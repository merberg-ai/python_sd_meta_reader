#!/usr/bin/env python3
"""
SD Meta Inspector — single-file GUI
- Extracts Stable Diffusion-style metadata from PNG/JPEG (A1111, ComfyUI, InvokeAI, EXIF/JPEG comment).
- Side-by-side: image + metadata JSON with a human summary.
- Persisted settings (~/.sd_meta_inspector.json): default_zoom (50%), rebuild_jpeg_from_png, last_dir.
- Ctrl+Wheel zoom, +/- buttons, zoom combo.
- No menu bar; buttons: Load, Export JSON (writes sidecar), Copy Prompt to Clipboard.

Tested with Python 3.10+ and PyQt6 + Pillow.
"""

import json
import sys
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from PIL import Image, PngImagePlugin, ExifTags, ImageQt
from PyQt6 import QtCore, QtGui, QtWidgets

# ----------------- Settings -----------------
def _config_path() -> Path:
    return Path.home() / ".sd_meta_inspector.json"

DEFAULT_SETTINGS = {
    "default_zoom": 50,            # percent
    "rebuild_jpeg_from_png": False,
    "last_dir": "",
}

def load_settings() -> dict:
    p = _config_path()
    if p.exists():
        try:
            data = json.load(open(p, "r", encoding="utf-8"))
            merged = DEFAULT_SETTINGS | {k: data.get(k, v) for k, v in DEFAULT_SETTINGS.items()}
            return merged
        except Exception:
            return DEFAULT_SETTINGS.copy()
    return DEFAULT_SETTINGS.copy()

def save_settings(s: dict) -> None:
    try:
        json.dump(s, open(_config_path(), "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    except Exception:
        pass

# ----------------- Extraction helpers (embedded from CLI) -----------------
def load_image(path: Path) -> Image.Image:
    try:
        return Image.open(path)
    except Exception as e:
        raise RuntimeError(f"Failed to open {path}: {e}")

def get_exif_dict(img: Image.Image) -> Dict[str, Any]:
    try:
        exif = img.getexif()
        if not exif:
            return {}
        return {ExifTags.TAGS.get(k, k): v for k, v in exif.items()}
    except Exception:
        return {}

def get_jpeg_comment(img: Image.Image) -> Optional[bytes]:
    try:
        return img.info.get("comment")
    except Exception:
        return None

def safe_json_loads(s: str) -> Optional[Any]:
    try:
        return json.loads(s)
    except Exception:
        return None

# ---- Automatic1111 parameter block parsing ----
A1111_KV_RE = re.compile(r"\s*([^:,]+)\s*:\s*([^,]+)\s*(?:,|$)")

def parse_a1111_parameters_block(text: str) -> Dict[str, Any]:
    """Parse A1111 'parameters' block into structured fields."""
    out: Dict[str, Any] = {}
    neg_key = "Negative prompt:"
    joined = (text or "").strip()
    if not joined:
        return out

    def split_tail(blob: str) -> Tuple[str, str]:
        m = re.search(r"\b(Steps?|Sampler|CFG scale|Seed|Size|Model|Model hash|Version)\b\s*:", blob)
        if m:
            return blob[:m.start()].strip(), blob[m.start():].strip()
        return blob, ""

    if neg_key in joined:
        p_txt, rest = joined.split(neg_key, 1)
        out["prompt"], rest = p_txt.strip(), rest
        neg, tail = split_tail(rest)
        out["negative_prompt"] = neg.strip(" :")
        tail_blob = tail
    else:
        # No explicit negative; try to split prompt vs KV tail
        p_txt, tail_blob = split_tail(joined)
        out["prompt"] = p_txt.strip()

    # parse KVs in tail
    for m in A1111_KV_RE.finditer(tail_blob):
        k = m.group(1).strip().lower().replace(" ", "_")
        v = m.group(2).strip()
        if k in {"steps", "seed"}:
            try: out[k] = int(v)
            except: out[k] = v
        elif k in {"cfg_scale"}:
            try: out[k] = float(v)
            except: out[k] = v
        elif k == "size":
            out["size"] = v
            try:
                w, h = v.lower().split("x", 1)
                out["width"], out["height"] = int(w), int(h)
            except:
                pass
        else:
            out[k] = v
    return out

# ---- PNG/JPEG extraction ----
def parse_png_info(info: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    # Automatic1111
    if "parameters" in info and isinstance(info["parameters"], str):
        out.update(parse_a1111_parameters_block(info["parameters"]))
    # ComfyUI
    if "prompt" in info and isinstance(info["prompt"], str):
        j = safe_json_loads(info["prompt"])
        out["comfyui_prompt"] = j if isinstance(j, (dict, list)) else info["prompt"]
    # InvokeAI
    if "sd-metadata" in info and isinstance(info["sd-metadata"], str):
        j = safe_json_loads(info["sd-metadata"])
        if isinstance(j, dict):
            out["invokeai_metadata"] = j
    # Other labels
    for k in ("Software", "Version"):
        if k in info:
            out[k.lower()] = info[k]
    return out

def try_extract_png(img: Image.Image) -> Dict[str, Any]:
    if not isinstance(img, PngImagePlugin.PngImageFile):
        return {}
    return parse_png_info(dict(img.info))

def try_extract_jpeg(img: Image.Image) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    exif = get_exif_dict(img)
    for k in ("UserComment", "ImageDescription", "XPComment"):
        v = exif.get(k)
        if isinstance(v, bytes):
            try:
                v = v.decode("utf-16-le") if k.startswith("XP") else v.decode("utf-8", "ignore")
            except:
                v = None
        if isinstance(v, str) and v.strip():
            out.update(parse_a1111_parameters_block(v) or {"raw_exif_comment": v})
    c = get_jpeg_comment(img)
    if c:
        s = c.decode("utf-8", "ignore")
        out.update(parse_a1111_parameters_block(s) or {"raw_jpeg_comment": s})
    return out

def extract_from_image(path: Path) -> Dict[str, Any]:
    img = load_image(path)
    meta: Dict[str, Any] = {}
    if img.format == "PNG":
        meta.update(try_extract_png(img))
    elif img.format in {"JPEG", "JPG"}:
        meta.update(try_extract_jpeg(img))
    else:
        meta.update(try_extract_png(img))
        meta.update(try_extract_jpeg(img))
    if not meta:
        meta["_note"] = "No SD-style metadata found."
    meta["_file"] = str(path)
    meta["_format"] = img.format
    meta["_size"] = {"width": img.width, "height": img.height}
    return meta

def sibling_png(path: Path) -> Optional[Path]:
    # try exact stem, plus removing common counters like _00000
    candidates = [
        path.with_suffix(".png"),
        path.with_name(path.stem.split(".")[0] + ".png"),
    ]
    for suffix in ("-00000", "-0000", "-000", "_00000", "_0000", "_000"):
        if path.stem.endswith(suffix):
            candidates.append(path.with_name(path.stem[: -len(suffix)] + ".png"))
    for c in candidates:
        if c.exists():
            return c
    return None

def rebuild_from_sibling_png(jpeg_path: Path) -> Optional[Dict[str, Any]]:
    sib = sibling_png(jpeg_path)
    if not sib:
        return None
    try:
        meta_png = extract_from_image(sib)
    except Exception:
        return None
    meta = dict(meta_png)
    meta["_sourced_from"] = str(sib)
    meta["_file"] = str(jpeg_path)
    try:
        img = Image.open(jpeg_path)
        meta["_format"] = img.format
        meta["_size"] = {"width": img.width, "height": img.height}
    except Exception:
        pass
    return meta

# ----------------- GUI utilities -----------------
def qt_pixmap_from_pil(img: Image.Image) -> QtGui.QPixmap | None:
    if img is None:
        return None
    qim = ImageQt.ImageQt(img.convert("RGBA"))
    return QtGui.QPixmap.fromImage(qim)

def pretty_json(obj) -> str:
    try:
        return json.dumps(obj, indent=2, ensure_ascii=False)
    except Exception:
        return str(obj)

def summary_line(meta: Dict[str, Any], max_prompt_len: int = 160) -> str:
    file = Path(meta.get("_file", "")).name
    fmt = meta.get("_format", "?")
    size = meta.get("_size", {})
    w, h = size.get("width"), size.get("height")
    seed = meta.get("seed")
    steps = meta.get("steps")
    cfg = meta.get("cfg_scale")
    model = meta.get("model") or meta.get("software") or meta.get("version")
    prompt = meta.get("prompt") or meta.get("comfyui_prompt")

    def shorten(text: str, limit: int = 160) -> str:
        t = " ".join((text or "").split())
        return t if len(t) <= limit else t[: max(0, limit - 1)] + "…"

    bits = [f"{file} [{fmt},{w}x{h}]"]
    if seed is not None: bits.append(f"seed {seed}")
    if steps is not None: bits.append(f"steps {steps}")
    if cfg is not None: bits.append(f"cfg {cfg}")
    if model: bits.append(f"model {str(model)[:40]}")
    head = " | ".join(bits)
    body = ""
    if isinstance(prompt, (dict, list)):
        body = shorten(json.dumps(prompt), max_prompt_len)
    elif isinstance(prompt, str):
        body = shorten(prompt, max_prompt_len)
    elif meta.get("_note"):
        body = "No prompt found"
    return head + ("\n" + body if body else "")

# ----------------- Widgets -----------------
class ZoomableImageLabel(QtWidgets.QLabel):
    wheelZoomRequested = QtCore.pyqtSignal(int)
    fileDropped = QtCore.pyqtSignal(str)

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.setAcceptDrops(True)
        self.setStyleSheet("QLabel{border:2px dashed #666; border-radius:12px; color:#aaa;}")

    def dragEnterEvent(self, e: QtGui.QDragEnterEvent):
        if e.mimeData().hasUrls():
            for u in e.mimeData().urls():
                if u.isLocalFile():
                    if Path(u.toLocalFile()).suffix.lower() in {".png", ".jpg", ".jpeg"}:
                        e.acceptProposedAction()
                        return
        e.ignore()

    def dropEvent(self, e: QtGui.QDropEvent):
        for u in e.mimeData().urls():
            if u.isLocalFile():
                p = u.toLocalFile()
                if Path(p).suffix.lower() in {".png", ".jpg", ".jpeg"}:
                    self.fileDropped.emit(p)
                    break

    def wheelEvent(self, e: QtGui.QWheelEvent):
        if QtWidgets.QApplication.keyboardModifiers() & QtCore.Qt.KeyboardModifier.ControlModifier:
            steps = int(e.angleDelta().y() / 120)
            if steps:
                self.wheelZoomRequested.emit(steps)
                e.accept()
                return
        super().wheelEvent(e)

class InspectorPane(QtWidgets.QWidget):
    statusMessage = QtCore.pyqtSignal(str)
    imageScaledSize = QtCore.pyqtSignal(int, int)

    def __init__(self, app_settings: dict, parent=None):
        super().__init__(parent)
        self.app_settings = app_settings

        # Left: image
        self.imageLabel = ZoomableImageLabel("Drop an image here\n(.png, .jpg)")
        self.imageLabel.setMinimumSize(320, 240)
        self.imageLabel.fileDropped.connect(self._on_drop)
        self.imageLabel.wheelZoomRequested.connect(self._on_wheel_zoom)

        self.imageScroll = QtWidgets.QScrollArea()
        self.imageScroll.setWidgetResizable(True)
        self.imageScroll.setWidget(self.imageLabel)

        # Right: meta
        self.metaSummary = QtWidgets.QTextEdit(readOnly=True)
        self.metaSummary.setMinimumHeight(80)
        self.metaJSON = QtWidgets.QPlainTextEdit(readOnly=True)

        rightBox = QtWidgets.QVBoxLayout()
        rightBox.addWidget(QtWidgets.QLabel("Summary"))
        rightBox.addWidget(self.metaSummary, 0)
        rightBox.addWidget(QtWidgets.QLabel("Extracted JSON"))
        rightBox.addWidget(self.metaJSON, 1)

        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        leftWrap = QtWidgets.QWidget()
        leftLay = QtWidgets.QVBoxLayout(leftWrap)

        # Zoom controls
        zoomRow = QtWidgets.QHBoxLayout()
        zoomRow.addWidget(QtWidgets.QLabel("Zoom"))
        self.zoomCombo = QtWidgets.QComboBox()
        for z in (25, 50, 75, 100, 125, 150, 200, 300, 400):
            self.zoomCombo.addItem(f"{z} %", z)
        self.zoomCombo.currentIndexChanged.connect(self._on_zoom_combo_changed)
        self.btnZoomOut = QtWidgets.QToolButton(text="–")
        self.btnZoomIn = QtWidgets.QToolButton(text="+")
        self.btnZoomOut.clicked.connect(lambda: self.set_zoom(self.zoomPercent - 25))
        self.btnZoomIn.clicked.connect(lambda: self.set_zoom(self.zoomPercent + 25))
        zoomRow.addWidget(self.zoomCombo)
        zoomRow.addWidget(self.btnZoomOut)
        zoomRow.addWidget(self.btnZoomIn)
        zoomRow.addStretch(1)

        leftLay.addLayout(zoomRow)
        leftLay.addWidget(self.imageScroll, 1)

        rightWrap = QtWidgets.QWidget()
        rightWrap.setLayout(rightBox)

        self.splitter.addWidget(leftWrap)
        self.splitter.addWidget(rightWrap)
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 2)

        # Top bar: Load / Export / Copy + extraction option
        self.btnLoad = QtWidgets.QPushButton("Load")
        self.btnExport = QtWidgets.QPushButton("Export JSON")
        self.btnCopyPrompt = QtWidgets.QPushButton("Copy Prompt to Clipboard")
        self.chkRebuildJPEG = QtWidgets.QCheckBox("Rebuild JPEG from sibling PNG")
        self.chkRebuildJPEG.setChecked(bool(self.app_settings.get("rebuild_jpeg_from_png", False)))
        self.chkRebuildJPEG.toggled.connect(self._on_rebuild_toggled)

        btnRow = QtWidgets.QHBoxLayout()
        btnRow.addWidget(self.btnLoad)
        btnRow.addWidget(self.btnExport)
        btnRow.addWidget(self.btnCopyPrompt)
        btnRow.addSpacing(16)
        btnRow.addWidget(self.chkRebuildJPEG)
        btnRow.addStretch(1)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(btnRow)
        layout.addWidget(self.splitter, 1)

        # State
        self.currentPath: Path | None = None
        self.currentMeta: Dict[str, Any] | None = None
        self._sourceImage: Image.Image | None = None
        self.zoomPercent: int = int(self.app_settings.get("default_zoom", 50))

        # Wire up
        self.btnLoad.clicked.connect(self.open_dialog)
        self.btnExport.clicked.connect(self.export_json)
        self.btnCopyPrompt.clicked.connect(self.copy_prompt_to_clipboard)

        # Init zoom UI
        self._apply_zoom_combo_from_settings()

    # ---- settings hooks ----
    def _on_rebuild_toggled(self, checked: bool):
        self.app_settings["rebuild_jpeg_from_png"] = bool(checked)
        save_settings(self.app_settings)

    def _apply_zoom_combo_from_settings(self):
        val = int(self.app_settings.get("default_zoom", 50))
        idx = self.zoomCombo.findData(val)
        if idx < 0:
            idx = self.zoomCombo.findData(50)
            val = 50
        self.zoomCombo.blockSignals(True)
        self.zoomCombo.setCurrentIndex(idx)
        self.zoomCombo.blockSignals(False)
        self.set_zoom(val, update_combo=False)

    def _persist_zoom(self, percent: int):
        if self.app_settings.get("default_zoom") != percent:
            self.app_settings["default_zoom"] = percent
            save_settings(self.app_settings)

    # ---- file actions ----
    def _on_drop(self, path: str):
        self.load_image(Path(path))

    def open_dialog(self):
        start_dir = self.app_settings.get("last_dir") or ""
        if start_dir and not Path(start_dir).exists():
            start_dir = ""
        patt = "Images (*.png *.jpg *.jpeg)"
        f, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open Image", start_dir, patt)
        if f:
            p = Path(f)
            self.app_settings["last_dir"] = str(p.parent)
            save_settings(self.app_settings)
            self.load_image(p)

    def load_image(self, p: Path):
        img = load_image(p)
        self._sourceImage = img
        self.currentPath = p

        meta = extract_from_image(p)
        if (self.chkRebuildJPEG.isChecked()
                and p.suffix.lower() in {".jpg", ".jpeg"} and ("_note" in meta)):
            salvaged = rebuild_from_sibling_png(p)
            if salvaged:
                meta = salvaged

        self.currentMeta = meta
        self.metaSummary.setPlainText(summary_line(meta, 160))
        self.metaJSON.setPlainText(pretty_json(meta))
        self.imageLabel.setStyleSheet("")

        # Apply zoom
        self.set_zoom(self.zoomPercent, update_combo=True)
        self.statusMessage.emit(f"Loaded {p}")

        # Resize window to scaled image
        pm = self.imageLabel.pixmap()
        if pm:
            self.imageScaledSize.emit(pm.width(), pm.height())

    # ---- zoom mechanics ----
    def _on_zoom_combo_changed(self, _index: int):
        val = self.zoomCombo.currentData()
        if isinstance(val, int):
            self.set_zoom(val, update_combo=False)
            self._persist_zoom(self.zoomPercent)

    def _on_wheel_zoom(self, steps: int):
        new_zoom = self.zoomPercent + (25 * steps)
        self.set_zoom(new_zoom)
        self._persist_zoom(self.zoomPercent)

    def set_zoom(self, percent: int, update_combo: bool = True):
        percent = max(25, min(400, int(percent)))
        self.zoomPercent = percent
        if update_combo:
            idx = self.zoomCombo.findData(percent)
            if idx >= 0:
                self.zoomCombo.blockSignals(True)
                self.zoomCombo.setCurrentIndex(idx)
                self.zoomCombo.blockSignals(False)

        if self._sourceImage is None:
            return

        w, h = self._sourceImage.size
        scale = percent / 100.0
        target_w = max(1, int(w * scale))
        target_h = max(1, int(h * scale))
        scaled = self._sourceImage.resize((target_w, target_h), Image.LANCZOS)
        pm = qt_pixmap_from_pil(scaled)
        self.imageLabel.setPixmap(pm)
        self.imageLabel.adjustSize()

        self.statusMessage.emit(f"Zoom {percent}%")
        self.imageScaledSize.emit(target_w, target_h)

    # ---- export / clipboard ----
    def export_json(self):
        if not self.currentPath or not self.currentMeta:
            QtWidgets.QMessageBox.information(self, "Nothing to export", "Load an image first.")
            return
        sidecar = self.currentPath.with_suffix(self.currentPath.suffix + ".json")
        try:
            with open(sidecar, "w", encoding="utf-8") as fp:
                json.dump(self.currentMeta, fp, indent=2, ensure_ascii=False)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Export failed", f"Sidecar write error:\n{e}")
            return
        self.statusMessage.emit(f"Saved sidecar: {sidecar}")

    def copy_prompt_to_clipboard(self):
        if not self.currentMeta:
            QtWidgets.QMessageBox.information(self, "Nothing to copy", "Load an image first.")
            return
        meta = self.currentMeta
        # Priority: plain 'prompt' text
        text = meta.get("prompt")
        if isinstance(text, str) and text.strip():
            QtWidgets.QApplication.clipboard().setText(text.strip())
            self.statusMessage.emit("Copied prompt to clipboard")
            return
        # Fallback: ComfyUI structured prompt -> JSON string
        comfy = meta.get("comfyui_prompt")
        if comfy is not None:
            QtWidgets.QApplication.clipboard().setText(
                json.dumps(comfy, indent=2, ensure_ascii=False) if not isinstance(comfy, str) else comfy
            )
            self.statusMessage.emit("Copied ComfyUI prompt JSON to clipboard")
            return
        QtWidgets.QMessageBox.information(self, "No prompt found", "This image doesn't contain a prompt field.")

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, app_settings: dict, start_path: Path | None = None):
        super().__init__()
        self.setWindowTitle("SD Meta Inspector")
        self.resize(1200, 800)

        # No menu bar (keep it empty)
        self.setMenuBar(QtWidgets.QMenuBar())

        self.inspector = InspectorPane(app_settings)
        self.setCentralWidget(self.inspector)

        self.setStatusBar(QtWidgets.QStatusBar())
        self.inspector.statusMessage.connect(self.statusBar().showMessage)
        self.inspector.imageScaledSize.connect(self._maybe_resize_window)

        if start_path and start_path.exists():
            QtCore.QTimer.singleShot(0, lambda: self.inspector.load_image(start_path))

    def _maybe_resize_window(self, img_w: int, img_h: int):
        right_min = 360
        splitter = self.centralWidget().splitter
        right_widget = splitter.widget(1)
        right_w = max(right_min, right_widget.width() or right_widget.sizeHint().width() or right_min)

        extra_w = 60
        extra_h = 140
        scr = QtGui.QGuiApplication.primaryScreen().availableGeometry()
        max_w = scr.width() - 40
        max_h = scr.height() - 80

        target_w = min(img_w + right_w + extra_w, max_w)
        target_h = min(max(img_h, 500) + extra_h, max_h)
        self.resize(target_w, target_h)

        left_target = min(img_w + 20, target_w - right_w - 40)
        if left_target < 200:
            left_target = max(200, target_w // 3)
        splitter.setSizes([left_target, target_w - left_target - 10])

def main():
    app = QtWidgets.QApplication(sys.argv)

    settings = load_settings()
    start_path = None
    if len(sys.argv) > 1:
        p = Path(sys.argv[1])
        if p.exists():
            start_path = p
            settings["last_dir"] = str(p.parent)
            save_settings(settings)

    mw = MainWindow(settings, start_path)
    mw.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
