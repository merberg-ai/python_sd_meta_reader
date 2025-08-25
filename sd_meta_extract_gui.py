#!/usr/bin/env python3
"""
GUI wrapper for sd_meta_extract.py

Features
- Open via command-line arg (image path) OR drag & drop OR File > Open
- Side-by-side: left = image preview, right = metadata (pretty JSON + summary)
- Load / Save buttons (Save writes sidecar .json next to image or lets you export)
- Simple Settings tab: max prompt length, auto write sidecar, rebuild from sibling PNG (JPEG),
  and extension filter used when opening via dialog
- Non-destructive: only writes sidecar on Save (or if auto-save toggled)
"""

import json
import sys
from pathlib import Path

from PIL import ImageQt, Image

from PyQt6 import QtCore, QtGui, QtWidgets

# ----- Import your extractor (same folder) -----
try:
    import sd_meta_extract as core  # your uploaded CLI module
except Exception as e:
    core = None
    _import_error = e
else:
    _import_error = None


# ----------------- Utilities -----------------
def pil_image_from_path(p: Path) -> Image.Image | None:
    try:
        return Image.open(p)
    except Exception:
        return None


def pixmap_from_pil(img: Image.Image, max_w: int, max_h: int) -> QtGui.QPixmap | None:
    if img is None:
        return None
    w, h = img.size
    scale = min(max_w / max(1, w), max_h / max(1, h), 1.0)
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    qim = ImageQt.ImageQt(img.convert("RGBA"))
    return QtGui.QPixmap.fromImage(qim)


def pretty_json(obj) -> str:
    try:
        return json.dumps(obj, indent=2, ensure_ascii=False)
    except Exception:
        return str(obj)


def summary_line(meta: dict, max_prompt_len: int = 160) -> str:
    # Roughly mirror your CLI summary
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
    else:
        body = "No prompt found" if meta.get("_note") else ""
    return head + ("\n" + body if body else "")


# ----------------- Widgets -----------------
class DropImageLabel(QtWidgets.QLabel):
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
                    suf = Path(u.toLocalFile()).suffix.lower()
                    if suf in {".png", ".jpg", ".jpeg"}:
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


class SettingsPane(QtWidgets.QWidget):
    settingsChanged = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)

        self.maxPrompt = QtWidgets.QSpinBox()
        self.maxPrompt.setRange(32, 10000)
        self.maxPrompt.setValue(160)

        self.autoSidecar = QtWidgets.QCheckBox("Auto-save sidecar on load")
        self.autoSidecar.setChecked(False)

        self.rebuildJPEG = QtWidgets.QCheckBox("Try rebuild JPEG sidecar from sibling PNG")
        self.rebuildJPEG.setToolTip("If a JPEG has no metadata, copy metadata from a sibling PNG with same stem.")

        self.extEdit = QtWidgets.QLineEdit(".png,.jpg,.jpeg")
        self.extEdit.setToolTip("Extensions used by the Open dialog (comma-separated).")

        form = QtWidgets.QFormLayout()
        form.addRow("Max prompt length:", self.maxPrompt)
        form.addRow(self.autoSidecar)
        form.addRow(self.rebuildJPEG)
        form.addRow("File extensions filter:", self.extEdit)

        wrap = QtWidgets.QVBoxLayout(self)
        wrap.addLayout(form)
        wrap.addStretch(1)

        for w in (self.maxPrompt, self.autoSidecar, self.rebuildJPEG, self.extEdit):
            if isinstance(w, QtWidgets.QAbstractButton):
                w.toggled.connect(self.settingsChanged.emit)
            elif isinstance(w, QtWidgets.QSpinBox):
                w.valueChanged.connect(self.settingsChanged.emit)
            elif isinstance(w, QtWidgets.QLineEdit):
                w.textChanged.connect(self.settingsChanged.emit)

    # Helpers to read settings
    def get_max_prompt_len(self) -> int:
        return int(self.maxPrompt.value())

    def get_auto_sidecar(self) -> bool:
        return self.autoSidecar.isChecked()

    def get_rebuild_jpeg(self) -> bool:
        return self.rebuildJPEG.isChecked()

    def get_extensions(self) -> set[str]:
        raw = self.extEdit.text().strip()
        if not raw:
            return {".png", ".jpg", ".jpeg"}
        parts = [p.strip().lower() for p in raw.split(",") if p.strip()]
        out = set()
        for p in parts:
            if not p.startswith("."):
                p = "." + p
            out.add(p)
        return out


class InspectorPane(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # Left: image; Right: metadata
        self.imageLabel = DropImageLabel("Drop an image here\n(.png, .jpg)")
        self.imageLabel.setMinimumSize(320, 240)
        self.imageLabel.fileDropped.connect(self._on_drop)

        self.metaSummary = QtWidgets.QTextEdit()
        self.metaSummary.setReadOnly(True)
        self.metaSummary.setMinimumHeight(80)

        self.metaJSON = QtWidgets.QPlainTextEdit()
        self.metaJSON.setReadOnly(True)

        rightBox = QtWidgets.QVBoxLayout()
        rightBox.addWidget(QtWidgets.QLabel("Summary"))
        rightBox.addWidget(self.metaSummary, 0)
        rightBox.addWidget(QtWidgets.QLabel("Extracted JSON"))
        rightBox.addWidget(self.metaJSON, 1)

        splitter = QtWidgets.QSplitter()
        splitter.setOrientation(QtCore.Qt.Orientation.Horizontal)
        leftWrap = QtWidgets.QWidget()
        leftLay = QtWidgets.QVBoxLayout(leftWrap)
        leftLay.addWidget(self.imageLabel, 1)
        splitter.addWidget(leftWrap)

        rightWrap = QtWidgets.QWidget()
        rightWrap.setLayout(rightBox)
        splitter.addWidget(rightWrap)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)

        # Buttons
        self.btnLoad = QtWidgets.QPushButton("Load")
        self.btnSave = QtWidgets.QPushButton("Save")
        self.btnExport = QtWidgets.QPushButton("Export JSON…")
        btnRow = QtWidgets.QHBoxLayout()
        btnRow.addWidget(self.btnLoad)
        btnRow.addWidget(self.btnSave)
        btnRow.addWidget(self.btnExport)
        btnRow.addStretch(1)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(btnRow)
        layout.addWidget(splitter, 1)

        self.currentPath: Path | None = None
        self.currentMeta: dict | None = None

        self.btnLoad.clicked.connect(self.open_dialog)
        self.btnSave.clicked.connect(self.save_sidecar)
        self.btnExport.clicked.connect(self.export_json)

        # Keep a pointer to settings provider
        self.settings_provider: SettingsPane | None = None

    # Wiring from MainWindow to read settings
    def set_settings_provider(self, sp: SettingsPane):
        self.settings_provider = sp

    # ---- Actions ----
    def _on_drop(self, path: str):
        self.load_image(Path(path))

    def open_dialog(self):
        exts = {".png", ".jpg", ".jpeg"}
        if self.settings_provider:
            exts = self.settings_provider.get_extensions()
        patt = "Images (" + " ".join(f"*{e}" for e in sorted(exts)) + ")"
        f, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open Image", "", patt)
        if f:
            self.load_image(Path(f))

    def load_image(self, p: Path):
        if core is None:
            QtWidgets.QMessageBox.critical(
                self, "Import error",
                f"Could not import sd_meta_extract.py:\n{_import_error}"
            )
            return
        img = pil_image_from_path(p)
        if img is None:
            QtWidgets.QMessageBox.warning(self, "Open failed", f"Could not open image:\n{p}")
            return
        pm = pixmap_from_pil(img, 2000, 2000)  # scale down if huge
        self.imageLabel.setPixmap(pm)
        self.imageLabel.setStyleSheet("")  # remove dashed border after load
        self.currentPath = p

        # Extract
        meta = core.extract_from_image(p)
        # Optional JPEG salvage
        if (self.settings_provider and self.settings_provider.get_rebuild_jpeg()
            and p.suffix.lower() in {".jpg", ".jpeg"} and ("_note" in meta)):
            salvaged = core.rebuild_from_sibling_png(p)
            if salvaged:
                meta = salvaged

        self.currentMeta = meta
        self.metaSummary.setPlainText(
            summary_line(meta, self.settings_provider.get_max_prompt_len() if self.settings_provider else 160)
        )
        self.metaJSON.setPlainText(pretty_json(meta))
        self.parent().parent().statusBar().showMessage(f"Loaded {p}")

        # Auto sidecar?
        if self.settings_provider and self.settings_provider.get_auto_sidecar():
            self.save_sidecar(auto=True)

    def save_sidecar(self, auto: bool = False):
        if not self.currentPath or not self.currentMeta:
            if not auto:
                QtWidgets.QMessageBox.information(self, "Nothing to save", "Load an image first.")
            return
        sidecar = self.currentPath.with_suffix(self.currentPath.suffix + ".json")
        try:
            with open(sidecar, "w", encoding="utf-8") as fp:
                json.dump(self.currentMeta, fp, indent=2, ensure_ascii=False)
        except Exception as e:
            if not auto:
                QtWidgets.QMessageBox.critical(self, "Save failed", str(e))
            return
        self.parent().parent().statusBar().showMessage(f"Saved sidecar: {sidecar}")

    def export_json(self):
        if not self.currentMeta:
            QtWidgets.QMessageBox.information(self, "Nothing to export", "Load an image first.")
            return
        f, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export JSON", "metadata.json", "JSON (*.json)")
        if not f:
            return
        try:
            with open(f, "w", encoding="utf-8") as fp:
                json.dump(self.currentMeta, fp, indent=2, ensure_ascii=False)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Export failed", str(e))
            return
        self.parent().parent().statusBar().showMessage(f"Exported: {f}")


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, start_path: Path | None = None):
        super().__init__()
        self.setWindowTitle("SD Meta Inspector")
        self.resize(1200, 800)

        tabs = QtWidgets.QTabWidget()
        self.inspector = InspectorPane()
        self.settings = SettingsPane()
        self.inspector.set_settings_provider(self.settings)

        tabs.addTab(self.inspector, "Inspector")
        tabs.addTab(self.settings, "Settings")

        self.setCentralWidget(tabs)

        # Menu
        fileMenu = self.menuBar().addMenu("&File")
        actOpen = fileMenu.addAction("Open…")
        actSave = fileMenu.addAction("Save sidecar")
        fileMenu.addSeparator()
        actQuit = fileMenu.addAction("Quit")
        actOpen.triggered.connect(self.inspector.open_dialog)
        actSave.triggered.connect(self.inspector.save_sidecar)
        actQuit.triggered.connect(lambda: QtWidgets.QApplication.instance().quit())

        self.setStatusBar(QtWidgets.QStatusBar())

        # If given a starting file, load it
        if start_path and start_path.exists():
            QtCore.QTimer.singleShot(0, lambda: self.inspector.load_image(start_path))


def main():
    app = QtWidgets.QApplication(sys.argv)
    # Optional CLI arg
    start_path = None
    if len(sys.argv) > 1:
        p = Path(sys.argv[1])
        if p.exists():
            start_path = p

    mw = MainWindow(start_path)
    mw.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
