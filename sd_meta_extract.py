#!/usr/bin/env python3
"""
sd_meta_extract.py — Extract Stable Diffusion-style generation metadata from images.

Highlights
- File(s) or directory path(s). Directories scanned recursively.
- PNG tEXt/iTXt: Automatic1111 ("parameters"), ComfyUI ("prompt" JSON), InvokeAI ("sd-metadata" JSON)
- JPEG EXIF/JPEG comment best-effort scraping
- Always writes a .json sidecar next to each image (disable with --no-sidecar)
- Fancy progress bar with 'rich' (fallback to simple bar)
- Tidy, colored, emoji summaries with truncation
- NEW: --extensions to control scanned file types
- NEW: --rebuild-jpeg-sidecar to copy metadata from a sibling PNG with the same stem

Usage
  python sd_meta_extract.py /path/to/img_or_dir
  python sd_meta_extract.py ./renders --json
  python sd_meta_extract.py ./renders --no-sidecar
  python sd_meta_extract.py ./renders --max-prompt-len 140
  python sd_meta_extract.py ./renders --extensions .png,.jpg
  python sd_meta_extract.py ./renders --rebuild-jpeg-sidecar
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from PIL import Image, PngImagePlugin, ExifTags

# ---------- Optional rich cosmetics ----------
_HAVE_RICH = False
try:
    from rich.console import Console
    from rich.progress import Progress, BarColumn, TimeElapsedColumn, TimeRemainingColumn, TaskProgressColumn
    from rich.theme import Theme
    RICH_THEME = Theme({
        "ok": "bold green",
        "warn": "bold yellow",
        "err": "bold red",
        "dim": "dim",
        "k": "bold white",
        "v": "cyan",
    })
    console = Console(theme=RICH_THEME)
    _HAVE_RICH = True
except Exception:
    console = None

# ---------- Simple color fallback ----------
class _Color:
    k = v = ok = warn = err = dim = reset = ""
C = _Color()
try:
    from colorama import init as colorama_init, Fore, Style
    colorama_init()
    C.k = Style.BRIGHT
    C.v = Fore.CYAN
    C.ok = Fore.GREEN + Style.BRIGHT
    C.warn = Fore.YELLOW + Style.BRIGHT
    C.err = Fore.RED + Style.BRIGHT
    C.dim = Style.DIM
    C.reset = Style.RESET_ALL
except Exception:
    pass

DEFAULT_EXTS = {".png", ".jpg", ".jpeg"}
EMO = {"ok":"✅","miss":"🫥","file":"🖼️","seed":"🌱","cfg":"⚙️","steps":"🧭","model":"🧬","copy":"📎"}

# ---------- Utility ----------
def shorten(text: str, limit: int = 160) -> str:
    if text is None:
        return ""
    t = " ".join(text.split())
    return t if len(t) <= limit else t[: max(0, limit - 1)] + "…"

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

# ---------- A1111 parsing ----------
A1111_KV_RE = re.compile(r"\s*([^:,]+)\s*:\s*([^,]+)\s*(?:,|$)")

def parse_a1111_parameters_block(text: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    neg_key = "Negative prompt:"
    joined = text.strip()
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
    else:
        p_only, tail = split_tail(joined)
        out["prompt"] = p_only.strip()

    for kv in A1111_KV_RE.finditer(tail):
        k = kv.group(1).strip().lower().replace(" ", "_")
        v = kv.group(2).strip()
        out[k] = v

    for key in ("steps", "seed"):
        if key in out:
            try: out[key] = int(out[key])
            except: pass
    if "cfg_scale" in out:
        try: out["cfg_scale"] = float(out["cfg_scale"])
        except: pass
    if "size" in out and isinstance(out["size"], str) and "x" in out["size"].lower():
        try:
            w, h = out["size"].lower().split("x", 1)
            out["width"], out["height"] = int(w), int(h)
        except: pass

    return out

# ---------- PNG/JPEG extraction ----------
def parse_png_info(info: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if "parameters" in info and isinstance(info["parameters"], str):
        out.update(parse_a1111_parameters_block(info["parameters"]))
    if "prompt" in info and isinstance(info["prompt"], str):
        j = safe_json_loads(info["prompt"])
        out["comfyui_prompt"] = j if isinstance(j, (dict, list)) else info["prompt"]
    if "Negative prompt" in info and isinstance(info["Negative prompt"], str):
        out.setdefault("negative_prompt", info["Negative prompt"])
    if "sd-metadata" in info and isinstance(info["sd-metadata"], str):
        j = safe_json_loads(info["sd-metadata"])
        if isinstance(j, dict):
            out["invokeai_metadata"] = j
    for k in ("Software", "Version"):
        if k in info: out[k.lower()] = info[k]
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
            try: v = v.decode("utf-16-le") if k.startswith("XP") else v.decode("utf-8", "ignore")
            except: v = None
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

# ---------- Discovery ----------
def normalize_exts(spec: Optional[str]) -> set:
    if not spec:
        return set(DEFAULT_EXTS)
    parts = [p.strip().lower() for p in spec.split(",") if p.strip()]
    exts = set()
    for p in parts:
        if not p.startswith("."):
            p = "." + p
        exts.add(p)
    return exts

def iter_image_files(paths: Iterable[Path], allowed_exts: set) -> List[Path]:
    out: List[Path] = []
    for p in paths:
        if p.is_dir():
            for sub in p.rglob("*"):
                if sub.is_file() and sub.suffix.lower() in allowed_exts:
                    out.append(sub)
        elif p.is_file() and p.suffix.lower() in allowed_exts:
            out.append(p)
    return sorted(out)

# ---------- Rebuild helpers ----------
def sibling_png(path: Path) -> Optional[Path]:
    # Look first for identical stem PNG, then common variants: "-0000", "_00000"
    candidates = [
        path.with_suffix(".png"),
        path.with_name(path.stem.split(".")[0] + ".png"),  # handle ".jpg.png" oddities
    ]
    # also try removing common counters
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
    # Tag the provenance and override file/format/size to reflect the JPEG target
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

# ---------- Output ----------
def print_summary(meta: Dict[str, Any], max_prompt_len: int = 160) -> None:
    file = Path(meta.get("_file", ""))
    w = meta.get("_size", {}).get("width")
    h = meta.get("_size", {}).get("height")
    seed = meta.get("seed")
    steps = meta.get("steps")
    cfg = meta.get("cfg_scale")
    model = meta.get("model") or meta.get("software") or meta.get("version")
    prompt = meta.get("prompt") or meta.get("comfyui_prompt")
    ok = EMO["ok"] if ("prompt" in meta or "comfyui_prompt" in meta) else EMO["miss"]

    ptxt = shorten(prompt if isinstance(prompt, str) else json.dumps(prompt) if prompt else "", max_prompt_len)
    left = f"{EMO['file']} {file.name} [{meta.get('_format','?')},{w}x{h}]"
    right_bits = []
    if seed is not None:   right_bits.append(f"{EMO['seed']} {seed}")
    if steps is not None:  right_bits.append(f"{EMO['steps']} {steps}")
    if cfg is not None:    right_bits.append(f"{EMO['cfg']} {cfg}")
    if model:              right_bits.append(f"{EMO['model']} {shorten(str(model), 40)}")
    if "_sourced_from" in meta: right_bits.append(f"{EMO['copy']} from PNG")
    right = "  ".join(right_bits)

    line1 = f"{ok} {left}  {right}"
    line2 = f"   🧠 {ptxt}" if ptxt else (f"   {EMO['miss']} No prompt found" if meta.get("_note") else "")

    if _HAVE_RICH:
        console.print(line1, style="ok" if ok == EMO["ok"] else "warn")
        if line2: console.print(line2, style="v")
    else:
        print(f"{C.ok}{line1}{C.reset}")
        if line2: print(f"{C.v}{line2}{C.reset}")

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Extract Stable Diffusion-style metadata from images.")
    ap.add_argument("paths", nargs="+", help="Image files or directories")
    ap.add_argument("--json", action="store_true", help="Print JSON to stdout (still writes sidecars unless --no-sidecar)")
    ap.add_argument("--no-sidecar", action="store_true", help="Disable writing .json sidecar files")
    ap.add_argument("--max-prompt-len", type=int, default=160, help="Prompt snippet length (default 160)")
    ap.add_argument("--extensions", type=str, default=None,
                    help="Comma-separated list of extensions to scan (e.g. .png,.jpg). Default: .png,.jpg,.jpeg")
    ap.add_argument("--rebuild-jpeg-sidecar", action="store_true",
                    help="If a JPEG has no metadata, try to copy metadata from a sibling PNG with the same stem.")
    args = ap.parse_args()

    allowed_exts = normalize_exts(args.extensions)
    inputs = [Path(p) for p in args.paths]
    files = iter_image_files(inputs, allowed_exts)

    if not files:
        msg = f"No images found (extensions: {', '.join(sorted(allowed_exts))})."
        if _HAVE_RICH: console.print(msg, style="err")
        else: print(f"{C.err}{msg}{C.reset}")
        sys.exit(1)

    results: List[Dict[str, Any]] = []

    # progress bar if multiple
    use_progress = (len(files) > 1 or any(p.is_dir() for p in inputs))
    if use_progress and _HAVE_RICH:
        with Progress(
            "[progress.description]{task.description}",
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Scanning images", total=len(files))
            for f in files:
                meta = extract_from_image(f)
                # JPEG salvage if requested
                if args.rebuild_jpeg_sidecar and f.suffix.lower() in {".jpg", ".jpeg"} and ("_note" in meta):
                    salvaged = rebuild_from_sibling_png(f)
                    if salvaged:
                        meta = salvaged
                results.append(meta)
                if not args.no_sidecar:
                    sidecar = f.with_suffix(f.suffix + ".json")
                    with open(sidecar, "w", encoding="utf-8") as fp:
                        json.dump(meta, fp, indent=2)
                print_summary(meta, max_prompt_len=args.max_prompt_len)
                progress.advance(task, 1)
    else:
        total = len(files)
        for i, f in enumerate(files, 1):
            meta = extract_from_image(f)
            if args.rebuild_jpeg_sidecar and f.suffix.lower() in {".jpg", ".jpeg"} and ("_note" in meta):
                salvaged = rebuild_from_sibling_png(f)
                if salvaged:
                    meta = salvaged
            results.append(meta)
            if not args.no_sidecar:
                sidecar = f.with_suffix(f.suffix + ".json")
                with open(sidecar, "w", encoding="utf-8") as fp:
                    json.dump(meta, fp, indent=2)
            # simple fallback progress
            if use_progress:
                bar_len = 24
                done = int(bar_len * i / total)
                bar = "[" + "#" * done + "-" * (bar_len - done) + f"] {i}/{total}"
                print(bar, end="\r", flush=True)
                print()
            print_summary(meta, max_prompt_len=args.max_prompt_len)

    if args.json:
        print(json.dumps(results if len(results) > 1 else results[0], indent=2))

if __name__ == "__main__":
    main()
