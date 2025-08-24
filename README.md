# sd_meta_extract

<h1>THIS IS MY PERSONAL PROJECT FOR PERSONAL USE. DO NOT USE IN A PRODUCTION ENVIRONMENT!</h1>

Extract Stable Diffusion–style generation metadata from images and write `.json` sidecars.

## Features
- **PNG**: reads `parameters` (Automatic1111), `prompt` JSON (ComfyUI), `sd-metadata` JSON (InvokeAI)
- **JPEG**: scrapes EXIF/JPEG comment if anything survived
- **Sidecars**: writes `image.ext.json` by default (disable with `--no-sidecar`)
- **Directory mode**: recursive scan with a **fancy progress bar** (via `rich`)
- **Tidy summaries**: colored, emoji one-liners with truncated prompts
- **Extensions filter**: `--extensions .png,.jpg,.jpeg`
- **Rebuild for JPEGs**: `--rebuild-jpeg-sidecar` copies metadata from a **same-stem PNG** if the JPEG is empty

## Install

### Linux / macOS
```bash
./install.sh
source .venv/bin/activate
```

### Windows
```bat
install.bat
call .venv\Scripts\activate
```

## Usage

### Single file
```bash
python sd_meta_extract.py my_image.png
```

### Directory (recursive), with JSON printout
```bash
python sd_meta_extract.py ./renders --json
```

### Limit scan to PNG + JPG only
```bash
python sd_meta_extract.py ./renders --extensions .png,.jpg
```

### Rebuild JPEG sidecars from sibling PNGs
```bash
python sd_meta_extract.py ./renders --rebuild-jpeg-sidecar
```

### Disable sidecars
```bash
python sd_meta_extract.py ./renders --no-sidecar
```

### Control prompt snippet length
```bash
python sd_meta_extract.py ./renders --max-prompt-len 240
```

## Notes
- PNGs preserve SD metadata; JPEGs usually don’t.
- Sidecars include `_file`, `_format`, `_size`. If rebuilt, `_sourced_from` contains the PNG path used.
- Works best with Automatic1111, but will surface ComfyUI/InvokeAI blobs when present.

- ## 📝 License

MIT License. Do whatever you want, just don’t blame me if zombies eat your files. 🧟
