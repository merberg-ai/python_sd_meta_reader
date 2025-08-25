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

## Graphical User Interface (GUI)

- Alongside the command-line tool, this project also includes a graphical user interface for inspecting image metadata. The GUI makes it easy to drag-and-drop, preview, and export extracted information without touching the terminal.

- Features

- Image preview + metadata view: See the source image on the left and the extracted JSON/summary on the right.

- Flexible input:

- Pass an image path on the command line (python sd_meta_extract_gui.py myimage.png)

- Or simply drag and drop a .png/.jpg onto the window

- Or use the Load button / File → Open menu

# Save / Export:

- Save creates a .json sidecar file next to the image

- Export JSON… lets you pick a custom filename and location

- Settings tab:

- Adjust maximum prompt length in summaries

- Enable/disable automatic sidecar saving on load

- Optionally rebuild missing JPEG metadata from sibling PNG files

- Configure which file extensions appear in the Open dialog



MIT License. Do whatever you want, just don’t blame me if zombies eat your files. 🧟
