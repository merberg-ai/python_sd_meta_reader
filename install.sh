#!/usr/bin/env bash
set -euo pipefail

# Create virtual env and install deps
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo
echo "✅ Installed. Activate with:"
echo "    source .venv/bin/activate"
echo "Run:"
echo "    python sd_meta_extract.py /path/to/images --rebuild-jpeg-sidecar"
