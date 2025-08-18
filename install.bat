@echo off
setlocal
python -m venv .venv
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt

echo.
echo ✅ Installed. Activate with:
echo     call .venv\Scripts\activate
echo Run:
echo     python sd_meta_extract.py C:\path\to\images --rebuild-jpeg-sidecar
endlocal
