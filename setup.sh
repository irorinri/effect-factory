#!/usr/bin/env bash
# Effect Factory - one-time setup for macOS and Linux.
# Creates a private Python environment in .venv and a desktop launcher
# (macOS: "Effect Factory.app" on the desktop; Linux: a .desktop launcher on
# the desktop and in the application menu). Run it again after moving this
# folder; "./setup.sh --remove" removes the launcher.
set -e
cd "$(dirname "$0")"
echo
echo "  Effect Factory setup"
echo "  ===================="
echo

if [ "$1" = "--remove" ]; then
  if [ -x .venv/bin/python ]; then .venv/bin/python tools/install_desktop.py --remove; else echo "Nothing to remove."; fi
  exit 0
fi

ok() { "$1" -c 'import sys, tkinter; sys.exit(0 if sys.version_info >= (3, 10) else 1)' >/dev/null 2>&1; }

PY=""
for cand in python3.13 python3.12 python3.11 python3.10 python3 \
            /Library/Frameworks/Python.framework/Versions/Current/bin/python3; do
  if command -v "$cand" >/dev/null 2>&1 && ok "$cand"; then PY="$cand"; break; fi
done
if [ -z "$PY" ]; then
  echo "Python 3.10 or later with Tkinter was not found."
  if [ "$(uname)" = "Darwin" ]; then
    echo "Install it from https://www.python.org/downloads/macos/ and run this setup again."
    open "https://www.python.org/downloads/macos/" 2>/dev/null || true
  else
    echo "Install it with your package manager, e.g.:  sudo apt install python3 python3-venv python3-tk"
  fi
  exit 1
fi

if [ ! -x .venv/bin/python ] || ! ok .venv/bin/python; then
  echo "Creating a private Python environment (.venv) ..."
  "$PY" -m venv --clear .venv
fi
echo "Installing numpy and Pillow ..."
.venv/bin/python -m pip install --disable-pip-version-check -q -r requirements.txt
echo
.venv/bin/python tools/install_desktop.py

if ! .venv/bin/python -c "import sys; sys.path.insert(0, '.'); from efx.export import find_ffmpeg; sys.exit(0 if find_ffmpeg() else 1)"; then
  echo
  echo "ffmpeg was not found. It is needed for MP4 / MOV export (PNG sequences work without it)."
  if [ "$(uname)" = "Darwin" ]; then echo "Install it with Homebrew:  brew install ffmpeg"
  else echo "Install it with your package manager, e.g.:  sudo apt install ffmpeg"; fi
fi
echo
echo "Done."
