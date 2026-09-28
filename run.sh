#!/usr/bin/env bash
# macOS / Linux launcher. Double-click run.command on a Mac, or run ./run.sh in Terminal.
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  echo "First run: creating a private Python environment..."
  python3 -m venv .venv || { echo "Could not create .venv. Install Python 3.10+ from python.org."; read -p "Press Enter to close"; exit 1; }
fi
source .venv/bin/activate
python3 -m pip install --upgrade pip >/dev/null 2>&1
echo "Installing requirements (first time takes a few minutes)..."
if ! python3 -m pip install -r requirements.txt; then
  echo; echo "Install failed. See the error above."; read -p "Press Enter to close"; exit 1
fi
echo "Starting the app. Leave this window open; close it to stop the app."
python3 -m streamlit run app.py
