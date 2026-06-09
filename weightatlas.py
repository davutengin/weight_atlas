#!/usr/bin/env python3
"""WeightAtlas entry point — starts the server and opens the browser."""
import sys
import os
import time
import threading
import webbrowser
import subprocess
from pathlib import Path

HOST = "127.0.0.1"
PORT = 8000
URL = f"http://{HOST}:{PORT}"

BACKEND_DIR = Path(__file__).parent / "backend"


def open_browser():
    time.sleep(1.5)
    webbrowser.open(URL)


def main():
    os.chdir(BACKEND_DIR)
    sys.path.insert(0, str(BACKEND_DIR))

    print(f"  WeightAtlas starting...")
    print(f"  {URL}")
    print(f"  Press Ctrl+C to stop\n")

    threading.Thread(target=open_browser, daemon=True).start()

    import uvicorn
    uvicorn.run("app.main:app", host=HOST, port=PORT, reload=False)


if __name__ == "__main__":
    main()
