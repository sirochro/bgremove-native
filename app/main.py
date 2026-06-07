"""bgremove-native entry point: launch FastAPI server, open pywebview window."""
from __future__ import annotations

import json
import socket
import sys
import threading
import time
from typing import Optional

import uvicorn
import webview
from webview.dom import DOMEventHandler

from app.server import app

WINDOW_TITLE = "bgremove-native"
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 960


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def run_server(port: int) -> None:
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning", access_log=False)


def wait_for_server(port: int, timeout: float = 30.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.2)
    return False


class JsAPI:
    """Exposed to JS as ``window.pywebview.api``. Video tab uses this for file dialogs."""

    def __init__(self) -> None:
        self._window: Optional[webview.Window] = None

    def open_input_video(self) -> Optional[str]:
        if self._window is None:
            return None
        result = self._window.create_file_dialog(
            webview.OPEN_DIALOG,
            allow_multiple=False,
            file_types=(
                "Video files (*.mp4;*.mov;*.mkv;*.webm;*.avi;*.m4v)",
                "All files (*.*)",
            ),
        )
        if not result:
            return None
        return result[0] if isinstance(result, (list, tuple)) else result

    def save_output_video(
        self, suggested_name: str = "output.mov", ext: str = "mov"
    ) -> Optional[str]:
        if self._window is None:
            return None
        label = {"mov": "MOV (*.mov)", "webm": "WebM (*.webm)"}.get(ext, f"*.{ext}")
        result = self._window.create_file_dialog(
            webview.SAVE_DIALOG,
            save_filename=suggested_name,
            file_types=(label, "All files (*.*)"),
        )
        if not result:
            return None
        return result if isinstance(result, str) else result[0]

    def save_output_image(self, suggested_name: str = "output.png") -> Optional[str]:
        if self._window is None:
            return None
        result = self._window.create_file_dialog(
            webview.SAVE_DIALOG,
            save_filename=suggested_name,
            file_types=("PNG (*.png)", "All files (*.*)"),
        )
        if not result:
            return None
        return result if isinstance(result, str) else result[0]

    def open_input_image(self) -> Optional[str]:
        if self._window is None:
            return None
        result = self._window.create_file_dialog(
            webview.OPEN_DIALOG,
            allow_multiple=False,
            file_types=(
                "Image files (*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff;*.webp;*.gif)",
                "All files (*.*)",
            ),
        )
        if not result:
            return None
        return result[0] if isinstance(result, (list, tuple)) else result


def _first_dropped_path(event: dict) -> Optional[str]:
    files = (event.get("dataTransfer") or {}).get("files") or []
    for f in files:
        full = f.get("pywebviewFullPath")
        if full:
            return full
    return None


def _bind_drop_handlers(window: webview.Window) -> None:
    """Register native drop handlers so dropped files surface their real OS path."""

    def on_video_drop(e: dict) -> None:
        path = _first_dropped_path(e)
        if not path:
            return
        window.evaluate_js(f"window.__princeOnVideoDrop({json.dumps(path)})")

    def on_image_drop(e: dict) -> None:
        path = _first_dropped_path(e)
        if not path:
            return
        window.evaluate_js(f"window.__princeOnImageDrop({json.dumps(path)})")

    v_drop = window.dom.get_element("#v-drop")
    if v_drop is not None:
        v_drop.events.drop += DOMEventHandler(on_video_drop, True, True)
    img_drop = window.dom.get_element("#img-drop")
    if img_drop is not None:
        img_drop.events.drop += DOMEventHandler(on_image_drop, True, True)


def main() -> int:
    port = find_free_port()
    server_thread = threading.Thread(target=run_server, args=(port,), daemon=True)
    server_thread.start()

    if not wait_for_server(port):
        print(f"[bgremove-native] server failed to start on port {port}", file=sys.stderr)
        return 1

    api = JsAPI()
    window = webview.create_window(
        WINDOW_TITLE,
        url=f"http://127.0.0.1:{port}/",
        width=WINDOW_WIDTH,
        height=WINDOW_HEIGHT,
        js_api=api,
    )
    api._window = window
    webview.start(_bind_drop_handlers, window, private_mode=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
