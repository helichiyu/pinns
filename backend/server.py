"""Tornado web server: serves the frontend and runs experiment queues.

Run from the project root:  python backend/server.py
Then open http://localhost:8770 in a browser.
"""

import base64
import ctypes
import io
import json
import os
import re
import subprocess
import sys
import threading
import time
import webbrowser

import numpy as np
import tornado.ioloop
import tornado.web
import tornado.websocket
from PIL import Image

# Ensure sibling modules (config, train, etc.) are importable.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config  # noqa: E402
from losses import dynamic_contour  # noqa: E402
from train import METRIC_PREFIX, RESULT_PREFIX, load_source  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND = os.path.join(ROOT, "frontend")
PORT = 8770

IMAGE_RE = re.compile(r"\.(png|jpg|jpeg|bmp|tif|tiff)$", re.IGNORECASE)

# ---- Windows NT API for process suspend/resume ----
# Must set argtypes/restype explicitly: HANDLE is pointer-sized on 64-bit.
# Without this, ctypes defaults to c_int (32-bit) and truncates the handle,
# causing NtSuspendProcess to receive an invalid handle and silently no-op.
_k32 = ctypes.windll.kernel32
_k32.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_bool, ctypes.c_uint32]
_k32.OpenProcess.restype = ctypes.c_void_p
_k32.CloseHandle.argtypes = [ctypes.c_void_p]
_k32.CloseHandle.restype = ctypes.c_bool

_ntdll = ctypes.windll.ntdll
_ntdll.NtSuspendProcess.argtypes = [ctypes.c_void_p]
_ntdll.NtSuspendProcess.restype = ctypes.c_int32
_ntdll.NtResumeProcess.argtypes = [ctypes.c_void_p]
_ntdll.NtResumeProcess.restype = ctypes.c_int32

_PROCESS_SUSPEND_RESUME = 0x0800


def _suspend_process(pid):
    handle = _k32.OpenProcess(_PROCESS_SUSPEND_RESUME, False, pid)
    if handle:
        _ntdll.NtSuspendProcess(handle)
        _k32.CloseHandle(handle)
        return True
    return False


def _resume_process(pid):
    handle = _k32.OpenProcess(_PROCESS_SUSPEND_RESUME, False, pid)
    if handle:
        _ntdll.NtResumeProcess(handle)
        _k32.CloseHandle(handle)
        return True
    return False


class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.render(os.path.join(FRONTEND, "index.html"))


class ImagesHandler(tornado.web.RequestHandler):
    def get(self):
        files = sorted(f for f in os.listdir(ROOT) if IMAGE_RE.search(f))
        self.write({"images": files})


class PreviewHandler(tornado.web.RequestHandler):
    def post(self):
        req = json.loads(self.request.body)
        device = "cpu"
        source, _ = load_source(req["image"], device, req["expand"])
        sigma = float(req.get("contour_sigma", config.CONTOUR_SIGMA))
        threshold = float(req.get("contour_threshold", config.CONTOUR_THRESHOLD))
        contour = dynamic_contour(source, sigma, threshold)
        arr = (contour.squeeze().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(arr, mode="L").save(buf, format="PNG")
        self.write({"image": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()})


class ExperimentSocket(tornado.websocket.WebSocketHandler):
    """Sequential experiment runner with stop support."""

    # Track live browser connections so the server can exit when idle.
    _connections = 0
    _last_disconnect = None

    def open(self):
        self.loop = tornado.ioloop.IOLoop.current()
        self.proc = None
        self.paused = False
        self.stopped = False
        self.remaining = []
        ExperimentSocket._connections += 1

    def on_message(self, raw):
        msg = json.loads(raw)
        t = msg.get("type")
        if t == "start":
            self.stopped = False
            self.paused = False
            self.remaining = list(enumerate(msg["experiments"]))
            self._next()
        elif t == "pause":
            if self.proc and not self.paused:
                if _suspend_process(self.proc.pid):
                    self.paused = True
                    self._send({"type": "paused"})
        elif t == "resume":
            if self.proc and self.paused:
                if _resume_process(self.proc.pid):
                    self.paused = False
                    self._send({"type": "resumed"})
        elif t == "stop":
            self._stop()

    def _stop(self):
        self.stopped = True
        self.remaining = []
        if self.proc:
            if self.paused:
                _resume_process(self.proc.pid)
                self.paused = False
            self.proc.kill()

    def _next(self):
        if self.stopped or not self.remaining:
            self._send({"type": "all_done"})
            return
        index, exp = self.remaining.pop(0)
        self._send({"type": "exp_start", "index": index, "config": exp})
        threading.Thread(target=self._run_thread, args=(index, exp), daemon=True).start()

    def _run_thread(self, index, exp):
        loop = self.loop
        cmd = [sys.executable, os.path.join("backend", "train.py")]
        cmd += ["--image", exp["image"]]
        cmd += ["--expand", str(exp["expand"])]
        cmd += ["--contour-sigma", str(exp["contour_sigma"])]
        cmd += ["--contour-threshold", str(exp["contour_threshold"])]
        cmd += ["--iterations", str(exp["iterations"])]
        cmd += ["--stream-metrics"]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                cwd=ROOT, text=True, encoding="utf-8", bufsize=1)
        self.proc = proc
        output_dir = None
        for line in proc.stdout:
            line = line.rstrip("\n")
            if line.startswith(METRIC_PREFIX):
                try:
                    metric = json.loads(line[len(METRIC_PREFIX):])
                    metric["type"] = "metric"
                    metric["index"] = index
                    loop.add_callback(self._send, metric)
                except json.JSONDecodeError:
                    pass
            elif line.startswith(RESULT_PREFIX):
                try:
                    output_dir = json.loads(line[len(RESULT_PREFIX):])["output_dir"]
                except (json.JSONDecodeError, KeyError):
                    pass
            else:
                loop.add_callback(self._send, {"type": "terminal", "index": index, "text": line})
        proc.wait()
        self.proc = None
        if self.stopped:
            result = {"type": "exp_stopped", "index": index}
        else:
            status = "exp_done" if proc.returncode == 0 else "exp_failed"
            result = {"type": status, "index": index,
                      "code": proc.returncode, "output_dir": output_dir}
        loop.add_callback(self._on_finished, result)

    def _send(self, msg):
        self.write_message(json.dumps(msg))

    def _on_finished(self, result):
        self._send(result)
        if self.stopped:
            self._send({"type": "all_done"})
        else:
            self._next()

    def on_close(self):
        self._stop()
        ExperimentSocket._connections -= 1
        if ExperimentSocket._connections <= 0:
            ExperimentSocket._connections = 0
            ExperimentSocket._last_disconnect = time.time()


# Auto-exit policy: shut the server down once no browser is connected,
# so closing the browser tab/window also stops the backend.
_STARTUP_GRACE = 60  # seconds after start before idle shutdown is allowed
_IDLE_TIMEOUT = 10   # seconds with no WS connection before auto-exit
_start_time = time.time()


def _idle_check():
    if ExperimentSocket._connections > 0:
        return
    if time.time() - _start_time < _STARTUP_GRACE:
        return
    deadline = ExperimentSocket._last_disconnect or _start_time
    if time.time() - deadline >= _IDLE_TIMEOUT:
        print("No browser connected for %ds, shutting down." % _IDLE_TIMEOUT)
        tornado.ioloop.IOLoop.current().stop()


def make_app():
    return tornado.web.Application([
        (r"/", MainHandler),
        (r"/api/images", ImagesHandler),
        (r"/api/preview", PreviewHandler),
        (r"/ws", ExperimentSocket),
        (r"/static/(.*)", tornado.web.StaticFileHandler, {"path": FRONTEND}),
        (r"/results/(.*)", tornado.web.StaticFileHandler, {"path": os.path.join(ROOT, "results")}),
    ])


if __name__ == "__main__":
    _start_time = time.time()
    app = make_app()
    app.listen(PORT)
    url = "http://localhost:{}".format(PORT)
    print("Server running at " + url)
    # Open the default browser once the event loop is running.
    tornado.ioloop.IOLoop.current().add_callback(lambda: webbrowser.open(url))
    # Auto-exit when the browser is closed.
    tornado.ioloop.PeriodicCallback(_idle_check, 1000).start()
    tornado.ioloop.IOLoop.current().start()
