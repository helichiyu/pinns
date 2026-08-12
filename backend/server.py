"""Tornado web server: serves the frontend and runs experiment queues.

Run from the project root:  python backend/server.py
Then open http://localhost:8770 in a browser.
"""

import base64
import io
import json
import os
import re
import subprocess
import sys
import threading
import time
import webbrowser

import tornado.ioloop
import tornado.web
import tornado.websocket

# Ensure sibling modules (config, train, etc.) are importable.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config  # noqa: E402
from losses import source_contour  # noqa: E402
from train import METRIC_PREFIX, RESULT_PREFIX, load_source  # noqa: E402
from visualization import plot_source_contour_explanation  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND = os.path.join(ROOT, "frontend")
IMAGES = os.path.join(ROOT, "images")
PORT = 8770

IMAGE_RE = re.compile(r"\.(png|jpg|jpeg|bmp|tif|tiff)$", re.IGNORECASE)


class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.render(os.path.join(FRONTEND, "index.html"))


class ImagesHandler(tornado.web.RequestHandler):
    def get(self):
        if not os.path.isdir(IMAGES):
            self.write({"images": []})
            return
        files = sorted(f for f in os.listdir(IMAGES) if IMAGE_RE.search(f))
        self.write({"images": files})


class PreviewHandler(tornado.web.RequestHandler):
    def post(self):
        req = json.loads(self.request.body)
        source, padding = load_source(req["image"], "cpu", req["expand"])
        sigma = float(req.get("contour_sigma", config.CONTOUR_SIGMA))
        threshold = float(req.get("contour_threshold", config.CONTOUR_THRESHOLD))
        mask = source_contour(source, sigma, threshold)
        ratio = mask.sum().item() / mask.numel()
        buf = io.BytesIO()
        plot_source_contour_explanation(source, mask, padding, ratio, buf)
        self.write({"image": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(),
                    "ratio": ratio})


class ExperimentSocket(tornado.websocket.WebSocketHandler):
    """Sequential experiment runner with stop support."""

    # Track live browser connections so the server can exit when idle.
    _connections = 0
    _last_disconnect = None

    def open(self):
        self.loop = tornado.ioloop.IOLoop.current()
        self.proc = None
        self.stopped = False
        self.remaining = []
        ExperimentSocket._connections += 1

    def on_message(self, raw):
        msg = json.loads(raw)
        t = msg.get("type")
        if t == "start":
            self.stopped = False
            self.remaining = list(enumerate(msg["experiments"]))
            self._next()
        elif t == "finish_current":
            self._finish_current()
        elif t == "stop":
            self._stop()

    def _finish_current(self):
        """结束当前这组实验：让 train.py 提前收尾出图，之后队列继续下一组。"""
        proc = self.proc
        if not proc or proc.poll() is not None:
            return
        try:
            proc.stdin.write("stop\n")
            proc.stdin.flush()
        except (OSError, ValueError):
            return
        self._send({"type": "finishing"})

    def _stop(self):
        self.stopped = True
        self.remaining = []
        if self.proc:
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
        cmd += ["--share-histogram", str(exp["share_histogram"])]
        cmd += ["--share-background", str(exp["share_background"])]
        cmd += ["--share-input-output", str(exp["share_input_output"])]
        if exp["enable_input_output_loss"]:
            cmd += ["--enable-input-output-loss"]
        cmd += ["--iterations", str(exp["iterations"])]
        cmd += ["--stream-metrics"]
        # 子进程默认按系统区域编码（GBK）写 stdout，这里按 utf-8 读，
        # 所以必须强制子进程也用 utf-8，否则中文日志会解码失败。
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        # stdin 用来发「结束当前」指令，让训练提前收尾而不是被 kill。
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, env=env,
                                cwd=ROOT, text=True, encoding="utf-8", bufsize=1)
        self.proc = proc
        output_dir = None
        stopped_early = False
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
                    payload = json.loads(line[len(RESULT_PREFIX):])
                    output_dir = payload["output_dir"]
                    stopped_early = payload.get("stopped_early", False)
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
            result = {"type": status, "index": index, "code": proc.returncode,
                      "output_dir": output_dir, "stopped_early": stopped_early}
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
