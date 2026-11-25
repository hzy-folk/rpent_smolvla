# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""RPent 兼容的 JSON-RPC over HTTP 传输层。

线格式与 ``rpent/utils/rpc/http_rpc.py`` 逐字段一致，因此 RPent 的
``HttpRpcClient`` / ``make_rpc_client`` 可以直接连本服务：

    请求  POST /call   {"method": str, "args": [...], "kwargs": {...}, "session_id": str|null}
    响应  HTTP 200     {"ok": true, "result": ...}
                       {"ok": false, "error": str, "traceback": str}
    ndarray            {"__ndarray__": <base64>, "dtype": "float32", "shape": [1, 8, 7]}
    numpy 标量          {"__npscalar__": <value>, "dtype": "float32"}

失败也返回 200，错误写在响应体里 —— 与 RPent 服务端行为一致。
"""

from __future__ import annotations

import base64
import json
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Mapping

import numpy as np

# ---------------------------------------------------------------------------
# numpy <-> JSON
# ---------------------------------------------------------------------------


class NumpyEncoder(json.JSONEncoder):
    """与 RPent 相同的 numpy 编码规则。"""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return {
                "__ndarray__": base64.b64encode(
                    np.ascontiguousarray(obj).tobytes()
                ).decode("ascii"),
                "dtype": str(obj.dtype),
                "shape": list(obj.shape),
            }
        if isinstance(obj, np.generic):
            item = obj.item()
            if obj.dtype.kind in "biuf" and isinstance(item, (bool, int, float)):
                return {"__npscalar__": item, "dtype": str(obj.dtype)}
        return super().default(obj)


def from_json(obj: Any) -> Any:
    """把带 tag 的 dict 还原成 ndarray / numpy 标量。"""
    if isinstance(obj, dict):
        if "__ndarray__" in obj and set(obj) <= {"__ndarray__", "dtype", "shape"}:
            raw = base64.b64decode(obj["__ndarray__"])
            arr = np.frombuffer(raw, dtype=obj.get("dtype"))
            return arr.reshape(obj.get("shape", (-1,))).copy()
        if "__npscalar__" in obj and set(obj) <= {"__npscalar__", "dtype"}:
            return np.dtype(obj["dtype"]).type(obj["__npscalar__"])
        return {k: from_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [from_json(v) for v in obj]
    return obj


def dumps(response: Mapping[str, Any]) -> bytes:
    return json.dumps(response, cls=NumpyEncoder).encode("utf-8")


# ---------------------------------------------------------------------------
# 服务端
# ---------------------------------------------------------------------------


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A002
        return

    def _write(self, payload: bytes, *, content_type: str = "application/json") -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        try:
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self) -> None:  # noqa: N802
        # 方便手工 curl 探活； RPent 只用 POST /call。
        if self.path in ("/healthz", "/"):
            self._write(dumps({"ok": True, "result": {"status": "ok"}}))
        else:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/call":
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return

        length = int(self.headers.get("Content-Length", 0) or 0)
        body = self.rfile.read(length) if length > 0 else b""
        try:
            request = json.loads(body)
            method = request["method"]
            args = tuple(from_json(v) for v in request.get("args", []))
            kwargs = {k: from_json(v) for k, v in (request.get("kwargs") or {}).items()}
            result = self.server.dispatch(method, args, kwargs)  # type: ignore[attr-defined]
            response: dict[str, Any] = {"ok": True, "result": result}
        except Exception as exc:  # noqa: BLE001 - 协议要求把异常转成响应体
            response = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": "".join(
                    traceback.format_exception(type(exc), exc, exc.__traceback__)
                ),
            }
        self._write(dumps(response))


class RpcServer(ThreadingHTTPServer):
    """注册若干方法并串行化执行的 HTTP RPC 服务。"""

    allow_reuse_address = True
    # server_close must wait until the shutdown RPC has sent its response.
    daemon_threads = False

    def __init__(
        self,
        address: tuple[str, int],
        methods: Mapping[str, Callable[..., Any]],
        *,
        lock: threading.Lock | None = None,
    ) -> None:
        super().__init__(address, _Handler)
        self._methods = dict(methods)
        self._lock = lock or threading.Lock()

    @property
    def url(self) -> str:
        host, port = self.server_address[:2]
        shown = "127.0.0.1" if host in ("0.0.0.0", "::") else host
        return f"http://{shown}:{port}"

    def dispatch(self, method: str, args: tuple, kwargs: dict) -> Any:
        handler = self._methods.get(method)
        if handler is None:
            raise KeyError(
                f"unknown RPC method {method!r}; available: {sorted(self._methods)}"
            )
        with self._lock:
            return handler(*args, **kwargs)
