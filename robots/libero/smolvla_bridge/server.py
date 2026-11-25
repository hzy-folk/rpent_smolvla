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

"""Standalone SmolVLA service compatible with RPent's LIBERO VLA client.

Run this file with the isolated SmolVLA interpreter, or use run_server.sh.
Use profiles/mock_osc7.json with --mock --self-test for a CPU-only check.
"""

from __future__ import annotations

import argparse
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any

if __package__:
    from .bridge.backend import build_backend
    from .bridge.demo import fake_obs
    from .bridge.profiles import Profile
    from .bridge.protocol import RpcServer
else:
    from bridge.backend import build_backend
    from bridge.demo import fake_obs
    from bridge.profiles import Profile
    from bridge.protocol import RpcServer


def _log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RPent 兼容的 SmolVLA VLA 服务（vla.predict over HTTP JSON-RPC）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--profile",
        default=str(Path(__file__).parent / "profiles/libero_pro_salad.json"),
        help="配置档 JSON 路径",
    )
    parser.add_argument("--host", default="127.0.0.1", help="绑定地址")
    parser.add_argument("--port", type=int, default=8320, help="端口（0 = 随机）")
    parser.add_argument(
        "--device", default=None, help="覆盖 profile 的 device (cpu/cuda)"
    )
    parser.add_argument(
        "--dtype", default=None, help="覆盖 profile 的 dtype (float32/bfloat16/float16)"
    )
    parser.add_argument("--chunk-size", type=int, default=None, help="覆盖 chunk_size")
    parser.add_argument(
        "--exec-steps", type=int, default=None, help="覆盖每次返回给环境的步数"
    )
    parser.add_argument("--num-steps", type=int, default=None, help="覆盖 flow 步数")
    parser.add_argument(
        "--state-adapter",
        default=None,
        choices=["strict", "pad", "truncate"],
        help="观测状态维度不符时的策略（pad/truncate 仅用于链路验证）",
    )
    parser.add_argument(
        "--action-adapter",
        default=None,
        choices=["strict", "pad", "truncate"],
        help="动作维度不符时的策略（pad/truncate 仅用于链路验证）",
    )
    parser.add_argument(
        "--allow-shape-mismatch",
        action="store_true",
        help="允许动作维度不符时自动截断/补零（非物理，仅供联调）",
    )
    parser.add_argument(
        "--mock", action="store_true", help="不加载模型，返回占位动作块"
    )
    parser.add_argument("--self-test", action="store_true", help="构建一次推理后退出")
    parser.add_argument(
        "--self-test-state-dim",
        type=int,
        default=None,
        help="自检时伪造的状态维度（默认与 profile 一致；可填 8 模拟 RPent 实际发送值）",
    )
    return parser.parse_args(argv)


def _overrides(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "device": args.device,
        "dtype": args.dtype,
        "chunk_size": args.chunk_size,
        "exec_steps": args.exec_steps,
        "num_steps": args.num_steps,
        "state_adapter": args.state_adapter,
        "action_adapter": args.action_adapter,
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    profile_path = Path(args.profile)
    if not profile_path.is_file():
        print(f"找不到配置档: {profile_path}", file=sys.stderr)
        return 2

    try:
        profile = Profile.load(profile_path, require_assets=not args.mock)
    except (ValueError, OSError, KeyError) as exc:
        print(f"配置错误: {exc}", file=sys.stderr)
        return 2
    backend = build_backend(
        profile,
        mock=args.mock,
        overrides=_overrides(args),
        allow_shape_mismatch=args.allow_shape_mismatch,
        log=_log,
    )

    _log(f"配置档 {profile.name}: {profile.source}")
    if profile.state_schema:
        _log(f"  state: {profile.state_schema}")
    if profile.action_schema:
        _log(f"  action: {profile.action_schema}")

    try:
        backend.profile.validate(require_assets=not args.mock)
        backend.load()
    except Exception as exc:  # noqa: BLE001
        print(f"\n加载失败: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    if args.self_test:
        try:
            obs = fake_obs(backend.profile, state_dim=args.self_test_state_dim)
            actions = backend.predict(obs)
        except Exception as exc:  # noqa: BLE001
            print(f"\n自检失败: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
        print("\n自检通过：")
        print(f"  动作形状 {actions.shape} dtype={actions.dtype}")
        print(f"  首步 {actions[0, 0].round(4).tolist()}")
        backend.close()
        return 0

    def _shutdown() -> dict[str, bool]:
        threading.Thread(target=server.shutdown, daemon=True).start()
        return {"ok": True}

    methods = {
        # RPent 的 wait_for_ready() 会先调 healthz
        "healthz": lambda: {"status": "ok"},
        "vla.predict": backend.predict,
        "vla.status": backend.status,
        "vla.get_model_meta": backend.model_meta,
        # 某些平台（如 RoboCasa）会用 session-aware 客户端；这里接受但无状态，
        # 因为 SmolVLA 推理是无历史的（每次调用独立）。
        "session.register": lambda *a, **kw: {
            "ok": True,
            "session_id": kw.get("session_id") or (a[0] if a else None),
        },
        "session.close": lambda *a, **kw: {
            "ok": True,
            "session_id": kw.get("session_id") or (a[0] if a else None),
        },
        "shutdown": _shutdown,
    }
    server = RpcServer((args.host, args.port), methods)

    def _on_signal(signum: int, _frame: Any) -> None:
        _log(f"收到信号 {signum}，关闭服务")
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    _log(f"服务就绪: {server.url}/call")
    _log(f"  RPent 接法: --am-backend pi05 --am-endpoint {server.url}")
    _log(f"  探活: curl {server.url}/healthz")
    try:
        server.serve_forever(poll_interval=0.2)
    finally:
        server.server_close()
        backend.close()
        _log("已退出")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
