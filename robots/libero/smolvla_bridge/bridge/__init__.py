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

"""Standalone RPent-compatible SmolVLA bridge modules.

模块划分：
    protocol  与 rpent/utils/rpc/http_rpc.py 完全同构的 JSON-RPC 线格式
    profiles  每个 checkpoint 的运行配置（维度、相机、解码步数）
    adapters  RPent 观测 -> SmolVLA 输入，SmolVLA 动作 -> RPent 动作
    backend   lerobot SmolVLA 加载与推理
"""
