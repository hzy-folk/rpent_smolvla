# RPent + SmolVLA：LIBERO-Pro 沙拉酱任务

[English](README.en.md)

本目录把 SmolVLA 包装成 RPent 可调用的 `vla.predict` HTTP 服务。代码、配置、启动脚本和一份精简历史记忆均随 RPent 发布，不需要原来的外部桥接仓库。高层 API 负责观察、定位、选择工具、搬运与释放；SmolVLA 提供抓取动作块。

本流程对应 `--libero-type pro --suite libero_object_task --task 8 --seed 0`：把沙拉酱放入篮子。已有一次完整成功记录，也有相同 seed 的失败记录；以下步骤用于复现调用流程，不保证每次任务成功。

## 准备环境

使用 Linux、NVIDIA GPU 和 EGL。开发机器为 8 GB 显存的 RTX 5060 Laptop，SmolVLA + SAM3 同时加载约占 5.6 GB，具体占用随依赖和配置变化。RPent 使用 Python 3.11，桥接使用独立 Python 3.10 或 3.11 环境，避免 Transformers 版本冲突。

从 **RPent 仓库根目录**执行。已有 RPent 环境可直接复用；新机器按 [LIBERO 安装指南](../../../docs/source-en/rst_source/simulators/libero.rst) 准备系统库和模型资源后安装：

```bash
uv venv --python 3.11 .venv-rpent-libero-pro
uv pip install --python .venv-rpent-libero-pro/bin/python -e '.[libero-pro]' --torch-backend cu128

uv venv --python 3.10 .venv-smolvla
uv pip install --python .venv-smolvla/bin/python \
  -r robots/libero/smolvla_bridge/requirements.txt --torch-backend cu128
```

不要把桥接 requirements 安装到 RPent 环境。桥接固定使用 [LeRobot 0.3.3](https://pypi.org/project/lerobot/0.3.3/)；本机实际使用的 SmolVLA 核心文件已与该公开发行包比对一致。主仓库的 `flywheel` 依赖不等于已安装桥接所需的完整模型依赖。


仿真资产和 SAM3 也需要单独准备。完成上述安装后，从仓库根目录执行：

```bash
.venv-rpent-libero-pro/bin/liberopro-download-assets --skip-existing
uv pip install --python .venv-rpent-libero-pro/bin/python modelscope
.venv-rpent-libero-pro/bin/modelscope download facebook/sam3 \
  --local-dir checkpoints/sam3
```

这些步骤可能需要网络访问及上游模型的访问权限。若已有文件，可以在本地配置中直接引用，无需重复下载。

## 准备权重与配置

推荐布局如下；`checkpoints/` 已被 Git 忽略：

```text
checkpoints/
  sam3/sam3.pt
  smolvla/
    checkpoint_015000.pt
    base/config.json
    base/model.safetensors
    vlm/config.json
    vlm/tokenizer.json
    vlm/tokenizer_config.json
    vlm/preprocessor_config.json
    vlm/processor_config.json
    vlm/special_tokens_map.json
    vlm/chat_template.json
```

需要三类 SmolVLA 资产：官方基础权重、VLM 的处理器/分词器文件，以及本项目微调的任务 checkpoint。任务 checkpoint 是包含 `trainable`、`stats`、`state_dim`、`action_dim` 的 PyTorch 字典，不能直接替换成任意 Hugging Face 模型目录。SAM3 权重按 LIBERO 安装指南准备。

基础模型和处理器可用固定版本下载；下载只在准备阶段联网，桥接推理使用本地文件：

```bash
.venv-smolvla/bin/hf download lerobot/smolvla_base \
  --revision 5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6 \
  --include config.json model.safetensors \
  --local-dir checkpoints/smolvla/base

.venv-smolvla/bin/hf download HuggingFaceTB/SmolVLM2-500M-Video-Instruct \
  --revision 4cde9d208e19519c36917aa170fffc7e381aa0ad \
  --include '*.json' \
  --local-dir checkpoints/smolvla/vlm

# 保留训练时实际使用的 chat template：
cp robots/libero/smolvla_bridge/processor_overrides/*.json checkpoints/smolvla/vlm/
```

`processor_overrides/` 保存训练时使用的两份处理器配置，只与官方配置的 chat template 字段不同。下载后需要覆盖这两份小型 JSON，以保持训练输入格式；不包含模型权重。

**微调的 `checkpoint_015000.pt` 尚未随代码发布，也没有公开下载地址。** 发布者需要另外通过 Hugging Face 或 GitHub Release 提供该文件，并填写 `assets_manifest.json` 的 `checkpoint_download_url`。在此之前，其他人可以完成 mock 自检，但必须从发布者处取得该文件才能运行真实任务。无需上传训练数据，也不要把约 200 MB 的 checkpoint 直接提交到普通 Git 历史。

准备好后检查文件哈希：

```bash
python3 robots/libero/smolvla_bridge/check_assets.py --root checkpoints/smolvla
cp robots/libero/smolvla_bridge/env.example.sh robots/libero/smolvla_bridge/env.local.sh
# 按实际位置编辑 env.local.sh，再从仓库根目录加载：
source robots/libero/smolvla_bridge/env.local.sh
```

`assets_manifest.json` 记录开发时实际使用资产的大小和 SHA256，不包含权重。若使用自己的另一份匹配 checkpoint，应保留自己的资产清单，不把哈希差异忽略成同一权重。`env.local.sh` 已被 Git 忽略，不要将凭据填入已跟踪文件。

## 先自检，再启动桥接

无需权重或 GPU 的接口/形状检查：

```bash
SMOLVLA_PYTHON="$PWD/.venv-rpent-libero-pro/bin/python" \
  bash robots/libero/smolvla_bridge/run_server.sh \
  --profile "$PWD/robots/libero/smolvla_bridge/profiles/mock_osc7.json" \
  --mock --self-test
```

预期动作形状为 `(1, 5, 7)`。mock 返回占位动作，不能用于评测。

真实模型自检会加载权重并对假观测执行一次推理；这不验证真实场景成功：

```bash
bash robots/libero/smolvla_bridge/run_server.sh --self-test
```

终端一正式启动模型服务：

```bash
bash robots/libero/smolvla_bridge/run_server.sh --port 8320
```

看到“服务就绪”后保持该终端运行。`curl http://127.0.0.1:8320/healthz` 可检查服务是否存活。用 Ctrl+C 关闭桥接。

## 运行 seed 0，使用历史记忆

终端二从仓库根目录加载同一份本地配置，设置你自己的视觉 API 模型与接口：

```bash
source robots/libero/smolvla_bridge/env.local.sh
export OPENAI_BASE_URL='https://your-provider.example/v1'
export RPENT_API_MODEL='openai-chat:your-vision-model'
read -rsp 'API Key: ' OPENAI_API_KEY
echo
export OPENAI_API_KEY

bash robots/libero/smolvla_bridge/run_episode.sh
```

API 模型需支持图像、工具调用及当前命令使用的 reasoning effort；服务商支持情况应单独确认。RPent 自动启动 LIBERO 和 SAM3，并连接已经运行的 SmolVLA。`--am-backend pi05` 只复用已有客户端的编码和 RPC 协议，实际模型由 8320 端口决定。

启动脚本默认读取本目录的 `memory/`：其中包含实际成功 seed-0 的 audit/recipe、抓取和容器定位经验。这是精简可发布子集，不是开发机完整记忆库；可用 `RPENT_MEMORY_DIR=/path/to/memory` 指向自己的完整记忆。不要直接照搬历史坐标。

三个新增功能默认关闭，需要时单独传入：

```bash
bash robots/libero/smolvla_bridge/run_episode.sh \
  --episode-brief --compact-observation-images --assess-movement
```

可以通过 `RPENT_PYTHON`、`SMOLVLA_PYTHON` 选择已有解释器，通过 `RPENT_AM_ENDPOINT` 改模型服务地址，通过 `RPENT_OUTPUT_DIR` 改输出目录。附加 RPent 参数会原样传递，例如 `--seed 1` 或 `--reasoning-effort medium`；初次复现先使用默认 seed 0。

结果保存在新的 `logs/smolvla_seed0_<时间>/`，包含 transcript、状态、图片、轨迹和视频。任务成功以 `environment_success=true` / 原生 `terminated=true` 为准，不能用进程退出码或工具抓取启发式代替。RPent 自动清理自己启动的环境与 SAM3；独立桥接仍需在终端一关闭。

## 模型输入约定与验证边界

profile 保持训练约定：8 维末端/夹爪状态、7 维 OSC 增量动作、两路 256×256 图片、chunk 16、每次返回前 5 步、flow 10 步。桥接只改变图像布局和数值范围，不额外旋转图像。

`task_prompt` **有意固定为 `pick up the salad dressing`**；高层工具调用里的其他文字也会被替换成该训练指令。这是当前小模型的使用约束，不应为了让提示词“更灵活”而删除。主任务仍由高层 API 完成。

迁入后已覆盖 mock CLI、本地真实 HTTP 客户端、提示词覆盖与图片方向检查；真实 checkpoint 推理使用已有环境和公开 LeRobot 0.3.3 源码进行检查。尚未在一台全新机器完整安装并重跑 episode。基础调用可运行不代表任务稳定成功，固定环境 seed 也不固定 API 决策或 VLA 随机噪声。

[详细验证记录](VALIDATION.md)
