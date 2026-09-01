# llm-shell

llama.cpp 的终端风格 GUI Shell（Python + Textual，全中文界面）。以子进程方式拉起官方预编译的 `llama-server.exe`，负责**启动模型与调节引擎参数**；右侧控制台显示服务端口、可用模型列表和 llama-server 实时日志。聊天在浏览器中通过 llama-server 自带的 WebUI（http://127.0.0.1:5801）进行，本程序不做界面内对话。

## 功能

- 终端/CLI 风格布局：状态栏 + 左侧中文参数面板 + 右侧控制台（无底部命令行、无界面内聊天）
- 引擎参数修改后按 F5（或点「加载服务」）重启服务生效；采样参数在 WebUI 中设置
- 自动扫描模型目录下（含子目录）全部 `*.gguf`（排除 mmproj-*），下拉选择，控制台列出大小；面板「模型目录」输入框可改扫描目录（多个用 `;` 分隔），改动后约 0.6s 自动重扫
- **视觉模型自动加载（LM Studio 行为）**：选中主模型后，若其同目录存在含 `mmproj`（去 `.gguf`）的视觉投影文件，启动时自动带上 `--mmproj <路径>`，无需手动选择；控制台与「预估显存」行会提示是否已加载视觉模型。传图片到 WebUI 即可用（不再报 `image input is not supported`）。「高级」区「加载视觉模型」开关默认开启；**关闭则不加载 mmproj**（纯文本省显存，即使同目录有视觉文件也不传 `--mmproj`）
- llama-server stdout 日志逐行转发到右侧控制台（dim 样式），状态栏同步显示最后一行
- 圆形开关（未选 ○ / 选中 ●，选中为蓝色 `$primary`）+ 参数持久化到 `config.json`

## 快速开始

```bat
python H:\Qwen\gui\llm-tui\main.py
rem 等价(在仓库根目录): python -m llm_tui
```

依赖：Python 3.10+、textual>=8.2、httpx>=0.27（本机已安装；新环境执行 `pip install -r requirements.txt`）。

## 布局说明

```
┌──────────────────────────────────────────────────────────────┐
│ [状态栏: 运行中/已停止 / 模型 / ctx / ngl / 线程]              │
├───────────────────────────┬──────────────────────────────────┤
│ 参数设置                  │ LLM 控制台                        │
│ -- 模型 --                │ server : H:\Qwen\gui\llama-server.exe│
│ 模型文件      [下拉选择]   │ WebUI  : http://127.0.0.1:5801    │
│ 聊天模板Jinja   ●         │ (服务就绪后浏览器打开, 聊天在此进行)│
│ -- 上下文与卸载 --         │                                  │
│ 上下文长度    [4096 ]      │ 可用模型:                         │
│ GPU卸载层数   [下拉 0-99]  │ *  1: Qwen3.8-27B-UD-Q4_K_XL (16.3GB)│
│ CPU线程池大小 [0    ]      │    2: Ornith-1.5-9B-Q8_0 (9.x GB) │
│ -- 高级 --                │                                  │
│ 评估批处理大小[2048 ]      │ >> 正在启动 llama-server: ...     │
│ 物理批处理大小[512  ]      │ [dim]引擎 stdout 逐行日志...[/dim] │
│ 并行请求数    [1    ]      │                                  │
│ 统一KV缓存    ●           │ >> 服务就绪 http://127.0.0.1:5801 │
│ 快速注意力    [自动 ▼]     │ 浏览器打开 WebUI 即可开始聊天      │
│ K/V缓存量化   [F16(默认)▼] │                                  │
│ RoPE频率基/比例[自动]      │                                  │
│ mmap加载模型  ●           │                                  │
│ 加载视觉模型  ●           │                                  │
│ 启用MTP加速   ○           │                                  │
│ 最大/最小草稿Token、草稿概率│                                 │
│ [加载服务]       │                                  │
│ [停止服务]                 │                                  │
└───────────────────────────┴──────────────────────────────────┘
```

参数行均为单行无边框控件（终端配置文件风格），面板内容超出高度时左侧自动出现垂直滚动条。开关项为 `RoundCheckbox`：继承 Textual 的 Checkbox，仅重写 `_button` 属性渲染圆形指示器（未选 ○ / 选中 ●），选中态颜色由 CSS `$primary`（默认主题蓝色 #0178D4）提供；CSS 同时覆盖 ToggleButton 默认的 tall 边框与聚焦高亮，保证单行不撑高。

## 操作方式

无命令行输入框，全部通过面板控件 + 快捷键：

| 操作 | 方式 |
|---|---|
| 切换模型 | 「模型文件」下拉框（当前选择持久化） |
| 调节参数 | 左侧各 Input / Select / 圆形开关，改动即时写入 `config.json` |
| 启动/重载服务 | F5 或「加载服务」按钮 |
| 停止服务 | 「停止服务」按钮 |
| 清空控制台 | Ctrl+L |
| 退出（同时停服） | Ctrl+Q |

聊天：服务就绪后浏览器打开 `http://127.0.0.1:5801`，在 llama-server 官方 WebUI 中对话；采样参数（温度/Top-P 等）也在该页面设置。

## 参数说明

引擎参数（llama-server 启动参数，修改后需 F5 重启；flag 均已对照本机 `llama.cpp-0.3.0/common/arg.cpp` 核实）：

| 字段 | 默认 | 对应命令行 |
|---|---|---|
| model_path (模型文件) | 扫描到的第一个 gguf | `-m` |
| jinja (聊天模板Jinja) | true | `--jinja`（启用模板变量/工具调用支持） |
| ctx_size (上下文长度) | 4096 | `-c` |
| ngl (GPU卸载层数) | 99 (全部上 GPU) | `-ngl` |
| threads (CPU线程池大小) | 0 (自动) | `-t`（为 0 时不传） |
| ubatch_size (评估批处理大小) | 2048 | `-ub`（为 0 时不传） |
| batch_size (物理批处理大小) | 512 | `-b` |
| parallel (并行请求数) | 1 | `-np`（始终显式下发；不传时 llama-server 进入 auto 模式强制 n_parallel=4 并自动开启 kv_unified，见 server.cpp:152） |
| kv_unified (统一KV缓存) | true | `--kv-unified` / `--no-kv-unified`（显式下发） |
| flash_attn (快速注意力) | auto | `-fa on/off`（auto 时不传，用默认） |
| cache_type_k/v (K/V缓存量化) | "" (F16 默认) | `-ctk` / `-ctv`；可选 F32/F16/BF16/Q8_0/Q4_0/Q4_1/IQ4_NL/Q5_0/Q5_1（空值不传；UI 显示大写，传递前统一转小写） |
| rope_freq_base/scale (RoPE频率基/比例) | 0.0 (自动) | `--rope-freq-base` / `--rope-freq-scale`（为 0 时不传） |
| mmap (mmap加载模型) | true | `-lm mmap`（关闭时不传，用默认 auto） |
| mmproj_auto (加载视觉模型) | true | 主模型同目录含 mmproj 时自动追加 `-mm/--mmproj <路径>`；false 时不加载（纯文本省显存） |
| spec_mtp (启用MTP加速) | false | `--spec-type draft-mtp`（用模型自带 nextn 层；关闭时全部不传） |
| spec_draft_max/min (最大/最小草稿Token) | 2 / 0 | `--spec-draft-n-max` / `--spec-draft-n-min` |
| spec_draft_p_split (草稿概率) | 0.75 | `--spec-draft-p-split` |
| reasoning (深度思考) | true | `--reasoning on/off`（on 时聊天模板设 `enable_thinking=true`，模型输出 `<think>...</think>` 推理） |
| reasoning_effort (思考强度) | 空=模板默认；GUI 首次运行自动归一化为 low | 官方思考强度档位（low/medium/high）：低/中/高 = `--reasoning-budget 512/2048/8192`（思考 token 预算，同官方 webui 的 `thinking_budget_tokens`）；模板原生支持 `reasoning_effort` 时（如 Qwen3.8 认 xhigh/medium/low）额外透传 `--reasoning-effort` 最接近档位 |

说明：

- MTP（Multi-Token Prediction）推测解码仅对带 nextn/MTP 层的模型有效（如 Qwen3.8-27B-MTP / UD 系列）；普通模型开启后无收益。面板已移除「推测解码 (MTP)」分区标签，MTP 相关控件归入「高级」区；不支持 MTP 的模型会自动置灰「启用MTP加速」。
- 随机种子控件已移除（模型默认随机采样即可，LM Studio 亦默认如此）。
- 思考强度（官方行为）：已合并为**单个下拉框**，仅提供「关闭思考 / 低 (512) / 中 (2048) / 高 (8192)」。低/中/高通过 `--reasoning-budget N` 限制思考 token 预算（对应官方 webui 随请求下发的 `thinking_budget_tokens=512/2048/8192`），**稠密模型与 MoE 思考模型均可用**。开启时模型输出 `<think>...</think>` 推理过程（OpenAI 接口放在 `message.reasoning_content`）。
- 是否提供思考强度菜单由 `memory.template_supports_thinking` 判断（移植官方 webui 的 `detectThinkingSupport`：模板含 `enable_thinking`/`reasoning_effort`/`thinking_budget` 任一 kwarg 引用、思考类 Jinja 条件，或成对思考标签 `<think>` 等即视为支持思考）。不支持思考的模型下拉框仅剩「关闭思考」并置灰，且会自动把深度思考关闭。
- 模板原生支持 `reasoning_effort` 档位时（如 Qwen3.8 认 `xhigh/medium/low`），「低/中/高」还会自动透传最接近的 `--reasoning-effort`（high→xhigh），与 token 预算互补（Qwen3.8 模板据此注入不同力度的思考提示词）；其余模型只靠预算控强度，与官方 webui 行为一致。关闭思考时即使残留档位也不会下发 `--reasoning-budget`/`--reasoning-effort`。
- 旧版配置的 `max`（不限）与 `xhigh` 会归一化到 `high`、`minimal` 归一化到 `low`；空串（模板默认）在 GUI 中不再提供选项，模型加载时自动归一化为 `low` 并写回配置。
- LM Studio 截图中的 Context Checkpoints、Reasoning Budget Message、「保持模型在内存中」「KV缓存卸载到GPU内存」等项为本构建 llama-server 不支持的 LM Studio 专有功能，未提供对应控件（不传即默认行为）。
- 采样参数不在本程序面板中：聊天已移至 WebUI，由该页面随请求下发。`llm_tui/engine.py` 仍保留 `SamplingConfig` / `chat_stream()` 作为库 API，供程序化调用 OpenAI 兼容接口时使用。
- 模型目录：`model_dirs` 为列表，支持多个目录（`;` 分隔）。`scan_models` 用 `os.walk` 递归检测每个目录的子目录里的 `.gguf`（如填 `H:\\Qwen\\model` 可检测其下各个模型子目录）。目录改后仅刷新下拉框与控制台模型列表，不影响已选模型（若仍存在则保留选择）。

服务固定监听 `http://127.0.0.1:5801`（可在 config.json 改端口），就绪判定为 `GET /health` 返回 `{"status":"ok"}`。

## 配置文件 config.json

运行时自动生成于 `llm_tui/config/config.json`（含本机绝对路径，已 gitignore），删除即恢复默认值。字段：

```json
{
  "bin_dir": "",                        // 空=自动: 打包成 exe 后取 exe 所在目录(与 llama-server.exe 同级), 开发期用内置固定路径
  "port": 5801,                        // 服务端口
  "model_dirs": [],                 // 空=开发期用内置默认目录; 打包后为空, 由用户自行添加(可多个, 递归检测子目录 .gguf)
  "engine": {
    "model_path": "",  "ctx_size": 4096, "threads": 0, "ngl": 99,
    "batch_size": 512, "ubatch_size": 2048, "parallel": 1, "jinja": true,
    "kv_unified": true, "flash_attn": "auto",
    "cache_type_k": "", "cache_type_v": "",
    "rope_freq_base": 0.0, "rope_freq_scale": 0.0,
    "mmap": true, "seed": "",
    "mmproj_auto": true,
    "spec_mtp": false, "spec_draft_max": 2, "spec_draft_min": 0,
    "spec_draft_p_split": 0.75, "reasoning": true, "reasoning_effort": ""
  }
}
```

兼容性：加载时只合并已知字段，旧版 config.json（缺少新增引擎字段）可继续使用，缺失项取默认值；未知字段被忽略。旧配置中的 `sampling` 键会被静默忽略（采样参数已移至 WebUI）。旧版 `reasoning_effort` 若存的是模板原生档位（如 `xhigh`/`minimal`/`max`），加载时自动归一化（`xhigh`→`high`、`minimal`→`low`、`max`→`high`），无需手工修改。

分发默认值：`bin_dir` 为空时自动定位——打包成 exe 后取 exe 所在目录（与 llama-server.exe 同级），无需改配置；`model_dirs` 打包后默认为空，`model_path` 默认为空（由使用者在界面选择）。所有引擎参数均以 `--alias` 给模型设置干净名字（取 gguf 文件名主干），因此 `/v1/models` 与聊天请求中的 `model` 字段显示的是模型名（如 `Ornith-1.5-9B-Q8_0`）而不是完整路径，第三方 Agent 对接时更友好。

## 目录结构

| 文件 | 说明 |
|---|---|
| `main.py` | 快速入口，`LlmShellApp().run()` |
| `pyproject.toml` | 打包元数据 + console script（`llm-tui`） |
| `.gitignore` | 排除 `__pycache__`、`*.svg`、运行时 `config.json` 等 |
| `llm_tui/__init__.py` | 包标识 + 版本号 |
| `llm_tui/__main__.py` | `python -m llm_tui` 入口 |
| `llm_tui/ui.py` | Textual 界面：中文布局、CSS、RoundCheckbox、控制台日志转发、服务控制 |
| `llm_tui/engine.py` | 引擎封装：EngineConfig dataclass、模型扫描、子进程管理、stdout 逐行回调（log_callback）、SSE 客户端（库 API） |
| `llm_tui/config.py` | 配置读写：CONFIG_PATH、load_config/save_config、_from_dict |
| `llm_tui/slider.py` | 自实现 RangeSlider 滑块（Textual 8.x 已移除 Slider），点击/方向键调节 |
| `llm_tui/config/config.json` | 运行时自动生成（含本机绝对路径，已 gitignore），勿手工编辑（可删） |
| `llm_tui/config/deep_thinking.jinja` | 模型“深度思考”聊天模板（按 `enable_thinking` 输出 `<think>`），入库 |
| `checks/ui_smoke.py` | UI 冒烟校验：run_test + 控件断言 + build_args 映射 + 布局/滚动 + SVG 截图 |
| `checks/engine_boot.py` | 引擎启动集成校验：真实拉起 llama-server 验证启动/log_callback 转发/健康检查/停止 |
| `requirements.txt` | textual>=8.2, httpx>=0.27 |

## 已验证记录（本机 2026-08-25/26）

- `python checks/ui_smoke.py` → SMOKE OK：130x50 终端下底部输入行已移除、`#console` 占满右列（含原输入框 3 行）、控件不超面板宽度、内容可滚动；RoundCheckbox 未选渲染 ○ / 选中渲染 ● 且带 `-on` class；控制台挂载后包含 WebUI 地址与模型列表（含大小）；Input/Select/RoundCheckbox 程序化赋值均正确联动到 config，Select 内部标签渲染出当前选中项文本（防空白回归）；`build_args()` flag 映射断言通过。
- `python checks/engine_boot.py` → 以 `Qwen3.8-27B-UD-Q4_K_XL.gguf`（16.35GB）真实启动 llama-server：新增 flag 全部被接受（含 `reasoning_effort="high"` 下发的 `--reasoning-budget 8192 --reasoning-effort xhigh`），log_callback 转发日志，约 10 秒内 `/health` 返回 READY（模型文件在系统内存缓存中），`stop()` 干净终止。
- 环境：RTX 4090 D 24GB + i9-13900KF；测试时 LM Studio 正占用约 22GB 显存，llama-server 仍正常启动并就绪（该构建对显存不足有优雅处理）。
- **GPU 加速修复**：此前 `ggml-cuda.dll` 因缺 `cublasLt64_12.dll`（CUDA 12 运行库）加载失败，模型静默跑在 CPU（decode 约 8.7 t/s）。补齐 `cudart64_12.dll`/`cublas64_12.dll`/`cublasLt64_12.dll` 到 `H:\Qwen\gui` 后，`using device CUDA0`、`offloaded 34/34 layers to GPU` 正常出现，同一配置解码达 **119.8 t/s**、prefill 约 **290 t/s**（MTP 草稿接受率 0.85），与 LM Studio 同级甚至更快。

## 注意事项

- 冷加载 16GB 模型需要较长时间（取决于磁盘速度），右侧控制台会逐行滚动引擎日志；文件进入系统缓存后再次加载显著变快。
- 若其他程序（如 LM Studio）占用大量显存，GPU offload 可能自动缩减或回退 CPU；追求完整 GPU 性能时先关闭其他模型实例。
- 端口被占用时服务无法就绪：控制台会显示引擎 bind 错误日志，可改 `config.json` 的 `port` 后 F5。
- K/V缓存量化下拉框显示大写（如 Q4_0），`build_args()` 传递前统一转小写（q4_0）：llama.cpp 的 `-ctk/-ctv` 解析大小写敏感（arg.cpp 与 ggml_type_name 比较），直接传 "Q4_0" 会报 `Unsupported cache type: Q4_0`。config.json 中已存的大写值自动兼容，无需手工修改。
- **视觉模型（mmproj）自动加载**：`build_args()` 依据当前主模型，在其同目录自动查找文件名含 `mmproj`（不区分大小写）的 `.gguf`（排除主模型自身），找到即追加 `--mmproj <路径>`（已对照 `llama-server --help` 的 `-mm/--mmproj`）。即「选大模型，同目录有视觉模型就自动带上」，与 LM Studio 一致；无需手动选择。若报 `image input is not supported`,通常是该模型目录缺少 mmproj 文件，或 mmproj 与模型不是同一套（如把 Qwen 的 mmproj 配到别的模型）。
- **CUDA 运行库缺失会静默回退 CPU（极慢）**：`ggml-cuda.dll` 动态依赖 `cudart64_12.dll`、`cublas64_12.dll`、`cublasLt64_12.dll`（以及 MSVC 运行库 `MSVCP140.dll`/`VCRUNTIME140*.dll`）。这些不在 `llama-server.exe` 同目录时，CUDA 后端加载失败，`-ngl` 形同虚设，模型完全跑在 CPU（实测 prefill 约 25 t/s、decode 约 8 t/s）。且 llama-server 的 stdout 在管道下为整块缓冲，启动日志里的 `using device CUDA0`/`offloaded N/N layers to GPU` 未必逐行显示，容易被误判成已上 GPU。**解决**：把上述 CUDA 运行库复制到 `llama-server.exe` 同目录（本机可从 LM Studio 的 `backends\vendor\win-llama-cuda12-vendor-v2` 复制）。
- 服务就绪后，本程序通过 `nvidia-smi` 校验该进程是否有 CUDA 上下文：为真则控制台显示「GPU 加速已启用」，否则红字警告「模型运行在 CPU」，此时应检查上一项的运行库是否齐全。
- Windows 下子进程以 CREATE_NO_WINDOW 方式启动，不会弹出控制台窗口；服务端 stdout 逐行转发到右侧控制台，状态栏同时保留最后一行摘要。
- Textual 8.2.8 无 Slider 控件（已从核心移除），故 `llm_tui/slider.py` 自实现了 `RangeSlider`（`──●── 值` 单行滑块，点击/方向键调节，自动钳制到 `[min,max]`）。「上下文长度」「GPU卸载层数」「CPU线程池大小」三个参数已改用滑块调节。
- `RoundCheckbox` 是 Checkbox 子类：Textual 的 ToggleButton 默认渲染「▐X▌」方块按钮且聚焦时加 tall 边框，本类仅重写 `_button` 属性为单字符圆形指示器，并在应用 CSS 中显式覆盖 `:focus` 边框（默认主题对 `ToggleButton:focus` 的特异性高于普通选择器，必须显式声明）。
- Select 下拉框必须用 `Select(compact=True)`：Textual 的 Select 外层是容器，真正显示内容的是内部 `SelectCurrent`（默认带 tall 边框占 3 行）。若在外层 CSS 强制 `height:1`/`border:none`，会把标签区域裁掉导致框内空白；compact 模式走官方紧凑样式（`border:none !important`、单行），聚焦时仅高亮不改变布局。
- 模型下拉框显示名为目录名（非完整路径），超过 22 字符截断为前 21 字符 + "…"，保证单行不换行（实测阈值：面板滚动条激活时 Select 宽 25、标签区 22 列）。文件大小在控制台启动时的模型列表中查看。
- 参数行之间留一行间距（`.param-row { margin-bottom: 1 }`），内容超出面板高度时左侧面板自动出现垂直滚动条，属预期行为。
