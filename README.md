# llm-tui

LLM 控制台 —— 基于 Textual 的终端风格 GUI，用于管理 llama.cpp 的推理服务。

## 功能概览

- **模型管理**：扫描本地 GGUF 模型，一键加载/切换
- **参数调节**：上下文长度、GPU 卸载层数、线程数等核心参数的实时调节
- **显存估算**：依据模型架构自动预估 GPU/CPU 显存占用
- **MTP 推测解码**：支持 Multi-Token Prediction 推理加速
- **深度思考控制**：低/中/高三档思考强度（token 预算 512/2048/8192）
- **视觉模型**：自动检测并加载视觉投影文件（mmproj），支持多模态
- **KV 缓存量化**：支持 F16/BF16/F32/Q8_0/Q5_1/Q5_0/Q4_1/Q4_0/IQ4_NL 等量化格式
- **实时日志**：引擎运行日志实时转发，带处理进度指示

## 安装

### 开发环境

```bash
git clone https://github.com/<your-username>/llm-tui.git
cd llm-tui
pip install -r requirements.txt
```

### 打包发布

使用 PyInstaller 打包为独立可执行文件（Windows）：

```bash
pip install pyinstaller
pyinstaller llamaGui.spec
```

## 使用

```bash
# 方式一：直接运行
python main.py

# 方式二：作为包运行
python -m llm_tui

# 方式三：安装后命令行调用
pip install .
llm-tui
```

首次运行时，在左侧参数面板中选择模型文件并点击「加载服务」。服务启动后，浏览器打开 `http://127.0.0.1:5801` 进入 WebUI 即可开始对话。

## 快捷键

| 快捷键 | 功能 |
|--------|------|
| `Ctrl+Q` | 退出程序 |
| `F5` | 加载/重新加载服务 |
| `Ctrl+L` | 清空控制台日志 |
| `Ctrl+Shift+C` | 复制控制台内容 |

## 项目结构

```
llm-tui/
├── llm_tui/              # 主程序包
│   ├── __init__.py       # 版本信息
│   ├── __main__.py       # 入口点
│   ├── ui.py             # Textual GUI 主界面
│   ├── engine.py         # llama-server 进程管理与 API 客户端
│   ├── config.py         # 配置读写
│   ├── memory.py         # GGUF 元数据解析与显存估算
│   ├── slider.py         # RangeSlider 自定义控件
│   └── config/           # 运行时配置
│       └── config.json   # 用户配置（已加入 .gitignore）
├── checks/               # 集成测试
│   ├── __init__.py
│   ├── engine_boot.py    # 引擎启动测试
│   ├── reasoning_effort.py  # 思考强度端到端测试
│   └── ui_smoke.py       # UI 冒烟测试
├── pyproject.toml        # 项目元数据与构建配置
├── requirements.txt      # 依赖列表
├── llamaGui.spec         # PyInstaller 打包配置
└── .gitignore
```

## 配置说明

运行时配置保存在 `llm_tui/config/config.json`，包含：

- `bin_dir`：llama-server.exe 所在目录
- `port`：服务端口
- `model_dirs`：模型扫描路径（可多个，以分号分隔）
- `engine`：引擎参数，含模型路径、上下文、线程、GPU 层数等

配置文件已加入 `.gitignore`，不会提交到版本库。

## 技术细节

### 引擎参数对照

以下参数与 llama.cpp 命令行参数直接对应（已对照官方 arg.cpp 源码核实）：

| 参数 | 标志 | 说明 |
|------|------|------|
| `ctx_size` | `-c` | 上下文长度 |
| `threads` | `-t` | CPU 线程数 |
| `ngl` | `-ngl` | GPU 卸载层数 |
| `batch_size` | `-b` | 物理批处理大小 |
| `ubatch_size` | `-ub` | 评估批处理大小 |
| `parallel` | `-np` | 并行请求数 |
| `flash_attn` | `--flash-attn` | 快速注意力 |
| `cache_type_k/v` | `-ctk/-ctv` | KV 缓存量化 |
| `rope_freq_base/scale` | `--rope-freq-base/--rope-freq-scale` | RoPE 频率 |
| `mmap` | `-lm mmap` | 内存映射加载 |
| `kv_unified` | `--kv-unified` | 统一 KV 缓存 |

### 思考强度档位

| 档位 | Token 预算 | 说明 |
|------|-----------|------|
| 低 | 512 | 轻量思考 |
| 中 | 2048 | 标准思考 |
| 高 | 8192 | 深度思考 |

### 显存估算公式

```
KV 缓存 = n_head_kv * head_dim * 2 * kv_layers * bytes_per_elem * ctx
总显存 = 模型权重 + KV 缓存 + 线性注意力状态 + 计算工作区
```

## 测试

运行集成测试（需要本地 llama-server.exe 和模型文件）：

```bash
# UI 冒烟测试
python checks/ui_smoke.py

# 引擎启动测试
python checks/engine_boot.py

# 思考强度端到端验证
python checks/reasoning_effort.py

# 或使用 pytest
pytest checks/
```

## 许可证

本项目开源供学习研究使用。
