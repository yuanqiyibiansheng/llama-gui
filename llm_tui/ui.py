"""llm-shell —— llama.cpp 终端风格 GUI Shell (Python + Textual)。

布局:
┌──────────────────────────────────────────────────────────┐
│ 状态栏: 服务状态 / 模型 / ctx / ngl                       │
├───────────────┬──────────────────────────────────────────┤
│ 参数面板      │  控制台 (RichLog)                        │
│ - 模型        │  - 服务启动/就绪信息, WebUI 地址          │
│ - 上下文与卸载│  - 引擎日志实时转发                      │
│ - 高级        │  - 可用模型列表(含大小)                  │
│ - 推测解码MTP │                                          │
│ [重新加载][停止]                                         │
└───────────────┴──────────────────────────────────────────┘

聊天在 llama-server WebUI (http://127.0.0.1:5801) 进行;
本程序只负责启动模型与调节参数。

运行: python main.py
"""
from __future__ import annotations

import asyncio
import os
import re
import subprocess
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from rich.markup import escape
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.content import Content
from textual.widgets import Button, Checkbox, Input, Label, RichLog, Select, Static

from llm_tui.config import _from_dict, load_config, save_config
from llm_tui.engine import (DEFAULT_MODEL_DIRS, DEFAULT_PORT,
                    EngineConfig, LlamaServer, default_bin_dir, default_model_dirs,
                    find_mmproj, model_alias, scan_models)
from llm_tui.memory import estimate_memory, read_model_arch, template_supports_thinking
from llm_tui.slider import RangeSlider

# 引擎数值型参数: (字段名, 中文显示名, 类型) —— 需重新加载生效
ENGINE_NUM_FIELDS = [
    ("ctx_size", "上下文长度", int),
    ("threads", "CPU线程池大小", int),
    ("ubatch_size", "评估批处理大小", int),
    ("batch_size", "物理批处理大小", int),
    ("parallel", "并行请求数", int),
    ("rope_freq_base", "RoPE频率基", float),
    ("rope_freq_scale", "RoPE频率比例", float),
    ("spec_draft_max", "最大草稿Token", int),
    ("spec_draft_min", "最小草稿Token", int),
    ("spec_draft_p_split", "草稿概率", float),
]

# KV 缓存量化可选值 (llama.cpp-0.3.0 common/arg.cpp kv_cache_types, 已核实)
KV_CACHE_TYPES = ["F16", "BF16", "F32", "Q8_0", "Q5_1", "Q5_0", "Q4_1", "Q4_0", "IQ4_NL"]

# 深度思考下拉(官方思考强度档位): 关闭/低/中/高。
# 低/中/高 = 思考 token 预算 512/2048/8192(与官方 webui thinking_budget_tokens 一致,
# 稠密/MoE 思考模型通用); 模板原生支持 reasoning_effort 时(如 Qwen3.8 认 xhigh/medium/low)
# 额外透传最接近档位, 二者互补。
REASONING_EFFORT_OPTIONS = [
    ("关闭思考", "off"),
    ("低 (512)", "low"),
    ("中 (2048)", "medium"),
    ("高 (8192)", "high"),
]





def model_label(path: str, max_width: int = 22) -> str:
    """模型下拉框显示名 (目录名)。实测阈值: 面板滚动条激活时 Select 宽=25,
    标签 <=22 字符单行显示, 超过则换行撑高该行。大小信息见控制台启动列表。"""
    name = os.path.basename(os.path.dirname(path)) or os.path.basename(path)
    if len(name) > max_width:
        return name[:max_width - 1] + "…"
    return name


class RoundCheckbox(Checkbox):
    """圆形勾选框: ○ 未选中 / ● 选中(蓝色), 单行显示。

    Textual 默认 Checkbox 渲染三字符 "▐X▌" 标记; 这里覆盖 _button
    只渲染单个圆形字形, 选中态颜色由 CSS (-on + $primary) 控制为蓝色。
    """

    @property
    def _button(self) -> Content:
        style = self.get_visual_style("toggle--button")
        ch = "●" if self.value else "○"
        return Content.assemble((ch, style))

    def render(self) -> Content:
        # 只渲染圆形指示器, 不渲染 Checkbox 标签(避免空标签的占位空格)
        return self._button


class LlmShellApp(App):
    TITLE = "LLM 控制台 (llama.cpp)"
    # 全局开启文本框选(控制台日志可拖选复制); 滑块等控件用各自 ALLOW_SELECT=False 排除
    ALLOW_SELECT = True

    CSS = """
    #status-bar {
        dock: top;
        height: 1;
        background: $surface;
        color: $text;
        padding: 0 1;
    }
    #status-text {
        width: 1fr;
        height: 1;
        content-align: left middle;
    }
    #status-api {
        width: auto;
        height: 1;
        content-align: center middle;
        color: $text;
    }
    #status-credit {
        width: 1fr;
        height: 1;
        content-align: right middle;
        color: $text;
        text-style: dim;
    }
    #main {
        height: 1fr;
    }
    #panel {
        width: 46;
        border: round $primary;
        padding: 0 1;
        overflow-y: auto;
        /* 滚动条: 细(竖1列) + 蓝色手柄(两端 ▁▂▃ 渐变圆头), 去掉默认刺眼的亮品红 \A.*/
        scrollbar-size-vertical: 1;
        scrollbar-color: $primary;
        scrollbar-background: $surface;
        scrollbar-corner-color: $primary;
    }
    .panel-title {
        text-style: bold;
        color: $accent;
        height: 1;
        content-align: left middle;
    }
    .mem-summary {
        height: 1;
        margin-bottom: 1;
        color: $accent;
        content-align: left middle;
    }
    /* 模型不支持 MTP 时, MTP 开关置灰不可勾选 */
    RoundCheckbox:disabled {
        opacity: 0.5;
    }
    RoundCheckbox:disabled > .toggle--button {
        color: $text-muted;
    }
    .section {
        color: $text-muted;
        height: 1;
        margin-top: 1;
        content-align: left middle;
    }
    .param-row {
        height: auto;
        min-height: 1;
        margin-bottom: 1; /* 行间距, 避免挤压 */
        align: left middle;
    }
    .param-name {
        width: 15;
        color: $text-muted;
        content-align: left middle;
    }
    .param-input {
        width: 1fr;
        border: none;
        background: transparent;
        height: 1;
    }
    .param-input:focus {
        background: $primary-background;
    }
    RangeSlider {
        width: 1fr;
        height: 1;
        color: $text;
        background: transparent;
    }
    RangeSlider:focus {
        background: transparent;
    }
    /* Select 外层是容器, 显示内容的是内部 SelectCurrent (默认带 tall 边框占3行)。
       不能在外层强制 height:1 —— 会把标签区域裁掉导致空白。
       改用 Select(compact=True) 走官方紧凑模式 (border:none !important, 单行)。 */
    .param-select {
        width: 1fr;
    }
    .param-select:focus > SelectCurrent {
        background-tint: $primary-background 30%;
    }
    /* 模型不支持思考时深度思考下拉置灰(仅关闭/默认可选, 与 MTP 开关一致) */
    .param-select:disabled {
        opacity: 0.5;
    }
    /* 圆形勾选框: ○ 未选中 / ● 选中($primary 蓝), 单行显示。
       ToggleButton 默认 border:tall 占3行, 覆盖为无边框紧凑样式;
       :focus 同样去边框, 避免点击聚焦时撑高该行。 */
    RoundCheckbox {
        border: none;
        background: transparent;
        height: 1;
    }
    RoundCheckbox:focus {
        border: none;
        background-tint: $primary-background 30%;
    }
    /* 默认主题选中态是绿色($text-success), 覆盖为蓝色 */
    RoundCheckbox > .toggle--button {
        color: $text-muted;
        background: transparent;
        text-style: bold;
    }
    RoundCheckbox.-on > .toggle--button {
        color: $primary;
        background: transparent;
        text-style: bold;
    }
    #btn-row {
        height: 3;
        margin-top: 1;
    }
    #btn-row Button {
        width: 1fr;
        margin: 0 1 0 0;   /* 两按钮之间留 1 格间隔 */
    }
    #btn-row Button:last-child {
        margin-right: 0;
    }
    #right {
        width: 1fr;
        height: 1fr;
    }
    #console-header {
        height: 3;
        border: solid $border;
        background: $surface;
        padding: 0 1;
        align: left middle;
    }
    #console-header .param-name {
        width: 13;
        color: $text-muted;
        content-align: left middle;
    }
    #slot-status {
        height: 1;
        width: 30;
        min-width: 30;
        margin: 0 0 0 1;
        padding: 0 1;
        border: none;
        color: $text;
        content-align: center middle;
        text-overflow: ellipsis;
        overflow: hidden;
    }
    #console {
        border: round $primary;
        width: 1fr;
        height: 1fr;
        /* 控制台滚动条同左面板: 细 1 列 + 蓝色手柄; 禁止横向滚动 */
        scrollbar-size-vertical: 1;
        scrollbar-color: $primary;
        scrollbar-background: $surface;
        scrollbar-corner-color: $primary;
        overflow-x: hidden;
    }
    """

    BINDINGS = [
        ("ctrl+q", "quit_app", "退出"),
        ("f5", "apply_engine", "重新加载"),
        ("ctrl+l", "clear_console", "清空控制台"),
        ("ctrl+shift+c", "copy_console", "复制控制台"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.cfg = load_config()
        self.engine_cfg = _from_dict(EngineConfig, self.cfg["engine"])
        self._model_dir_timer = None  # 模型目录输入的防抖计时器
        self._sync_timer = None        # 滑块拖动时的配置落盘防抖计时器
        # 实时处理状态 (转圈 + PROCESSING PROMPT X%)
        self._slot_processing = False  # 槽位是否正在处理 prompt
        self._slot_progress = 0.0      # prompt 处理进度 0.0-1.0
        self._spinner_idx = 0
        self._slot_timer = None        # 转圈动画计时器
        # 实际生效的 GPU 卸载层数(从引擎日志解析 "offloaded N/M layers"/"n_gpu_layers=N")
        self._offload_text = ""
        if not self.engine_cfg.model_path:
            models = scan_models(self._model_dirs())
            self.engine_cfg.model_path = models[0] if models else ""
        self.server = LlamaServer(
            str(self.cfg.get("bin_dir") or default_bin_dir()),
            port=int(self.cfg.get("port", DEFAULT_PORT)),
        )
        # 引擎日志逐行转发到右侧控制台 (dim 样式)
        self.server.log_callback = self._on_engine_log

    # ------------------------------------------------------------------ UI

    def compose(self) -> ComposeResult:
        with Horizontal(id="status-bar"):
            yield Static("", id="status-text")
            yield Static(f"API 请求地址  http://127.0.0.1:{self.server.port}/v1  |  API Key 任意", id="status-api")
            yield Static("作者:B站 睿智君弹皮筋", id="status-credit")
        with Horizontal(id="main"):
            with Vertical(id="panel"):
                yield Static("参数设置", classes="panel-title")
                yield Static("", id="mem-estimate", classes="mem-summary")

                yield Static("-- 模型 --", classes="section")
                with Horizontal(classes="param-row"):
                    yield Label("模型文件", classes="param-name")
                    yield Select(self._model_options(), value=self._model_select_value(),
                                 id="model-select", classes="param-select", compact=True)
                with Horizontal(classes="param-row"):
                    yield Label("深度思考", classes="param-name")
                    yield Select(self._reasoning_effort_options(),
                                 value=self._reasoning_effort_select_value(),
                                 id="eng-reasoning_effort", classes="param-select",
                                 compact=True, allow_blank=False)

                yield Static("-- 上下文与卸载 --", classes="section")
                with Horizontal(classes="param-row"):
                    yield Label("上下文长度", classes="param-name")
                    yield RangeSlider(minimum=0, maximum=262144, step=1024,
                                      value=self.engine_cfg.ctx_size, id="eng-ctx_size")
                with Horizontal(classes="param-row"):
                    yield Label("GPU卸载层数", classes="param-name")
                    yield RangeSlider(minimum=0, maximum=99, step=1,
                                      value=self.engine_cfg.ngl, id="eng-ngl")
                with Horizontal(classes="param-row"):
                    yield Label("CPU线程池大小", classes="param-name")
                    yield RangeSlider(minimum=0, maximum=(os.cpu_count() or 32), step=1,
                                      value=self.engine_cfg.threads, id="eng-threads")

                yield Static("-- 高级 --", classes="section")
                for key, label, _t in ENGINE_NUM_FIELDS[2:5]:  # ubatch/batch/parallel
                    with Horizontal(classes="param-row"):
                        yield Label(label, classes="param-name")
                        yield Input(str(getattr(self.engine_cfg, key)),
                                    id=f"eng-{key}", classes="param-input")
                with Horizontal(classes="param-row"):
                    yield Label("统一KV缓存", classes="param-name")
                    yield RoundCheckbox(value=self.engine_cfg.kv_unified, id="eng-kv_unified")
                with Horizontal(classes="param-row"):
                    yield Label("快速注意力", classes="param-name")
                    yield Select([("自动", "auto"), ("开启", "on"), ("关闭", "off")],
                                 value=self.engine_cfg.flash_attn,
                                 id="eng-flash_attn", classes="param-select", compact=True)
                with Horizontal(classes="param-row"):
                    yield Label("K缓存量化", classes="param-name")
                    yield Select([("F16 (默认)", "")] + [(t, t) for t in KV_CACHE_TYPES],
                                 value=self.engine_cfg.cache_type_k or "",
                                 id="eng-cache_type_k", classes="param-select", compact=True)
                with Horizontal(classes="param-row"):
                    yield Label("V缓存量化", classes="param-name")
                    yield Select([("F16 (默认)", "")] + [(t, t) for t in KV_CACHE_TYPES],
                                 value=self.engine_cfg.cache_type_v or "",
                                 id="eng-cache_type_v", classes="param-select", compact=True)
                for key, label, _t in ENGINE_NUM_FIELDS[5:7]:  # rope base/scale
                    with Horizontal(classes="param-row"):
                        yield Label(label, classes="param-name")
                        yield Input(str(getattr(self.engine_cfg, key))
                                    if getattr(self.engine_cfg, key) else "",
                                    placeholder="自动",
                                    id=f"eng-{key}", classes="param-input")
                with Horizontal(classes="param-row"):
                    yield Label("mmap加载模型", classes="param-name")
                    yield RoundCheckbox(value=self.engine_cfg.mmap, id="eng-mmap")
                with Horizontal(classes="param-row"):
                    yield Label("加载视觉模型", classes="param-name")
                    yield RoundCheckbox(value=self.engine_cfg.mmproj_auto, id="eng-mmproj_auto")
                with Horizontal(classes="param-row"):
                    yield Label("启用MTP加速", classes="param-name")
                    yield RoundCheckbox(value=self.engine_cfg.spec_mtp, id="eng-spec_mtp")
                for key, label, _t in ENGINE_NUM_FIELDS[7:10]:  # spec draft max/min/p
                    with Horizontal(classes="param-row"):
                        yield Label(label, classes="param-name")
                        yield Input(str(getattr(self.engine_cfg, key)),
                                    id=f"eng-{key}", classes="param-input")

                with Horizontal(id="btn-row"):
                    yield Button("加载服务", variant="success", id="btn-apply")
                    yield Button("停止服务", variant="error", id="btn-stop")
            with Vertical(id="right"):
                with Horizontal(id="console-header"):
                    yield Label("模型目录", classes="param-name")
                    yield Input(self._model_dir_text(), id="eng-model-dir",
                                classes="param-input", placeholder=r"C:\Qwen\model\models")
                    yield Static("已停止", id="slot-status", classes="param-status")
                yield RichLog(markup=True, id="console")

    def _model_dirs(self) -> List[str]:
        """去重、过滤后的模型扫描目录。

        - 配置里没有 model_dirs 键(旧配置) -> 用默认目录
        - 配置里是空列表(用户清空) -> 返回 [], 扫描无模型
        """
        raw = self.cfg.get("model_dirs")
        if raw is None:
            # 无配置(或旧配置无 model_dirs 键)时: 打包后回退空目录, 开发期用固定目录
            return default_model_dirs()
        seen: set = set()
        out: List[str] = []
        for d in raw:
            if isinstance(d, str) and d.strip() and d.strip() not in seen:
                seen.add(d.strip())
                out.append(d.strip())
        return out

    def _model_dir_text(self) -> str:
        return "; ".join(self._model_dirs())

    def _model_options(self) -> List[tuple]:
        # 只列当前目录真实扫描到的模型; 目录为空时不应硬塞当前模型
        options = [(model_label(p), p) for p in scan_models(self._model_dirs())]
        return options or [("未检测出可用模型", "")]

    def _model_select_value(self) -> str:
        """模型下拉框的初始值: 当前模型在选项中则保留, 否则取第一个(或空占位),
        避免 Select 挂载时因 value 不在选项中而校验失败。"""
        opts = self._model_options()
        mp = self.engine_cfg.model_path
        if any(p == mp for _l, p in opts):
            return mp
        return opts[0][1] if opts and opts[0][1] else ""

    def _reasoning_effort_options(self) -> list:
        """深度思考下拉选项: 模型支持思考时给低/中/高档(稠密/MoE 通用),
        否则只留「关闭思考」并置灰(官方 webui 对非思考模型不显示该菜单)。"""
        if not template_supports_thinking(self.engine_cfg.model_path):
            return [("关闭思考", "off")]
        return list(REASONING_EFFORT_OPTIONS)

    def _reasoning_effort_select_value(self) -> str:
        """深度思考下拉的初始值: 不支持思考或关闭思考->off; 否则当前档位在选项中
        则保留, 否则回低(最低档), 避免 Select 挂载时 value 不在选项中而校验失败。"""
        if not template_supports_thinking(self.engine_cfg.model_path):
            return "off"
        if not self.engine_cfg.reasoning:
            return "off"
        opts = self._reasoning_effort_options()
        cur = self.engine_cfg.reasoning_effort or ""
        if cur and any(p == cur for _l, p in opts):
            return cur
        return "low"

    def _refresh_reasoning_effort(self) -> None:
        """切模型后刷新深度思考下拉: 更新选项(模型不支持思考则置灰并强制关闭),
        恢复当前选择; 旧配置/模板默认档位(空)不存在于选项时归一化为低档。"""
        sel = self.query_one("#eng-reasoning_effort", Select)
        supports = template_supports_thinking(self.engine_cfg.model_path)
        opts = self._reasoning_effort_options()
        sel.set_options(opts)
        sel.disabled = not supports
        if not supports:
            # 模型不支持思考: 下拉只剩「关闭思考」, 同步关闭避免配置与显示不一致
            sel.value = "off"
            if self.engine_cfg.reasoning:
                self.engine_cfg.reasoning = False
                self.engine_cfg.reasoning_effort = ""
                self._sync_config()
            return
        if not self.engine_cfg.reasoning:
            sel.value = "off"
            return
        cur = self.engine_cfg.reasoning_effort or ""
        if cur and any(p == cur for _l, p in opts):
            sel.value = cur
            return
        # 旧配置存空档(模板默认)或非法档位: 归一化为最低档, 避免下发后模板 raise_exception
        sel.value = "low"
        if self.engine_cfg.reasoning_effort != "low":
            self.engine_cfg.reasoning_effort = "low"
            self._sync_config()

    def _apply_model_dir(self) -> None:
        """读取模型目录输入, 更新配置并重扫模型(由防抖计时器触发)。"""
        text = (self.query_one("#eng-model-dir", Input).value or "").strip()
        if text:
            self.cfg["model_dirs"] = [d.strip() for d in text.split(";") if d.strip()]
        else:
            # 清空目录 -> 不扫描任何目录, 模型列表为空
            self.cfg["model_dirs"] = []
        save_config(self.cfg)
        self._refresh_models()

    def _refresh_models(self) -> None:
        """模型目录变化后: 刷新模型下拉框 + 控制台模型列表。"""
        sel = self.query_one("#model-select", Select)
        opts = self._model_options()
        sel.set_options(opts)
        # set_options 会重置选择项; 若当前模型仍在选项中则恢复, 避免下拉框变空
        cur = self.engine_cfg.model_path
        if cur and any(p == cur for _l, p in opts):
            sel.value = cur
        self._write_console_header()
        # 目录变化后模型列表可能为空/不含当前模型, 同步刷新预估显存与上限
        self._update_mem_estimate()
        self._update_model_limits()

    def _mem_arch(self):
        if not self.engine_cfg.model_path:
            return None
        return read_model_arch(self.engine_cfg.model_path)

    def _offload_summary(self) -> str:
        """返回 GPU 卸载层数的可读说明。

        优先用引擎日志解析到的实际值(offloaded N/M 或 n_gpu_layers=N);
        否则按模型层数与 ngl 目标估算。ngl>=模型层数视为"全部"(llama.cpp 约定 99/-1=all)。
        """
        if self._offload_text:
            return self._offload_text
        arch = self._mem_arch()
        n_layer = int((arch or {}).get("n_layer", 0) or 0)
        ngl = int(self.engine_cfg.ngl)
        if n_layer <= 0:
            return "未知"
        if ngl >= n_layer:
            return f"全部 {n_layer} 层"
        if ngl > 0:
            return f"{ngl}/{n_layer} 层"
        return "0 层 (纯 CPU)"

    def _update_mem_estimate(self) -> None:
        """依据当前模型与引擎参数实时刷新「预估显存」显示。"""
        try:
            w = self.query_one("#mem-estimate", Static)
        except Exception:
            return
        # 以模型下拉框当前实际选中的值为准(用户选中占位/未选, 就显示未检测出)
        try:
            sel_value = self.query_one("#model-select", Select).value
        except Exception:
            sel_value = None
        mp = str(sel_value) if sel_value is not None else ""
        if not mp or mp in ("", "Select.NULL") or not os.path.isfile(mp):
            w.update("预估显存  未检测出可用模型")
            return
        arch = read_model_arch(mp)
        if not arch:
            w.update("预估显存  未检测出可用模型")
            return
        gpu, total = estimate_memory(self.engine_cfg, arch)
        # 视觉模型: 受「加载视觉模型」开关控制; 关闭则不显示加载信息(不会传 --mmproj)
        if self.engine_cfg.mmproj_auto:
            mmproj = find_mmproj(mp)
            vision = f" | 视觉 {os.path.basename(mmproj)}" if mmproj else " | 视觉 无"
        else:
            vision = " | 视觉 关闭"
        if gpu <= 0:
            w.update(f"预估显存  总 {total / 1024 ** 3:.1f} GB · CPU{vision}")
        else:
            w.update(f"预估显存  GPU {gpu / 1024 ** 3:.1f} GB | 总 {total / 1024 ** 3:.1f} GB{vision}")

    def _update_model_limits(self) -> None:
        """按当前模型刷新滑块上限与 MTP 可勾选性(不显示文字标签)。"""
        try:
            mtp_cb = self.query_one("#eng-spec_mtp", RoundCheckbox)
            ctx_slider = self.query_one("#eng-ctx_size", RangeSlider)
            ngl_slider = self.query_one("#eng-ngl", RangeSlider)
        except Exception:
            return
        arch = self._mem_arch()
        if not arch:
            mtp_cb.disabled = True
            return
        max_ctx = arch.get("context_length", 0) or 262144
        max_ngl = arch.get("n_layer", 0) + 1
        mtp = (arch.get("nextn", 0) or 0) > 0
        # 各滑块上限随模型真实参数决定
        ctx_slider.maximum = max_ctx
        if ctx_slider.value > max_ctx:
            ctx_slider.value = max_ctx
        ngl_slider.maximum = max_ngl
        if ngl_slider.value > max_ngl:
            ngl_slider.value = max_ngl   # 触发 Changed -> 同步/显存刷新
        # 只改 maximum 而不改值时不会自动重绘, 需手动刷新, 切换模型后条/数字才更新
        ctx_slider.refresh()
        ngl_slider.refresh()
        mtp_cb.disabled = not mtp
        if mtp:
            # 模型支持 MTP: 自动启用, 无需手动点击; 仍可手动关闭(override)
            if not self.engine_cfg.spec_mtp or not mtp_cb.value:
                self.engine_cfg.spec_mtp = True
                mtp_cb.value = True   # enabled 状态下赋值会触发 Changed -> 落盘
        else:
            # 模型不支持 MTP: 置灰关闭。disabled 状态下程序赋 value 不触发 Changed,
            # 故显式落盘; 视觉上同步复位
            if mtp_cb.value:
                mtp_cb.value = False
            if self.engine_cfg.spec_mtp:
                self.engine_cfg.spec_mtp = False
                try:
                    self._sync_config()
                except Exception:
                    pass

    def on_mount(self) -> None:
        self.refresh_status()
        self._write_console_header()
        self._update_mem_estimate()
        self._update_model_limits()
        self._refresh_reasoning_effort()
        self._update_slot_status()
        self._slot_timer = self.set_interval(0.12, self._tick_slot_spinner)
        # 仅状态栏(API地址等)保留框选; 面板内所有控件不参与框选, 避免拖出蓝色蒙版
        for w in self.query("#panel *"):
            try:
                w.ALLOW_SELECT = False
            except Exception:
                pass

    async def on_unmount(self) -> None:
        """app 关闭(含窗口关闭)时: 停掉本 app 的 llama-server, 并清理占用本程序端口的残留进程。"""
        try:
            await self.server.stop()
        except Exception:
            pass
        try:
            self._kill_port_owner(self.server.port)
        except Exception:
            pass
        if self._slot_timer is not None:
            self._slot_timer.stop()

    @staticmethod
    def _kill_port_owner(port: int) -> None:
        """杀掉监听指定端口的进程(可能是上次运行遗留、仍占着端口的 llama-server)。"""
        try:
            out = subprocess.check_output(["netstat", "-ano"], text=True, timeout=6)
        except Exception:
            return
        seen = set()
        for line in out.splitlines():
            if f":{port} " not in line or "LISTENING" not in line:
                continue
            parts = line.split()
            pid = parts[-1] if parts else ""
            if pid.isdigit() and pid not in seen:
                seen.add(pid)
                try:
                    subprocess.run(["taskkill", "/F", "/T", "/PID", pid],
                                   capture_output=True)
                except Exception:
                    pass

    def _write_console_header(self) -> None:
        """写入控制台简洁头部: 标题 + WebUI 地址 + 可用模型 + 一行操作提示。"""
        # 只列当前目录真实扫描到的模型, 不再硬塞当前模型
        models = scan_models(self._model_dirs())
        lines = [
            f"[bold cyan]LLM 控制台[/bold cyan]  |  [bold]WebUI  http://127.0.0.1:{self.server.port}[/bold]\n",
        ]
        if models:
            lines.append("\n[bold]可用模型:[/bold]\n")
            for i, p in enumerate(models, start=1):
                mark = "*" if p == self.engine_cfg.model_path else " "
                try:
                    size_s = f" ({os.path.getsize(p) / 1024 ** 3:.1f} GB)"
                except OSError:
                    size_s = ""
                lines.append(f"[dim]{mark} {i:>2}: [/dim]{escape(model_label(p))}{size_s}\n")
        else:
            lines.append("\n[yellow]未检测出可用模型[/yellow]\n")
        # 视觉模型: 受「加载视觉模型」开关控制(默认开=自动, 关=不加载省显存)
        cur = self.engine_cfg.model_path
        if cur:
            if not self.engine_cfg.mmproj_auto:
                lines.append("\n[dim]视觉模型: 已关闭(纯文本, 不加载)[/dim]\n")
            else:
                mmproj = find_mmproj(cur)
                if mmproj:
                    lines.append(f"\n[green]视觉模型: 已自动加载 {escape(os.path.basename(mmproj))}[/green]\n")
                else:
                    lines.append("\n[dim]视觉模型: 未检测到(不支持图片输入)[/dim]\n")
        lines.append("\n[dim]F5 加载服务 | Ctrl+L 清空 | Ctrl+Shift+C 复制 | Ctrl+Q 退出[/dim]\n")
        self.log_console.clear()
        self.log_console.write("".join(lines))

    @property
    def log_console(self) -> RichLog:
        # 右侧引擎控制台 (RichLog)。不能命名 console: App.console 是 Textual
        # 内部诊断用的 rich Console, __init__ 会赋值该属性。
        return self.query_one("#console", RichLog)

    # ------------------------------------------------------------- 状态栏

    def refresh_status(self) -> None:
        bar = self.query_one("#status-text", Static)
        if self.server.running:
            name = os.path.basename(os.path.dirname(self.engine_cfg.model_path)) or "模型"
            threads = self.engine_cfg.threads or (os.cpu_count() or 8)
            bar.update(
                f"[bold green]运行中[/bold green] {escape(name)}"
                f" | ctx {self.engine_cfg.ctx_size} ngl {self.engine_cfg.ngl}"
                f" t {threads}"
            )
        else:
            tail = ""
            if self.server.last_log_line:
                tail = f"  [dim]{escape(self.server.last_log_line[:70])}[/dim]"
            bar.update(
                "[bold red]已停止[/bold red] 按 F5 / 加载服务" + tail
            )

    # ------------------------------------------------------------- 事件处理

    async def on_button_pressed(self, e: Button.Pressed) -> None:
        if e.button.id == "btn-apply":
            await self.action_apply_engine()
        elif e.button.id == "btn-stop":
            await self._stop_server()

    def on_input_changed(self, e: Input.Changed) -> None:
        wid = e.input.id or ""
        if wid == "eng-model-dir":
            # 仅当用户改动了目录文本才防抖重扫; Input 初次挂载赋初值也会触发
            # Changed, 但值未变时不应排队重扫(否则 set_options 会让下拉框变空)
            new_text = (e.input.value or "").strip()
            if new_text != self._model_dir_text():
                if not new_text:
                    # 清空目录: 立即持久化 [], 避免重启又回退默认路径
                    self.cfg["model_dirs"] = []
                    save_config(self.cfg)
                if self._model_dir_timer is not None:
                    self._model_dir_timer.stop()
                self._model_dir_timer = self.set_timer(0.6, self._apply_model_dir)
            return
        if wid.startswith("eng-"):
            key = wid[len("eng-"):]
            ftype = dict((k, t) for k, _l, t in ENGINE_NUM_FIELDS).get(key)
            if ftype is None:
                return
            val = self._parse_number(e.input.value, ftype)
            if val is not None and (ftype is float or val >= 0):
                setattr(self.engine_cfg, key, val)
                self._sync_config()
                if key in ("ctx_size", "ubatch_size"):
                    self._update_mem_estimate()

    def on_checkbox_changed(self, e: Checkbox.Changed) -> None:
        mapping = {
            "eng-kv_unified": "kv_unified",
            "eng-mmap": "mmap",
            "eng-mmproj_auto": "mmproj_auto",
            "eng-spec_mtp": "spec_mtp",
        }
        key = mapping.get(e.checkbox.id or "")
        if key is not None:
            setattr(self.engine_cfg, key, bool(e.value))
            self._sync_config()

    def on_range_slider_changed(self, e: "RangeSlider.Changed") -> None:
        sid = e.slider.id or ""
        key = sid[len("eng-"):] if sid.startswith("eng-") else ""
        if key in ("ctx_size", "ngl", "threads"):
            setattr(self.engine_cfg, key, int(e.value))
            if key in ("ctx_size", "ngl"):
                self._update_mem_estimate()
            # 拖动时实时更新界面/显存, 但落盘防抖避免频繁写 config
            if self._sync_timer is not None:
                self._sync_timer.stop()
            self._sync_timer = self.set_timer(0.4, self._sync_config)

    def on_select_changed(self, e: Select.Changed) -> None:
        sid = e.select.id or ""
        if sid == "model-select":
            # 仅当选到有效模型路径才更新 model_path 并落盘; 占位/空值只刷新显示(不污染配置)
            if e.value is not Select.NULL and str(e.value) not in ("", "Select.NULL"):
                self.engine_cfg.model_path = str(e.value)
                self._sync_config()
            self._update_mem_estimate()
            self._update_model_limits()
            self._refresh_reasoning_effort()
        elif sid == "eng-reasoning_effort":
            # 单个下拉同时表达「深度思考开关」+「档位」:
            #   off = 关闭思考(--reasoning off);
            #   档位 = 开启 + --reasoning-budget N(+模板原生 effort, 若支持)
            v = str(e.value) if e.value is not Select.NULL else ""
            if v == "off":
                self.engine_cfg.reasoning = False
                self.engine_cfg.reasoning_effort = ""
            else:
                self.engine_cfg.reasoning = True
                self.engine_cfg.reasoning_effort = "" if v in ("", "Select.NULL") else v
            self._sync_config()
        elif sid == "eng-flash_attn":
            self.engine_cfg.flash_attn = str(e.value)
            self._sync_config()
        elif sid in ("eng-cache_type_k", "eng-cache_type_v"):
            key = "cache_type_k" if sid.endswith("_k") else "cache_type_v"
            setattr(self.engine_cfg, key, str(e.value or ""))
            self._sync_config()
            self._update_mem_estimate()

    @staticmethod
    def _parse_number(text, ftype):
        """把输入框文本解析为数值。int 字段返回 int, float 字段返回 float, 失败返回 None。

        刻意不带类型注解: Cython 编译时若写 `-> Optional[float]`, 会把 `return int(v)`
        结果强转成 float(如 512 -> 512.0), 导致打包后 `-c`/`-ub`/`-np` 等出现浮点参数。"""
        try:
            v = float(str(text).strip())
            if ftype is int:
                return int(v)
            return v
        except (ValueError, AttributeError):
            return None

    def _sync_config(self) -> None:
        self.cfg["engine"] = self.engine_cfg.to_dict()
        save_config(self.cfg)

    # ------------------------------------------------------------- 服务器控制

    async def action_apply_engine(self) -> None:
        if not self.engine_cfg.model_path or not os.path.isfile(
                self.engine_cfg.model_path):
            self.log_console.write("[bold red]未检测出可用模型 - 请在左侧面板的模型下拉框中选择[/bold red]\n")
            return
        self._sync_config()
        self.log_console.write(f"[bold yellow]>> 正在启动 llama-server: {escape(self.engine_cfg.model_path)}[/bold yellow]\n")
        # 视觉模型自动加载提示(LM Studio 行为): 同目录带 mmproj 时引擎已传 --mmproj
        if self.engine_cfg.mmproj_auto:
            mmproj = find_mmproj(self.engine_cfg.model_path)
            if mmproj:
                self.log_console.write(f"[dim]>> 已自动加载视觉模型: {escape(os.path.basename(mmproj))}[/dim]\n")
        try:
            await self.server.spawn(self.engine_cfg)
        except Exception as e:
            self.log_console.write(f"[bold red]>> 启动失败: {escape(str(e))}[/bold red]\n")
            return
        deadline = time.time() + 600.0
        while True:
            if await self.server.is_healthy():
                break
            if not self.server.running:
                code = self.server.proc.returncode if self.server.proc else "?"
                self.log_console.write(
                    f"[bold red]>> llama-server 已退出 (code {code})\n"
                    f">> 最后日志: {escape(self.server.last_log_line)}[/bold red]\n")
                return
            if time.time() > deadline:
                await self._stop_server()
                self.log_console.write("[bold red]>> 模型加载超时 (600s)[/bold red]\n")
                return
            self.refresh_status()
            await asyncio.sleep(1.0)
        # 控制台明确提示 GPU/CPU: 通过 nvidia-smi 判断进程是否有 CUDA 上下文
        if await self.server.uses_gpu():
            self.log_console.write("[green]>> GPU 加速已启用[/green]\n")
        else:
            self.log_console.write(
                "[bold red]>> 警告: 模型运行在 CPU (未检测到 CUDA 上下文), 生成速度会非常慢\n"
                ">> 请确认 llama-server.exe 同目录存在 cudart64_12.dll / cublas64_12.dll / cublasLt64_12.dll[/bold red]\n")
        # 稳定显示 GPU 卸载层数: 优先用引擎日志里解析到的实际值, 否则按模型层数+ngl 估算
        self.log_console.write(f"[dim]>> 卸载层数: {self._offload_summary()}[/dim]\n")
        self.log_console.write(f"[bold green]>> 服务就绪 http://127.0.0.1:{self.server.port}"
                           f"[/bold green]\n[green]浏览器打开 WebUI 即可开始聊天[/green]\n")
        self.refresh_status()

    async def _stop_server(self) -> None:
        if not self.server.running:
            return
        await self.server.stop()
        self.log_console.write("[yellow]>> 服务已停止[/yellow]\n")
        self.refresh_status()

    async def action_quit_app(self) -> None:
        try:
            await self._stop_server()
        except Exception:
            pass
        self._sync_config()
        self.exit()

    # ------------------------------------------------------------- 控制台

    def action_clear_console(self) -> None:
        self.log_console.clear()

    def action_copy_console(self) -> None:
        """复制控制台全部文本到系统剪贴板(Windows)。RichLog 不支持拖选, 用此快捷键复制。"""
        text = "\n".join(strip.text for strip in self.log_console.lines)
        if not text.strip():
            return
        try:
            import ctypes
            CF_UNICODETEXT = 13
            GMEM_MOVEABLE = 0x0002
            ctypes.windll.user32.OpenClipboard(0)
            ctypes.windll.user32.EmptyClipboard()
            data = ctypes.create_unicode_buffer(text)
            h = ctypes.windll.kernel32.GlobalAlloc(GMEM_MOVEABLE, ctypes.sizeof(data))
            p = ctypes.windll.kernel32.GlobalLock(h)
            ctypes.memmove(p, data, ctypes.sizeof(data))
            ctypes.windll.kernel32.GlobalUnlock(h)
            ctypes.windll.user32.SetClipboardData(CF_UNICODETEXT, h)
            ctypes.windll.user32.CloseClipboard()
            self.log_console.write(f"[dim]>> 控制台已复制到剪贴板[/dim]\n")
        except Exception as e:
            self.log_console.write(f"[bold red]>> 复制失败: {escape(str(e))}[/bold red]\n")

    def _on_engine_log(self, line: str) -> None:
        """引擎 stdout 逐行转发到控制台 (LM Studio 风格):
        原始 slot 行以 [DEBUG] [模型名] 显示; 进度行额外输出 [INFO] Prompt processing progress: X%;
        结束输出 [INFO] Finished streaming response。"""
        line = line.rstrip("\n")
        if not line.strip():
            return
        low = line.lower()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        alias = model_alias(self.engine_cfg.model_path)
        model_txt = f"[{alias}] " if alias else ""
        # 解析实时处理进度 + 生成友好行
        prog_pct = None
        if "launch_slot_" in low or "get_availabl" in low:
            self._slot_processing = True
            self._slot_progress = 0.0
        elif "print_timing" in low:
            pm = re.search(r"progress\s*=\s*([0-9.]+)", line)
            if pm:
                self._slot_progress = min(1.0, float(pm.group(1)))
                self._slot_processing = True
                prog_pct = self._slot_progress * 100
        elif "release" in low and "stop processing" in low:
            self._slot_processing = False
            self._wlog(now, "[INFO]", model_txt, "Finished streaming response")
        # 解析实际生效的 GPU 卸载层数: "offloaded N/M layers to GPU" 或 "n_gpu_layers = N"
        # (llama-server 的 stdout 在管道下可能整块缓冲, 故记录下来供后续稳定显示)
        mo = re.search(r"offloaded\s+(\d+)/(\d+)\s+layers\s+to\s+gpu", low)
        if mo:
            self._offload_text = f"{mo.group(1)}/{mo.group(2)} 层"
        elif "n_gpu_layers" in low:
            mn = re.search(r"n_gpu_layers\s*=\s*(\d+)", low)
            if mn:
                self._offload_text = f"{mn.group(1)} 层 (实际)"
        self._update_slot_status()
        if prog_pct is not None:
            self._wlog(now, "[INFO]", model_txt, f"Prompt processing progress: {prog_pct:.1f}%")
        # 原始行: I->[DEBUG](对齐 LM Studio), W/E 保留真实级别
        m = re.match(r"^(\S+\s+)([IWED])(\s+)(.*)$", line)
        lvl = m.group(2) if m else "I"
        lvl_txt = {"I": "[DEBUG]", "W": "[WARN]", "E": "[ERROR]", "D": "[DEBUG]"}.get(lvl, "[DEBUG]")
        self._wlog(now, lvl_txt, model_txt, escape(line))

    def _wlog(self, now: str, lvl: str, model_txt: str, content: str) -> None:
        self.log_console.write(
            f"[dim][{now}][/dim] {lvl} {model_txt}[dim]{content}[/dim]\n")

    def _update_slot_status(self) -> None:
        """更新控制台头部的实时处理状态 (转圈 + PROCESSING PROMPT X%)。"""
        try:
            w = self.query_one("#slot-status", Static)
        except Exception:
            return
        if self._slot_processing:
            ch = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"[self._spinner_idx % 10]
            w.update(f"{ch} PROCESSING PROMPT {min(1.0, self._slot_progress) * 100:.2f}%")
        else:
            w.update("就绪" if self.server.running else "已停止")

    def _tick_slot_spinner(self) -> None:
        """转圈动画: 处理中每 0.12s 前进一格。"""
        if self._slot_processing:
            self._spinner_idx += 1
        self._update_slot_status()


def main() -> None:
    LlmShellApp().run()


if __name__ == "__main__":
    main()
