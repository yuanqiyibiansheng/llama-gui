"""llama-server 进程管理与 OpenAI 兼容 API 客户端。

职责:
- 以子进程方式启动/停止官方预编译的 llama-server.exe
- 异步读取其 stdout: 保留最后一行日志(状态栏), 并逐行回调转发(控制台实时显示)
- 通过 /health 轮询等待模型加载完成
- 提供 SSE 流式 chat completions 接口
"""
from __future__ import annotations

import asyncio
import ctypes
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from typing import AsyncIterator, Callable, List, Optional

import httpx

from llm_tui.memory import read_reasoning_effort

DEFAULT_BIN_DIR = r"H:\Qwen\gui"
DEFAULT_MODEL_DIRS = [r"H:\Qwen\model\models"]
DEFAULT_PORT = 5801

# 思考强度档位 —— GUI 提供低/中/高三档(官方 Off/Low/Medium/High 的可用子集)。
# 值 -> 思考 token 预算(低/中/高), 与官方 thinking_budget_tokens 的 512/2048/8192 相同;
# 旧配置的 max(不限)会归一化为 high(最接近的有预算档位)。
REASONING_EFFORT_LEVELS = ("low", "medium", "high")
_REASONING_EFFORT_BUDGET = {"low": 512, "medium": 2048, "high": 8192}
# 官方档位 -> 模型模板可能认可的原生 reasoning_effort 档位(按优先级)。
# 模板白名单含其一才透传 --reasoning-effort(如 Qwen3.8 认 xhigh/medium/low, 且把 high 归一为 xhigh)
_EFFORT_NATIVE_PREF = {
    "low": ("low",),
    "medium": ("medium",),
    "high": ("high", "xhigh"),
}
# 旧版配置可能存模板原生档位(xhigh/minimal 等), 归一化到官方档位
_LEGACY_REASONING_EFFORT = {
    "minimal": "low",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "high",
    "max": "high",
}


def _create_kill_job():
    """创建 Windows Job Object(设 KILL_ON_JOB_CLOSE), 让子进程随父进程退出被 OS 自动杀死。

    llama-server 以 CREATE_NO_WINDOW 独立子进程启动, 父进程(GUI)无论正常还是被强制
    结束, 只要本进程退出, 该 Job 句柄随之被 OS 关闭并触发 KILL_ON_JOB_CLOSE,
    自动杀掉 Job 内所有进程, 避免 llama-server.exe 残留。非 win32 或创建失败返回 None。
    """
    if sys.platform != "win32":
        return None
    try:
        from ctypes import wintypes

        class _IO_COUNTERS(ctypes.Structure):
            _fields_ = [("ReadOperationCount", ctypes.c_ulonglong),
                        ("WriteOperationCount", ctypes.c_ulonglong),
                        ("OtherOperationCount", ctypes.c_ulonglong),
                        ("ReadTransferCount", ctypes.c_ulonglong),
                        ("WriteTransferCount", ctypes.c_ulonglong),
                        ("OtherTransferCount", ctypes.c_ulonglong)]

        class _BASIC_LIMIT(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", wintypes.LARGE_INTEGER),
                        ("PerJobUserTimeLimit", wintypes.LARGE_INTEGER),
                        ("LimitFlags", wintypes.DWORD),
                        ("MinimumWorkingSetSize", ctypes.c_size_t),
                        ("MaximumWorkingSetSize", ctypes.c_size_t),
                        ("ActiveProcessLimit", wintypes.DWORD),
                        ("Affinity", ctypes.c_size_t),
                        ("PriorityClass", wintypes.DWORD),
                        ("SchedulingClass", wintypes.DWORD)]

        class _EXTENDED_LIMIT(ctypes.Structure):
            _fields_ = [("BasicLimitInformation", _BASIC_LIMIT),
                        ("IoInfo", _IO_COUNTERS),
                        ("ProcessMemoryLimit", ctypes.c_size_t),
                        ("JobMemoryLimit", ctypes.c_size_t),
                        ("PeakProcessMemoryUsed", ctypes.c_size_t),
                        ("PeakJobMemoryUsed", ctypes.c_size_t)]

        kernel32 = ctypes.windll.kernel32
        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            return None
        info = _EXTENDED_LIMIT()
        # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
        info.BasicLimitInformation.LimitFlags = 0x00002000
        ok = kernel32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info))
        if not ok:
            kernel32.CloseHandle(job)
            return None
        return job
    except Exception:
        return None


def default_bin_dir() -> str:
    """bin_dir 默认值: 打包成 exe 后回退到 exe 所在目录(与 llama-server.exe 同级);
    开发期回退到固定路径。"""
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return DEFAULT_BIN_DIR


def default_model_dirs() -> List[str]:
    """model_dirs 默认值: 打包成 exe 后为空(分发时由用户自行添加模型目录);
    开发期回退到固定目录。"""
    if getattr(sys, "frozen", False):
        return []
    return list(DEFAULT_MODEL_DIRS)


def model_alias(path: str) -> str:
    """API 模型别名: 取 gguf 文件名主干(去目录、去 .gguf 扩展), 供 --alias 使用。

    使 /v1/models 与 chat 请求的 model 字段显示干净模型名, 而不是完整路径
    (如 Ornith-1.5-9B-Q8_0 而非 H:\\Qwen\\model\\models\\Ornith-1.5-9B-Q8_0\\Ornith-1.5-9B-Q8_0.gguf)。"""
    base = os.path.basename(path)
    if base.lower().endswith(".gguf"):
        base = base[: -len(".gguf")]
    return base


def find_mmproj(model_path: str) -> str:
    """在主模型同目录自动查找视觉投影(multimodal projector)文件, 找不到返回空串。

    命名约定(与 LM Studio 的自动加载行为一致): 文件名含 "mmproj"(不区分大小写)
    且以 .gguf 结尾, 并排除与主模型同文件。这样「选中大模型, 同目录只要有视觉
    模型就自动带上 --mmproj」, 无需用户手动选择。
    多个时优先与主模型文件名主干同名者, 否则按字典序取第一个。"""
    if not model_path:
        return ""
    d = os.path.dirname(model_path)
    if not os.path.isdir(d):
        return ""
    base = os.path.basename(model_path).lower()
    stem = os.path.splitext(base)[0]
    candidates: List[str] = []
    for f in os.listdir(d):
        low = f.lower()
        if not low.endswith(".gguf"):
            continue
        if low == base:
            continue
        if "mmproj" not in low:
            continue
        candidates.append(os.path.join(d, f))
    if not candidates:
        return ""
    if len(candidates) > 1:
        pref = [c for c in candidates if stem in os.path.basename(c).lower()]
        if pref:
            candidates = pref
    candidates.sort()
    return candidates[0]


@dataclass
class EngineConfig:
    """引擎级参数 —— 修改后需要重启 llama-server 才生效。

    flag 映射均已对照 llama.cpp-0.3.0/common/arg.cpp 源码核实:
      -c ctx-size, -t threads, -ngl n-gpu-layers, -b batch-size,
      -ub ubatch-size(评估批处理), -np parallel(并行请求数), --jinja,
      --kv-unified/--no-kv-unified(统一KV缓存),
      -fa flash-attn [on|off|auto](快速注意力),
      -ctk/-ctv cache-type-k/v(KV缓存量化, UI显示大写/传递小写: f32,f16,bf16,q8_0,q4_0,q4_1,iq4_nl,q5_0,q5_1),
      --rope-freq-base/--rope-freq-scale(RoPE 频率基/比例),
      -lm load-mode [auto|none|mmap|mlock|mmap+mlock|dio](--no-mmap 已弃用),
      -s seed(随机种子, 缺省=随机),
      --spec-type draft-mtp + --spec-draft-n-max/-n-min/--spec-draft-p-split(MTP 推测解码)
    """

    model_path: str = ""
    ctx_size: int = 4096
    threads: int = 0          # 0 = 自动(CPU 核心数)
    ngl: int = 99             # GPU offload 层数, 99 = 全部; CPU-only 机器可设 0
    batch_size: int = 512     # 物理批处理大小 (-b)
    ubatch_size: int = 2048   # 评估批处理大小 (-ub), 0 = 不传(用内置默认)
    parallel: int = 1         # 并行请求数 (-np), 始终显式下发(缺省会走 auto 4 槽)
    jinja: bool = True        # 使用模型自带 jinja chat template(Qwen3 系列需要)
    kv_unified: bool = True   # 统一 KV 缓存 (--kv-unified / --no-kv-unified)
    flash_attn: str = "auto"  # 快速注意力 auto|on|off, on/off 才传递
    cache_type_k: str = ""    # K 缓存量化 (-ctk), 空 = 默认 F16
    cache_type_v: str = ""    # V 缓存量化 (-ctv)
    rope_freq_base: float = 0.0   # RoPE 频率基, 0 = 自动(不传)
    rope_freq_scale: float = 0.0  # RoPE 频率比例, 0 = 自动(不传)
    mmap: bool = True         # mmap 加载模型 (-lm mmap), off = 内置 auto
    mmproj_auto: bool = True  # 自动加载视觉模型: 主模型同目录含 mmproj 时自动带上 --mmproj; off = 不加载(纯文本省显存)
    seed: str = ""            # 随机种子 (-s), 空 = 随机
    spec_mtp: bool = False        # 启用 MTP 推测解码 (--spec-type draft-mtp)
    spec_draft_max: int = 2       # 最大草稿 token (--spec-draft-n-max)
    spec_draft_min: int = 0       # 最小草稿 token (--spec-draft-n-min)
    spec_draft_p_split: float = 0.75  # 草稿概率 (--spec-draft-p-split)
    reasoning: bool = True        # 深度思考 (--reasoning on/off), on 时模板设 enable_thinking=true
    reasoning_effort: str = ""   # 思考强度(官方档位): low/medium/high; 空=模板默认(仅旧配置/后端)

    def __post_init__(self) -> None:
        """兼容旧配置: 旧版 reasoning_effort 存模板原生档位(xhigh/minimal/max 等),
        归一化到 GUI 可用档位(low/medium/high); 未知值回退空(模板默认, 不设预算)。"""
        v = (self.reasoning_effort or "").strip().lower()
        if v not in REASONING_EFFORT_LEVELS:
            v = _LEGACY_REASONING_EFFORT.get(v, "")
        self.reasoning_effort = v

    def native_effort_level(self) -> str:
        """把官方思考强度档位映射为模型模板认可的原生 reasoning_effort 档位。

        模板白名单(从 GGUF 内嵌 chat template 解析)不含任何候选时返回空串,
        此时只靠 --reasoning-budget 控制思考 token 预算(官方 webui 的做法,
        稠密/MoE 思考模型通用)。"""
        if not self.reasoning_effort or not self.model_path:
            return ""
        levels = read_reasoning_effort(self.model_path)
        for cand in _EFFORT_NATIVE_PREF.get(self.reasoning_effort, ()):
            if cand in levels:
                return cand
        return ""

    def to_dict(self) -> dict:
        return asdict(self)

    def build_args(self, bin_dir: str, host: str, port: int) -> List[str]:
        exe = os.path.join(bin_dir, "llama-server.exe")
        threads = self.threads or (os.cpu_count() or 8)
        args = [
            exe,
            "-m", self.model_path,
            "-c", str(self.ctx_size),
            "-t", str(threads),
            "-ngl", str(self.ngl),
            "-b", str(self.batch_size),
        ]
        # 给模型设一个干净的 API 别名: /v1/models 与 chat 的 model 字段用它而非完整路径
        alias = model_alias(self.model_path)
        if alias:
            args += ["--alias", alias]
        # 视觉模型: 主模型同目录含 mmproj 时自动加载(LM Studio 行为, 无需手动选);
        # mmproj_auto=False 时不加载(纯文本省显存)
        if self.mmproj_auto:
            mmproj = find_mmproj(self.model_path)
            if mmproj:
                args += ["--mmproj", mmproj]
                # Qwen-VL 动态分辨率视觉模型 grounding 需至少 1024 图像 token;
                # llama.cpp 默认读模型元数据(常为 8, 偏小), 官方建议显式 --image-min-tokens 1024
                # (clip.cpp 据此把 image_min_pixels 抬到阈值, 消除 "require at minimum 1024 image tokens" 警告)
                args += ["--image-min-tokens", "1024"]
                # ngl<=0(纯 CPU)时 mmproj 默认仍尝试 GPU(--mmproj-offload), 造成 CPU/CUDA 分配不一致;
                # 显式 --no-mmproj-offload 让视觉投影也留在 CPU, 保持一致
                if self.ngl <= 0:
                    args += ["--no-mmproj-offload"]
        if self.ubatch_size > 0:
            args += ["-ub", str(self.ubatch_size)]
        # 显式下发 -np: 不传时 llama-server 进入 auto 模式(n_parallel=-1), 强制
        # n_parallel=4 并自动开启 kv_unified (server.cpp), 与单用户单槽预期不符
        args += ["-np", str(self.parallel)]
        args += ["--host", host, "--port", str(port)]
        # 聊天模板对 Qwen 系模型必须, 隐藏开关后始终启用
        args.append("--jinja")
        args.append("--kv-unified" if self.kv_unified else "--no-kv-unified")
        if self.flash_attn in ("on", "off"):
            args += ["--flash-attn", self.flash_attn]
        # llama.cpp -ctk/-ctv 解析大小写敏感(arg.cpp kv_cache_type_from_str
        # 与 ggml_type_name 小写名比较), UI 显示值统一转小写后传递
        if self.cache_type_k:
            args += ["-ctk", self.cache_type_k.strip().lower()]
        if self.cache_type_v:
            args += ["-ctv", self.cache_type_v.strip().lower()]
        if self.rope_freq_base > 0:
            args += ["--rope-freq-base", str(self.rope_freq_base)]
        if self.rope_freq_scale > 0:
            args += ["--rope-freq-scale", str(self.rope_freq_scale)]
        if self.mmap:
            args += ["-lm", "mmap"]
        seed = (self.seed or "").strip()
        if seed:
            args += ["-s", seed]
        if self.spec_mtp:
            args += [
                "--spec-type", "draft-mtp",
                "--spec-draft-n-max", str(self.spec_draft_max),
                "--spec-draft-n-min", str(self.spec_draft_min),
                "--spec-draft-p-split", str(self.spec_draft_p_split),
            ]
        # 深度思考: --reasoning on/off (arg.cpp), on 时模板设 enable_thinking=true
        args += ["--reasoning", "on" if self.reasoning else "off"]
        # 思考强度: 仅在开启思考时透传; 关闭思考时即使残留档位也忽略(避免模板 raise_exception)
        if self.reasoning and self.reasoning_effort:
            # 思考 token 预算(官方 webui 的 thinking_budget_tokens): low=512/medium=2048/high=8192。
            # 模板原生支持 effort 的模型(如 Qwen3.8)同时透传最接近档位(系统提示风格), 二者互补。
            budget = _REASONING_EFFORT_BUDGET.get(self.reasoning_effort)
            if budget:
                args += ["--reasoning-budget", str(budget)]
            native = self.native_effort_level()
            if native:
                args += ["--reasoning-effort", native]
        return args


@dataclass
class SamplingConfig:
    """采样参数 —— 每次请求随 chat completions body 下发, 无需重启。"""

    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 40
    min_p: float = 0.05
    repeat_penalty: float = 1.1
    max_tokens: int = 2048

    def to_dict(self) -> dict:
        return asdict(self)

    def to_request(self) -> dict:
        d = self.to_dict()
        # llama.cpp OpenAI 兼容接口支持的采样字段
        return {
            "temperature": d["temperature"],
            "top_p": d["top_p"],
            "top_k": d["top_k"],
            "min_p": d["min_p"],
            "repeat_penalty": d["repeat_penalty"],
            "max_tokens": d["max_tokens"],
        }


def scan_models(dirs: Optional[List[str]] = None) -> List[str]:
    """扫描目录下的 .gguf 模型文件(排除 mmproj 视觉投影文件)。"""
    found: List[str] = []
    # dirs 为 [] 时(用户清空)扫描结果为空; dirs 为 None 时才用默认目录
    for d in (dirs if dirs is not None else DEFAULT_MODEL_DIRS):
        if not os.path.isdir(d):
            continue
        for root, _dirs, files in os.walk(d):
            for f in files:
                if f.lower().endswith(".gguf") and not f.startswith("mmproj"):
                    found.append(os.path.join(root, f))
    return sorted(found)


class LlamaServer:
    """llama-server 子进程 + HTTP API 封装。所有方法均为 async。"""

    def __init__(self, bin_dir: str = "", host: str = "127.0.0.1",
                 port: int = DEFAULT_PORT):
        # 空 = 回退到 default_bin_dir(); 打包后即 exe 所在目录(与 llama-server.exe 同级)
        self.bin_dir = bin_dir or default_bin_dir()
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self.proc: Optional[asyncio.subprocess.Process] = None
        self.last_log_line = ""
        # 引擎日志行回调 (控制台实时显示用); 在 asyncio 任务内同步调用
        self.log_callback: Optional[Callable[[str], None]] = None
        self._log_task: Optional[asyncio.Task] = None
        self._job: Optional[int] = None  # 子进程 Job 句柄(KILL_ON_JOB_CLOSE)

    @property
    def running(self) -> bool:
        return self.proc is not None and self.proc.returncode is None

    async def spawn(self, cfg: EngineConfig) -> None:
        """启动 llama-server 子进程(不等待模型加载完成)。"""
        if self.running:
            await self.stop()
        args = cfg.build_args(self.bin_dir, self.host, self.port)
        kwargs: dict = {}
        if sys.platform == "win32":
            # CREATE_NO_WINDOW: 不在后台弹出控制台窗口
            kwargs["creationflags"] = 0x08000000
        self.proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=self.bin_dir,
            **kwargs,
        )
        self.last_log_line = ""
        self._log_task = asyncio.get_event_loop().create_task(self._read_log())
        # 把子进程放入 KILL_ON_JOB_CLOSE 的 Job: 父进程退出时由 OS 自动杀它, 防残留
        self._job = _create_kill_job()
        if self._job is not None:
            try:
                # AssignProcessToJobObject 需要进程句柄, 从 pid 打开
                PROCESS_SET_QUOTA = 0x0100
                PROCESS_TERMINATE = 0x0001
                PROCESS_QUERY_INFORMATION = 0x0400
                h = ctypes.windll.kernel32.OpenProcess(
                    PROCESS_SET_QUOTA | PROCESS_TERMINATE | PROCESS_QUERY_INFORMATION,
                    False, self.proc.pid)
                if h:
                    ctypes.windll.kernel32.AssignProcessToJobObject(self._job, h)
                    ctypes.windll.kernel32.CloseHandle(h)
            except Exception:
                pass

    async def _read_log(self) -> None:
        """持续读取 stdout, 保留最后一行(状态栏), 并逐行回调(控制台)。"""
        assert self.proc is not None and self.proc.stdout is not None
        while True:
            line = await self.proc.stdout.readline()
            if not line:
                break
            text = line.decode("utf-8", "replace").strip()
            if text:
                self.last_log_line = text
                if self.log_callback is not None:
                    self.log_callback(text)

    async def is_healthy(self) -> bool:
        """模型加载完成且服务就绪时返回 True。"""
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                r = await client.get(f"{self.base_url}/health")
                if r.status_code != 200:
                    return False
                data = r.json()
                return data.get("status", "") == "ok"
        except Exception:
            return False

    async def uses_gpu(self) -> bool:
        """通过 nvidia-smi 判断本进程是否有 CUDA 上下文, 即模型是否真正跑在 GPU 上。

        不依赖日志文本(stdout 在管道下整块缓冲), 直接查 nvidia-smi 计算进程列表。
        模型刚加载完成时 CUDA 上下文可能尚未登记到 nvidia-smi, 故轮询重试几次避免误报 CPU。
        """
        if self.proc is None or self.proc.pid is None:
            return False
        for _ in range(4):
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                    text=True, timeout=6)
                pids = {int(x.strip()) for x in out.splitlines() if x.strip().isdigit()}
                if self.proc.pid in pids:
                    return True
            except Exception:
                pass
            await asyncio.sleep(1.0)
        return False

    async def stop(self) -> None:
        """终止 llama-server 子进程。"""
        if self._log_task is not None:
            self._log_task.cancel()
            try:
                await self._log_task
            except (asyncio.CancelledError, Exception):
                pass
            self._log_task = None
        if self.proc is not None and self.running:
            pid = self.proc.pid
            try:
                if sys.platform == "win32" and pid:
                    # taskkill /T 连子进程一起结束, 比 terminate/kill 可靠(避免旧服务残留占端口)
                    subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)],
                                   capture_output=True)
                else:
                    self.proc.terminate()
            except Exception:
                pass
            try:
                await asyncio.wait_for(self.proc.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                try:
                    self.proc.kill()
                except ProcessLookupError:
                    pass
        self.proc = None
        self.last_log_line = ""
        # 关闭 Job: KILL_ON_JOB_CLOSE 会杀掉 Job 内残留子进程(兜底, 正常已 taskkill 则无剩余)
        if self._job is not None:
            try:
                ctypes.windll.kernel32.CloseHandle(self._job)
            except Exception:
                pass
            self._job = None

    async def chat_stream(
        self,
        messages: List[dict],
        sampling: SamplingConfig,
        model_name: str = "local",
        abort_event: Optional[asyncio.Event] = None,
    ) -> AsyncIterator[str]:
        """流式调用 /v1/chat/completions, 逐块 yield assistant 文本增量。"""
        payload = {
            "model": model_name,
            "messages": messages,
            "stream": True,
            **sampling.to_request(),
        }
        # read 超时给足: reasoning 模型首 token 延迟可能很长
        timeout = httpx.Timeout(connect=10.0, read=600.0, write=30.0, pool=10.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream(
                "POST", f"{self.base_url}/v1/chat/completions", json=payload
            ) as r:
                if r.status_code != 200:
                    body = (await r.aread()).decode("utf-8", "replace")[:300]
                    raise RuntimeError(f"HTTP {r.status_code}: {body}")
                async for line in r.aiter_lines():
                    if abort_event is not None and abort_event.is_set():
                        break
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        obj = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    choices = obj.get("choices") or [{}]
                    delta = (choices[0].get("delta") or {}).get("content")
                    if delta:
                        yield delta
