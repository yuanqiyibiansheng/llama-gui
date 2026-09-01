"""预估显存估算: 依据模型架构与参数(上下文/GPU卸载层数/KV量化)估算内存占用。

KV 缓存公式(已对照 llama-server 实际日志核实):
  kv_bytes = n_head_kv * head_dim * 2(K+V) * kv_layers * bytes_per_elem * ctx
例如 Ornith(Qwen3.5 9B, head_count_kv=4, head_dim=256, kv_layers=8, F16):
  4 * 256 * 2 * 8 * 2 * 4096 = 134217728 B = 128 MiB, 与日志 "KV buffer size = 128 MiB" 一致。
"""
from __future__ import annotations

import os
import re
import struct
from functools import lru_cache
from typing import Dict, Optional, Tuple

# GGUF 元数据类型码 (gguf/gguf.h gguf_type)
_GGUF_T = {0: "U8", 1: "I8", 2: "U16", 3: "I16", 4: "U32", 5: "I32",
           6: "F32", 7: "BOOL", 8: "STR", 9: "ARRAY", 10: "U64", 11: "I64", 12: "F64"}


def _read_gguf_metadata(path: str) -> Dict[str, object]:
    """读取 .gguf 文件头部的元数据键值。

    只读 header + 元数据(在文件最前), 不读 tensor 数据, 避免十几 GB 大模型
    切换时整文件读入内存导致界面卡死。"""
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError("not a gguf file")
        f.read(4)      # version
        f.read(8)      # tensor_count(跳过)
        nkv = struct.unpack("<Q", f.read(8))[0]

        def rd_str() -> str:
            n = struct.unpack("<Q", f.read(8))[0]
            return f.read(n).decode("utf-8", "replace")

        def rd_val(t: int):
            if t == 0: v = struct.unpack("<B", f.read(1))[0]
            elif t == 1: v = struct.unpack("<b", f.read(1))[0]
            elif t == 2: v = struct.unpack("<H", f.read(2))[0]
            elif t == 3: v = struct.unpack("<h", f.read(2))[0]
            elif t == 4: v = struct.unpack("<I", f.read(4))[0]
            elif t == 5: v = struct.unpack("<i", f.read(4))[0]
            elif t == 6: v = struct.unpack("<f", f.read(4))[0]
            elif t == 7: v = bool(struct.unpack("<B", f.read(1))[0])
            elif t == 8: v = rd_str()
            elif t == 9:
                at = struct.unpack("<I", f.read(4))[0]
                n = struct.unpack("<Q", f.read(8))[0]
                v = [rd_val(at) for _ in range(n)]
            elif t == 10: v = struct.unpack("<Q", f.read(8))[0]
            elif t == 11: v = struct.unpack("<q", f.read(8))[0]
            elif t == 12: v = struct.unpack("<d", f.read(8))[0]
            else: v = None
            return v

        meta: Dict[str, object] = {}
        for _ in range(nkv):
            k = rd_str()
            t = struct.unpack("<I", f.read(4))[0]
            meta[k] = rd_val(t)
        return meta


@lru_cache(maxsize=8)
def read_model_arch(gguf_path: str) -> Optional[Dict[str, int]]:
    """解析 .gguf 架构参数; 失败返回 None(不做估算)。"""
    if not os.path.isfile(gguf_path):
        return None
    try:
        meta = _read_gguf_metadata(gguf_path)
    except Exception:
        return None
    arch = str(meta.get("general.architecture", ""))
    if not arch:
        return None
    def gi(key, default=0):
        v = meta.get(f"{arch}.{key}")
        return int(v) if isinstance(v, (int, float)) else default
    block_count = gi("block_count")
    n_embd = gi("embedding_length")
    n_head = gi("attention.head_count")
    n_head_kv = gi("attention.head_count_kv")
    head_dim = gi("attention.key_length") or (n_embd // n_head if n_head else 0)
    nextn = max(0, gi("nextn_predict_layers"))
    interval = gi("full_attention_interval")
    n_layer = max(0, block_count - nextn)          # 真实 transformer 层数
    if interval > 0 and n_layer > 0:
        kv_layers = (n_layer + interval - 1) // interval   # 全注意力层数(带 KV)
    else:
        kv_layers = n_layer                                 # 非混合模型: 全部带 KV
    if kv_layers < 1 and n_layer > 0:
        kv_layers = n_layer
    return {
        "arch": arch,
        "n_layer": n_layer,
        "n_linear": max(0, n_layer - kv_layers),            # 线性注意力层数(无 KV)
        "kv_layers": kv_layers,
        "n_embd": n_embd,
        "n_head": n_head,
        "n_head_kv": n_head_kv,
        "head_dim": head_dim,
        "nextn": nextn,                                     # MTP/nextn 层数, >0 即支持 MTP
        "context_length": gi("context_length"),            # 训练最大上下文
        "model_bytes": os.path.getsize(gguf_path),          # 权重大致=文件大小
    }


# 每元素字节(按 KV 量化类型); "" 视为默认 F16
_KV_BYTES = {
    "": 2.0, "F16": 2.0, "BF16": 2.0, "F32": 4.0,
    "Q8_0": 34.0 / 32, "Q5_0": 22.0 / 32, "Q5_1": 24.0 / 32,
    "Q4_1": 20.0 / 32, "Q4_0": 18.0 / 32, "IQ4_NL": 17.0 / 32,
}


@lru_cache(maxsize=8)
def read_reasoning_effort(gguf_path: str) -> List[str]:
    """从模型内嵌的 chat template 解析 reasoning_effort 合法档位; 读不到则返回空表。

    llama.cpp 只透传 effort 值, 合法集合由模型模板校验(如 Qwen3.5 模板只认
    xhigh/medium/low)。这里直接读 GGUF 内嵌模板, 按模型自动适配; 模型未选/
    模板不可读时返回空表, 让下拉框仅保留"关闭思考"项。"""
    if not os.path.isfile(gguf_path):
        # 模型未选/文件缺失: 无模板可读, 返回空(下拉框仅留"关闭思考")
        return []
    try:
        meta = _read_gguf_metadata(gguf_path)
        tmpl = meta.get("tokenizer.chat_template", "")
        if isinstance(tmpl, str) and tmpl:
            m = re.search(r"reasoning_effort[^)]*?not\s+in\s*\(([^)]*)\)", tmpl, re.I)
            if not m:
                m = re.search(r"reasoning_effort[^)]*?in\s*\(([^)]*)\)", tmpl, re.I)
            if m:
                levels = [x.strip().strip("\'\"").lower()
                          for x in m.group(1).split(",") if x.strip()]
                levels = [l for l in levels if l]
                if levels:
                    return levels
    except Exception:
        pass
    return []


# 思考控制相关 Jinja kwarg 变量 —— 与官方 webui 的 chat-template-thinking-detector.ts 一致:
# 模板引用任一变量即视为支持思考强度控制(如 Qwen3.x 的 enable_thinking / reasoning_effort)
_THINKING_KWARG_VARS = ("enable_thinking", "reasoning_effort", "thinking_budget")
# 成对思考内容标签: 自闭合条目(<think></think>)只有起始标签且相邻出现
_THINKING_TAG_PATTERNS = (
    ("<think>", "</think>"),
    ("<|channel>thought", "<|channel|>"),
    ("<|think|>", "</|think|>"),
    ("<seed:think|>", "</seed:think|>"),
    ("<think></think>", None),
)
# Jinja 思考条件(模板原生 on/off 逻辑):
#   {% if enable_thinking %}/{% if enable thinking %}/{% if (enable_thinking is defined) %}
#   {% if not enable ... %}/{% if ns.enable_thinking %} 等
_JINJA_THINKING_CONDITIONALS = (
    re.compile(r"\{%-?\s*if\s+\(?\s*\w*enable[\s_]+\w*(thinking|think|reasoning)", re.I),
    re.compile(r"\{%-?\s*if\s+\w*(thinking|reasoning)\s*(is not|==|!=)", re.I),
    re.compile(r"\{%-?\s*if\s+not\s+\w*enable", re.I),
    re.compile(r"\{%-?\s*if\s+ns\.enable_thinking", re.I),
)


@lru_cache(maxsize=8)
def template_supports_thinking(gguf_path: str) -> bool:
    """判断模型 chat template 是否支持思考/推理(官方 webui 的 supportsThinking 等价物)。

    与模型是稠密还是 MoE 无关: 只要模板含 enable_thinking/reasoning_effort/
    thinking_budget 任一 kwarg 引用、思考类 Jinja 条件, 或成对思考标签(<think> 等),
    即视为支持思考。据此启用思考强度下拉(官方档位)。
    """
    if not os.path.isfile(gguf_path):
        return False
    try:
        meta = _read_gguf_metadata(gguf_path)
        t = meta.get("tokenizer.chat_template", "")
    except Exception:
        return False
    if not isinstance(t, str) or not t:
        return False
    for kw in _THINKING_KWARG_VARS:
        # {{ ...enable_thinking... }} 或 {% ...enable_thinking... %}
        if re.search(
            r"(\{\{[^{}]*\b" + kw + r"\b[^{}]*\}\}|\{%[^{}]*\b" + kw + r"\b[^{}]*%\})",
            t, re.I):
            return True
    for pat in _JINJA_THINKING_CONDITIONALS:
        if pat.search(t):
            return True
    for s, e in _THINKING_TAG_PATTERNS:
        if s in t and (not e or e in t):
            return True
    return False


def estimate_memory(engine: object, arch: Optional[Dict[str, int]]) -> Tuple[float, float]:
    """返回 (gpu_bytes, total_bytes); arch 为 None 或未知时返回 (0, 0)。"""
    if not arch or not arch.get("n_head_kv") or not arch.get("head_dim"):
        return 0.0, 0.0
    ctx = max(0, int(getattr(engine, "ctx_size", 0) or 0))
    ngl = max(0, int(getattr(engine, "ngl", 0) or 0))
    kv_k = (getattr(engine, "cache_type_k", "") or "").upper()
    kv_v = (getattr(engine, "cache_type_v", "") or "").upper()
    ubatch = max(1, int(getattr(engine, "ubatch_size", 0) or 0))

    n_head_kv = arch["n_head_kv"]
    head_dim = arch["head_dim"]
    kv_layers = arch["kv_layers"]
    model_bytes = arch["model_bytes"]
    n_embd = arch["n_embd"]

    # KV 缓存 (K,V 各取各自量化; 默认 F16=2B)
    b_k = _KV_BYTES.get(kv_k, 2.0)
    b_v = _KV_BYTES.get(kv_v, 2.0)
    kv_per_layer = n_head_kv * head_dim  # 单层单量的元素数
    kv_bytes = kv_per_layer * (b_k + b_v) * kv_layers * ctx

    # 线性注意力(混合模型)的循环状态, 与上下文无关, 粗略按每层 ~8 MiB
    recur_bytes = arch["n_linear"] * 8 * 1024 * 1024

    # 计算/工作区缓冲, 随 n_embd 与 ubatch 大致增长(含 CUDA graph 预留)
    compute_bytes = int(0.3 * 1024 ** 3 + n_embd * ubatch * 2 * 1.5)

    total = model_bytes + kv_bytes + recur_bytes + compute_bytes

    # GPU 部分: 权重按 ngl 卸载比例; KV/循环状态/工作区在启用 GPU 时随之上卡
    n_layer = arch["n_layer"]
    if ngl <= 0:
        gpu = 0.0
    else:
        frac = min((min(ngl, n_layer) + 1) / (n_layer + 1), 1.0) if n_layer > 0 else 0.0
        gpu = model_bytes * frac + kv_bytes + recur_bytes + compute_bytes
    return float(gpu), float(total)
