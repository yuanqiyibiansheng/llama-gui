"""配置读写与路径。运行状态(config.json)与聊天模板统一放在 config/ 子目录。"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import fields as dc_fields
from typing import Any, Dict, Optional

from llm_tui.engine import (DEFAULT_BIN_DIR, DEFAULT_MODEL_DIRS, DEFAULT_PORT,
                            EngineConfig, default_bin_dir, default_model_dirs)

HERE = os.path.dirname(os.path.abspath(__file__))
if getattr(sys, "frozen", False):
    # 打包成单文件 exe 后, __file__ 位于临时解压目录(重启即丢);
    # 改为把运行状态存到 exe 同级目录, 保证配置持久化且随 exe 一起分发。
    HERE = os.path.dirname(os.path.abspath(sys.executable))
# 配置(含运行时状态)统一放在 config/ 子目录: config.json + 聊天模板
CONFIG_PATH = os.path.join(HERE, "config", "config.json")
TEMPLATES_DIR = os.path.join(HERE, "config")


def _from_dict(cls, data: Optional[dict]):
    names = {f.name for f in dc_fields(cls)}
    return cls(**{k: v for k, v in (data or {}).items() if k in names})


def load_config() -> Dict[str, Any]:
    # 采样参数已移至 WebUI 设置; 旧配置中的 sampling 键会被忽略 (向后兼容)
    defaults: Dict[str, Any] = {
        "bin_dir": default_bin_dir(),
        "port": DEFAULT_PORT,
        "model_dirs": default_model_dirs(),
        "engine": EngineConfig().to_dict(),
    }
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                user = json.load(f)
            for key in ("engine",):
                if isinstance(user.get(key), dict):
                    merged = dict(defaults[key])
                    merged.update({k: v for k, v in user[key].items()
                                   if k in defaults[key]})
                    defaults[key] = merged
            for key in ("bin_dir", "port"):
                if key in user:
                    defaults[key] = user[key]
            if isinstance(user.get("model_dirs"), list):
                defaults["model_dirs"] = [d for d in user["model_dirs"]
                                          if isinstance(d, str) and d.strip()]
        except Exception as e:  # 配置损坏时回退默认值, 不阻塞启动
            print(f"[llm-shell] config load failed ({e}); using defaults")
    return defaults


def save_config(cfg: Dict[str, Any]) -> None:
    try:
        # 配置目录可能被清理, 写前确保存在
        os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[llm-shell] config save failed: {e}")
