"""思考强度端到端验证: 真实启动 llama-server, 用同一问题对比不同思考档位。

验证点:
1. reasoning_effort 档位映射的 --reasoning-budget 是否真正生效:
   小预算(128)思考被截断, 大预算(2048)思考更长, 可量化对比
2. 关闭思考(--reasoning off)后 message.reasoning_content 为空
3. /v1/chat/completions 响应带 reasoning_content(供第三方 Agent 读取)

运行: python checks/reasoning_effort.py
"""
import asyncio
import os
import sys

import httpx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_tui.engine import EngineConfig, LlamaServer, model_alias

# 本机缓存加载快的模型; 可换成任意 Qwen3.x/思考模型
MODEL = r"H:\Qwen\model\models\Qwen3.8-27B-UD-Q4_K_XL\Qwen3.8-27B-UD-Q4_K_XL.gguf"
PROMPT = ("请认真思考并逐步推理: 一个笼子里鸡和兔共 35 个头、94 只脚, "
          "鸡和兔各多少只? 请先充分推理再给出最终答案。")


class _CfgOverride:
    """临时覆盖 EngineConfig.build_args 产出的 --reasoning-budget(测试档位截断用)。"""

    def __init__(self, cfg: EngineConfig, budget: int | None, reasoning: bool):
        self._cfg = cfg
        self._budget = budget
        self._reasoning = reasoning

    def build_args(self, bin_dir: str, host: str, port: int) -> list:
        self._cfg.reasoning = self._reasoning
        args = self._cfg.build_args(bin_dir, host, port)
        if self._reasoning and self._budget is not None:
            if "--reasoning-budget" in args:
                args[args.index("--reasoning-budget") + 1] = str(self._budget)
            else:
                args += ["--reasoning-budget", str(self._budget)]
        return args


async def run_case(label: str, budget: int | None, reasoning: bool) -> dict:
    s = LlamaServer()
    base = EngineConfig(model_path=MODEL, ctx_size=1024, reasoning_effort="low")
    cfg = _CfgOverride(base, budget, reasoning)
    await s.spawn(cfg)  # type: ignore[arg-type]
    try:
        for _ in range(60):
            await asyncio.sleep(1.0)
            if not s.running:
                raise RuntimeError(f"server exited early: {s.last_log_line}")
            if await s.is_healthy():
                break
        else:
            raise RuntimeError("server not ready in 60s")
        payload = {
            "model": model_alias(MODEL),
            "messages": [{"role": "user", "content": PROMPT}],
            "stream": False,
            "max_tokens": 512,
            "temperature": 0.6,
        }
        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as c:
            r = await c.post(f"{s.base_url}/v1/chat/completions", json=payload)
            r.raise_for_status()
            data = r.json()
        msg = (data.get("choices") or [{}])[0].get("message") or {}
        reasoning = msg.get("reasoning_content") or ""
        content = msg.get("content") or ""
        return {
            "label": label,
            "reasoning_len": len(reasoning),
            "content_len": len(content),
            "reasoning_head": reasoning[:80].replace("\n", " "),
            "content_tail": content[-60:].replace("\n", " "),
            "usage": data.get("usage"),
        }
    finally:
        await s.stop()


async def main() -> None:
    cases = [
        ("关闭思考 (--reasoning off)", None, False),
        ("低 (预算 128)", 128, True),
        ("高 (预算 2048)", 2048, True),
    ]
    results = []
    for label, budget, reasoning in cases:
        print(f"[run] {label} ...")
        results.append(await run_case(label, budget, reasoning))

    print("\n==== 结果对比 ====")
    print(f"{'档位':<24}{'思考长度':>10}{'回答长度':>10}")
    for x in results:
        print(f"{x['label']:<24}{x['reasoning_len']:>10}{x['content_len']:>10}")
        print(f"  思考开头: {x['reasoning_head']}")
        print(f"  回答结尾: {x['content_tail']}")
        print(f"  usage: {x['usage']}")
    # 断言: 小预算思考显著短于大预算, 关闭思考无思考
    low = next(x for x in results if "预算 128" in x["label"])
    high = next(x for x in results if "预算 2048" in x["label"])
    off = next(x for x in results if "关闭思考" in x["label"])
    assert off["reasoning_len"] == 0, f"关闭思考仍有 reasoning_content: {off}"
    assert low["reasoning_len"] > 0, "低档位没有产生思考(思考未开启?)"
    assert low["reasoning_len"] < high["reasoning_len"], \
        f"思考预算未生效: 128={low['reasoning_len']} 应小于 2048={high['reasoning_len']}"
    print("\nREASONING EFFORT OK - 思考强度档位真实生效")


if __name__ == "__main__":
    asyncio.run(main())
