"""集成测试: 真实启动 llama-server, 验证进程拉起与模型加载进度。

不等待 16GB 模型完整加载(耗时过长), 只验证:
1. 子进程成功拉起且参数被接受(无 unknown argument / 立即退出)
2. stdout 出现加载进度日志
3. stop() 能干净终止进程

运行: python tests/test_server_boot.py
"""
import asyncio
import os
import sys

# 保证从仓库根目录导入 llm_tui 包
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_tui.engine import EngineConfig, LlamaServer

MODEL = r"H:\Qwen\model\models\Qwen3.8-27B-UD-Q4_K_XL\Qwen3.8-27B-UD-Q4_K_XL.gguf"


async def main() -> None:
    s = LlamaServer()
    # 验证新增的 log_callback: TUI 用它把引擎日志逐行转发到控制台
    got_lines: list[str] = []
    s.log_callback = lambda line: got_lines.append(line)
    # 小 ctx 加快加载; ngl=99 默认(有 CUDA dll, 无 GPU 时 llama-server 自动回退 CPU)
    # reasoning_effort="high": 同时验证官方思考强度 flag(--reasoning-budget 8192 +
    # Qwen3.8 模板原生 --reasoning-effort xhigh)被真实 llama-server 接受
    cfg = EngineConfig(model_path=MODEL, ctx_size=1024, reasoning_effort="high")
    args = cfg.build_args(s.bin_dir, s.host, s.port)
    assert "--reasoning-budget" in args and "--reasoning-effort" in args, \
        f"thinking flags missing: {args}"
    print("[args] reasoning flags ok:", [
        f"{args[i]}={args[i + 1]}" for i in range(len(args) - 1)
        if args[i] in ("--reasoning-budget", "--reasoning-effort")])
    await s.spawn(cfg)
    print(f"[boot] pid={s.proc.pid}")

    for i in range(30):
        await asyncio.sleep(1.0)
        if not s.running:
            code = s.proc.returncode
            print(f"[FAIL] server exited early, code={code}")
            print("[last]", s.last_log_line)
            sys.exit(1)
        print(f"[{i + 1:>2}s] {s.last_log_line[:110]}")

    assert len(got_lines) > 0, "log_callback never fired"
    print(f"[callback] ok - {len(got_lines)} log lines forwarded")
    healthy = await s.is_healthy()
    print("[health]", "READY" if healthy else "still loading (expected for 16GB model)")
    await s.stop()
    assert not s.running, "stop failed: process still running"
    print("[stop] ok - process terminated cleanly")


if __name__ == "__main__":
    asyncio.run(main())
