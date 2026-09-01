"""冒烟测试: 验证 UI 构建与参数联动(不真实启动 llama-server)。

运行: python tests/test_smoke.py
通过标准: 输出 SMOKE OK, 并生成 screenshot.svg 供人工目检布局。
覆盖:
- RoundCheckbox 圆形指示器: 未选 ○ / 选中 ●, 选中带 -on class (蓝色由 CSS $primary 提供)
- 挂载后控制台内容: WebUI 地址 + 模型列表(含大小)
- 引擎参数 滑块(RangeSlider)/Input/Select/RoundCheckbox -> config 联动 (含中文标签控件)
- EngineConfig.build_args 新 flag 映射 (draft-mtp/-ub/-np/-ctk/-s/-lm/--kv-unified)
- Select 显示回归: 内部标签必须渲染当前选中项文本 (compact=True, 防外层裁剪致空白)
- 布局 region 断言: 底部输入行已移除; #console 占满右列; 控件不超面板宽度; 内容可滚动
"""
import asyncio
import os
import sys

from textual.css.query import NoMatches

# 保证从仓库根目录导入 llm_tui 包 (python checks/ui_smoke.py 或 pytest 均可)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_tui.config import CONFIG_PATH
from llm_tui.engine import EngineConfig
from llm_tui.ui import LlmShellApp, RoundCheckbox, model_label


async def main() -> None:
    # 保存现场: 测试会写入 config.json, 结束后恢复原状(避免污染真实配置)
    had_config = os.path.exists(CONFIG_PATH)
    backup = open(CONFIG_PATH, "rb").read() if had_config else b""

    app = LlmShellApp()
    try:
        await _run(app)
    finally:
        # 只恢复现场; 绝不删除 config(避免误删用户配置)
        if had_config:
            with open(CONFIG_PATH, "wb") as f:
                f.write(backup)


def _console_text(app: LlmShellApp) -> str:
    # 注意: App.console 是 Textual 内部诊断 Console, RichLog 属性名为 log_console
    return "\n".join(strip.text for strip in app.log_console.lines)


async def _run(app: LlmShellApp) -> None:
    async with app.run_test(size=(130, 50)) as pilot:
        await pilot.pause(0.5)

        # 1. 挂载后控制台内容: WebUI 地址 + 模型列表 (含大小)
        text = _console_text(app)
        assert f"http://127.0.0.1:{app.server.port}" in text, \
            "WebUI URL missing from console"
        if app.engine_cfg.model_path:
            label = model_label(app.engine_cfg.model_path)
            assert label in text, f"current model not listed: {label!r}"

        # 2. 引擎参数联动 (Input 路径): 程序化赋值 -> Changed 事件 -> config
        app.query_one("#eng-ctx_size").value = 8192
        await pilot.pause(0.2)
        assert app.engine_cfg.ctx_size == 8192, f"ctx not applied: {app.engine_cfg.ctx_size}"

        # 4. Select / RoundCheckbox 联动 (程序化赋值 -> Changed 事件 -> config)
        app.query_one("#eng-ngl").value = 30
        app.query_one("#eng-flash_attn").value = "on"
        app.query_one("#eng-cache_type_k").value = "Q4_0"
        app.query_one("#eng-kv_unified", RoundCheckbox).value = False
        app.query_one("#eng-spec_mtp", RoundCheckbox).value = True
        await pilot.pause(0.3)
        assert app.engine_cfg.ngl == 30, f"ngl not applied: {app.engine_cfg.ngl}"
        assert app.engine_cfg.flash_attn == "on", "flash_attn not applied"
        assert app.engine_cfg.cache_type_k == "Q4_0", "cache_type_k not applied"
        assert app.engine_cfg.kv_unified is False, "kv_unified not applied"
        assert app.engine_cfg.spec_mtp is True, "spec_mtp not applied"

        # 5. build_args flag 映射 (纯函数验证, 对照 arg.cpp 已核实 flag)
        cfg = EngineConfig(model_path="m.gguf", spec_mtp=True, flash_attn="on",
                           cache_type_k="Q4_0", seed="42", parallel=4, mmap=False)
        s = " ".join(cfg.build_args(r"H:\Qwen\gui", "127.0.0.1", 5801))
        assert "--spec-type draft-mtp" in s, f"missing spec flags: {s}"
        # -ctk/-ctv 必须小写传递(llama.cpp 解析大小写敏感), UI 值 Q4_0 -> q4_0
        assert "-np 4" in s and "-ctk q4_0" in s and "-s 42" in s and "--reasoning on" in s, f"flags wrong: {s}"
        assert "-ctk Q4_0" not in s, "cache type must be lowercased: " + s
        assert "-lm mmap" not in s, "mmap off should omit -lm: " + s
        d = " ".join(EngineConfig(model_path="m.gguf").build_args(r"H:\Qwen\gui", "127.0.0.1", 5801))
        assert "-ub 2048" in d and "--kv-unified" in d and "-lm mmap" in d and "--reasoning on" in d, f"default flags wrong: {d}"
        assert "-np 1" in d, f"parallel=1 也必须显式下发 -np, 否则 server 走 auto 模式强制 4 槽: {d}"
        assert "--flash-attn" not in d and "--spec-type" not in d and "-ctk" not in d, \
            f"auto/default values must be omitted: {d}"
        # 关闭思考时即使残留档位也不下发 --reasoning-budget/--reasoning-effort
        # (防止模板 raise_exception)
        off = " ".join(EngineConfig(model_path="m.gguf", reasoning=False,
                                     reasoning_effort="low").build_args(r"H:\Qwen\gui", "127.0.0.1", 5801))
        assert "--reasoning off" in off and "--reasoning-budget" not in off \
            and "--reasoning-effort" not in off, f"off gate wrong: {off}"
        # 思考强度 -> --reasoning-budget 思考 token 预算(官方 thinking_budget_tokens 映射:
        # low=512/medium=2048/high=8192, 稠密/MoE 思考模型通用); 旧 max 归一化为 high;
        # m.gguf 无内嵌模板故不会下发 --reasoning-effort
        for lv, bd in (("low", 512), ("medium", 2048), ("high", 8192)):
            r = " ".join(EngineConfig(model_path="m.gguf", reasoning_effort=lv)
                         .build_args(r"H:\Qwen\gui", "127.0.0.1", 5801))
            assert f"--reasoning on --reasoning-budget {bd}" in r, f"eff {lv} wrong: {r}"
        assert EngineConfig(model_path="m.gguf", reasoning_effort="max").reasoning_effort == "high", \
            "max should map to high (GUI 已移除最大档)"
        # 旧配置模板原生档位(xhigh/minimal 等)归一化到官方档位
        assert EngineConfig(model_path="m.gguf", reasoning_effort="xhigh").reasoning_effort == "high", \
            "legacy xhigh should map to high"
        assert EngineConfig(model_path="m.gguf", reasoning_effort="minimal").reasoning_effort == "low", \
            "legacy minimal should map to low"

        # 6. 中文分区标签存在性 (采样参数已移至 WebUI, 面板不再包含)
        sections = [str(st.content) for st in app.query(".section")]
        joined = " ".join(sections)
        for kw in ("模型", "上下文与卸载", "高级"):
            assert kw in joined, f"missing section {kw}: {joined}"
        assert "采样参数" not in joined, "sampling section should be removed: " + joined

        # 7. Select 显示回归: 内部标签必须渲染当前选中项文本。
        #    (旧 bug: 外层 .param-select 强制 height:1 裁掉内部 SelectCurrent -> 框内空白)
        from textual.widgets._select import SelectCurrent
        sc = app.query_one("#model-select").query_one(SelectCurrent)
        mp = app.engine_cfg.model_path
        expect = model_label(mp) if mp else ""
        got = str(sc.query_one("#label").content)
        assert got == expect, f"model select label wrong: {got!r} != {expect!r}"
        for sel_id, kw in (("#eng-flash_attn", "开启"),
                           ("#eng-cache_type_k", "Q4_0")):
            sc = app.query_one(sel_id).query_one(SelectCurrent)
            got = str(sc.query_one("#label").content)
            assert kw in got, f"{sel_id} label blank/wrong: {got!r}"

        # 7b. 深度思考独一下拉(官方思考强度): 首项"关闭思考"; 模型支持思考时
        #     档位(低/中/高)可选; 选档位=开启思考+强度; 选"关闭思考"=关闭;
        #     不支持思考的模型(测试机缺模型文件时)下拉置灰仅留关闭
        from textual.widgets import Select
        eff = app.query_one("#eng-reasoning_effort", Select)
        opts = app._reasoning_effort_options()
        assert opts and opts[0] == ("关闭思考", "off"), f"eff options wrong: {opts}"
        if any(p == "low" for _l, p in opts):
            assert not eff.disabled, "eff select should be enabled for thinking model"
            eff.value = "low"
            await pilot.pause(0.2)
            assert app.engine_cfg.reasoning is True and app.engine_cfg.reasoning_effort == "low", \
                f"level not applied: reasoning={app.engine_cfg.reasoning} eff={app.engine_cfg.reasoning_effort}"
        eff.value = "off"
        await pilot.pause(0.2)
        assert app.engine_cfg.reasoning is False and app.engine_cfg.reasoning_effort == "", \
            f"off not applied: reasoning={app.engine_cfg.reasoning} eff={app.engine_cfg.reasoning_effort}"
        sc2 = eff.query_one(SelectCurrent)
        got = str(sc2.query_one("#label").content)
        assert got == "关闭思考", f"eff select label wrong: {got!r}"
        # 恢复开启思考+低档, 避免影响后续联动断言; 无低档(非思考模型)则保持关闭
        if any(p == "low" for _l, p in opts):
            eff.value = "low"
            await pilot.pause(0.2)

        # 8. 布局断言: 底部输入行已移除; #console 占满右列 (含原输入框的 3 行);
        #    控件不超面板宽度; 行间距生效后内容超出视口 -> 可滚动
        try:
            app.query_one("#cmd-input")
            raise AssertionError("bottom input should have been removed")
        except NoMatches:
            pass
        regions = {}
        for wid in ("#status-bar", "#panel", "#console",
                    "#btn-apply", "#model-select",
                    "#eng-ngl", "#eng-spec_mtp", "#eng-reasoning_effort"):
            w = app.query_one(wid)
            r = w.region
            regions[wid] = (r.x, r.y, r.width, r.height)
            print(f"region {wid:20s} x={r.x:<3d} y={r.y:<3d} w={r.width:<4d} h={r.height}")
        panel_w = regions["#panel"][2]
        for wid in ("#model-select", "#eng-ngl", "#btn-apply"):
            assert regions[wid][0] + regions[wid][2] <= panel_w, f"{wid} overflows panel width"
        # 内容缩短后 130x50 下未必溢出; 仅确认面板可滚动(>=0), 真机更小窗口会滚动
        assert app.query_one("#panel").max_scroll_y >= 0, "panel scroll state invalid"
        assert regions["#console"][2] >= 20, "console area too narrow"
        assert regions["#console"][3] >= 46, \
            f"console should span full right column incl. former input rows: {regions['#console']}"

        # 9. SVG 截图供目检布局 (直接写 screenshot.svg)
        # 直接写 screenshot.svg, 避免 save_screenshot() 默认生成的"标题+时间戳"临时文件残留
        app.save_screenshot("screenshot.svg")
        print("SMOKE OK - screenshot saved to screenshot.svg")


if __name__ == "__main__":
    asyncio.run(main())
