# -*- mode: python ; coding: utf-8 -*-
import os
from PyInstaller.utils.hooks import collect_all

datas = []
binaries = []
hiddenimports = []
# 显式收集常用第三方包; llm_tui 在 .py 模式收集源码+数据, 在 .pyd 模式收集(可能被跳过, 由下方兜底)
for _pkg in ('llm_tui', 'textual', 'rich', 'httpx'):
    _r = collect_all(_pkg)
    datas += _r[0]
    binaries += _r[1]
    hiddenimports += _r[2]

# ---------------------------------------------------------------------------
# 兼容 Cython/Encrypt 编译的 .pyd:
#   PyInstaller 无法静态分析 .pyd(编译后的扩展模块)内部的 import, 且 collect_all
#   对"纯 .pyd 构成的包"采集不到子模块 —— 这就是此前打包后运行时报
#   "ModuleNotFoundError: No module named 'llm_tui.config'" 的根因。
#   解决: 显式把 llm_tui 全部子模块加入 hiddenimports, 并把 .pyd 作为二进制打进包,
#         这样运行时 import llm_tui.config / engine / memory / slider 才能命中。
#   .py 源码模式下这些 .pyd 不存在(下方自动跳过), 不影响; .py 子模块由
#   collect_all('llm_tui') 已收录, 这里再兜底一遍。
# ---------------------------------------------------------------------------
_lt = os.path.join('H:\\Qwen\\gui\\llm-tui', 'llm_tui')
for _m in ('config', 'engine', 'memory', 'slider', 'ui', '__main__'):
    hiddenimports.append('llm_tui.' + _m)
    _pyd = os.path.join(_lt, _m + '.pyd')
    if os.path.exists(_pyd):
        binaries.append((_pyd, 'llm_tui'))


a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='llamaGui',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['H:/Qwen/gui/llm-tui/SetupIcon.ico'],
)
