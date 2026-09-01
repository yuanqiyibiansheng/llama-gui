"""RangeSlider: 单行数值滑块控件(Textual 8.x 核心已移除 Slider, 此处自实现)。

渲染一个水平条 + 手柄 "────●──── 值", 支持:
- 鼠标点击/按住拖动定位
- 左右/上下方向键按 step 微调
- 程序化赋 .value 会钳制到 [minimum, maximum] 并按 step 对齐, 变化时发 Changed 消息
"""
from __future__ import annotations

from textual.content import Content
from textual.reactive import reactive
from textual.widget import Widget
from textual.message import Message


class RangeSlider(Widget):
    class Changed(Message):
        """值变化消息(由滑动/按键/程序化赋值触发)。"""

        def __init__(self, slider: "RangeSlider", value: int) -> None:
            self.slider = slider
            self.value = value
            super().__init__()

    can_focus = True
    # 滑块拖动时不参与文本框选(避免拖出蓝色蒙版)
    ALLOW_SELECT = False

    def __init__(self, minimum: int = 0, maximum: int = 100, step: int = 1,
                 value: int = 0, *, name: str | None = None, id: str | None = None,
                 classes: str | None = None, disabled: bool = False) -> None:
        super().__init__(name=name, id=id, classes=classes, disabled=disabled)
        self.minimum = int(minimum)
        self.maximum = int(maximum)
        self.step = int(step)
        self._value = self._clamp(int(value))
        self._dragging = False

    @property
    def value(self) -> int:
        return self._value

    @value.setter
    def value(self, v: int) -> None:
        v = self._clamp(int(v))
        if v != self._value:
            self._value = v
            self.refresh()
            self.post_message(self.Changed(self, v))

    def _clamp(self, v: int) -> int:
        v = max(self.minimum, min(self.maximum, v))
        if self.step > 1:
            v = self.minimum + round((v - self.minimum) / self.step) * self.step
            v = max(self.minimum, min(self.maximum, v))
        return v

    def _val_width(self) -> int:
        # 数值固定宽度(按最大值的位数), 拖动时数字变长不会让条跳动
        return max(len(str(self.maximum)), len(str(self.minimum)))

    def _bar_width(self) -> int:
        return max(self.size.width - self._val_width() - 1, 2)

    def render(self):
        vtext = str(self._value).rjust(self._val_width())
        w = self._bar_width()
        span = self.maximum - self.minimum
        frac = 0.0 if span <= 0 else (self._value - self.minimum) / span
        pos = max(0, min(w - 1, round(frac * (w - 1))))
        # 手柄用主题蓝($primary), 轨道/数字用 widget 默认色(白)
        try:
            handle_color = self.app.get_css_variables().get("primary", "#0178D4")
        except Exception:
            handle_color = "#0178D4"
        parts = []
        if pos > 0:
            parts.append(("━" * pos, None))
        parts.append(("●", handle_color))
        if w - 1 - pos > 0:
            parts.append(("━" * (w - 1 - pos), None))
        parts.append((" " + vtext, None))
        return Content.assemble(*parts)

    def _set_value_from_x(self, x: int) -> None:
        w = self._bar_width()
        span = self.maximum - self.minimum
        x = max(0, min(w - 1, x))
        frac = max(0.0, min(1.0, x / max(w - 1, 1)))
        self.value = self.minimum + round(frac * span)

    def on_mouse_down(self, event) -> None:
        if event.button == 1:
            # 只点滑块条(不含末尾数字)才触发; 点击数字不要改值
            if event.x < self._bar_width():
                self._dragging = True
                self.capture_mouse()  # 捕获鼠标, 拖动全程事件归滑块, 直到松开
                self._set_value_from_x(event.x)
            event.stop()

    def on_mouse_move(self, event) -> None:
        # 拖动时实时跟随
        if self._dragging:
            self._set_value_from_x(event.x)
            event.stop()

    def on_mouse_up(self, event) -> None:
        if event.button == 1:
            self._dragging = False
            self.release_mouse()  # 松开后释放鼠标捕获

    def on_key(self, event) -> None:
        if event.key in ("left", "down"):
            self.value = self._value - self.step
            event.stop()
        elif event.key in ("right", "up"):
            self.value = self._value + self.step
            event.stop()
