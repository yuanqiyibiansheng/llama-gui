"""llm-shell 入口。运行: python main.py"""
from llm_tui.ui import LlmShellApp


def main() -> None:
    LlmShellApp().run()


if __name__ == "__main__":
    main()
