"""python -m llm_tui 入口。"""
from llm_tui.ui import LlmShellApp


def main() -> None:
    LlmShellApp().run()


if __name__ == "__main__":
    main()
