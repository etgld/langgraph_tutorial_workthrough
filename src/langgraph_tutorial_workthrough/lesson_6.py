from .lesson_6_utils import AgenticWriter, WriterGUI


def main() -> None:
    MultiAgenticWriter = AgenticWriter()
    app = WriterGUI(MultiAgenticWriter.get_state_graph())
    app.launch()


if __name__ == "__main__":
    main()
