"""
Entry point — launch the Gradio UI or run one question in the terminal.

Usage:
    python main.py                  # open UCF Unofficial Course Guide UI
    python main.py --cli "question" # print answer in terminal
"""

from __future__ import annotations

import argparse
import sys

from query import ask


def main() -> None:
    parser = argparse.ArgumentParser(description="UCF Unofficial Course Guide")
    parser.add_argument(
        "--cli",
        action="store_true",
        help="Run a single question in the terminal instead of launching the UI",
    )
    parser.add_argument("question", nargs="?", help="Question to ask (CLI mode only)")
    args = parser.parse_args()

    if args.cli:
        if not args.question:
            print('Usage: python main.py --cli "your question here"')
            sys.exit(1)
        result = ask(args.question)
        print(result["answer"])
        print("\nRetrieved from:")
        print(result["sources_text"])
        return

    from app import main as launch_app

    launch_app()


if __name__ == "__main__":
    main()
