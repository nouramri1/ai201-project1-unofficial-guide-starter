"""
End-to-end query — retrieval + generation in one call.

Used by app.py (Gradio UI) and main.py (CLI).

Usage:
    python query.py "How is Mark Llewellyn's CNT4714 class graded?"
"""

from __future__ import annotations

import sys

from generate import generate_answer


def ask(question: str) -> dict:
    """Main entry point: question in → answer + sources out."""
    return generate_answer(question)


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python query.py "your question here"')
        sys.exit(1)

    question = " ".join(sys.argv[1:])
    result = ask(question)
    print(result["answer"])
    print()
    print("Retrieved from:")
    print("\n".join(f"• {source}" for source in result["sources"]))


if __name__ == "__main__":
    main()
