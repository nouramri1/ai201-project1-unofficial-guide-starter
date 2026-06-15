"""
Gradio web UI — the front door of the pipeline.

Wires the question box to query.ask(), which runs retrieval + grounded generation.
Sources are formatted in code (not by the LLM) and shown in a separate box.

Usage:
    source .venv/bin/activate && python app.py
"""

from __future__ import annotations

import gradio as gr

from query import ask

# Brand colors: dark green background, light pink panels
DARK_GREEN = "#1a3a2a"
LIGHT_PINK = "#ffc8dd"
SOFT_PINK = "#fff0f5"

# Demo / eval questions — click an example to try it
EXAMPLE_QUESTIONS = [
    "What do students say about Kevin Pfeil's exam grading policy in COP3223C?",
    "How is Mark Llewellyn's CNT4714 class graded?",
    "What is Paul Gazzillo's attendance policy for COP3402?",
    "How is Heath Martin's MAC2312 course graded?",
    "What teaching style does Paul Lawrence use for CHM2210?",
    "What do students say about dining halls at UCF?",
]

theme = (
    gr.themes.Base(
        primary_hue=gr.themes.colors.green,
        secondary_hue=gr.themes.colors.pink,
    )
    .set(
        body_background_fill=DARK_GREEN,
        body_background_fill_dark=DARK_GREEN,
        block_background_fill=LIGHT_PINK,
        block_border_color=DARK_GREEN,
        block_label_text_color=DARK_GREEN,
        block_title_text_color=DARK_GREEN,
        body_text_color=DARK_GREEN,
        input_background_fill=SOFT_PINK,
        button_primary_background_fill=DARK_GREEN,
        button_primary_text_color=LIGHT_PINK,
        button_secondary_background_fill=LIGHT_PINK,
        button_secondary_text_color=DARK_GREEN,
    )
)


def handle_query(question: str) -> tuple[str, str]:
    """Take a user question, return (answer, bullet-list of source files)."""
    question = question.strip()
    if not question:
        return "Please enter a question.", ""

    try:
        result = ask(question)
        sources = "\n".join(f"• {source}" for source in result["sources"])
        return result["answer"], sources
    except ValueError as error:
        return str(error), ""
    except Exception as error:
        return f"Error: {error}", ""


with gr.Blocks(title="UCF Unofficial Course Guide", theme=theme) as demo:
    gr.Markdown(
        "# UCF Unofficial Course Guide\n"
        "*No secrets will stay hidden — know your professors and choose your classes wisely.*"
    )
    inp = gr.Textbox(
        label="Your question",
        placeholder="e.g. How is Kevin Pfeil's COP3223C class graded?",
        lines=2,
    )
    btn = gr.Button("Ask", variant="primary")
    answer = gr.Textbox(label="Answer", lines=8)
    sources = gr.Textbox(label="Retrieved from", lines=4)
    gr.Examples(examples=[[q] for q in EXAMPLE_QUESTIONS], inputs=inp)

    btn.click(handle_query, inputs=inp, outputs=[answer, sources])
    inp.submit(handle_query, inputs=inp, outputs=[answer, sources])


def main() -> None:
    demo.launch()


if __name__ == "__main__":
    main()
