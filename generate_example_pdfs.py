from __future__ import annotations
import re
from pathlib import Path
from typing import List

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "example_pdfs"


def discover_examples(base_dir: Path) -> List[Path]:
    ignore_names = {"example_pdfs", ".git", "__pycache__", "README.md", Path(__file__).name}
    examples: List[Path] = []
    for entry in sorted(base_dir.iterdir()):
        if entry.name in ignore_names or entry.name.startswith('.'):
            continue
        if entry.is_dir():
            examples.append(entry)
        elif entry.suffix.lower() in {".py", ".md"}:
            # treat standalone top-level resources as examples too
            examples.append(entry)
    return examples


def human_title(name: str) -> str:
    cleaned = re.sub(r"[_-]+", " ", name)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned.title()


def draw_wrapped_lines(c: canvas.Canvas, text: str, x: float, y: float, max_width: float, line_height: float = 16):
    from textwrap import wrap

    for line in wrap(text, width=int(max_width / 7)):  # rough character-to-width heuristic
        c.drawString(x, y, line)
        y -= line_height
    return y


def add_title_page(c: canvas.Canvas, title: str):
    width, height = letter
    c.setFont("Helvetica-Bold", 32)
    c.drawCentredString(width / 2, height - 200, title)
    c.setFont("Helvetica", 18)
    c.drawCentredString(width / 2, height - 240, "Solution Overview")
    c.showPage()


def add_architecture_page(c: canvas.Canvas, title: str):
    width, height = letter
    c.setFont("Helvetica-Bold", 22)
    c.drawCentredString(width / 2, height - 72, f"{title} Architecture")

    box_width = 180
    box_height = 60
    y = height - 200
    x_positions = [72, width / 2 - box_width / 2, width - box_width - 72]
    labels = ["Inputs", "Processing", "Outputs"]

    c.setStrokeColor(colors.darkblue)
    c.setFillColor(colors.lightblue)

    boxes = []
    for x, label in zip(x_positions, labels):
        c.rect(x, y, box_width, box_height, fill=1, stroke=1)
        c.setFillColor(colors.darkblue)
        c.setFont("Helvetica-Bold", 14)
        c.drawCentredString(x + box_width / 2, y + box_height / 2, label)
        c.setFillColor(colors.lightblue)
        boxes.append((x, y))

    c.setStrokeColor(colors.gray)
    c.setLineWidth(2)
    for idx in range(len(boxes) - 1):
        x1 = boxes[idx][0] + box_width
        x2 = boxes[idx + 1][0]
        c.line(x1, y + box_height / 2, x2, y + box_height / 2)
        c.line(x2 - 10, y + box_height / 2 + 5, x2, y + box_height / 2)
        c.line(x2 - 10, y + box_height / 2 - 5, x2, y + box_height / 2)

    c.setFillColor(colors.black)
    c.setFont("Helvetica", 12)
    details = [
        "Sample data sources, prompts, and telemetry feed the system.",
        "Models, tools, or pipelines orchestrate processing and reasoning.",
        "Insights, actions, or API responses are returned to applications.",
    ]
    for idx, detail in enumerate(details):
        draw_wrapped_lines(c, detail, x_positions[idx], y - 20, box_width, 14)

    c.showPage()


def add_technical_page(c: canvas.Canvas, title: str):
    width, height = letter
    c.setFont("Helvetica-Bold", 22)
    c.drawCentredString(width / 2, height - 72, "Technical Definition")

    c.setFont("Helvetica", 12)
    y = height - 120
    bullet_points = [
        f"{title} establishes a reusable pattern for building AI-first workflows.",
        "Core components: data ingestion, model execution, orchestration, and observability.",
        "Interfaces expose clear contracts for prompts, inputs, and downstream integrations.",
        "Security considerations include input validation, safe tool invocation, and governance hooks.",
    ]

    for point in bullet_points:
        y = draw_wrapped_lines(c, f"• {point}", 72, y, width - 144)
        y -= 8

    c.showPage()


def add_business_value_page(c: canvas.Canvas, title: str):
    width, height = letter
    c.setFont("Helvetica-Bold", 22)
    c.drawCentredString(width / 2, height - 72, "Business Value")

    c.setFont("Helvetica", 12)
    y = height - 120
    benefits = [
        "Accelerates delivery of AI features with consistent architecture and documentation.",
        "Reduces operational risk via transparent data flows and measurable checkpoints.",
        "Improves team alignment by linking technical design to measurable outcomes.",
        f"Differentiates the product by packaging {title} capabilities into customer-facing value.",
    ]

    for benefit in benefits:
        y = draw_wrapped_lines(c, f"• {benefit}", 72, y, width - 144)
        y -= 8

    c.showPage()


def build_pdf(example_path: Path):
    title = human_title(example_path.name)
    OUTPUT_DIR.mkdir(exist_ok=True)
    pdf_path = OUTPUT_DIR / f"{example_path.name}.pdf"
    c = canvas.Canvas(str(pdf_path), pagesize=letter)

    add_title_page(c, title)
    add_architecture_page(c, title)
    add_technical_page(c, title)
    add_business_value_page(c, title)

    c.save()
    return pdf_path


def main():
    examples = discover_examples(BASE_DIR)
    if not examples:
        print("No examples found.")
        return

    OUTPUT_DIR.mkdir(exist_ok=True)
    for example in examples:
        pdf_path = build_pdf(example)
        print(f"Created: {pdf_path.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
