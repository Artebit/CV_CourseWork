from __future__ import annotations

from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from PIL import Image as PillowImage


def markdown_to_blocks(text: str) -> list[tuple[str, str]]:
    blocks: list[tuple[str, str]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            blocks.append(("space", ""))
        elif line.startswith("# "):
            blocks.append(("title", line[2:]))
        elif line.startswith("## "):
            blocks.append(("heading2", line[3:]))
        elif line.startswith("### "):
            blocks.append(("heading3", line[4:]))
        elif line.startswith("- "):
            blocks.append(("body", "- " + line[2:]))
        elif line.startswith("|"):
            blocks.append(("body", line.replace("|", " | ")))
        else:
            blocks.append(("body", line))
    return blocks


def scaled_image(path: Path, max_width: float = 230, max_height: float = 150) -> Image:
    with PillowImage.open(path) as image:
        width, height = image.size
    scale = min(max_width / width, max_height / height)
    flowable = Image(str(path))
    flowable.drawWidth = width * scale
    flowable.drawHeight = height * scale
    return flowable


def add_result_images(story: list, root: Path, styles) -> None:
    outputs = root / "outputs"
    if not outputs.exists():
        return

    stages = [
        ("Original", "01_original.jpg"),
        ("Enhanced", "02_enhanced.jpg"),
        ("Segmentation mask", "03_segmentation_mask.jpg"),
        ("Cleaned mask", "04_cleaned_mask.jpg"),
        ("Detection result", "05_detection_result.jpg"),
        ("Scanned output", "06_scanned_document.jpg"),
    ]

    story.append(PageBreak())
    story.append(Paragraph("Stage-by-stage Results", styles["Heading2"]))
    story.append(Spacer(1, 8))

    for folder in sorted(path for path in outputs.iterdir() if path.is_dir()):
        story.append(Paragraph(folder.name, styles["Heading3"]))
        rows = []
        row = []
        for index, (caption, filename) in enumerate(stages, start=1):
            image_path = folder / filename
            if not image_path.exists():
                continue
            cell = [Paragraph(caption, styles["BodyText"]), scaled_image(image_path)]
            row.append(cell)
            if len(row) == 2 or index == len(stages):
                rows.append(row)
                row = []
        table = Table(rows, colWidths=[250, 250])
        table.setStyle(
            TableStyle(
                [
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("BOX", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                    ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )
        story.append(table)
        story.append(Spacer(1, 12))


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    markdown = root / "report.md"
    output = root / "report.pdf"

    styles = getSampleStyleSheet()
    story = []
    for kind, text in markdown_to_blocks(markdown.read_text(encoding="utf-8")):
        if kind == "space":
            story.append(Spacer(1, 8))
        elif kind == "title":
            story.append(Paragraph(text, styles["Title"]))
            story.append(Spacer(1, 12))
        elif kind == "heading2":
            story.append(Paragraph(text, styles["Heading2"]))
        elif kind == "heading3":
            story.append(Paragraph(text, styles["Heading3"]))
        else:
            story.append(Paragraph(text, styles["BodyText"]))

    add_result_images(story, root, styles)

    document = SimpleDocTemplate(str(output), pagesize=A4, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    document.build(story)
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
