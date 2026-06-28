from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

SAMPLE_IMAGES = {
    "receipt_swiss.jpg": "https://commons.wikimedia.org/wiki/Special:Redirect/file/ReceiptSwiss.jpg",
    "receipt_agr.jpg": "https://commons.wikimedia.org/wiki/Special:Redirect/file/Receipt.agr.jpg",
    "invoice.jpg": "https://commons.wikimedia.org/wiki/Special:Redirect/file/Invoice.jpg",
}

SAMPLE_SOURCES = {
    "receipt_swiss.jpg": "https://commons.wikimedia.org/wiki/File:ReceiptSwiss.jpg",
    "receipt_agr.jpg": "https://commons.wikimedia.org/wiki/File:Receipt.agr.jpg",
    "invoice.jpg": "https://commons.wikimedia.org/wiki/File:Invoice.jpg",
}


@dataclass
class Decision:
    image: str
    status: str
    message: str
    confidence: float
    document_area_ratio: float
    quadrilateral_detected: bool
    output_width: int
    output_height: int


def safe_name(path: Path) -> str:
    name = path.stem.lower()
    name = re.sub(r"[^a-z0-9_-]+", "_", name)
    return name.strip("_") or "image"


def download_samples(input_dir: Path) -> None:
    input_dir.mkdir(parents=True, exist_ok=True)
    for filename, url in SAMPLE_IMAGES.items():
        destination = input_dir / filename
        if destination.exists() and destination.stat().st_size > 0:
            continue
        print(f"Downloading {filename} ...")
        request = urllib.request.Request(url, headers={"User-Agent": "cv-course-project/1.0"})
        with urllib.request.urlopen(request, timeout=45) as response:
            destination.write_bytes(response.read())

    source_file = input_dir / "SOURCES.md"
    lines = ["# Sample Image Sources", ""]
    for filename, source in SAMPLE_SOURCES.items():
        lines.append(f"- `{filename}`: {source}")
    source_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def resize_for_processing(image: np.ndarray, max_side: int = 1400) -> tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    largest = max(height, width)
    if largest <= max_side:
        return image.copy(), 1.0
    scale = max_side / largest
    resized = cv2.resize(image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
    return resized, scale


def order_points(points: np.ndarray) -> np.ndarray:
    pts = points.reshape(4, 2).astype("float32")
    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).reshape(4)
    ordered = np.zeros((4, 2), dtype="float32")
    ordered[0] = pts[np.argmin(sums)]
    ordered[2] = pts[np.argmax(sums)]
    ordered[1] = pts[np.argmin(diffs)]
    ordered[3] = pts[np.argmax(diffs)]
    return ordered


def perspective_warp(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    rect = order_points(points)
    tl, tr, br, bl = rect

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)

    max_width = max(1, int(max(width_a, width_b)))
    max_height = max(1, int(max(height_a, height_b)))

    destination = np.array(
        [[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]],
        dtype="float32",
    )
    matrix = cv2.getPerspectiveTransform(rect, destination)
    return cv2.warpPerspective(image, matrix, (max_width, max_height))


def enhance_image(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(8, 8))
    lightness = clahe.apply(lightness)
    enhanced = cv2.merge((lightness, a_channel, b_channel))
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 5, 5, 7, 21)

    gamma = 1.08
    table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(enhanced, table)


def segment_document(enhanced: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    median = float(np.median(blurred))
    lower = int(max(0, 0.66 * median))
    upper = int(min(255, 1.33 * median))
    edges = cv2.Canny(blurred, lower, upper)

    adaptive = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        6,
    )
    adaptive_edges = cv2.Canny(adaptive, 50, 160)
    mask = cv2.bitwise_or(edges, adaptive_edges)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return cv2.dilate(mask, kernel, iterations=1)


def clean_mask(mask: np.ndarray) -> np.ndarray:
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, open_kernel, iterations=1)

    components, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    filtered = np.zeros_like(cleaned)
    min_area = max(150, int(cleaned.shape[0] * cleaned.shape[1] * 0.0005))
    for label in range(1, components):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            filtered[labels == label] = 255
    return filtered


def detect_document(cleaned: np.ndarray, image_shape: tuple[int, int, int]) -> tuple[np.ndarray | None, float, bool]:
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, 0.0, False

    image_area = float(image_shape[0] * image_shape[1])
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    best_quad = None
    best_ratio = 0.0

    for contour in contours[:12]:
        area = cv2.contourArea(contour)
        if area < image_area * 0.025:
            continue
        perimeter = cv2.arcLength(contour, True)
        for epsilon_factor in (0.015, 0.02, 0.03, 0.04, 0.06):
            approx = cv2.approxPolyDP(contour, epsilon_factor * perimeter, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                ratio = area / image_area
                if ratio > best_ratio:
                    best_quad = approx.reshape(4, 2)
                    best_ratio = ratio
                break

    if best_quad is not None:
        return best_quad, best_ratio, True

    largest = contours[0]
    area = cv2.contourArea(largest)
    rect = cv2.minAreaRect(largest)
    box = cv2.boxPoints(rect).astype("float32")
    return box, area / image_area, False


def draw_detection(image: np.ndarray, points: np.ndarray | None, decision: str, confidence: float) -> np.ndarray:
    result = image.copy()
    if points is not None:
        polygon = points.reshape(-1, 1, 2).astype("int32")
        cv2.polylines(result, [polygon], True, (0, 180, 0), 4)
        for idx, point in enumerate(points.reshape(-1, 2).astype("int32"), start=1):
            cv2.circle(result, tuple(point), 7, (0, 80, 255), -1)
            cv2.putText(result, str(idx), tuple(point + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 80, 255), 2)

    label = f"{decision} | confidence={confidence:.2f}"
    cv2.rectangle(result, (12, 12), (min(result.shape[1] - 1, 560), 62), (255, 255, 255), -1)
    cv2.putText(result, label, (24, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (20, 20, 20), 2)
    return result


def readable_scan(scan: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(scan, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        25,
        12,
    )


def process_image(image_path: Path, output_root: Path) -> Decision:
    original = cv2.imread(str(image_path))
    if original is None:
        raise ValueError(f"Cannot read image: {image_path}")

    image, scale = resize_for_processing(original)
    enhanced = enhance_image(image)
    mask = segment_document(enhanced)
    cleaned = clean_mask(mask)
    points, area_ratio, is_quad = detect_document(cleaned, image.shape)

    confidence = min(1.0, max(0.0, area_ratio / 0.45))
    status = "PASS" if points is not None and area_ratio >= 0.08 else "FAIL"
    message = (
        "Document detected and perspective/readability output was generated."
        if status == "PASS"
        else "Document was not detected with enough confidence."
    )

    if points is not None and status == "PASS":
        scan = perspective_warp(enhanced, points)
        scan_output = readable_scan(scan)
    else:
        scan_output = readable_scan(enhanced)

    decision = Decision(
        image=image_path.name,
        status=status,
        message=message,
        confidence=round(float(confidence), 3),
        document_area_ratio=round(float(area_ratio), 3),
        quadrilateral_detected=bool(is_quad),
        output_width=int(scan_output.shape[1]),
        output_height=int(scan_output.shape[0]),
    )

    image_output = output_root / safe_name(image_path)
    image_output.mkdir(parents=True, exist_ok=True)

    detection = draw_detection(image, points, decision.status, decision.confidence)
    cv2.imwrite(str(image_output / "01_original.jpg"), image)
    cv2.imwrite(str(image_output / "02_enhanced.jpg"), enhanced)
    cv2.imwrite(str(image_output / "03_segmentation_mask.jpg"), mask)
    cv2.imwrite(str(image_output / "04_cleaned_mask.jpg"), cleaned)
    cv2.imwrite(str(image_output / "05_detection_result.jpg"), detection)
    cv2.imwrite(str(image_output / "06_scanned_document.jpg"), scan_output)
    (image_output / "decision.txt").write_text(
        "\n".join(
            [
                f"Image: {decision.image}",
                f"Final decision: {decision.status}",
                f"Message: {decision.message}",
                f"Confidence: {decision.confidence}",
                f"Document area ratio: {decision.document_area_ratio}",
                f"Quadrilateral detected: {decision.quadrilateral_detected}",
                f"Output size: {decision.output_width}x{decision.output_height}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (image_output / "decision.json").write_text(json.dumps(asdict(decision), indent=2), encoding="utf-8")
    return decision


def collect_images(input_dir: Path) -> list[Path]:
    return sorted(path for path in input_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)


def write_summary(output_dir: Path, decisions: list[Decision]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        "# Processing Summary",
        "",
        "| Image | Status | Confidence | Area Ratio | Quad |",
        "|---|---:|---:|---:|---:|",
    ]
    for decision in decisions:
        rows.append(
            "| "
            f"{decision.image} | {decision.status} | {decision.confidence:.3f} | "
            f"{decision.document_area_ratio:.3f} | {decision.quadrilateral_detected} |"
        )
    (output_dir / "summary.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (output_dir / "summary.json").write_text(
        json.dumps([asdict(decision) for decision in decisions], indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Document Scanner & Enhancer CV pipeline.")
    parser.add_argument("--input", type=Path, default=Path("data/test"), help="Folder with input images.")
    parser.add_argument("--output", type=Path, default=Path("outputs"), help="Folder for stage outputs.")
    parser.add_argument(
        "--download-samples",
        action="store_true",
        help="Download three open sample images from Wikimedia Commons before processing.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.download_samples:
        download_samples(args.input)

    if not args.input.exists():
        print(f"Input folder does not exist: {args.input}", file=sys.stderr)
        return 2

    images = collect_images(args.input)
    if not images:
        print(f"No images found in {args.input}", file=sys.stderr)
        return 2

    args.output.mkdir(parents=True, exist_ok=True)
    decisions: list[Decision] = []
    for image_path in images:
        print(f"Processing {image_path.name} ...")
        decisions.append(process_image(image_path, args.output))

    write_summary(args.output, decisions)
    passed = sum(1 for decision in decisions if decision.status == "PASS")
    print(f"Done. PASS: {passed}/{len(decisions)}. Results: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
