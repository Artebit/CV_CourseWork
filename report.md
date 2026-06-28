# Document Scanner & Enhancer

## Problem Description

The selected project theme is **Documents - Document Scanner & Enhancer**. The goal is to process real images containing receipts, passport pages, or desk documents and automatically produce an improved document view. The system must detect the document region, correct perspective when possible, improve readability, and output an interpretable final decision.

## Team Roles and Task Division

| Role | Responsibility |
|---|---|
| Lead CV Engineer | Pipeline integration, contour detection, perspective correction, final decision logic |
| Image Processing Specialist | CLAHE enhancement, denoising, gamma correction, segmentation mask |
| Morphology & Report Lead | Morphological cleaning, output visualization, report preparation |
| Data & Testing Engineer | Test image collection, repeated runs, failure-case notes |

For a 3-student team, the Lead CV Engineer also covers Data & Testing.

## Pipeline Design

The system follows the required course pipeline:

`image -> enhance -> segment -> clean -> detect -> decision`

For each input image, the program writes the original image, enhanced image, segmentation mask, cleaned mask, detection result, scanned document, and decision files.

## Methods Used

### Enhance

The image is converted to LAB color space. CLAHE is applied to the luminance channel to increase local contrast while limiting over-amplification. The image is then denoised and adjusted with a mild gamma correction.

### Segment

The enhanced image is converted to grayscale and blurred. The segmentation stage combines Canny edges with edges extracted from an adaptive-threshold view. This makes the pipeline work on both high-contrast paper boundaries and documents with visible internal text lines.

### Clean

Morphological closing connects broken edge segments, while opening removes small isolated noise. Connected-component filtering removes tiny regions that cannot plausibly represent document boundaries.

### Detect

Contours are sorted by area. The detector searches for a large convex quadrilateral using polygon approximation. If no clean quadrilateral is found, it falls back to the minimum-area rectangle around the largest contour. The selected region is drawn on the detection output.

### Decide

The decision is automatic:

- `PASS`: a document-like region is detected and covers at least 8% of the image.
- `FAIL`: no region is detected with sufficient confidence.

Confidence is estimated from the detected area ratio. The decision is saved as text and JSON.

## Results

The project includes a command that downloads three open sample images from Wikimedia Commons:

- `receipt_swiss.jpg`
- `receipt_agr.jpg`
- `invoice.jpg`

After processing, the `outputs` folder contains one subfolder per test image. Each subfolder includes all required stage-by-stage files:

1. `01_original.jpg`
2. `02_enhanced.jpg`
3. `03_segmentation_mask.jpg`
4. `04_cleaned_mask.jpg`
5. `05_detection_result.jpg`
6. `06_scanned_document.jpg`
7. `decision.txt`
8. `decision.json`

The current run produced `PASS` decisions for all three test images:

| Image | Status | Confidence | Area Ratio | Quadrilateral |
|---|---:|---:|---:|---:|
| invoice.jpg | PASS | 1.000 | 0.909 | True |
| receipt_agr.jpg | PASS | 1.000 | 0.975 | True |
| receipt_swiss.jpg | PASS | 1.000 | 0.995 | True |

The detailed visual results are saved in:

- `outputs/invoice`
- `outputs/receipt_agr`
- `outputs/receipt_swiss`

## Failure Cases

The main expected failure cases are:

- A document has the same color as the background, so the boundary is weak.
- The document is heavily occluded by hands or other objects.
- The image is strongly blurred or too dark for reliable edge detection.
- The document is folded or curved, so a single quadrilateral is not a good model.

## Conclusion

The implemented system is a complete OpenCV-based document scanner pipeline. It satisfies all five required stages, runs automatically on real images, saves interpretable intermediate outputs, and produces a final decision for each test image.

## Contribution Statement

| Student | Role | Completed Tasks | Signature |
|---|---|---|---|
| Student 1 | Lead CV Engineer | Detection, decision logic, integration | |
| Student 2 | Image Processing Specialist | Enhancement and segmentation | |
| Student 3 | Morphology & Report Lead | Cleaning, visualization, report | |
| Student 4 | Data & Testing Engineer | Dataset and testing notes | |
