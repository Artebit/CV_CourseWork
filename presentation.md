# 10-Minute Presentation Plan

## Slide 1 - Title

Document Scanner & Enhancer

Team roles:

- Lead CV Engineer
- Image Processing Specialist
- Morphology & Report Lead
- Data & Testing Engineer

## Slide 2 - Problem and Motivation

Real images of documents often have shadows, blur, perspective distortion, and noisy backgrounds. The goal is to automatically detect the document, improve readability, and output a clear final decision.

## Slide 3 - Full Pipeline

`image -> enhance -> segment -> clean -> detect -> decision`

Show one example folder from `outputs`.

## Slide 4 - Enhancement

Methods:

- LAB color conversion
- CLAHE on luminance
- denoising
- mild gamma correction

Show `01_original.jpg` and `02_enhanced.jpg`.

## Slide 5 - Segmentation

Methods:

- grayscale conversion
- Gaussian blur
- Canny edges
- adaptive threshold support

Show `03_segmentation_mask.jpg`.

## Slide 6 - Cleaning

Methods:

- morphological closing
- morphological opening
- connected-component filtering

Show `04_cleaned_mask.jpg`.

## Slide 7 - Detection and Perspective Output

Methods:

- contour sorting by area
- quadrilateral approximation
- fallback minimum-area rectangle
- perspective transform

Show `05_detection_result.jpg` and `06_scanned_document.jpg`.

## Slide 8 - Decision Logic

Automatic output:

- `PASS` if a document-like region is detected and covers at least 8% of the image
- `FAIL` otherwise

Current result: 3/3 test images passed.

## Slide 9 - Failure Cases

Expected weaknesses:

- weak boundary between paper and background
- heavy occlusion
- strong blur or darkness
- curved/folded document instead of flat quadrilateral

## Slide 10 - Demo and Conclusion

Run:

```powershell
& 'C:\Users\gg\AppData\Local\Python\bin\python.exe' src\document_scanner.py --input data\test --output outputs
```

Open the generated `outputs` folders and show the decision files.
