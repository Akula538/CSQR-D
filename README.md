# CSQR-D — Curved Surface QR Decode

[Русская версия](README.ru.md)

QR code detection and decoding for QR codes placed on arbitrary smooth non-planar surfaces.

CSQR-D reconstructs the geometry of a curved QR code, rectifies the image, and then uses conventional QR decoding to recover the encoded data.

On the current benchmark dataset, the C++ implementation of CSQR-D decodes **163 of 169 images (96.45%)**. Detailed results and measurement conditions are provided below.

![Curved QR geometric correction](docs/images/Original_Corrected.png)

## What is CSQR-D?

Most QR decoders are designed to work with images where the QR code can be approximated by a planar projective transformation. When a QR code is printed on a curved surface, such as a cylinder or another smooth non-planar object, this assumption no longer holds: different parts of the code can be distorted differently.

CSQR-D is designed specifically for this type of geometric deformation. Instead of trying to make a conventional decoder handle the distorted image directly, it first recovers the geometry of the QR code and reconstructs an approximately regular representation that can be decoded by conventional QR readers.

The current implementation is intended for QR codes located on **smooth curved surfaces**. It is not designed primarily for arbitrary physical damage, tearing, or highly non-smooth deformation.

## Application

CSQR-D is integrated into **Curved QR Scanner**, an Android application designed for practical QR scanning. The application provides a convenient camera-based interface while using the CSQR-D scanner to handle QR codes that are difficult for conventional scanners to read.

[Download Curved QR Scanner from GitHub Releases](https://github.com/Akula538/CSQR-D/releases)

## Benchmark

### Dataset

The accuracy benchmark uses all **169 smartphone images** in `data`. Deformation levels are read from the `_d0`–`_d3` filename suffixes; QR error-correction levels L, M, Q and H are also encoded in the filenames.

- **0** — almost flat (12 samples)
- **1** — slightly curved (53 samples)
- **2** — noticeably curved (92 samples)
- **3** — strongly curved (12 samples)

For accuracy, images are downscaled to **2000 pixels on the longer side**, preserving aspect ratio (normally 2000×924 or 924×2000; one image is 965×2000). All five scanners receive the same resized images. Resizing uses Pillow LANCZOS; no other external preprocessing is applied.

### Recognition accuracy

Recognition rate is the fraction of images for which the scanner returns a nonempty decoded QR payload. CSQR-D outputs `None` and `CTF` count as failures. This measures decoding success, without checking payloads against an independent ground-truth annotation.

| Deformation | Samples | ZXing-C++ | ZBar / pyzbar | OpenCV | QReader | CSQR-D |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 12 | 58.33% | 50.00% | 33.33% | 75.00% | 100.00% |
| 1 | 53 | 37.74% | 16.98% | 13.21% | 33.96% | 100.00% |
| 2 | 92 | 9.78% | 6.52% | 0.00% | 6.52% | 93.48% |
| 3 | 12 | 0.00% | 0.00% | 0.00% | 0.00% | 100.00% |
| **Overall** | 169 | 21.30% | 12.43% | 6.51% | 19.53% | 96.45% |

### Runtime comparison

Runtime is measured on all **135 PNG images (640×640)** in `data2`, without resizing. The accompanying `labels.csv.gz` is not an image and is excluded. Each scanner performs one untimed warm-up, followed by **three measured calls per image** (405 measurements per scanner). Mean, median and P95 are calculated over all calls, including failed decoding attempts; P95 uses linear percentile interpolation.

| Scanner | Mean (ms) | Median (ms) | P95 (ms) |
|---|---:|---:|---:|
| ZXing-C++ | 8.70 | 8.55 | 11.27 |
| ZBar / pyzbar | 14.89 | 14.89 | 17.43 |
| OpenCV | 26.28 | 27.10 | 38.58 |
| QReader | 238.71 | 178.00 | 680.58 |
| CSQR-D | 39.55 | 34.47 | 78.73 |

The dataset is easier than the curved-QR dataset but is not decoded successfully by every scanner. Success counts on the first measured call per image: ZXing-C++: 112/135, ZBar: 98/135, OpenCV: 54/135, QReader: 126/135, CSQR-D: 132/135.

### CSQR-D runtime

This benchmark uses all **169 images in `data`**, downscaled to **1000 pixels on the longer side**, preserving aspect ratio. Each cell is the **mean runtime in milliseconds**, including successful and failed attempts, for that deformation / error-correction group. There is one measured call per image after an untimed warm-up. **Resizing and saving the resized image happen before timing and are excluded.**

| Deformation / EC | L | M | Q | H |
|---|---:|---:|---:|---:|
| 0 | 83.97 | 48.79 | 70.41 | 46.94 |
| 1 | 72.38 | 60.41 | 74.60 | 56.21 |
| 2 | 90.55 | 74.35 | 77.91 | 59.47 |
| 3 | 113.40 | 97.50 | 104.58 | 68.79 |

### Measurement environment and reproduction

Measured on Windows 11 with an **AMD Ryzen 7 6800HS with Radeon Graphics**, using Python 3.12.4. All benchmarks run sequentially on CPU. Comparison libraries: zxing-cpp 3.1.1, pyzbar 0.1.9, opencv-python 5.0.0.93, qreader 3.16, qrdet 2.5, torch 2.14.1. Input resizing used Pillow 10.3.0. QReader uses its small (`s`) model and confidence threshold 0.5; model loading, downloading and initialization are excluded from timing.

CSQR-D uses the existing **C++ Release CLI** for every benchmark, with `--seed 1`:

```powershell
.\src\cpp\build\bin\qrscanner_cli.exe <image-path> --seed 1
```

Timing uses `perf_counter_ns` and includes image reading. CSQR-D timings also include CLI process startup and shutdown; the other scanners use Python bindings in an already running process. These are invocation times for the specified interfaces, rather than isolated decoder kernel times. Some `data2` images contain multiple QR codes: CSQR-D returns one decoded payload, while the comparison bindings may return several. ZXing-C++ and ZBar are restricted to QR codes; OpenCV uses `QRCodeDetector.detectAndDecodeMulti`. Scanners retain their default internal preprocessing.

The [benchmark script](test/benchmark_readme.py), [pinned requirements](test/benchmark_requirements.txt), [per-image measurements](test/benchmark_results), [summary](test/benchmark_results/summary.json), and [environment / CLI hash](test/benchmark_results/environment.json) are included for reproduction. From the repository root:

```powershell
python -m pip install --target test/deps -r test/benchmark_requirements.txt
python test/benchmark_readme.py
python test/benchmark_readme.py --summarize --update-readme
```

Completed scanner / dataset runs are resumed from the JSONL files; move those files aside before a fresh measurement. Input hashes, dimensions and group labels are recorded in [the dataset manifest](test/benchmark_results/manifest.json).

## How it works

CSQR-D is organized as a multi-stage recovery pipeline. It first tries ordinary QR decoding and only performs the more expensive geometric reconstruction when necessary.

### 1. Initial decoding

The input image is first passed to conventional QR decoders. If the QR code is already readable, no geometric reconstruction is required.

### 2. Finder pattern detection

When standard decoding fails, CSQR-D searches for the three QR finder patterns.

The contour-based detector uses the structure of nested contours together with geometric constraints to identify candidate finder patterns and determine the configuration of the three patterns. This provides the geometric reference needed for the following reconstruction stages.

### 3. QR geometry reconstruction

The detected finder patterns are used to recover the position and orientation of the QR grid. For a planar QR code, a perspective transformation is usually sufficient. On a curved surface, however, the deformation varies across the code, so a single planar transformation is not enough.

CSQR-D therefore reconstructs additional information about the QR grid, including the distribution of its internal structure and, when available, alignment patterns.

### 4. Local deformation recovery

For strongly curved or otherwise difficult samples, CSQR-D estimates the local deformation of the QR image from image gradients.

First, Sobel derivatives are used to obtain local gradient directions. The image is then considered in local regions where the dominant gradient directions are estimated from peaks in the local orientation distribution. These directions form a vector field describing the local structure of the distorted QR image.

Integral curves are traced through this vector field. Their intersections provide a reconstructed grid of points corresponding to the regular structure of the QR code. A **Thin Plate Spline (TPS)** transformation is then fitted to these points and used to rectify the image.

### 5. Final decoding

After geometric correction, the reconstructed QR image is passed to a conventional QR decoder. The purpose of CSQR-D is therefore primarily to solve the **geometric recovery problem**, while the standard decoder handles the actual QR symbol decoding.

## Limitations

CSQR-D currently works best when:

- all three finder patterns are clearly visible;
- QR modules are approximately square rather than rounded;
- the central part of the QR code is not significantly occluded;
- the background does not contain strong directional textures or line patterns;
- the QR code is captured at sufficient resolution;
- the deformation is smooth rather than arbitrary or discontinuous.

The current implementation does not reliably support multiple QR codes in the same image.

## Dataset

The benchmark dataset is available directly in the repository:

[**CSQR-D dataset**](https://github.com/Akula538/CSQR-D/tree/main/data)

## Usage and integration

The repository contains both Python and C++ implementations of CSQR-D, so the scanner can also be integrated into other projects.

- [Python implementation](src/python)
- [C++ implementation](src/cpp)

The Android application is provided as a practical demonstration of the scanner:

[**Curved QR Scanner releases**](https://github.com/Akula538/CSQR-D/releases)

## License

See [LICENSE](LICENSE).
