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

The benchmark uses **169 smartphone images** of QR codes on curved surfaces from `data`. Samples are grouped into four deformation levels:

- **0** — almost flat (12 samples)
- **1** — slightly curved (53 samples)
- **2** — noticeably curved (92 samples)
- **3** — strongly curved (12 samples)

All scanners receive the same images, downscaled before evaluation without additional external preprocessing.

### Recognition accuracy

Recognition rate is the fraction of images successfully decoded at each deformation level. The overall result covers all 169 samples.

| Deformation | Samples | ZXing-C++ | ZBar / pyzbar | OpenCV | QReader | CSQR-D |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 12 | 58.33% | 50.00% | 33.33% | 75.00% | 100.00% |
| 1 | 53 | 37.74% | 16.98% | 13.21% | 33.96% | 100.00% |
| 2 | 92 | 9.78% | 6.52% | 0.00% | 6.52% | 93.48% |
| 3 | 12 | 0.00% | 0.00% | 0.00% | 0.00% | 100.00% |
| **Overall** | 169 | 21.30% | 12.43% | 6.51% | 19.53% | 96.45% |

### Runtime comparison

Runtime is measured on **135 images** from the simpler `data2` dataset. Each image is scanned three times; the table reports mean, median and P95 processing time. P95 is the time within which 95% of calls complete.

| Scanner | Mean (ms) | Median (ms) | P95 (ms) |
|---|---:|---:|---:|
| ZXing-C++ | 8.70 | 8.55 | 11.27 |
| ZBar / pyzbar | 14.89 | 14.89 | 17.43 |
| OpenCV | 26.28 | 27.10 | 38.58 |
| QReader | 238.71 | 178.00 | 680.58 |
| CSQR-D | 39.55 | 34.47 | 78.73 |

### CSQR-D runtime

This test measures CSQR-D on the curved-QR dataset. The table shows mean processing time in milliseconds, grouped by deformation and QR error-correction level (L, M, Q, H). Images are downscaled before scanning; resizing time is excluded.

| Deformation / EC | L | M | Q | H |
|---|---:|---:|---:|---:|
| 0 | 83.97 | 48.79 | 70.41 | 46.94 |
| 1 | 72.38 | 60.41 | 74.60 | 56.21 |
| 2 | 90.55 | 74.35 | 77.91 | 59.47 |
| 3 | 113.40 | 97.50 | 104.58 | 68.79 |

CSQR-D uses the C++ implementation in all tests. [Detailed methodology, environment and reproduction](docs/BENCHMARKS.md).

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
