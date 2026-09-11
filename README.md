# CSQR-D — Curved Surface QR Decode

QR code detection and decoding for QR codes placed on arbitrary smooth non-planar surfaces.

CSQR-D reconstructs the geometry of a curved QR code, rectifies the image, and then uses conventional QR decoding to recover the encoded data.

On the current benchmark dataset, conventional QR scanners recognize only about 20–30% of the samples, while CSQR-D reaches 95–98% depending on the deformation level.

> **Project status:** active experimental project. The current release focuses on curved QR codes on smooth surfaces rather than arbitrary QR damage or occlusion.

![Curved QR geometric correction](docs/images/placeholder-correction.png)

## What is CSQR-D?

Most QR decoders are designed to work with images where the QR code can be approximated by a planar projective transformation. When a QR code is printed on a curved surface, such as a cylinder or another smooth non-planar object, this assumption no longer holds: different parts of the code can be distorted differently.

CSQR-D is designed specifically for this type of geometric deformation. Instead of trying to make a conventional decoder handle the distorted image directly, it first recovers the geometry of the QR code and reconstructs an approximately regular representation that can be decoded by conventional QR readers.

The current implementation is intended for QR codes located on **smooth curved surfaces**. It is not designed primarily for arbitrary physical damage, tearing, or highly non-smooth deformation.

## Application

CSQR-D is integrated into **Curved QR Scanner**, an Android application designed for practical QR scanning. The application provides a convenient camera-based interface while using the CSQR-D scanner to handle QR codes that are difficult for conventional scanners to read.

[Download Curved QR Scanner from GitHub Releases](https://github.com/Akula538/CSQR-D/releases)

## Benchmark

### Dataset

The benchmark dataset contains **169 smartphone images** of curved QR codes. Before evaluation, images are downscaled to **2000×923**.

Four deformation levels are used:

- **0** — almost flat
- **1** — slightly curved
- **2** — noticeably curved
- **3** — strongly curved

The samples are distributed approximately uniformly between the four deformation levels.

All scanners are evaluated on the same input images without additional preprocessing.

### Recognition accuracy

Recognition rate is the fraction of samples successfully decoded by each scanner.

| Deformation | Samples | ZXing-C++ | ZBar / pyzbar | OpenCV | QReader | CSQR-D |
|---|---:|---:|---:|---:|---:|---:|
| 0 | — | — | — | — | — | — |
| 1 | — | — | — | — | — | — |
| 2 | — | — | — | — | — | — |
| 3 | — | — | — | — | — | — |
| **Overall** | **169** | — | — | — | — | — |

### Runtime comparison

Runtime is measured separately on a simple dataset containing QR codes that can be decoded by conventional scanners without geometric correction. This makes the comparison focus on the computational overhead of the scanners rather than on whether a particular scanner succeeds on difficult curved samples.

| Scanner | Mean (ms) | Median (ms) | P95 (ms) |
|---|---:|---:|---:|
| ZXing-C++ | — | — | — |
| ZBar / pyzbar | — | — | — |
| OpenCV | — | — | — |
| QReader | — | — | — |
| CSQR-D | — | — | — |

### CSQR-D runtime

The following benchmark characterizes CSQR-D itself on the curved-QR dataset. Runtime is measured for combinations of deformation level and QR error-correction level.

| Deformation / EC | L | M | Q | H |
|---|---:|---:|---:|---:|
| 0 | — | — | — | — |
| 1 | — | — | — | — |
| 2 | — | — | — | — |
| 3 | — | — | — | — |

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
