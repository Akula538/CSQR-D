# CSQR-D — Curved Surface QR Decode

[Русская версия](README.ru.md)

CSQR-D detects and decodes QR codes on smooth curved surfaces. It reconstructs the geometry of a distorted code and rectifies the image so that conventional QR decoders can read it.

The C++ implementation decodes **163 of 169 images (96.45%)** in the curved-QR benchmark.

![Curved QR geometric correction](docs/images/Original_Corrected.png)

## Application

CSQR-D powers **Curved QR Scanner**, an Android app for scanning QR codes that conventional readers struggle with.

[Download Curved QR Scanner](https://github.com/Akula538/CSQR-D/releases)

## Benchmark

### Dataset

The [dataset](data) contains **169 smartphone images** of curved QR codes, grouped by deformation:

- **0** — almost flat
- **1** — slightly curved
- **2** — noticeably curved
- **3** — strongly curved

### Recognition accuracy

Percentage of images successfully decoded by each scanner.

| Deformation | Samples | ZXing-C++ | ZBar / pyzbar | OpenCV | QReader | CSQR-D |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 12 | 58.33% | 50.00% | 33.33% | 75.00% | 100.00% |
| 1 | 53 | 37.74% | 16.98% | 13.21% | 33.96% | 100.00% |
| 2 | 92 | 9.78% | 6.52% | 0.00% | 6.52% | 93.48% |
| 3 | 12 | 0.00% | 0.00% | 0.00% | 0.00% | 100.00% |
| **Overall** | 169 | 21.30% | 12.43% | 6.51% | 19.53% | 96.45% |

### Runtime comparison

Processing time on **135 images** from the simpler [dataset](data2).

| Scanner | Mean (ms) | Median (ms) | P95 (ms) |
|---|---:|---:|---:|
| ZXing-C++ | 8.70 | 8.55 | 11.27 |
| ZBar / pyzbar | 14.89 | 14.89 | 17.43 |
| OpenCV | 26.28 | 27.10 | 38.58 |
| QReader | 238.71 | 178.00 | 680.58 |
| CSQR-D | 39.55 | 34.47 | 78.73 |

### CSQR-D runtime

Mean processing time in milliseconds on the curved-QR dataset, grouped by deformation and QR error-correction level (L, M, Q, H).

| Deformation / EC | L | M | Q | H |
|---|---:|---:|---:|---:|
| 0 | 83.97 | 48.79 | 70.41 | 46.94 |
| 1 | 72.38 | 60.41 | 74.60 | 56.21 |
| 2 | 90.55 | 74.35 | 77.91 | 59.47 |
| 3 | 113.40 | 97.50 | 104.58 | 68.79 |

[Benchmark methodology and reproduction](docs/TECHNICAL.md#benchmark-methodology)

## How it works

CSQR-D first attempts conventional decoding. If that fails, it locates the QR finder patterns, reconstructs the distorted grid and rectifies the image before decoding again.

[Algorithm details](docs/TECHNICAL.md#algorithm)

## Limitations

CSQR-D is designed for smooth geometric deformation. It works best with clearly visible finder patterns, square QR modules, an unobstructed centre and sufficient image resolution. Strong background textures, physical damage and abrupt deformation can reduce reliability.

The current implementation does not reliably support multiple QR codes in the same image.

## Usage and integration

The scanner is available in two implementations:

- [Python](src/python)
- [C++](src/cpp)

## License

See [LICENSE](LICENSE).
