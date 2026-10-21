# Benchmark methodology

[Back to README](../README.md) · [Русская версия](BENCHMARKS.ru.md)

### Dataset

The accuracy benchmark uses all **169 smartphone images** in `data/qr/data1`. Deformation levels are read from the `_d0`–`_d3` filename suffixes; QR error-correction levels L, M, Q and H are also encoded in the filenames.

- **0** — almost flat (12 samples)
- **1** — slightly curved (53 samples)
- **2** — noticeably curved (92 samples)
- **3** — strongly curved (12 samples)

For accuracy, images are downscaled to **2000 pixels on the longer side**, preserving aspect ratio (normally 2000×924 or 924×2000; one image is 965×2000). All five scanners receive the same resized images. Resizing uses Pillow LANCZOS; no other external preprocessing is applied.

### Recognition accuracy

Recognition rate is the fraction of images for which the scanner returns a nonempty decoded QR payload. CSQR-D outputs `None` and `CTF` count as failures. This measures decoding success, without checking payloads against an independent ground-truth annotation.


### Runtime comparison

Runtime is measured on all **135 PNG images (640×640)** in `data/qr/data2`, without resizing. The accompanying `labels.csv.gz` is not an image and is excluded. Each scanner performs one untimed warm-up, followed by **three measured calls per image** (405 measurements per scanner). Mean, median and P95 are calculated over all calls, including failed decoding attempts; P95 uses linear percentile interpolation.


The dataset is easier than the curved-QR dataset but is not decoded successfully by every scanner. Success counts on the first measured call per image: ZXing-C++: 112/135, ZBar: 98/135, OpenCV: 54/135, QReader: 126/135, CSQR-D: 133/135.

### CSQR-D runtime

This benchmark uses all **169 images in `data/qr/data1`**, downscaled to **1000 pixels on the longer side**, preserving aspect ratio. Each cell is the **mean runtime in milliseconds**, including successful and failed attempts, for that deformation / error-correction group. There is one measured call per image after an untimed warm-up. **Resizing and saving the resized image happen before timing and are excluded.**


### Measurement environment and reproduction

All five scanners were remeasured on 20 October 2026 in one sequential CPU session using the same input caches.

Measured on Windows 11 with an **AMD Ryzen 7 6800HS with Radeon Graphics**, using Python 3.12.4. All benchmarks run sequentially on CPU. Comparison libraries: zxing-cpp 3.1.1, pyzbar 0.1.9, opencv-python 5.0.0.93, qreader 3.16, qrdet 2.5, torch 2.14.1. Input resizing used Pillow 12.3.0. QReader uses its small (`s`) model and confidence threshold 0.5; model loading, downloading and initialization are excluded from timing.

CSQR-D uses the existing **C++ Release CLI** for every benchmark, with `--seed 1`:

```powershell
.\src\cpp\build\bin\qrscanner_cli.exe <image-path> --seed 1
```

Timing uses `perf_counter_ns` and includes image reading. CSQR-D timings also include CLI process startup and shutdown; the other scanners use Python bindings in an already running process. These are invocation times for the specified interfaces, rather than isolated decoder kernel times. Some `data/qr/data2` images contain multiple QR codes: CSQR-D returns one decoded payload, while the comparison bindings may return several. ZXing-C++ and ZBar are restricted to QR codes; OpenCV uses `QRCodeDetector.detectAndDecodeMulti`. Scanners retain their default internal preprocessing.

The [benchmark script](../test/benchmark_readme.py), [pinned requirements](../test/benchmark_requirements.txt), [per-image measurements](../test/benchmark_results), [summary](../test/benchmark_results/summary.json), and [environment / CLI hash](../test/benchmark_results/environment.json) are included for reproduction. From the repository root:

```powershell
python -m pip install --target test/deps -r test/benchmark_requirements.txt
python test/benchmark_readme.py
python test/benchmark_readme.py --fresh
python test/benchmark_readme.py --scanners CSQR-D --fresh
python test/benchmark_readme.py --summarize --update-readme
```

Completed scanner / dataset runs are resumed from the JSONL files; `--fresh` replaces only the selected scanners' measurements. Input hashes, dimensions and group labels are recorded in [the dataset manifest](../test/benchmark_results/manifest.json). Resized input caches are generated on demand and are not committed.

### Additional real datasets and Python port

The C++ scanner also decodes 14/20 images in `data/qr/data3` and 41/68 in `data/qr/data4`, at their original resolutions. `data/qr/data3` checks exact payloads against its hash-verified manifest; `data/qr/data4` uses the decoding-success criterion above. The [retained four-set comparison](../test/benchmark_results/scanner_improvements.json) records three interleaved runs of the baseline and improved C++ executables, including gains, losses and latency tails. Additional recovery attempts can increase P95 on difficult inputs.

The Python port was checked on all 392 real images: 166/169, 134/135, 14/20 and 41/68 respectively, with no execution errors. Some multiple-QR images in `data/qr/data2` return a different symbol through the Python ZXing build. [Per-image Python results and source hashes](../test/benchmark_results/python_scanner_results.json). README timing tables use C++ for CSQR-D; they do not compare Python pipeline speed.
