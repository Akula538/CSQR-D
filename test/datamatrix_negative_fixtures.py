"""Generate fixed finder-pattern controls with erased or uncoded payloads.

Requires NumPy and OpenCV (also accepts the local test/deps installation).
These are failure controls, not a representative camera dataset. No examples
are selected based on a scanner's results. Benchmark them with
datamatrix_benchmark.py --dataset test/benchmark_results/dm_negative_control/images.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'test/deps'))
import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'test/benchmark_results/dm_negative_control')
    args = parser.parse_args()
    images = args.output / 'images'
    images.mkdir(parents=True, exist_ok=True)
    seed = 20261101
    rng = np.random.default_rng(seed)
    manifest = {}
    for height, width in [(22, 22), (8, 32)]:
        for body in ['erased', 'invalid']:
            for module in [6, 12]:
                for angle in [0, 23]:
                    shape = (height, width)
                    cells = (np.full(shape, 128, np.uint8) if body == 'erased'
                             else rng.integers(0, 2, shape, dtype=np.uint8) * 255)
                    cells[:, 0] = cells[-1, :] = 0
                    cells[0, :] = np.where(np.arange(width) % 2 == 0, 0, 255)
                    cells[:, -1] = np.where(np.arange(height) % 2 == 1, 0, 255)
                    code = cv2.resize(cells, (width * module, height * module),
                                      interpolation=cv2.INTER_NEAREST)
                    image = np.full((768, 1024), 255, np.uint8)
                    y, x = (768 - code.shape[0]) // 2, (1024 - code.shape[1]) // 2
                    image[y:y + code.shape[0], x:x + code.shape[1]] = code
                    if angle:
                        image = cv2.warpAffine(image, cv2.getRotationMatrix2D((512, 384), angle, 1),
                                               (1024, 768), borderValue=255)
                    cv2.putText(image, 'PRODUCT / LOT 12345', (250, 160), cv2.FONT_HERSHEY_SIMPLEX,
                                1.2, 0, 2, cv2.LINE_AA)
                    name = f'{body}_{width}x{height}_m{module}_r{angle}.png'
                    path = images / name
                    if not cv2.imwrite(str(path), image):
                        raise RuntimeError(f'Cannot write {path}')
                    manifest[name] = {
                        'kind': body, 'width_modules': width, 'height_modules': height,
                        'module_pixels': module, 'rotation': angle,
                        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
    report = {'seed': seed, 'numpy': np.__version__, 'opencv': cv2.__version__,
              'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'files': manifest}
    (args.output / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(f'Generated {len(manifest)} fixed controls: {images}')


if __name__ == '__main__':
    main()
