"""Resize all existing Kodak24 negatives to a controlled input resolution.

This is a resolution stress control, not new photographic detail or a camera
video dataset. Originals are verified against the existing download manifest.
Requires OpenCV (also accepts test/deps). No downloads or case selection.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'test/deps'))
import cv2


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'test/benchmark_results/negative_kodak24')
    parser.add_argument('--output', type=Path, default=ROOT / 'test/benchmark_results/dm_negative_large')
    parser.add_argument('--long-side', type=int, default=4080)
    args = parser.parse_args()
    if args.long_side < 24:
        parser.error('--long-side must be at least 24')
    source, output = args.source.resolve(), args.output.resolve()
    if source == output:
        parser.error('Output must differ from the source directory')
    manifest = json.loads((source / 'manifest.json').read_text(encoding='utf-8'))
    if len(manifest) != 24:
        raise ValueError('Expected the complete 24-image Kodak manifest')
    (output / 'images').mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    files = {}
    for name, original in sorted(manifest.items()):
        path = source / 'images' / name
        if sha(path) != original['sha256']:
            raise ValueError(f'Source hash mismatch: {name}')
        image = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        height, width = image.shape[:2]
        scale = args.long_side / max(width, height)
        size = (round(width * scale), round(height * scale))
        image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
        dest = output / 'images' / name
        if not cv2.imwrite(str(dest), image, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
            raise RuntimeError(f'Cannot write {dest}')
        files[name] = {'sha256': sha(dest), 'source_sha256': original['sha256'],
                       'url': original['url'], 'width': size[0], 'height': size[1]}
    report = {'opencv': cv2.__version__, 'interpolation': 'INTER_LINEAR',
              'long_side': args.long_side, 'generator_sha256': sha(Path(__file__)), 'files': files}
    (output / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(f'Generated {len(files)} resolution controls: {output / "images"}')


if __name__ == '__main__':
    main()
