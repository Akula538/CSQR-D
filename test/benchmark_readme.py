"""Reproduce the README benchmarks. Run from any working directory.

python test/benchmark_readme.py --scanners CSQR-D
python test/benchmark_readme.py --scanners ZXing-C++ ZBar OpenCV QReader
python test/benchmark_readme.py --summarize
Dependencies can be installed locally into test/deps.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / 'benchmark_results'
EXE = ROOT / 'src/cpp/build/bin/qrscanner_cli.exe'
os.environ['YOLO_CONFIG_DIR'] = str(HERE / 'yolo_config')
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
(HERE / 'yolo_config').mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(HERE / 'deps'))

def images(folder):
    return sorted(p for p in folder.rglob('*') if p.suffix.lower() in {'.png', '.jpg', '.jpeg'})

def prepare():
    import PIL
    from PIL import Image
    previous_path = OUT / 'manifest.json'
    previous = {row['source']: row for row in json.loads(previous_path.read_text(encoding='utf-8'))} if previous_path.exists() else {}
    manifest = []
    for source in images(ROOT / 'data'):
        match = re.fullmatch(r'img\d+_v\d+_([LMQH])_\w+_d([0-3])', source.stem)
        if not match:
            raise ValueError(f'Unrecognized metadata: {source}')
        with Image.open(source) as original:
            original = original.convert('RGB')
            row = dict(source=source.relative_to(ROOT).as_posix(), ec=match[1], deformation=int(match[2]),
                       original_size=list(original.size), sha256=hashlib.sha256(source.read_bytes()).hexdigest())
            old = previous.get(row['source'], {})
            cached = old.get('sha256') == row['sha256'] and all((OUT / f'inputs_{side}' / source.name).exists() for side in (2000, 1000))
            row['resizer_version'] = old.get('resizer_version', PIL.__version__) if cached else PIL.__version__
            for side in (2000, 1000):
                destination = OUT / f'inputs_{side}' / source.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                scale = min(1, side / max(original.size))
                size = tuple(round(v * scale) for v in original.size)
                if not destination.exists() or previous.get(row['source'], {}).get('sha256') != row['sha256']:
                    original.resize(size, Image.Resampling.LANCZOS).save(destination)
                row[f'size_{side}'] = list(size)
                row[f'sha256_{side}'] = hashlib.sha256(destination.read_bytes()).hexdigest()
            manifest.append(row)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    simple_manifest = []
    for source in images(ROOT / 'data2'):
        with Image.open(source) as image:
            simple_manifest.append(dict(source=source.relative_to(ROOT).as_posix(), size=list(image.size),
                                        sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
    (OUT / 'manifest_data2.json').write_text(json.dumps(simple_manifest, indent=2), encoding='utf-8')
    return manifest

def cli(path):
    run = subprocess.run([str(EXE), str(path), '--seed', '1'], cwd=HERE, capture_output=True, timeout=300)
    if run.returncode:
        raise RuntimeError(f'CLI exit {run.returncode}: {run.stderr!r}')
    text = run.stdout.decode('utf-8', errors='replace').strip()
    return [text] if text and text not in {'None', 'CTF'} else []

def reader(name):
    if name == 'CSQR-D':
        return cli
    import cv2
    def load(path):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if image is None:
            raise ValueError(f'Cannot read {path}')
        return image
    if name == 'ZXing-C++':
        import zxingcpp
        return lambda path: [r.text for r in zxingcpp.read_barcodes(load(path), formats=zxingcpp.BarcodeFormat.QRCode) if r.valid]
    if name == 'ZBar':
        from pyzbar.pyzbar import decode, ZBarSymbol
        return lambda path: [r.data.decode('utf-8', errors='replace') for r in decode(load(path), symbols=[ZBarSymbol.QRCODE])]
    if name == 'OpenCV':
        detector = cv2.QRCodeDetector()
        return lambda path: [s for s in detector.detectAndDecodeMulti(load(path))[1] if s]
    if name == 'QReader':
        import torch
        from qreader import QReader
        torch.manual_seed(1)
        model = QReader(model_size='s', min_confidence=0.5, weights_folder=str(HERE / 'weights'))
        return lambda path: [s for s in model.detect_and_decode(image=load(path), is_bgr=True) if s]
    raise ValueError(name)

def benchmark(name, manifest, repeats):
    decode = reader(name)
    stages = ['accuracy', 'runtime'] + (['curved_runtime'] if name == 'CSQR-D' else [])
    for stage in stages:
        log = OUT / f'{stage}_{name}.jsonl'
        existing = [json.loads(line) for line in log.read_text(encoding='utf-8').splitlines()] if log.exists() else []
        done = {r['source'] for r in existing}
        if stage == 'runtime':
            entries = [dict(source=p.relative_to(ROOT).as_posix()) for p in images(ROOT / 'data2')]
            paths = [ROOT / r['source'] for r in entries]
            count = repeats
        else:
            entries = manifest
            side = 2000 if stage == 'accuracy' else 1000
            paths = [OUT / f'inputs_{side}' / Path(r['source']).name for r in entries]
            count = 1
        if len(done) == len(entries):
            print(f'{stage} {name}: already complete', flush=True)
            continue
        # Untimed warm-up; model loading, downloads and resizing are excluded.
        decode(paths[0])
        with log.open('a', encoding='utf-8', buffering=1) as stream:
            for i, (entry, path) in enumerate(zip(entries, paths), 1):
                if entry['source'] in done:
                    continue
                times, outputs = [], []
                for _ in range(count):
                    start = time.perf_counter_ns()
                    payloads = decode(path)
                    times.append((time.perf_counter_ns() - start) / 1e6)
                    outputs.append(payloads)
                record = dict(entry, scanner=name, stage=stage, times_ms=times, decoded=outputs,
                              success=bool(outputs[0]))
                stream.write(json.dumps(record, ensure_ascii=False) + '\n')
                print(f'{stage} {name} {i}/{len(entries)}: {bool(outputs[0])}, {times[0]:.2f} ms', flush=True)

def summarize():
    import numpy as np
    summary = dict(accuracy={}, runtime={}, curved_runtime={})
    names = ['ZXing-C++', 'ZBar', 'OpenCV', 'QReader', 'CSQR-D']
    for name in names:
        for stage in summary:
            log = OUT / f'{stage}_{name}.jsonl'
            if not log.exists():
                continue
            rows = [json.loads(line) for line in log.read_text(encoding='utf-8').splitlines()]
            assert len(rows) == len({row['source'] for row in rows}), f'Duplicate measurements in {log}'
            assert all(r['scanner'] == name and r['stage'] == stage for r in rows)
            assert all(r['success'] == bool(r['decoded'][0]) for r in rows)
            if stage == 'accuracy':
                summary[stage][name] = {}
                for deformation in [0, 1, 2, 3, 'Overall']:
                    group = [r for r in rows if deformation == 'Overall' or r['deformation'] == deformation]
                    summary[stage][name][str(deformation)] = dict(samples=len(group), decoded=sum(r['success'] for r in group),
                        rate_percent=100 * sum(r['success'] for r in group) / len(group))
            elif stage == 'runtime':
                values = [t for r in rows for t in r['times_ms']]
                summary[stage][name] = dict(samples=len(rows), measurements=len(values), decoded=sum(r['success'] for r in rows),
                    mean_ms=float(np.mean(values)), median_ms=float(np.median(values)), p95_ms=float(np.percentile(values, 95)))
            else:
                for deformation in range(4):
                    summary[stage][str(deformation)] = {}
                    for ec in 'LMQH':
                        values = [r['times_ms'][0] for r in rows if r['deformation'] == deformation and r['ec'] == ec]
                        summary[stage][str(deformation)][ec] = dict(samples=len(values), mean_ms=float(np.mean(values)))
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    return summary

def update_readme(summary):
    # Update results without replacing the editorial text in either README.
    names = ['ZXing-C++', 'ZBar', 'OpenCV', 'QReader', 'CSQR-D']
    for name in names:
        assert summary['accuracy'][name]['Overall']['samples'] == 169
        assert summary['runtime'][name]['samples'] == 135
        assert summary['runtime'][name]['measurements'] == 405
    assert sum(cell['samples'] for row in summary['curved_runtime'].values() for cell in row.values()) == 169
    overall = summary['accuracy']['CSQR-D']['Overall']
    for filename, russian in [('README.md', False), ('README.ru.md', True)]:
        readme = ROOT / filename
        text = readme.read_text(encoding='utf-8')
        def number(value):
            formatted = f"{value:.2f}"
            return formatted.replace('.', ',') if russian else formatted
        accuracy_rows = []
        for level in ['0', '1', '2', '3', 'Overall']:
            label = ('**Всего**' if russian else '**Overall**') if level == 'Overall' else level
            count = summary['accuracy']['CSQR-D'][level]['samples']
            accuracy_rows.append('| ' + ' | '.join([label, str(count)] +
                [number(summary['accuracy'][name][level]['rate_percent']) + '%' for name in names]) + ' |')
        runtime_rows = []
        for name in names:
            stats = summary['runtime'][name]
            runtime_rows.append('| ' + ' | '.join(['ZBar / pyzbar' if name == 'ZBar' else name] +
                [number(stats[key]) for key in ['mean_ms', 'median_ms', 'p95_ms']]) + ' |')
        curved_rows = ['| ' + ' | '.join([str(level)] +
            [number(summary['curved_runtime'][str(level)][ec]['mean_ms']) for ec in 'LMQH']) + ' |'
            for level in range(4)]
        tables = iter([accuracy_rows, runtime_rows, curved_rows])
        def replace_table(match):
            headers = match.group().splitlines()[:2]
            return '\n'.join(headers + next(tables)) + '\n'
        text, count = re.subn(r'(?m)^\|[^\n]*\n(?:\|[^\n]*\n?)+', replace_table, text)
        assert count == 3, f'Expected three benchmark tables in {filename}'
        result = (f"**{overall['decoded']} из {overall['samples']} изображений ({number(overall['rate_percent'])}%)**"
                  if russian else f"**{overall['decoded']} of {overall['samples']} images ({number(overall['rate_percent'])}%)**")
        pattern = r'\*\*\d+ из \d+ изображений \([\d,]+%\)\*\*' if russian else r'\*\*\d+ of \d+ images \([\d.]+%\)\*\*'
        text, count = re.subn(pattern, lambda _: result, text)
        assert count == 1, f'Expected one overall result in {filename}'
        readme.write_text(text, encoding='utf-8')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scanners', nargs='+', default=['ZXing-C++', 'ZBar', 'OpenCV', 'QReader', 'CSQR-D'])
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--summarize', action='store_true')
    parser.add_argument('--update-readme', action='store_true')
    args = parser.parse_args()
    if args.summarize:
        summary = summarize()
        if args.update_readme:
            update_readme(summary)
        return
    manifest = prepare()
    import winreg
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
        cpu = winreg.QueryValueEx(key, 'ProcessorNameString')[0].strip()
    packages = {}
    for package in ['numpy', 'Pillow', 'opencv-python', 'zxing-cpp', 'pyzbar', 'qreader', 'qrdet', 'torch', 'ultralytics']:
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    metadata = dict(platform=platform.platform(), cpu=cpu, python=sys.version, packages=packages,
                    input_resizer_version=', '.join(sorted({r['resizer_version'] for r in manifest})),
                    qreader_weights_sha256=hashlib.sha256((HERE / 'weights/qrdet-s.pt').read_bytes()).hexdigest() if (HERE / 'weights/qrdet-s.pt').exists() else None,
                    cli_sha256=hashlib.sha256(EXE.read_bytes()).hexdigest(), seed=1, runtime_repeats=args.repeats,
                    resize='Pillow LANCZOS, preserve aspect ratio, before timing',
                    timer='perf_counter_ns; image loading included; CSQR-D process startup included',
                    qreader='small model, confidence 0.5, CPU; initialization and warm-up excluded')
    (OUT / 'environment.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    for name in args.scanners:
        benchmark(name, manifest, args.repeats)

if __name__ == '__main__':
    main()
