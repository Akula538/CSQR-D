"""Hash and benchmark every Data Matrix photograph; requires only Python stdlib.

python test/datamatrix_benchmark.py --repeat 3
Scanner time includes preprocessing, localization, recovery and decoding, but
excludes file I/O, JPEG decompression and process startup. Both scanners run in
one process, with alternating order and one initial warmup per scanner.
"""
import argparse
import hashlib
import json
import platform
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentile(values, fraction):
    values = sorted(values)
    index = (len(values) - 1) * fraction
    low = int(index)
    return values[low] + (values[min(low + 1, len(values) - 1)] - values[low]) * (index - low)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=ROOT / 'data/datamatrix/data1')
    parser.add_argument('--exe', type=Path, default=ROOT / 'src/cpp/build/bin/datamatrix_cli.exe')
    parser.add_argument('--repeat', type=int, default=3)
    parser.add_argument('--output', type=Path, default=None)
    parser.add_argument('--thorough', action='store_true', help='Enable the slower extended recovery')
    parser.add_argument('--legacy-source', type=Path,
                        help='Source compiled into the optional three-scanner datamatrix_compare executable')
    parser.add_argument('--labels', type=Path, help='Optional independent JSON mapping of relative file names to expected HRI text')
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error('--repeat must be positive')
    if args.legacy_source and args.thorough:
        parser.error('Historical comparison supports only fast mode')
    scanners = ['ZXing-C++', 'CSQR-D-DataMatrix'] + (['Legacy-DataMatrix'] if args.legacy_source else [])
    if args.output is None:
        filename = 'datamatrix_thorough.json' if args.thorough else 'datamatrix.json'
        args.output = ROOT / 'test/benchmark_results' / filename
    dataset, exe = args.dataset.resolve(), args.exe.resolve()
    files = sorted(p for p in dataset.rglob('*') if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'})
    if not files:
        parser.error('No input images')
    manifest = {p.relative_to(dataset).as_posix(): {'sha256': sha256(p), 'bytes': p.stat().st_size} for p in files}
    labels = json.loads(args.labels.read_text(encoding='utf-8')) if args.labels else None
    if args.labels and (not isinstance(labels, dict) or set(labels) != set(manifest) or
                               any(not isinstance(value, str) for value in labels.values())):
        parser.error('Labels must cover exactly the input images')
    source_paths = ['src/cpp/DMscanner.cpp', 'src/cpp/DMscanner.hpp', 'src/cpp/datamatrix_main.cpp',
                    'src/cpp/find_dm.cpp', 'src/cpp/find_dm.hpp', 'src/cpp/dm_correct.cpp', 'src/cpp/dm_decode.cpp', 'src/cpp/dm_decode.hpp', 'src/cpp/dm_correct.hpp', 'src/cpp/scan_result.hpp', 'src/cpp/QRscanner.hpp',
                    'src/cpp/decoder.cpp', 'src/cpp/decoder.hpp',
                    'src/cpp/binarizer.cpp', 'src/cpp/binarizer.hpp',
                    'src/cpp/vectorfield_correct.cpp', 'src/cpp/CMakeLists.txt',
                    'src/cpp/vectorfield_correct.hpp', 'src/cpp/datamatrix_checks.cpp',
                    'test/datamatrix_benchmark.py']
    source_hashes = {name: sha256(ROOT / name) for name in source_paths}
    if args.legacy_source:
        source_hashes['src/cpp/datamatrix_legacy_compare.cpp'] = sha256(ROOT / 'src/cpp/datamatrix_legacy_compare.cpp')
    legacy_hash = sha256(args.legacy_source) if args.legacy_source else None
    executable_hash = sha256(exe)
    labels_hash = sha256(args.labels) if args.labels else None
    command = [str(exe), str(dataset), '--compare', '--json', '--repeat', str(args.repeat)]
    if args.thorough:
        command.append('--thorough')
    records = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    raw_path = args.output.with_suffix('.jsonl')
    # Save incremental raw records so an interrupted measurement is inspectable.
    with raw_path.open('w', encoding='utf-8') as raw, subprocess.Popen(
            command, stdout=subprocess.PIPE, encoding='utf-8') as process:
        for line in process.stdout:
            record = json.loads(line)
            records.append(record)
            raw.write(json.dumps(record, ensure_ascii=False) + '\n')
            raw.flush()
            if len(records) % (args.repeat * len(scanners) * 10) == 0:
                print(f'{len(records) // (args.repeat * len(scanners))}/{len(files)} images', flush=True)
        if process.wait():
            raise RuntimeError(f'Scanner failed with exit {process.returncode}; raw records: {raw_path}')
    if len(records) != len(files) * args.repeat * len(scanners):
        raise RuntimeError('Incomplete measurement')
    summary = {}
    by_scanner = {}
    for scanner in scanners:
        grouped = {}
        for record in records:
            if record['scanner'] == scanner:
                grouped.setdefault(record['file'], []).append(record)
        if set(grouped) != set(manifest):
            raise RuntimeError(f'{scanner}: unexpected input coverage')
        by_scanner[scanner] = {}
        for name, rows in grouped.items():
            if {row['repeat'] for row in rows} != set(range(args.repeat)) or len(rows) != args.repeat:
                raise RuntimeError(f'{scanner}: incomplete repeats for {name}')
            if len({(row['decoded'], row['text']) for row in rows}) != 1:
                raise RuntimeError(f'{scanner}: nondeterministic decode for {name}')
            manifest[name]['width'], manifest[name]['height'] = rows[0]['width'], rows[0]['height']
            by_scanner[scanner][name] = {
                'decoded': rows[0]['decoded'], 'text': rows[0]['text'],
                'success': rows[0]['decoded'] and (labels is None or rows[0]['text'] == labels[name]),
                'times_ms': [row['ms'] for row in rows],
                'median_ms': statistics.median(row['ms'] for row in rows),
            }
        values = list(by_scanner[scanner].values())
        times = [row['median_ms'] for row in values]
        summary[scanner] = {
            'successes': sum(row['success'] for row in values), 'total': len(values),
            'rate_percent': 100 * sum(row['success'] for row in values) / len(values),
            'mean_ms': statistics.mean(times), 'median_ms': statistics.median(times),
            'p95_ms': percentile(times, .95), 'max_ms': max(times),
        }
    before, after = (by_scanner[name] for name in ['ZXing-C++', 'CSQR-D-DataMatrix'])
    gains = [name for name in manifest if after[name]['success'] and not before[name]['success']]
    losses = [name for name in manifest if before[name]['success'] and not after[name]['success']]
    disagreements = [name for name in manifest if before[name]['decoded'] and after[name]['decoded'] and before[name]['text'] != after[name]['text']]
    if sha256(exe) != executable_hash or any(sha256(ROOT / name) != value for name, value in source_hashes.items()):
        raise RuntimeError('Scanner executable or sources changed during benchmark')
    if args.labels and sha256(args.labels) != labels_hash:
        raise RuntimeError('Labels changed during benchmark')
    if args.legacy_source and sha256(args.legacy_source) != legacy_hash:
        raise RuntimeError('Historical source changed during benchmark')
    for path in files:
        if sha256(path) != manifest[path.relative_to(dataset).as_posix()]['sha256']:
            raise RuntimeError(f'Input changed during benchmark: {path}')
    report = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'metric': 'exact HRI payload match' if labels else 'valid ECC200 decode; no independent ground-truth labels supplied',
        'method': 'original JPEG dimensions, ignore EXIF orientation; one OpenCV thread; alternate scanners; one initial warmup; median of repeats per image; scanner time includes all preprocessing, excludes image loading and process startup',
        'mode': 'thorough' if args.thorough else 'fast',
        'repeat': args.repeat, 'platform': platform.platform(), 'processor': platform.processor(),
        'command': command, 'executable_sha256': executable_hash,
        'sources': source_hashes,
        'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'zxing_head': subprocess.check_output(['git', '-C', 'src/cpp/zxing-cpp_finder/zxing-cpp', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'labels_sha256': labels_hash,
        'legacy_source': {'path': str(args.legacy_source.resolve()), 'sha256': legacy_hash} if args.legacy_source else None,
        'manifest': manifest, 'summary': summary, 'gains': gains, 'losses': losses,
        'payload_disagreements': disagreements, 'results': by_scanner,
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)
    print(f'Gains: {len(gains)}; losses: {len(losses)}; payload disagreements: {len(disagreements)}')
    print(args.output)


if __name__ == '__main__':
    main()
