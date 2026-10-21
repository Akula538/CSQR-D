"""Compare scanners on frozen single-QR crops of real fold photographs.

    python test/fold_test.py
    python test/fold_test.py --baseline path/to/without_patterns.exe --repeat 5 --details path/to/results.json

Only the standard library is required. The node filter is a property of the
selected binaries; this runner does not modify or rebuild the scanner.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_cases(manifest_path):
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('version') != 1 or not manifest.get('cases'):
        raise ValueError('Expected a nonempty version-1 manifest')
    cases = manifest['cases']
    source_dir = manifest_path.parent / manifest.get('source_directory', '.')
    if len(cases) != manifest['samples'] or len({r['id'] for r in cases}) != len(cases):
        raise ValueError('Invalid case count or duplicate IDs')
    checked_sources = set()
    for r in cases:
        if not isinstance(r.get('expected'), str) or not r['expected']:
            raise ValueError(f'Missing expected payload: {r["id"]}')
        p = (manifest_path.parent/r['file']).resolve()
        if not p.is_relative_to(manifest_path.parent.resolve()):
            raise ValueError('Fixture path outside dataset')
        if sha(p) != r['sha256']:
            raise ValueError(f'Frozen fixture changed: {p}')
        if r.get('source_sha256') and r['source'] not in checked_sources:
            source = source_dir/r['source']
            if source.exists() and sha(source) != r['source_sha256']:
                raise ValueError(f'Original photograph changed: {source}')
            checked_sources.add(r['source'])
    return cases


def scan(exe, path, expected, timeout):
    start = time.perf_counter_ns()
    output, stderr, code, error = '', '', None, None
    try:
        process = subprocess.run([str(exe), str(path), '--seed', '1'], cwd=ROOT,
                                 capture_output=True, timeout=timeout)
        output = process.stdout.decode('utf-8', errors='replace').strip()
        stderr = process.stderr.decode('utf-8', errors='replace').strip()
        code = process.returncode
        if code:
            error = f'exit {code}'
    except subprocess.TimeoutExpired:
        error = 'timeout'
    elapsed = (time.perf_counter_ns()-start)/1e6
    return dict(output=output, stderr=stderr, returncode=code, error=error,
                success=error is None and output == expected, time_ms=elapsed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=ROOT/'data/qr/data3/manifest.json')
    parser.add_argument('--exe', type=Path, default=ROOT/'src/cpp/build/bin/qrscanner_cli.exe')
    parser.add_argument('--baseline', type=Path, help='Optional comparison executable')
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--timeout', type=float, default=30)
    parser.add_argument('--details', type=Path)
    args = parser.parse_args()
    if args.repeat < 1 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error('repeat and timeout must be positive')
    manifest_path = args.manifest.resolve()
    cases = load_cases(manifest_path)
    executables = {'current':args.exe.resolve()}
    if args.baseline:
        executables['baseline'] = args.baseline.resolve()
    hashes = {n:sha(p) for n,p in executables.items()}
    manifest_hash = sha(manifest_path)
    records = []
    for repeat in range(args.repeat):
        for index, case in enumerate(cases):
            names = list(executables)
            if (repeat+index) % 2:
                names.reverse()
            results = {name:scan(executables[name], manifest_path.parent/case['file'],
                                 case['expected'], args.timeout) for name in names}
            records.append(dict(id=case['id'], source=case['source'], ec=case['ec'],
                                repeat=repeat, expected=case['expected'],
                                sha256=case['sha256'], results=results))
        if args.repeat > 1:
            print(f'Completed repeat {repeat+1}/{args.repeat}', file=sys.stderr, flush=True)
    if hashes != {n:sha(p) for n,p in executables.items()} or sha(manifest_path) != manifest_hash:
        raise RuntimeError('Scanner or manifest changed during measurement')
    load_cases(manifest_path)
    summary = {}
    for name in executables:
        values = [r['results'][name] for r in records]
        times = sorted(v['time_ms'] for v in values)
        successful = [r['id'] for r in records if r['results'][name]['success']]
        varying = [c['id'] for c in cases if len({
            (r['results'][name]['returncode'],r['results'][name]['output'])
            for r in records if r['id']==c['id']}) > 1]
        summary[name] = dict(samples=len(cases), invocations=len(values),
                             decoded=sum(v['success'] for v in values),
                             score_percent=100*len(successful)/len(values),
                             decoded_ids=sorted(set(successful)), varying_ids=varying,
                             mean_ms=statistics.mean(times), median_ms=statistics.median(times),
                             p95_ms=times[math.ceil(.95*len(times))-1],
                             errors=sum(v['error'] is not None for v in values),
                             wrong_payloads=sum(v['error'] is None and v['output'] not in
                                                ['', 'None', 'CTF', r['expected']]
                                                for r,v in zip(records,values)))
    report = dict(suite=manifest_path.parent.name, samples=len(cases), repeats=args.repeat,
                  seed=1, measurement='CLI wall time including startup, imread, decoding, shutdown; no external resize',
                  manifest_sha256=manifest_hash,
                  binaries={n:dict(path=str(p),sha256=hashes[n]) for n,p in executables.items()},
                  summary=summary, records=records)
    if 'baseline' in executables:
        first = [r for r in records if r['repeat']==0]
        report['gains'] = [r['id'] for r in first if r['results']['current']['success'] and not r['results']['baseline']['success']]
        report['losses'] = [r['id'] for r in first if r['results']['baseline']['success'] and not r['results']['current']['success']]
        report['mean_time_delta_percent'] = 100*(summary['current']['mean_ms']/summary['baseline']['mean_ms']-1)
    if args.details:
        args.details.parent.mkdir(parents=True, exist_ok=True)
        args.details.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n',encoding='utf-8')
    if args.baseline:
        for name,s in summary.items():
            print(f'{name}: {s["score_percent"]:.2f}% ({len(s["decoded_ids"])}/{len(cases)} unique; {s["mean_ms"]:.2f} ms mean)')
    else:
        print(f'{summary["current"]["score_percent"]:.2f}')


if __name__ == '__main__':
    main()
