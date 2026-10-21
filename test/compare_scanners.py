"""Compare two C++ CLIs on all four real datasets without changing fixtures.

data/qr/data1 uses the existing README 2000px cache; data/qr/data2 through data4 use original
images. data/qr/data3 checks its independently annotated payloads. Other sets retain
the README nonempty-payload criterion and separately report disagreements.
Times include process startup, image loading and shutdown; runs are interleaved.
"""
import argparse
import hashlib
import json
import statistics
import subprocess
import time
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_inputs():
    sys.path.insert(0, str(ROOT / "test/deps"))
    import PIL
    from PIL import Image
    entries = json.loads((ROOT / "test/benchmark_results/manifest.json").read_text())
    folder = ROOT / "test/benchmark_results/inputs_2000"
    folder.mkdir(parents=True, exist_ok=True)
    for row in entries:
        source = ROOT / row["source"]
        if sha(source) != row["sha256"]:
            raise ValueError(f"Source differs from README manifest: {source}")
        with Image.open(source) as image:
            image.convert("RGB").resize(tuple(row["size_2000"]), Image.Resampling.LANCZOS).save(folder / source.name)
    (folder / "resizer.json").write_text(json.dumps(
        dict(pillow=PIL.__version__, method="RGB, LANCZOS; dimensions from README manifest")))


def cases():
    labels = json.loads((ROOT / "data/qr/data3/manifest.json").read_text(encoding="utf-8"))
    expected = {r["file"]: r for r in labels["cases"]}
    originals = {r["source"]: r for r in json.loads(
        (ROOT / "test/benchmark_results/manifest.json").read_text())}
    for dataset in ("data/qr/data1", "data/qr/data2", "data/qr/data3", "data/qr/data4"):
        for source in sorted((ROOT / dataset).rglob("*")):
            if source.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                continue
            path = source
            label = None
            if dataset == "data/qr/data1":
                path = ROOT / "test/benchmark_results/inputs_2000" / source.name
                if sha(source) != originals[source.relative_to(ROOT).as_posix()]["sha256"]:
                    raise ValueError(f"Source differs from README manifest: {source}")
                if not path.exists():
                    raise ValueError("Missing resized input cache; run with --prepare-inputs")
            elif dataset == "data/qr/data3":
                row = expected[source.relative_to(ROOT / dataset).as_posix()]
                if sha(source) != row["sha256"]:
                    raise ValueError(f"Changed fixture: {source}")
                label = row["expected"]
            yield dict(dataset=dataset, source=source.relative_to(ROOT).as_posix(),
                       input=path.relative_to(ROOT).as_posix(),
                       sha256=sha(path), source_sha256=sha(source), expected=label)


def scan(exe, path, expected):
    start = time.perf_counter_ns()
    try:
        proc = subprocess.run([str(exe), str(path), "--seed", "1"], cwd=ROOT,
                              capture_output=True, timeout=30)
        output = proc.stdout.decode("utf-8", errors="replace").strip()
        error = proc.stderr.decode("utf-8", errors="replace") if proc.returncode else None
        success = proc.returncode == 0 and (
            output == expected if expected is not None else output not in {"", "None", "CTF"})
        return dict(output=output, success=success, returncode=proc.returncode,
                    error=error, time_ms=(time.perf_counter_ns() - start) / 1e6)
    except subprocess.TimeoutExpired:
        return dict(output="", success=False, returncode=None, error="timeout",
                    time_ms=(time.perf_counter_ns() - start) / 1e6)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--current", type=Path)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--details", type=Path, required=True)
    parser.add_argument("--prepare-inputs", action="store_true",
                        help="Build the 2000px cache with Pillow before timing")
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be positive")
    if args.prepare_inputs:
        prepare_inputs()
    executables = {"baseline": args.baseline.resolve()}
    if args.current:
        executables["current"] = args.current.resolve()
    hashes = {name: sha(path) for name, path in executables.items()}
    inputs = list(cases())
    rows = []
    for exe in executables.values():
        scan(exe, ROOT / inputs[0]["input"], inputs[0]["expected"])
    for repeat in range(args.repeat):
        for i, case in enumerate(inputs):
            names = list(executables)
            if (i + repeat) % 2:
                names.reverse()
            results = {name: scan(executables[name], ROOT / case["input"], case["expected"])
                       for name in names}
            rows.append(dict(case, repeat=repeat, results=results))
            if (i + 1) % 50 == 0:
                print(f"Repeat {repeat + 1}: {i + 1}/{len(inputs)}", flush=True)
    summary = {}
    for dataset in ("data/qr/data1", "data/qr/data2", "data/qr/data3", "data/qr/data4"):
        subset = [r for r in rows if r["dataset"] == dataset]
        summary[dataset] = {}
        for name in executables:
            values = [r["results"][name] for r in subset]
            times = sorted(v["time_ms"] for v in values)
            rank = (len(times) - 1) * .95
            index = int(rank)
            p95 = times[index] + (rank - index) * (times[min(index + 1, len(times) - 1)] - times[index])
            summary[dataset][name] = dict(
                images=len({r["source"] for r in subset}),
                invocations=len(values), decoded=sum(v["success"] for v in values),
                unique_decoded=len({r["source"] for r in subset if r["results"][name]["success"]}),
                mean_ms=statistics.mean(times), median_ms=statistics.median(times),
                p95_ms=p95,
                errors=sum(v["returncode"] != 0 for v in values),
                varying_results=sorted({r["source"] for r in subset
                    if len({(s["results"][name]["returncode"], s["results"][name]["output"])
                            for s in subset if s["source"] == r["source"]}) > 1}))
        if args.current:
            summary[dataset]["gains"] = sorted({r["source"] for r in subset
                if r["results"]["current"]["success"] and not r["results"]["baseline"]["success"]})
            summary[dataset]["losses"] = sorted({r["source"] for r in subset
                if r["results"]["baseline"]["success"] and not r["results"]["current"]["success"]})
            summary[dataset]["disagreements"] = sorted({r["source"] for r in subset
                if all(v["success"] for v in r["results"].values())
                and r["results"]["baseline"]["output"] != r["results"]["current"]["output"]})
    if hashes != {name: sha(path) for name, path in executables.items()}:
        raise RuntimeError("Executable changed during measurement")
    if inputs != list(cases()):
        raise RuntimeError("Input changed during measurement")
    resize_metadata = ROOT / "test/benchmark_results/inputs_2000/resizer.json"
    report = dict(measurement="CLI wall time; data/qr/data1 at 2000px, other sets original",
                  seed=1, repeats=args.repeat, environment=dict(platform=platform.platform(),
                      python=sys.version, processor=platform.processor()),
                  resizer=json.loads(resize_metadata.read_text()) if resize_metadata.exists() else None,
                  binaries={n: dict(path=str(p), sha256=hashes[n]) for n, p in executables.items()},
                  summary=summary, records=rows)
    args.details.parent.mkdir(parents=True, exist_ok=True)
    args.details.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
