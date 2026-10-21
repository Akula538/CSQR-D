"""Check the Python port on the same real inputs as compare_scanners.py."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "test/deps"), str(ROOT / "src/python")]
from PIL import Image
from QRscanner import detectAndDecode
from compare_scanners import cases, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--details", type=Path, default=ROOT / "test/benchmark_results/python_scanner_results.json")
    args = parser.parse_args()
    cpp = json.loads((ROOT / "test/benchmark_results/scanner_improvements.json").read_text())
    reference = {r["source"]: r["results"]["current"] for r in cpp["records"] if r["repeat"] == 0}
    sources = sorted((ROOT / "src/python").glob("*.py"))
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in sources}
    inputs = list(cases())
    records = []
    for i, case in enumerate(inputs, 1):
        start = time.perf_counter()
        captured = io.StringIO()
        output, error = None, None
        try:
            with Image.open(ROOT / case["input"]) as image, contextlib.redirect_stdout(captured):
                output = detectAndDecode(image)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        success = error is None and (output == case["expected"] if case["expected"] is not None
                                    else output not in (None, "", "None", "CTF"))
        records.append(dict(case, output=output, error=error, success=success,
                            diagnostic_stdout=captured.getvalue(),
                            elapsed_ms=(time.perf_counter() - start) * 1000,
                            cpp=reference[case["source"]]))
        if i % 20 == 0 or i == len(inputs):
            print(f"{i}/{len(inputs)}", flush=True)
    if hashes != {str(p.relative_to(ROOT)): sha(p) for p in sources}:
        raise RuntimeError("Python sources changed during validation")
    if inputs != list(cases()):
        raise RuntimeError("Inputs changed during validation")
    summary = {}
    for dataset in ("data/qr/data1", "data/qr/data2", "data/qr/data3", "data/qr/data4"):
        rows = [r for r in records if r["dataset"] == dataset]
        summary[dataset] = dict(samples=len(rows), decoded=sum(r["success"] for r in rows),
            errors=[r["source"] for r in rows if r["error"]],
            cpp_only=[r["source"] for r in rows if r["cpp"]["success"] and not r["success"]],
            python_only=[r["source"] for r in rows if not r["cpp"]["success"] and r["success"]],
            payload_disagreements=[r["source"] for r in rows if r["success"] and r["cpp"]["success"]
                                   and r["output"] != r["cpp"]["output"]])
    args.details.parent.mkdir(parents=True, exist_ok=True)
    args.details.write_text(json.dumps(dict(summary=summary, source_sha256=hashes, records=records),
                                      ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if any(r["error"] for r in records):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
