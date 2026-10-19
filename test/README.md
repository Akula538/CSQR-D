# Tests

- `benchmark_readme.py` reproduces the three README benchmarks using `data` and `data2`. `--scanners CSQR-D --fresh` refreshes only our scanner; `--summarize --update-readme` updates both README tables.
- `benchmark_results/` contains the retained measurements, summary and input hashes. Resized input caches are generated on demand and ignored by Git.
- `fold_test.py` checks exact payloads on the 20 real crops in [data3](../data3/). Its manifest supplies labels and verifies fixture hashes.

Raw photographs (`fold_data/`), diagnostic galleries (`fold_timing/`), `wave_test.py` and its frozen fixtures remain local and are excluded from commits, along with all synthetic datasets. Local dependencies (`deps/`), model weights and configuration are also ignored.
