[2026-11-15T21:14Z] ddd1f27e2843
TRN-158, the interface change: draft #10401, branch cursor/training-inference-only-interface-4207, head d4ced2e5c577. It is a merge of #10391 at bc590e7b6502 and #10378 at 602c0dc0b0ef, plus one commit. It is not marked.

Two things you asked me to tell you:
1. In-process coins are `ran with: not stated`. PoUW's piece states in-process coins only under the retired `coin_source` key. The only reader of that key outside the allowlist is compute's #10389, and it reads the key unallowed, so it is not a lint-clean route. I added no lint exception. The live-challenger case reads `coins.check.session`, which the piece writes only for live coins, and prints `local challenger`. `deployed challenger` is never claimed.
2. The samples are not read. Their key is retired, and the piece's sample count is prose. DETAIL holds the scheme only.

Ratchet gate against main: it fails on 6 retired-term sites, `draws`, all in #10391's own lines: benchmarks/README.md:115 (the interface's DETAIL text), harness/tests/test_interface.py:139-141 and :178, and harness/verity_bench/kit_report.py:67. On my commit alone (535b551d832..HEAD) the gate passes. This is for the benchmarks lane to fix; I did not touch their files.

Suites, run with --fresh and lake: benchmarks/inference_only passes 63, and the harness (verity-bench) passes 753. The root retired-terms, doc-paths and repo retired-terms tests pass 72, with the same `draws` warning on #10391's README line.

The no-record path, compared to the merge base with `commit` normalised:
- with --stated-only, inference_only.json and result.json are the same byte for byte;
- in the full run, only the Lean count's measured cost differs;
- stdout differs only at line 2, the stated-policy line.

