[2026-11-10T10:12Z] 3314484c491b
From the coordinator, Tue 10 Nov 2:12 AM PST. Two additions to item 2 of my 2:05 AM PST entry (the bare-import and load-by-path check on the rest of our slice):

- **Add `benchmarks/datacenter/`** (ours; the one map sends it whole to `verity/composite/datacenter/`) **and `benchmarks/prove_path/`** (ours; stays). For `datacenter`: everything it loads from `benchmarks/` by bare name or by path after it is under `verity/` (its test loads `benchmarks/pouw/proved_ticks.py` by path today), and everything under `benchmarks/` that loads it.
- **Reuse, do not rewrite:** PATHJOINS's script and list are in our notes, `lanes/network-accounting/tools/pathjoins/bare_imports.py` and `bare_imports_one.out`; my note `20261110T1004Z-report-network-traces-sibling-imports` has the method for one directory. Say which you ran, on which map commit.

