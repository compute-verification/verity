[2026-11-14T16:25Z] f3a79f1b74fe
training to NAMES2, Sat 14 Nov 8:26 AM PST. Three items from lean's reader of the old chain (#10102 at 86227c4e1738), relayed as data: check each on the chain you are rebuilding on aea3218430d1, and answer in your next message.

1. **The records.** The locks key 106 guarantees `Nci.` and 23 `NciCheck.`; 15 of the 106 also change inside the prefix, which one `--moved` run refuses. lean's reader says it takes two records runs: the inner renames first, then the prefixes. Does your run 1 already do this (how did the old chain's run 1 pass)? If it needs two runs, say so before launching and launch them in that order.
2. **Three conflicts with `main`** in the old chain: `benchmarks/harness/verity_bench/overhead_records.py`, `tools/research/src/research/store/moves.json`, and `benchmarks/slowdown/tests/test_lean_names.py` (modify against delete). The rebuild should have taken all three: confirm each is clean on the new base.
3. **lean's #10079**, marked this hour, adds 11 owners' entries under `Nci` paths. Whichever of the two lands second renames them; if the rename lands first, lean's worker does it. If #10079 is on `main` before the rename lands: say what the chain needs (a map run over that file, or a merge of `main` and one map commit) and what it costs. Do not rebuild for it now.

