[2026-11-11T05:14Z] 929fde7e1fab
Coordinator to CROSSB, Tue 10 Nov 9:14 PM PST. **#8709's run `f92f` at the pushed head `1862e5084f05` is red on the read guard, not on a test: one commit from you, then one run.** [the run fetched and read by me]

1. **What failed:** `verity-pouw-benchmarks`, 1216 passed, 21 skipped, then: "its tests read 2 files outside its inputs, for example `verity/configuration/__init__.py` (read by `benchmarks/pouw/tests/test_proved_ticks_placed.py`), `verity/protocols/poub/__init__.py` (read by the same test)". The other six suites passed, none from the cache (research 4022, repository 280, verity-datacenter 49, verity-graph 181, verity-netsim-workload 35, verity-network-sim 21).
2. **Why:** the move gave two of the PR's imports a new parent package (`verity.core.wire` → `verity.configuration.wire`; `verity.core.protocols.remote_bandwidth` → `verity.protocols.poub.verifier`), and the scripts map the entries the PR lists but cannot add a parent's `__init__.py` that `main`'s list never needed.
3. **The mend:** one commit on `cursor/net-crossb-8709-6c8b` that adds exactly those two paths to `[tool.verity.tests] inputs` in `benchmarks/pouw/pyproject.toml`, beside the PR's own seven. Nothing else. If the guard at your head names a third file, add it and say so.
4. **Then ONE run** at the mended head, as before (`--changed 5c095bb4ab70 --quick`, `--on auto`), and your entry with the new head. I push it as a fast-forward: the PR's new head is not marked and is in no candidate, so it may take the commit.
5. Same order as my last: the entries and verdict lines you owe first, #8742's mend, then this. If you had already seen `f92f` and done this, send the head.

