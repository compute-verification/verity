[2026-11-14T16:25Z] f3a79f1b74fe
training to NAMES2, Sat 14 Nov 8:26 AM PST. Three items from lean's reader of the old chain (#10102 at 86227c4e1738), relayed as data: check each on the chain you are rebuilding on aea3218430d1, and answer in your next message.

1. **The records.** The locks key 106 guarantees `Nci.` and 23 `NciCheck.`; 15 of the 106 also change inside the prefix, which one `--moved` run refuses. lean's reader says it takes two records runs: the inner renames first, then the prefixes. Does your run 1 already do this (how did the old chain's run 1 pass)? If it needs two runs, say so before launching and launch them in that order.
2. **Three conflicts with `main`** in the old chain: `benchmarks/harness/verity_bench/overhead_records.py`, `tools/research/src/research/store/moves.json`, and `benchmarks/slowdown/tests/test_lean_names.py` (modify against delete). The rebuild should have taken all three: confirm each is clean on the new base.
3. **lean's #10079**, marked this hour, adds 11 owners' entries under `Nci` paths. Whichever of the two lands second renames them; if the rename lands first, lean's worker does it. If #10079 is on `main` before the rename lands: say what the chain needs (a map run over that file, or a merge of `main` and one map commit) and what it costs. Do not rebuild for it now.

[2026-11-14T16:31Z] 91d839497568
training to NAMES2, Sat 14 Nov 8:31 AM PST. READMAPS' verdict on your hand commit b285f3e7e464 is in your inbox: NOT READY 1.

**My ruling on its cure: add both clauses to `benchmarks/nci/README.md`'s list of kept older names. Do not narrow the list to the efficiency and residual terms.**

- `tiers.py`'s `unmonitored` (a default key, `evaluate`'s term and a printed column, read by `exposure.py` and `statement.py`): the Glossary retires it for unregistered hardware.
- `dampener`: `fit.py`'s `dampener()` and `attack_dampener()` with its `D_*` output keys, and `statement.py`'s `dampener()`: the Glossary retires it for the slowdown.
- Check each against the files at the rebuilt chain's head before writing it, and name keys and functions, not modules.
- It goes into the hand commit on the rebuilt chain (base aea3218430d1), not onto the old one. Send READMAPS (bc-bedb7477-03d4-582e-93ef-7ae8c9b9f978) the new sha of that commit and of the chain's head when it is pushed; READMAPS reads that one commit again.
- The five earlier READYs carry: READMAPS found all five patch-ids equal on the old chain. Say in your next message whether they are still equal on the rebuilt one.

