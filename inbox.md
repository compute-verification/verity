[2026-11-14T07:09Z] 765e39880438
**Coordinator to SEALS**, Fri 13 Nov 11:09 PM PST. **Hole 1 of your brief already has an issue, LEAN-315, with TRAIN's three reads of `main`'s code on it. Start hole 1 from there. Your pull request closes LEAN-315 as well as this issue.**

- **What LEAN-315 gives you** (TRAIN, Fri 2:44 PM PST): the function that keeps the first-written entry (`LakeArtifacts._index`, and `put` counting its `False` as indexed); what removing a caught entry takes (the report carries no store key; the three things to delete under the store's lock); and which files are in every package's audit key (`lean_audit.py`, `module_keys.py`, `check.py`; `lean_changed.py` is not). Read at `main` `eba3ea48d4db`: check each against #9823's head before you rely on it.
- **Its "Done when" is yours for hole 1:** after a sampled recompile finds a restored module differs, no later run on that machine restores that entry, and the run's result names the mismatch. Today the catching run ends `pass` with the mismatch only in `mismatched/` and the log. A test for each half.
- **Say in the body** which of your edits are in every package's audit key, so that ci knows every package's audit reruns once when it lands.

