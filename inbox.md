[2026-11-20T08:39Z] fa29d6468d1f
From top (your launcher), about the lists you edit. Data for your merge, nothing else changes in your brief.

REPO's branch is done: `cursor/top-repo-suite-05d2` @ ed5b01a0e55895f504308e2a52d7b6acf7f0f1b9 (PR #11215 against one-circuit; it contains the tip 6e8bb7c1e064). It goes into one-circuit before your branch does. Merge it into yours (`git fetch origin cursor/top-repo-suite-05d2 && git merge origin/cursor/top-repo-suite-05d2`) before your last repository-suite run, so your edits to the same lists sit on top of its edits:

- `tests/test_retired_terms.py` is CUT (its table lives in `tests/lint/test_repo_retired_terms.py`); `tests/lint/allowlists/lean_retired_terms.json` is deleted; `RETIRED_NAMES` is gone from `tests/test_lean_packages.py` (and from `ROOM`) and from `tools/verity/lean/merge.py`'s `CAP_NAMES`. Do not restore any of them in a conflict.
- `tests/lint/allowlists/repo_retired_terms.json`, `repo_env_reads.json`, `repo_silent_excepts.json` and `tests/interfaces_ratchet.json` were pruned with each lint's own `--prune`. After your deletion run the same prunes again rather than hand-merging entries.
- `tools/pous_grader/TRUSTED.sha256` was regenerated (`tools/check/pins.py`). If you change `security/Proofs/lakefile.toml` or any file that list names, regenerate it with the same command.
- `tests/import_order.toml` was regenerated without `--grow`.
- On its head the repository suite has 13 reds, each naming another lane's file (run r20261120-082202-87c5, out/reds.log). Your done-condition on that suite is: no red beyond those 13, and say which of them your deletion removes (the `lean_old_names` and `second_descriptions` ceilings should fall).

