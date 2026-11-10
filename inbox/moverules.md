# Inbox: MOVERULES (TOP-397)

Read at each hourly checkpoint: `git fetch -q origin cursor/verity-top-4db4 && git show origin/cursor/verity-top-4db4:inbox/moverules.md`. Items are numbered and only added; each has its time in UTC.

1. (Tue 10 Nov 19:54Z) **Two more test cases for rule 1, the cover, from circuits** (TOP-394, comment of 19:35Z). They are facts about `main` after #8694 lands, in review since 19:34Z; do not wait for it.
   - #8694 makes the staging key's four paths one constant, `KEY_PATHS`, in `benchmarks/served_units`'s `prove_time.sh` (a shell list) and `window.py` (a Python list), with a test `test_key_paths_in_tree` that holds both to the tree. `FILES` in `tests/test_names.py` is the third list. On a moved tree that test is red until the three lists name the new paths.
   - For rule 1: say in your note whether the cover would write each of the three lists on the old one map (`tools/move/layout_one_map.toml`), and if not, which kind the line is. A shell list is a form the table's `cover` rows may not hold; if rule 1 does not reach it, say so, and do not widen the rule to reach it without a row of HANDROWS's lists that shows the same form.
   - Not a rule, only so you do not chase it: 13 scripts (8 under `verity/ml/flock/pod/`, 5 under `benchmarks/`) key a cache on `$(cd verity/core/service/flock && find . ...)`. The old one map moves that directory whole (line 240), so the mover rewrites the path. Leave them alone.
