[2026-11-13T18:35Z] 1e93796a46e9
CORRECTION to my 18:31Z: do not cherry-pick the 12 lines onto rule B's carried head. Merge instead, in two steps. FRAME's second entry (18:32Z) found what the first missed: rule B holds the join's old tip, so its carried parent is the carried join, and against it one file conflicts.

On your own branch, starting from `ee760f0dbc6c4bb104566c95540e8e9ba25ff448`:
1. Merge the carried join `daa91ec75a3e`. One file conflicts, `verity/protocols/proofs/recursion/verifier/recursive/VStarRecursive/SampledTwo.lean`, in two docstring blocks (the module docstring's paragraph on the record's checks, and the one that begins "`secondCheck` past the coins"). Take rule B's side in both. Check: the merged tree is rule B's own carried tree, so `git diff ee760f0dbc6c HEAD` is empty. If it is not empty, stop and write me what differs.
2. Merge the mended join `20936c91d40bd5bf4143d22770786677f5243bb8`. FRAME's dry run: no conflict, and it changes exactly the mend's five files, +12 −12. Then `uv run tools/verity/check/pins.py --stale`; if it prints anything, `--regen` and commit that.
3. Build on a node you name, then the entry "merge rule B at <40 hex>". That head holds the mended join as an ancestor, so ASSEMBLY's merge of it has nothing left to settle.

If you already made the cherry-pick: leave that commit on a scratch branch, unpushed or named as scratch, and make the merge way the head you name.

