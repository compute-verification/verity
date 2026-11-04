
## 23:17Z, proofs coordinator: pins.py --merged runs Lean builds locally

`pins.py --merged` runs pinned fixtures' generators, which `lake build` on your VM, and nothing refuses it yet (ci CI-383,
hook refusal Wed). Until then, run it only through `research run --on auto`, or recompute pins by hashing.

## 00:53Z, proofs coordinator: check shape.inline before every mark (until LEAN-172)

lean found that something run AFTER a merge (pins.py --merged, check.py --update, or committing a records run's lock)
can drop `shape.inline` entries from a lean-audit.json: #6676 lost 22 that way, and #6075's committed record may have too.
Until lean's fix (LEAN-172) lands: before marking, diff each lock's `shape.inline` against the merge driver's output for
your merge (`python3 tools/verity/lean/merge.py`) and against next's. Restore any entry whose premise the merged tree still
takes inline, as a normal commit, and say so in the PR body.
