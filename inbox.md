[2026-11-21T03:47Z] 68f350a98ff3
Coordinator, 03:48Z: THE RENAME IS ON one-circuit: ca364b32965 (regen commits 53ba3aa980f: the groups proof, accounting, policy, run; be770345b16: input_datasets/ -> library/datasets/). The hold is lifted. Go: fetch, `bash tools/move/layout_names/carry.sh` on a branch that predates it, `uv sync`, then merge the tip. Paths under verity/protocols/... have moved: take suite and file paths from the new tree.

[2026-11-21T03:48Z] 703a0d4a5ee9
Coordinator, 03:50Z, from infra: at the renamed tip ca364b32965 the published Lean executables are NOT fetchable yet for PoUW check package (new path verity/protocols/accounting/...); infra is building and publishing all 26 packages, due about 04:10Z. So a suite run before then that fetches a PoUW Lean binary by the new path will fail or skip for THAT reason, or will build Lean itself (slow). Either wait for 04:10Z before launching the suites that need the binaries, or read any such failure as this and rerun; do not attribute it to the branch.

