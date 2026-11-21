[2026-11-21T03:47Z] 08420d945cee
Coordinator, 03:48Z: THE RENAME IS ON one-circuit: ca364b32965 (regen commits 53ba3aa980f: the groups proof, accounting, policy, run; be770345b16: input_datasets/ -> library/datasets/). The hold is lifted. Go: fetch, `bash tools/move/layout_names/carry.sh` on a branch that predates it, `uv sync`, then merge the tip. Paths under verity/protocols/... have moved: take suite and file paths from the new tree.

