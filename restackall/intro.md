# Restack forecast: every open pull request across the staged layout

@LEAD@

## What was measured

- The heads: the 299 open pull requests against `main` at 23:53Z Tue 10 Nov (3:53 PM PST), each at the head commit it
  had then. Owner is the `Lane NAME` line of the pull request's body, else `unknown`.
- The moved tree: staging A, `origin/cursor/top-layout-staged-e15c` at `50f940c4f28531bc33f23bbb73a2970d5820caf1`, on
  `main` `2b869ed35db5d0efdb844a5f684e6fecbe658114`. It holds two moves: `e15c5efc84b9`
  (`Layout-move: tools/move/layout_one_map.toml`, on `f4d2ff2f8fc9`) and `6751480ae211`
  (`Layout-move: tools/move/security_dir_map.toml`, on `524192c487a4`).
- Per head, on a node, in a `research run --cwd clone` clone (the seven merge drivers registered): `git merge` of
  `main` as a tree merge, then `main`'s `tools/move/restack.py` (blob `bff0d7729cef`) as
  `restack.py MERGED --main 50f940c4f285 --kept --dry-run --json --no-pins`. Nothing was pushed.
- The classes:
  - **clean**: `git merge main` and every step of `restack.py` end without a conflict, and would without the merge
    drivers too.
  - **clean with the drivers only**: clean, but one of the merges conflicts in a clone that has no merge driver
    registered. The list below says which merge and which files.
  - **stops at `git merge main`**: the branch conflicts with `main` as it is today, before any move. `restack.py`
    was not run on it.
  - **stops at the layout**: `restack.py` stops. Its step is named: `base` (the merge of a move's base), `script`
    (the base's `layout.py` on the branch) or `moved` (the merge of the moved tree).
- A restack ends at its first stop, so a head's files are those of its first stop only. A later step may stop it too.

## What was not measured

- The pins step of `restack.py` (`--no-pins`): see the first finding. Each result's stale pins are listed with
  `pins.py --stale`, which runs nothing.
- No check or test was run on any restacked result. "Clean" says the merges and the script ended; it does not say
  the result passes `check`.
- The final moved tree. This is staging A; the table is rerun on the final tree's commit when top names it.
- Heads pushed after 23:53Z.

## Findings about `restack.py` on this tree

@FINDINGS@

