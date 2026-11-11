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
- The runs: second round r20261111-002554-0164, r20261111-002642-ac2f, r20261111-002733-3cb5, r20261111-002825-926f,
  r20261111-002909-46cb, r20261111-002945-e3ce (266 heads and the control); first round, cancelled, whose 33 final
  records are kept (each stopped before the pins step): r20261111-000738-014f, r20261111-000910-9644,
  r20261111-001342-07c6, r20261111-001418-0622, r20261111-001605-2076, r20261111-001843-2eb9; the control with the
  pins step: r20261111-003521-5d6d. Each run's `out/heads/N.json` is one head's record; `out/results-*.json` and
  `out/results-*.bundle` hold each clean head's restacked commit (`refs/restackall/result/N`).
- The driver and this report's script: `restackall/run.py` and `restackall/report.py` on `cursor/top-restackall-7c3e`
  (scratch; it lands nowhere).
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

