# GROUPS: the map that puts the protocols in their three group directories (from top, Thu 12 Nov 11:10 PM PST)

You are a cloud worker of `top`. Read this whole file, then `git show origin/main:AGENTS.md` (your snapshot can be older
than `main`), `tools/move/README.md` whole, and the docstring of `tools/move/layout.py`.

## Why

Daniel changed the architecture page tonight (10:25 to 10:50 PM PST): the protocols sit in two layers plus a proofs
module, each group a directory of its own. At 11:03 PM PST he made the structure tonight's top priority: "having the
repo in this structure when I wake up if that's possible". The page's text is `inbox/ARCHFILE-page.md` on
`origin/cursor/verity-top-4db4` (section 6 has the tree and its rules).

```
verity/protocols/
  proofs/        flock, spot, recursion: what the other protocols call to prove
  accounting/    pouw, pous, poub, certifier: compute, memory and network
  policy/        inference_only, on accounting/
  compose/       stays where it is
  experimental/  stays where it is
```

## The task

1. Create a new branch `cursor/top-protocol-groups-<4 hex>` from `origin/main` and push only to it. Never push to the
   launcher's branch, to a PR's head or to `main`.
2. Write one layout map, `tools/move/protocol_groups_map.toml`, whose `[paths]` move today's eight flat directories:
   `verity/protocols/flock/`, `spot/`, `recursion/` under `verity/protocols/proofs/`; `pouw/`, `pous/`, `poub/`,
   `certifier/` under `verity/protocols/accounting/`; `inference_only/` under `verity/protocols/policy/`. Read two maps
   on `main` that move Lake packages (`tools/move/compose_map.toml`, `tools/move/world_map.toml`) for the tables such a
   map needs (`[roots]`, `[modules]`, `[split]`, `[rewrite]`). Python module names follow the paths
   (`verity.protocols.pouw` becomes `verity.protocols.accounting.pouw`). Lean module and area names do not change; only
   the packages' directories do. `verity/protocols/interface/`, `compose/`, `experimental/`, `remote/`, `onsite/`,
   `tests/` and the loose files stay.
3. Say what the mover does not rewrite and a hand commit must: each group's `__init__.py` if imports need one, Lake
   `path =` requires between packages, `lean-audit.json` paths, workspace members and suite inputs in the pyprojects,
   entries of `tools/verity/check/*.toml`, `tests/import_order.toml`, `tests/test_boundaries.py`'s tables,
   `tests/tree.toml`. Write those hand commits on the staged branch, each small and named.
4. Stage it as a rehearsal on `origin/main` with `tools/move/stage.py` (one commit per map, certified by
   `move_check.py`, rerun to check it changes nothing). Run the staging on a node, never on your VM:
   `research run --on auto --cwd clone -- ...`, as the writing-runs skill says. `python3 tools/move/layout.py --map M
   --dry-run` is light enough to run locally first. Push the rehearsal as `cursor/top-protocol-groups-<hex>-rehearsal`.
5. On the rehearsal's tree, list every import that breaks the page's new rule, with file and line, and fix none:
   - a module under `proofs/` that imports from `accounting/`, `policy/` or `compose/`;
   - a module under `accounting/` that imports from `policy/` or `compose/`;
   - a module under `policy/` that imports from `compose/`.
   The same for Lean: a Lake package under `proofs/` that requires one under `accounting/` or `policy/`, and so on.
6. Do not open a pull request. top's chain builder puts your map and hand commits into tonight's second chain, on the
   tree the first chain leaves. So the map must not depend on anything `main` holds only today; say in your report which
   rows would change if `verity/protocols/composition/`, `remote/` or `onsite/` had already moved.
7. `experimental/`: if `git show origin/cursor/verity-top-4db4:inbox/GROUPS.md` gains a section naming Dither's and the
   FP4 variant's directories, add their rows to `verity/protocols/experimental/<name>/` as a second map,
   `tools/move/experimental_map.toml`. Until it does, leave them out.

## Hours

- The map and its hand-commit list pushed by **Fri 13 Nov 1:00 AM PST**.
- The rehearsal branch and the report by **2:00 AM PST**. If the staging run is still going then, end your turn naming
  its run id and its expected end; top resumes you.

## Rules

- No tool call blocks more than 20 minutes. Builds, suites and staging run through `research run`, never locally.
- Read this file again at each hourly checkpoint
  (`git fetch -q origin cursor/verity-top-4db4 && git show origin/cursor/verity-top-4db4:inbox/GROUPS.md`).
- Push after every commit. Write a one-line checkpoint in a note under `~/.research/notes/lanes/top/` at least every
  60 minutes and `research notes sync --path lanes/top`.
- Do not post on Slack or Linear. Do not mark anything ready. Never print a secret.

## Your final message, to top

The branch names and shas; the map's rows; the hand commits, each in one line; files moved and references rewritten;
every file `move_check.py` refused and why; which `lean-audit.json` files and Lake requires the move touches; the list
of rule-breaking imports (count per pair of groups, then the lines); how many open pull requests touch a moved path
(`gh pr list --state open --json number,files` is enough for a count); and anything you were unsure of.
