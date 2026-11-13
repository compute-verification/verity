# INVENTORY: every line between `main` and the page's tree, and top's remaining maps (from top, Thu 12 Nov 11:10 PM PST)

You are a cloud worker of `top`. Read this whole file, then `git show origin/main:AGENTS.md` (your snapshot can be older
than `main`), `tools/move/README.md` whole, `tests/tree.toml` whole and `tests/test_tree.py`.

## Why

At 11:03 PM PST Daniel made the repository's structure tonight's top priority: "having the repo in this structure when
I wake up if that's possible". The structure is section 6 of his architecture page, whose text is
`inbox/ARCHFILE-page.md` on `origin/cursor/verity-top-4db4`. `tests/tree.toml`'s `[[not_yet]]` list is what `main` holds
that the tree does not. top needs, tonight, one table that says for every line how it gets there and when, and needs
the path-only lines that have no map written.

Tonight has two chains of moves. **Chain 1** is rehearsed (`origin/cursor/top-moves-chain-rehearsal-e0f3`) and is being
rebuilt on `main` by another worker. **Chain 2** is everything else that only moves paths, staged on chain 1's tree by
5 AM PST: the maps on `origin/cursor/top-moves-a-41d7-pr2b` and `origin/cursor/top-moves-c-b396` (top's), lean's on
`origin/cursor/lean-period-maps-fc8d`, network-accounting's on `origin/cursor/na-nci-rho-map-8d6e`, the Lake-package
maps on `main` (`tools/move/compose_map.toml`, `world_map.toml`, `layout_one_map.toml`), and a new map another worker
(GROUPS) writes that puts the protocols under `verity/protocols/proofs/`, `accounting/` and `policy/`.

## The task

1. Create a new branch `cursor/top-tree-inventory-<4 hex>` from `origin/main` and push only to it. Never push to the
   launcher's branch, to a PR's head or to `main`.
2. **The table, first, by Fri 13 Nov 12:30 AM PST.** For every `[[not_yet]]` line of `tests/tree.toml` on
   `origin/main`, and every `[[tree]]` entry that is still to come (it has an owner and a day), one row: the path, where
   it goes, its owner, and one of:
   - `chain 1`: the rehearsal branch already moves it (check the branch's tree);
   - `chain 2, map M on branch B`: a map on one of the branches above, or on `main`, moves it. Name the map and its row;
   - `path-only, no map`: nothing moves it yet, and moving it changes no code beyond references;
   - `code change`: it cannot be a pure move (a fold into one file, a split, a rewrite). Say in a few words why.
   For a `code change` row whose owner is `top` (the fold of `verity/periods/` into `verity/protocol.py` is one), add
   the brief a worker would need to do it tonight: the files, what goes where, which tests hold it.
   Also list what the page's tree names that neither list of `tests/tree.toml` knows, and any `[[open]]` question that
   blocks a row. Write the table as a note, `~/.research/notes/lanes/top/<YYYYMMDDTHHMMZ>-report-tree-inventory.md`
   (front-matter as the other notes there have), `research notes sync --path lanes/top`, and put the same table in your
   final message.
3. **Then the maps for `path-only, no map` rows whose owner is `top`.** One map per independent move,
   `tools/move/<name>_map.toml`, on your branch. `tools/move/layout_one_map.toml` is known to be stale (its `[split]`
   rows): regenerate it with `tools/move/layout_maps.py` from `layout_maps.json` and say what changed, and whether
   anything in it is still to move after chain 1 and the other maps of chain 2. Do not write a map for a row another
   lead owns: list it for top instead, with the map you would write in three lines.
4. Rehearse each of your maps on `origin/cursor/top-moves-chain-rehearsal-e0f3` with `tools/move/stage.py`
   (`--base`), on a node through `research run --on auto --cwd clone -- ...` as the writing-runs skill says, never on
   your VM. Push each rehearsal as `cursor/top-tree-inventory-<hex>-rehearsal-<name>`.
5. Do not open a pull request. top's chain builder takes your maps into chain 2.

## Hours

- The table by **12:30 AM PST**. If it is done sooner, end nothing: go on to the maps, and top reads the note.
- Your maps pushed by **2:00 AM PST**, their rehearsals reported by **3:00 AM PST**. If a staging run is still going
  then, end your turn naming its run id and its expected end; top resumes you.

## Rules

- No tool call blocks more than 20 minutes. Builds, suites and staging run through `research run`, never locally.
- Read this file again at each hourly checkpoint
  (`git fetch -q origin cursor/verity-top-4db4 && git show origin/cursor/verity-top-4db4:inbox/INVENTORY.md`).
- Push after every commit. One-line checkpoint in your note at least every 60 minutes, synced.
- Do not post on Slack or Linear. Do not mark anything ready. Never print a secret.

## Your final message, to top

The table; the note's id; your branch names and shas; each map with its rows and its rehearsal's result (files moved,
references rewritten, files `move_check.py` refused and why); the rows you left for other leads; anything unsure.
