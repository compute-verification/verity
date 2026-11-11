# tools/move

Codemods that move or rename part of the tree and rewrite every reference to it, so a move is one generated commit that
`tools/verity/check/move_check.py` can certify as a pure move. Each is stdlib and git, run as a file from the repository
root, and refuses a dirty worktree; a second run changes nothing.

| Tool | What it does |
|---|---|
| `stage.py` | Stages a move on a base: one commit per map, each certified by `move_check.py` and rerun to check it changes nothing, for its lander (`MAP...`, `--base`, `--branch`, `--push`, `--certificate`). |
| `layout.py` | Moves the paths a layout map names and rewrites every path, module name, import, pyproject and anchor that pointed at them (`--dry-run`, `--modules`, `--resolve M`). |
| `layout_maps.py` | Writes the layout as one map, `layout_one_map.toml` (or, by `--form moves`, as one map a move in the order they run, `layout_NN_NAME_map.toml`), from `layout_maps.json` (each moved component's new directory and files) and the files HEAD tracks; it fails on a tracked file with no destination, so it is the one rerun when `main` moves (`--form`, `--ref`, `--targets`, `--created`). |
| `rename.py` | Renames the words a word map names, in paths, modules, Lean names and prose, leaving the Lean locks' records for `tools/verity/lean/check.py --update --moved` and sorting again a sorted JSON object whose keys it renames (`--map MAP`, `--dry-run`, `--moves`). |
| `restack.py` | Carries an open branch across every move `main` has and it lacks, by rerunning each move's script on it, then merging the moved `main` (`BRANCH`, `--push-to`). |

A layout map (`layout.py --map`) maps old paths to new ones; a word map (`rename.py --map`) maps old words to new ones:

~~~toml
[paths]                                   # layout.py: a file, or a directory ending in "/"
"verity/primitives/circuits/" = "verity/ml/circuits/"

[words]                                   # rename.py: applied in order, as a word or one part of an identifier
"network warden" = "network certifier"
"Warden" = "NetworkCertifier"

[rewrite]
exclude = ["archive/**", "**/CHANGELOG.md"]
~~~

The docstrings give the full format: `[roots]`, `[modules]`, `[split]` and `[rewrite]` for `layout.py`, `[[words_in]]`,
`[keep]`, `[keep_in]` and `scope` for `rename.py`, which refuses a map whose `[[words_in]]` files, `[keep_in]` or
`[keep_paths]` name a path no tracked file matches, since its move would skip what the map says of it. A word map with `scope = "prose"` renames words only where they stand in
prose (comments, docstrings, Markdown's text), so its move moves no path and leaves every lock and pin as it was;
`move_check.py` judges that move by the map its commit's `Move:` line names (`Terms`). Neither script rewrites research's
`moves.json`, to which each move appends its renamed paths.

A map lives only as long as its move. `move_check.py` reads the word maps in this directory at a move's base, so a word map
lands just before its move and is deleted once the move has landed; git keeps it in the commits that ran it. Each concept
has one map: two maps here that rename one word must reach one name whichever of their moves runs first, and
`tests/test_maps.py` fails the map that would give a word a second name, when that map's PR runs. The toy tests
keep two past word maps as fixtures (`tests/fixtures/`). `layout_map.toml` stays while `tools/lean/setup.sh` maps old Lake
package paths through it.

## Staging a move

A move lands alone, as one commit per map on the commit it was made on. Its lead:

1. lands each map first, as an ordinary change: `tools/move/<name>.toml`, directly in this directory, where `move_check.py`
   reads a word map at the move's base and anyone reruns the move from;
2. stages the move on the base it will land on, `main`:

       python3 tools/move/stage.py tools/move/NAME.toml [tools/move/MORE.toml...] --branch BRANCH --push --certificate cert.json

   For each map in order, on `origin/main` (fetched first; `--base REF` names another) or on the commit the map before it
   made, it runs the base's own `rename.py` (a map with `[words]` or `[[words_in]]`, under a `[words]` table that may be
   empty) or `layout.py` (a map with `[paths]`; `--no-lock` skips
   `uv lock`) at the root of a fresh worktree, commits what that changed as one commit whose message names the command and
   ends in its move line, `Move: MAP` or `Layout-move: MAP`, certifies it with the base's `move_check.py PARENT COMMIT`,
   and runs the map again to check that changes nothing. Nothing in the checkout changes but the new branch, pushed and never forced.
   `--certificate` writes each commit's `move_check.py --json` report;
3. hands the lander the branch and the command `stage.py` printed for each commit, `verity check move PARENT COMMIT`
   (`move_check.py`, which exits 0 on a pure move, 1 naming what isn't).

Over the whole tree `rename.py` and `move_check.py` each run for minutes, so the lead stages on a node, from a checkout at
the window's base. The node's clone can't push and its git has no identity (`stage.py` refuses before running), so the
run names one and bundles the branch into its outputs, and the lead pushes it from the checkout:

    git fetch origin main && git checkout --detach origin/main
    research run --on auto --cwd clone -- bash -c 'export GIT_{AUTHOR,COMMITTER}_NAME=LEAD GIT_{AUTHOR,COMMITTER}_EMAIL=EMAIL;
      python3 tools/move/stage.py tools/move/NAME.toml --base HEAD --no-fetch --branch BRANCH \
        --bundle "$OUT/move.bundle" --certificate "$OUT/cert.json"'
    research fetch RUN --all && git fetch ~/.research/runs/RUN/out/move.bundle BRANCH:BRANCH && git push -u origin BRANCH

If `main` moves before the move lands, the lead stages it again on the new `main`: the map reruns over whatever landed
since. A word map can also go through the queue, as a PR whose body says `Move: tools/move/NAME.toml`: drive regenerates
its move on the tip below it with `checked`'s scripts, as `stage.py` does with its base's, lands it only if certified
(the team skill's "Landing"), and restacks `next`'s members across it. Drive knows only the moves it regenerated: a move
landed outside the queue leaves each open branch, `next`'s members among them, to cross it with `restack.py`.

`stage.py` refuses (exit 1) a map outside this directory or missing at the commit it would run on, a map with both
`[words]` and `[paths]` or neither (a move that needs both stages two maps), a map with `[[words_in]]` and no `[words]`
table, a script that fails, a map that changes
nothing, a commit `move_check.py` doesn't certify, and a move whose second run changes anything. `move_check.py` refuses,
with each path's first offending hunk, every change its docstring doesn't list as part of a pure move. Among them:
- a file's text changed beyond its moved paths, module names and the words of the word maps at the base, as `rename.py`
  writes them;
- under `scope = "prose"`, a word renamed in code, a string literal, backticks or a data file; a name renamed to one the
  file already spells, or to different names in different files; a Lean name a lock records; any edit to a file a JSON
  pin records;
- a lock's records (`pins`, `guarantees`, `reads`) changed, or a pin rewritten to other than its file's digest at the
  head;
- a file added or deleted that no rename pairs, other than a new Lake package's files copied from its source package.

Once the moves have landed, each open branch crosses all of them with one command:

    python3 tools/move/restack.py BRANCH --push-to NEWBRANCH --kept

It finds each move `origin/main` (`--main`; fetch it first) has and the branch lacks by its move line, oldest first, reruns that move's
script and map on the branch, its own additions included, and merges the moved `main` last. It changes no branch: the
result is pushed to the new branch, never over one. It reports `clean`, or where it stopped: a conflict of the branch's
own, a script that failed, or pins or Lean locks to regenerate. Its first step merges each move's base into the branch, so
a branch behind `main` merges `main` up to the window's base before the window: otherwise `restack.py` stops at that
merge's conflicts, which are the branch's own. Before a branch crosses a move, `restack.py` checks that the move's commit
is what its script writes on its base, which takes as long as the script. With `--kept` a check that passed is kept in the
check's cache (under `~/.cache/verity-check/restack-generated/`), so on one machine only the first restack across a move
runs it. A word move that renames declarations a lock records leaves
those records for a commit of their own: `tools/verity/lean/check.py --update --moved` with the map `rename.py --moves`
prints.

    uv run tools/verity/check/suites.py tools/move     # the tests (remotely: AGENTS.md "Tests")
