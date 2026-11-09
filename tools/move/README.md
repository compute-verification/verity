# tools/move

Codemods that move or rename part of the tree and rewrite every reference to it, so a move is one generated commit that
`tools/verity/check/move_check.py` can certify as a pure move. Each is stdlib and git, run as a file from the repository
root, and refuses a dirty worktree; a second run changes nothing.

| Tool | What it does |
|---|---|
| `layout.py` | Moves the paths a layout map names and rewrites every path, module name, import, pyproject and anchor that pointed at them (`--dry-run`, `--modules`, `--resolve M`). |
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

## Landing a move

A move lands through the queue, drive's or the verified queue's, as one commit made on the tip it lands on. Its lead:

1. lands its map first, as an ordinary change: `tools/move/<name>.toml`, directly in this directory, where `move_check.py`
   reads a word map at the move's base and anyone reruns the move from;
2. opens the move's PR, whose body says `Move: tools/move/NAME.toml`. Its head is only the request; it needs a ready mark and
   a land verdict like any PR, and never lands on its mark alone.

The queue regenerates the move on the tip below it with `main`'s scripts: `rename.py --map MAP` for a word map (a
non-empty `[words]`), `layout.py --map MAP` for a path map (a non-empty `[paths]`; `--no-lock` on a tree with no
`uv.lock`), at the root of a fresh worktree. What that changed is one commit, whose message names the command and ends in
the PR's trailers and its move line, `Move: MAP` or `Layout-move: MAP`, by which `restack.py` finds the move and its
script; `main`'s `move_check.py PARENT COMMIT` (`verity check move`) must certify it. A tip that changes regenerates it
again, so the map reruns over whatever landed since.

The queue leaves the PR out, telling its owner why once a head, for a map outside this directory or missing at the tip, one
that doesn't parse, one with both `[words]` and `[paths]` or neither (a move that needs both is two move PRs, each with its
map), a script that fails, a map that changes nothing and a commit `move_check.py` doesn't certify; a PR below the move that
changes this directory's scripts holds it back until it lands. A PR above the move whose change names a word, path or module
of its map, or whose plain merge conflicts with it, crosses it by `restack.py` (below) before it merges; a move that
landed outside the queue, which no queue regenerated, leaves each open branch to cross it so. `move_check.py` refuses,
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

    python3 tools/move/restack.py BRANCH --push-to NEWBRANCH

It finds each move `origin/main` (`--main`; fetch it first) has and the branch lacks by its move line, oldest first, reruns that move's
script and map on the branch, its own additions included, and merges the moved `main` last. It changes no branch: the
result is pushed to the new branch, never over one. It reports `clean`, or where it stopped: a conflict of the branch's
own, a script that failed, or pins or Lean locks to regenerate. Its first step merges each move's base into the branch, so
a branch behind `main` merges `main` up to the window's base before the window: otherwise `restack.py` stops at that
merge's conflicts, which are the branch's own. A word move that renames declarations a lock records leaves
those records for a commit of their own: `tools/verity/lean/check.py --update --moved` with the map `rename.py --moves`
prints.

    uv run tools/verity/check/suites.py tools/move     # the tests (remotely: AGENTS.md "Tests")
