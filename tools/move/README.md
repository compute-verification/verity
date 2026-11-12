# tools/move

Codemods that move or rename part of the tree and rewrite every reference to it, so a move is one generated commit that
`tools/verity/check/move_check.py` can certify as a pure move. Each is stdlib and git, run as a file from the repository
root, and refuses a dirty worktree; a second run changes nothing.

| Tool | What it does |
|---|---|
| `stage.py` | Stages a move on a base: one commit per map, each certified by `move_check.py` and rerun to check it changes nothing, for its PR (`MAP...`, `--base`, `--branch`, `--push`, `--certificate`). |
| `layout.py` | Moves the paths a layout map names and rewrites every path, module name, import, pyproject and anchor that pointed at them (`--dry-run`, `--modules`, `--resolve M`). |
| `layout_maps.py` | Writes the layout as one map, `layout_one_map.toml` (or, by `--form moves`, as one map a move in the order they run, `layout_NN_NAME_map.toml`), from `layout_maps.json` (each moved component's new directory and files) and the files HEAD tracks; it fails on a tracked file with no destination, so it is the one rerun when `main` moves (`--form`, `--ref`, `--targets`, `--created`). |
| `rename.py` | Renames the words a word map names, in paths, modules, Lean names and prose, leaving the Lean locks' records for `tools/verity/lean/check.py --update --moved` and sorting again a sorted JSON object whose keys it renames (`--map MAP`, `--dry-run`, `--moves`). |
| `restack.py` | Carries an open branch across every move `main` has and it lacks, by rerunning each move's script on it, then merging the moved `main` (`BRANCH`, `--push-to`). |
| `flock_base.py` | What follows the modules of Flock's Lean package that moved to the base packages it requires by path: `private` names the module in each `open private … from Flock.M`, `moves` prints the map `check.py --update --moved` takes, and `rewrite` makes every import of an old name, and every lock's `layers` entry for one, name the base module and deletes the re-exports (`--dry-run`). It goes with the re-exports. |

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
3. opens the move's PR from the branch, naming in its body the command `stage.py` printed for each commit, `verity check
   move PARENT COMMIT` (`move_check.py`, which exits 0 on a pure move, 1 naming what isn't).

A move that makes or removes a directory an index lists (`benchmarks/README.md`, which `tests/test_benchmarks_index.py`
holds to one row for each directory) carries the index's row as a second commit of its PR, by hand and changing the index
alone, since no map adds a line. The PR's body says so first, and its reviewer certifies the first commit and reads the
second.

Over the whole tree `rename.py` and `move_check.py` each run for minutes, so the lead stages on a node, from a checkout at
the window's base. The node's clone can't push and its git has no identity (`stage.py` refuses before running), so the
run names one and bundles the branch into its outputs, and the lead pushes it from the checkout:

    git fetch origin main && git checkout --detach origin/main
    research run --on auto --cwd clone -- bash -c 'export GIT_{AUTHOR,COMMITTER}_NAME=LEAD GIT_{AUTHOR,COMMITTER}_EMAIL=EMAIL;
      python3 tools/move/stage.py tools/move/NAME.toml --base HEAD --no-fetch --branch BRANCH \
        --bundle "$OUT/move.bundle" --certificate "$OUT/cert.json"'
    research fetch RUN --all && git fetch ~/.research/runs/RUN/out/move.bundle BRANCH:BRANCH && git push -u origin BRANCH

A move's PR, whose body says `Move: tools/move/NAME.toml`, lands on its ready mark alone, with no land verdict, whatever
it touches: the verified queue reruns the map, from `main`, on its batch's tip, over whatever landed since, and takes it
once `move_check.py` certifies that run (the team skill's "Landing"). It needs a land verdict while drive.toml's
`lowrisk_lane` is off, and under `signed_verdicts` when it changes a `SIGNED_GUARDS` file. Once it lands, each open branch
crosses it with `restack.py`.

Where a `[[words_in]]` row renames a Lean declaration that a file outside the row still spells, `move_check.py`
certifies the move and only a build would find the file. Neither `stage.py` nor `check` looks for it on a move: whoever
stages a move, or stages it again, runs `main`'s `python3 tools/move/resolve_check.py PARENT COMMIT` on each commit of a
word map once `stage.py` has made it, before the move's Lean build, and reads each issue. The script compares the two
commits' text and elaborates nothing, so it names some files that are fine and misses others: an issue is where to look,
and the Lean build stays the authority on whether the move builds.

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

A certificate says that nothing but paths, modules and words changed. It does not say that every path which had to
change did: a path no pass of the rewrite reads stays as written, and its commit is still a pure move. For Lean the mover
checks it itself: `layout.py` stops (exit 1) a move after which a Lake package's path to another package (a requirement's
`path` in a `lakefile.toml`, a path package's `dir` in a `lake-manifest.json`) names no package where it named one
before, and names the file and the path. A path of any other kind that a move leaves behind is found by the check on
its candidate.

Once the moves have landed, each open branch crosses all of them with one command:

    python3 tools/move/restack.py BRANCH --push-to NEWBRANCH --kept

It finds each move `origin/main` (`--main`; fetch it first) has and the branch lacks by its move line, oldest first, reruns that move's
script and map on the branch, its own additions included, and merges the moved `main` last. It changes no branch: the
result is pushed to the new branch, never over one. It reports `clean`, or where it stopped: a conflict of the branch's
own, files of the branch with no row (`no-row`, below), a script that failed, or pins or Lean locks to regenerate. Its
first step merges each move's base into the branch, so
a branch behind `main` merges `main` up to the window's base before the window: otherwise `restack.py` stops at that
merge's conflicts, which are the branch's own. Before a branch crosses a move, `restack.py` checks that the move's commit
is what its script writes on its base, which takes as long as the script. With `--kept` a check that passed is kept in the
check's cache (under `~/.cache/verity-check/restack-generated/`), so on one machine only the first restack across a move
runs it. A word move that renames declarations a lock records leaves
those records for a commit of their own: `tools/verity/lean/check.py --update --moved` with the map `rename.py --moves`
prints. `stage.py` makes the move's commits alone: it makes no records commit and runs no records run.

A layout map is written from `main`'s tree, so a file only a branch has gets a row only where a directory row carries it.
Where the branch holds such a file under a path the map retires (`[retired]`: a directory the map empties, which
`layout_maps.py` lists), `restack.py` stops as `no-row` and names each file, which would otherwise stay at its old path.
The fix is the branch's own rows file: `tools/move/<name>.rows.toml` beside the map `tools/move/<name>.toml`, committed
on the branch, with `[paths]` alone in the map's format:

~~~toml
[paths]                                   # a file, or a directory ending in "/": every file under it
"old/place/added.py" = "new/place/added.py"
"old/place/added/" = "new/place/added/"
~~~

`layout.py` reads it after the map. A row there moves its file and rewrites each reference to it as a row of the map
does; the module renames it implies are derived. The script deletes the rows file once read, so no restacked result
holds one, and it keeps the rows out of `moves.json`, which records the move's own. It refuses a rows
file that would change what the map says (a row for a path the map sends elsewhere or deletes, a directory row over a
path the map names, a module the map renames another way) and a row no tracked file is at or under. A file under a
directory the map does not retire is not stopped: it stays unless a directory row or a row of the rows file moves it.

A branch that shares with `main` a commit a move's base lacks is based after that move, whose script never runs on it:
the move's last merge would bring those commits in a second time. Where a later move's base contains that move, it is
skipped and named (`skipped`): the later move's base merge brings it in, as a merge of `main` would, and the owner looks
for its map's old names in what the branch adds. A crossing that skipped a move reports `clean-skipped`, never `clean`,
pushes nothing and exits 1, and prints the command that crosses the same moves named (`--move-commit` for each, `--onto`),
which skips none: running it is the owner saying the skip is right. The last move, one no later move's base contains, and
one named with `--move-commit` are never skipped: the branch is refused (`based-after`), and its owner merges `main`
instead.

    uv run tools/verity/check/suites.py tools/move     # the tests (remotely: AGENTS.md "Tests")
