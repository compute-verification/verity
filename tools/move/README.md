# `tools/move`

A move relocates or renames part of the tree: a directory goes somewhere else, or a word gives way to its new name.
These tools turn a move into one generated commit that rewrites every reference to what moved, which
`verity check move` can certify as a pure move: nothing changed but paths, module names and the map's words. Because the
commit is generated, anyone can rerun it, so a move lands without a line-by-line review, and every open branch crosses
it by rerunning it on the branch.

Each tool is the standard library and git, run as a file from the repository's root; it refuses a dirty worktree,
and a second run changes nothing.

## Try one

~~~sh
python3 tools/move/rename.py --help
~~~

## The tools

| Tool | What it does |
| --- | --- |
| `stage.py` | Stages a move on a base: one certified commit per map, on a branch, for its pull request. |
| `layout.py` | Moves the paths a layout map names and rewrites every reference to them. |
| `rename.py` | Renames the words a word map names, in paths, modules, Lean names and prose. |
| `layout_maps.py` | Writes the layout as one map from the tree, and fails on a tracked file with no destination. |
| `tests_root.py` | Writes the maps that move every test under the root's `tests/`, mirrored by path. |
| `restack.py` | Carries an open branch across every move `main` has and it lacks, by rerunning each move on it. |
| `resolve_renames.py`, `stale_names.py` | After a plain merge of `main`: resolve the conflicts that are only a move's renames, and list the lines still in old names. |
| `flock_base.py` | Follows the Flock modules that moved to the base Lean packages. |

Each tool's `--help` gives its usage.

## Staging and landing a move

The map lands first, as an ordinary change. Then `stage.py` makes the move's commits on `main` and certifies each:

    python3 tools/move/stage.py tools/move/NAME.toml [tools/move/MORE.toml...] --branch BRANCH --push --certificate cert.json

Over the whole tree that runs for minutes, so the lead stages on a node. The node's clone can't push and its git has no
identity, so the run names the identity of the checkout that will push the result, and bundles the branch into its
outputs:

    git fetch origin main && git checkout --detach origin/main
    research run --on auto --cwd clone -- env \
      GIT_AUTHOR_NAME="$(git config user.name)" GIT_AUTHOR_EMAIL="$(git config user.email)" \
      GIT_COMMITTER_NAME="$(git config user.name)" GIT_COMMITTER_EMAIL="$(git config user.email)" \
      bash -c 'python3 tools/move/stage.py tools/move/NAME.toml --base HEAD --no-fetch --branch BRANCH \
        --bundle "$OUT/move.bundle" --certificate "$OUT/cert.json"'
    research fetch RUN --all && git fetch ~/.research/runs/RUN/out/move.bundle BRANCH:BRANCH && git push -u origin BRANCH

A move's pull request, whose body says `Move: tools/move/NAME.toml`, lands on its ready mark alone, with no land verdict
while `tools/check/drive.toml`'s `lowrisk_lane` is on: the verified queue reruns the map on its batch's tip and takes
the move once `move_check.py` certifies that run.

Once it lands, each open branch crosses it:

    python3 tools/move/restack.py BRANCH --push-to NEWBRANCH --kept

A commit written by a script with no map has no move line, so `restack.py` merges it as a plain commit and leaves the
branch's own lines in its old names. The Lean names pass's docket part, `5916146988e4`
(`tools/move/lean_names_pass.py --apply docket`: eight names of SPOT's run record become `docket`, in code only), is
one. A branch made before it that touches those names crosses in three runs, each `--dry-run` printing the result
the next one reads, the middle one naming the part and its script:

    python3 tools/move/restack.py BRANCH --main a98c0b3a1d48 --dry-run
    python3 tools/move/restack.py RESULT1 --move-commit 5916146988e4 --script tools/move/lean_names_pass.py \
      --script-arg=--apply --script-arg docket --dry-run
    python3 tools/move/restack.py RESULT2 --push-to NEWBRANCH --kept
