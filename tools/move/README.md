# tools/move

Codemods that move or rename part of the tree and rewrite every reference to it, so a move is one generated commit that
`tools/verity/check/move_check.py` can certify as a pure move. Each is stdlib and git, run as a file from the repository
root, and refuses a dirty worktree; a second run changes nothing.

| Tool | What it does |
|---|---|
| `layout.py` | Moves the paths a layout map names and rewrites every path, module name, import, pyproject and anchor that pointed at them (`--dry-run`, `--modules`, `--resolve M`). |
| `rename.py` | Renames the words a word map names, in paths, modules, Lean names and prose, leaving the Lean locks' records for `tools/verity/lean/check.py --update --moved` (`--map MAP`, `--dry-run`, `--moves`). |
| `restack.py` | Carries an open branch across a move by rerunning the move's script on it, then merging the moved `main`. |

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
`[keep]` and `[keep_in]` for `rename.py`.

A map lives only as long as its move. `move_check.py` reads the word maps in this directory at a move's base, so a word map
lands just before its move and is deleted once the move has landed; git keeps it in the commits that ran it. The toy tests
keep two past word maps as fixtures (`tests/fixtures/`). `layout_map.toml` stays while `tools/lean/setup.sh` maps old Lake
package paths through it.

    uv run tools/verity/check/suites.py tools/move     # the tests (remotely: AGENTS.md "Tests")
