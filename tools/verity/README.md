# `tools/verity`

The repository's own tools: the `verity` command and the checks it runs. None of them is imported by `verity`; each reads
the tree and judges it.

| Directory | What it is |
| --- | --- |
| `cli/` | `verity`, the repository's one command: `verity check`, `verity bench NAME`, `verity run`, `verity example NAME`, `verity lean try FILE` |
| `check/` | `check`, the one check: every test suite, `circuit-check --all`, C-Flock's Rust, the Lean verifier's build and the Lean check, as jobs in `ci.toml`; the suite runner, the read guard, verdict packs and the landing gates |
| `circuit_check/` | `circuit-check`, the property checks every circuit passes whenever it is built or changed |
| `lean/` | the Lean check (`check.py`): every Lake package audited against its lock |

Each directory is its own test suite (its `tests/`), and its README says what it holds and how to run it.

~~~sh
uv run verity --help                       # every subcommand
uv run tools/verity/check/suites.py --list # each suite, its inputs, and whether a pass is cached (reads only)
~~~

Builds and tests run on the cluster's nodes, not on an agent VM: wrap a command in
`research run --on auto --cwd source -- bash -lc 'uv sync -q --all-packages --extra torch-cpu && CMD'`.
