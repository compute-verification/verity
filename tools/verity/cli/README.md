# `verity`

`verity` is the repository's one command (the distribution `verity-cli`, its package `verity_cli`). It dispatches to the
tools beside it and holds no check of its own. `uv run verity --help` prints the usage `verity_cli/__init__.py` states.

| Command | What it runs |
| --- | --- |
| `verity check [ARGS]` | the whole check, `tools/verity/check/check.py` |
| `verity check tests\|circuits\|lean` | only those parts of it, with the jobs `check/ci.toml` puts in each |
| `verity check --record [ARGS]` | the check of a clean commit, recorded as an Attempt |
| `verity check move BASE HEAD` | the certificate that BASE..HEAD is a pure move (`check/move_check.py`) |
| `verity bench NAME [ARGS]` | `benchmarks/<NAME>/`'s runner, its `run.py` or `run.sh` |
| `verity example NAME [ARGS]` | `examples/<NAME>/`'s command: `verity run` on its `config.toml`, else its runner |
| `verity example --status` | every example's place on its two axes, how far it runs and how far its guarantee is proved, and what each next step needs |
| `verity run CONFIG` | the parties a deployment configuration places on this machine, and its verdict |
| `verity config new\|check\|hash` | write, check or hash a deployment configuration |
| `verity lean try FILE` | `lake env lean FILE` in the Lake package that holds it, the Lean edit loop; with `--on NODE` on a project machine |

In a NAME, a `-` stands for a `_`.

## Modules

| Module | What it holds |
| --- | --- |
| `verity_cli/__init__.py` | the dispatch, `verity check` and `verity lean try` |
| `verity_cli/run.py` | `verity run` and `verity config`, over `verity.core.deploy.config` |
| `verity_cli/examples.py` | the examples' ladder, read from each `examples/<name>/example.toml` |
| `verity_cli/settings.py` | the environment variables `verity run` reads, declared once |

`verity run` and `verity config` refuse a configuration (`no-table`) until a workload's protocols bring a table of parties;
the suite runs one of its own (`tests/step0.py`).

~~~sh
uv run tools/verity/check/suites.py tools/verity/cli   # the suite, on a node through `research run --on auto`
~~~
