# `verity`: design

What `verity run` runs for the onsite demo, and how its two commands of record are launched.

## The onsite served window

`demos/onsite_full.toml` is one configuration (`verity/config/v0`, `verity.config`) of onsite's seven
subprotocols on the `served-window` workload: one execution phase of Llama-3.2-1B in int8 served through the vLLM adapter's
committing endpoint. `verity run` executes it, reading `verity_cli/served_table.py`. The file is
`served_table.configuration` rendered by `verity.config.render`, and `tests/test_onsite_served.py` holds it to that byte for
byte, as it does each other demo file. It names no host and no address: those are the bindings' (`--bindings`, `verity_cli.run`'s
docstring), so binding another host changes no hash. It pins the memory attester's two keys, one for PoUS's runs
and one for the wipe's (`[[pin]]`, their only statement: `served_table.configuration` reads them from this file, and each
smaller demo file holds those of the subprotocols it lists), and they are inside its hash, which
`verity_cli/configurations.toml` holds.

### Interface

| Entry point | What it is |
|---|---|
| `verity run demos/onsite_full.toml [--dry-run] [--on MACHINE] [--bindings FILE] [--state DIR]` | Reads the configuration and the bindings, runs the pieces of the parties placed on this machine, and keeps the run under `STATE`: the served window's public record in `STATE/out`, and the monitoring period's `VERDICT` line, printed and written to `STATE/record/verdicts.jsonl`. |
| `served_table.configuration(commit: str, demo: str = "onsite_full") -> dict[str, Any]` | A demo file's configuration as data (`served_table.DEMOS`). |
| `served_table` | What `verity run` reads of the served window: `MANIFESTS` (each subprotocol's declaration, a `Component`), `WORKLOADS` (the workload's declaration with its piece), `PARTIES` (each subprotocol's piece for each of its parties, a `Piece`), `ORDER`, and `live(cfg: Mapping[str, Any], state: Path, here: Sequence[str]) -> Result`, which runs the pieces of the parties here and returns each party's outcome or refusal. |

### Running it

Every party resolved, every channel and piece, and nothing run (no GPU); with `--bindings`, a missing binding refused:

    verity run demos/onsite_full.toml --dry-run --bindings BINDINGS

On the developer's node, from the commit `research deploy install` installed (set `[program] commit` to it, else the verdict
names the difference under `simulated`): the served window, every piece inside it, the auditor's sides opened through
their bindings:

    verity run demos/onsite_full.toml --on prover-node --bindings BINDINGS --state STATE

The gate's and the auditor device's sides are started by their gates when the developer's pieces open them, so
`verity run --on gate-host` starts nothing. On the auditor device, with `STATE/out` holding the served window's public
record, the onsite check and the period's verdict:

    verity run demos/onsite_full.toml --on auditor-host --state STATE

A run of record launches each of the two commands under `research run`, and `benchmarks/demo_minutes/onsite_minutes.py`
reads the minutes from those two runs (`vllm/onsite-demo-minutes` in `benchmarks/demo_minutes/metrics.toml`).
