# `circuit-check`

`circuit-check` is the property checks every circuit passes whenever it is built or changed: a Definition, a subcircuit
template or one of its subcircuits, a served workload's request Program, and their Boolean lowerings. A change that adds or
modifies a circuit puts its report in the PR, and `check` runs `circuit-check --all`.

~~~sh
uv run circuit-check F32Add_v1 'RMSNormTriton_v1{N=64,EPS=1e-05}'    # Definitions, at a binding
uv run circuit-check MODULE.circuit-check.py                         # what a sidecar's bindings reach in its module
uv run circuit-check --all                                           # every registered Definition and template
uv run circuit-check --structural UNIT.tmpl TARGET                   # a template as the Definition's lowering
uv run circuit-check --ambient TARGET...                             # each lowering, read under a perturbed interpreter
~~~

It exits 1 on a failure `known/` doesn't list, and under `--all` on a known entry that no longer fails. Redundant and dead
gates are warnings with counts (`--fail-on-warnings` makes them fail). A target whose every input is unchanged is reused
from the result cache (`--no-cache` checks it now). It needs about 24 GB a process, so run it on a node through
`research run --on auto`.

| Module (`src/circuit_check/`) | What it holds |
| --- | --- |
| `checks.py` | the checks: correspondence with the reference on every realization, declared port intervals, redundant and dead gates, the partition |
| `bodies.py` | a composite's evaluation and its non-constant gates, body by body |
| `targets.py` | what a target string names, and the catalog `--all` runs over |
| `sidecars.py` | each Definition module's bindings and pins, in its `<module>.circuit-check.py` and `.circuit-check.json` |
| `vectors.py` | the edge and seeded random input vectors |
| `structural.py`, `ambient.py`, `by_calls.py` | `--structural`, `--ambient`, and a Program root checked by its calls alone |
| `cache.py`, `units.py` | the result cache, and the catalog bindings' records, each reused while what it read is unchanged |
| `known.py`, `entries.py` | the failures `main` already has, each with its cause, one file per entry (`known/`, `pins/`) |
| `cli.py` | the command line and the report (`verity-circuit-check/v1`) |

`heavy.json` holds the peak memory, in GiB, of each target that grows past 3 GiB: the `--all` scheduler reads it to
decide which targets run side by side, never what a check computes. A new Definition gets a binding in its
module's sidecar: the suite fails any registered Definition family the catalog doesn't reach.
