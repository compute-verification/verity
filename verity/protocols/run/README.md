# run

`run` is the top-level protocol: it establishes one verdict a monitoring period over every protocol an application
lists, from those protocols' own verifiers, whose answers the Lean composite combines; `run` decides none itself. It
takes the public parameters and the workload, holds the run's state between periods (`--state`), and runs each
protocol's parties in the order its table gives.

Its Lean statement: `Properties.Composition.AcceptMeans` (`security/Properties/Composition.lean`).

A period runs in the order `run_period` (`period.py`) writes once: the memory wipe, registration, then for each window
the accounting layer, compilation first, and the policy layer on what accounting returned, then the period's result.

Not yet: compilation accepts every circuit, the wipe is the adapter's launcher's, and each step's body is the runner's (`benchmarks/one_stage/served_verdicts.sh`).

## Pieces

| piece | what it is | where |
| --- | --- | --- |
| `verity run` | reads an application, places its parties on machines and runs them for each period | `tools/verity/cli/verity_cli/run.py` |
| the period's order | `run_period`: the stages in their order, the public parameters, the state between windows | `period.py` |
| the parties' table | each listed protocol's entry for each party of the served window, each stage's body and the steps' two lists (`ACCOUNTING`, `POLICY`) | `tools/verity/cli/verity_cli/served_table.py` |
| the accounting protocols' declarations | each protocol's name to the module that declares it | `schemes.py` |
| the Lean verifier | the composition: one verdict rule over any list of protocols | `verifier/lean/` |
| its two composites | onsite's and remote's lists of protocols | `verifier/onsite/`, `verifier/remote/` |
| the auditor's device | the challenger service: each session's coins, its signed record written before the first | `verifier/live_challenger/` |
| its circuits, its device code | none of its own: each protocol brings its own | |

## What varies

| implementation | SPOT | PoUW (Pearl+) | inference-only | PoUS | PoUB | network attester | memory wipe |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `applications/remote.toml` | exists | exists | exists | never written | never written | never written | never written |
| `applications/onsite_full.toml` | exists | exists | exists | exists | exists | exists | exists |

exists: the application lists the protocol and `verity run --dry-run` resolves its parties; never written: it does not list it.

A new implementation supplies: an application file listing its protocols, parties, channels and pins, and for a set of
protocols no composite holds, its composite, named in `Core.Composites.ofIds` (`verifier/onsite/CoreComposites.lean`).
Fixed for every implementation: the configuration format `verity/config/v0`, the names in `schemes.py`, and the verdict
rule: any protocol's reject rejects, and an accept needs every protocol's accept and the commitment binding their digests.

## Interfaces

- From the application: `format`, `protocols`, `workload` (`name`, `params`), `service` (the sampling design), `program` (`commit`),
  `params` (each protocol's, by its id), `party`, `pin`, `channel`; the hosts come in `--bindings`, which change no hash.
- To each protocol: its `params`, its parties on their machines, and the run's state directory.
- It gives: each period's `VERDICT` line (the composite's verdict, each protocol's under `components`), printed and
  appended to `STATE/record/verdicts.jsonl`; exit 0 when every period accepts, 1 when one rejects, 2 otherwise; and
  last, the period's `PERIOD` line: each stage's end, the outcome over both, and what compilation, the wipe and
  registration said.

## Try it

~~~sh
uv run verity run applications/remote.toml --dry-run
~~~

It runs nothing and needs no GPU: it prints a `PLAN` line for each of the four parties (workers, challenger, ZK server,
verifier: its machine and each protocol's entry), a `CHANNEL` line for each of the five channels, and a `DRY` line of
the counts, the configuration's hash and no unbound binding.
