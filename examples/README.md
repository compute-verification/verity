# Examples

One directory per example. An example holds what is its own: the workload it serves, its placement (where each party
runs), and the commands a person runs to serve and audit it. What attaches Verity to the serving stack is the integration's
(`integrations/`), and the protocols it runs stay in core (`verity/core/`). `verity` imports nothing from `examples/`
(`tests/test_boundaries.py`), and `verity example NAME` runs `examples/<NAME>/config.toml` (`verity run`), else its `run.py`
or `run.sh`.

## The remote protocol

Remote Verity (`verity/remote/v0`, `Core.Remote` in `verity/core/protocols/remote/CoreRemote.lean`) is what the verifier
establishes with no hardware of its own at the prover's site. It composes three protocols (`Core.Remote.components`):

- integrity, by sampled proofs' one-stage audit (`verity/sampled-proofs/one-stage/v2`): the prover commits every value of
  a served window under hiding leaves, a live coin from the challenger draws proof blocks, the prover proves each drawn
  block in C-Flock `--zk`, and the Lean verifier checks each proof against the committed roots;
- work, by PoUW's window check (`verity/pouw/v1`): the GPU's time is credited to useful matmuls through receipts the live
  coins open;
- the inference-only policy (`verity/inference-only/v0`): the declared circuit divides into inference blocks within
  their gate and fresh-input budgets, checked from the declared circuit and the registered information alone.

Remote accepts only when all three accept over one commitment that binds them, each having committed to the same record
(`Core.Composition.same`); any one's rejection rejects it, whatever the others answer. It rests on live coins (drawn after
the commitment, never redrawn) and on what each component's own
guarantee assumes; each component states its guarantee in `Security/`, and the composite states none of its own in Lean
yet, so no example's `proof` is past `none`. Its verifier, `remote check`, has no run of record yet; its first is due
Wed 28 Oct, 11 AM PDT. `lifecycle/` is the only example that steps through the whole remote composite, on CPU with a
stand-in prover. Its demo is the vLLM demo (`verity example vllm`): one served run of Llama-3.2-1B INT8 on one H200,
sampled proofs and PoUW on the served model, whose PoUW verdict is a development run that credits nothing; `h200/` runs
remote's sampled-proofs half. The inference-only policy runs in `lifecycle/` only: no served run produces the declared
division yet. Five other things are called "remote" (PoUB among them);
`verity/core/protocols/remote/README.md`, "Which remote", lists them.

## How far an example runs, and how far it's proved

An example stands on two independent axes, and its `example.toml` gives each. `run` is how far it runs:

| `run` | what the example has | evidence |
| --- | --- | --- |
| `moved` | its workload, placement and commands, and no end-to-end run yet | none |
| `runs` | an end-to-end run: a run of record on its hardware, or a CPU test that runs it with stand-ins | a run of record (`runs`: `rYYYYMMDD-HHMMSS-xxxx` or `art:` ids), or a CPU test in a suite that runs it (`tests`: `PATH::NAME`) |
| `placed` | a run of record where the auditor's parties run on machines the prover doesn't control | a run on a placement that puts every auditor role (`verifier`, `challenger`, `certifier`, `memory_challenger`) on `auditor-host`, and none of the prover's |

`proof` is how far its guarantee is proved: a proof needs no run, and a run needs no proof.

| `proof` | evidence |
| --- | --- |
| `none` | none |
| `stated` | `guarantee` listed in a `lean-audit.json`, with its `honest` and `nontrivial` lemmas declared in Lean |
| `proved` | `[evidence] proof`, the guarantee's `<Area>.SecurityProofs.G`, declared in Lean |
| `measured` | `[evidence.premises]`: each premise, an assumption a lock lists, to the run that measured it |

A file that still gives the retired `level` instead of the two axes is read as `run` `moved`, `runs` or `placed` for 0, 1
or 2, with `proof` `none`.

## `example.toml`

`vllm/example.toml` is one. It gives the example's `protocol` (a package under `verity/core/protocols/`, the composite
`onsite` or `remote`, or a list), its `workload`, its `[placement]` (each role to a machine class), its `run` and `proof`,
and the `[evidence]` every step up to each needs. A workload is what runs: its Program, its inputs and its placement. The
`workload` field names the integration (a directory under `integrations/`, or `none`); once workload specs exist, it names
a workload spec id, `<frontend>:<id>`, whose spec carries the model and the load. A directory's name is lowercase letters,
digits and `_`, since `verity example` reads a `-` in NAME as a `_`.

`tests/test_examples.py` fails an example that lacks the evidence for its `run` or its `proof`, saying what to add, and
`verity example --status` prints every example's next missing evidence on each. The roles, the machine classes and the
checks are `tools/verity/cli/verity_cli/examples.py`'s; a new role goes in its `ROLES`, with its side.

## The examples

- `lifecycle/` (`run` `runs`, `proof` `none`): both composites, `onsite` and `remote`, stepped through one monitoring
  period on CPU in about 2 s, from the hardware registration to the period's verdict and accounting. Every party is
  simulated in one process, the prover is a labelled stand-in for C-Flock, and the verdict is the composition's Python
  mirror (`verity/core/protocols/composition/tests/reference.py`), not `remote check`. It is the only example that steps
  through the whole remote composite; its tests (`examples/lifecycle/tests/test_lifecycle.py`) run each composite honest
  and with every deviation it lists refused. Nothing about it is proved.
- `vllm/` (`run` `moved`, `proof` `none`): the vLLM demo, the remote protocol on a real served model. Llama-3.2-1B INT8
  served by vLLM on one H200 through `verity_vllm`'s committing endpoint; `verity example vllm` serves one sampling block,
  commits its values, and audits it through `benchmarks/one_stage/served_verdicts.sh`: live coins sample proof blocks,
  C-Flock `--zk` proves them and the Lean verifier checks them, PoUW runs on the served model, and onsite's subprotocols
  run where each has its piece. Its run is `moved`: PoUW's verdict is a development run that credits nothing, so no run of
  record of the remote verdict has passed. Its workload is the integration's one that `placement.toml`'s `row` names,
  its placement is `placement.toml`, which `run.sh` reads, and its scripts reach the integration by `PYTHONPATH`; it is
  not a package. The auditor's devices (the verifier, the network certifier and the memory challenger) are not placed:
  the Lean verifier runs on the GPU node with the endpoint and the prover, and the other two may run there too.
- `h200/` (`run` `runs`, `proof` `none`): remote's sampled-proofs half at a frozen shape on one H200. The
  frozen multi-query attention shape is served on one GPU node with every value committed a step. `verity example h200
  --out OUT` runs its step driver, `run.py` (on the CPU at small shapes, or an H200 at the frozen shape), and writes a
  replay bundle and a block record; `phases.sh` runs the drawn-block proving, one phase a run, from the step's commit on
  the H200 to its drawn blocks proved. In its run of record, 8 proof blocks drawn by one live coin are each proved by
  C-Flock `--zk` and accepted by the Lean verifier, and a flipped bit is refused. It is not `placed`: the Lean verifier ran
  on a CPU node beside the prover. It claims only sampled proofs: PoUS and network certification (onsite's
  subprotocols), the memory wipe (onsite's) and the bandwidth audit over PoUS's store (neither composite's yet), exist
  here as a skeleton on fixture coins, not as runs, and nothing claims them.
- `onsite_served/` (`run` `moved`, `proof` `none`): onsite's run configuration for the served window. One
  `verity run` configuration (`config.toml`) of onsite's five components and the wipe on the served int8 window: its
  parties, channels and each component's piece, PoUS's at 320 challenges, which `verity run --on MACHINE --bindings FILE`
  executes on each machine, the hosts and addresses only in the deploy bindings. `verity run CONFIG --dry-run` resolves
  it without a GPU. It has no run of record yet.
- `dither_full_util/` (`run` `moved`, `proof` `none`): the remote protocol's PoUW half, as PoUW's work unit. The H200 at
  full utilization on useful s8 GEMMs, credited by PoUW over Dither's no-fold receipt. It has no runner and no run of
  record, so `verity example dither_full_util` runs nothing yet, and it is not a remote run.
