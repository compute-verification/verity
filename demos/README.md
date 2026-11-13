# Demos

One directory per demo. A demo holds what is its own: the workload it serves, its placement (where each party
runs), and the commands a person runs to serve and audit it. What attaches Verity to the serving stack is the integration's
(`adapters/`), and the protocols it runs stay in `verity/protocols/`. `verity` imports nothing from `demos/`
(`tests/test_boundaries.py`), and `verity demo NAME` runs `demos/<NAME>/config.toml` (`verity run`), else its `run.py`
or `run.sh`.

## The remote protocol

Remote Verity (`verity/remote/v1`, `Core.Remote.V1` in `verity/protocols/remote/verifier/CoreRemote.lean`) is what the
verifier establishes with no hardware of its own at the developer's site. It composes three protocols
(`Core.Remote.V1.components`):

- integrity, by SPOT (`verity/sampled-proofs/v1`, in its one-stage form): the prover commits every value of
  a served window under hiding leaves, a live coin from the challenger samples proof blocks, the prover proves each sampled
  proof block in Flock `--zk`, and the Lean verifier checks each proof against the committed roots;
- work, by Pearl+'s run verdict (`verity/pouw/pearl-plus/v1`): tiles of the run's retained passes, sampled by work
  weight on a live coin, are recomputed, and the run's commitment names the one record the policy and SPOT
  committed to (`Core.Remote.V1.run`). H1's development executable, `verity/remote/v0` (`Core.Remote.protocol`),
  which no configuration names, has PoUW's period check (`verity/pouw/period/v1`, onsite's work subprotocol too) in its place: the GPU's time is credited
  to useful matmuls through receipts the live coins open, the period's windows all or nothing, each window's Commit
  record that one record;
- the inference-only policy (`verity/inference-only/v0`): the registered circuit divides into inference blocks within
  their gate and fresh-input budgets, checked from the registered circuit and the registered information alone.

Remote accepts only when all three accept over one commitment that binds them, each having committed to the same record
(`Core.Composition.same`); any one's rejection rejects it, whatever the others answer. It rests on live coins (flipped after
the commitment, never flipped again) and on what each subprotocol's own
guarantee assumes; each subprotocol states its guarantee in `security/`, and the composite's
(`Verity.RemoteSoundness`, `security/Properties/Verity/Soundness.lean`) is proved from theirs, which it takes as
hypotheses, so no demo's `proof` is past `none`. `lifecycle/` is the only demo that steps through the whole
remote composite, H1's development one, on CPU with a stand-in prover. Its demo is the vLLM demo
(`verity demo vllm`), on H1's development executable: one served run of Llama-3.2-1B INT8 on one H200, SPOT
and PoUW on the served model, whose PoUW verdict is a development run that credits nothing. The inference-only policy
runs in `lifecycle/` only: no served run produces the declared
division yet. Five other things are called "remote" (PoUB among them);
`verity/protocols/remote/verifier/README.md`, "Which remote", lists them.

## How far a demo runs, and how far it's proved

A demo stands on two independent axes, and its `demo.toml` gives each. `run` is how far it runs:

| `run` | what the demo has | evidence |
| --- | --- | --- |
| `moved` | its workload, placement and commands, and no end-to-end run yet | none |
| `runs` | an end-to-end run: a run of record on its hardware, or a CPU test that runs it with stand-ins | a run of record (`runs`: `rYYYYMMDD-HHMMSS-xxxx` or `art:` ids), or a CPU test in a suite that runs it (`tests`: `PATH::NAME`) |
| `placed` | a run of record where the auditor's parties run on machines the developer doesn't control | a run on a placement that puts every auditor role (`verifier`, `challenger`, `certifier`, `memory_challenger`) on `auditor-host`, and none of the developer's |

`proof` is how far its guarantee is proved: a proof needs no run, and a run needs no proof.

| `proof` | evidence |
| --- | --- |
| `none` | none |
| `stated` | `guarantee` listed in a `lean-audit.json`, with its `honest` and `nontrivial` lemmas declared in Lean |
| `proved` | `[evidence] proof`, the guarantee's `<Area>.SecurityProofs.G`, declared in Lean |
| `measured` | `[evidence.premises]`: each premise a lock lists (under `assumptions`, `domain` or `open`), to the run that measured it |

A file that still gives the retired `level` instead of the two axes is read as `run` `moved`, `runs` or `placed` for 0, 1
or 2, with `proof` `none`.

## `demo.toml`

`vllm/demo.toml` is one. It gives the demo's `protocol` (a package under `verity/protocols/`, the composite
`onsite` or `remote`, or a list), its `workload`, its `[placement]` (each role to a machine class), its `run` and `proof`,
and the `[evidence]` every step up to each needs. A workload is what runs: its Program, its inputs and its placement. The
`workload` field names the integration (a directory under `adapters/`, or `none`); once workload specs exist, it names
a workload spec id, `<frontend>:<id>`, whose spec carries the model and the load. A directory's name is lowercase letters,
digits and `_`, since `verity demo` reads a `-` in NAME as a `_`.

`tests/test_examples.py` fails a demo that lacks the evidence for its `run` or its `proof`, saying what to add, and
`verity demo --status` prints every demo's next missing evidence on each. The roles, the machine classes and the
checks are `tools/verity/cli/verity_cli/examples.py`'s; a new role goes in its `ROLES`, with its side.

## The demos

- `lifecycle/` (`run` `runs`, `proof` `none`): both composites, `onsite` and `remote`, stepped through one monitoring
  period on CPU in about 2 s, from the hardware registration to the period's verdict and accounting. Every party is
  simulated in one process, the prover is a labelled stand-in for Flock, and the verdict is the composition's Python
  mirror (`verity/protocols/remote/verifier/composition/tests/reference.py`), not `remote check`. It is the only demo that steps
  through the whole remote composite; its tests (`demos/lifecycle/tests/test_lifecycle.py`) run each composite honest
  and with every deviation it lists refused. Nothing about it is proved.
- `vllm/` (`run` `moved`, `proof` `none`): the vLLM demo, the remote protocol on a real served model. Llama-3.2-1B INT8
  served by vLLM on one H200 through `verity_vllm`'s committing endpoint; `verity demo vllm` serves one execution phase,
  commits its values, and audits it through `benchmarks/one_stage/served_verdicts.sh`: live coins sample proof blocks,
  Flock `--zk` proves them and the Lean verifier checks them, PoUW runs on the served model, and onsite's subprotocols
  run where each has its piece. Its run is `moved`: PoUW's verdict is a development run that credits nothing, so no run of
  record of the remote verdict has passed. Its workload is the integration's one that `placement.toml`'s `row` names,
  its placement is `placement.toml`, which `run.sh` reads, and its scripts reach the integration by `PYTHONPATH`; it is
  not a package. The auditor's devices (the verifier, the network certifier and the memory challenger) are not placed:
  the Lean verifier runs on the GPU node with the endpoint and the prover, and the other two may run there too.
- `onsite_served/` (`run` `moved`, `proof` `none`): onsite's run configuration for the served window. One
  `verity run` configuration (`config.toml`) of onsite's seven subprotocols on the served int8 window: its
  parties, channels and each subprotocol's piece, PoUS's at 320 challenges, which `verity run --on MACHINE --bindings FILE`
  executes on each machine, the hosts and addresses only in the deploy bindings. `verity run CONFIG --dry-run` resolves
  it without a GPU. It has no run of record yet.
