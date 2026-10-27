# Examples

One directory per example. An example holds what is its own: the workload it serves, its placement (where each party
runs), and the commands a person runs to serve and audit it. What attaches Verity to the serving stack is the integration's
(`integrations/`), and the protocols it runs stay in core (`verity/core/`). `verity` imports nothing from `examples/`
(`tests/test_boundaries.py`), and `verity example NAME` runs `examples/<NAME>/config.toml` (`verity run`), else its `run.py`
or `run.sh`.

## Levels

Each example has one level, how far it runs, and its `example.toml` gives it as `run`:

| level | `run` | what the example has | evidence |
| --- | --- | --- | --- |
| 0 | `moved` | its workload, placement and commands, and no end-to-end run yet | none |
| 1 | `runs` | an end-to-end run: a run of record on its hardware, or a CPU test that runs it with stand-ins | a run of record (`runs`: `rYYYYMMDD-HHMMSS-xxxx` or `art:` ids), or a CPU test in a suite that runs it (`tests`: `PATH::NAME`) |
| 2 | `placed` | a run of record where the auditor's parties run on machines the prover doesn't control | a run on a placement that puts every auditor role (`verifier`, `challenger`, `certifier`) on `auditor-host`, and none of the prover's |

How far the example's guarantee is proved is not a level: a proof needs no run, and a run needs no proof. `example.toml`
gives it as `proof`, with the evidence each step needs:

| `proof` | evidence |
| --- | --- |
| `none` | none |
| `stated` | `guarantee` listed in a `lean-audit.json`, with its `honest` and `nontrivial` lemmas declared in Lean |
| `proved` | `[evidence] proof`, the guarantee's `<Area>.SecurityProofs.G`, declared in Lean |
| `measured` | `[evidence.premises]`: each premise, an assumption a lock lists, to the run that measured it |

## `example.toml`

`vllm/example.toml` is one. It gives the example's `protocol` (a package under `verity/core/protocols/`, the composite
`onsite` or `remote`, or a list), its `workload`, its `[placement]` (each role to a machine class), its `run` (its level)
and `proof`, and the `[evidence]` every step up to each needs. A workload is what runs: its Program, its inputs and its
placement. The `workload` field names the integration (a directory under `integrations/`, or `none`); once workload specs
exist, it names a workload spec id, `<frontend>:<id>`, whose spec carries the model and the load. A directory's name is
lowercase letters, digits and `_`, since `verity example` reads a `-` in NAME as a `_`.

`tests/test_examples.py` fails an example that lacks the evidence for its level or its `proof`, saying what to add, and
`verity example --status` prints every example's next missing evidence on each. The roles, the machine classes and the
checks are `tools/verity/cli/verity_cli/examples.py`'s; a new role goes in its `ROLES`, with its side.

## The examples

- `dither_full_util/`, level 0: the H200 at full utilization on useful s8 GEMMs, with PoUW over Dither's no-fold receipt.
  It has no runner and no run of record. Proof: none.
- `h200/`, level 1: the one-H200 demo, with its subprotocols' interfaces in `interfaces/`: the frozen MQA shape served on
  one GPU node with every op's output committed a step. Its `run.py` is the step driver, and `phases.sh` runs the drawn-block
  proving one phase a run, from the step's commit on the H200 to its drawn blocks proved. In its run of record, 8 proof
  blocks drawn by one live coin are each proved by C-Flock `--zk` and accepted by Lean, and a flipped bit is refused. It is
  not level 2: the Lean verifier ran on a CPU node beside the prover. Its memory audits (PoUS and the bandwidth audit on the
  H200) aren't claimed. Proof: none.
- `vllm/`, level 1: Llama-3.2-1B served by vLLM on one GPU through `verity_vllm`'s committing endpoint, a served window
  audited by sampled proofs, and by the other subprotocols where each has its piece. Its row is in its own `workloads/`, its placement is `placement.toml`, which `run.sh`
  reads, and its scripts reach the integration by `PYTHONPATH`; it is not a package. It is not level 2: its runs of record
  ran the Lean verifier on the GPU node with the endpoint and the prover. Proof: none.
