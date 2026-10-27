# Examples

One directory per example (Daniel, 25 Oct). An example holds what is its own: the workload it serves, its placement (where
each party runs), and the commands a person runs to serve and audit it. What attaches Verity to the serving stack is the
integration's (`integrations/`), and the protocols it runs stay in core (`verity/core/`). `verity` imports nothing from
`examples/` (`tests/test_boundaries.py`), and `verity example NAME` runs `examples/<NAME>/config.toml` (`verity run`), else
its `run.py` or `run.sh`.
Each example stands on two independent axes of doneness (Daniel, 26 Oct): how far it runs, and how far its guarantee is
proved. A milestone is a set of (example, run, proof) triples, which top keeps. Its `example.toml` says where it stands
(`vllm/example.toml` is one): its `protocol` (a package under `verity/core/protocols/`, the composite `onsite` or `remote`,
or a list), its `workload`, its `[placement]` (each role to a machine class), its `run` and `proof`, and the `[evidence]`
every step up to each needs on its own axis. A workload is what runs: its Program, its inputs and its placement. Until the
workload specs' `workloads/` exists (after vLLM's freeze, Wed 28 Oct 23:00Z), the field names the integration (a directory
under `integrations/`, or `none`); afterwards it names a workload spec id, `<frontend>:<id>`, whose spec carries the model
and the load. A directory's name is lowercase letters, digits and `_`, since `verity example` reads a `-` in NAME as a `_`.

| axis | step | evidence |
| --- | --- | --- |
| `run` | `moved` | none |
| `run` | `runs` | a run of record (`runs`: `rYYYYMMDD-HHMMSS-xxxx` or `art:` ids), or a CPU test in a suite that runs it (`tests`: `PATH::NAME`) |
| `run` | `placed` | a run on a placement that puts every auditor role (`verifier`, `challenger`, `certifier`) on `auditor-host`, and none of the prover's |
| `proof` | `none` | none |
| `proof` | `stated` | `guarantee` listed in a `lean-audit.json`, with its `honest` and `nontrivial` lemmas declared in Lean |
| `proof` | `proved` | `[evidence] proof`, the guarantee's `<Area>.SecurityProofs.G`, declared in Lean |
| `proof` | `measured` | `[evidence.premises]`: each premise, an assumption a lock lists, to the run that measured it |

`tests/test_examples.py` fails an example that lacks what it claims on either axis, saying what to add, and
`verity example --status` prints every example's next missing evidence on each. The roles, the machine classes and the
checks are `tools/verity/cli/verity_cli/examples.py`'s; a new role goes in its `ROLES`, with its side.

- `dither_full_util/`: CMP-176's full-utilization demo on the H200, with PoUW over Dither's no-fold receipt. Level 0, with no
  runner and no run of record yet.
- `h200/`: the one-H200 demo, with its subprotocols' interfaces in `interfaces/`: the frozen MQA shape served on one GPU
  node with every op's output committed a step. Its run.py is the step driver, and `phases.sh` runs Demo A's M3 proper one
  phase a run, from the step's commit on the H200 to its drawn blocks proved. Level 1 on DRAWPROVE r20261027-013625-8e25: 8
  live-drawn proof blocks are proved by C-Flock `--zk` and accepted by Lean, and the flipped bit is refused. Demo B's memory
  terms (memory-accounting's, M4) have a slot in its `example.toml` and aren't claimed.
- `toy_onsite/`: the CPU toy (`integrations/torch_loop`) through every protocol and the wipe, composed through onsite, one run
  on one CPU machine. Its run.sh is `benchmarks/all_protocols/run.sh`, which `verity bench
  all-protocols` runs. Level 1 on S1's run of record, r20261026-161826-eb34: a window with one egress byte changed is
  rejected, and the honest window is accepted over all six subprotocols. The sampled-proofs roles, each run on its own, are
  DEPLOYSP's separate toy_network example (#4928).
- `vllm/`: Llama-3.2-1B served by vLLM on one GPU through `verity_vllm`'s committing endpoint, a served window audited
  through every subprotocol. Its row is in its own `workloads/`, its placement is `placement.toml`, which `run.sh` reads, and
  its scripts reach the integration by `PYTHONPATH`; it is not a package.
- `toy_network/`: sampled proofs and the network certifier on the toy (`integrations/torch_loop`), every party on one
  CPU machine through `verity run`, its deployment configuration as `verity config new` writes it.
