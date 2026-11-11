# prove

`verity bench prove` is the one command for what a proof system's prover costs on a workload. Its code is
`verity-bench`'s (`benchmarks/harness`: `verity_bench/prove.py` holds the rules, `prove_records.py` the store side and
the table); this directory holds its runner. It reads the store's labelled records and runs on any machine that reaches
the store, with no GPU and no node.

```text
verity bench prove --system flock --workload glm53                  every block proved
verity bench prove --system spot --workload glm53 [--sampling ART]  Flock at the workload's sampling plan of record
    [--rows N]                                                      the N costliest block types, the rest in one row
    [--json]                                                        the table as JSON, every row whole
    [--local-only]                                                  this machine's catalog alone, not the store's remote
```

## What it prints

- **A row a block type** of the workload's census under its cut: the blocks of it proved per served token, the prover's
  ms per block, and where that figure is from: measured at steady state (a proof server's report), measured in a single
  session (the `proof_overhead` card), modeled from k_log (the card's types' median seconds a witness bit), or missing.
  Each row names the run or art its figure is from. A type several steady-state reports measured takes the fullest
  statement's, the newest run's among equals; a steady-state attempt that didn't stand is named on its row.
- **The totals:** the prover's GPU-seconds per served token, the tokens a prover GPU covers per second, and the prover
  GPUs a serving GPU needs, each with its range, whose other end prices every type not measured at steady state at the
  steady-state types' median seconds a witness bit. A total is called measured only when steady-state measurements
  cover all of its proving time. The verifier's seconds and the proof's bytes are given where a record holds them, and
  "not measured" where none does. The served MACs the census holds in no proof block are in no figure, and the totals
  give their share.
- **The share of the proving time each source covers.**
- **The totals per monitoring period over all compute:** with every block proved, the census's datacenter and the
  world's serving for a week, a month and a year; under a sampling plan, the compute and period the plan is sized for.

`--system flock` proves every block. `--system spot` takes the same costs at the blocks a sampling plan samples: a
system is a `prove.ProofSystem`, a census in and a table out, and SPOT wraps one with a plan.

## Inputs

Every input is a record of the store, named there by a label and never in code: what carries
`run_of_record=prove-WORKLOAD-SLOT` and no `superseded_by` (`prove_records`'s docstring has each slot's format).

| Slot | What carries the label |
| --- | --- |
| `census` | the workload's block census (`benchmarks/one_stage/block_census.py`'s art) |
| `cut` | the `partition` card that recuts it |
| `flock-steady` | each steady-state proof server run, or an art naming them in its `runs` |
| `flock-single` | the `proof_overhead` card |
| `spot-sampling` | the sampling plan's art, for `--system spot` |

```text
research data label TARGET run_of_record prove-glm53-census --by LANE     name TARGET the glm53 census of record
research data label OLD superseded_by NEW --by LANE                       retire OLD once NEW replaces it
```

An input that isn't labelled or is superseded is refused, by name and with the label that would fix it; so is a slot
two unsuperseded targets claim, and a sampling plan and a census that don't cover each other. A steady-state run that is
superseded or holds no report is left out and listed under "Refused".

The serving rate is not a record. It is a stated, modeled input: the census's MACs a served token at the fleet's assumed
serving MFU (`verity/protocols/inference_only/rho/data/reference.toml`) of the card's data-sheet peak (`benchmarks/census/hardware.json`).
