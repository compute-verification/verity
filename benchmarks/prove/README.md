# prove

`verity bench prove` is the one command for what a proof system's prover costs on a workload. Its code is
`verity-bench`'s (`benchmarks/harness`: `verity_bench/prove.py` holds the rules, `prove_records.py` the store side and
the table); this directory holds its runner. It reads the store's labelled records and runs on any machine that reaches
the store, with no GPU and no node.

```text
verity bench prove --system flock --workload glm53                  every block proved
verity bench prove --system spot --workload glm53 [--sampling ART]  Flock at the workload's sampling plan of record
    [--provers N]                                                   the prover count each type is priced at (default 1)
    [--rows N]                                                      the N costliest block types, the rest in one row
    [--json]                                                        the table as JSON, every row whole
    [--local-only]                                                  this machine's catalog alone, not the store's remote
```

## What it prints

- **A row a block type** of the workload's census under its cut: the blocks of it proved per served token, the prover's
  ms per block, and where that figure is from: measured at steady state (a row of the proof server's rate), steady
  state, its family's (no row of its own, so the dearest steady-state row of its Definition and K at other statics, at
  2^Δk its seconds a block; priced, not measured, and counted in no steady-state share), measured in a single session
  (the `proof_overhead` card), modeled from k_log (the card's types' median seconds a witness bit), or
  missing. Each row names the run or art its figure is from, and a steady-state row its prover count.
- **A type is priced at a stated prover count**, `--provers N`, one by default: only the steady-state rows at N provers
  measure it, so its figure doesn't depend on which counts were run or in what order. Among those rows it takes the
  fullest statement's, then the newest run's. A row at N provers that didn't stand is named on its type's row with
  its reason, and so are the other counts that measured the type.
- **The totals:** the prover GPU-seconds per served token, the tokens a prover GPU covers per second, and the prover
  GPUs a serving GPU needs, each with its range, whose other end prices every type not measured at steady state at the
  steady-state types' median seconds a witness bit. A total is called measured only when every block type is measured
  at steady state. When a block type has no figure the totals leave its proving out and are called partial, naming each
  such type with its share of the blocks, and of the MACs where the census holds them (a type with no time has no
  share of the proving time). Otherwise a total is modeled. The verifier's seconds and the proof's bytes are given
  where a record holds them, and "not measured" where none does. The served MACs the census holds in no proof block
  are in no figure, and the totals give their share.
- **Under a sampling plan that states its own prices, both totals.** A plan's sample is sized by its planner at the
  planner's price of each block, which the plan's record may state (`prove_s` a cell: its seconds for the blocks the
  cell takes). The table prices the same blocks from its own sources, so the two totals differ wherever those prices
  do. The command prints the prover GPUs per serving GPU at each, the plan's ms per block beside the table's on every
  row, each source's proving time at the table's prices over the plan's, and the block type that holds most of the
  difference. The plan's prices are its record's: they are no measurement the table reads, and they hold whatever its
  planner counts in a block's price. A plan that prices only some of its cells is refused.
- **The share of the proving time each source covers.**
- **The steady-state rows it didn't read,** under "Unread": a labelled row whose staged Definition is no block type
  of the census under its cut, at any prover count. Each is named with what holds it, the type and Definition it
  staged, and its own ms per block where it is a steady-state measurement, else why it isn't one. It is on no row and
  in no figure, the range's other end included: a census that no longer holds a type leaves its measurements listed,
  not dropped.
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
| `flock-steady` | each run of `benchmarks/one_stage/server_rate.py`, an art naming them in its `runs`, or an art holding one run's rows as its `rate.json` |
| `flock-single` | the `proof_overhead` card |
| `spot-sampling` | the sampling plan's art, for `--system spot` |

```text
research data label TARGET run_of_record prove-glm53-census --by LANE     name TARGET the glm53 census of record
research data label OLD superseded_by NEW --by LANE                       retire OLD once NEW replaces it
```

An input that isn't labelled or is superseded is refused, by name and with the label that would fix it; so is a slot
two unsuperseded targets claim, and a sampling plan and a census that don't cover each other. A steady-state run or
art that is superseded, a run that holds neither `out/rate/rate.json` nor a proof server report and an art that names
no run are left out and listed under "Refused".

## A steady-state row

A steady-state run's format is `out/rate/rate.json` as `benchmarks/one_stage/server_rate.py` writes it: a list of
rows, one a staged type and prover count. The command reads, by the script's names:

| Field | What it is |
| --- | --- |
| `stage.definition`, `stage.k_log` | the staged Definition's id, matched to the census's block types, and log2 of a block's witness bits |
| `stage.blocks`, `stage.full` | the blocks in the statement proved, and in the type's full statement |
| `provers`, `gpus` | the provers that proved it, and the GPUs they held |
| `statements_per_s` | statements a second over the steady window |
| `warm`, `sessions`, `accepted`, `refused_n` | the warm sessions a prover and the timed sessions, those accepted and those refused |
| `exits`, `void` | each process's exit code, and why the row has no rate (only on a row that has none) |
| `flat` | whether the statement cache's hit rate was flat |
| `verify_s` | the verifier's seconds a statement (optional: "not measured" without it) |

A statement's GPU-seconds are `gpus / statements_per_s`, and a block's ms are that over `stage.blocks`, times 1000. A
row is a steady-state measurement when every session it ran was accepted (`accepted` is `provers` times `warm` plus
`sessions`), none was refused, it isn't void, every exit is 0 and `flat` is true. One that isn't, or that lacks a
field, prices nothing and is named with its reason (the field, by name) on its block type's row. A row with no
`stage.definition`, one that failed or was skipped, is listed under "Refused".

A run's rows completed or corrected after the run go in an art that holds them as `rate.json`, the same list of rows,
with the run in its meta's `runs`. Labelled, the art gives those rows in place of the run's own: they are ordered as
that run's, each type's row names the art as its record and the run that measured it, and the run's files aren't
read. Such an art holds one run's rows: one that names several runs is listed under "Refused". An art that holds no
`rate.json` stands for the runs it names, each read from its own files.

A run from before `server_rate.py` holds no `rate.json`, only its proof server reports: each `*/server-kK/server.json`
with the `types.json` beside it. Such a run is read from those, each staged type of a report as its row at K provers,
a steady-state measurement when the report wasn't stopped or timed out, every prover exited 0, every session was
accepted and its cache was flat. A run that holds `rate.json` is read from it alone.

The serving rate is not a record. It is a stated, modeled input: the census's MACs a served token at the fleet's assumed
serving MFU (`verity/protocols/inference_only/rho/data/reference.toml`) of the card's data-sheet peak (`benchmarks/census/hardware.json`).
