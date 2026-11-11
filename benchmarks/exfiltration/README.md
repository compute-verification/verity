# exfiltration

`verity bench exfiltration` prints GLM-5.3's exfiltration capacity in GB over a monitoring period, at 10% of total
compute: the most bits about the weights that can leave over the world's year, as two parts side by side, with every
input labelled measured, modeled or assumed. It computes neither part. It composes the two calculators that do, so it
needs no GPU and no node, only the evidence store. Its code is `benchmarks/harness/verity_bench/exfiltration.py`; this
directory holds its runner.

```text
verity bench exfiltration                 the figure, the inputs, each part, and the two side by side a period
verity bench exfiltration --json          the same as one document: value, unit, inputs, parts, periods
verity bench exfiltration --share 0.05    read at another share of total compute
verity bench exfiltration --local-only    read this machine's catalog alone, not the store's remote
verity bench exfiltration --help          every parameter and its default
```

## The two parts

- **The compute audit's part** is the break-even planner's bound (`benchmarks/spot_figures/breakeven/`): what the
  wrong blocks of a sampled audit can carry at δ. The planner prices one case of its plan over one scope at a pinned
  audit budget β: the share of total compute it costs (serving with its wait, proving, the replay, the pool, the edge
  and compiling) and the bound in bits. The command takes the row for the registered and served cut with every
  modeled lever at its measured value, at the step that prices the verifier as measured, and reads it at the β
  where its total is the share: the total is a line in β fitted over the pinned budgets, and ln(bound) is linear in ln β
  between the two that bracket it. The row plans for the weights / λ, and its part is λ times its bound. `--row` and
  `--step` take another case of the plan.
- **The network certifier's part** is `verity bench certifier_term`'s (`benchmarks/certifier_term/`): the schedules,
  the fail cells and the jitter, for the scan and N2 as the monitoring-period check charges them today, for the scan
  under each change to its check, and for the design with no run behind it.

A link's execution phase carries K = schedules + cells + jitter + min(entries, advice + the wrong blocks' output bits
at δ). The first three terms are the network certifier's part and the last term's second branch is the compute
audit's, so the command also prints their sum under the network certifier's rows: the scan and N2 as charged today,
the scan with all three changes, and the design. Where a row charges the entries and they are fewer than the compute
audit's part, they stand in its place. The command prices no advice, and says so among its inputs.

Both parts count the same requests: the planner's, of the plan its rows are priced on, which the command reads from
the planner in the tree and hands to the network certifier's calculator as the world's requests a year. They are
fewer than the census's, which `verity bench certifier_term` counts by default: a request of the plan has fewer
computed tokens at more served MACs a token. The command refuses if the two parts don't cover the same GPU-seconds
or aren't shares of the same weights.

## The records it reads

The planner's rows are a labelled record of the store, never named in code. The command reads every run that carries
`run_of_record=exfiltration-glm53-compute-audit` and no `superseded_by`, and in each the file `out/rows.jsonl` of its
run files: one JSON object a line, a row of the plan at one scope (`world year`, `datacenter week`, ...) and one
pinned budget, with its `world`, `row`, `step`, `scope`, `b10` (the budget), `total` and its parts, `bound_bits`,
`weights_bits`, `target_bits`, `delta_seen`, `S`, `gpu_seconds` and its proving time's `split` by block family.

```text
research data label RUN run_of_record exfiltration-glm53-compute-audit --by LANE     a run that holds the rows
research data label RUN superseded_by NEWER --by LANE                                a run a newer one replaces
```

Several runs may hold the record, each its own scopes: the world's year is required, and the other scopes appear in
the tables as their runs are labelled. The command refuses when nothing carries the label and when two unsuperseded
runs hold the same scope, and says which. A run that is superseded, or holds no rows of the case, is left out and
listed under "Refused".

## What it prints

- **The figure**: over the world's year, the compute audit's part in GB with its copies of the weights and its bits a
  request, then the network certifier's part under each of its rows, then the two summed. Beside them are the
  errors: the compute audit's δ over the monitoring period, and the scan's ε, its execution phases' δ summed, which
  bounds nothing at 1 or more. One more line gives the compute audit's part with the network check's own proving
  taken off the audit budget (`--network-check-share`, assumed: no run measures it and the planner's rows don't
  count it).
- **Inputs**: every input's part, value, unit, mark and source. A mark is `measured` (a run measured it), `modeled`
  (derived from other figures) or `assumed`. The compute audit's come from its rows of record: the share of proving
  time each way of pricing a block family covers (measured statements, computed, or both), δ, λ and the most
  bits a request's output holds. The network certifier's are its calculator's own marks. The fleet, a request of the plan and the
  weights are read by both.
- **The compute audit's part**: each scope of record, with its run, its β at the share, its bound at λ = 1 and at the
  row's λ, and each part of the total there. A β outside the pinned budgets is marked extrapolated.
- **The network certifier's part over the world's year**: each of its rows' schedules, cells, term at D_j = 1 and
  D_j = 2, and the entries.
- **Each monitoring period over all compute**: a week, a month and a year of the reference datacenter and of the
  world, with the compute audit's part as a pooled total and in bits a request, the network certifier's term at
  D_j = 1 under four of its rows, and the two together.
