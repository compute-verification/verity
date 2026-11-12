# exfiltration

`verity bench exfiltration` prints GLM-5.3's exfiltration capacity in GB over a monitoring period, at 10% of total
compute: the most bits about the weights that can leave over the world's year, as two parts side by side, with every
input labelled measured, modeled, simulated or assumed. It computes neither part. It composes the two calculators that
do, so it needs no GPU and no node, only the evidence store. Its code is
`benchmarks/harness/verity_bench/exfiltration.py`; this directory holds its runner.

```text
verity bench exfiltration                 the figure, the inputs, each part, and the two side by side a period
verity bench exfiltration --json          the same as one document: value, unit, inputs, parts, periods
verity bench exfiltration --share 0.05    read at another share of total compute
verity bench exfiltration --variant V     read one variant of the row, in a run that holds several
verity bench exfiltration --response-ids N the network certifier's part for a class of at most N ids a response
verity bench exfiltration --scan-delta D  the network certifier's errors at another δ of its scans (a fraction)
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
  under each change to its check, and for the design with no run behind it, before a served audit has sampled a
  stop block and once one has. The design's two rows are also charged the count: how many of a link's releases
  each content class holds in an execution phase. The command reads that command's document (`certifier-term/v2`)
  and refuses another schema.
  - **Its figures of record are at two release schedules**, D_j = 2, the jitter bound of that document's term of
    record. A figure at D_j = 1, and every figure of a row whose release times are taken once (`timing`,
    `all three`), is what a network certifier with a set-time card would be charged. That card is simulated: none
    is deployed, and it is an input of the document, `set_time_card`, the one input marked `simulated`. Each such
    figure is printed only under the label the document gives it, "with a simulated set-time card", in a line, a
    column or a row's name that carries the label. The command reads the label from the document, at its top and
    on each row that carries it, and holds no copy.
  - **Its class is the one the plan of record serves**: at most 1,024 ids a response (`--response-ids`), assumed, as
    every class is while no run has certified one. `verity bench certifier_term` assumes a wider class by default,
    so the headline gives that command's own term of record beside this one's.

A link's execution phase carries K = schedules + cells + jitter + min(entries, advice + the wrong blocks' output bits
at δ). The first three terms are the network certifier's part and the last term's second branch is the compute
audit's, so the command also prints their sum under the network certifier's rows: the scan and N2 as charged today,
the scan with all three changes, and the design, before its stop block and with it. Where a row charges the entries
and they are fewer than the compute audit's part, they stand in its place. The command prices no advice, and says
so among its inputs.

The sums under the rows whose schedules are 0 (`all three`, `design`) take a response's output count at 0: a stop
block, sampled on a served audit, holds that count to the response's ids. Until a served audit has sampled one the
server chooses the count, and the network certifier's row `design, no stop block` charges it, as its schedules.
The command reads the output count from that row and prints it beside those sums, marked assumed, in every
monitoring period and where it says no advice is priced.

## The errors

Every figure carries the error summed over everything it covers. A figure whose error is 1 or more is printed as
"no bound at this period", with that error and no amount: a union bound of 1 or more bounds nothing.

- **The compute audit's part** fails with at most its rows' δ over the monitoring period (`delta_seen`).
- **The network certifier's part** fails with at most ε, which its document gives each row (`eps`): the δ of each
  execution phase's scan, summed over every link of the scope and every execution phase of the monitoring period. A
  scope's monitoring period has one ε, whichever row charges it, and the command refuses a document whose rows of
  one carry two. So a link's week can carry an amount where the world's year carries none, from the same scans.
- **The two together** fail with at most the sum of both parts' errors.

The scans' δ is an input of the network certifier's calculator, and this command gives it one. By default it is the δ
at which the row printed that sums the most execution phases, the world's year, carries an error of 2^-20: 2^-20
divided by that row's execution phases, which the calculator's own document counts. Every row printed then carries
that error or less. `verity bench certifier_term`'s result of record takes its δ the same way
(`benchmarks/certifier_term/README.md`). `--scan-delta D` gives another. Either is marked assumed among the inputs,
with how it was chosen. No term depends on the scans' δ, so an amount that is printed is the same at any δ; only the
errors move, and with them which rows are printed at all. The δ printed beside a row is the one the calculator's
document states among its inputs, never a copy the command keeps.

## The requests

Both parts count the same requests: the sampling plan's, which the network certifier's calculator counts by default
and reads from the planner in the tree. The command holds no count of its own, so its inputs list that calculator's
request count and what its request is, as the calculator marks them. The requests one of the planner's rows is
priced over are its `replay_sample / q`, the sampled replays over the probability a request is sampled; the row's `requests`
field is the planner's default scope's count, another request's, and is not read. The command refuses if the two
parts don't cover the same GPU-seconds, if the rows of record are priced over other requests than the network
certifier's part counts, or if the two aren't shares of the same weights.

## The records it reads

The planner's rows are a labelled record of the store, never named in code. The command reads every run that carries
`run_of_record=exfiltration-glm53-compute-audit` and no `superseded_by`, and in each the file `out/rows.jsonl` of its
run files: one JSON object a line, a row of the plan at one scope (`world year`, `datacenter week`, ...) and one
pinned budget, with its `world`, `row`, `step`, `scope`, `b10` (the budget), `total` and its parts, `bound_bits`,
`weights_bits`, `target_bits`, `delta_seen`, `S`, `gpu_seconds`, `q` and `replay_sample`, and its `split`: each block
family's share of proving and how it is priced (`split.families`, `share` and `priced`). A run holds one variant of a
row: where the planner's driver wrote several (`h8`), the command refuses to mix them, and `--variant` reads the one
it names.

How a row's proving is priced is read from the row's plan, each family priced as the rows say (a family they don't
name is priced in no stated way). The command finds the plan in one of two places:

- **The rows' own.** A row's `split` is its own plan where it says the `q` and the `replay_sample` it is the plan of
  (`split.q`, `split.replay_sample`) and they are the row's. The command reads the row at the pinned budget nearest
  the β it reads the bound at.
- **A plan of record**, `out/mix.json` in a labelled run: the sampling plan the row is priced over at one audit budget
  (`row`, `step`, `world`, `scope`, `beta`, `q`, `replay_sample`, `bound_bits`), with each cell's `block` and its
  `share` of the plan's proving time (`cells`, `left_out`). It is read before the rows' own. A plan must be its rows'
  plan: priced over the same requests, with the bound the rows give at its budget. The row a plan's run holds at the
  plan's budget is the plan's own and is not read off.

A `split` that doesn't say whose plan it is, is not one: it is written before the row's levers multiply its blocks,
and its shares are not read. With no plan in either place the command says so and marks no proving measured.

```text
research data label RUN run_of_record exfiltration-glm53-compute-audit --by LANE     a run that holds the rows, or a plan
research data label RUN superseded_by NEWER --by LANE                                a run a newer one replaces
```

Several runs may hold the record, each its own scopes: the world's year is required, and the other scopes appear in
the tables as their runs are labelled. The command refuses when nothing carries the label and when two unsuperseded
runs hold the same scope's rows, or its plan, and says which. A run that is superseded, or holds neither rows nor a
plan of the case, is left out and listed under "Refused".

## What it prints

- **The figure**: over the world's year, the compute audit's part in GB with its copies of the weights, its bits a
  request and the requests it is priced over, how its proving time is priced by the row's plan, then the network
  certifier's part under each of its rows at D_j = 2, then the two summed, each followed by a line under the
  network certifier's label for what rests on the simulated card, then the output count. The network certifier's
  figures end with their ε over the world's year and the δ of the scans it sums, and the sums with their error.
  Where the ε is 1 or more the network certifier's part is "no bound at this period", with the ε, in place of every
  amount of it, the output count among them; where the two parts' errors sum to 1 or more, so are the two together.
  One line gives the errors: the compute audit's δ over the monitoring period, the network certifier's ε with the
  scans' δ and the execution phases it sums, and their sum. One more line gives the compute audit's part with the
  network check's own proving taken off the audit budget (`--network-check-share`, assumed: no run measures it and
  the planner's rows don't count it).
- **Inputs**: every input's part, value, unit, mark and source. A mark is `measured` (a run measured it), `modeled`
  (derived from other figures), `simulated` (what a stand-in does in its place: the network certifier's set-time
  card, the one such input) or `assumed`. The compute audit's come from its rows of record: δ, λ and the most
  bits a request's output holds; and from the row's plan, where it is read: the share of proving time each way of
  pricing a block family covers (measured statements, computed, or both). The network certifier's are its
  calculator's own marks. The fleet, the request counted, a request of the plan and the weights are read by both.
  The advice and the output count are listed at 0, each with what it leaves out.
- **The compute audit's part**: each scope of record, with its run, its β at the share, its bound at λ = 1 and at the
  row's λ, the requests its rows are priced over, and each part of the total there. A β outside the pinned budgets is
  marked extrapolated. Under it, for each scope whose plan is read, its proving time by how its block families are
  priced, with the plan and its budget.
- **The network certifier's part over the world's year**: each of its rows' schedules, cells, term at D_j = 2 with
  its multiple of the weights, term at D_j = 1 in a column whose header carries the label, and the entries, then the
  δ of its scans and its ε. A row whose release times are taken once carries the label in its name. A row whose ε is
  1 or more has "no bound at this period" in place of each amount.
- **Each monitoring period over all compute**: a week, a month and a year of the reference datacenter and of the
  world, with the compute audit's part as a pooled total and in bits a request, the network certifier's term of
  record under the scan, N2 and the design before its stop block and with it, the two together, and the output
  count. Each part's amounts are followed by its error over that scope and monitoring period: the compute audit's
  δ, the scans' δ and the network certifier's ε, and the error of the two together. A figure whose error is 1 or more
  is "no bound at this period".
- **Each monitoring period over all compute, under the label**: the same scopes, with the network certifier's term
  and the two together under `all three`, and under each other row at D_j = 1. Every amount's column carries the
  label in its header, and no amount of the table is of record. Each line ends with its errors, the table above's.

With `--json` the document holds the same: a scope's monitoring period has its `eps`, its `error` (the two together's,
null for a scope no run of record holds) and its `phases`, and an amount that has no bound is null.
