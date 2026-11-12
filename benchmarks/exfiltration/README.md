# exfiltration

`verity bench exfiltration` prints GLM-5.3's exfiltration capacity in GB over a monitoring period, at 10% of total
compute: the most bits about the weights that can leave over the world's year, as two parts side by side, with every
input labelled measured, modeled, assumed, missing or run, not proved. It computes neither part. It composes the two calculators that do, so it
needs no GPU and no node, only the evidence store. Its code is `benchmarks/harness/verity_bench/exfiltration.py`; this
directory holds its runner.

```text
verity bench exfiltration                 the figure, the inputs, each part, and the two side by side a period
verity bench exfiltration --json          the same as one document: value, unit, inputs, parts, periods
verity bench exfiltration --share 0.05    read at another share of total compute
verity bench exfiltration --served-run F  beside it, the share a served two-stage run's record F measures
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
  and compiling) and the bound in bits. The command takes row `(ii) honest`, the plan of record's row, with every
  modeled lever at its measured value, at the step that prices the verifier as measured, and reads it at the β
  where its total is the share: the total is a line in β fitted over the pinned budgets, and ln(bound) is linear in ln β
  between the two that bracket it. The row plans for the weights / λ, and its part is λ times its bound. `--row` and
  `--step` take another case of the plan.
  - **Row `(i) honest`** prices its proving at the planner's assumed prices. Beside the figure it is read over the
    world's year at the same share and printed only under the label "at the planner's assumed prices", never as the
    figure; the document holds it as `at_assumed_prices`. Read with `--row "(i) honest"`, the figure itself carries
    the label.
- **The network certifier's part** is `verity bench certifier_term`'s (`benchmarks/certifier_term/`): the schedules,
  the fail cells and the jitter, for the scan and N2 as the monitoring-period check charges them today, for the scan
  under each change to its check, and for the design with no run behind it, before a served audit has sampled a
  stop block and once one has. The command reads that command's document (`certifier-term/v2`) and refuses another
  schema.
  - **Its figures of record are at two release schedules**, D_j = 2, the jitter bound of that document's term of
    record. A figure at D_j = 1, and every figure of a row whose release times are taken once (`timing`,
    `all three`), is what a network certifier with a set-time card would be charged, and no timing test has shown
    such a card. Each is printed only under the label the document gives it, in a line, a column or a row's name
    that carries the label. The command reads the label from the document, at its top and on each row that carries
    it, and holds no copy.
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

## The stop class's verdicts

The output count's 0 in those sums stays assumed until a run backs it: the four verdicts a stop class needs, each
accepted and complete, are the stop record, the response tie, the request tie and the compute audit's second stage.
The command reads them from the store, from every run that carries
`run_of_record=exfiltration-glm53-stop-verdicts` and no `superseded_by`, in the file `out/verdicts.jsonl` of its run
files, one verdict's record a line (`VERDICT_FIELDS` in `exfiltration.py`). A record lacking a field is refused by the
field's name:

- `piece`: `stop_record`, `response_tie`, `request_tie` or `second_stage`.
- `control`: null on the honest run, or the negative control it is: `id` (a changed id) or `max_tokens` (a changed
  `max_tokens`), a run on the same path with that one thing changed.
- `run`: the id of the run that produced the verdict.
- `verifier`: `{"module": ..., "commit": ...}`, the verifier that checked it and the commit it ran at.
- `verdict`: `accept` (accepted and complete), `not complete` (accepted, which moves no figure) or `refuse`.
- `proved_by`: null until a Lean proof covers that verifier, then the Lean guarantee's name.
- `open`: while `proved_by` is null, what that proof waits on.

The class is complete when every piece's honest run is `accept`, and a changed id and a changed `max_tokens` are each
refused, on a piece, by the verifier that accepted that piece's honest run, and no control is accepted. Then the
output count's 0 is marked `run, not proved`, printed "run (Python verifier, not proved)" with each verdict's run and
verifier and counted apart from measured, and the class prints as `complete (run, not proved: OPEN open; Python
verifier for PIECES)`. Only where every piece's record names its `proved_by` is it `measured`; the command takes that
from the field, never from a flag or a name. Short of complete, the inputs print as with no verdicts and the output
count's line names what is missing. With no run carrying the record the print is as it was before the command read
any.

## The share of compute

Every figure is read at a share of total compute that is given (`--share`, 10% by default) and marked assumed. Beside
it the command prints the share a served two-stage run measures: the audit's GPU-seconds on the run over the window's
serving GPU-seconds. Without a run's record it prints that share as "not measured: no served run's record".

With `--served-run FILE`, the record as the reader of a served run's record gives it, the command reads each part of
the audit's GPU time from it and marks it measured: the sessions' proving, the staging of the sampled proof blocks,
the replay of the sampled spans, the interior commitment of their proof blocks, and the serving charge. Where the
record lacks a field, the command names it and what the run doesn't write, marks it missing, and claims no share.
Where it claims one, it prints the compute audit's part over the world's year at that share beside the part at the
given share, at which every other figure stays.

The record is one JSON object (`served_share` in `exfiltration.py`):

- `run`: the served run's id.
- `prove`, `staging`, `replay`, `interior`: each a list of what held GPUs for that part, one entry a session or a
  step, each `{"seconds": S, "gpus": G}`: the seconds it held them and how many it held.
- `serving`: one such entry, the window's load: the seconds it was served over and the GPUs its engine held.
- `stock`: one such entry, stock vLLM serving the same load at the same launch settings save Verity's changes. The
  serving charge is the served GPU-seconds less stock's, measured.
- `serving_cost` and `serving_cost_source`, where there is no `stock`: `verity bench serve_tax`'s cost of the served
  arm over stock, as a fraction (stock's output tokens a second over the arm's, less one), and the run and load it is
  of. The serving charge is then the served GPU-seconds times c / (1 + c), modeled, since those loads are not the
  window's.

`verity bench serve_tax` (`benchmarks/serve_tax/`) serves the two-stage window as its `two-stage` arm, and each load's
`meta.json` there holds the seconds it was served over and the GPUs' busy seconds (`wall_s`, `gpu`), for every arm the
job serves. So the window's serving and, where the same job served `compiled` on the same plan and rate, stock's are
read from that record, and no stock serve of this command's own is needed.

The reader is `python -m verity_bench.served_record --second DIR [--first DIR] [--serve DIR] [--replay DIR]
[--serving META] [--stock META]` (`benchmarks/harness/verity_bench/served_record.py`). A served run is four runs
(the serve, the first stage, the replay of the sampled spans, the second stage), and each `DIR` is one's directory
as it is fetched, its `job.json` beside its `out/`; only the second stage is required. It prints that object from the
files those runs published and the two loads' `meta.json`, with `None` for each field those files lack, and names
each one with the file it read and what that file holds instead. It computes no share and decides no verdict.

- A session's seconds are those its own `prove_seconds_source` names in `audit.timing`: `busy`, the proof server's
  clock on the session's prover (`busy_s`), or `session`, the audit's clock on the whole session (`session_s`), which
  counts a session's wait behind the others of its prover and so cannot read low. The record's
  `prove_seconds_source` is the sessions' own; where a session names none its seconds are `session_s` and the field is
  named as absent.
- A part that started no process on a card is written on 0 GPUs, and the command reads it as no GPU-seconds.
- Beside the share's fields the record holds `commit`, the commit the second stage's run ran, `runs`, each of the four
  runs' id, commit and machine, and `audit`, what the audit showed: the sessions' verdicts counted by kind, the
  driver's checks, the stream's outcome with its codes, the layout the serve committed, and what holds the four
  runs' files to each other.

From a record that holds `audit` the command prints one more section, before the resampling holes
(`benchmarks/harness/verity_bench/served_row.py`), read against the tree:

- **The verdict**, in the verifier's own words: how many of the sampled proofs the Lean verifier accepted, then the
  final accept as the stream's outcome gave it: given, refused with its codes, or withheld, where every check of the
  second stage held and the outcome is refused by one code the table names a reason for. An accept the verifier
  withheld is never printed as given.
- **What the audit covered**: the replay block as the layout names it, the stand-in (instances of the check's
  template) or a Definition of the class's own.
- **To fully proved**: the replay block where it is the stand-in; every verdict a verifier that is not Lean
  decided; every Lean verdict whose lock entry (`security/lean-audit.json`) is missing or lists open premises; and
  the final accept where it was not given. Which verifier decides each kind of verdict, and the lock entry that
  covers it, are data (`verity_bench/data/served_row.json`). The resampling holes and the cited theorem are the next
  block's, and this one prints no line on them.
- **The mark** is `proved` only where that list is empty, the final accept is given, the cited theorem's lock entry
  lists no open premise and each resampling hole is closed at the run's commit; `run, not proved` otherwise.
- **The prover's seconds** as the record holds them, said as "at most" where they are a session's, with the cards
  the provers ran on and what the data file notes of a card.
- **The security level at the run's own sizes**: the planner's binding bound at one verification phase of the run's
  sessions and sampled proof blocks, modeled, with what it assumes.
- **The verdict as a record** of the stop class's verdicts of record, for the piece `second_stage`.

## The security level

Beside the GB the command prints the binding bound the row of record holds: the probability that an accepted run
opens a value off the committed trace, by the bound Lean states of a two-stage sample counted by classes of gates
(`CompiledTwo.eventTwoX_le_cls_at_budget`'s right-hand side). The planner computes it for each
row's own design (`benchmarks/spot_figures/binding.py`, `over_phases`) and writes it in the row's
`verification_phase`; the command computes none of it. Each figure is named by what it is over, one verification
phase or the world's year of them, and is given counted in calls of SHA-512, as the Lean counts, and in
compressions. It is marked modeled, and the inputs it assumes are listed and are among the inputs, each marked
assumed: the classes of gates a sampled proof block, the symbols of a table's level-0 codewords, and that a prover
makes one commitment for one sample and presents every committed window. Beside each figure the command prints what
Lean proves of it at numbers: a theorem of `Claims.Flock.Numbers` by name with its bound, or "not proved at these
sizes" with the nearest instance and each numeral the row's design is past. No theorem sums a monitoring period's
verification phases. Rows written before the planner computed a bound hold none, and the command says so.

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
`run_of_record=exfiltration-glm53-compute-audit` and no `superseded_by`, and in each the files `out/rows.jsonl` and
`out/three/rows.jsonl` of its run files (the planner's driver's default command's rows, and its rows over the world's
year alone), a case's rows over a scope from the first file that holds them: where the second holds other rows of
that case and scope, the run is left out. Each file is one JSON object a line, a row of the plan at one scope (`world year`, `datacenter week`, ...) and one
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

What prices a row's proving is read from the row's own plan too: the share each source prices (a block type's own
steady-state row, its Definition's family row, the planner's price), as `split.sources` records it, else by
`split.basis` (`steady` an own row, `family` a family row, the copy stratum its own, every other key the planner's
price), and the share at the planner's assumed default (`split.basis`'s `reprice.DEFAULT`). The command prints it
beside the figure and beside row `(i) honest`, and "not recorded" for each where the row's plan records neither or a
plan of record is read.

```text
research data label RUN run_of_record exfiltration-glm53-compute-audit --by LANE     a run that holds the rows, or a plan
research data label RUN run_of_record exfiltration-glm53-stop-verdicts --by LANE     a run that holds stop verdicts
research data label RUN superseded_by NEWER --by LANE                                a run a newer one replaces
```

Several runs may hold the record, each its own scopes: the world's year is required, and the other scopes appear in
the tables as their runs are labelled. The command refuses when nothing carries the label and when two unsuperseded
runs hold the same scope's rows, or its plan, and says which. A run that is superseded, or holds neither rows nor a
plan of the case, is left out and listed under "Refused".

## What it prints

- **The figure**: over the world's year, the compute audit's part in GB with its copies of the weights, its bits a
  request and the requests it is priced over, how its proving time is priced by the row's plan and what prices it by
  source, row `(i) honest` under its label at the planner's assumed prices, then the network
  certifier's part under each of its rows at D_j = 2, then the two summed, each followed by a line under the
  network certifier's label for what rests on the set-time card, then the output count. The network certifier's
  figures end with their ε over the world's year and the δ of the scans it sums, and the sums with their error.
  Where the ε is 1 or more the network certifier's part is "no bound at this period", with the ε, in place of every
  amount of it, the output count among them; where the two parts' errors sum to 1 or more, so are the two together.
  One line gives the errors: the compute audit's δ over the monitoring period, the network certifier's ε with the
  scans' δ and the execution phases it sums, and their sum. One more line gives the compute audit's part with the
  network check's own proving taken off the audit budget (`--network-check-share`, assumed: no run measures it and
  the planner's rows don't count it).
- **Inputs**: every input's part, value, unit, mark and source. A mark is `measured` (a run measured it), `modeled`
  (derived from other figures), `assumed`, `missing` (a field a served run's record lacks, with its gap), or `run, not
  proved` (the stop class's verdicts, a run's, checked by a verifier no Lean proof covers yet). The compute audit's come from its rows of record: δ, λ and the most
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
- **SPOT's resampling holes in the code that ran**, over the compute audit's runs of record: the label (`SPOT`, or
  `SPOT soundness` only when both holes are closed), each hole's status, and the theorem with its open premises as
  the lock lists them (`verity_bench.spot_status`; `spot_holes` in the JSON).

With `--json` the document holds the same: a scope's monitoring period has its `eps`, its `error` (the two together's,
null for a scope no run of record holds) and its `phases`, and an amount that has no bound is null.
