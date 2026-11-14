# exfiltration

`verity bench exfiltration` prints GLM-5.3's exfiltration capacity in GB over a monitoring period, at 10% of total
compute: the most bits about the weights that can leave over the world's year, as two parts side by side, with every
input labelled measured, modeled, simulated, assumed, missing or run, not proved. It computes neither part. It
composes the two calculators that do, so it needs no GPU and no node, only the evidence store. Its code is
`benchmarks/harness/verity_bench/exfiltration.py`; this directory holds its runner.

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
  `--step` take another case of the plan. This is the headline figure. The planner's driver reads the same rows two
  other ways, and the print says how each differs: `headline plan` gives the bound at the plan's own pinned budget,
  whose total is that row's rather than the share, at λ = 1; `headline rows` gives λ times the bound at 10%, the
  total read between the two pinned budgets that bracket it alone.
  - **Row `(i) honest`**, the unfused tokens line, prices its proving at the planner's prices but where a table's
    rows price its blocks: at `verity bench prove`'s table, under half of it. Beside the figure it is read over the
    world's year at the same share and printed only under its label, never as the figure; the document holds it as
    `at_assumed_prices`. Its label is what its own rows say of their pricing: the note the planner's driver writes on
    them (no table's row priced their proving, or they priced under half of it, with the share each source prices),
    else the share the table's rows price, else "at the planner's assumed prices" for rows that record no table's
    row. Read with `--row "(i) honest"`, the figure itself carries the label.
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
- `verdict`: `accept` (accepted and complete), `not complete` (accepted, which moves no figure), `refuse`, or
  `withheld`: the verifier held back its last accept itself because the form it would accept is not yet proved (every
  SPOT proof accepted, every check before the sample held, every tampered input refused). `withheld` is neither an
  accept nor a refusal.
- `proved_by`: null until a Lean proof covers that verifier, then the Lean guarantee's name.
- `open`: while `proved_by` is null, and on every `withheld` verdict, what stands between the verdict and proved,
  stated in words (`the two-round form's soundness is not proved`). A record whose `open` names an issue's id is
  refused: the print carries no issue id, and the issue goes on the board.

An honest verdict (`control` null) also names the class it is on, in six top-level fields (`CLASS_FIELDS`). One it
lacks keeps the class short of complete and is named; one that holds what it can't is refused by its name:

- `registration`: the digest of the class's boundary record, the same on all four.
- `stream_registration`: the digest of the verdict's own stream's registration.
- `stream_names_boundary`: true or false, the verifier's recorded check that that registration names the boundary.
  The command reads it and never recomputes it.
- `serve`: the store id of the served run the class came from.
- `class`: `w<window>/p<P>-o<O>`, with `-nostop` for a class with no stop block.
- `rate`: `NUM/DEN` in lowest terms, from the registration record; every stop and tie block proved is `1/1`.

A control's are not read: a control changes a committed value, so its registration differs. It is held to its piece
and its verifier (module and commit) alone.

The class is complete when every piece's honest run is `accept`, the four name one class (each with its
`stream_names_boundary` true, all four agreeing on `registration`, `serve`, `class` and `rate`) at the rate the row
assumes, `1/1`, and a changed id and a changed `max_tokens` are each
refused, on a piece, by the verifier that accepted that piece's honest run, and no control is accepted. Where they
don't name one class the print says which verdict lacks a field or records a failed check, or which field the four
disagree on with each value's pieces, and claims nothing. Where the class is complete the
output count's 0 is marked `run, not proved`, printed "run (Python verifier, not proved)" with each verdict's run and
verifier and counted apart from measured, its source naming the boundary's registration, and the class prints as
`complete on class C of serve S at rate 1/1 (run, not proved: OPEN; Python verifier for PIECES)`, OPEN being each
unproved piece's `open`. Only where every piece's record names its `proved_by` is it `measured`; the command takes that
from the field, never from a flag or a name. Short of complete, the inputs print as with no verdicts and the output
count's line names what is missing; a `withheld` piece or control prints as `withheld: OPEN`, and a withheld control
is not refused. A control on a withheld piece refused by that piece's verifier (the same module and commit) counts as
refused. With no run carrying the record the print is as it was before the command read
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
- `commit`: the commit its second stage ran; `runs`: each of its four runs (`serve`, `first`, `replay`, `second`)
  as `{"run", "commit", "dirty", "why", "machine"}`, null for a run whose directory the reader wasn't given. The
  resampling holes are read at these commits (below).
- `prove`, `staging`, `replay`, `interior`: each a list of what held GPUs for that part, one entry a session or a
  step, each `{"seconds": S, "gpus": G}`: the seconds it held them and how many it held. A step that started no
  process on a card is on 0 GPUs and adds no GPU-seconds; a part whose entries are all on 0 GPUs prints "on CPUs: no
  GPU-seconds" beside its 0. A count under 0 or not a whole number is refused.
- `prove_seconds_source`: which seconds `prove`'s are. `busy`, the proof server's busy seconds a card: the proving is
  measured. `session`, a session's seconds with its wait behind the others of its prover included: the proving and
  the share that sums it print `measured, at most`, an upper bound. Absent, `prove` lacks it and no share is claimed;
  any other value is refused. Either way a part's GPU-seconds are the seconds held times the GPUs held.
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
family's share of proving and how it is priced (`split.families`, `share`, `priced` and, where the family is priced
several ways, its share by way, `ways`, which the command splits it by). A way is measured only where it is the
planner's measured price or a block type's own steady-state row (`steady`); a family's row and the Lean verifier's
seconds, which no row times, are modeled. A run holds one variant of a
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
steady-state row, its Definition's family row, the planner's price, the Lean verifier's seconds apart from the row
they sit on), as `split.sources` records it, else by
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
  source, row `(i) honest` under its label, what its rows say of their pricing, then the network
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
  card, the one such input), `assumed`, `missing` (a field a served run's record lacks, with its gap), `measured, at
  most` or `modeled, at most` (an upper bound: a served run's proving read from its sessions' seconds), or `run, not
  proved` (the stop class's verdicts, a run's, checked by a verifier no Lean proof covers yet). The compute audit's
  come from its rows of record: δ, λ and the most
  bits a request's output holds; and from the row's plan, where it is read: the share of proving time each way of
  pricing a block family covers (measured statements, computed, or both). The network certifier's are its
  calculator's own marks. The fleet, the request counted, a request of the plan and the weights are read by both.
  The advice and the output count are listed at 0, each with what it leaves out.
- **The compute audit's part**: each scope of record, with its run, its β at the share, its bound at λ = 1 and at the
  row's λ, the requests its rows are priced over, and each part of the total there. A β outside the pinned budgets is
  marked extrapolated. Under it, for each scope whose plan is read, its proving time by how its block families are
  priced, with the plan and its budget.
- **The network certifier's part over the world's year**: a legend line a row, with its standing and, where the
  calculator gives the row one, its mark beside it (the `scan` row's accept, `run, not proved`); a row's mark the
  command has no name for is refused. Then each of its rows' schedules, cells, term at D_j = 2 with
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
- **SPOT's resampling holes in the code that ran**, over the compute audit's runs of record, or with `--served-run`
  over the served run's own code instead: its record's `commit` and each of its runs' commits, each listed, a hole
  closed only where its fix is an ancestor of every one of them. Where any of the four runs is not given, such a
  hole prints "closed at the runs read", naming the runs read and those not given, and the label stays `SPOT`. The
  block holds the label (`SPOT`, or
  `SPOT soundness` only when both holes are closed), each hole's status, and the theorem with its open premises as
  the lock lists them (`verity_bench.spot_status`; `spot_holes` in the JSON).

With `--json` the document holds the same: a scope's monitoring period has its `eps`, its `error` (the two together's,
null for a scope no run of record holds) and its `phases`, and an amount that has no bound is null.

## Under a research run

`run.py`'s `METRICS` are the rows of `benchmarks/exfiltration/metrics.toml` this benchmark is held to. Under a research
run (`RESEARCH_RUN_DIR` set) the command writes them into the run's `result.json` through the harness kit, each at the
share over each monitoring period of the document it has a value over (a week, a month and a year of a datacenter's
and of the world's compute), with its `period` and `over` (`year` and `world` for the world's year), in GB a world
year at 10% of compute and so on for each period. A period it has no value over (one no run of record holds, or where
the network certifier's ε is 1 or more) is left out, but the world's year, the figure each metric's title names, which
is always written and written last, so a reader that keys a result's measurements by name alone, the last of a name
winning, reads it:

- `benchmarks/exfiltration-gb-world-year`: the headline figure, the document's `value` over the world's year (today
  the compute audit's part, λ times the bound where row `(ii) honest`'s total is 10% of compute), and that part over
  each other period.
- `benchmarks/exfiltration-compute-audit-gb`: the compute audit's part over each period.
- `benchmarks/exfiltration-network-certifier-gb`: the network certifier's part of record, the `scan` row at D_j = 2;
  over the world's year with no value and "no bound at this period" with its ε as its why where that ε is 1 or more.

Each measurement's first input opens
with its mark (`modeled`, and the network certifier's with the mark its calculator gives the `scan` row beside it), its
second is its error over its period (the compute audit's δ, or the network certifier's ε at its scans' δ), and the
rest are the runs of record it read. The document is kept as
`out/exfiltration.json`, the run's measurement file and its validation's evidence, and the fingerprint holds what
ran. The print is the same in a run as outside one.

```text
research run --on auto --project verity --source . --cwd source --store-read -- uv run python benchmarks/exfiltration/run.py
```
