# certifier_term

`verity bench certifier_term` prints the network certifier's term for GLM-5.3: the bits its accepted transcripts can
carry over a monitoring period, at the load of a node, the census's reference datacenter and the world, in bits and
GB, beside a target and one copy of the weights. Each row counts all its scope serves by its check's own rule, as
the monitoring-period check charges it (`verity/protocols/certifier/verifier/lean/NetworkCheck/`, `rowsTerms` and `terms`).
Its code is the calculator of K, `benchmarks/network_traces/leak_bounds.py` (`table`, `document`, `render`); this
directory holds its runner, which with `--out` also writes the term of record's row as a result (The result, below).

```text
verity bench certifier_term                              the inputs, the scopes, and the terms in GB and in bits
verity bench certifier_term --json                       the same as one document: value, unit, commit, run, inputs, rows
verity bench certifier_term --scope world --period year  only those rows (each flag repeats)
verity bench certifier_term --requests census            the census's request in place of the sampling plan's
verity bench certifier_term --out DIR                    also its result, for DIR: the amounts of the term of record's row
verity bench certifier_term --out DIR --registration F   with its capacity and verified, read from the registration log F
verity bench certifier_term --response-ids 1024 --scan-delta D --out DIR   the result of record (The result, below)
verity bench certifier_term --help                       every parameter and its default
```

It reads files and computes, so it needs no GPU and no run. Under `research run`, give the document its run and
commit: `--run "$RESEARCH_RUN_ID" --commit "$RESEARCH_SOURCE_SHA"`. Without `--commit` it names the checkout's HEAD.

The document's `value` is the term of record: the scan's term over the world's year, charged at two release
schedules (D_j = 2), in GB. The term at D_j = 1 is what a network certifier with a set-time card, a network card
that releases at a set time, would be charged. That card is simulated: none is deployed, and the network
certifier's software pacer stands in for it. So a figure at D_j = 1 is never the term of record and is printed only
under the label "with a simulated set-time card": the document's `with_set_time_card` holds the scan's term over
the world's year at D_j = 1, its unit, that label and its mark, `simulated`. A row whose release times are taken
once (`timing`, `all three`) has a jitter of 0 at either D_j, which rests on the same card, so every figure of it is
under the label: the row's own `with_set_time_card` is the label, where every other row's is null, and the tables
print the row's name with the label.

The card is an input of the document, `set_time_card`, the one input marked `simulated`. Its source is what a
figure under the label rests on, a clause of `trusted-host`: the network certifier's card releases each bucket at
the tick its record names, whatever the host and the server send it. The clause holds only of a calibrated hardware
card; no such card is deployed, and the software pacer that stands in for it does not meet it.

## The result

With `--out DIR` the runner also writes its result, through the harness kit's writer for `DIR` (`verity_bench.kit`)
and through nothing else. The kit says where the file is, `DIR/result.json` outside a research run
(`benchmarks/harness/README.md`). The result holds the amounts every resource's benchmark reports, for the row the
term of record is read from, the `scan` row at D_j = 2, at each scope and monitoring period the command prints. What
it prints doesn't change. Every measurement is the model's and its name says so; none is computed from a run's K.

- `certifier_term/modeled/metric`: the term of record, the document's `value`, in GB, against the document's target.
  Its verdict, `within target` or `over target`, is the document's multiple of the target in words. Its row is the
  scan's over the world's year, whichever rows the command prints.
- `certifier_term/modeled/simulated/metric`: the card's figure, the same row's term at D_j = 1, which the document
  gives as `with_set_time_card`, in GB, against the same target. It is no term of record. Its name says `simulated`,
  the card's mark, and its `inputs` open with the row it is of and the card as the document states it, whose source
  is the clause the figure rests on.
- `certifier_term/modeled/<scope>/<period>/free`: the term K, in bits. The check bounds the free amount from above,
  so this is the network's amount of record. It stands against the same target, in bits.
- `certifier_term/modeled/<scope>/<period>/capacity`: the bits the scope's certified boundary links carry out over
  the period, the scope's machines × a registered uplink's `beta` × the period's ticks. `beta` is bits a tick each
  way (`verity.hardware.site`) and the term is egress, so one direction counts. The uplink is read from a
  registration log, `--registration FILE` (`verity.hardware.registration`'s committed log): the boundary link of its
  one host that a network certifier splits. The log is read as given, and one that registers anything but the
  model's machine, one host behind one network certifier, is refused. Every machine of the scope is taken as
  registered like that one, which is the model's statement and no registration's. With no `--registration` the
  capacity has no value and its `inputs` say why: it is read from a registered link or not at all, never copied
  from a data sheet, a schedule or the datacenter model.
- `certifier_term/modeled/<scope>/<period>/verified`: capacity less free, in bits, where both have a value. It
  isn't clamped, so it is negative where the term is more than the links carry.
- `certifier_term/modeled/<scope>/<period>/error`: the row's ε, a probability, which the calculator's document holds
  as `eps`. It is a union bound over the period's execution phases, their number × the scan's δ (`--scan-delta`).

A row whose error is 1 or more has no bound at its period, since a union bound of 1 bounds nothing. Its `free` and
`verified` are written with no value and the reason, `no bound at this period`, in their `inputs`, and so are
`metric` and the card's figure where the row is the term of record's. Its `error` is written with its value, and its
`capacity`, which no error enters. What the command prints is the calculator's still, those rows' terms among it.

A measurement's `inputs` name what it was computed from: its row, the document's inputs with their values, marks and
sources, the registration read, and the measurements it is a difference of. Every one names the class the term is at,
the most ids a response holds, and the scan's δ its error is at, as the document states them: terms at two classes
don't compare, and neither do errors at two δ. A δ the command gives is named as given and assumed, since no scan
has run at it.

The command of record passes its class and its scan's δ, whatever the calculator's defaults are: `--response-ids 1024`
and `--scan-delta D`, and the result's `inputs` name both. `D` is the δ at which the world's year has an error of
2^-20, so that every row is bounded: 2^-20 divided by the world year's execution phases, which the command's own
document counts (the `scan` row's `phases`). It is computed from them each time and never copied, and `--scan-delta`
takes a fraction, so it holds `D` exactly. No term depends on the scan's δ: only the errors move with it.

```text
verity bench certifier_term --json --response-ids 1024 --scope world --period year |
    python3 -c '
import fractions, json, sys
phases = next(r["phases"] for r in json.load(sys.stdin)["rows"] if r["row"] == "scan")
print(fractions.Fraction(1, 2 ** 20) / fractions.Fraction(phases))'
verity bench certifier_term --response-ids 1024 --scan-delta D --out DIR
```

The suite beside the runner (`tests/`) holds the result to the calculator's document and to the kit's schema. What
the command prints without `--out`, and the calculator's numbers, are held where the calculator's tests are,
`benchmarks/network_traces/tests/test_certifier_term.py`.

## The rows

A scope is a node (one machine), the reference datacenter or the world, each machine behind one network certifier.
Over a week, a month and a year it serves its GPU-seconds × the serving MFU × a GPU's peak FLOP/s / (a request's
tokens × 2 × the served MACs a token) requests, each on one connection. Which request that is is the row's request
count, which every row names (Where the scale comes from, below). Each scope and monitoring period has eight rows,
and each row says where it stands:

- `scan` (on main): the rows form, the scan on the network certifier's own run, as the check charges it today. A
  connection is charged for its schedule and again for its fail cell, one connection a request, so these two terms
  are counted by request. The jitter is counted by link, for each of its execution phases.
- `N2` (on main): the clock form. The schedules are counted by link, over its execution phases, whatever it serves.
  Its parameters list no connection, so it has no fail cell. The jitter is the scan's.
- `lengths`, `timing`, `entries` (each a change to the scan's check) and `all three`: the scan with a response's
  length held to its Program's output count, with the release times taken once by the compute side's tick, and with
  the entries charged through the compute audit. The schedules' 0 under `lengths` and `all three` rests on a stop
  block, a proof block that checks where a response's ids stop, sampled on a served audit: it holds that output count
  to the response's ids. The jitter's 0 under `timing` and `all three` rests on the set-time card, which is
  simulated.
- `design, no stop block` (no run behind it): the design, below, before a stop block has been sampled on a served
  audit. The lengths unit holds a response's length to its Program's output count, but until a stop block holds that
  count to the ids the server picks it, so each connection is still charged its schedule, as the scan charges it.
  Its failed connections' charge, its jitter, its count and its entries are the design's.
- `design` (no run behind it): head checks, a lengths unit on the network certifier's records, and a stop block
  sampled on a served audit. How its stop class is charged is read from a served run's verdict records and from
  nothing else (The stop class's verdicts, below). Charged below its schedule, the lengths are held, the stop block
  holding a response's output count to its ids, and its term is the jitter, the failed connections' charge and the
  count. With no record read, or one verdict lacking, each connection is charged its schedule and the row is the
  one before it. The entries are the compute audit's either way.

### The stop class's verdicts

The row `design` is charged below its schedules only as the network certifier's own rule says
(`verity.protocols.certifier.verifier.class_charge.charge`), from the signed head of an execution phase and the
verdict records on a class of it:

- a class that holds a row owes four verdicts, the stop record, the response tie, the certified read of the request's
  three words and the compute audit's second stage, each accepted, complete and from a run on the head's serve;
- with one missing, refused, not complete or of another serve the class is charged its whole schedule on every row;
  a record without `complete` reads as not complete;
- a class with a count of 0 owes none.

`benchmarks/network_traces/stop_verdicts.py` reads the records and hands the calculator the answer (`of_runs`, `stop`).
The records are in the evidence store, in the run files of the runs that carry
`run_of_record=exfiltration-glm53-stop-verdicts`: `stop-verdicts.jsonl`, one verdict record a line, and
`stop-heads.jsonl`, one signed head a line, the serve it is of and its registration. The row is below its schedules
only where every head read is of one serve, some class holds a row, and every class that holds one, in every head
read, has its four verdicts. The calculator takes that answer as an argument (`document`'s and `table`'s `stop`). The
command has no option for it and no input holds it, and it reads no store itself: run alone it reads no record,
charges the row its whole schedules and says so. A caller that reaches the store names the runs and passes the
answer.

The lines print how the row is charged, one line for each class of each head read, and each verdict read under the
mark `run, not proved`: a verifier judged it on a run, and no proof of that verifier is read here, whatever the
record's `proved_by` says. A head is read as its registration lists it, held to the digest its certificate names; its
signature is not checked here.

Nothing a check counts by capacity is scaled by requests: twice the requests are twice the scan's schedules and fail
cells, and nothing else. The count is by capacity too: with the lanes given it does not move with the requests, and
with none given it moves only as the least lanes that serve them do.

## The terms

- **schedules**: for the scan, ⌈log2(cells + 1)⌉ bits a connection, where cells is the most cells a response takes,
  one more than the most ids it holds; 0 where the lengths are held with a stop block (`lengths`, `all three`, and
  `design` where its verdict records charge its stop class below its schedule), and the scan's where they are held
  without one (`design, no stop block`, and `design` otherwise). For N2, ⌈log2 #Σ_sync^m⌉ over a link's m execution
  phases (`--syncs`).
- **cells**: the fail cells, the same bits again a connection. Where the lengths are held, with a stop block or
  without, it is the failed connections' charge instead, pooled over the period: which connections failed and where
  each one's stream ended, at most F·(log2(e·N / F) + log2(cells + 1)) bits for F failures among N connections, and
  0 with none (`--failure-rate`, the share that fail, which no run has measured).
- **jitter**: ⌈log2 D_j^T⌉ bits for each execution phase of T buckets on each link. At D_j = 2, two release
  schedules, it is a bit a bucket, and that is the jitter of record. At D_j = 1 it is 0, which is what a network
  certifier with a set-time card would be charged: it is printed as "with a simulated set-time card". Where
  the timing is taken once it is 0 at either D_j, and that 0 rests on the same card, so those rows carry the label
  in their name.
- **count**: on the design's two rows, classes · log2(R + 1) bits for each execution phase on each link, whatever
  it serves, and 0 on every other row. Under the design the network certifier signs a head over each execution
  phase's tables, and a content class's tables are the rows it holds and no other, so the head states how many of
  the execution phase's releases each class holds: one of R + 1 values a class.
  - R is the most connections the schedule releases in an execution phase of T buckets: lanes · ⌈(T + ingress
    buckets − 1) / (D + 1)⌉, D being the delay. It is a bound: at some schedules no load releases as many.
  - The classes, the delay and the ingress buckets are inputs (`--classes`, `--delay`, `--ingress-buckets`), by
    default those of the lengths rows' cost runner (`benchmarks/network_traces/connection_rows_run.py cost`): six
    classes and a delay of 30 buckets, which its ingress buckets equal.
  - The lanes are the input the entries read (`--lanes`). With none given the schedule is not known, so the count
    is at the least lanes whose schedule releases what a link serves in an execution phase: any schedule that
    serves the load releases at least as many, so it is the least the rule can charge. With lanes given the count
    is at them. Lanes whose schedule releases fewer than a link serves in an execution phase are refused, with both
    numbers and the least lanes that do: the rows would count a schedule that does not carry the connections they
    charge.
  - The rows `lengths` and `all three` are charged no count, and each says so in its own line: its tables are of
    the lengths rows' form 1, the schedule's rows whatever the load, so the head states none.
  - The count's line is the count alone. The head's time is not held to a tick: nothing holds when the head of a
    link's execution phase leaves, so the line says so, and prints no charge for a head that leaves late, of 0 or
    of anything else.
- **term**: the four summed, the network certifier's own term, at each D_j. The term of record is the one at
  D_j = 2. The one at D_j = 1 is printed beside it only as "with a simulated set-time card", never as the
  term.
- **entries**: the slot entries, which are the compute side's and not part of the term. The scan's check charges
  every slot's entry at its whole width, a count by capacity: with `--lanes N`, the connections a link holds at once,
  it is printed so. Without it the column is one entry a request, the least the check can charge. Where the compute
  audit charges the entries, its own bound stands in their place, and this command doesn't count it.

## The tables

- **Inputs**: each input's value, unit, mark and source. A mark is `measured` (a run measured it), `modeled`
  (derived from other figures), `assumed` or `simulated` (a stand-in runs in place of a device that is not deployed,
  and the figure is what that device would be charged: the set-time card, `set_time_card`, alone); the source is the
  file and key it is read from, the parameter that gave it, or for the card the clause its figures rest on. No run
  has certified a class for GLM-5.3, so the class is assumed: the most ids a response holds
  (`--response-ids`), an id's bits (`--id-bits`), an execution phase's buckets and their seconds
  (`--phase-buckets`, `--bucket-s`), and the schedule the count is at (`--lanes`, `--classes`, `--delay`,
  `--ingress-buckets`).
- **The scopes**: the compute each row covers, which is all its scope serves: machines, GPU-seconds, requests and
  execution phases, and ε, the scan's execution phases' δ summed (`--scan-delta`). ε is a union bound, so at 1 or more
  it bounds nothing.
- **request count**: a column of every table, and `request_count` on every row of the document and at its top:
  `plan`, `census` or `given`, the request the row's requests are counted of. One table holds one request count;
  rows of two don't compare.
- **The terms in GB**: schedules, cells, the count, the jitter at each D_j, and the term at each D_j with its
  multiple of the target (`x target`, `--target-gb`) and of one copy of the weights in FP4 (`x weights`); then the
  entries. The columns at D_j = 2, the term of record, come before those at D_j = 1. Every column at D_j = 1, the
  multiples too, has the label in its header, as `simulated set-time card`, and so has the name of a row whose
  release times are taken once, in the `row` column: `timing, simulated set-time card`.
- **The terms in bits**: the same pooled totals in bits, beside the term for one execution phase of one link
  (`a link's phase`) and for one request (`a request`), the term as a share of the compute it covers.

The lines above the tables give the rule for the term of record, that the card is simulated, and the rule for the
rows that carry the label, each row's rule, one link's execution phase (its requests, a connection's bits, N2's
schedules, the jitter at each D_j, a slot entry's bits), the count (its rows, its bits a link's execution phase, R
and the schedule R is of: its lanes, given or the least, and its classes, delay and ingress buckets, the cost
runner's or as given; and the same rule at the cost runner's own schedule, 16 lanes and an execution phase of 600
buckets, which is that runner's and not these rows'), and the target as bits a link's execution phase and a request
over the world's year. Then they say
what the design's term over the world's year is with the release times taken once by a network certifier whose
network card releases at a set time (simulated), the failed connections' charge and the count: the largest failure
rate at which it meets the target, the document's `failure_rate_at_target`, or, where the count alone is over the
target, that no failure rate meets it, and the document's rate is null. Last they give what the jitter alone is
without that card, at D_j = 2 over the world's year, in GB and as a multiple of the target.

The document's `count` holds how the count is counted: its rows, the schedule (`classes`, `lanes` and whether they
were given, `delay`, `ingress_buckets`), R as `releases`, the bits a class and a link's execution phase, its charge
over the world's year beside the target, and the cost runner's own schedule with its R and bits (`cost_runner`).
Each row holds its own `count`, in bits and GB.

## Where the scale comes from

- The world is rho's world fleet (`verity/protocols/inference_only/rho/data/reference.toml`, `fleet.world-h200`): the census's inference
  (`benchmarks/census/adversary.json`) on eight-GPU machines at the fleet's serving MFU. The datacenter is the census's
  reference datacenter (`benchmarks/census/deployments.json`).
- The requests are counted of one request, by the rule `benchmarks/spot_figures/figure1.py` plans with, at the
  H200's dense FP8 peak (the H100 SXM5's entry of `benchmarks/census/hardware.json`, one die and one clock):
  - `plan`, the default: the sampling plan's request, so that the rows count the requests the compute audit's plan
    prices. The world's requests a year, a request's tokens and the served MACs a token are read from the break-even
    planner (`benchmarks/spot_figures/breakeven/`, `sweep.build` on the case `leak_bounds.PLAN_CASE`), which prices
    the 128k serving plan's request: fewer tokens than the census's, at more work a token.
  - `census` (`--requests census`): the census's tokens a request at the served MACs a token of GLM-5.3's workload
    card (`benchmarks/spot_figures/cards/workload.glm-5.3.json`).
  - `given` (`--world-requests N`): N requests a world-year, of no stated request, in place of either. Every scope
    scales with it.
- The weights are the workload card's parameters at FP4's 4 bits each.
- A bucket is the network certifier's default, and N2's #Σ_sync the most a calibration chooses by default
  (`verity/protocols/certifier/verifier/calibration.py`).

The scaling by links (`leak_bounds.at_link_rate`, which `chain_record.py leak` reads on a run) is a different count:
every link at the rate one run certified, whatever its scope serves.
