# certifier_term

`verity bench certifier_term` prints the network certifier's term for GLM-5.3: the bits its accepted transcripts can
carry over a monitoring period, at the load of a node, the census's reference datacenter and the world, in bits and
GB, beside a target and one copy of the weights. Each row counts all its scope serves by its check's own rule, as
the monitoring-period check charges it (`verity/protocols/certifier/verifier/lean/NetworkCheck/`, `rowsTerms` and `terms`).
Its code is the calculator of K, `benchmarks/network_traces/leak_bounds.py` (`table`, `document`, `render`); this
directory holds its runner.

```text
verity bench certifier_term                              the inputs, the scopes, and the terms in GB and in bits
verity bench certifier_term --json                       the same as one document: value, unit, commit, run, inputs, rows
verity bench certifier_term --scope world --period year  only those rows (each flag repeats)
verity bench certifier_term --requests census            the census's request in place of the sampling plan's
verity bench certifier_term --help                       every parameter and its default
```

It reads files and computes, so it needs no GPU and no run. Under `research run`, give the document its run and
commit: `--run "$RESEARCH_RUN_ID" --commit "$RESEARCH_SOURCE_SHA"`. Without `--commit` it names the checkout's HEAD.

## The rows

A scope is a node (one machine), the reference datacenter or the world, each machine behind one network certifier.
Over a week, a month and a year it serves its GPU-seconds × the serving MFU × a GPU's peak FLOP/s / (a request's
tokens × 2 × the served MACs a token) requests, each on one connection. Which request that is is the row's request
count, which every row names (Where the scale comes from, below). Each scope and monitoring period has seven rows,
and each row says where it stands:

- `scan` (on main): the rows form, the scan on the network certifier's own run, as the check charges it today. A
  connection is charged for its schedule and again for its fail cell, one connection a request, so these two terms
  are counted by request. The jitter is counted by link, for each of its execution phases.
- `N2` (on main): the clock form. The schedules are counted by link, over its execution phases, whatever it serves.
  Its parameters list no connection, so it has no fail cell. The jitter is the scan's.
- `lengths`, `timing`, `entries` (each a change to the scan's check) and `all three`: the scan with a response's
  length held to its Program's output count, with the release times taken once by the compute side's tick, and with
  the entries charged through the compute audit.
- `design` (no run behind it): head checks and a lengths unit on the network certifier's records. The lengths are
  held and the entries are the compute audit's, so its term is the jitter and the failed connections' charge.

Nothing a check counts by capacity is scaled by requests: twice the requests are twice the scan's schedules and fail
cells, and nothing else.

## The terms

- **schedules**: for the scan, ⌈log2(cells + 1)⌉ bits a connection, where cells is the most cells a response takes,
  one more than the most ids it holds; 0 where the lengths are held. For N2, ⌈log2 #Σ_sync^m⌉ over a link's m
  execution phases (`--syncs`).
- **cells**: the fail cells, the same bits again a connection. Where the lengths are held it is the failed
  connections' charge instead, pooled over the period: which connections failed and where each one's stream ended,
  at most F·(log2(e·N / F) + log2(cells + 1)) bits for F failures among N connections, and 0 with none
  (`--failure-rate`, the share that fail, which no run has measured).
- **jitter**: ⌈log2 D_j^T⌉ bits for each execution phase of T buckets on each link. At D_j = 1 it is 0; at D_j = 2 it
  is a bit a bucket. Both are printed; where the timing is taken once it is 0.
- **term**: the three summed, the network certifier's own term, at each D_j.
- **entries**: the slot entries, which are the compute side's and not part of the term. The scan's check charges
  every slot's entry at its whole width, a count by capacity: with `--lanes N`, the connections a link holds at once,
  it is printed so. Without it the column is one entry a request, the least the check can charge. Where the compute
  audit charges the entries, its own bound stands in their place, and this command doesn't count it.

## The tables

- **Inputs**: each input's value, unit, mark and source. A mark is `measured` (a run measured it), `modeled`
  (derived from other figures) or `assumed`; the source is the file and key it is read from, or the parameter that
  gave it. No run has certified a class for GLM-5.3, so the class is assumed: the most ids a response holds
  (`--response-ids`), an id's bits (`--id-bits`), and an execution phase's buckets and their seconds
  (`--phase-buckets`, `--bucket-s`).
- **The scopes**: the compute each row covers, which is all its scope serves: machines, GPU-seconds, requests and
  execution phases, and ε, the scan's execution phases' δ summed (`--scan-delta`). ε is a union bound, so at 1 or more
  it bounds nothing.
- **request count**: a column of every table, and `request_count` on every row of the document and at its top:
  `plan`, `census` or `given`, the request the row's requests are counted of. One table holds one request count;
  rows of two don't compare.
- **The terms in GB**: schedules, cells, the jitter at each D_j, and the term at each D_j with its multiple of the
  target (`x target`, `--target-gb`) and of one copy of the weights in FP4 (`x weights`); then the entries.
- **The terms in bits**: the same pooled totals in bits, beside the term for one execution phase of one link
  (`a link's phase`) and for one request (`a request`), the term as a share of the compute it covers.

The lines above the tables give each row's rule, one link's execution phase (its requests, a connection's bits, N2's
schedules, a slot entry's bits), the target as bits a link's execution phase and a request over the world's year, and
the largest failure rate at which, with the lengths held and the timing taken once, the world's year meets the
target.

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
