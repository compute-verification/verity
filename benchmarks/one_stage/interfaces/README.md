# E2E demo interfaces

The interfaces between the six subprotocols of the one-H200 E2E demo: the block record.
Not a package: run the modules with this directory on `PYTHONPATH` (the suite's `pyproject.toml` sets it).

## The block record (`verity/block-record/v2`)

`verity_e2e_demo/block_record.py`, schema `verity_e2e_demo/schemas/block-record.v2.schema.json`. What each subprotocol writes
for one server block and the Lean check's answer for it. It extends the all-protocols runner's report
(`verity/all-protocols-period/v1` and `-window/v1`) with:

- the block's server and steps, `server_time: [{server, steps: [lo, hi)}]`;
- the subprotocols, keyed `integrity` (SPOT), `pouw`, `pous`, `bandwidth`, `attester` and `wipe`. Each is
  `null` when the block ran no such audit, else its protocol, the core answer, its certificate (integrity's count bound,
  `profile.Stratified.count_bound` over its sample: over the answer's own `strata`, which the quantity then carries,
  where its profile samples each stratum on its own, and over a uniform sample of the checked level otherwise; W_cert,
  S0, Q_cert, K, the wipe's foreign_bits at its reads `k` and ratio `r`), its error and the certificate at δ = 1/100
  and 1/10^6, its run ids, and `source: real|fixture`. S0 is PoUS's state bound ⌊ρ·|C|⌋; the schema also reads
  `S_cert`, that quantity's older name, which nothing writes;
- the terms an error leaves out, `named_terms`. The bandwidth audit's error has one, with no number and in no total:
  the PRF advantage of the keyed stream its openings are sampled from. Its entry carries PoUB's `ledger.NAMED_TERMS` as
  its answer's profile names it, and a bandwidth answer that names none is refused: no entry is built from it, and
  `check` and the schema refuse a record whose bandwidth entry has none. The error, its certificates at each δ and
  every sum over it read with the term beside them (`error_form`, the ledger's `DELTA_FORM` with the error for δ);
- the composed verdict, which is the answer's: the Lean check of the block's product (`CHECKS`: `verity/onsite/v1`,
  whose work is Pearl+'s run verdict, or `verity/onsite/v0`, H1's, in records written before it; or remote's composite
  of record `verity/remote/v1`, as the composite Lean check writes its
  `answer.json` and `composite.py` its public `composite.json`), `accept`, `reject` or `undecided`, and
  `undecided` with no answer. Nothing in Python combines the subprotocols' verdicts;
- what the product's subprotocols say (`compose`) and the error of `ResourceAccounting.Guarantees.JointPooledLive` over the
  block: the accepted subprotocols' errors summed, with `joint_horizon.named_terms` naming, by subprotocol, the terms
  those errors leave out, which the sum and the monitoring period's bounds leave out too. The product (`PRODUCTS`) is
  onsite's, every subprotocol, unless `composed.product` names remote's, SPOT and PoUW, whose other
  subprotocols are `dropped`: not run and not composed;
- the monitoring period: week, month and year over a census deployment's machines (a parameter), with JointPooledLive's
  union bound over that many blocks and the per-block error each δ needs.

`check` recomputes every derived field from the subprotocols' entries and the answer.

    python -m verity_e2e_demo.block_record check tests/fixtures/block-record-r2-period0.json
    python -m verity_e2e_demo.block_record from-runs --r2 R2/reports.jsonl --r1 R1/reports.jsonl --wipe WIPE/results.json --out DIR

The fixtures are records of real runs, in the record's form of today: a bandwidth entry names the term its error leaves
out, which those runs' answers did not, so `from-runs` refuses a report of theirs that has a bandwidth row.
`block-record-r2-period0` is one period of an all-protocols run: five
subprotocols accept, and a wipe is carried but not composed, since it ran on another server. It has no Lean answer, so the
verdict is undecided, and the error sum is at least SPOT's 1/2 from its 2-unit sample. `block-record-b71a-w30`
carries its window's Lean `onsite check` answer, which accepts. `block-record-r1-window10` is one window with
SPOT at δ = 1/100 and the wipe on the same server, the other subprotocols absent.
`integrity-profile-stratified` is the profile of a served run's SPOT answer, which samples each template's proof blocks as a
stratum of their own (all 160 of one, 2 of another's 4): its bound is 2 of 164 at δ = 1/100, where a uniform sample of
162 of 164 would say 1.
