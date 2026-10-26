# all_protocols

`verity bench all-protocols` (`run.sh`): every window, one run of the torch loop (`integrations/torch_loop`, a two-layer
W8A8 MLP on CPU) audited by every protocol's own code, each decided on its own and composed through onsite Verity
(`verity/core/protocols/onsite`). The result is per monitoring period, over all compute: a week, a month and a year of
every machine of a census deployment, the pooled total first and the per-window figure beside it.

- `verity_all_protocols/monitoring.py`: the monitoring-period report (`$OUT/monitoring-period.json`). A monitoring period
  of a deployment (`benchmarks/census/deployments.json`: its machines and window length, stated parameters) is
  `T = machines × ⌊period/window⌋` windows, each on its own live draw of `kd` of its `n` units (`q = kd/n`).
  - Pooled (the result): `IntegrityHorizon` (#4358; its sampling core `horizon_escape_log`, #4350). If every window is
    accepted, the period's wrong units total fewer than `ln(1/δ)/q`, except with probability `δ + Σ_t (ε_ks,t + δ_link,t)`.
    At k = 64 of n = 1024 that is at most 73 units at δ = 1% and 221 at 10⁻⁶, whatever `T` is.
  - Per window (beside it): `IntegrityPeriod` (#4358) at `δ/T` a window, the exact hypergeometric threshold.
  - Proved: each window's verdict, by the Lean verifier of record. Scaled: the number of windows, and that every window
    is audited as the run's are. The run doesn't compute `ε_ks` or `δ_link`, so each figure names that term.
  - Space, bandwidth and communication pool from the run's own answers over the same `T`, when their legs report.
    Space is a count: PoUS's shortness reads each window's own coins, so the period certifies at least `T − M + 1`
    windows at each δ, by `pooledShort`'s Chernoff form at `p` the windows' `eps` (#4418, given `FixedBefore`; #4413).
    Bandwidth fails with probability `T · eps`, not `eps`. Communication pays each window's own terms of the
    certifier's itemized K every window and the faulty units once, `⌈ln(1/δ)/q⌉ · (16 + log2(N + 1))` bits
    (`HorizonLeakAccepted`, #4367, conditional on its per-window inputs until they are discharged).
  - `python -m verity_all_protocols.monitoring A/reports.jsonl B/reports.jsonl` pools windows run on several machines.
- `verity_all_protocols/table.py`: the protocols (`TABLE`) and the one leg interface. A leg is
  `legs/<component>.py` defining `leg(period_dir, node, out) -> dict`. `period_dir` is the window's loop records (the
  window the loop wrote) with `period.json` (the window's index, the circuit registered before the loop ran, the run
  root, the law, the window's δ, the built tools) and `answers/<component>.json`, the answers of the legs before it. It
  returns its protocol's core answer (`Answer.canonical`, `verity/core/protocols/interface/README.md`), whose `public` is
  what acceptance certifies, and may write `out/<component>.case.json` (its port's case, for onsite) and `out/runs.json`.
  A lane plugs its protocol in by adding its file. A protocol with no leg file reports
  `{"verdict": "missing", "owner": LANE}`, and a leg that raises reports `error`. Neither is ever a verdict.
- `legs/sampled_proofs.py`: integrity, real. `benchmarks/prove_path/records_audit.py` on the window, from its records
  alone, at k = 64 and δ = 1/100 by default: live coins, C-Flock `--zk` behind the Lean firewall, the Lean verifier of
  record's audit record, turned into sampled proofs' answer (`verity/sampled-proofs/one-stage/v1`). It also hands on
  `sampled-proofs.case.json`, the case `SampledProofs.OneStage` reads (written by `benchmarks/one_stage/a5.py`), with
  whether `OneStage.check`'s Python on it gives the leg's verdict (`case-check.json`).
- `verity_all_protocols/compose.py`: the composed verdict. A rejecting leg rejects it (onsite's reject dominance). It is
  `pending-ports` while a leg is missing or has no case, or any of onsite's components (read from `CoreOnsite.lean`) is
  still a stub, since a stub answers for itself, not for the protocol. Once all of them are ports, `onsite check` decides
  on the legs' cases. Bandwidth is not an onsite component yet: a rejection there rejects, but its acceptance completes
  nothing.
- `verity_all_protocols/runner.py`: the windows, each report in `$OUT/w<W>/report.json` and `$OUT/reports.jsonl`, and one
  `ALLP` line per window, led by the monitoring period. `--jobs J` audits J windows at once, each on its own ports. An
  audit's drawn members' proof sessions run one at a time, since outer proofs go through the firewall one at a time
  (`--session-jobs` takes only 1).
  `--start I` numbers another machine's windows apart. `--tamper W,...` makes those windows' loop record outputs that
  aren't its model's. `--pouw h1` (until ALLP-H1LOOP's `serve --pouw h1` lands) also serves PoUW's H1 window and audits
  it with `verity_torch_loop.pouw_audit` before the legs.

| name | protocol | leg file | owner | state |
| --- | --- | --- | --- | --- |
| integrity | sampled proofs | `legs/sampled_proofs.py` | compute-accounting | real |
| work | PoUW | `legs/pouw.py` | compute-accounting | missing |
| space | PoUS | `legs/pous.py` | memory-accounting | missing |
| bandwidth | the bandwidth audit over PoUS's store, or remote bandwidth | `legs/bandwidth.py` | memory-accounting | missing |
| communication | the network certifier | `legs/certifier.py` | network-accounting | missing |

~~~sh
research run --on vy-nebius-cpu-4 --project verity --source . --cwd clone --mem-gb 96 -- \
  bash benchmarks/all_protocols/run.sh --windows 4 --jobs 4 --tamper 3
research run --on vy-nebius-cpu-5 --project verity --source . --cwd clone --mem-gb 96 -- \
  bash benchmarks/all_protocols/run.sh --windows 4 --start 4 --jobs 4
~~~
