# all_protocols

`verity bench all-protocols` (`run.sh`): every period, one run of the torch loop (`integrations/torch_loop`, a two-layer
W8A8 MLP on CPU) audited by every protocol's own code, each decided on its own and composed through onsite Verity
(`verity/core/protocols/onsite`), with a bound over the periods so far and projected to a month.

- `verity_all_protocols/table.py`: the protocols (`TABLE`) and the one leg interface. A leg is
  `legs/<component>.py` defining `leg(period_dir, node, out) -> dict`. `period_dir` is the period's loop records (the
  window the loop wrote) with `period.json` (the circuit registered before the loop ran, the run root, the law, the built
  tools) and `answers/<component>.json`, the answers of the legs before it. It returns its protocol's core answer
  (`Answer.canonical`, `verity/core/protocols/interface/README.md`), whose `public` is what acceptance certifies, and may
  write `out/<component>.case.json` (its port's case, for onsite) and `out/runs.json`. A lane plugs its protocol in by
  adding its file. A protocol with no leg file reports `{"verdict": "missing", "owner": LANE}`, and a leg that raises
  reports `error`. Neither is ever a verdict.
- `legs/sampled_proofs.py`: integrity, real. `benchmarks/prove_path/records_audit.py` on the period's window, from its
  records alone: live coins, C-Flock `--zk` behind the Lean firewall, the Lean verifier of record's audit record, turned
  into sampled proofs' answer (`verity/sampled-proofs/one-stage/v1`): at most `public.bound` of the window's units wrong,
  except with probability δ.
- `verity_all_protocols/compose.py`: the composed verdict. A rejecting leg rejects it (onsite's reject dominance). It is
  `pending-ports` while a leg is missing or has no case, or any of onsite's components (read from `CoreOnsite.lean`) is
  still a stub, since a stub answers for itself, not for the protocol. Once all of them are ports, `onsite check` decides
  on the legs' cases. Bandwidth is not an onsite component yet: a rejection there rejects, but its acceptance completes
  nothing. `horizon` is the union bound over periods, each audited on its own live coins, `vacuous` once the δs sum to 1,
  and beside it the network certifier's `K` over a week, a month and a year of the run's periods (`communication`), at
  δ = 1/100 and 10⁻⁶. Each window draws its own live coins, so the faulty units are paid for once over the whole span
  (`HorizonLeakAccepted`, #4367): `Fmax = ⌈ln(1/δ)/q⌉` with `q` the least `k_i/n_i`, `Fmax · (16 + log2(N + 1))` bits
  for them with `N = Σ n_i`, plus each period's schedule, cell, free-input and jitter terms, except with probability
  `δ + Σ_i (e_i + e'_i)`. Its `share` is K over the span's output bits, 16 a unit; `fleet` is 1,000 machines joined by a
  union bound at δ/1000 each (not in Lean); each period's own `K` stands beside it. #4367 is conditional on two
  per-window inputs not yet discharged (proofs' `SampledKSLive` and the live `SampledLeakAccepted`), so this K is cited
  as conditional.
- `legs/certifier.py`: communication, real. One network certifier session over the run's periods, in front of the
  loop's model (`benchmarks/network_traces/torch_loop_stub.py`), a link-window per period; N2's verdict of record on each
  link-window (`--zk` behind the Lean firewall); the certified outputs checked against the period's registered ones; then
  `NetworkCheck.Period` (`verity/certifier/v0`, `network-period check`) on the period's link-windows and its compute
  audit: on acceptance `public.K`, the bits a covert sender in the server can move over the link in the period, except
  with probability `public.eps`, at δ = 1/100 (beside it the case at 10⁻⁶ and the tampered run's). Its `summary.json`'s
  `stubbed` names what the verdict still stands on: the records are the certifier-only path's, not the network
  certifier's own cells, and the device runs beside the server, not on the auditor's machine through the gate.
- `verity_all_protocols/runner.py`: the periods, each report in `$OUT/p<P>/report.json` and `$OUT/reports.jsonl`, and one
  `ALLP` line per period. `--tamper P,...` makes those periods' loop record outputs that aren't its model's.

| name | protocol | leg file | owner | state |
| --- | --- | --- | --- | --- |
| integrity | sampled proofs | `legs/sampled_proofs.py` | compute-accounting | real |
| work | PoUW | `legs/pouw.py` | compute-accounting | missing |
| space | PoUS | `legs/pous.py` | memory-accounting | missing |
| bandwidth | the bandwidth audit over PoUS's store, or remote bandwidth | `legs/bandwidth.py` | memory-accounting | missing |
| communication | the network certifier | `legs/certifier.py` | network-accounting | real |

~~~sh
research run --on vy-nebius-cpu-4 --project verity --source . --cwd clone --mem-gb 32 -- \
  bash benchmarks/all_protocols/run.sh --periods 2 --period-s 0 --tamper 1
~~~
