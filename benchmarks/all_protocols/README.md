# all_protocols

`verity bench all-protocols` (`run.sh`): every period, one run of the torch loop (`integrations/torch_loop`, a two-layer
W8A8 MLP on CPU) audited by every protocol's own code, each decided on its own and composed through onsite Verity
(`verity/core/protocols/onsite`), with a bound over the periods so far and projected to a month.

- `verity_all_protocols/legs.py`: the leg interface and `LEGS`, the runner's whole table. A leg is
  `leg(period: Period) -> Leg`: it audits the period's records (`Period.records`, the window the loop wrote) on
  `Period.node` and returns its protocol's core answer (`Answer.canonical`, `verity/core/protocols/interface/README.md`),
  the quantity its acceptance certifies, its port's case (what onsite's evidence carries for it) and its run ids. A lane
  plugs its protocol in by replacing its entry, and touches nothing else. A protocol with no leg reports
  `{"verdict": "missing", "owner": LANE}`; a leg that raises reports `error`. Neither is ever a verdict.
- `verity_all_protocols/integrity.py`: sampled proofs' leg, real. `benchmarks/prove_path/records_audit.py` on the period's
  window, from its records alone: live coins, C-Flock `--zk` behind the Lean firewall, the Lean verifier of record's
  audit record, turned into sampled proofs' answer (`verity/sampled-proofs/one-stage/v1`). Its quantity is the bound on
  wrong units at δ over the window's units.
- `verity_all_protocols/compose.py`: the composed verdict. A rejecting leg rejects it (onsite's reject dominance). It is
  `pending-ports` while a leg is missing or has no case, or any of onsite's components (read from `CoreOnsite.lean`) is
  still a stub, since a stub answers for itself, not for the protocol. Once all of them are ports, `onsite check` decides
  on the legs' cases. Bandwidth is not an onsite component yet: a rejection there rejects, but its acceptance completes
  nothing. `horizon` is the union bound over periods, each audited on its own live coins, `vacuous` once the δs sum to 1.
- `verity_all_protocols/runner.py`: the periods, each report in `$OUT/p<P>/report.json` and `$OUT/reports.jsonl`, and one
  `ALLP` line per period. `--tamper P,...` makes those periods' loop record outputs that aren't its model's.

| leg | protocol | onsite component | owner | state |
| --- | --- | --- | --- | --- |
| integrity | sampled proofs | `sampled-proofs` | compute-accounting | real |
| work | PoUW | `pouw` | compute-accounting | missing |
| space | PoUS | `pous` | memory-accounting | missing |
| bandwidth | the bandwidth audit over PoUS's store, or remote bandwidth | none yet | memory-accounting | missing |
| communication | the network certifier | `certifier` | network-accounting | missing |

~~~sh
research run --on vy-nebius-cpu-4 --project verity --source . --cwd clone --mem-gb 32 --timeout 120 -- \
  bash benchmarks/all_protocols/run.sh --periods 2 --period-s 0 --tamper 1
~~~
