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
- `legs/certifier.py`: communication, real. A link carries the loop's requests and step outputs (read back from its
  window) through the network certifier, certifier-serve's planned profile on an injected clock; each link-window's unit
  is proved `--zk` behind the Lean firewall and decided by N2's verifier of record (`n2_zk.py prove`;
  `ALLP_CERTIFIER_PROVE=0` decides in the clear, which answers `undecided`), and `network-period check`
  (`verity/certifier/v0`) decides the period with the integrity leg's audit, publishing `K`. `certifier.month.json` has
  the month's bound at δ = 1% and 10⁻⁶ (`HorizonLeakAccepted`, conditional). The certifier runs on the prover's node, a
  rehearsal; `ALLP_CERTIFIER_RUN` names a run of vy-challenger-1's gate instead.
- `verity_all_protocols/compose.py`: the composed verdict. A rejecting leg rejects it (onsite's reject dominance). It is
  `pending-ports` while a leg is missing or has no case, or any of onsite's components (read from `CoreOnsite.lean`) is
  still a stub, since a stub answers for itself, not for the protocol. Once all of them are ports, `onsite check` decides
  on the legs' cases. Bandwidth is not an onsite component yet: a rejection there rejects, but its acceptance completes
  nothing. `horizon` is the union bound over periods, each audited on its own live coins, `vacuous` once the δs sum to 1.
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
