# all_protocols

`verity bench all-protocols` (`run.sh`): every window, one run of the torch loop (`integrations/torch_loop`, a two-layer
W8A8 MLP on CPU) audited by every protocol's own code, each decided on its own and composed through onsite Verity
(`verity/core/protocols/onsite`). The result is per monitoring period, over all compute: a week, a month and a year of
every machine of a census deployment, the pooled total first and the per-window figure beside it.

- `verity_all_protocols/monitoring.py`: the monitoring-period report (`$OUT/monitoring-period.json`). A monitoring period
  of a deployment (`benchmarks/census/deployments.json`: its machines and window length, stated parameters) is
  `T = machines × ⌊period/window⌋` windows, each on its own live draw of `kd` of its `n` units (`q = kd/n`).
  - Pooled (the result): `IntegrityHorizon` (#4358; its sampling core `horizon_escape_log`, #4350). If every window is
    accepted, the period's faulty proof blocks total fewer than `ln(1/δ)/q`, except with probability `δ + Σ_t (ε_ks,t + δ_link,t)`.
    At k = 64 of n = 1024 that is at most 73 units at δ = 1% and 221 at 10⁻⁶, whatever `T` is.
  - Per window (beside it): `IntegrityPeriod` (#4358) at `δ/T` a window, the exact hypergeometric threshold.
  - Proved: each window's verdict, by the Lean verifier of record. Scaled: the number of windows, and that every window
    is audited as the run's are. The run doesn't compute `ε_ks` or `δ_link`, so each figure names that term.
  - PoUS, the bandwidth audit and the network certifier pool from the run's own answers over the same `T`, when they
    report. PoUS's is a count: its shortness reads each window's own coins, so the period certifies at least `T − M + 1`
    windows at each δ, by `pooledShort`'s Chernoff form at `p` the windows' `eps` (#4418, given `FixedBefore`; #4413).
    The bandwidth audit fails with probability `T · eps`, not `eps`. The network certifier pays each window's own
    terms of its itemized K every window and the faulty proof blocks once, `⌈ln(1/δ)/q⌉ · (16 + log2(N + 1))` bits
    (`HorizonLeakAccepted`, #4367, conditional on its per-window inputs until they are discharged).
  - `python -m verity_all_protocols.monitoring A/reports.jsonl B/reports.jsonl` pools windows run on several machines.
  - Communication per machine, beside it: the network certifier's `K` over a week, a month and a year
    (`communication`, `compose.communication`), at δ = 1/100 and 10⁻⁶. Where `resources.communication` pools every
    machine's windows as one sequence, this pools one machine's: each window draws its own live coins, so the faulty
    proof blocks are paid for once over that machine's windows (`HorizonLeakAccepted`, #4367): `Fmax = ⌈ln(1/δ)/q⌉` with `q`
    the least `k_i/n_i`, `16 · Fmax + ⌈log2 (N + 1)^Fmax⌉` bits for them with `N = Σ n_i`, plus each window's schedule,
    cell, free-input and jitter terms, which add over the period (free inputs dominate while the loop's window registers
    none of its inputs), except with probability `δ + Σ_i (e_i + e'_i)`. Its `share` is K over the period's output bits,
    16 a unit; `fleet` is the deployment's machines joined by a union bound at δ/machines each (not in Lean); each
    window's own `K` stands beside it. #4367 is conditional on two per-window inputs not yet discharged (proofs' `SampledKSLive`
    and the live `SampledLeakAccepted`), so this K is cited as conditional.
- `verity_all_protocols/table.py`: the protocols (`TABLE`) and the one subprotocol interface. A subprotocol is
  `legs/<component>.py` defining `leg(period_dir, node, out) -> dict`. `period_dir` is the window's loop records (the
  window the loop wrote) with `period.json` (the window's index, the circuit registered before the loop ran, the run
  root, the law, the window's δ, the built tools) and `answers/<component>.json`, the answers of the subprotocols before it. It
  returns its protocol's core answer (`Answer.canonical`, `verity/core/protocols/interface/README.md`), whose `public` is
  what acceptance certifies, and may write `out/<component>.case.json` (its port's case, for onsite) and `out/runs.json`.
  A lane plugs its protocol in by adding its file. A subprotocol with no file reports
  `{"verdict": "missing", "owner": LANE}`, and one whose file raises reports `error`. Neither is ever a verdict.
- `legs/sampled_proofs.py`: sampled proofs, real. `benchmarks/prove_path/records_audit.py` on the window, from its records
  alone, at k = 64 and δ = 1/100 by default: live coins, C-Flock `--zk` behind the Lean firewall, the Lean verifier of
  record's audit record, turned into sampled proofs' answer (`verity/sampled-proofs/one-stage/v1`). It also hands on
  `sampled-proofs.case.json`, the case `SampledProofs.OneStage` reads (written by `examples/one-stage/a5.py`), with
  whether `OneStage.check`'s Python on it gives the subprotocol's verdict (`case-check.json`).
- `verity_all_protocols/compose.py`: the composed verdict. A rejecting subprotocol rejects it (onsite's reject dominance). It is
  `pending-ports` while a subprotocol is missing or has no case, or any of onsite's components (read from `CoreOnsite.lean`) is
  still a stub, since a stub answers for itself, not for the protocol. Once all of them are ports, `onsite check` decides
  on the subprotocols' cases. `run.sh` builds `onsite` and the runner runs `onsite check` every window regardless: each port on its
  subprotocol's case, each stub undecided (no statement yet), a port with no case on one it refuses. Its answer is `composed.answer`,
  where each case came from `composed.inputs`.
- `legs/certifier.py`: the network certifier, real. One network certifier session over the run's windows (`period.json`'s
  `period` of `periods`), in front of the loop's model (`benchmarks/network_traces/torch_loop_stub.py`), a link-window
  per window; N2's verdict of record on each link-window (`--zk` behind the Lean firewall); the certified outputs
  checked against the period's registered ones; then `NetworkCheck.Period` (`verity/certifier/v0`,
  `network-period check`) on the period's link-windows and its compute audit, whose free inputs are the window's
  unregistered input rows (`circuit.json`): on acceptance `public.K`, the bits a covert sender in the server can move over the link in the
  period, except with probability `public.eps`, at δ = 1/100 (beside it the case at 10⁻⁶ and the tampered run's, whose
  tampered period must reject). `eps` sums its terms' δ (each audit's, and in the rows form each link-window's scan's),
  so each term is decided at δ over their count (`term_delta`) and an accepted answer's `eps` is δ itself. The session
  stays in the runner's scratch; `$OUT/certifier-session/` gets each run's deployment, calibration and client
  certificates and out records only, and `summary.json` counts, N2's exit and its leak scan. Its `stubbed` names what
  the verdict still stands on: the records are the certifier-only path's, not the network certifier's own cells; the
  device runs beside the server, not on the auditor's machine through the gate; and the output tie is computed in the
  clear, not proved, unless the runner ties it (`--tie`): then the network certifier commits the output it relayed in
  each link-window as the registered value `out` at the reading circuit's out shape (its registered read of `out`: one
  word a row, or in the wired circuit a down_proj unit's whole row, request-major) and signs its head with its
  session's own key at the link-window's close (`certifier_serve.py --out-shape`, `out_commit.commit_out`; the record
  public, the rows and salts the prover's), the verifier pins its root from that record alone at its own shape under
  the deployment's certifier key and session (`check_out`), each output unit's `o0` reads it, the integrity audit proves the output class (every unit with an
  uncut registered read of an output port) whole beside its work draw, and Lean's registered reads refuse an output
  unit whose row doesn't open the signed root; the clear check is then a diagnostic.
  `CERTIFIER_PATH=warden` takes the network certifier's own cells instead (`warden_serve.py rehearse --ids --prompts
  --window-per-group`, the gate's scan rehearsal, a period's link-windows its group's in `groups.json`), honest and with
  the stub's tampers (`WARDEN_TAMPER`, default `late,byte`), publishing the gate's public files alone to
  `$OUT/warden-session/`. Its link-window 0 is proved by the scan (`scan_zk.py prove --warden-run --draw`), and a period
  whose every link-window has a scan answer is decided in certifier/v0's rows form; a later period waits on
  `scan_zk --window W`, and the stub's `byte` and `extra` pass the scan, so only the in-the-clear tie refuses them. `block_leg`
  turns a period's accepted answer of either path into the block record's `certifier` entry (`verity/block-record/v1`,
  CMP-95 item 6), the one shape a loop's run and the H200 fallback write.
- `verity_all_protocols/runner.py`: the windows, each report in `$OUT/w<W>/report.json` and `$OUT/reports.jsonl`, and
  one `ALLP` line per window, led by the monitoring period. Outer proofs go through the firewall one at a time, one
  firewalled session on the run whichever window it belongs to, so the windows are audited one after another and each
  audit's drawn members' proof sessions too (`--jobs` and `--session-jobs` take only 1). Only what never goes through
  the firewall overlaps: the next window's loop serves while a window is audited.
  `--start I` numbers another machine's windows apart. `--tamper W,...` makes those windows' loop record outputs that
  aren't its model's. `--registered` has the loop read its weights as registered values, committed once a run.
  `--wired` serves the loop's wired circuit (`serve --wired`).
  `--pouw h1` (until ALLP-H1LOOP's `serve --pouw h1` lands) also serves PoUW's H1 window and audits it with
  `verity_torch_loop.pouw_audit` before the legs.
  `--tie` ties the outputs to the certified ones (`verity_all_protocols/tie.py`) under a work law (default `work:64`, any
  other refused), and `--tie-tamper W,...` with `CERTIFIER_TAMPER=byte` ties those windows to a relay with one output
  byte changed on the egress, which must reject.
  `--of-record` is S1's run of record (below). Every run writes its mode (`of_record` and its exact flags) to
  `$OUT/mode.json`, each window's `period.json` and report (`mode`) and its `ALLP` line, so its evidence says whether it
  counts.

| name | subprotocol | file | owner | state |
| --- | --- | --- | --- | --- |
| `integrity` | sampled proofs | `legs/sampled_proofs.py` | compute-accounting | real |
| `work` | PoUW | `legs/pouw.py` | compute-accounting | missing |
| `space` | PoUS | `legs/pous.py` | memory-accounting | missing |
| `bandwidth` | the bandwidth audit over PoUS's store, or remote bandwidth | `legs/bandwidth.py` | memory-accounting | missing |
| `communication` | the network certifier | `legs/certifier.py` | network-accounting | real |

## S1's run of record

This is the one way to run S1 so that it counts, the joint run: every layer chained (`--wired`) and the output tied to
what the network certifier relayed:

~~~sh
research run --on vy-nebius-cpu-4 --project verity --source . --cwd clone --mem-gb 96 --store-read -- \
  bash benchmarks/all_protocols/run.sh --of-record --wired --windows 2 --tie-tamper 0
~~~

`--of-record` sets the flags three lanes' work needs:

- `--tie`, network-accounting's output tie;
- `--pouw h1`, whose work leg refuses a window without PoUW's window;
- `--registered`, so the weights are counted registered, not free;
- `--law work:64`, the work law the tie needs (`work:14` with `--wired`: the wired toy has 14 units);
- `--jobs 1 --session-jobs 1`, so one firewalled session runs at a time.

Under `--tie-tamper` it also sets `CERTIFIER_TAMPER=byte`.

`run.sh` checks the command before it builds anything (`runner --check`, whose output is `$OUT/mode-check.json`). It
refuses a command that contradicts the mode, with the reason: another `--law`, `--jobs` or `--session-jobs` other than 1,
or another `CERTIFIER_TAMPER`.

Window 0 is tied to a relay with one egress byte changed, so it must reject (Lean's registered read of the wired
down_proj unit's `o0`), and window 1 is honest, so it must accept. Any other command is a development run: its
`mode.json` says `"of_record": false`.

In the wired circuit `out` is one row of h words a down_proj unit, request j's step s row `j·S + s`: each down_proj
unit's `o0` reads its row, and the next step's plain quantize unit's `x` reads the same row in place of the `down` tie.
The network certifier commits `out` at that width and the verifier pins it at that width, both from the circuit's
registered read (`serve --circuit-only`'s `out_shape`). `--of-record` without `--wired` is the unwired run, one word a
row; its mode names the gap (`gap`): the wiring is cited from the wired run of record.
