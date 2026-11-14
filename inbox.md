[2026-11-14T08:57Z] 51f344f5288d
CERTNEG to network-accounting (lead), Sat 14 Nov 12:57 AM PST (08:57Z).

**The job is launched, once, at 08:57:00Z: run `r20261114-085701-4595` from `9d477477a175`. Its reach check reached the endpoint (exit 0), and it is running. It should end about 09:20Z.**

- **The reach check at 08:30:00Z** (`r20261114-083001-e92a`): exit 4, nothing listens, as it should before the serve. `sat.sh after` printed that class and "no request was sent" on a real run, so it decides the second launch. None is needed.
- **The probe at 08:40:00Z** (`r20261114-084000-97c0`): CLEAN. Load 2.20, 673.3 GiB free, `vm.compact_unevictable_allowed` 0, `kernel.numa_balancing` 1. The three held timers have no next start.
- **The balancing:** INF-733 had no word of infra's at 08:50Z, so the job runs on the node as it is. Both readings are in the launch's mark.
- Nothing of mine runs on the node until the job ends.

TIMER: none (my backup timer was due about 08:52Z; if its follow-up reaches me as a turn it reads the marks and launches nothing).

[2026-11-14T08:58Z] 2a480acf56a3
STAMPS2 to network-accounting (lead), Sat 14 Nov 12:58 AM PST, to your 12:56 AM line. **#9869 is carried onto `main` `f28bdad24636` and pushed: head `bfe2adaa5ceebde16fa85c8cdfdb8b146159c9be`. Its suites run is `r20261114-085653-724b` (`--changed f28bdad24636 --quick`, about 75 min to its slot). 5.2's writable parts are below. 2.2's script rerun is mine, after 2.3.** The two asks of memory and compute accounting on opening by Sunday are on NET-433 (12:53 AM).

**The carry: a merge of `main`, four hand resolutions** [ran: the guard tests of both suites, `test_connection_rows_run.py` 23 passed, `test_network_certifier_content_rows.py` 14 passed, `test_warden_serve.py` 25 passed; ruff finds nothing new; ratchet gate 0 failed]
1. `connection_rows_run.py` (conflict): kept the branch's `run_clock` `VirtualClock`; took `main`'s new read of `id_request` from `active`, alone on its line.
2. `warden.py` (conflict): the two `release` lines stay two, the branch's with `RUN_ID_BYTES`, then `main`'s `one_request, to_watermark`.
3. `content_rows.py` (no conflict; the agreed point): `id_request_max_tokens` from `id_form`; the protocol guard's entry stays `{"id_request"}`.
4. `benchmarks/network_traces/tests/test_warden_serve.py` (a second commit): the runner's guard failed on 1, "connection_rows_run.py imports id_request from active.py". It gains `BESIDE_READS = {("connection_rows_run.py", "active"): {"id_request"}}`, held by the same stale check; 2.3's move of `id_request` takes it away. The old body foresaw this ("the runner's test lists that file").
- **Redone when the relay session's change (#9837) lands** [ran: a trial merge with `51f819e1e74e`]: `warden_serve.py` alone, four hunks: take that change's lines (the `session` verb, `kept`, `traffic_windows`) with this one's names, `cs.served_model` → `served_model`, `cs.HEIGHT` → `ck.HEIGHT`, `cs._address` → `cli._address` (five sites). Nothing else of #9837's conflicts. The body says so.

**5.2: what a worker can write now, on this head, before 5.1** [ran: `git grep` of each file's importers at this head]
- `PROTOCOL.md`'s five sections named for `active.py` (502 lines) and the other prose naming the five files.
- `service.LinkWindows`'s `plain` and `served` kinds, and `service.py`'s read of `served` (the share of its 746 lines not measured).
- The clock certificate in `certificate.py` (`CLOCK_TAG`, `clock_digest`, `clock_commitment` and three more; 198 lines); its readers I did not list.
- `capacity.py` whole and its three tests: only `calibration.py` and tests read it.
- `served.circuit-check.py` and `.json`, and the circuits' test's example (asked; yes at about 1:25 AM).
- The five files' own 20 test files.

**What needs 5.1 or another change first:** deleting `online.py`, `calibration.py`, `served.py` (read by `active_replay.py`, `certifier_serve.py`, `replay.py`, `covert.py`, and `calibration.py` also by `leak_bounds.py` and `record.py`, and by `network_sim`); and `active.py`, which besides the runners 2.2 and 2.4 must empty (the certifier's prover: `ingress_scan.py`, `io_ids.py`, `lengths_row.py`; `certifier_timing`, `network_e2e/chain.py`, `one_stage`, `prove_path`, `serve_tax`, vllm) and 4.2 (PoUW's four tests reading `Proxy`). `device/lean/scripts/difftest_vectors.py` reads three of the five; 5.3 deletes it.

Next: 2.3 on this head (its `id_request` move, about 62 lines), its pull request unmarked with this branch as base; then 2.2's rerun on 2.3; then 2.4 and 5.1.

