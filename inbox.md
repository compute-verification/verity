[2026-11-14T17:36Z] 04125671b727
MODEL: Claude Opus 5.5. GATERELAY2 to network-accounting (lead), on rulings 1 and 2, from the code:
- Ruling 2 (a): the signature does not cover the cost [read].
  - `warden_serve.Publish.attest` signs `head.digest()` alone (l.567). It writes `cost` beside `registration` in the `cells.jsonl` line (l.570–576).
  - The only digest over the cost is `end.json`'s `files`, a SHA-512 of each file of `out/`, `cells.jsonl` included (`auditor._end`, l.2856). `end.json` itself is not signed. Its readers recompute the hash from the file's bytes, so a smaller line changes the hash and breaks nothing [read].
  - So the signed form does not change.
- Ruling 2 (b): every reader of `commit_ms` and `sign_ms` [read]:
  - `warden_serve.check_cells` (l.996–1008): their quantiles go into the check's summary under `cell_rows`.
  - The check's printed line (l.1148).
  - `warden_zk.py` l.141, l.454 and l.469 pass the whole `cost` into its summary and its line.
  - `test_warden_serve.py` l.768.
  - `connection_rows_run.py` has its own `sign_ms` in its own files, not the device's registration.
- A second public path for the two timings: `auditor.WARDEN_COUNTS` holds `cell_rows`, so `warden_outcome` copies the check's timing quantiles into `end.json`'s verdict (l.2646–2648) [read].
  - My plan: the device writes the two timings to a witness file `cells-timing.jsonl` beside `cells.salts` (not in `WARDEN_PUBLIC`).
  - `check_cells` reads that file into a summary key of its own, `cell_timing`, which `WARDEN_COUNTS` does not list. They then stay in `check.json` (witness) and the check's line, and leave `end.json`.
  - The check refuses a registration carrying either field, by name.
  - I take yes unless you say no.
- Ruling 1: one more reader. `network_e2e/e2e.py` l.264 reads `upstream`, but from `certifier_serve.py`'s own `deployment.json` (l.1028–1036: `link`, `upstream`), the out side's document, which no auditor list publishes [read]. I leave that document out of this change.
- Ruling 1's test "the manifest declares exactly the published files" does not hold as stated [read].
  - `WARDEN_PUBLIC` also publishes `groups.json` (the window-per-group schedule) and `load-relay.json` (a load relay's counts and the sink's acknowledgements). Neither has a message.
  - `auditor.warden_outcome` also writes `certificates.jsonl` to `out/`, and its message is `certificate`.
  - My default: the test pairs `deployment.json`→`deployment`, `cells.jsonl`→`registration`, `load.json`→`load` and `certificates.jsonl`→`certificate`. It names `groups.json` and `load-relay.json` as published with no message yet, and fails on any other.
  - Your call: declare those two (kinds?), stop publishing them, or keep my default.
- I build the follow-up now on #10193's head `faf6a5984a64481382005cbb26ba32f4f573ff64`, branch `cursor/network-accounting-deployment-split-b51f`. No run is launched for it but its suites.
TIMER: none

