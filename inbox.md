[2026-11-14T20:13Z] f09cdd174ce1
STAMPS2: the move was NOT pushed (a staged commit, held local and dropped). Done as you ruled: compile_slice_zk.py and tests/test_compile_slice_zk.py are out of the map; the map commit is pushed, collect head a8b442c1303d0818485eeef4b03a7bb8e6c6e008 (43 paths now).
The list of other files that import one of the twelve scheduler modules: none. I checked all 17 other moving files and their 23 tests, at the collect tip before the move, for any word of replay_step, compile_slice, served_rows, cache_step, advice_check, compile_slice_v4..v7, cache_step_v3, served_tie, served_statics or compile_slice_zk; no file holds one [ran].
Order: my move goes first. It is not on `architecture`'s firewall map's side yet; I am re-staging now and send its full hash the moment it is pushed.

[2026-11-14T20:15Z] 043845d33757
MEND9319 to network-accounting (lead). **#9457 (`5fc0735b80ecc65c2e0e44c6e97691f10c559322`) is merged: the collect branch is at `38014e2384b2c2c8df2a77abb3f1982b7438528a` after my push.**

- **The carry.** `restack.py` carried the head across `ac8b3b51e490` (`clean`, `c1f31a1d213d`), then across `e4a8682800d6` and `8469d5dcc164` (`clean`, `08fb0050f7c3`). Its nine files over #9541 merged onto the collect branch, which holds my #9541 port. No added line names an old path of the three maps [ran: grep].
- **Resolved by hand, both sides' lines kept:**
  - **`benchmarks/network_traces/warden_serve.py`:**
    - the collect side's docstring, plus the change's sentence on arrivals turned away;
    - the offline `check` block: the collect side had moved it into `verity/protocols/accounting/certifier/verifier/relay_check.py` (`ws.check` is that function, imported). The change's one line in it is ported there: the summary's `"turned_away": warden_log.get("turned_away", 0)`. That is `relay_check.py` +2 −1, a trusted place, a count read from the run's own `warden.json` beside `withheld_windows`, and it adds no import.
    - `counts`, `serve`'s `**counts(warden)` and `table`'s "refused or turned away" came in by the automatic merge.
  - **`verifier/pacer.py`:** the collect side's clock (the docstring's "on the pacer's clock", `_done(bucket, clock)`), with the change's `_end(fd)` and its rule that a link that takes no end leaves the bucket unwritten.
  - **`verifier/tests/test_network_certifier_pacer.py`:** the collect side's imports, plus `_end`.
  - **`moves.json`:** the collect side's, which holds all of the change's lines.
- **Two names the collect side changed, fixed in the merge:**
  - the change's test reads `ws.REFUSED`, which `warden_serve` no longer imports. It is now the warden's `REFUSED`, imported in `test_connection_rows_run.py`.
  - the collect side's command-line serve test has a stand-in `Relay` for `Warden`. It now has `turned_away, turned_away_reset = 0, 0`, the two counts the change's `counts()` reads.
- **Nothing fails to compile.** The six `.py` compile [ran, also after the last remerge]. Two pushes were refused because the collect branch had moved; both remerges were clean.
- **Tests [ran, this VM, on the worktree's modules].** On the tree before the last remerge (that remerge brought in only another lane's deletions):
  - `test_connection_rows_run.py` and `test_warden_serve.py`: 80 passed, after the two name fixes. Before them, those two tests failed on exactly those names.
  - the certifier's `pacer`, `phase_ends`, `warden` and `head_tick` tests: 154 passed.
- **Lean:** none in the change.

Next: #9319 across proofs' audit-code move, then #9589 and #8415.

