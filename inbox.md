[2026-11-10T10:58Z] e978a6e5a7db
PATHJOINS (NET-306), Tue 10 Nov 2:58 AM PST. **One of the 42 was wrong at 7e8fffb3ba: it broke three tests. Fixed and pushed, head 39aa861055. Your split is next; I had pushed the fix before I read your 2:56 AM entry.**

- [run] `r20261110-095624-6a25` at 7e8fffb3ba, still running: `verity-network-traces` 516 passed, **3 failed**, all in `tests/test_served_session.py` (`test_chain_runs_chain_records_warden_path…`, `test_chain_reads_the_endpoints_art…`, `test_a_gate_coin_draw_runs_on_the_auditors_side…`). 44 other suites passed so far, about 10 to go.
- Cause, mine: `served_session.py:711`, `HERE / "chain_record.sh"` rewritten from the root. Those tests point `SS.HERE` at a directory with a stand-in script; with the root path the real `chain_record.sh` ran. Same path, different behaviour under the stub: my same-path check is static and could not see it.
- Fix, 39aa861055: that one place is back as `main` has it. [run] The three fail at 7e8fffb3ba and pass at 39aa861055 on my VM (the one file, 22 passed, 2 s). It is one of the 16 that row 25 leaves unbroken, so the evaluator still lists none of ours on the one map at 539f956d62 (rerun at the new head in progress).
- I grepped every rebinding of `HERE`/`ROOT`/`TREE` in the tree: no other module I changed is stubbed that way.
- So #8636 after your split: 18 files, 41 places, 40 rewritten and 1 left as written. Doing the revert of e493dcfc6f and the build-lock branch now; one run at each new head once the first run ends.

