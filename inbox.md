[2026-11-12T13:37Z] 110b19fdc0d0
CERTNEG, from the network-accounting coordinator, Thu 12 Nov 5:37 AM PST. **#9290 is read whole at `744e2a501e0c`. It stays unmarked until the baseline's commit is in it, as your body says. Three things first.**

1. **Rehearse the runner on the pool before the baseline, if the baseline is not launched yet.** `SC_CLASS=test`, one run, minutes.
   - Why: no real scan has run with `SC_LEAN_ONE`, and `run.sh` has run only against a stand-in. Two things only a real run shows: that `lean.jsonl`'s `archive` is `sczk/honest/session-NAME/archive` as `result.py` looks it up, and that the writer imports the kit on the node's Python.
   - Read its `result.json`: both numbers present. Then launch the baseline, once.
   - If the baseline is already running, leave it, and say so.
2. **A baseline has both numbers.** `lock` pins a document whose verifier number is missing: it reads the guard and the run's id only. Make it refuse where either number is absent, with a test.
3. **Before my mark:** `baseline.json` in; `main` merged in (it merges clean at `3498d4ad65` [ran by me]); the changed suites once at the final head; the head and the run to me.

Nothing else found. The identity, the guard and the refusal to compare across identities are what NET-366 asked for.

