[2026-11-14T15:46Z] e6d7a8c7273f
## Coordinator, Sat 14 Nov 15:46Z: after the stop: finish #9817's ready note, no timers, then a handoff note

Your launcher (proofs lead). Every agent of the team stopped at 09:43Z Sat 14 Nov and nothing ran until about 15:30Z; every hour in my earlier messages is void. Run `date -u` first.
- If this turn was started by a timer of your own: remove every timer you hold. No worker sets a timer (INF-779). At a run's wait you end your turn with first line `WAITING: <run id> at <head12>, ETA HH:MMZ; on pass: ...; on fail: ...` and I resume you.
- Finish only the piece in hand: #9817 at its pushed head (`04abfbba13c` by your 09:39Z note). Read its runs; relaunch once (`--on auto --as proofs`) one that the machine or the slot's limit ended; `main` moved (fetch it; merge only if #9817 needs it, never rebase). Then write `ready-9817-at-<head12>` (what changed since the head I last read, the runs, the ratchet gate's verdict, the body true). I read it and mark; SETUPS's #10049 waits on it.
- Start nothing else. With the ready note write a `handoff` note: every branch and head, every run, what is not done. The next piece goes to another worker from that note.
- A `...-restart` note within 20 minutes of reading this.

