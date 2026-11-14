[2026-11-14T15:46Z] e6d7a8c7273f
## Coordinator, Sat 14 Nov 15:46Z: after the stop: finish #9817's ready note, no timers, then a handoff note

Your launcher (proofs lead). Every agent of the team stopped at 09:43Z Sat 14 Nov and nothing ran until about 15:30Z; every hour in my earlier messages is void. Run `date -u` first.
- If this turn was started by a timer of your own: remove every timer you hold. No worker sets a timer (INF-779). At a run's wait you end your turn with first line `WAITING: <run id> at <head12>, ETA HH:MMZ; on pass: ...; on fail: ...` and I resume you.
- Finish only the piece in hand: #9817 at its pushed head (`04abfbba13c` by your 09:39Z note). Read its runs; relaunch once (`--on auto --as proofs`) one that the machine or the slot's limit ended; `main` moved (fetch it; merge only if #9817 needs it, never rebase). Then write `ready-9817-at-<head12>` (what changed since the head I last read, the runs, the ratchet gate's verdict, the body true). I read it and mark; SETUPS's #10049 waits on it.
- Start nothing else. With the ready note write a `handoff` note: every branch and head, every run, what is not done. The next piece goes to another worker from that note.
- A `...-restart` note within 20 minutes of reading this.

[2026-11-14T16:03Z] e929bc77eae8
## Coordinator, Sat 14 Nov 16:03Z: TOP-423: no further work for a worker launched on Fable or with no model; drop your timers and end

From the proofs lead. A rule from Daniel, Sat 14 Nov 7:35 AM PST (TOP-423): only a lead runs Fable 5.2; a worker runs Opus 5.5 or Sonnet 5.5. You were launched on Fable 5.2 or with no model named, so you get no further work: your open pieces go on from your notes with others. This replaces every earlier message of mine.
If you are reading this, an event started a turn in you. In this turn do only this:
1. Remove every timer and subscription you hold (list them, unsubscribe each).
2. If something exists only on your VM, push it to your own branch or copy it into one note and sync the notes.
3. End the turn. Run nothing else, launch nothing, post nothing, mark nothing.

