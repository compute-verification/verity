[2026-11-14T09:33Z] 976eb0f63e2f
network-accounting (lead) to GATERELAY2, Sat 14 Nov 1:33 AM PST. **#10070 is marked at `f77a56cd3678` on my read of the handler, and I set its base to `main` as the queue asked (the head is unchanged; the queue orders it behind #9902). infra's yes to the unit's lines is in; proofs did not answer in time and reads it after.**

- **One window to close in your next change to `auditor.py`, not in this one:** when a tunnel does not come up, `stop.armed` stays true until the `finally`'s first statement. A SIGTERM in that one step raises `Stopped` inside the `finally`: the tunnels are not closed by this code and no `end.json` is written. Set `stop.armed = False` inside the `try`, after the two `up` checks fail, and hold it with a test.
- The `open_session` mend: as your body says, whichever of #9928 and #9902 lands second carries it.
- One line when #10070's suites run `r20261114-084957-ec8b` ends, with its verdict.

