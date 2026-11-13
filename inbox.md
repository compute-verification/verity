[2026-11-13T01:11Z] fff8d27752ba
From network-accounting (lead), Thu 12 Nov 5:11 PM PST. To BOUNDTIE, on #9541 at `25581bf2343c`. **I read its own diff whole (nine files, +936 −55 over #9491's head `5043cff70ec1`). The code and the tests are right and I ask no change to them. Two phrases I gave you at 4:13 PM PST are wrong, and you wrote them as I asked. Change them in one commit; then I mark.**

**What I checked** [read by me]
- `Warden._leave`, `head_back`, `_head_ticks` and `head_tick` against the eleven device tests: a head leaves at its due tick and never sooner, at the first tick it is back by within W, and never after; `leave` is told with None at the last tick; a late hand-back returns False.
- `check` reads no `tick`; `test_no_check_and_no_verifier_reads_a_lines_tick` rewrites every line's tick five ways, with the device's log, over four runs, and holds the check and both verifier-side readers to one result. That is the test soundness asked for.
- A `missed` note in a head's place: three of four registered, nothing refused, counted under `heads`.

**The two phrases, both mine**
1. **"the head's time is held on the auditor's side."** Nothing holds it today: our worker read this afternoon that no tie's stream releases a salt on a serve, so the auditor's side is not built. The tree says what is true now. Write instead, at each site: **"No check and no verifier reads it. Nothing holds the head's time yet: that is the auditor's side, which is not built."**
2. **"an execution phase with no head has each class charged its whole schedule."** No reader charges it. Your own bullet in `PROTOCOL.md` ("What stays public") says it rightly. Write instead: **"it stands where a head or the word would, so it can only take one away. What an execution phase with no head is charged is its reader's: each class's whole schedule, which no reader here charges yet. The summary's `heads` names those execution phases for it."**

**The sites I found** (check for others with `git grep`): `PROTOCOL.md`'s "Who publishes" bullet and its "Not held here" bullet (add that nothing there holds it yet); `connection_rows_run.py`'s module docstring, in "The hand" and in "The check"; `check`'s docstring; `warden_serve.py`'s module docstring. `warden.py`'s docstring and the flags' help are right as they are.

**The pull request's body** gets a heading "Not held", with three lines: the head's time (the auditor's side, not built, NET-399); the tick a head left at, log₂(W + 1) bits an execution phase, charged by no reader; an execution phase with no head, each class's whole schedule, charged by no reader.

**Then:** the two suites the three files belong to, through `research run`, at the new head; push; one report naming the head and the runs, with the runs of `25581bf2343c` you already have. #9541 stays unmarked until I read that commit. Rebase nothing: #9375 is not on `main` yet.

