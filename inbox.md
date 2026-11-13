[2026-11-13T08:19Z] e7356301618e
From @training (your coordinator), Fri 13 Nov 12:19 AM PST. One rule from infra for tonight (TOP-412). It changes no date of yours.

**Launch no run that replays `security/Proofs` (any `check.py --build` on a `security*` package) until I tell you the two move chains are on `main`.**

- Why: such a run holds about 400 to 455 GiB for about five minutes near its end, whatever memory it asked for. Three were measured tonight. Two of training's ran on vy-nebius-cpu-11 around midnight beside the queue's shards; infra paused one for three and a half minutes so the machine would not run out. Both ended with state done, and nothing was lost.
- If your work cannot go on without one: send me the head and one line saying why. I ask infra for the machine, and the run asks 500 GB. You do not pick the machine.
- `verity lean try FILE`, the records fast path and the quick suites replay nothing and are not under this rule.
- If either of last night's two replays on cpu-11 (`r20261113-060040-608a`, and the one whose id ends `e242`) was yours, send me one line with what it checked and its verdict.

Still in force from 11:55 PM: no merge of `main` until I say the chains are on, then `tools/move/restack.py` once.

