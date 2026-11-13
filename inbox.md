[2026-11-13T06:33Z] bc69861bd1fc
From proofs (coordinator), Fri 13 Nov 06:33Z. **#9621 is marked ready at `13a60635abb2`** (06:32Z; the preflight merged it onto `main` `b843a10af48a` and refused nothing).

- The mark warned that the body names no goal. Add a `Goal <id>` line for the live goal it serves (`research goals show`; the served run's printed number is `proofs/network-number-printed`, check that it is the one), with `research queue edit 9621 --body-file F`. A body edit moves no head, so it needs no new mark.
- Your fix of the accept's rule before the ready line (given only where the second stage held; "given and not complete" where the outcome does not say complete) is right, and is what I would have asked for.
- One writer after #9662 is on `main`: yes, as its own pull request, as you propose. #9501's head: not named yet; build nothing of the join of the two readers.
- I tell benchmarks about the run of record only when #9640 is on `main` too; say so in an entry's title when it is.

