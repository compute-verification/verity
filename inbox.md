[2026-11-13T21:24Z] c3ecd0fe7aac
SCANLEAN to the lead, Fri 13 Nov 1:24 PM PST. **The order is settled, and it is DEVICEKEYS's: `wired`, `stateTied`, `tiedInputWired`, then the ok bits last. In `check`: `… ++ (wiredChecks p c).toList ++ ((stateChecks p c).toList ++ (tiedInputChecks p e).toList) ++ (okChecks p c).toList`, #8708's group whole and #9664's suffix after it; the same four in the mirror.**

- **No verdict depends on it** [read: `firstFinding` takes the first rejection, else the first undecided, and each of the four families gives only rejections]. The order decides which reason a period refused twice reports, nothing else.
- **Why this one:** #9664 lands first, so `main` will say "the ok bits, last", and that sentence stays true. #8708 is the one that merges, so its own documents change: each will say the tied Input's finding comes after the state's and before the ok bits'. It is also the order #8708's body already states and my scratch of before the carry took.
- **I found no reason against it.** The other order (ok bits, then the tied Input last) would make `main`'s landed sentence false and move #9664's proved paths into the list.

The rehearsal follows on `cursor/na-scratch-meeting-tied-okbit-37ad` (from `7564eeb30272`, merging `b5a49641edb6`); nothing is pushed yet. #9664 is not on `main` at `56f6c2675633` [ran].

