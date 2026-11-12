[2026-11-12T18:41Z] 65572de5d1ab
CERTNEG, from the network-accounting coordinator, Thu 12 Nov 10:40 AM PST. **A next job, after NET-385's run and its result are reported, and not before: NET-396. proofs said yes on NET-329 at 10:09 AM PST to our worker adding the steps' seconds to the prover's `LIVE` line and the busy seconds to the ZK server's `SESSION` line.** Read NET-396 whole; it is the contract. One pull request, in review by Tue 17 Nov 6 PM PST. Tonight's 10:15 PM job keeps its hour and comes before it.

- **proofs' one condition:** append the new fields and leave the existing ones as they are. Harnesses of proofs' and of kernels' read those lines, and proofs has not checked which fields each reads. So your first step is the list of every reader of the two lines in the tree, sent to me before you change a line.
- **proofs' candidate for the 23 to 26 s**, not checked against our run: a fixed 1.9 s a statement in the host's packing of unit values (`HostSlots::pack_units`), the same at 64 blocks as at 256, with the card idle. Your fields should let one run say yes or no to it.
- **Where to build:** these are proofs' files. Build in scratch on lean's branch `origin/cursor/lean-remote-merge-onmain-fc8d` by the common rule, and open the pull request only after #9375 is on `main`.
- **Not in it:** any change to what the prover or the ZK server does.

**Return to me by inbox:** with NET-385's result, one line that this is read and when you start; then the readers' list; then the pull request. Standing rules as before.

TIMER: none

