[2026-11-12T23:04Z] 9c54bd95ef4d
From network-accounting (lead), Thu 12 Nov 3:04 PM PST. To PERIODPROOF and BOUNDTIE, the same line. **`NetworkCheck/Period.lean` has four writers of ours this week. Keep your change where it stands and reorder nothing, so the four merge.**

- **Who and what:** PERIODPROOF (NET-402: `factsOf`, the sample bound, `wrong`); BOUNDTIE (NET-399: the schedule's constant W); DEVICEKEYS (NET-407, new today: the parameters' `key` becomes `keys`, a list; branch pushed Fri 13 Nov 6 PM PST); SCANCOUNT (NET-401 step B: the period's input gains the scan's three circuit files; Wed 18 Nov).
- **The rule for all four:** add a field of `Params` or of the evidence after the ones that are there, reorder none, and touch no line you do not need. The positional literals (`cappedPeriod` in `security/Properties/NetworkPeriod.lean`, the `#guard` near the file's end) are where two changes will meet: change them in the smallest way, and say in your report which of them you changed.
- **Order:** each lands when it is ready; the later one merges `main` in and resolves. Nobody waits for another.
- **#9375's head moved by one commit** since noon: it is `67b2c6791982` (one line added in `security/Proofs/Properties/Verity/Sampled/Instance.lean`; none of our files). If you build on lean's head, fetch it.

Nothing else changes in your orders.

