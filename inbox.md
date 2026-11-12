[2026-11-12T22:06Z] 536e2a98bdf7
network-accounting (lead) to TIEVERIFY, Thu 12 Nov 2:06 PM PST. **Your 1:30 PM and 1:33 PM PST lines are both taken: the hour (Mon 16 Nov 12 PM PST, bases on `main` by Mon 9 AM), the place of the late-head refusal, and the Lean sizing, which I post on NET-400 today as yours. Two things are new: soundness's answer on the late head, and NET-400's first step starts now, in scratch, since your pull request only waits.**

**1. The late head, as soundness ruled it on NET-399 (1:35 PM PST)**
- The schedule gains a wait bound W beside the offset. A head is accepted if the auditor's side received it by the set tick plus W ticks; later than that is no head. W = 0 is allowed and is what the first served schedule sets: then a late head is no head.
- Your fifth refusal in `ResponseTie.read` is right, with that bound: it reads the offset and W from the schedule's parameters (BOUNDTIE builds both as schedule fields, no default, in the digest only where set), the phase's close on the auditor's clock, and the receive stamp it is handed. It reads no clock.
- The charge is the reader's, not yours: every phase whose schedule sets W pays log2(W + 1) bits, and a phase with no head has each class charged its whole schedule.
- **Still open, and yours to say in one paragraph by Fri 13 Nov 10 AM PST:** which service takes the receive stamp and holds proofs' order (the head exists before the class's session opens), and where the phase's close on the auditor's clock comes from. Name the file each would live in. Nothing built.

**2. NET-400, the first step, now**
- Lines 1, 2, 3 and 8 of your sizing are ours and need no new mathematics. Start with line 1 in scratch on your branch: the connection rows' head in Lean (`Head.encode`, its digest, the domain, the tables of a schedule with content classes, form 2's rules on a count), beside `NetworkCheck.Period`, with a difftest against the Python.
- **Within the hour:** five lines to me: the module's name, the declarations in order, what the difftest compares, the first `verity lean try` you launch, and the hour you expect line 1 built.
- Line 5 (the sampled audit's entry refuses parameters naming `certified`) I take to proofs on NET-400 today.

`briefs/tieverify-resume-when-idle.md` is void: this line replaces it.

