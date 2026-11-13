[2026-11-13T03:43Z] 2942b099e5b2
BACKWARD2, Thu 12 Nov 7:43 PM PST

**Both runs at `3b384b35f2` passed, so the head stays where TILES is reading it; #9417's body is replaced and holds your three statements.**

- RAN r20261113-021349-f32c (suites at `3b384b35f2`, `--changed 6d95fcc68d --quick`, rc 0): "59 suites: 59 passed (0 cached), 0 failed; 156.2 min of test time".
- RAN r20261113-021421-6f37 (circuit-check on both sidecars at `3b384b35f2`, rc 0): "circuit-check: 16 target(s), 0 failure(s), 14 unchecked (warnings, which never fail) on 14 Definition(s)".
- RAN on the VM: two mutations of the new Pre, each against the whole test file. Taking dim 0's product as the FMUL (the source's order) and taking FMUL then FADD are each caught by both dP-sum tests.
- READ (`gh pr view 9417`): the body is replaced, the PR is still a draft, the head is `3b384b35f2`, and the Lane, Refs and Goal lines are kept. It states, in plain words:
  - the review's first must was right that the sum is fused and wrong on which product is rounded alone: its fix took dim 0's, and the second probe told that it is dim 1's;
  - default mode has no Definition, and 55 of its 2,097,152 dQ words differ from the card's;
  - only the three changed orders were fitted, so every other order of dims 0 and 2 to 7 is not shown to be separated by any captured or probed head.
- The headline now says only what holds of each row: every dQ, dK and dV word of the 32 deterministic heads and the 40 probe heads, and that default mode has no Definition.

I hold the head until TILES's must-fixes come through you. The handoff note is current.

