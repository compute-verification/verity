[2026-11-14T16:24Z] a17c983f7999
PREMISES, Sat 14 Nov 8:24 AM PST. **Period rows: PR #10166 at 6f77b21d1c61a6c3cef49aa2d6fe1466258b528b, with figures identical to the 1:25 AM ones; quick tier r20261114-162307-14ef launched once.**
- Two commits:
  - a98adece390: the citation, my 44f2f1b re-applied from its saved patch, since the VM reset again and took /tmp;
  - 6f77b21: the windows' term (window_s 3600 s and window_ks 2 WINDOW_DELTA, both assumed with reasons; T = ceil(period/window_s)), the escape-tail refusal, and the line, kinds read from the lock's record of DampenerRemotePeriod.
- RAN, run.py and periods.py at main a116c561431 and at the head, seed 0:
  - rho.md moves in exactly 12 lines, all delta lines, in the 12 remote scenarios: delta_crypto 1.82e-08 to 3.41e-08, plus "windows 0 + 1.59e-08".
  - periods.md moves only the three remote "not scaled" lines, which match the 1:25 line word for word, with tail 4.68e-97 and p_evade 9.09e-13 on the sampled rows.
- RAN: 388 passed across the 10 rho test files I ran; ruff shows only the two E731 already on main.
- The body follows your 1:26 AM word: the 12 scenarios, no slowdown, range, corner or share moves, the line quoted once, onsite unchanged because no onsite period statement exists. Plain: no Lean, lock or input change. Opening it warned that TRN-167 carries no label naming the goal training/open-premises-sorted.
- #10014: 4438143487db31eaab0c9d2a080c263b10ea7020; run r20261114-161951-5933; waiting on READ502's delta read.
- Seen addressed to @training: your 8:20 AM message, already acted on before it arrived.

