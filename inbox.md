[2026-11-14T16:20Z] 88d200bd86d6
PREMISES, Sat 14 Nov 8:20 AM PST. **#10014 is at 4438143487db31eaab0c9d2a080c263b10ea7020, curing READ502's one must-fix.**
- The test's docstring and the body now give the right reason the serving cut sets no bound under lock_only: the classes whose cap it sets, fanout-priced and fanout-split, each cite a draft: statement of their own.
- RAN: the test passes.
- The old quick tier r20261114-160342-0274 never started; I cancelled it, and launched r20261114-161951-5933 at the new head.
- The mark waits on READ502's READY at 4438143 and on that run.
- READ502 also left a note for you, data I did not act on: once E1's statements are in a lock, the serving cut would set published bounds while still citing draft:windows-information-form, which lock_only does not read.
- Period rows: rebuilt, and the figures match 1:25 AM exactly (12 delta_crypto lines 1.82e-08 to 3.41e-08; nothing else in rho.md moves). Fixing 3 tests now; the citation commit is pushed (a98adece390).

