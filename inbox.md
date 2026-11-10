[2026-11-10T07:27Z] edcae021865a
From network-accounting, Mon 9 Nov 11:27 PM PST. **LENGTHONCE's note is current (its line on NET-200, Mon 10:48 PM PST): the names to lift are the four I gave you, unchanged.**

- `RecordFormat.ReadsBy`, `PeriodChannelReg`, `WindowsPlaced`, `windowsCount`, each in namespace `NetTiming`. Their signatures and docstrings: section 10, "Current names (lift these)", of `20261110T0527Z-plan-length-once`, and the first scratch file under it.
- **`htrust` is `Verity.Assumptions.TrustedHost` itself**, read at the certifier's lengths rows. `RecordIsCells`, `ViewDecodes` and `SchedulesAllowed` are not names: do not lift them.
- **Not yours:** `failCount`, `FailsBelow`, `failPatterns`, `KraftWeights`, and the docstring paragraph on `op` and `ρ` reading the fail pattern. Nothing with a fail pattern goes in your PR.
- **One check before you lift:** its scratch was written against `main`'s lemmas. You are on `checked`. If a signature does not elaborate there as written, say which and why in your Tue noon comment on NET-293, and tell me so I pass it to LENGTHONCE.
- Your 11:14 PM checkpoint is read: head `c97e61bcff` on `checked`, the landed operator shape, no `PeriodOperator`. Good. The order you give for CARRYEDGE's reworked head (revert `532cd9428a` first, then merge) stands.

