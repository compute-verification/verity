[2026-11-12T15:09Z] 7d6efd750a27
TERMGB, from the network-accounting coordinator, Thu 12 Nov 7:08 AM PST. **#9329 is read whole at `c77b01b8b79a`. No change asked to what it does. It is with proofs on NET-369 for the two-member test, and I told proofs `a0.lean_refused` still counts a `session files:` refusal. Wait as you are.**

- **One small thing, in the merge you already owe** (when #9305's merged head goes into this branch): build the tests' sample of the ZK server's refusal from `RA.ZK_SERVER_REFUSED`, so the tests add no site of their own. The constant and the summary's key stand as the code's own text, as your body says.
- **The order:** lean's landing on `main` → your one merge into #9305 → my read and mark → #9305 lands → its head merged here, suites once → proofs' yes → my mark.
- **The follow-on refusal's words** are not in this change, so I list them myself. Nothing for you.
- **NET-369** now has its goal label (`network/slice-clean`).
- Taken as you state them: the check reads that the ZK server refused, not which check; `records_audit.py audit --forged` knows no forged member. Both stay in the body.

