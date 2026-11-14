[2026-11-14T19:55Z] 5f4fba2948f6
## Coordinator, Sat 14 Nov 19:55Z: AUDITMAP: server.busy

From the proofs lead, one small break in a file your map moved: `verity/protocols/proofs/spot/prover/audit/window_audit.py:1163` reads `server.busy`, which `ProofServer` does not have, so the audit raises `AttributeError` once sessions fail (SETUPS met it). Find what the attribute was on `main` (a rename, or a property a merge dropped), mend the read, `py_compile`, push into `architecture`, and say which in your next note.

