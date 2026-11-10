"""PoUB's sessions between two processes over one TCP stream, the verifier's side: the verifier (`verifier`), the Lean
verifier it runs beside `audit.AuditRun` (`lean_verifier`), its worker processes (`checks`), where a session's coins
come from (`challengers`), and the framed messages between the verifier's process and the prover's (`wire`). The
verifier's process is `verity/protocols/poub/verifier/remote_session/verifier_process.py`, which hands `verifier.main` what a verifier
imports none of: the LMS verify a served or gate session's signatures are checked with, and the challenger's own
process. The prover's process is `benchmarks/remote_session/prover.py`."""
