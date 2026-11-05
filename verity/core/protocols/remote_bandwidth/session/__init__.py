"""PoUB's sessions between two processes over one TCP stream, the verifier's side: the verifier's process (`verifier`), the
Lean verifier it runs beside `audit.AuditRun` (`lean_verifier`), its worker processes (`checks`), where a session's coins
come from (`challengers`), and the framed messages between the verifier's process and the prover's (`wire`). The prover's
process is `benchmarks/remote_session/prover.py`."""
