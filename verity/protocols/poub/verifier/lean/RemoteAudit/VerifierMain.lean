module

public import RemoteAudit.Verifier

/-! `remote-audit-verifier`: `RemoteAudit.Verifier.serve`, the verifier's process, which holds the registry and every
session's state. It reads one request a line on standard input, a JSON object with its `op` (after some, the bytes the
request counts), until the input ends, and prints one line for each, `{"answer": JSON}` or `{"error": MESSAGE}`, and a
line `{"done": id, "accepted": bool}` when a check of openings it started for a `verify` request ends; it exits 0.
`session/lean_verifier.py` and `p2_lean.py`, in the Python package this one sits in, start it. -/

@[expose] public section

def main : IO UInt32 := RemoteAudit.Verifier.serve
