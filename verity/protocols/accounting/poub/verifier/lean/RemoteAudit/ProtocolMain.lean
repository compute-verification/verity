module

public import RemoteAudit.Protocol

/-! `remote-audit-protocol`: the core protocol command line (`Core.Protocol.main`) of `RemoteAudit.Protocol.protocol`,
`verity/remote-bandwidth/v0`, a PoUB session over a registered store. `id` prints the protocol id; `check PARAMS
COMMITMENT EVIDENCE` reads three files and prints the outcome's canonical JSON line; `serve` reads one case a line on
standard input, `{"commitment": hex, "evidence": hex, "params": JSON}`, and prints one outcome line for each. An
accepted session's outcome publishes `q_bytes` and `eps`. `tests/test_protocol.py` drives `serve` against the Python
verifier, and `period.py` runs it on a period's case when the period names it among its tools. -/

@[expose] public section

def main (args : List String) : IO UInt32 := RemoteAudit.Protocol.protocol.main args
