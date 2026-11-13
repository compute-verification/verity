module

public import RemoteAudit.P2Protocol

/-! `remote-audit-p2-protocol`: the core protocol command line (`Core.Protocol.main`) of
`RemoteAudit.P2Protocol.protocol`, `verity/remote-bandwidth/p2/v0`, a PoUB session over PoUS's P2 store. `id` prints
the protocol id; `check PARAMS COMMITMENT EVIDENCE` reads three files and prints the outcome's canonical JSON line;
`serve` reads one case a line on standard input, `{"commitment": hex, "evidence": hex, "params": JSON}`, and prints one
outcome line for each. An accepted session's outcome publishes `eps`, `q_bytes` and `span_ns`. `p2_case.py` writes its
cases and runs `serve` on one (`judged`), and `tests/test_p2_case.py` holds its answers to each session's own. -/

@[expose] public section

def main (args : List String) : IO UInt32 := RemoteAudit.P2Protocol.protocol.main args
