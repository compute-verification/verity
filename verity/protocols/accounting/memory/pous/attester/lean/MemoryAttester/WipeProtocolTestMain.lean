module

public import MemoryAttester.WipeProtocol

/-! `wipe-protocol-test`: the core protocol command line (`Core.Protocol.main`) of
`MemoryChallenger.WipeProtocol.testProtocol`, `wipe-protocol`'s verdict under a protocol id of its own, which accepts
the test signers' schemes and only them. It takes `wipe-protocol`'s forms, each held to the served node NAME and the
fill floor BYTES (`--server NAME --fill BYTES serve`, `--server NAME --fill BYTES check PARAMS COMMITMENT
EVIDENCE`), and the test's form, `serve` and `check` held to no node, which no deployed command has.
`tests/test_wipe_protocol.py` drives `serve` against `wipe_protocol_case.test_answer` on sampled sessions. -/

@[expose] public section

def main (args : List String) : IO UInt32 := MemoryAttester.WipeProtocol.testMain args
