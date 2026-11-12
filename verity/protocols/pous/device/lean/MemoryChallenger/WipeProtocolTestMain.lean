module

public import MemoryChallenger.WipeProtocol

/-! `wipe-protocol-test`: the core protocol command line (`Core.Protocol.main`) of
`MemoryChallenger.WipeProtocol.testProtocol`, `wipe-protocol`'s verdict under a protocol id of its own, which accepts
the test signers' schemes and only them. It reads and prints what `wipe-protocol` does: `id`, `check PARAMS COMMITMENT
EVIDENCE` on three files, or `serve`, one case a line on standard input and one outcome line out for each.
`tests/test_wipe_protocol.py` drives `serve` against `wipe_protocol_case.test_answer` on sampled sessions. -/

@[expose] public section

def main (args : List String) : IO UInt32 := MemoryChallenger.WipeProtocol.testProtocol.main args
