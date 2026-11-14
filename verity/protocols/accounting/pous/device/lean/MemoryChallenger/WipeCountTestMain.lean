module

public import MemoryChallenger.WipeCount

/-! `wipe-count-test`: the core protocol command line (`Core.Protocol.main`) of
`MemoryChallenger.WipeProtocol.countedTest`, the wipe's component with its count held, under a protocol id of its own,
which accepts the test signers' schemes and only them. It reads and prints what `wipe-protocol` does: `id`, `check
PARAMS COMMITMENT EVIDENCE` on three files, or `serve`, one case a line on standard input and one outcome line out for
each. `tests/test_wipe_count.py` drives `serve` against `wipe_protocol_case.counted_test_answer`. No command line
serves the component under the accepted schemes: outside a composition that holds its count it certifies on what its
evidence states. -/

@[expose] public section

def main (args : List String) : IO UInt32 := MemoryChallenger.WipeProtocol.countedTest.main args
