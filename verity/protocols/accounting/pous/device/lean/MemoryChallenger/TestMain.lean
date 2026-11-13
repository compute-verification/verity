module

public import CoreProtocol.Cli
public import MemoryChallenger.Protocol
import Flock.Sizes
import Std.Data.HashMap
import Std.Data.HashSet
import Std.Data.TreeMap.Raw.Lemmas

/-! `memory-challenger-test`: the core protocol command line (`Core.Protocol.main`) of `MemoryChallenger.testProtocol`,
`memory-challenger`'s verdict under a protocol id of its own, which accepts the test signers' schemes and only them. It
reads and prints what `memory-challenger` does: `id`, `check PARAMS COMMITMENT EVIDENCE` on three files, or `serve`, one
case a line on standard input and one outcome line out for each. `tests/test_protocol.py` drives `serve` against
`memory_challenger.verdict` under `pous_test_signer.SCHEMES`. -/

@[expose] public section

def main (args : List String) : IO UInt32 := MemoryChallenger.testProtocol.main args
