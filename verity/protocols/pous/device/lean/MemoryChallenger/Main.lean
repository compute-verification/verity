module

public import CoreProtocol.Cli
public import MemoryChallenger.Protocol
import Flock.Sizes
import Std.Data.HashMap
import Std.Data.HashSet
import Std.Data.TreeMap.Raw.Lemmas

/-! `memory-challenger`: the core protocol command line (`Core.Protocol.main`) of `MemoryChallenger.protocol`, the
verdict on a signed session of PoUS's timed rounds under the accepted schemes (S1's LMS). `id` prints the protocol id;
`check PARAMS COMMITMENT EVIDENCE` reads three files and prints the outcome's canonical JSON line; `serve` reads one
case a line on standard input, `{"commitment": hex, "evidence": hex, "params": JSON}`, and prints one outcome line for
each. An accepted session's outcome publishes the verdict alone. `tests/test_protocol.py` drives `serve` against
`memory_challenger.verdict` under `memory_challenger.SCHEMES`. -/

@[expose] public section

def main (args : List String) : IO UInt32 := MemoryChallenger.protocol.main args
