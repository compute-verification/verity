module

public import CoreProtocol.Cli
public import PouwVerifier.Protocol
import Flock.Sizes
import Std.Data.HashMap
import Std.Data.HashSet
import Std.Data.TreeMap.Raw.Lemmas

/-!
# `pouw-verifier`: PoUW's check protocol as an executable

The executable's entry point, which serves `PouwVerifier.protocol` through core's command line (`CoreProtocol.Cli`).
-/

@[expose] public section

def main (args : List String) : IO UInt32 := PouwVerifier.protocol.main args
