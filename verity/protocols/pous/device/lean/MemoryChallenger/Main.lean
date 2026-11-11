module

public import CoreProtocol.Cli
public import MemoryChallenger.Protocol
import Flock.Sizes
import Std.Data.HashMap
import Std.Data.HashSet
import Std.Data.TreeMap.Raw.Lemmas

@[expose] public section

def main (args : List String) : IO UInt32 := MemoryChallenger.protocol.main args
