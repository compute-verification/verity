module

public import CoreProtocol.Cli
public import PouwCheck.Protocol
import Flock.Sizes
import Std.Data.HashMap
import Std.Data.HashSet
import Std.Data.TreeMap.Raw.Lemmas

@[expose] public section

def main (args : List String) : IO UInt32 := PouwCheck.protocol.main args
