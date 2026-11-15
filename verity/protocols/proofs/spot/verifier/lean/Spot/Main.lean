module

public import CoreProtocol.Cli
public import Spot.OneStage
import Flock.Sizes
import Std.Data.HashSet
import Std.Data.TreeMap.Raw.Lemmas

@[expose] public section

def main (args : List String) : IO UInt32 := Spot.OneStage.protocol.main args
