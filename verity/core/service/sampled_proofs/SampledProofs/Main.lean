module

public import CoreProtocol.Cli
public import SampledProofs.OneStage
import Flock.Sizes
import Std.Data.HashSet
import Std.Data.TreeMap.Raw.Lemmas

@[expose] public section

def main (args : List String) : IO UInt32 := SampledProofs.OneStage.protocol.main args
