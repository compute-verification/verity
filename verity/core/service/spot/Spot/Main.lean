module

public import Spot.OneStage

@[expose] public section

def main (args : List String) : IO UInt32 := Spot.OneStage.protocol.main args
