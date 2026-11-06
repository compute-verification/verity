module

public import SampledProofs.OneStage

@[expose] public section

def main (args : List String) : IO UInt32 := SampledProofs.OneStage.protocol.main args
