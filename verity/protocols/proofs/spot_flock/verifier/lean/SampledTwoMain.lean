module

public import VStarRecursive.SampledTwo

@[expose] public section

def main (args : List String) : IO UInt32 := VStar.Recursive.SampledTwo.protocol.main args
