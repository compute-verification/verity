module

public import VStarRecursive.Sampled
public import VStarRecursive.Protocol

@[expose] public section

def main (args : List String) : IO UInt32 := VStar.Recursive.Sampled.main args
