module

public import CoreProtocol.Cli
public import CoreOnsite
public import VStarRecursive.ChallengerPin

@[expose] public section

def main (args : List String) : IO UInt32 :=
  VStar.Recursive.ChallengerPin.compositeMain Core.Onsite.protocolOf Core.Onsite.components args
