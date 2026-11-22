module

public import CoreProtocol.Cli
public import CoreOnsite

@[expose] public section

def main (args : List String) : IO UInt32 := Core.Onsite.protocol.main args
