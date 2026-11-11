module

public import CoreProtocol.Cli
public import CoreProtocol.Registration

@[expose] public section

def main (args : List String) : IO UInt32 := Core.Registration.protocol.main args
