module

public import CoreProtocol.Cli
public import CoreProtocol.Example

@[expose] public section

def main (args : List String) : IO UInt32 := Core.Example.protocol.main args
