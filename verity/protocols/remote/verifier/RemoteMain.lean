module

public import CoreProtocol.Cli
public import CoreRemote

@[expose] public section

def main (args : List String) : IO UInt32 := Core.Remote.protocol.main args
