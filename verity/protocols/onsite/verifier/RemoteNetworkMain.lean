module

public import CoreProtocol.Cli
public import CoreRemoteNetwork

@[expose] public section

def main (args : List String) : IO UInt32 := Core.RemoteNetwork.protocol.main args
