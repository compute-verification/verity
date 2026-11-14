module

public import CoreProtocol.Cli
public import CoreRemoteNetworkNoPouw

@[expose] public section

def main (args : List String) : IO UInt32 := Core.RemoteNetworkNoPouw.protocol.main args
