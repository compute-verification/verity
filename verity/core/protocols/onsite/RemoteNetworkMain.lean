import CoreProtocol.Cli
import CoreRemoteNetwork

def main (args : List String) : IO UInt32 := Core.RemoteNetwork.protocol.main args
