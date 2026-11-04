import CoreProtocol.Cli
import CoreRemote

def main (args : List String) : IO UInt32 := Core.Remote.V1.protocol.main args
