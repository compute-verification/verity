import CoreProtocol.Cli
import CoreRemote

def main (args : List String) : IO UInt32 := Core.Remote.protocol.main args
