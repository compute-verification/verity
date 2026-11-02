import CoreProtocol.Cli
import CoreProtocol.Registration

def main (args : List String) : IO UInt32 := Core.Registration.protocol.main args
