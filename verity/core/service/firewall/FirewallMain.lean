import Firewall.Protocol
import Firewall.Json

def main : List String → IO UInt32
  | ["decide"] => Core.Firewall.decideMain
  | args => Core.Firewall.protocol.main args
