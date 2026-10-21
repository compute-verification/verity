import Firewall.Protocol
import Firewall.Json
import Firewall.Loop

def main : List String → IO UInt32
  | ["decide"] => Core.Firewall.decideMain
  | "session" :: args => Core.Firewall.Loop.main args
  | args => Core.Firewall.protocol.main args
