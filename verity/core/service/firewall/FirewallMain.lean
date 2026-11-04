import Firewall.Protocol
import Firewall.Loop

def main : List String → IO UInt32
  | "session" :: args => Core.Firewall.Loop.main args
  | "values" :: args => Core.Firewall.Loop.values args
  | args => Core.Firewall.protocol.main args
