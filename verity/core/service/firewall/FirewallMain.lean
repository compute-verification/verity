module

public import Firewall.Protocol
public import Firewall.Loop

@[expose] public section

def main : List String → IO UInt32
  | "session" :: args => Core.Firewall.Loop.main args
  | args => Core.Firewall.protocol.main args
