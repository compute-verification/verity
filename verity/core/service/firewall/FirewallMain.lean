module

public import CoreProtocol.Cli
public import Firewall.Loop
public import Firewall.Protocol
import Flock.Sizes
import Std.Data.HashMap
import Std.Data.HashSet
import Std.Data.TreeMap.Raw.Lemmas

@[expose] public section

def main : List String → IO UInt32
  | "session" :: args => Core.Firewall.Loop.main args
  | "values" :: args => Core.Firewall.Loop.values args
  | args => Core.Firewall.protocol.main args
