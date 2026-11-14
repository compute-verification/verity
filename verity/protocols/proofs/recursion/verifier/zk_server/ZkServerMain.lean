module

public import ZkServer.Protocol
public import ZkServer.Loop

@[expose] public section

def main : List String → IO UInt32
  | "session" :: args => Core.ZkServer.Loop.main args
  | "values" :: args => Core.ZkServer.Loop.values args
  | args => Core.ZkServer.protocol.main args
