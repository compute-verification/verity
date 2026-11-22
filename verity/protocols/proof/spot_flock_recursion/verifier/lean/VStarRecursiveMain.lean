module

public import VStarRecursion.Protocol
public import VStarRecursive.Sampled

@[expose] public section

/-- `vstar-recursive`'s command line: `sampled ...` SPOT's (`VStar.Recursive.Sampled.protocol`, `Core.BlobProtocol.main`),
anything else the recursive audit's (`VStar.Recursive.main`). -/
def main : List String → IO UInt32
  | "sampled" :: args => VStar.Recursive.Sampled.protocol.main args
  | args => VStar.Recursive.main args
