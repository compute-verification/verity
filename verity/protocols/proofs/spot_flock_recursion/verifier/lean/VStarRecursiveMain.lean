module

public import VStarRecursion.Protocol
public import VStarRecursive.Sampled
public import VStarRecursive.ChallengerPin

@[expose] public section

/-- `vstar-recursive`'s command line: `sampled [--challenger-key-sha512 H [--challenger-retired FILE]] ...` SPOT's
(`VStar.Recursive.Sampled.protocol`, under the pin when given: `ChallengerPin.sampled`; `Core.BlobProtocol.main`),
anything else the recursive audit's (`VStar.Recursive.main`). -/
def main : List String → IO UInt32
  | "sampled" :: args =>
    Spot.ChallengerPin.pinMain (fun pin => (VStar.Recursive.ChallengerPin.sampled pin).main)
      VStar.Recursive.Sampled.protocol.main args
  | args => VStar.Recursive.main args
