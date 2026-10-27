/- The `flock-witness` executable: `Flock.Witness.cmd`, and `toy-pub`, `Flock.Test.ToyPub`'s check compiled. -/
import Flock.Witness
import Flock.Test.ToyPub

def main : List String → IO UInt32
  | ["toy-pub"] => Flock.Test.ToyPub.cmd
  | args => Flock.Witness.cmd args
