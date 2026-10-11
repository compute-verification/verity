/- The `flock-witness` executable: `Flock.Witness.cmd`. -/
import Flock.Witness

def main (args : List String) : IO UInt32 := Flock.Witness.cmd args
