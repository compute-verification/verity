/- The `flock-witness` executable: `Flock.Witness.cmd`. -/
module

public import Flock.Witness

@[expose] public section

def main (args : List String) : IO UInt32 := Flock.Witness.cmd args
