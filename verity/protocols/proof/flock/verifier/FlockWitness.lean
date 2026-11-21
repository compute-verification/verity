module

public import Flock.Witness

/-! The `flock-witness` executable: `Flock.Witness.cmd`. -/

@[expose] public section

def main (args : List String) : IO UInt32 := Flock.Witness.cmd args
