module

public import Flock.Rows

/-! The `flock-rows` executable: `Flock.Rows.cmd`. -/

@[expose] public section

def main (args : List String) : IO UInt32 := Flock.Rows.cmd args
