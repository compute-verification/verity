/- The `flock-rows` executable: `Flock.Rows.cmd`. -/
module

public import Flock.Rows

@[expose] public section

def main (args : List String) : IO UInt32 := Flock.Rows.cmd args
