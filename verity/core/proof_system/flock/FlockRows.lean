/- The `flock-rows` executable: `Flock.Rows.cmd`. -/
import Flock.Rows

def main (args : List String) : IO UInt32 := Flock.Rows.cmd args
