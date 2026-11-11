/- The `flock-rec-opening` executable: `Flock.RecOpeningCmd.cmd`. -/
module

public import Flock.RecOpeningCmd

@[expose] public section

def main (args : List String) : IO UInt32 := Flock.RecOpeningCmd.cmd args
