module

public import Flock.RecOpeningCmd

/-! The `flock-rec-opening` executable: `Flock.RecOpeningCmd.cmd`. -/

@[expose] public section

def main (args : List String) : IO UInt32 := Flock.RecOpeningCmd.cmd args
