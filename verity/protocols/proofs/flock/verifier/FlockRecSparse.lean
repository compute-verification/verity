module

public import Flock.RecSparseCmd

/-! The `flock-rec-sparse` executable: `Flock.RecSparseCmd.cmd`. -/

@[expose] public section

def main (args : List String) : IO UInt32 := Flock.RecSparseCmd.cmd args
