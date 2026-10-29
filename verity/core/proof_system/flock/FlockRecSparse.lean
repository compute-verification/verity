/- The `flock-rec-sparse` executable: `Flock.RecSparseCmd.cmd`. -/
import Flock.RecSparseCmd

def main (args : List String) : IO UInt32 := Flock.RecSparseCmd.cmd args
