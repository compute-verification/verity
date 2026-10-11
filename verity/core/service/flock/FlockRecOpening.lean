/- The `flock-rec-opening` executable: `Flock.RecOpeningCmd.cmd`. -/
import Flock.RecOpeningCmd

def main (args : List String) : IO UInt32 := Flock.RecOpeningCmd.cmd args
