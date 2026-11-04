/- The `flock-rec-links` executable: `Flock.RecLinksCmd.cmd`. -/
import Flock.RecLinksCmd

def main (args : List String) : IO UInt32 := Flock.RecLinksCmd.cmd args
