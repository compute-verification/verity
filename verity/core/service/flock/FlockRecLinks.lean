/- The `flock-rec-links` executable: `Flock.RecLinksCmd.cmd`. -/
module

public import Flock.RecLinksCmd

@[expose] public section

def main (args : List String) : IO UInt32 := Flock.RecLinksCmd.cmd args
