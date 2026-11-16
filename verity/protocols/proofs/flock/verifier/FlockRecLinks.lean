module

public import Flock.RecLinksCmd

/-! The `flock-rec-links` executable: `Flock.RecLinksCmd.cmd`. -/

@[expose] public section

def main (args : List String) : IO UInt32 := Flock.RecLinksCmd.cmd args
