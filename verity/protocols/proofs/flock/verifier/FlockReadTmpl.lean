/- The `flock-read-tmpl` executable: `Flock.ReadTmplCmd.cmd`. -/
module

public import Flock.ReadTmplCmd

@[expose] public section

def main (args : List String) : IO UInt32 := Flock.ReadTmplCmd.cmd args
