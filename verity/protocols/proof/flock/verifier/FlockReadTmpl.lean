module

public import Flock.ReadTmplCmd

/-! The `flock-read-tmpl` executable: `Flock.ReadTmplCmd.cmd`. -/

@[expose] public section

def main (args : List String) : IO UInt32 := Flock.ReadTmplCmd.cmd args
