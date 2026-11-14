module

public import PouwCheck.Period

/-!
# `pouw-period`: PoUW's credit-period protocol as an executable

The executable's entry point, which serves `PouwCheck.Period.protocol` through core's command line.
-/

@[expose] public section

def main (args : List String) : IO UInt32 := PouwCheck.Period.protocol.main args
