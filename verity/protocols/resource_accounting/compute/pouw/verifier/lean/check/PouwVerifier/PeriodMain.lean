module

public import PouwVerifier.Period

/-!
# `pouw-period`: PoUW's credit-period protocol as an executable

The executable's entry point, which serves `PouwVerifier.Period.protocol` through core's command line.
-/

@[expose] public section

def main (args : List String) : IO UInt32 := PouwVerifier.Period.protocol.main args
