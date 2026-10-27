import PouwWindow.RegFile

/-! `pouw-leaf-own`: the leaf verifier's check of a drawn-leaf statement's `--registered` file against the accepted
registration and produced record (`PouwWindow.Lanes.leafOwnCmd`). -/

def main (args : List String) : IO UInt32 := PouwWindow.Lanes.leafOwnCmd args
