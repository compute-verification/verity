module

public import PouwCheck.LeafOwn

/-! `pouw-leaf-own-window`: the leaf verifier's check of a window unit's leaf statement's `--registered` file at the
unit's placement, from the window's descriptor and round log (`PouwCheck.LeafOwn.leafOwnCmd`). -/

@[expose] public section

def main (args : List String) : IO UInt32 := PouwCheck.LeafOwn.leafOwnCmd args
