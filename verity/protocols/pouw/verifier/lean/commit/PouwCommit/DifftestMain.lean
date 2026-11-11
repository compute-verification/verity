module

public import PouwCommit.Difftest
meta import PouwCommit.Difftest

/-! The difftest's run (`runs` in `lean-audit.json`): `PouwCommit.Difftest.main`, audited under `PouwCommit`. It is the one
test that holds the Python's coins check, bindings and tree roots to the Lean, PoUW's verifier: each side's own
tests pass a definition that drifts from the other. -/

@[expose] public section

def main (args : List String) : IO UInt32 := PouwCommit.Difftest.main args
