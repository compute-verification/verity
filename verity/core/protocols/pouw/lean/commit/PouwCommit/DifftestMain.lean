import PouwCommit.Difftest

/-! The difftest's run (`runs` in `lean-audit.json`): `PouwCommit.Difftest.main`, audited under `PouwCommit`. It is the one
test that holds the Python's coins check, bindings and tree roots to the port: each side's own tests pass a definition that
drifts from the other. -/

def main (args : List String) : IO UInt32 := PouwCommit.Difftest.main args
