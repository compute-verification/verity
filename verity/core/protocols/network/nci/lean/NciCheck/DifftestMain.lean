import NciCheck.Difftest

/-! The difftest's run (`runs` in `lean-audit.json`): `NciCheck.Difftest.main`, audited under `NciCheck`. It is the one
test that holds the Lean check to the Python reference, which computes the same numbers straight from the definitions. -/

def main (args : List String) : IO UInt32 := NciCheck.Difftest.main args
