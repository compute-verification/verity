import PouwWindow.Difftest

/-! The difftest's run (`runs` in `lean-audit.json`): `PouwWindow.Difftest.main`, audited under `PouwWindow`. It holds the
window's Lean to `window.py` on the instances `scripts/difftest_vectors.py` generates; `Security/Properties/Pouw.lean`
states what the Lean must satisfy. -/

def main (args : List String) : IO UInt32 := PouwWindow.Difftest.main args
