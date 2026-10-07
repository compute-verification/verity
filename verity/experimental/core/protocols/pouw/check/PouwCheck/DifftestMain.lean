import PouwCheck.Difftest

/-! The difftest's run (`runs` in `lean-audit.json`): `PouwCheck.Difftest.main`, audited under `PouwCheck`. It holds the
check to `core_check.py` on the cases `scripts/difftest_vectors.py` generates; `Security/Properties/Pouw/Check.lean`
states what the check must satisfy. -/

def main (args : List String) : IO UInt32 := PouwCheck.Difftest.main args
