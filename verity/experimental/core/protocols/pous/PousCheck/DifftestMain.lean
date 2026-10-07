import PousCheck.Difftest

/-! The difftest's run (`runs` in `lean-audit.json`): `PousCheck.Difftest.main`, audited under `PousCheck`. -/

def main (args : List String) : IO UInt32 := PousCheck.Difftest.main args
