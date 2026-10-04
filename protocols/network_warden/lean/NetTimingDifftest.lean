import NetTiming.Difftest

/-! The difftest's run (`runs` in `lean-audit.json`): `NetTiming.Difftest.main`, audited under `NetTiming`. -/

def main (args : List String) : IO UInt32 := NetTiming.Difftest.main args
