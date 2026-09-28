import Difftest
import PousCheck.Difftest

/-! The difftest's run (`runs` in `lean-audit.json`): `PousCheck.Difftest.main`, audited under `PousCheck`. With the one
argument `serve`, `PousCheck.Difftest.transcript` behind `Difftest.serve`, which `interop.py` drives against `wire.py`. -/

def main (args : List String) : IO UInt32 := do
  if args == ["serve"] then
    Difftest.serve (Difftest.table [("transcript", PousCheck.Difftest.transcript)])
    return 0
  PousCheck.Difftest.main args
