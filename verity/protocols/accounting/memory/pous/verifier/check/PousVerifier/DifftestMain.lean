module

public import Difftest
public import PousVerifier.Difftest
meta import Difftest
meta import PousVerifier.Difftest

/-! The difftest's run (`runs` in `lean-audit.json`): `PousVerifier.Difftest.main`, audited under `PousVerifier`. With the one
argument `serve`, `PousVerifier.Difftest.transcript` behind `Difftest.serve`, which `interop.py` drives against `wire.py`. -/

@[expose] public section

def main (args : List String) : IO UInt32 := do
  if args == ["serve"] then
    Difftest.serve (Difftest.table [("transcript", PousVerifier.Difftest.transcript)])
    return 0
  PousVerifier.Difftest.main args
