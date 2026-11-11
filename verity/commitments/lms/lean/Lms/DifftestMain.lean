module

public import Lms.Difftest
meta import Lms.Difftest

/-! The difftest's run (`runs` in `lean-audit.json`): `Lms.Difftest.main`, audited under `Lms`. It holds
`verity/ml/crypto/lms.py`'s verifier to `Lms.Verify` on every accept and reject, and on `root_of`. -/

@[expose] public section

def main (args : List String) : IO UInt32 := Lms.Difftest.main args
