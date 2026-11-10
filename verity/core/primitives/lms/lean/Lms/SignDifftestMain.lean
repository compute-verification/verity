module

public import Lms.SignDifftest
meta import Lms.SignDifftest

/-! The sign difftest's run (`runs` in `lean-audit.json`): `Lms.SignDifftest.main`, audited under `Lms`. It holds
`lms.py`'s `keygen`, `key`, `public_key` and `sign_at` to `Lms.Sign` byte for byte on the generated keys. -/

@[expose] public section

def main (args : List String) : IO UInt32 := Lms.SignDifftest.main args
