module

public import NetworkCheck.Difftest
meta import NetworkCheck.Difftest

/-! The difftest's run (`runs` in `lean-audit.json`): `NetworkCheck.Difftest.main`, audited under `NetworkCheck`. It holds
`certificate.py`'s encodings to `NetworkCheck.Certificate` byte for byte on the generated instances. -/

@[expose] public section

def main (args : List String) : IO UInt32 := NetworkCheck.Difftest.main args
