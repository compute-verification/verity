module

public import NetworkVerifier.Difftest
meta import NetworkVerifier.Difftest

/-! The difftest's run (`runs` in `lean-audit.json`): `NetworkVerifier.Difftest.main`, audited under `NetworkVerifier`. It holds
`certificate.py`'s encodings to `NetworkVerifier.Certificate` byte for byte on the generated instances. -/

@[expose] public section

def main (args : List String) : IO UInt32 := NetworkVerifier.Difftest.main args
