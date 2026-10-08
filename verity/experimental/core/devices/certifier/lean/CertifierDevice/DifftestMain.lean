import CertifierDevice.Difftest

/-! The difftest's run (`runs` in `lean-audit.json`): `CertifierDevice.Difftest.main`, audited under `CertifierDevice`. It
holds `active.Proxy` to `step` action for action on the generated traces. -/

def main (args : List String) : IO UInt32 := CertifierDevice.Difftest.main args
