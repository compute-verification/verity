import CertifierDevice.RelayDifftest

/-! The relay difftest's run (`runs` in `lean-audit.json`): `CertifierDevice.RelayDifftest.main`, audited under
`CertifierDevice`. It holds `warden.Warden` to `Relay.run` decision for decision on the generated connections. -/

def main (args : List String) : IO UInt32 := CertifierDevice.RelayDifftest.main args
