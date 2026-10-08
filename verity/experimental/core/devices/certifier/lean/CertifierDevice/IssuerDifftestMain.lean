import CertifierDevice.IssuerDifftest

/-! The issuer difftest's run (`runs` in `lean-audit.json`): `CertifierDevice.IssuerDifftest.main`, audited under
`CertifierDevice`. It holds `certificate.Issuer` to `Issuer.certify` call for call on the generated records. -/

def main (args : List String) : IO UInt32 := CertifierDevice.IssuerDifftest.main args
