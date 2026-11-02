module

public import CertifierDevice.IssuerDifftest
meta import CertifierDevice.IssuerDifftest

/-! The issuer difftest's run (`runs` in `lean-audit.json`): `CertifierDevice.IssuerDifftest.main`, audited under
`CertifierDevice`. It holds `certificate.Issuer` to `Issuer.certify` call for call on the generated records. -/

@[expose] public section

def main (args : List String) : IO UInt32 := CertifierDevice.IssuerDifftest.main args
