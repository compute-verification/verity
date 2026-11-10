import NciCheck.ThetaDifftest

/-! The provenance check's difftest run (`runs` in `lean-audit.json`): `NciCheck.ThetaDifftest.main`, audited under
`NciCheck`. It holds `NciCheck.Theta.check` to the Python reference and to the unrolled check on the same cases. -/

def main (args : List String) : IO UInt32 := NciCheck.ThetaDifftest.main args
