import NciCheck.BaselineDifftest

/-! The provenance check's difftest run (`runs` in `lean-audit.json`): `NciCheck.BaselineDifftest.main`, audited under
`NciCheck`. It holds `NciCheck.Baseline.check` to the Python reference and to the unrolled check on the same cases. -/

def main (args : List String) : IO UInt32 := NciCheck.BaselineDifftest.main args
