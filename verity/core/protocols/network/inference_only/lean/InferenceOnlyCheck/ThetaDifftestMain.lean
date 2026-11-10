import InferenceOnlyCheck.ThetaDifftest

/-! The provenance check's difftest run (`runs` in `lean-audit.json`): `InferenceOnlyCheck.ThetaDifftest.main`, audited under
`InferenceOnlyCheck`. It holds `InferenceOnlyCheck.Theta.check` to the Python reference and to the unrolled check on the same cases. -/

def main (args : List String) : IO UInt32 := InferenceOnlyCheck.ThetaDifftest.main args
