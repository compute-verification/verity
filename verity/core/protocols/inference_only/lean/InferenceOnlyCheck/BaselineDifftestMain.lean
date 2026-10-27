import InferenceOnlyCheck.BaselineDifftest

/-! The provenance check's difftest run (`runs` in `lean-audit.json`): `InferenceOnlyCheck.BaselineDifftest.main`, audited under
`InferenceOnlyCheck`. It holds `InferenceOnlyCheck.Baseline.check` to the Python reference and to the unrolled check on the same cases. -/

def main (args : List String) : IO UInt32 := InferenceOnlyCheck.BaselineDifftest.main args
