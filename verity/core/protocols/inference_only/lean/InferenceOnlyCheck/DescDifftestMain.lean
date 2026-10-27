import InferenceOnlyCheck.DescDifftest

/-! The description-sized check's difftest run (`runs` in `lean-audit.json`): `InferenceOnlyCheck.DescDifftest.main`, audited under
`InferenceOnlyCheck`. It holds `InferenceOnlyCheck.Desc.check` to the unrolled check, `InferenceOnlyCheck.check`, on the same declarations. -/

def main (args : List String) : IO UInt32 := InferenceOnlyCheck.DescDifftest.main args
