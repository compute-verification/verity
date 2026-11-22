module

public import InferenceOnlyVerifier.DescDifftest
meta import InferenceOnlyVerifier.DescDifftest

/-! The description-sized check's difftest run (`runs` in `lean-audit.json`): `InferenceOnlyVerifier.DescDifftest.main`, audited under
`InferenceOnlyVerifier`. It holds `InferenceOnlyVerifier.Desc.check` to the unrolled check, `InferenceOnlyVerifier.check`, on the same declarations. -/

@[expose] public section

def main (args : List String) : IO UInt32 := InferenceOnlyVerifier.DescDifftest.main args
