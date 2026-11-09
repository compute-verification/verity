module

public import InferenceOnlyCheck.BaselineDifftest
meta import InferenceOnlyCheck.BaselineDifftest

/-! The provenance check's difftest run (`runs` in `lean-audit.json`): `InferenceOnlyCheck.BaselineDifftest.main`, audited under
`InferenceOnlyCheck`. It holds `InferenceOnlyCheck.Baseline.check` to the Python reference and to the unrolled check on the same cases. -/

@[expose] public section

def main (args : List String) : IO UInt32 := InferenceOnlyCheck.BaselineDifftest.main args
