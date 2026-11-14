module

public import InferenceOnlyCheck.CountDifftest
meta import InferenceOnlyCheck.CountDifftest

/-! The count mode's difftest run (`runs` in `lean-audit.json`): `InferenceOnlyCheck.CountDifftest.main`, audited under
`InferenceOnlyCheck`. It holds `InferenceOnlyCheck.Count.checkDetail` and `InferenceOnlyCheck.Count.check` to `scripts/inference_only_reference.py`'s
`check_count_detail` and `check_count`. -/

@[expose] public section

def main (args : List String) : IO UInt32 := InferenceOnlyCheck.CountDifftest.main args
