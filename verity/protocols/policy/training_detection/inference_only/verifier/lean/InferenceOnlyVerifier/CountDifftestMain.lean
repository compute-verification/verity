module

public import InferenceOnlyVerifier.CountDifftest
meta import InferenceOnlyVerifier.CountDifftest

/-! The count mode's difftest run (`runs` in `lean-audit.json`): `InferenceOnlyVerifier.CountDifftest.main`, audited under
`InferenceOnlyVerifier`. It holds `InferenceOnlyVerifier.Count.checkDetail` and `InferenceOnlyVerifier.Count.check` to `scripts/inference_only_reference.py`'s
`check_count_detail` and `check_count`. -/

@[expose] public section

def main (args : List String) : IO UInt32 := InferenceOnlyVerifier.CountDifftest.main args
