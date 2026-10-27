import InferenceOnlyCheck.DescFuzz

/-! The desc fuzzer's run (`runs` in `lean-audit.json`): `InferenceOnlyCheck.DescFuzz.main`, audited under `InferenceOnlyCheck`. It decides
the statements behind `DescAgrees` on `scripts/desc_fuzz_programs.py`'s Programs, `InferenceOnlyCheck.Desc.check` against the
unrolled check, `InferenceOnlyCheck.check`, among them. -/

def main (args : List String) : IO UInt32 := InferenceOnlyCheck.DescFuzz.main args
