module

public import InferenceOnlyVerifier.DescFuzz
meta import InferenceOnlyVerifier.DescFuzz

/-! The desc fuzzer's run (`runs` in `lean-audit.json`): `InferenceOnlyVerifier.DescFuzz.main`, audited under `InferenceOnlyVerifier`. It decides
the statements behind `DescAgrees` on `scripts/desc_fuzz_programs.py`'s Programs, `InferenceOnlyVerifier.Desc.check` against the
unrolled check, `InferenceOnlyVerifier.check`, among them. -/

@[expose] public section

def main (args : List String) : IO UInt32 := InferenceOnlyVerifier.DescFuzz.main args
