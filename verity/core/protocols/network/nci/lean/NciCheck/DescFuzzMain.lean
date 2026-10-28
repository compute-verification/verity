import NciCheck.DescFuzz

/-! The desc fuzzer's run (`runs` in `lean-audit.json`): `NciCheck.DescFuzz.main`, audited under `NciCheck`. It decides
the statements behind `DescAgrees` on `scripts/desc_fuzz_programs.py`'s Programs, `NciCheck.Desc.check` against the
unrolled check, `NciCheck.check`, among them. -/

def main (args : List String) : IO UInt32 := NciCheck.DescFuzz.main args
