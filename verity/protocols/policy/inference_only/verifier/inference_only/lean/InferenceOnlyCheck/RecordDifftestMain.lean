module

public import InferenceOnlyCheck.RecordDifftest
meta import InferenceOnlyCheck.RecordDifftest

/-! The record check's difftest run (`runs` in `lean-audit.json`): `RecordDifftest.main`, audited under
`InferenceOnlyCheck`. It holds `Record` and `RecordCount` to `scripts/inference_only_record_reference.py`, and the wiring
the form reads to the one SPOT's reader reads. -/

@[expose] public section

def main (args : List String) : IO UInt32 := InferenceOnlyCheck.RecordDifftest.main args
