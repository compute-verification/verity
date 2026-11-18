module

public import InferenceOnlyVerifier.RecordDifftest
meta import InferenceOnlyVerifier.RecordDifftest

/-! The record check's difftest run (`runs` in `lean-audit.json`): `RecordDifftest.main`, audited under
`InferenceOnlyVerifier`. It holds `Record`, `RecordCount`, `RecordRows` and `RecordChosen` to
`scripts/inference_only_record_reference.py`, and the wiring the form reads to the one SPOT's reader reads. -/

@[expose] public section

def main (args : List String) : IO UInt32 := InferenceOnlyVerifier.RecordDifftest.main args
