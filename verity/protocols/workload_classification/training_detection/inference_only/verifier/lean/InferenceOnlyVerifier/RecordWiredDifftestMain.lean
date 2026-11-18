module

public import InferenceOnlyVerifier.RecordWiredDifftest
meta import InferenceOnlyVerifier.RecordWiredDifftest

/-! The record check's wired reading's difftest: `RecordWiredDifftest.main`. It holds `RecordWired` to the wired
reading of `scripts/inference_only_record_reference.py`, and to `Record` on a root with no edge. -/

@[expose] public section

def main (args : List String) : IO UInt32 := InferenceOnlyVerifier.RecordWiredDifftest.main args
