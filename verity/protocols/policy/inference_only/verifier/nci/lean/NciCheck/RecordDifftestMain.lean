module

public import NciCheck.RecordDifftest
meta import NciCheck.RecordDifftest

/-! The record check's difftest run (`runs` in `lean-audit.json`): `RecordDifftest.main`, audited under
`NciCheck`. It holds `Record` and `RecordCount` to `scripts/nci_record_reference.py`, and the wiring
the form reads to the one SPOT's reader reads. -/

@[expose] public section

def main (args : List String) : IO UInt32 := NciCheck.RecordDifftest.main args
