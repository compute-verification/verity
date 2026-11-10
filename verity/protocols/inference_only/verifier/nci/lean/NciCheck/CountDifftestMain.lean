module

public import NciCheck.CountDifftest
meta import NciCheck.CountDifftest

/-! The count mode's difftest run (`runs` in `lean-audit.json`): `NciCheck.CountDifftest.main`, audited under
`NciCheck`. It holds `NciCheck.Count.checkDetail` and `NciCheck.Count.check` to `scripts/nci_reference.py`'s
`check_count_detail` and `check_count`. -/

@[expose] public section

def main (args : List String) : IO UInt32 := NciCheck.CountDifftest.main args
