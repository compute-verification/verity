module

public import PousVerifier.SetupDifftest

/-! `pous-setup-difftest`: `PousVerifier.SetupDifftest.table` behind `Difftest.serve`, which `tests/test_setup_difftest.py`
drives against the Python reference. -/

@[expose] public section

def main : IO Unit := Difftest.serve PousVerifier.SetupDifftest.table
