import PousCheck.SetupDifftest

/-! `pous-setup-difftest`: `PousCheck.SetupDifftest.table` behind `Difftest.serve`, which `tests/test_setup_difftest.py`
drives against the Python reference. -/

def main : IO Unit := Difftest.serve PousCheck.SetupDifftest.table
