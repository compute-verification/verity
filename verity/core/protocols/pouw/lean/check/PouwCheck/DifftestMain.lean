import PouwCheck.Difftest

/-! The port's difftest server (`verity/core/difftest`): the functions of `PouwCheck.Difftest` behind `Difftest.serve`,
which `tests/test_difftest.py` drives against `core_check.py`. -/

def main : IO Unit := Difftest.serve PouwCheck.Difftest.table
