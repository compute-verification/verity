import PouwWindow.Difftest

/-! The port's difftest server (`verity/core/difftest`): the functions of `PouwWindow.Difftest` behind
`Difftest.serve`, which `tests/test_difftest.py` drives against `window.py`. -/

def main : IO Unit := Difftest.serve PouwWindow.Difftest.table
