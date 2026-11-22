module

public import ProgramDifftest

/-! `program-difftest`: `ProgramDifftest.table` behind `Difftest.serve`, which `tests/tools/difftest/test_program_decode_port.py`
drives against `codec.decode_program`. -/

@[expose] public section

def main : IO Unit := Difftest.serve ProgramDifftest.table
