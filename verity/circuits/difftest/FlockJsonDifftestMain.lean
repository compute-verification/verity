module

public import FlockJsonDifftest

/-! `flock-json-difftest`: `FlockJsonDifftest.table` behind `Difftest.serve`, which `tests/test_flock_json_port.py`
drives against the Python mirror. -/

@[expose] public section

def main : IO Unit := Difftest.serve FlockJsonDifftest.table
