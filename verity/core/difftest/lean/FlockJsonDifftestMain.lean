import FlockJsonDifftest

/-! `flock-json-difftest`: `FlockJsonDifftest.table` behind `Difftest.serve`, which `tests/test_flock_json_port.py`
drives against the Python mirror. -/

def main : IO Unit := Difftest.serve FlockJsonDifftest.table
