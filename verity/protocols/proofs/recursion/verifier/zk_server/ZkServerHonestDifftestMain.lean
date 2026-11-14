module

public import ZkServerHonestDifftest

/-! `zk-server-honest-difftest`: `ZkServerHonestDifftest.table` behind `Difftest.serve`, which
`verity/ml/flock/tests/test_honest_reqs.py` drives against `rec_live`'s requests. -/

@[expose] public section

def main : IO Unit := Difftest.serve ZkServerHonestDifftest.table
