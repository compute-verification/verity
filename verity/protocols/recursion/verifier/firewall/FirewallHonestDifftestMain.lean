module

public import FirewallHonestDifftest

/-! `firewall-honest-difftest`: `FirewallHonestDifftest.table` behind `Difftest.serve`, which
`verity/ml/flock/tests/test_honest_reqs.py` drives against `rec_live`'s requests. -/

@[expose] public section

def main : IO Unit := Difftest.serve FirewallHonestDifftest.table
