module

public import Difftest
public import PousVerifier.VerdictDifftest

/-! `pous-verdict-difftest`: `PousVerifier.VerdictDifftest`'s `gates` and `eval` behind `Difftest.serve`, which
`tests/test_verdict_difftest.py` drives against the Definition `verdict_circuit.PousVerdict`, compiled. -/

@[expose] public section

def main : IO Unit :=
  Difftest.serve (Difftest.table [("gates", PousVerifier.VerdictDifftest.gates), ("eval", PousVerifier.VerdictDifftest.eval)])
