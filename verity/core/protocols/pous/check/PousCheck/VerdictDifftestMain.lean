import Difftest
import PousCheck.VerdictDifftest

/-! `pous-verdict-difftest`: `PousCheck.VerdictDifftest`'s `gates` and `eval` behind `Difftest.serve`, which
`tests/test_verdict_difftest.py` drives against the Definition `verdict_circuit.PousVerdict`, compiled. -/

def main : IO Unit :=
  Difftest.serve (Difftest.table [("gates", PousCheck.VerdictDifftest.gates), ("eval", PousCheck.VerdictDifftest.eval)])
