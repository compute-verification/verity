import Difftest
import PousCheck.Obligation

/-! `pous-obligation-difftest`: `PousCheck.Obligation.verdict` behind `Difftest.serve`, which
`tests/test_obligation_difftest.py` drives against `obligation.verdict`. -/

def main : IO Unit := Difftest.serve (Difftest.table [("verdict", PousCheck.Obligation.verdict)])
