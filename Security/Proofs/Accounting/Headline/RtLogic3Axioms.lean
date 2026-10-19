import Proofs.Accounting.Headline.RtLogic3ForkAt
import Proofs.Accounting.Headline.Sm120
import Proofs.Accounting.Headline.Sm120Measured

/-! RT3-LOGIC smell check on #2791 @ 8ce15bb6f5 (not on main): the two sm_120 headlines' types and axioms, and the
witness's axioms. -/

example : Accounting.Guarantees.TS.Sm120SheetHeadline := Accounting.SecurityProofs.TS.Sm120SheetHeadline
example : Accounting.Guarantees.TS.Sm120MeasuredHeadline := Accounting.SecurityProofs.TS.Sm120MeasuredHeadline

#print axioms Accounting.SecurityProofs.TS.Sm120SheetHeadline
#print axioms Accounting.SecurityProofs.TS.Sm120MeasuredHeadline
#print axioms Accounting.SecurityProofs.TS.RtLogic3.coinLayer_budget_unsat
#print axioms Accounting.SecurityProofs.TS.RtLogic3.coinLayer_moved_budget_unsat_dc
#print axioms Accounting.SecurityProofs.TS.RtLogic3.game_accepts_last
