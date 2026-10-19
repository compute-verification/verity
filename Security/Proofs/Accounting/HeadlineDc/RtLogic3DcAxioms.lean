import Proofs.Accounting.HeadlineDc

/-! RT3-LOGIC smell check on #2786 @ 217f29a644 (not on main): each data-centre proof has #2699's statement as its
type, with no hypothesis added, and its axioms. -/

example : Accounting.Guarantees.TS.Sm90SheetHeadline := Accounting.SecurityProofs.TS.Sm90SheetHeadline
example : Accounting.Guarantees.TS.Sm90MeasuredHeadline := Accounting.SecurityProofs.TS.Sm90MeasuredHeadline
example : Accounting.Guarantees.TS.Sm100SheetHeadline := Accounting.SecurityProofs.TS.Sm100SheetHeadline
example : Accounting.Guarantees.TS.Sm100MeasuredHeadline := Accounting.SecurityProofs.TS.Sm100MeasuredHeadline

#print axioms Accounting.SecurityProofs.TS.Sm90SheetHeadline
#print axioms Accounting.SecurityProofs.TS.Sm90MeasuredHeadline
#print axioms Accounting.SecurityProofs.TS.Sm100SheetHeadline
#print axioms Accounting.SecurityProofs.TS.Sm100MeasuredHeadline
