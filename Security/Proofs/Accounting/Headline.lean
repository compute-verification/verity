import Proofs.Accounting.Headline.Encoding
import Proofs.Accounting.Headline.EpsJl
import Proofs.Accounting.Headline.FpG1
import Proofs.Accounting.Headline.FpG1Measured
import Proofs.Accounting.Headline.Luck
import Proofs.Accounting.Headline.Sm120
import Proofs.Accounting.Headline.Sm120Measured
import Proofs.Accounting.Headline.Whole

/-! The families' headline guarantees (`Definitions.Accounting.Guarantees.Headline`): the encoding's lemmas
(`Headline/Encoding.lean`), (I7) from the receipt split (`Headline/Luck.lean`), Theorem D-ideal with fold skippers
charged (`Headline/Whole.lean`: FOLD-OPT per word folded and the whole-unit luck) and one file per family and reading
(`Headline/Sm120.lean`: sm_120 at its data sheet;
`Headline/Sm120Measured.lean`: sm_120 on VF3s-16); and FP8-g1's and W4A8-g1's headlines on sm_120
(`Definitions.Accounting.Guarantees.FpG1`), with an honest g1 window that meets every premise, at the data sheet
(`Headline/FpG1.lean`) and on the measured vector (`Headline/FpG1Measured.lean`). -/
