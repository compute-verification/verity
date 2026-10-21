import Proofs.Nci.Cover.Paths
import Proofs.Nci.Cover.Charge
import Proofs.Nci.Cover.Area
import Proofs.Nci.Cover.Additive
import Proofs.Nci.Cover.Width
import Proofs.Nci.Cover.Witness
import Proofs.Nci.Cover.FanOut
import Proofs.Nci.Cover.FanOutWitness
import Proofs.Nci.Cover.FanOutChain
import Proofs.Nci.Cover.FanOutValue
import Proofs.Nci.Cover.FanOutValueWitness
import Proofs.Nci.Cover.PartialSum
import Proofs.Nci.Cover.PartialSumWitness
import Proofs.Nci.Cover.Cut
import Proofs.Nci.Cover.CutWitness
import Proofs.Nci.Cover.Bits
import Proofs.Nci.Cover.AtMost
import Proofs.Nci.Cover.Sum
import Proofs.Nci.Cover.Training
import Proofs.Nci.Cover.Aligned
import Proofs.Nci.Cover.Checked

/-!
# Proofs of the cover theorem

`Models.Nci.Guarantees.Cover` states the guarantees. Each piece is its own file: paths into a block (`Paths`), what a
block pays for the products it finishes (`Charge`), the area bound (`Area`), its additive and width variants
(`Additive`, `Width`), and the non-vacuity witness (`Witness`). `Models.Nci.Guarantees.CoverWidth` states the width
bound from the wiring, proved in `Cut`, with its witness and its limits in `CutWitness`. The fan-in/fan-out bounds
(`Models.Nci.Guarantees.CoverFanOut`) have their plan (`FanOutPlan`), its pieces (`FanOutWalk`, `FanOutRootCost`), the
bounds (`FanOut`), their witness (`FanOutWitness`), and the key chains' one spent value each (`FanOutChain`). The read
tier from wiring (`Models.Nci.Guarantees.CoverFanOutValue`) is `FanOutValue`, with its witness
(`FanOutValueWitness`). The partial-sum bounds (`Models.Nci.Guarantees.CoverPartialSum`) have their pieces
(`PartialSumRead`, `PartialSumCube`, `PartialSumRate`, `PartialSumCover`, `PartialSumKeyed`), the bounds (`PartialSum`)
and their witness (`PartialSumWitness`). `Bits` translates a declaration's gates to the bits the cover theorem counts:
the block of a set of gates, what it computes and what it is handed. `AtMost` gives the per-block bounds on blocks of at
most `s` gates, from the guarantees at the block's own size, and `Sum` adds them over a division at the NCI policy's
rate. `Training` joins them to a declaration that complies with the NCI policy (`Models.Nci.Guarantees.CoverTraining`),
`Aligned` gives its width variant on whole-word activations, and `Checked` carries it to a verdict of the NCI check: the
gate-level check's, and the check of record's given `DescAgrees`'s open statements.
-/
