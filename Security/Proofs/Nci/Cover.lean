import Proofs.Nci.Cover.Paths
import Proofs.Nci.Cover.Charge
import Proofs.Nci.Cover.Area
import Proofs.Nci.Cover.Additive
import Proofs.Nci.Cover.Witness
import Proofs.Nci.Cover.Gates
import Proofs.Nci.Cover.CoveringAdditive
import Proofs.Nci.Cover.CoveringAdditiveWitness
import Proofs.Nci.Cover.FanOut
import Proofs.Nci.Cover.FanOutWitness
import Proofs.Nci.Cover.FanOutChain
import Proofs.Nci.Cover.FanOutValue
import Proofs.Nci.Cover.FanOutValueWitness
import Proofs.Nci.Cover.PartialSum
import Proofs.Nci.Cover.PartialSumWitness
import Proofs.Nci.Cover.Bits
import Proofs.Nci.Cover.Sum
import Proofs.Nci.Cover.Training
import Proofs.Nci.Cover.Aligned
import Proofs.Nci.Cover.Checked
import Proofs.Nci.Cover.Chained
import Proofs.Nci.Cover.CheckedChained
import Proofs.Nci.Cover.ChainedWitness

/-!
# Proofs of the cover theorem

Each piece is its own file. The steps every tier shares: paths into a block (`Paths`), what a block pays for the
products it finishes (`Charge`), the per-block area count and the sum over a cover (`Area`), the additive charge's two
steps (`Additive`), and a witness that `UpdateHardness` and the upstream certificate hold together (`Witness`). `Gates`
relates this vocabulary to the covering bound's (`Nci.Covering`) on a declaration's gates: inference blocks, covers and
parameter-hardness agree. The additive covering bound (`Definitions.Nci.Guarantees.CoverAdditive`), stated in
`Nci.Covering`'s vocabulary, is `CoveringAdditive`, with its witness (`CoveringAdditiveWitness`). The fan-in/fan-out
bounds
(`Definitions.Nci.Guarantees.CoverFanOut`) have their plan (`FanOutPlan`), its pieces (`FanOutWalk`, `FanOutRootCost`), the
bounds (`FanOut`), their witness (`FanOutWitness`), and the key chains' one spent value each (`FanOutChain`). The read
tier from wiring (`Definitions.Nci.Guarantees.CoverFanOutValue`) is `FanOutValue`, with its witness
(`FanOutValueWitness`). The partial-sum bounds (`Definitions.Nci.Guarantees.CoverPartialSum`) have their pieces
(`PartialSumRead`, `PartialSumCube`, `PartialSumRate`, `PartialSumCover`, `PartialSumKeyed`), the bounds (`PartialSum`)
and their witness (`PartialSumWitness`). `Bits` translates a declaration's gates to the bits the cover theorem counts:
the block of a set of gates, what it computes and what it is handed. `Sum` adds the per-block bounds over a division at
the NCI policy's rate. `Training` joins them to a declaration that complies with the NCI policy
(`Definitions.Nci.Guarantees.CoverTraining`), `Aligned` gives its variant on whole-word activations, and `Checked` carries it
to a verdict of the NCI check: the gate-level check's, and the check of record's (`SecurityProofs.DescAgrees`).
`Chained` is the same join counting the covering bound's chained multiplication blocks on the reference
circuit's bits as a declaration (`Definitions.Nci.Guarantees.CoverChained`), with its witness (`ChainedWitness`), and
`CheckedChained` carries it to a verdict as `Checked` does.
-/
