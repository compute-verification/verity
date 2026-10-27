import Proofs.InferenceOnly.Cover.Paths
import Proofs.InferenceOnly.Cover.Charge
import Proofs.InferenceOnly.Cover.Area
import Proofs.InferenceOnly.Cover.Additive
import Proofs.InferenceOnly.Cover.Witness
import Proofs.InferenceOnly.Cover.Gates
import Proofs.InferenceOnly.Cover.CoveringAdditive
import Proofs.InferenceOnly.Cover.CoveringAdditiveWitness
import Proofs.InferenceOnly.Cover.FanOut
import Proofs.InferenceOnly.Cover.FanOutWitness
import Proofs.InferenceOnly.Cover.FanOutChain
import Proofs.InferenceOnly.Cover.FanOutValue
import Proofs.InferenceOnly.Cover.FanOutValueWitness
import Proofs.InferenceOnly.Cover.PartialSum
import Proofs.InferenceOnly.Cover.PartialSumWitness
import Proofs.InferenceOnly.Cover.Bits
import Proofs.InferenceOnly.Cover.Sum
import Proofs.InferenceOnly.Cover.Training
import Proofs.InferenceOnly.Cover.Aligned
import Proofs.InferenceOnly.Cover.Checked
import Proofs.InferenceOnly.Cover.Chained
import Proofs.InferenceOnly.Cover.CheckedChained
import Proofs.InferenceOnly.Cover.ChainedWitness

/-!
# Proofs of the cover theorem

Each piece is its own file. The steps every tier shares: paths into a block (`Paths`), what a block pays for the
products it finishes (`Charge`), the per-block area count and the sum over a cover (`Area`), the additive charge's two
steps (`Additive`), and a witness that `UpdateHardness` and the upstream certificate hold together (`Witness`). `Gates`
relates this vocabulary to the covering bound's (`InferenceOnly.Covering`) on a declaration's gates: inference blocks, covers and
parameter-hardness agree. The additive covering bound (`Definitions.InferenceOnly.Guarantees.CoverAdditive`), stated in
`InferenceOnly.Covering`'s vocabulary, is `CoveringAdditive`, with its witness (`CoveringAdditiveWitness`). The fan-in/fan-out
bounds
(`Definitions.InferenceOnly.Guarantees.CoverFanOut`) have their plan (`FanOutPlan`), its pieces (`FanOutWalk`, `FanOutRootCost`), the
bounds (`FanOut`), their witness (`FanOutWitness`), and the key chains' one spent value each (`FanOutChain`). The read
tier from wiring (`Definitions.InferenceOnly.Guarantees.CoverFanOutValue`) is `FanOutValue`, with its witness
(`FanOutValueWitness`). The partial-sum bounds (`Definitions.InferenceOnly.Guarantees.CoverPartialSum`) have their pieces
(`PartialSumRead`, `PartialSumCube`, `PartialSumRate`, `PartialSumCover`, `PartialSumKeyed`), the bounds (`PartialSum`)
and their witness (`PartialSumWitness`). `Bits` translates a declaration's gates to the bits the cover theorem counts:
the block of a set of gates, what it computes and what it is handed. `Sum` adds the per-block bounds over a division at
the NCI policy's rate. `Training` joins them to a declaration that complies with the NCI policy
(`Definitions.InferenceOnly.Guarantees.CoverTraining`), `Aligned` gives its variant on whole-word activations, and `Checked` carries it
to a verdict of the NCI check: the gate-level check's, and the check of record's (`SecurityProofs.DescAgrees`).
`Chained` is the same join counting the covering bound's chained multiplication blocks on the reference
circuit's bits as a declaration (`Definitions.InferenceOnly.Guarantees.CoverChained`), with its witness (`ChainedWitness`), and
`CheckedChained` carries it to a verdict as `Checked` does.
-/
