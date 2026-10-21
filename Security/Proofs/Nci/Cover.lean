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

/-!
# Proofs of the cover theorem

`Models.Nci.Guarantees.Cover` states the guarantees. Each piece is its own file: paths into a block (`Paths`), what a
block pays for the products it finishes (`Charge`), the area bound (`Area`), its additive and width variants
(`Additive`, `Width`), and the non-vacuity witness (`Witness`). The fan-in/fan-out bounds
(`Models.Nci.Guarantees.CoverFanOut`) have their plan (`FanOutPlan`), its pieces (`FanOutWalk`, `FanOutRootCost`), the
bounds (`FanOut`), their witness (`FanOutWitness`), and the key chains' one spent value each (`FanOutChain`). The read
tier from wiring (`Models.Nci.Guarantees.CoverFanOutValue`) is `FanOutValue`, with its witness
(`FanOutValueWitness`).
-/
