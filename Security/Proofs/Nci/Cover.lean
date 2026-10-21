import Proofs.Nci.Cover.Paths
import Proofs.Nci.Cover.Charge
import Proofs.Nci.Cover.Area
import Proofs.Nci.Cover.Additive
import Proofs.Nci.Cover.Width
import Proofs.Nci.Cover.Witness
import Proofs.Nci.Cover.Bits

/-!
# Proofs of the cover theorem

`Models.Nci.Guarantees.Cover` states the guarantees. Each piece is its own file: paths into a block (`Paths`), what a
block pays for the products it finishes (`Charge`), the area bound (`Area`), its additive and width variants
(`Additive`, `Width`), and the non-vacuity witness (`Witness`). `Bits` translates a declaration's gates to the bits the
cover theorem counts: the block of a set of gates, what it computes and what it is handed.
-/
