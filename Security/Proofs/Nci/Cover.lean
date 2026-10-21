import Proofs.Nci.Cover.Paths
import Proofs.Nci.Cover.Charge
import Proofs.Nci.Cover.Area
import Proofs.Nci.Cover.Additive
import Proofs.Nci.Cover.Width
import Proofs.Nci.Cover.Witness
import Proofs.Nci.Cover.Bits
import Proofs.Nci.Cover.AtMost
import Proofs.Nci.Cover.Sum
import Proofs.Nci.Cover.Training
import Proofs.Nci.Cover.Checked

/-!
# Proofs of the cover theorem

`Models.Nci.Guarantees.Cover` states the guarantees. Each piece is its own file: paths into a block (`Paths`), what a
block pays for the products it finishes (`Charge`), the area bound (`Area`), its additive and width variants
(`Additive`, `Width`), and the non-vacuity witness (`Witness`). `Bits` translates a declaration's gates to the bits the
cover theorem counts: the block of a set of gates, what it computes and what it is handed. `AtMost` gives the per-block
bounds on blocks of at most `s` gates, from the guarantees at the block's own size, and `Sum` adds them over a division
at the NCI policy's rate. `Training` joins them to a declaration that complies with the NCI policy
(`Models.Nci.Guarantees.CoverTraining`), and `Checked` carries it to a verdict of the NCI check: the gate-level check's,
and the check of record's given `DescAgrees`'s open statements.
-/
