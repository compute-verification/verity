import Proofs.Nci.Products.Cap
import Proofs.Nci.Products.Witnesses
import Proofs.Nci.Products.NoCell
import Proofs.Nci.Products.NoCellWitnesses
import Proofs.Nci.Products.Whole
import Proofs.Nci.Products.WholeWitnesses
import Proofs.Nci.Products.Sum
import Proofs.Nci.Products.SumWitnesses

/-!
# Proofs of the fresh-products theorems

`Nci.Products.SecurityProofs.G : Nci.Products.Guarantees.G` for each theorem of `Definitions.Nci.Guarantees.Products`:
the cap at the bit level, its word-level form, the cap along chains and the slowdown the bit-level cap gives
(`Proofs.Nci.Products.Cap`), and their honest and nontrivial lemmas (`Proofs.Nci.Products.Witnesses`); for each of
`Definitions.Nci.Guarantees.Products.NoCell`: the cap without the word level and its slowdown
(`Proofs.Nci.Products.NoCell`), and their honest and nontrivial lemmas (`Proofs.Nci.Products.NoCellWitnesses`); for
each of `Definitions.Nci.Guarantees.Products.Whole`: the class bound, the cap with operands made whole and its slowdown
(`Proofs.Nci.Products.Whole`), and their honest and nontrivial lemmas (`Proofs.Nci.Products.WholeWitnesses`); and for
each of `Definitions.Nci.Guarantees.Products.Sum`: the cap and its slowdown summed over a set of products, several in a
block, with P-cell and without it (`Proofs.Nci.Products.Sum`), and their honest and nontrivial lemmas
(`Proofs.Nci.Products.SumWitnesses`).
-/
