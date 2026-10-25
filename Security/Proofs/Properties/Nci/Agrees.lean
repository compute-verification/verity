import Proofs.Properties.Nci.BodiesOk
import Proofs.Properties.Nci.BodiesOk.ReadsDecoded
import Proofs.Properties.Nci.Novel
import Proofs.Properties.Nci.OperandsResolve

/-!
# The description-sized check agrees with the check gate by gate

`descAgrees_of_open` proves `DescAgrees` from three statements: `Resolve.bodiesOk` (`Proofs.Properties.Nci.BodiesOk`)
proves `Resolve.BodiesOk`, `operandsResolve_of` (`Proofs.Properties.Nci.OperandsResolve`) proves `Walk.OperandsResolve`
from it, and `readsExact` proves `Novel.ReadsExact` (`Desc.analyze`'s reads are the unrolled operands outside each root
node) from `BodiesOk.rootSet` (`readsExact_of_set`, `Proofs.Properties.Nci.BodiesOk.ReadsDecoded`).
`descAgrees_of_readsExact` proves `DescAgrees` from `ReadsExact`, and `SecurityProofs.DescAgrees` is the guarantee.
-/

namespace NciCheck.Properties.SecurityProofs.Novel

/-- **`DescAgrees`, from `ReadsExact`.** -/
theorem descAgrees_of_readsExact (hR : ReadsExact) : Properties.DescAgrees :=
  Novel.descAgrees_of_open (OperandsResolve.operandsResolve_of Resolve.bodiesOk) Resolve.bodiesOk hR

/-- **`analyze`'s reads are the unrolled operands outside each root node.** -/
theorem readsExact : ReadsExact :=
  BodiesOk.readsExact_of_set BodiesOk.rootSet

end NciCheck.Properties.SecurityProofs.Novel

namespace NciCheck.Properties.SecurityProofs

/-- **The check at the size of the description agrees with the check gate by gate.** -/
theorem DescAgrees : Properties.DescAgrees :=
  Novel.descAgrees_of_readsExact Novel.readsExact

end NciCheck.Properties.SecurityProofs
