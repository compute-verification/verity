import Proofs.Properties.Nci.BodiesOk
import Proofs.Properties.Nci.BodiesOk.ReadsDecoded
import Proofs.Properties.Nci.Fresh
import Proofs.Properties.Nci.OperandsResolve

/-!
# The description-sized check agrees with the check gate by gate

`descAgrees_of_open` proves `DescAgrees` from three statements: `Resolve.bodiesOk` (`Proofs.Properties.Nci.BodiesOk`)
proves `Resolve.BodiesOk`, `operandsResolve_of` (`Proofs.Properties.Nci.OperandsResolve`) proves `Walk.OperandsResolve`
from it, and `readsExact` proves `Fresh.ReadsExact` (`Desc.analyze`'s reads are the unrolled operands outside each root
node) from `BodiesOk.rootSet` (`readsExact_of_set`, `Proofs.Properties.Nci.BodiesOk.ReadsDecoded`).
`descAgrees_of_readsExact` proves `DescAgrees` from `ReadsExact`, and `SecurityProofs.DescAgrees` is the guarantee.
-/

namespace NciCheck.Properties.SecurityProofs.Fresh

/-- **`DescAgrees`, from `ReadsExact`.** -/
theorem descAgrees_of_readsExact (hR : ReadsExact) : Properties.DescAgrees :=
  Fresh.descAgrees_of_open (OperandsResolve.operandsResolve_of Resolve.bodiesOk) Resolve.bodiesOk hR

/-- **`analyze`'s reads are the unrolled operands outside each root node.** -/
theorem readsExact : ReadsExact :=
  BodiesOk.readsExact_of_set BodiesOk.rootSet

end NciCheck.Properties.SecurityProofs.Fresh

namespace NciCheck.Properties.SecurityProofs

/-- **The check at the size of the description agrees with the check gate by gate.** -/
theorem DescAgrees : Properties.DescAgrees :=
  Fresh.descAgrees_of_readsExact Fresh.readsExact

end NciCheck.Properties.SecurityProofs
