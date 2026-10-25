import Properties.Proofs.Nci.BodiesOk
import Properties.Proofs.Nci.BodiesOk.ReadsDecoded
import Properties.Proofs.Nci.Novel
import Properties.Proofs.Nci.OperandsResolve

/-!
# The description-sized check agrees with the check gate by gate

`descAgrees_of_open` proves `DescAgrees` from three statements: `Resolve.bodiesOk` (`Properties.Proofs.Nci.BodiesOk`)
proves `Resolve.BodiesOk`, `operandsResolve_of` (`Properties.Proofs.Nci.OperandsResolve`) proves `Walk.OperandsResolve`
from it, and `readsExact` proves `Novel.ReadsExact` (`Desc.analyze`'s reads are the unrolled operands outside each root
node) from `BodiesOk.rootSet` (`readsExact_of_set`, `Properties.Proofs.Nci.BodiesOk.ReadsDecoded`).
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
