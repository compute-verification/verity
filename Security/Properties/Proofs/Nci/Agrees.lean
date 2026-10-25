import Properties.Proofs.Nci.BodiesOk
import Properties.Proofs.Nci.Novel
import Properties.Proofs.Nci.OperandsResolve

/-!
# The description-sized check agrees with the check gate by gate, from `ReadsExact`

`descAgrees_of_open` proves `DescAgrees` from three statements, and two of them are proved: `Resolve.bodiesOk`
(`Properties.Proofs.Nci.BodiesOk`) proves `Resolve.BodiesOk`, and `operandsResolve_of`
(`Properties.Proofs.Nci.OperandsResolve`) proves `Walk.OperandsResolve` from it. `descAgrees_of_readsExact` proves
`DescAgrees` from the one open statement, `Novel.ReadsExact` (`Desc.analyze`'s reads are the unrolled operands outside
each root node).
-/

namespace NciCheck.Properties.SecurityProofs.Novel

/-- **`DescAgrees`, from `ReadsExact`.** -/
theorem descAgrees_of_readsExact (hR : ReadsExact) : Properties.DescAgrees :=
  Novel.descAgrees_of_open (OperandsResolve.operandsResolve_of Resolve.bodiesOk) Resolve.bodiesOk hR

end NciCheck.Properties.SecurityProofs.Novel
