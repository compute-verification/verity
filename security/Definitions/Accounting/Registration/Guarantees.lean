module

public import Definitions.Accounting.Registration.View

/-!
# What registration establishes

`RegistrationSound`: the view of a log registration accepts holds the registered facts the theorems read of it
(`Registration.View.Holds`), and the log is chained (`Registration.Chained`).
-/

@[expose] public section

namespace Verity.Guarantees

open Verity

/-- **What registration establishes.** For all digests `D`, auditors `A`, composites `c` and logs: if the log is valid
(`Registration.Valid`: each entry passes its checks against the log before it), its view holds the registered facts the
theorems read (`Registration.View.Holds`): every registered device has accepted figures, onsite every boundary link is
certified and every memory leaf in a reached host, θ's values and the declared days are each registered once, and the
whole Site passes every check of a site entry (`Site.Checked`: fresh ids and sheets, known references, kinds, sheets,
figures, capacity, owners, coverage and reach); and each entry names the head of the log before it
(`Registration.Chained`). -/
def RegistrationSound : Prop :=
  ∀ (D : Registration.Digests) (A : Registration.Auditor) (c : Registration.Composite) (log : Registration),
    Registration.Valid D A c log → (Registration.view D c log).Holds D A c ∧ Registration.Chained D c log

end Verity.Guarantees
