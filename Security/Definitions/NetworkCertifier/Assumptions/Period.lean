import Definitions.NetworkCertifier.SampledAudit

/-!
# A period's named assumption: the trace carries the certifier's rows

`Guarantees.PeriodLeakAccepted` takes one named assumption beyond the audit's and the tap's host's. It ties the audit's
extracted trace to the certifier: the trace's transfer gates carry the certifier's rows (`RowsCarried`). The scan's
registered reads of the warden's rows, which the Lean verifier opens against the warden's registered root, are the check
it rests on; no lemma here derives it from them. That the wardens see what the certifier observed is no assumption: the
warden and the certifier are one tap (`WardenTap`), and it is a lemma about the tap's code.
-/

namespace NetTiming

open Claims Claims.Sampling

/-- **The transfer gates carry the certifier's rows** (`rows-carried`): at the statement `x` and the audit's extracted
witness `w`, the trace's transfer gates carry the certifier's observations `o` at the operator `op` (`Carries`). -/
def RowsCarried {Coin Col T : Type} {L K S : ℕ} (H : Msg → Msg) (D : Deploy Coin Col T) (lay : Stmt → Layout L S)
    (k : Stmt → ℕ) (op : ℕ → List (Obs L K S) → List Bool) (x : Msg) (w : Wit) (o : List (Obs L K S)) : Prop :=
  ∀ s, D.parse x = some s → Carries (lay s) s.prog (w.val H D.commit s) op (k s) o

end NetTiming
