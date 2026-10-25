import Definitions.NetworkCertifier.Warden
import Definitions.NetworkCertifier.SampledAudit

/-!
# A period's named assumptions: the wardens see what the certifier observed, and the trace carries it

`Guarantees.PeriodLeakAccepted` takes two named assumptions beyond the audit's and the warden host's. One ties two
auditor devices: what reaches the wardens from the server is what the certifier observed of that server, framed into
cells (`WardenCertifierEgress`). The other ties the audit's extracted trace to the certifier: the trace's transfer gates
carry the certifier's rows (`RowsCarried`). The scan's registered reads of the warden's rows, which the Lean verifier
opens against the warden's registered root, are the check it rests on; no lemma here derives it from them.
-/

namespace NetTiming

open Claims Claims.Sampling

/-- **The wardens see what the certifier observed** (`warden-certifier-egress`): at every server behavior `σ`, the data
reaching the wardens on each connection is the certifier's observations of `σ` over the period's windows (`obsOf σ`),
framed into cells (`cells`). Both devices are the auditor's, each on a machine the prover doesn't control. -/
def WardenCertifierEgress {Srv Conn Cell Ob : Type} (arr : Srv → Conn → Arrivals Cell) (obsOf : Srv → Ob)
    (cells : Ob → Conn → List Cell) : Prop :=
  ∀ σ, (fun c => (arr σ c).stream) = cells (obsOf σ)

/-- **The transfer gates carry the certifier's rows** (`rows-carried`): at the statement `x` and the audit's extracted
witness `w`, the trace's transfer gates carry the certifier's observations `o` at the operator `op` (`Carries`). -/
def RowsCarried {Coin Col T : Type} {L K S : ℕ} (H : Msg → Msg) (D : Deploy Coin Col T) (lay : Stmt → Layout L S)
    (k : Stmt → ℕ) (op : ℕ → List (Obs L K S) → List Bool) (x : Msg) (w : Wit) (o : List (Obs L K S)) : Prop :=
  ∀ s, D.parse x = some s → Carries (lay s) s.prog (w.val H D.commit s) op (k s) o

end NetTiming
