import Definitions.NetworkCertifier.Warden

/-!
# A period's named assumption: the wardens see what the certifier observed

`Guarantees.PeriodLeakAccepted` takes one named assumption beyond the audit's and the warden host's. It ties two
auditor devices: what reaches the wardens from the server is what the certifier observed of that server, framed into
cells (`WardenCertifierEgress`). That the audits' traces carry the certifier's rows at their transfer gates is not an
assumption: it is part of each window's good event (`WindowGood`), as it is a condition inside `RAcc`'s conclusion.
-/

namespace NetTiming

/-- **The wardens see what the certifier observed** (`warden-certifier-egress`): at every server behavior `σ`, the data
reaching the wardens on each connection is the certifier's observations of `σ` over the period's windows (`obsOf σ`),
framed into cells (`cells`). Both devices are the auditor's, each on a machine the prover doesn't control. -/
def WardenCertifierEgress {Srv Conn Cell Ob : Type} (arr : Srv → Conn → Arrivals Cell) (obsOf : Srv → Ob)
    (cells : Ob → Conn → List Cell) : Prop :=
  ∀ σ, (fun c => (arr σ c).stream) = cells (obsOf σ)

end NetTiming
