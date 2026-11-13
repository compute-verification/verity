module

public import Proofs.Properties.Verity.Sampled.Members.Widen

/-!
# Scratch, never to merge: `JointAuditV5.extend`'s `hend` as one term

Whether the widened audit's `hend` elaborates as the term that hands the registered audit's proof over unchanged, with
no `dsimp` of `sampleEnded` first.
-/

public section

namespace Flock.Guarantees.JointAuditV5

theorem probe_hend_term (w : JointAuditV5) (S : Verity.Sampled.Widening) :
    ∃ ω : w.Ω × S.T, sampleEnded (fun ω : w.Ω × S.T => w.src ω.1) w.law w.nJ ω = true :=
  w.hend.elim fun ω h => ⟨(ω, Classical.arbitrary S.T), h⟩

end Flock.Guarantees.JointAuditV5
