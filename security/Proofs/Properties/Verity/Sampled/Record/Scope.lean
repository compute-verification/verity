module

public import Proofs.Flock.Claims.NumbersDeploy

/-! A session's setup in Flock's masked scope. -/

@[expose] public section

namespace Verity.Sampled.Record

def SessionInScope (spec : Flock.SessionSpec) (sts : Array Flock.Stmt) : Prop :=
  sts.size = 1 ∧ (∀ st ∈ sts.toList, 25 ≤ st.m ∧ st.m ≤ 35) ∧
    (∀ st ∈ sts.toList, st.c.kLog ≤ 27 ∧ st.regions.size ≤ 1024 ∧ spec.mPts ≤ 64) ∧ spec.points ≤ 2

theorem zkIn_of_inScope {x : Claims.Msg} {s : FlockDeploy.Ins} {spec : Flock.SessionSpec} {sts : Array Flock.Stmt}
    (hs : FlockDeploy.setupOf true x = some (s, spec, sts)) (h : SessionInScope spec sts) :
    FlockDeploy.ZkIn x s spec sts :=
  ⟨hs, by have := h.1; omega, by have := h.1; omega, h.2.2.2,
    fun st hst => ⟨(h.2.1 st hst).1, (h.2.1 st hst).2, h.2.2.1 st hst⟩⟩

end Verity.Sampled.Record
