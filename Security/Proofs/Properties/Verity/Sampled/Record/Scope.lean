module

public import Proofs.Flock.Claims.NumbersDeploy

/-!
# A session's setup in Flock's masked scope

`SessionInScope spec sts` reads a session's setup (`sessionAt`'s `(spec, sts)` in one-stage v2, `Ins.setup`'s in
flock-verify `--zk`): one table, at an `m` the masked verifier serves (`25 ≤ m ≤ 35`), in the statement range the
soundness bound covers (`Flock.checkInRange_ok`'s three bounds, `FlockDeploy.acct`'s `InRange`), at at most two link
points. It reads only Flock's own fields. With the setup's equation it is `FlockDeploy.ZkIn` at one table
(`zkIn_of_inScope`), the scope `SessionsIn` reads; every record session that sets up is in it (`Record.Inner`).
-/

@[expose] public section

namespace Verity.Sampled.Record

/-- **A session's setup in Flock's masked scope**: one table, each table at `25 ≤ m ≤ 35` with `k_log ≤ 27`, at most
`1024` regions and `m_pts ≤ 64`, at at most two link points. -/
def SessionInScope (spec : Flock.SessionSpec) (sts : Array Flock.Stmt) : Prop :=
  sts.size = 1 ∧ (∀ st ∈ sts.toList, 25 ≤ st.m ∧ st.m ≤ 35) ∧
    (∀ st ∈ sts.toList, st.c.kLog ≤ 27 ∧ st.regions.size ≤ 1024 ∧ spec.mPts ≤ 64) ∧ spec.points ≤ 2

/-- **In scope with its setup is `ZkIn`**: one table is between `1` and `200`, and the range is `acct`'s `InRange`. -/
theorem zkIn_of_inScope {x : Claims.Msg} {s : FlockDeploy.Ins} {spec : Flock.SessionSpec} {sts : Array Flock.Stmt}
    (hs : FlockDeploy.setupOf true x = some (s, spec, sts)) (h : SessionInScope spec sts) :
    FlockDeploy.ZkIn x s spec sts :=
  ⟨hs, by have := h.1; omega, by have := h.1; omega, h.2.2.2,
    fun st hst => ⟨(h.2.1 st hst).1, (h.2.1 st hst).2, h.2.2.1 st hst⟩⟩

end Verity.Sampled.Record
