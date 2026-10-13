import Proofs.Flock.Recursive.Raw

/-!
# The registrant's openings: the recursive audit's premise on the provers it covers

A `--zk` transcript holds a registered row's `b ‖ c` and path only, and `ReadsX.committed` asks for the row and salt
under it, so acceptance says nothing about whether the prover can open what it registered. Under `cr/sha-512` alone, a
prover can register a `c` it knows no salt for. `RegistrantCovers` (`registrant-holds-openings`) is the premise that
excludes it: the guarantee covers only provers that hold a row and salt under every leaf they registered at a row the
session's reads file `own` names, dishonest provers included, and a prover that registers a hash it can't open is
outside it. `FlockC3.Compose.registrantCovers_attained` (`Proofs.Flock.C3.Compose.RowsReg`) meets it at every family
of V*'s sessions whose registrant opens the rows the verifier reads, together with the guarantee's other hypotheses on
them.
-/

namespace Flock.Assumptions

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind Integrate Placed StrictCR
  DecodeZL Composed ZkReg FailClosed

/-- Premise (`registrant-holds-openings`) on the provers the guarantee covers. **The prover holds its openings.** At
each typing of V*'s session `x` as the prover supplies it, every row of a port the session's reads file `own` names is
a registered read of the session's record `rd`, at every unit: the prover holds a row and salt under each leaf it
registered there. The record is the prover's, honest or not; the registered roots and positions are the verifier's own
(`own`, `--registered`). A prover that registers a hash it can't open is outside the guarantee. -/
def RegistrantCovers {mPts nv J : ℕ} {Reg' : Type} {c : Flock.Circuit} {hU : Built c.unit}
    {hscope : Layout.scopeOk c = true} {Lv : Law nv} (x : ZkOuterR mPts nv J Reg' c hU hscope Lv) : Prop :=
  ∀ ht S₀ d₀ y₀ hc hpub (u : Fin nv) (q : Fin c.ports.size), (∃ pt ∈ x.own, pt.name = c.ports[q].name) →
    (x.rd ht S₀ d₀ y₀ hc hpub).isRead (.inl (u, q))

end Flock.Assumptions
