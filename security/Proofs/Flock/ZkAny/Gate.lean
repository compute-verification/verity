module

public import Proofs.Flock.ZkAny.Tables
public import Proofs.Flock.Claims.ZK.FinishStopAt
public import Proofs.Flock.VBridge.Family.Openings

/-! The ZK server's check gates each table's file at any clear values (`gated_of_cons`); the triple's three rows need no check, since they vanish by the pads' own products, which the server samples. -/

@[expose] public section

namespace FlockSoundness.Discharge.Composed

open Flock Model ZkSession ZkView ZK Refine

local notation "𝔽" => FlockLevel3.GF128

variable (P : LigProver) {mPts : ℕ}

def ConsAt (Tz : TabZK 𝔽 mPts) (st : Stmt) (c : CoinsZK (coinedOf P Tz st).Rest)
    (p : Witness (coinedOf P Tz st).S.m × ((coinedOf P Tz st).T c).tab.Vals)
    (ω : ((coinedOf P Tz st).T c).tab.onVals.Rand) (r : Fin 2) (cons : Array (String × Zk.Lin)) : Prop :=
  cons.size = 7 ∧ (∀ q ∈ cons, q.2.a.size = (layOf Tz st).len) ∧
    (∀ k : Fin 7, (ofZk cons[k.val]!.2 : ZkSession.Lin 𝔽 (((coinedOf P Tz st).T c).tab.onVals.nE + 4)) =
      rowsOf Tz st c r ((((coinedOf P Tz st).T c).tab.onVals.view p ω).pre.2.1 r)
        (((coinedOf P Tz st).T c).tab.onVals.view p ω).pre.1
        ((((coinedOf P Tz st).T c).tab.onVals.view p ω).pre.2.2.2.2.2 r).1 (0, 0) (Fin.castAdd 3 k)) ∧
    Flock.ZkServer.Finish.failingOn cons
      (arrOf (((coinedOf P Tz st).T c).tab.onVals.hFull (ω.2.2.1 r) ∘
        (padEquiv ((coinedOf P Tz st).T c).tab.onVals.nE).symm)) = #[]

theorem holdsAt_of_cons (Tz : TabZK 𝔽 mPts) (st : Stmt) (c : CoinsZK (coinedOf P Tz st).Rest)
    (p : Witness (coinedOf P Tz st).S.m × ((coinedOf P Tz st).T c).tab.Vals)
    (ω : ((coinedOf P Tz st).T c).tab.onVals.Rand) (h : ∀ r, ∃ cons, ConsAt P Tz st c p ω r cons) :
    ((coinedOf P Tz st).T c).tab.onVals.HoldsAt p ω :=
  rowsAt_of_cons P Tz st c p ω h ω rfl rfl

theorem gated_of_cons (Tz : TabZK 𝔽 mPts) (st : Stmt) (c : CoinsZK (coinedOf P Tz st).Rest)
    (p : Witness (coinedOf P Tz st).S.m × ((coinedOf P Tz st).T c).tab.Vals)
    (go : ((coinedOf P Tz st).T c).tab.onVals.Rand → Bool)
    (h : ∀ ω, go ω = true → ∀ r, ∃ cons, ConsAt P Tz st c p ω r cons) :
    ((coinedOf P Tz st).T c).tab.onVals.Gated go p :=
  fun ω hω => holdsAt_of_cons P Tz st c p ω (h ω hω)

end FlockSoundness.Discharge.Composed

namespace FlockSoundness.ZK.Release

theorem gated_of_tables {F K W Salt Seed D Hi VL : Type} [Field F] [Field K] [CharP F 2] {J : ℕ}
    (T : Fin J → Table F K W Salt Seed D Hi VL) {St : Type} (out : (∀ j, (T j).Rand) → St) (full : St → Fin J → Bool)
    (w : Fin J → W) (go : ∀ j, (T j).Rand → Bool) (hgo : ∀ ω j, full (out ω) j = true → go j (ω j) = true)
    (hT : ∀ j, (T j).Gated (go j) (w j)) : Gated T out full w :=
  fun ω j h => hT j (ω j) (hgo ω j h)

end FlockSoundness.ZK.Release
