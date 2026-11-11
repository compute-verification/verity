module

public import Proofs.Flock.ZkAny.Tables
public import Proofs.Flock.Claims.ZK.FinishStopAt
public import Proofs.Flock.VBridge.Family.Openings

/-!
# The ZK server's check gates each table's file, at any values

`Flock.Firewall.Finish.repFw` releases a rep's inner proof only where no constraint it hands the inner proof fails on
the rep's own pads (`failingOn`). At a sample's `--zk` table at a worker's clear values (`coinedOf`'s, at `onVals`), that
check is `Table.Gated`'s: where it passes at both reps, the worker's rows hold at the sample's mask slot and extra lanes
at every pad (`rowsAt_of_cons`), so at the sampled pads (`Table.HoldsAt`). The triple's three rows need no check: they
vanish by the pads' own products `h_ab = h_fa·h_fb` and `h_tw = h_tx·h_ty`, which the server samples.

* `ConsAt`: at the sample `ω`, rep `r`'s constraints are its first seven rows on the view's masked values, each on the
  layout's length, and none fails on the sampled pads.
* **`holdsAt_of_cons`**, **`gated_of_cons`**: `Table.HoldsAt` where `ConsAt` holds at both reps, and `Table.Gated`
  behind a check that passes only there.
* `Release.gated_of_tables`: a release's files are gated where each table's is, behind checks that pass wherever the
  release's outcome releases that table's file.
-/

@[expose] public section

namespace FlockSoundness.Discharge.Composed

open Flock Model ZkSession ZkView ZK Refine

local notation "𝔽" => FlockLevel3.GF128

variable (P : LigProver) {mPts : ℕ}

/-- **The ZK server goes on at rep `r` of the sample `ω`** on the constraints `cons`: they are rep `r`'s first seven rows
on the view's masked values (`rowsOf`; the triple's inputs `(ρ, σ)` are unread, `rowsOf_castAdd`), each on the layout's
length, and none fails on the sampled pads (`failingOn`, the pads at their layout indices). -/
def ConsAt (Tz : TabZK 𝔽 mPts) (st : Stmt) (c : CoinsZK (coinedOf P Tz st).Rest)
    (p : Witness (coinedOf P Tz st).S.m × ((coinedOf P Tz st).T c).tab.Vals)
    (ω : ((coinedOf P Tz st).T c).tab.onVals.Rand) (r : Fin 2) (cons : Array (String × Zk.Lin)) : Prop :=
  cons.size = 7 ∧ (∀ q ∈ cons, q.2.a.size = (layOf Tz st).len) ∧
    (∀ k : Fin 7, (ofZk cons[k.val]!.2 : ZkSession.Lin 𝔽 (((coinedOf P Tz st).T c).tab.onVals.nE + 4)) =
      rowsOf Tz st c r ((((coinedOf P Tz st).T c).tab.onVals.view p ω).pre.2.1 r)
        (((coinedOf P Tz st).T c).tab.onVals.view p ω).pre.1
        ((((coinedOf P Tz st).T c).tab.onVals.view p ω).pre.2.2.2.2.2 r).1 (0, 0) (Fin.castAdd 3 k)) ∧
    Flock.Firewall.Finish.failingOn cons
      (arrOf (((coinedOf P Tz st).T c).tab.onVals.hFull (ω.2.2.1 r) ∘
        (padEquiv ((coinedOf P Tz st).T c).tab.onVals.nE).symm)) = #[]

/-- **Where the ZK server goes on at both reps of a sample, the worker's values pass the inner constraint there**: its
rows hold at the sample's mask slot and extra lanes at every pad (`rowsAt_of_cons`), the sampled ones among them. -/
theorem holdsAt_of_cons (Tz : TabZK 𝔽 mPts) (st : Stmt) (c : CoinsZK (coinedOf P Tz st).Rest)
    (p : Witness (coinedOf P Tz st).S.m × ((coinedOf P Tz st).T c).tab.Vals)
    (ω : ((coinedOf P Tz st).T c).tab.onVals.Rand) (h : ∀ r, ∃ cons, ConsAt P Tz st c p ω r cons) :
    ((coinedOf P Tz st).T c).tab.onVals.HoldsAt p ω :=
  rowsAt_of_cons P Tz st c p ω h ω rfl rfl

/-- **The ZK server's check gates the table's file at any clear values**: a check that passes only where the ZK server
goes on at both reps passes only where the worker's values pass the inner constraint at the sampled pads. -/
theorem gated_of_cons (Tz : TabZK 𝔽 mPts) (st : Stmt) (c : CoinsZK (coinedOf P Tz st).Rest)
    (p : Witness (coinedOf P Tz st).S.m × ((coinedOf P Tz st).T c).tab.Vals)
    (go : ((coinedOf P Tz st).T c).tab.onVals.Rand → Bool)
    (h : ∀ ω, go ω = true → ∀ r, ∃ cons, ConsAt P Tz st c p ω r cons) :
    ((coinedOf P Tz st).T c).tab.onVals.Gated go p :=
  fun ω hω => holdsAt_of_cons P Tz st c p ω (h ω hω)

end FlockSoundness.Discharge.Composed

namespace FlockSoundness.ZK.Release

/-- **A release's files are gated where each table's is**: behind per-table checks `go j` that pass wherever the
release's outcome releases table `j`'s file, each table gated by its own check. -/
theorem gated_of_tables {F K W Salt Seed D Hi VL : Type} [Field F] [Field K] [CharP F 2] {J : ℕ}
    (T : Fin J → Table F K W Salt Seed D Hi VL) {St : Type} (out : (∀ j, (T j).Rand) → St) (full : St → Fin J → Bool)
    (w : Fin J → W) (go : ∀ j, (T j).Rand → Bool) (hgo : ∀ ω j, full (out ω) j = true → go j (ω j) = true)
    (hT : ∀ j, (T j).Gated (go j) (w j)) : Gated T out full w :=
  fun ω j h => hT j (ω j) (hgo ω j h)

end FlockSoundness.ZK.Release
