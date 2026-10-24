import Proofs.Flock.Claims.ZK.FinishStop

/-!
# The firewall's stop at a table's own draw

`FinishStop` decides the firewall's stop at any pads and masked values related as a draw relates them. Here it is at a
`--zk` table's own draw `ω₀` of a worker's values `p`: the masked values are the view's, the pads are the draw's
(`hFull`, by layout index), and where the firewall goes on at both reps with constraints that are the rows, the worker's
rows hold at `ω₀`'s mask slot and extra lanes (`Table.RowsAt`), so the switched view is the worker's own.

* `arrOf_padsAt`: the pads on the layout's slots are the pads, as the executable's array.
* `viewPads`: at a draw, the view's masked values are the worker's clear values plus the exposed pads, and the drawn
  pads have `h_ab = h_fa·h_fb` and `h_tw = h_tx·h_ty`.
* **`rowsAt_of_cons`**: where, at each rep, the firewall's constraints are the first seven rows on the view's masked
  values, each on the layout's length, and none fails on the drawn pads, `RowsAt p ω₀.1 ω₀.2.1.2.1`.
-/

namespace FlockSoundness.Discharge.Composed

open Flock Model ZkSession ZkView ZK Refine

local notation "𝔽" => FlockLevel3.GF128
local notation "𝕂" => FlockLevel3.GF256

/-- The pads on the layout's slots are the pads, as the executable's array. -/
theorem arrOf_padsAt {nE s : ℕ} (hs : s = nE + 4) (hp : Fin (nE + 4) → 𝔽) : arrOf (padsAt hs hp) = arrOf hp := by
  subst hs
  rfl

variable (P : LigProver) {mPts : ℕ}

/-- **A draw's masked values and pads**: the view's masked values at rep `r` are the worker's clear values plus the
exposed pads, and the drawn pads have `h_ab = h_fa·h_fb` and `h_tw = h_tx·h_ty`, by layout index. -/
theorem viewPads (Tz : TabZK 𝔽 mPts) (st : Stmt) :
    let C := coinedOf P Tz st
    ∀ c (r : Fin 2) p (ω : (C.T c).tab.onVals.Rand),
      (∀ i, ((C.T c).tab.onVals.view p ω).pre.2.1 r i = (C.T c).tab.onVals.vals r p ω.1 ω.2.1.2.1 i +
        ((C.T c).tab.onVals.hFull (ω.2.2.1 r) ∘ (padEquiv (C.T c).tab.onVals.nE).symm) (Fin.castAdd 4 i)) ∧
      ((C.T c).tab.onVals.hFull (ω.2.2.1 r) ∘ (padEquiv (C.T c).tab.onVals.nE).symm)
          (ix (C.T c).tab.onVals.nE (layOf Tz st).hab) =
        ((C.T c).tab.onVals.hFull (ω.2.2.1 r) ∘ (padEquiv (C.T c).tab.onVals.nE).symm)
            (ix (C.T c).tab.onVals.nE (layOf Tz st).fa) *
          ((C.T c).tab.onVals.hFull (ω.2.2.1 r) ∘ (padEquiv (C.T c).tab.onVals.nE).symm)
            (ix (C.T c).tab.onVals.nE (layOf Tz st).fb) ∧
      ((C.T c).tab.onVals.hFull (ω.2.2.1 r) ∘ (padEquiv (C.T c).tab.onVals.nE).symm)
          (ix (C.T c).tab.onVals.nE (layOf Tz st).tw) =
        ((C.T c).tab.onVals.hFull (ω.2.2.1 r) ∘ (padEquiv (C.T c).tab.onVals.nE).symm)
            (ix (C.T c).tab.onVals.nE (layOf Tz st).tx) *
          ((C.T c).tab.onVals.hFull (ω.2.2.1 r) ∘ (padEquiv (C.T c).tab.onVals.nE).symm)
            (ix (C.T c).tab.onVals.nE (layOf Tz st).ty) := by
  intro C c r p ω
  have hE : (layOf Tz st).hab = (C.T c).tab.onVals.nE := by
    show _ = padLenOf _ _ - 4
    simp only [padLenOf, layOf, Zk.PadLayout.len, Zk.PadLayout.tw]
    omega
  obtain ⟨hfa, hfb, htx, hty, htw⟩ := lay_triple _ _ hE
  refine ⟨fun i => ?_, ?_, ?_⟩
  · show _ + _ = _ + _
    refine congrArg (_ + ·) ?_
    rw [Function.comp_apply, show (padEquiv (C.T c).tab.onVals.nE).symm (Fin.castAdd 4 i) = .inl i from
      (Equiv.symm_apply_eq _).2 (Fin.ext (by simp [padEquiv]))]
    rfl
  · rw [comp_symm_lt _ _ hfa, comp_symm_lt _ _ hfb,
      show ix (C.T c).tab.onVals.nE (layOf Tz st).hab = ix (C.T c).tab.onVals.nE ((C.T c).tab.onVals.nE + 0) from
        congrArg _ hE, comp_symm_tail _ 0 (by omega)]
    rfl
  · rw [htw, htx, hty, comp_symm_tail _ 1 (by omega), comp_symm_tail _ 2 (by omega), comp_symm_tail _ 3 (by omega)]
    rfl

/-- **Where the firewall goes on at a table's draw, the worker's rows hold there**: where, at each rep, the firewall's
constraints are the first seven rows on the view's masked values, each on the layout's length, and none fails on the
drawn pads, the worker's rows hold at the draw's mask slot and extra lanes at every pad. -/
theorem rowsAt_of_cons (Tz : TabZK 𝔽 mPts) (st : Stmt) :
    let C := coinedOf P Tz st
    ∀ c p (ω₀ : (C.T c).tab.onVals.Rand),
      (∀ r : Fin 2, ∃ cons : Array (String × Zk.Lin), cons.size = 7 ∧
        (∀ q ∈ cons, q.2.a.size = (layOf Tz st).len) ∧
        (∀ k : Fin 7, (ofZk cons[k.val]!.2 : ZkSession.Lin 𝔽 ((C.T c).tab.onVals.nE + 4)) =
          rowsOf Tz st c r (((C.T c).tab.onVals.view p ω₀).pre.2.1 r) ((C.T c).tab.onVals.view p ω₀).pre.1
            (((C.T c).tab.onVals.view p ω₀).pre.2.2.2.2.2 r).1 (0, 0) (Fin.castAdd 3 k)) ∧
        Flock.Firewall.Finish.failingOn cons
          (arrOf ((C.T c).tab.onVals.hFull (ω₀.2.2.1 r) ∘ (padEquiv (C.T c).tab.onVals.nE).symm)) = #[]) →
      (C.T c).tab.onVals.RowsAt p ω₀.1 ω₀.2.1.2.1 := by
  intro C c p ω₀ h
  refine rowsAt_of_rows P Tz st c p ω₀.1 ω₀.2.1.2.1 fun r => ?_
  obtain ⟨cons, hsz, hlen, hform, hf⟩ := h r
  have hE : (layOf Tz st).hab = (C.T c).tab.onVals.nE := by
    show _ = padLenOf _ _ - 4
    simp only [padLenOf, layOf, Zk.PadLayout.len, Zk.PadLayout.tw]
    omega
  have hs := len_eq Tz st hE
  obtain ⟨hvv, hab, htw⟩ := viewPads P Tz st c r p ω₀
  rw [← arrOf_padsAt hs] at hf
  exact (noFail_iff_rows Tz st c r _ _ _ hE hvv hab htw _ _ hs cons hsz hlen hform).1 hf

end FlockSoundness.Discharge.Composed
