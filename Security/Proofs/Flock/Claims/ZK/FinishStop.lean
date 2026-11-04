import Proofs.Flock.Claims.ZK.PadFree
import Proofs.Flock.Claims.ZK.FinishRep

/-!
# The firewall's stop at a worker's values

`repFw` stops where a constraint it hands the inner proof fails on the rep's pads (`failingOn`). Here that stop is put in
the analysis's terms, at a worker's clear values, toward `LeanRelease.FinishIs`: where the firewall goes on, the
worker's rows hold (`Table.RowsAt`), so the switched view is the worker's own (`Table.swView`).

* `failingOn_eq_empty_iff`: no constraint fails on the pads exactly where each vanishes there (`linAt`).
* `rowsOf_castAdd`: the first seven of rep `r`'s rows read no triple input `(ρ, σ)`.
* **`rowsAt_of_rows`**: where the first seven rows vanish at a worker's clear values at zero pads, for both reps, at the
  mask slot `u` and the extra lanes `R`, its rows hold there at every pad: the residual at every draw through `(u, R)`
  is the rows' batch at the clear values (`resid_padFree`), and the triple's rows vanish at zero pads (`triple_zero`).
* **`noFail_iff_rows`**: at constraints whose forms are rep `r`'s first seven rows on the masked values `v' = v + h`,
  each on the layout's length, at pads with `h_ab = h_fa·h_fb` and `h_tw = h_tx·h_ty`, the firewall's stop does not
  fire on the pads exactly where those rows vanish at the clear values at zero pads (`rows_padFree`). The stop bit is
  `1[some row fails at v]`, a function of the witness, the worker's values, the mask slot and the extra lanes, never of
  the pads.
-/

namespace Flock.Firewall.Finish

/-- No constraint fails on the pads exactly where each vanishes there. -/
theorem failingOn_eq_empty_iff {cons : Array (String × Zk.Lin)} {h : Array F128} :
    failingOn cons h = #[] ↔ ∀ q ∈ cons, linAt q.2 h = 0 := by
  unfold failingOn
  rw [Array.map_eq_empty_iff, Array.filter_eq_empty_iff]
  simp [bne]

end Flock.Firewall.Finish

namespace FlockSoundness.Discharge.Composed

open Flock Model ZkSession ZkView ZK Refine

local notation "𝔽" => FlockLevel3.GF128
local notation "𝕂" => FlockLevel3.GF256

/-- The first seven of rep `r`'s rows read no triple input `(ρ, σ)`. -/
theorem rowsOf_castAdd {mPts : ℕ} (Tz : TabZK 𝔽 mPts) (st : Stmt) (c : CoinsZK (RestOf Tz st)) (r : Fin 2) {nE : ℕ}
    (v : Fin nE → 𝔽) {nC : ℕ} (sh : Fin nC → Fin 128 → 𝔽) (T' : 𝕂) (ρσ ρσ' : 𝔽 × 𝔽) (k : Fin 7) :
    rowsOf Tz st c r v sh T' ρσ (Fin.castAdd 3 k) = rowsOf Tz st c r v sh T' ρσ' (Fin.castAdd 3 k) := by
  simp only [rowsOf, Fin.append_left]

/-- **The firewall's stop at a worker's values is its rows' at zero pads**: at constraints whose forms are rep `r`'s
first seven rows on the masked values `v' = v + h`, each on the layout's length, at pads with `h_ab = h_fa·h_fb` and
`h_tw = h_tx·h_ty`, none fails on the pads exactly where those rows vanish at the clear values `v` at zero pads. -/
theorem noFail_iff_rows {mPts : ℕ} (Tz : TabZK 𝔽 mPts) (st : Stmt) (c : CoinsZK (RestOf Tz st)) (r : Fin 2) {nE : ℕ}
    (v v' : Fin nE → 𝔽) (hp : Fin (nE + 4) → 𝔽) (hE : (layOf Tz st).hab = nE)
    (hvv : ∀ i, v' i = v i + hp (Fin.castAdd 4 i))
    (hab : hp (ix nE (layOf Tz st).hab) = hp (ix nE (layOf Tz st).fa) * hp (ix nE (layOf Tz st).fb))
    (htw : hp (ix nE (layOf Tz st).tw) = hp (ix nE (layOf Tz st).tx) * hp (ix nE (layOf Tz st).ty))
    {nC : ℕ} (sh : Fin nC → Fin 128 → 𝔽) (T' : 𝕂) (hs : (layOf Tz st).len = nE + 4)
    (cons : Array (String × Zk.Lin)) (hsz : cons.size = 7) (hlen : ∀ q ∈ cons, q.2.a.size = (layOf Tz st).len)
    (hform : ∀ k : Fin 7,
      (ofZk cons[k.val]!.2 : ZkSession.Lin 𝔽 (nE + 4)) = rowsOf Tz st c r v' sh T' (0, 0) (Fin.castAdd 3 k)) :
    Flock.Firewall.Finish.failingOn cons (arrOf (padsAt hs hp)) = #[] ↔
      ∀ k : Fin 7, (rowsOf Tz st c r v sh T' (0, 0) (Fin.castAdd 3 k)).eval 0 = 0 := by
  have key : ∀ k : Fin 7, (rowsOf Tz st c r v sh T' (0, 0) (Fin.castAdd 3 k)).eval 0 =
      fOf (Flock.Firewall.Finish.linAt cons[k.val]!.2 (arrOf (padsAt hs hp))) := fun k => by
    have hk : k.val < cons.size := by omega
    have hmem : cons[k.val]! ∈ cons := by
      rw [getElem!_pos cons k.val hk]
      exact Array.getElem_mem hk
    have hl := hlen cons[k.val]! hmem
    rw [Flock.Firewall.Finish.linAt_eq, ← linVal_eq_evalL _ _ hl, ← ofZk_eval hs, hform k,
      rowsOf_castAdd Tz st c r v' sh T' (0, 0)
        (innerEps Tz st c r * hp (ix nE (layOf Tz st).fa) + hp (ix nE (layOf Tz st).tx),
          hp (ix nE (layOf Tz st).fb) + hp (ix nE (layOf Tz st).ty)) k,
      rows_padFree Tz st c r v v' hp hE hvv hab htw sh T' (Fin.castAdd 3 k)]
  rw [Flock.Firewall.Finish.failingOn_eq_empty_iff]
  constructor
  · intro hf k
    have hk : k.val < cons.size := by omega
    have hmem : cons[k.val]! ∈ cons := by
      rw [getElem!_pos cons k.val hk]
      exact Array.getElem_mem hk
    rw [key k, hf cons[k.val]! hmem, fOf_zero]
  · intro hr q hq
    obtain ⟨i, hi, rfl⟩ := Array.mem_iff_getElem.1 hq
    have e := key ⟨i, by omega⟩
    rw [hr, getElem!_pos cons i hi] at e
    exact fOf_injective (e.symm.trans fOf_zero.symm)

variable (P : LigProver) {mPts : ℕ}

/-- **A worker's rows hold where they vanish at its clear values**: where the first seven of each rep's rows vanish on
the worker's clear values at the mask slot `u` and the extra lanes `R`, at zero pads, the rows hold at `(u, R)` at every
pad (`Table.RowsAt`). -/
theorem rowsAt_of_rows (Tz : TabZK 𝔽 mPts) (st : Stmt) :
    let C := coinedOf P Tz st
    ∀ c p (u : (C.T c).tab.onVals.Mask) (R : (C.T c).tab.onVals.XLanes),
      (∀ (r : Fin 2) (k : Fin 7), (rowsOf Tz st c r ((C.T c).tab.onVals.vals r p u R)
          ((C.T c).tab.onVals.nu p + (C.T c).tab.onVals.maskHom u)
          (∑ col, (C.T c).tab.onVals.y1' p u R r col * bFold Tz st c r col.val) (0, 0) (Fin.castAdd 3 k)).eval 0 = 0) →
      (C.T c).tab.onVals.RowsAt p u R := by
  intro C c p u R h ω hu hR r
  have e := resid_padFree P Tz st c r p ω
  rw [hu, hR] at e
  refine sub_eq_zero.mp (e.trans (Finset.sum_eq_zero fun k _ => ?_))
  refine Fin.addCases (fun k => ?_) (fun k => ?_) k
  · rw [h r k, mul_zero]
  · simp only [rowsOf, Fin.append_right]
    exact (congrArg (_ * ·) (triple_zero _ _ k)).trans (mul_zero _)

end FlockSoundness.Discharge.Composed
