import Proofs.Flock.Claims.Complete.Zk.ConsRingSwitch
import Proofs.Flock.Soundness.Discharge.Composed.Inner.Holds
import Proofs.Flock.Claims.ZK.Worker

/-!
# The inner constraint at a worker's values reads no pad

A worker's clear values `v` never see the pads; the replay reads them masked, `v' = v + h`. Each of rep `r`'s ten rows,
on `v'` at the pads `h`, evaluates to what it does on `v` at zero pads (`rows_padFree`): the zerocheck, lincheck and ring
switch rows by their masked evaluations (`ZerocheckMasked.c_eval_eval`, `final_eval`, `FlockComplete.Replay`'s
`lcRunning_eval`, `lcAbValue_eval`), applied once at `(v, v', h)` and once at `(v, v, 0)`; the lane rows by the lane run
at the pads (`laneRun_padFree`) and their coordinates (`limb_padFree`); the triple's rows vanish at both. So the batched
residual `⟨λ, h⟩ − θ` at a draw is a function of the witness, the worker's values, the mask slot and the extra lanes
alone (`resid_padFree`), and the constraint holds at the drawn pads exactly where it holds at every pad
(`rowsAt_iff_pass`): the firewall's own check of the drawn pads decides `Table.RowsAt`, and its stop reveals
`1[¬RowsAt (w, v)]`, a function of the witness and the worker's values, `0` for the honest prover's (`innerOf_holds`).
-/

namespace FlockSoundness.Discharge.Composed

open Flock Model ZkSession ZkView ZK Refine

local notation "𝔽" => FlockLevel3.GF128
local notation "𝕂" => FlockLevel3.GF256

/-! ## Forms and coordinates -/

/-- A masked form is pad-free where its executable value at the pads `h` on `l'` is its value at zero pads on `l`. -/
theorem ofZk_padFree {mPts : ℕ} (Tz : TabZK 𝔽 mPts) (st : Stmt) {nE : ℕ} (hp : Fin (nE + 4) → 𝔽)
    (hs : (layOf Tz st).len = nE + 4) (l l' : Zk.Lin) (hl : l.a.size = (layOf Tz st).len)
    (hl' : l'.a.size = (layOf Tz st).len)
    (he : ZerocheckMasked.evalL l' (arrOf (padsAt hs hp)) = ZerocheckMasked.evalL l (arrOf (padsAt hs 0))) :
    (ofZk l' : ZkSession.Lin 𝔽 (nE + 4)).eval hp = (ofZk l : ZkSession.Lin 𝔽 (nE + 4)).eval 0 := by
  rw [ofZk_eval hs, ofZk_eval hs, linVal_eq_evalL _ _ hl', linVal_eq_evalL _ _ hl, he]

/-- A `K` row's value is its two coordinate rows' (`limbRows`). -/
theorem eval_limbRows {np : ℕ} (row : LinK 𝕂 np) (h : Fin np → 𝔽) :
    LinK.eval execArith h row =
      execArith.ofLimbs ((limbRows execArith row 0).eval h) ((limbRows execArith row 1).eval h) := by
  rw [show (limbRows execArith row 0).eval h =
      ∑ i, (coords execArith (row.1 i)).1 * h i + (coords execArith row.2).1 from rfl,
    show (limbRows execArith row 1).eval h =
      ∑ i, (coords execArith (row.1 i)).2 * h i + (coords execArith row.2).2 from rfl,
    ← ofLimbs_add, ← ofLimbs_sum, ofLimbs_coords execArith execArith_correct]
  simp only [LinK.eval, LinearMap.coe_mk, AddHom.coe_mk]
  congr 1
  refine Finset.sum_congr rfl fun i _ => ?_
  rw [← ofLimbs_mul_embed, ofLimbs_coords execArith execArith_correct, mul_comm]

/-- **Coordinate rows of `K` rows with one value have one value each.** -/
theorem limb_padFree {np : ℕ} (row row' : LinK 𝕂 np) (h h' : Fin np → 𝔽)
    (he : LinK.eval execArith h' row' = LinK.eval execArith h row) (k : Fin 2) :
    (limbRows execArith row' k).eval h' = (limbRows execArith row k).eval h := by
  rw [eval_limbRows row' h', eval_limbRows row h] at he
  obtain ⟨h0, h1⟩ := ofLimbs_inj execArith execArith_correct he
  fin_cases k
  · exact h0
  · exact h1

/-- The triple's rows vanish at zero pads and `ρ = σ = 0`. -/
theorem triple_zero {s : ℕ} (tr : Triple s) (ε : 𝔽) (k : Fin 3) :
    (tripleRows tr ε 0 0 k).eval (0 : Fin s → 𝔽) = 0 := by
  fin_cases k
  · rw [show ((⟨0, by omega⟩ : Fin 3)) = 0 from rfl, tripleRows_eval_zero]
    simp
  · rw [show ((⟨1, by omega⟩ : Fin 3)) = 1 from rfl, tripleRows_eval_one]
    simp
  · rw [show ((⟨2, by omega⟩ : Fin 3)) = 2 from rfl, tripleRows_eval_two]
    simp

/-- A masked value unmasked by its pad (characteristic 2), as an executable form reads it. -/
theorem evalL_masked_pad {x x' : F128} {i s : ℕ} (hi : i < s) (h : Array F128) (hx : x' = x + h[i]!) :
    ZerocheckMasked.evalL (Zk.Lin.masked x' i s) h = x := by
  rw [ZerocheckMasked.evalL_masked _ _ _ hi, hx, ZerocheckMasked.add_add_self]

/-- `z_partial`'s forms at the pads, on its masked values: its clear values. -/
theorem zp_map (lay : Zk.PadLayout) (h : Array F128) (zp zp' : Array F128)
    (hm : ∀ i < 64, zp'[i]! = zp[i]! + h[lay.zp i]!) :
    (Lincheck.lcZp lay zp').map (fun l => ZerocheckMasked.evalL l h) = (Array.range 64).map fun i => zp[i]! := by
  unfold Lincheck.lcZp
  apply Array.ext
  · simp
  · intro i hi _
    simp only [Array.size_map, Array.size_range] at hi
    simp only [Array.getElem_map, Array.getElem_range]
    exact evalL_masked_pad (FlockComplete.Replay.zp_lt_len lay i hi) h (hm i hi)

theorem zp_size (lay : Zk.PadLayout) (zp : Array F128) : ∀ l ∈ Lincheck.lcZp lay zp, l.a.size = lay.len :=
  fun _ hl => by
    obtain ⟨i, _, rfl⟩ := ZerocheckMasked.mem_map_range hl
    simp

/-! ## Rep `r`'s rows on masked values -/

section Rows

variable {mPts : ℕ} (Tz : TabZK 𝔽 mPts) (st : Stmt) (c : CoinsZK (RestOf Tz st)) (r : Fin 2) {nE : ℕ}
  (v v' : Fin nE → 𝔽) (hp : Fin (nE + 4) → 𝔽)

/-- Clear values are masked by zero pads. -/
theorem add_zeroPad : ∀ i, v i = v i + (0 : Fin (nE + 4) → 𝔽) (Fin.castAdd 4 i) := fun i => by simp

/-- A masked value plus its pad is its clear value (characteristic 2). -/
theorem msgAt_pad (hvv : ∀ i, v' i = v i + hp (Fin.castAdd 4 i)) (k : ℕ) (hk : k < nE) :
    msgAt v' k + hp (ix nE k) = msgAt v k := by
  have hix : ix nE k = Fin.castAdd 4 ⟨k, hk⟩ := Fin.ext (by simp [ix, Nat.mod_eq_of_lt (by omega : k < nE + 4)])
  simp only [msgAt, dif_pos hk, hvv, hix]
  rw [add_assoc, CharTwo.add_self_eq_zero, add_zero]

/-- A masked value is its clear value plus its pad, as the executable's arrays hold them. -/
theorem msg_masks (hE : (layOf Tz st).hab = nE) (hvv : ∀ i, v' i = v i + hp (Fin.castAdd 4 i)) (k : ℕ) (hk : k < nE) :
    (msgAt v' k).val = (msgAt v k).val + (arrOf (padsAt (len_eq Tz st hE) hp))[k]! := by
  have hpad : FlockLevel3.GF128.of (arrOf (padsAt (len_eq Tz st hE) hp))[k]! = hp ⟨k, by omega⟩ := by
    rw [show FlockLevel3.GF128.of _ = fOf _ from rfl, arrOf_get _ k (by rw [len_eq Tz st hE]; omega)]
    rfl
  have hix : (Fin.castAdd 4 ⟨k, hk⟩ : Fin (nE + 4)) = ⟨k, by omega⟩ := rfl
  apply Zerocheck.eq_of_of
  rw [Zerocheck.of_val, FlockLevel3.of_add, hpad, Zerocheck.of_val,
    show msgAt v' k = v' ⟨k, hk⟩ from dif_pos hk, show msgAt v k = v ⟨k, hk⟩ from dif_pos hk, hvv, hix]

/-- **The masked zerocheck is the clear one under the pads**, at any clear values. -/
theorem masks_of (hE : (layOf Tz st).hab = nE) (hvv : ∀ i, v' i = v i + hp (Fin.castAdd 4 i)) (n : ℕ)
    (hn : n ≤ st.m - 6) :
    ZerocheckMasked.Masks (layOf Tz st) (arrOf (padsAt (len_eq Tz st hE) hp)) (zcOf Tz st v) (zcOf Tz st v') n := by
  have hv := msg_masks Tz st v v' hp hE hvv
  refine ⟨fun i hi => ?_, fun i hi => ?_, fun j hj => ?_, ?_, ?_, ?_⟩
  · simp only [zcOf]
    rw [Zerocheck.get_map_range _ hi, Zerocheck.get_map_range _ hi]
    exact hv _ (by rw [← hE]; (simp only [layOf, Zk.PadLayout.new, Zk.PadLayout.fa, Zk.PadLayout.fb, Zk.PadLayout.fc, Zk.PadLayout.lc, Zk.PadLayout.zp, Zk.PadLayout.lgOod, Zk.PadLayout.lgMsg, Zk.PadLayout.lgE, Zk.PadLayout.hab, Zk.PadLayout.ml, Zk.PadLayout.r1ab, Zk.PadLayout.r1c, Zk.PadLayout.len, Zk.PadLayout.tw, Zk.PadLayout.tx, Zk.PadLayout.ty] at *; omega))
  · simp only [zcOf]
    rw [Zerocheck.get_map_range _ hi, Zerocheck.get_map_range _ hi]
    exact hv _ (by rw [← hE]; (simp only [layOf, Zk.PadLayout.new, Zk.PadLayout.fa, Zk.PadLayout.fb, Zk.PadLayout.fc, Zk.PadLayout.lc, Zk.PadLayout.zp, Zk.PadLayout.lgOod, Zk.PadLayout.lgMsg, Zk.PadLayout.lgE, Zk.PadLayout.hab, Zk.PadLayout.ml, Zk.PadLayout.r1ab, Zk.PadLayout.r1c, Zk.PadLayout.len, Zk.PadLayout.tw, Zk.PadLayout.tx, Zk.PadLayout.ty] at *; omega))
  · simp only [zcOf]
    rw [Zerocheck.get_map_range _ (by omega : j < st.m - 6), Zerocheck.get_map_range _ (by omega : j < st.m - 6)]
    exact Prod.ext (hv _ (by rw [← hE]; (simp only [layOf, Zk.PadLayout.new, Zk.PadLayout.fa, Zk.PadLayout.fb, Zk.PadLayout.fc, Zk.PadLayout.lc, Zk.PadLayout.zp, Zk.PadLayout.lgOod, Zk.PadLayout.lgMsg, Zk.PadLayout.lgE, Zk.PadLayout.hab, Zk.PadLayout.ml, Zk.PadLayout.r1ab, Zk.PadLayout.r1c, Zk.PadLayout.len, Zk.PadLayout.tw, Zk.PadLayout.tx, Zk.PadLayout.ty] at *; omega))) (hv _ (by rw [← hE]; (simp only [layOf, Zk.PadLayout.new, Zk.PadLayout.fa, Zk.PadLayout.fb, Zk.PadLayout.fc, Zk.PadLayout.lc, Zk.PadLayout.zp, Zk.PadLayout.lgOod, Zk.PadLayout.lgMsg, Zk.PadLayout.lgE, Zk.PadLayout.hab, Zk.PadLayout.ml, Zk.PadLayout.r1ab, Zk.PadLayout.r1c, Zk.PadLayout.len, Zk.PadLayout.tw, Zk.PadLayout.tx, Zk.PadLayout.ty] at *; omega)))
  · exact hv (layOf Tz st).fa (by rw [← hE]; (simp only [layOf, Zk.PadLayout.new, Zk.PadLayout.fa, Zk.PadLayout.fb, Zk.PadLayout.fc, Zk.PadLayout.lc, Zk.PadLayout.zp, Zk.PadLayout.lgOod, Zk.PadLayout.lgMsg, Zk.PadLayout.lgE, Zk.PadLayout.hab, Zk.PadLayout.ml, Zk.PadLayout.r1ab, Zk.PadLayout.r1c, Zk.PadLayout.len, Zk.PadLayout.tw, Zk.PadLayout.tx, Zk.PadLayout.ty] at *; omega))
  · exact hv (layOf Tz st).fb (by rw [← hE]; (simp only [layOf, Zk.PadLayout.new, Zk.PadLayout.fa, Zk.PadLayout.fb, Zk.PadLayout.fc, Zk.PadLayout.lc, Zk.PadLayout.zp, Zk.PadLayout.lgOod, Zk.PadLayout.lgMsg, Zk.PadLayout.lgE, Zk.PadLayout.hab, Zk.PadLayout.ml, Zk.PadLayout.r1ab, Zk.PadLayout.r1c, Zk.PadLayout.len, Zk.PadLayout.tw, Zk.PadLayout.tx, Zk.PadLayout.ty] at *; omega))
  · exact hv (layOf Tz st).fc (by rw [← hE]; (simp only [layOf, Zk.PadLayout.new, Zk.PadLayout.fa, Zk.PadLayout.fb, Zk.PadLayout.fc, Zk.PadLayout.lc, Zk.PadLayout.zp, Zk.PadLayout.lgOod, Zk.PadLayout.lgMsg, Zk.PadLayout.lgE, Zk.PadLayout.hab, Zk.PadLayout.ml, Zk.PadLayout.r1ab, Zk.PadLayout.r1c, Zk.PadLayout.len, Zk.PadLayout.tw, Zk.PadLayout.tx, Zk.PadLayout.ty] at *; omega))

theorem lcr_masks (hE : (layOf Tz st).hab = nE) (hvv : ∀ i, v' i = v i + hp (Fin.castAdd 4 i)) :
    ∀ j < st.c.kLog - 6, (lcRoundsOf Tz st v')[j]! =
      ((lcRoundsOf Tz st v)[j]!.1 + (arrOf (padsAt (len_eq Tz st hE) hp))[(layOf Tz st).lc j 0]!,
        (lcRoundsOf Tz st v)[j]!.2 + (arrOf (padsAt (len_eq Tz st hE) hp))[(layOf Tz st).lc j 1]!) := by
  intro j hj
  simp only [lcRoundsOf]
  rw [Zerocheck.get_map_range _ hj, Zerocheck.get_map_range _ hj]
  exact Prod.ext (msg_masks Tz st v v' hp hE hvv _ (by rw [← hE]; (simp only [layOf, Zk.PadLayout.new, Zk.PadLayout.fa, Zk.PadLayout.fb, Zk.PadLayout.fc, Zk.PadLayout.lc, Zk.PadLayout.zp, Zk.PadLayout.lgOod, Zk.PadLayout.lgMsg, Zk.PadLayout.lgE, Zk.PadLayout.hab, Zk.PadLayout.ml, Zk.PadLayout.r1ab, Zk.PadLayout.r1c, Zk.PadLayout.len, Zk.PadLayout.tw, Zk.PadLayout.tx, Zk.PadLayout.ty] at *; omega)))
    (msg_masks Tz st v v' hp hE hvv _ (by rw [← hE]; (simp only [layOf, Zk.PadLayout.new, Zk.PadLayout.fa, Zk.PadLayout.fb, Zk.PadLayout.fc, Zk.PadLayout.lc, Zk.PadLayout.zp, Zk.PadLayout.lgOod, Zk.PadLayout.lgMsg, Zk.PadLayout.lgE, Zk.PadLayout.hab, Zk.PadLayout.ml, Zk.PadLayout.r1ab, Zk.PadLayout.r1c, Zk.PadLayout.len, Zk.PadLayout.tw, Zk.PadLayout.tx, Zk.PadLayout.ty] at *; omega)))

theorem zp_masks (hE : (layOf Tz st).hab = nE) (hvv : ∀ i, v' i = v i + hp (Fin.castAdd 4 i)) :
    ∀ i < 64, (zPartialOf Tz st v')[i]! =
      (zPartialOf Tz st v)[i]! + (arrOf (padsAt (len_eq Tz st hE) hp))[(layOf Tz st).zp i]! := by
  intro i hi
  simp only [zPartialOf]
  rw [Zerocheck.get_map_range _ hi, Zerocheck.get_map_range _ hi]
  exact msg_masks Tz st v v' hp hE hvv _ (by rw [← hE]; (simp only [layOf, Zk.PadLayout.new, Zk.PadLayout.fa, Zk.PadLayout.fb, Zk.PadLayout.fc, Zk.PadLayout.lc, Zk.PadLayout.zp, Zk.PadLayout.lgOod, Zk.PadLayout.lgMsg, Zk.PadLayout.lgE, Zk.PadLayout.hab, Zk.PadLayout.ml, Zk.PadLayout.r1ab, Zk.PadLayout.r1c, Zk.PadLayout.len, Zk.PadLayout.tw, Zk.PadLayout.tx, Zk.PadLayout.ty] at *; omega))

/-- **The masked lincheck is the clear one under the pads**, at any clear values. -/
theorem lcMasks_of (hE : (layOf Tz st).hab = nE) (hvv : ∀ i, v' i = v i + hp (Fin.castAdd 4 i)) :
    FlockComplete.Replay.LcMasks (layOf Tz st) (arrOf (padsAt (len_eq Tz st hE) hp))
      ⟨lcRoundsOf Tz st v, zPartialOf Tz st v⟩ ⟨lcRoundsOf Tz st v', zPartialOf Tz st v'⟩ (st.c.kLog - 6) :=
  ⟨lcr_masks Tz st v v' hp hE hvv, zp_masks Tz st v v' hp hE hvv⟩

theorem arr_hab (hE : (layOf Tz st).hab = nE)
    (hab : hp (ix nE (layOf Tz st).hab) = hp (ix nE (layOf Tz st).fa) * hp (ix nE (layOf Tz st).fb)) :
    (arrOf (padsAt (len_eq Tz st hE) hp))[(layOf Tz st).hab]! =
      (arrOf (padsAt (len_eq Tz st hE) hp))[(layOf Tz st).fa]! *
        (arrOf (padsAt (len_eq Tz st hE) hp))[(layOf Tz st).fb]! := by
  have hs := len_eq Tz st hE
  obtain ⟨hfa, hfb, -, -, -⟩ := lay_triple Tz st hE
  rw [ix_eq (nE := nE) _ (by omega), ix_eq (nE := nE) _ (by omega), ix_eq (nE := nE) _ (by omega)] at hab
  apply Zerocheck.eq_of_of
  rw [FlockLevel3.of_mul]
  rw [show FlockLevel3.GF128.of (arrOf (padsAt hs hp))[(layOf Tz st).hab]! = fOf _ from rfl,
    show FlockLevel3.GF128.of (arrOf (padsAt hs hp))[(layOf Tz st).fa]! = fOf _ from rfl,
    show FlockLevel3.GF128.of (arrOf (padsAt hs hp))[(layOf Tz st).fb]! = fOf _ from rfl,
    arrOf_get _ _ (by rw [hs]; omega), arrOf_get _ _ (by rw [hs]; omega), arrOf_get _ _ (by rw [hs]; omega)]
  exact hab

/-- **Row 0 (zerocheck c_eval) reads no pad.** -/
theorem row0_padFree (hE : (layOf Tz st).hab = nE) (hvv : ∀ i, v' i = v i + hp (Fin.castAdd 4 i)) :
    (rowCEval Tz st c r v').eval hp = (rowCEval Tz st c r v).eval 0 := by
  refine ofZk_padFree Tz st hp (len_eq Tz st hE) _ _ (by exact size_rowCEval_form _ _ _)
    (by exact size_rowCEval_form _ _ _) ?_
  rw [ZerocheckMasked.c_eval_eval _ _ (zcOf Tz st v) _ 0 (masks_of Tz st v v' hp hE hvv 0 (Nat.zero_le _)),
    ZerocheckMasked.c_eval_eval _ _ (zcOf Tz st v) _ 0 (masks_of Tz st v v 0 hE (add_zeroPad v) 0 (Nat.zero_le _))]

/-- **Row 1 (zerocheck final a·b) reads no pad**, at pads with `h_ab = h_fa·h_fb`. -/
theorem row1_padFree (hE : (layOf Tz st).hab = nE) (hvv : ∀ i, v' i = v i + hp (Fin.castAdd 4 i))
    (hab : hp (ix nE (layOf Tz st).hab) = hp (ix nE (layOf Tz st).fa) * hp (ix nE (layOf Tz st).fb)) :
    (rowAB Tz st c r v').eval hp = (rowAB Tz st c r v).eval 0 := by
  have hn : st.m - 6 ≤ (layOf Tz st).nMlv := by simp [layOf, Zk.PadLayout.new]
  refine ofZk_padFree Tz st hp (len_eq Tz st hE) _ _ (by exact size_rowAB_form _ _ _ _ _ _)
    (by exact size_rowAB_form _ _ _ _ _ _) ?_
  rw [ZerocheckMasked.final_eval _ _ (zcOf Tz st v) _ _ hn (masks_of Tz st v v' hp hE hvv _ le_rfl)
      (arr_hab Tz st hp hE hab),
    ZerocheckMasked.final_eval _ _ (zcOf Tz st v) _ _ hn (masks_of Tz st v v 0 hE (add_zeroPad v) _ le_rfl)
      (arr_hab Tz st 0 hE (by simp))]

/-- **Row 2 (lincheck final) reads no pad.** -/
theorem row2_padFree (hE : (layOf Tz st).hab = nE) (hvv : ∀ i, v' i = v i + hp (Fin.castAdd 4 i)) :
    (rowLC Tz st c r v').eval hp = (rowLC Tz st c r v).eval 0 := by
  have hm1 := masks_of Tz st v v' hp hE hvv 0 (Nat.zero_le _)
  have hm0 := masks_of Tz st v v 0 hE (add_zeroPad v) 0 (Nat.zero_le _)
  have hn : st.c.kLog - 6 ≤ (layOf Tz st).nLc := by simp [layOf, Zk.PadLayout.new]
  refine ofZk_padFree Tz st hp (len_eq Tz st hE) _ _ (by exact Lincheck.lcFinal_size _ _ _ _ _ _ _ _ _ _)
    (by exact Lincheck.lcFinal_size _ _ _ _ _ _ _ _ _ _) ?_
  unfold Lincheck.lcFinal
  rw [ZerocheckMasked.evalL_add _ _ _
      (by rw [FlockComplete.Replay.lcRunning_size, ZerocheckMasked.size_combo]),
    ZerocheckMasked.evalL_add _ _ _
      (by rw [FlockComplete.Replay.lcRunning_size, ZerocheckMasked.size_combo]),
    FlockComplete.Replay.lcRunning_eval _ _ _ _ _ hn (lcMasks_of Tz st v v' hp hE hvv) _ _ _ _ _ _ hm1.finalA
      hm1.finalB _ _ le_rfl,
    FlockComplete.Replay.lcRunning_eval _ _ _ _ _ hn (lcMasks_of Tz st v v 0 hE (add_zeroPad v)) _ _ _ _ _ _
      hm0.finalA hm0.finalB _ _ le_rfl,
    ZerocheckMasked.evalL_combo _ _ _ _ (zp_size _ (zPartialOf Tz st v')),
    ZerocheckMasked.evalL_combo _ _ _ _ (zp_size _ (zPartialOf Tz st v)),
    zp_map _ _ _ _ (zp_masks Tz st v v' hp hE hvv), zp_map _ _ _ _ (zp_masks Tz st v v 0 hE (add_zeroPad v))]

/-- **Row 3 (ring switch ab) reads no pad.** -/
theorem row3_padFree (hE : (layOf Tz st).hab = nE) (hvv : ∀ i, v' i = v i + hp (Fin.castAdd 4 i)) {nC : ℕ}
    (sh : Fin nC → Fin 128 → 𝔽) :
    (rowRsAb Tz st c r v' sh).eval hp = (rowRsAb Tz st c r v sh).eval 0 := by
  refine ofZk_padFree Tz st hp (len_eq Tz st hE) _ _ (by exact Lincheck.lcAbValue_size _ _ _)
    (by exact Lincheck.lcAbValue_size _ _ _) ?_
  rw [FlockComplete.Replay.evalL_addConst, FlockComplete.Replay.evalL_addConst,
    FlockComplete.Replay.lcAbValue_eval _ _ _ _ (zp_masks Tz st v v' hp hE hvv),
    FlockComplete.Replay.lcAbValue_eval _ _ _ _ (zp_masks Tz st v v 0 hE (add_zeroPad v))]

/-- **Row 4 (ring switch c) reads no pad.** -/
theorem row4_padFree (hE : (layOf Tz st).hab = nE) (hvv : ∀ i, v' i = v i + hp (Fin.castAdd 4 i)) {nC : ℕ}
    (sh : Fin nC → Fin 128 → 𝔽) :
    (rowRsC Tz st c r v' sh).eval hp = (rowRsC Tz st c r v sh).eval 0 := by
  refine ofZk_padFree Tz st hp (len_eq Tz st hE) _ _ (by exact ZerocheckMasked.size_masked _ _ _)
    (by exact ZerocheckMasked.size_masked _ _ _) ?_
  rw [FlockComplete.Replay.evalL_addConst, FlockComplete.Replay.evalL_addConst,
    evalL_masked_pad (ZerocheckMasked.fc_lt_len _) _ (masks_of Tz st v v' hp hE hvv 0 (Nat.zero_le _)).finalC,
    evalL_masked_pad (ZerocheckMasked.fc_lt_len _) _
      (masks_of Tz st v v 0 hE (add_zeroPad v) 0 (Nat.zero_le _)).finalC]

/-- A masked `K` message at the pads: its clear coordinates. -/
theorem eval_mkK_pad (hvv : ∀ i, v' i = v i + hp (Fin.castAdd 4 i)) (i₀ i₁ : ℕ) (h₀ : i₀ < nE) (h₁ : i₁ < nE) :
    LinK.eval execArith hp (mkK v' i₀ i₁) = execArith.ofLimbs (msgAt v i₀) (msgAt v i₁) := by
  simp only [mkK, map_add, eval_cK, eval_padK]
  rw [ofLimbs_add, msgAt_pad v v' hp hvv i₀ h₀, msgAt_pad v v' hp hvv i₁ h₁]

/-- **The lane run reads no pad**: after `j ≤ k₀` lane rounds. -/
theorem laneRun_padFree (hE : (layOf Tz st).hab = nE) (hvv : ∀ i, v' i = v i + hp (Fin.castAdd 4 i)) {nC : ℕ}
    (sh : Fin nC → Fin 128 → 𝔽) :
    ∀ j ≤ Tz.sch.k0, LinK.eval execArith hp (laneRun Tz st c r v' sh j) =
      LinK.eval execArith 0 (laneRun Tz st c r v sh j)
  | 0, _ => by
      simp only [laneRun, map_add, map_sum, map_smul, eval_cK, eval_padF, smul_eq_mul]
      refine congrArg (_ + ·) (Finset.sum_congr rfl fun i _ => congrArg (_ * ·) ?_)
      rw [← map_add, ← map_add, msgAt_pad v v' hp hvv _ (lgOod_lt Tz st hE i i.isLt)]
      simp
  | j + 1, hj => by
      have ih := laneRun_padFree hE hvv sh j (by omega)
      have hj' : j < Tz.sch.k0 := by omega
      simp only [laneRun, QuadL.eval, fromMsgL, map_add, map_smul, smul_eq_mul]
      rw [eval_mkK_pad v v' hp hvv _ _ (lgMsg_lt Tz st hE j 0 hj' (by omega)) (lgMsg_lt Tz st hE j 1 hj' (by omega)),
        eval_mkK_pad v v' hp hvv _ _ (lgMsg_lt Tz st hE j 2 hj' (by omega)) (lgMsg_lt Tz st hE j 3 hj' (by omega)),
        eval_mkK_pad v v 0 (add_zeroPad v) _ _ (lgMsg_lt Tz st hE j 0 hj' (by omega))
          (lgMsg_lt Tz st hE j 1 hj' (by omega)),
        eval_mkK_pad v v 0 (add_zeroPad v) _ _ (lgMsg_lt Tz st hE j 2 hj' (by omega))
          (lgMsg_lt Tz st hE j 3 hj' (by omega)), ih]

/-- **Level 0's lane-phase row reads no pad.** -/
theorem laneRow_padFree (hE : (layOf Tz st).hab = nE) (hvv : ∀ i, v' i = v i + hp (Fin.castAdd 4 i)) {nC : ℕ}
    (sh : Fin nC → Fin 128 → 𝔽) (T' : 𝕂) :
    LinK.eval execArith hp (laneRow Tz st c r v' sh T') = LinK.eval execArith 0 (laneRow Tz st c r v sh T') := by
  simp only [laneRow, map_add, map_sub, map_sum, map_smul, smul_eq_mul, eval_cK]
  rw [laneRun_padFree Tz st c r v v' hp hE hvv sh Tz.sch.k0 le_rfl]
  refine congrArg (· - T') (congrArg (_ + ·) (Finset.sum_congr rfl fun i _ => ?_))
  rw [eval_mkK_pad v v' hp hvv _ _ (lgE_lt Tz st hE i 0 i.isLt (by omega)) (lgE_lt Tz st hE i 1 i.isLt (by omega)),
    eval_mkK_pad v v 0 (add_zeroPad v) _ _ (lgE_lt Tz st hE i 0 i.isLt (by omega))
      (lgE_lt Tz st hE i 1 i.isLt (by omega))]

/-- **Rep `r`'s ten rows read no pad**: on the masked values `v' = v + h` at pads `h` with `h_ab = h_fa·h_fb` and
`h_tw = h_tx·h_ty`, and the triple's `(ρ, σ)` at `h`, each row evaluates to what it does on `v` at zero pads and
`(ρ, σ) = (0, 0)`. -/
theorem rows_padFree (hE : (layOf Tz st).hab = nE) (hvv : ∀ i, v' i = v i + hp (Fin.castAdd 4 i))
    (hab : hp (ix nE (layOf Tz st).hab) = hp (ix nE (layOf Tz st).fa) * hp (ix nE (layOf Tz st).fb))
    (htw : hp (ix nE (layOf Tz st).tw) = hp (ix nE (layOf Tz st).tx) * hp (ix nE (layOf Tz st).ty))
    {nC : ℕ} (sh : Fin nC → Fin 128 → 𝔽) (T' : 𝕂) (k : Fin (7 + 3)) :
    (rowsOf Tz st c r v' sh T' (innerEps Tz st c r * hp (ix nE (layOf Tz st).fa) + hp (ix nE (layOf Tz st).tx),
        hp (ix nE (layOf Tz st).fb) + hp (ix nE (layOf Tz st).ty)) k).eval hp =
      (rowsOf Tz st c r v sh T' (0, 0) k).eval 0 := by
  refine Fin.addCases (fun k => ?_) (fun k => ?_) k
  · rw [rowsOf, rowsOf, Fin.append_left, Fin.append_left]
    fin_cases k
    · exact row0_padFree Tz st c r v v' hp hE hvv
    · exact row1_padFree Tz st c r v v' hp hE hvv hab
    · exact row2_padFree Tz st c r v v' hp hE hvv
    · exact row3_padFree Tz st c r v v' hp hE hvv sh
    · exact row4_padFree Tz st c r v v' hp hE hvv sh
    · exact limb_padFree _ _ _ _ (laneRow_padFree Tz st c r v v' hp hE hvv sh T') 0
    · exact limb_padFree _ _ _ _ (laneRow_padFree Tz st c r v v' hp hE hvv sh T') 1
  · rw [rowsOf, rowsOf, Fin.append_right, Fin.append_right]
    exact (triple_rows_zero (trOf nE (layOf Tz st)) _ hp hab htw k).trans (triple_zero _ _ k).symm

end Rows

/-! ## The batched residual at a worker's values -/

/-- The batch's residual `⟨λ, h⟩ − θ` over `PadIdx` is `Σ_k γ_k·R_k(h)` by layout index (`padEquiv`). -/
theorem batch_resid {F : Type} [Field F] {nE n : ℕ} (γ : Fin n → F) (R : Fin n → ZkSession.Lin F (nE + 4))
    (hF : ZK.PadIdx nE → F) :
    ∑ i, batchCoef γ R (padEquiv nE i) * hF i - batchT γ R =
      ∑ k, γ k * (R k).eval (hF ∘ (padEquiv nE).symm) := by
  rw [← batch_eval γ R (hF ∘ (padEquiv nE).symm), ← Equiv.sum_comp (padEquiv nE)]
  simp

variable (P : LigProver) {mPts : ℕ}

/-- **The residual at a draw reads no pad**: at a `--zk` table of a setup, on any clear values a worker supplies
(`Table.onVals`), rep `r`'s batched residual `⟨λ, h⟩ − θ` at the draw `ω` is a function of the witness, the worker's
values, the mask slot and the extra lanes alone: the ten rows on the clear values at zero pads. -/
theorem resid_padFree (Tz : TabZK 𝔽 mPts) (st : Stmt) :
    let C := coinedOf P Tz st
    ∀ c (r : Fin 2) p ω,
      ∑ i, (C.T c).tab.onVals.lam r ((C.T c).tab.onVals.view p ω).pre i * (C.T c).tab.onVals.hFull (ω.2.2.1 r) i -
          (C.T c).tab.onVals.theta r ((C.T c).tab.onVals.view p ω).pre =
        ∑ k, innerGamma Tz st c r k * (rowsOf Tz st c r ((C.T c).tab.onVals.vals r p ω.1 ω.2.1.2.1)
          ((C.T c).tab.onVals.nu p + (C.T c).tab.onVals.maskHom ω.1)
          (∑ col, (C.T c).tab.onVals.y1' p ω.1 ω.2.1.2.1 r col * bFold Tz st c r col.val) (0, 0) k).eval 0 := by
  intro C c r p ω
  have hE : (layOf Tz st).hab = (C.T c).tab.onVals.nE := by
    show _ = padLenOf _ _ - 4
    simp only [padLenOf, layOf, Zk.PadLayout.len, Zk.PadLayout.tw]
    omega
  obtain ⟨hfa, hfb, htx, hty, htw⟩ := lay_triple _ _ hE
  refine (batch_resid (innerGamma Tz st c r)
    (rowsOf Tz st c r (((C.T c).tab.onVals.view p ω).pre.2.1 r) ((C.T c).tab.onVals.view p ω).pre.1
      (((C.T c).tab.onVals.view p ω).pre.2.2.2.2.2 r).1 (((C.T c).tab.onVals.view p ω).pre.2.2.2.2.1 r))
    ((C.T c).tab.onVals.hFull (ω.2.2.1 r))).trans ?_
  refine Finset.sum_congr rfl fun k _ => congrArg (innerGamma Tz st c r k * ·) ?_
  have hvv : ∀ i, (((C.T c).tab.onVals.view p ω).pre.2.1 r) i = (C.T c).tab.onVals.vals r p ω.1 ω.2.1.2.1 i +
      ((C.T c).tab.onVals.hFull (ω.2.2.1 r) ∘ (padEquiv (C.T c).tab.onVals.nE).symm) (Fin.castAdd 4 i) := fun i => by
    show _ + _ = _ + _
    refine congrArg (_ + ·) ?_
    rw [Function.comp_apply, show (padEquiv (C.T c).tab.onVals.nE).symm (Fin.castAdd 4 i) = .inl i from
      (Equiv.symm_apply_eq _).2 (Fin.ext (by simp [padEquiv]))]
    rfl
  have hinner : (((C.T c).tab.onVals.view p ω).pre.2.2.2.2.1 r) =
      (innerEps Tz st c r *
          ((C.T c).tab.onVals.hFull (ω.2.2.1 r) ∘ (padEquiv (C.T c).tab.onVals.nE).symm)
            (ix (C.T c).tab.onVals.nE (layOf Tz st).fa) +
        ((C.T c).tab.onVals.hFull (ω.2.2.1 r) ∘ (padEquiv (C.T c).tab.onVals.nE).symm)
          (ix (C.T c).tab.onVals.nE (layOf Tz st).tx),
       ((C.T c).tab.onVals.hFull (ω.2.2.1 r) ∘ (padEquiv (C.T c).tab.onVals.nE).symm)
          (ix (C.T c).tab.onVals.nE (layOf Tz st).fb) +
        ((C.T c).tab.onVals.hFull (ω.2.2.1 r) ∘ (padEquiv (C.T c).tab.onVals.nE).symm)
          (ix (C.T c).tab.onVals.nE (layOf Tz st).ty)) := by
    rw [comp_symm_lt _ _ hfa, comp_symm_lt _ _ hfb, htx, hty, comp_symm_tail _ 1 (by omega),
      comp_symm_tail _ 2 (by omega)]
    rfl
  have hab : ((C.T c).tab.onVals.hFull (ω.2.2.1 r) ∘ (padEquiv (C.T c).tab.onVals.nE).symm)
        (ix (C.T c).tab.onVals.nE (layOf Tz st).hab) =
      ((C.T c).tab.onVals.hFull (ω.2.2.1 r) ∘ (padEquiv (C.T c).tab.onVals.nE).symm)
          (ix (C.T c).tab.onVals.nE (layOf Tz st).fa) *
        ((C.T c).tab.onVals.hFull (ω.2.2.1 r) ∘ (padEquiv (C.T c).tab.onVals.nE).symm)
          (ix (C.T c).tab.onVals.nE (layOf Tz st).fb) := by
    rw [comp_symm_lt _ _ hfa, comp_symm_lt _ _ hfb,
      show ix (C.T c).tab.onVals.nE (layOf Tz st).hab = ix (C.T c).tab.onVals.nE ((C.T c).tab.onVals.nE + 0) from
        congrArg _ hE, comp_symm_tail _ 0 (by omega)]
    rfl
  have htw' : ((C.T c).tab.onVals.hFull (ω.2.2.1 r) ∘ (padEquiv (C.T c).tab.onVals.nE).symm)
        (ix (C.T c).tab.onVals.nE (layOf Tz st).tw) =
      ((C.T c).tab.onVals.hFull (ω.2.2.1 r) ∘ (padEquiv (C.T c).tab.onVals.nE).symm)
          (ix (C.T c).tab.onVals.nE (layOf Tz st).tx) *
        ((C.T c).tab.onVals.hFull (ω.2.2.1 r) ∘ (padEquiv (C.T c).tab.onVals.nE).symm)
          (ix (C.T c).tab.onVals.nE (layOf Tz st).ty) := by
    rw [htw, htx, hty, comp_symm_tail _ 1 (by omega), comp_symm_tail _ 2 (by omega), comp_symm_tail _ 3 (by omega)]
    rfl
  rw [hinner]
  exact rows_padFree Tz st c r ((C.T c).tab.onVals.vals r p ω.1 ω.2.1.2.1) (((C.T c).tab.onVals.view p ω).pre.2.1 r)
    _ hE hvv hab htw' _ _ k

/-- **Passing at the drawn pads is passing at every pad**: where a worker's values pass rep `r`'s inner constraint at
one draw for every rep, the rows hold at its mask slot and extra lanes (`Table.RowsAt`), at every pad. -/
theorem rowsAt_of_pass (Tz : TabZK 𝔽 mPts) (st : Stmt) :
    let C := coinedOf P Tz st
    ∀ c p (ω₀ : (C.T c).tab.onVals.Rand),
      (∀ r, ∑ i, (C.T c).tab.onVals.lam r ((C.T c).tab.onVals.view p ω₀).pre i *
          (C.T c).tab.onVals.hFull (ω₀.2.2.1 r) i = (C.T c).tab.onVals.theta r ((C.T c).tab.onVals.view p ω₀).pre) →
      (C.T c).tab.onVals.RowsAt p ω₀.1 ω₀.2.1.2.1 := by
  intro C c p ω₀ h0 ω hu hR r
  have e := resid_padFree P Tz st c r p ω
  have e0 := resid_padFree P Tz st c r p ω₀
  rw [hu, hR] at e
  exact sub_eq_zero.mp (e.trans (e0.symm.trans (sub_eq_zero.mpr (h0 r))))

/-- **The constraint at the drawn pads decides `RowsAt`**: a worker's values pass rep `r`'s inner constraint at the
draw `ω₀`, for both reps, exactly where its rows hold at `ω₀`'s mask slot and extra lanes at every pad. The firewall's
own `Finish` check at the drawn pads therefore stops exactly on `¬RowsAt (w, v)`, a function of the witness and the
worker's values. -/
theorem rowsAt_iff_pass (Tz : TabZK 𝔽 mPts) (st : Stmt) :
    let C := coinedOf P Tz st
    ∀ c p (ω₀ : (C.T c).tab.onVals.Rand),
      (C.T c).tab.onVals.RowsAt p ω₀.1 ω₀.2.1.2.1 ↔
        ∀ r, ∑ i, (C.T c).tab.onVals.lam r ((C.T c).tab.onVals.view p ω₀).pre i *
          (C.T c).tab.onVals.hFull (ω₀.2.2.1 r) i = (C.T c).tab.onVals.theta r ((C.T c).tab.onVals.view p ω₀).pre :=
  fun c p ω₀ => ⟨fun h => h ω₀ rfl rfl, rowsAt_of_pass P Tz st c p ω₀⟩

end FlockSoundness.Discharge.Composed
