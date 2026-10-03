import FlockSoundness.Discharge.Composed.Defs
import FlockSoundness.Discharge.Composed.Inner.Lane
import FlockSoundness.Discharge.Composed.Inner.ZerocheckGlue
import FlockSoundness.Discharge.Composed.Inner.RingSwitch

/-!
# `innerOf_holds`: the honest prover's pads satisfy the inner proof's constraint at the draw

`InnerHolds` is `⟨λ, h⟩ = θ` for `verifyInner`'s batch of rep `r`'s ten rows (`lamOf`, `thetaOf`), at the honest pads
`hFull` (`h_ab = h_fa·h_fb`, `h_tw = h_tx·h_ty`). By `batch_eval` it holds once every row vanishes at the pads by layout
index (`batch_of_rows`). The triple's three rows vanish at the honest `(ρ_in, σ_in)` (`triple_rows_zero`).
-/

namespace FlockSoundness.Discharge.Composed

open Game Audit Exec Refine Flock Model ZkSession ZkLink ZkExec DecodeZL FlockLevel3 ZkView

/-- **The batch holds where its rows vanish**: `⟨λ, h⟩ = θ` over `PadIdx` once each row vanishes at `h` by layout index
(`padEquiv`). -/
theorem batch_of_rows {F : Type} [Field F] {nE n : ℕ} (γ : Fin n → F) (R : Fin n → ZkSession.Lin F (nE + 4))
    (hF : ZK.PadIdx nE → F) (h0 : ∀ k, (R k).eval (hF ∘ (padEquiv nE).symm) = 0) :
    ∑ i, batchCoef γ R (padEquiv nE i) * hF i = batchT γ R := by
  have hb := batch_eval γ R (hF ∘ (padEquiv nE).symm)
  have hz : ∑ k, γ k * (R k).eval (hF ∘ (padEquiv nE).symm) = 0 :=
    Finset.sum_eq_zero fun k _ => by rw [h0 k, mul_zero]
  rw [hz, sub_eq_zero] at hb
  rw [← hb, ← Equiv.sum_comp (padEquiv nE)]
  simp

/-- **The triple's rows vanish** at the honest `ρ_in = ε·h_fa + h_tx`, `σ_in = h_fb + h_ty`, where `h_ab = h_fa·h_fb`
and `h_tw = h_tx·h_ty` (characteristic 2). -/
theorem triple_rows_zero {F : Type} [Field F] [CharP F 2] {s : ℕ} (tr : Triple s) (ε : F) (h : Fin s → F)
    (hab : h tr.ab = h tr.fa * h tr.fb) (htw : h tr.tw = h tr.tx * h tr.ty) (k : Fin 3) :
    (tripleRows tr ε (ε * h tr.fa + h tr.tx) (h tr.fb + h tr.ty) k).eval h = 0 := by
  have h2 : (2 : F) = 0 := CharTwo.two_eq_zero
  fin_cases k
  · rw [show ((⟨0, by omega⟩ : Fin 3)) = 0 from rfl, tripleRows_eval_zero]
    linear_combination (ε * h tr.fa + h tr.tx) * h2
  · rw [show ((⟨1, by omega⟩ : Fin 3)) = 1 from rfl, tripleRows_eval_one]
    linear_combination (h tr.fb + h tr.ty) * h2
  · rw [show ((⟨2, by omega⟩ : Fin 3)) = 2 from rfl, tripleRows_eval_two, hab, htw]
    linear_combination (ε * h tr.fa * h tr.fb + h tr.tx * h tr.ty + h tr.fb * h tr.tx + ε * h tr.fa * h tr.ty +
      h tr.tx * h tr.ty) * h2

section Glue

open Flock Model ZkSession ZkView ZK Refine

local notation "𝔽" => FlockLevel3.GF128
local notation "𝕂" => FlockLevel3.GF256

/-- Pads as the executable's array. -/
def arrOf {s : ℕ} (h : Fin s → 𝔽) : Array F128 := Array.ofFn fun i => (h i).val

theorem arrOf_get {s : ℕ} (h : Fin s → 𝔽) (k : ℕ) (hk : k < s) : fOf (arrOf h)[k]! = h ⟨k, hk⟩ := by
  rw [arrOf, getElem!_pos _ k (by simpa using hk), Array.getElem_ofFn]
  rfl

theorem linVal_eq_evalL {s : ℕ} (h : Fin s → 𝔽) (l : Zk.Lin) (hl : l.a.size = s) :
    Lincheck.linVal h l = fOf (ZerocheckMasked.evalL l (arrOf h)) := by
  refine Eq.trans ?_ (ZerocheckMasked.of_evalL l _ s hl).symm
  exact congrArg (fOf l.c + ·) (Finset.sum_congr rfl fun i _ => congrArg (fOf l.a[i.val]! * ·) (arrOf_get h i i.isLt).symm)

/-- The pads on the layout's `len` slots. -/
def padsAt {nE s : ℕ} (hs : s = nE + 4) (hp : Fin (nE + 4) → 𝔽) : Fin s → 𝔽 := fun i => hp (Fin.cast hs i)

theorem ofZk_eval {nE s : ℕ} (hs : s = nE + 4) (hp : Fin (nE + 4) → 𝔽) (l : Zk.Lin) :
    (ofZk l : ZkSession.Lin 𝔽 (nE + 4)).eval hp = Lincheck.linVal (padsAt hs hp) l := by
  subst hs
  simp only [ZkSession.Lin.eval, ofZk, Lincheck.linVal, padsAt]
  rw [add_comm]
  rfl

set_option linter.unusedSimpArgs false in
/-- Close a goal about the pads layout's indices by unfolding them to arithmetic. -/
local macro "lay_omega" : tactic =>
  `(tactic| (simp only [layOf, Zk.PadLayout.new, Zk.PadLayout.fa, Zk.PadLayout.fb, Zk.PadLayout.fc,
    Zk.PadLayout.lc, Zk.PadLayout.zp, Zk.PadLayout.lgOod, Zk.PadLayout.lgMsg, Zk.PadLayout.lgE,
    Zk.PadLayout.hab, Zk.PadLayout.ml, Zk.PadLayout.r1ab, Zk.PadLayout.r1c, Zk.PadLayout.len, Zk.PadLayout.tw,
    Zk.PadLayout.tx, Zk.PadLayout.ty] at *; omega))

theorem val_mul_cast (x y : ZMod 2) : (((x * y).val : ℕ) : 𝔽) = ((x.val : ℕ) : 𝔽) * ((y.val : ℕ) : 𝔽) := by
  simp only [ZMod.natCast_val]
  exact ZMod.cast_mul (dvd_refl 2) x y

/-- `modelFold` realizes the statement by construction. -/
theorem modelFold_realizes (S : Statement) : FoldRealizes (modelFold S) S where
  kLog := rfl
  pin := rfl
  fold α lw ρ v _ _ hv col := by
    simp only [modelFold, Except.ok.injEq] at hv
    subst hv
    rw [getElem!_pos _ col.val (by simp), Array.getElem_map, Array.getElem_range, dif_pos col.isLt]
    rfl
  folded α β lw ρ ts w hw := by
    refine ⟨_, rfl, by simp, ?_⟩
    simp only [modelFold, Except.ok.injEq] at hw
    exact hw.symm

section Rows

variable {mPts : ℕ} (Tz : TabZK 𝔽 mPts) (st : Stmt) (c : CoinsZK (RestOf Tz st)) (r : Fin 2)
  (zw : Witness Tz.S.m) {nE L : ℕ} (R : Fin 2 → Fin L → 𝔽) (hp : Fin (nE + 4) → 𝔽)

theorem ix_eq (k : ℕ) (hk : k < nE + 4) : ix nE k = ⟨k, hk⟩ := Fin.ext (Nat.mod_eq_of_lt hk)

/-- A masked value is its clear value plus its pad (characteristic 2). -/
theorem msgAt_honest (k : ℕ) (hk : k < nE) :
    msgAt (honestMasked Tz st c r zw R hp) k = honestAt Tz st c r zw R k + hp ⟨k, by omega⟩ := by
  rw [← msgAt_add_pad Tz st c r zw R hp k hk, add_assoc, ix_eq k (by omega), CharTwo.add_self_eq_zero, add_zero]

/-! ### The honest values at the zerocheck's and the lincheck's indices -/

theorem honestAt_r1ab (i : ℕ) (hi : i < 64) :
    honestAt Tz st c r zw R ((layOf Tz st).r1ab i) = fOf (zcH Tz st c r zw).round1ab[i]! := by
  unfold honestAt; dsimp only
  rw [ite_eq_left (by lay_omega)]
  rfl

theorem honestAt_r1c (i : ℕ) (hi : i < 64) :
    honestAt Tz st c r zw R ((layOf Tz st).r1c i) = fOf (zcH Tz st c r zw).round1c[i]! := by
  unfold honestAt; dsimp only
  rw [ite_eq_right (by lay_omega), ite_eq_left (by lay_omega), show (layOf Tz st).r1c i - 64 = i by lay_omega]

theorem honestAt_ml (j t : ℕ) (hj : j < st.m - 6) (ht : t < 2) :
    honestAt Tz st c r zw R ((layOf Tz st).ml j t) =
      if t = 0 then fOf (zcH Tz st c r zw).rounds[j]!.1 else fOf (zcH Tz st c r zw).rounds[j]!.2 := by
  unfold honestAt; dsimp only
  rw [ite_eq_right (by lay_omega), ite_eq_right (by lay_omega), ite_eq_left (by lay_omega),
    show ((layOf Tz st).ml j t - 128) / 2 = j by lay_omega, show ((layOf Tz st).ml j t - 128) % 2 = t by lay_omega]

theorem honestAt_fa : honestAt Tz st c r zw R (layOf Tz st).fa = fOf (zcH Tz st c r zw).finalA := by
  unfold honestAt; dsimp only
  rw [ite_eq_right (by lay_omega), ite_eq_right (by lay_omega), ite_eq_right (by lay_omega), ite_eq_left rfl]

theorem honestAt_fb : honestAt Tz st c r zw R (layOf Tz st).fb = fOf (zcH Tz st c r zw).finalB := by
  unfold honestAt; dsimp only
  rw [ite_eq_right (by lay_omega), ite_eq_right (by lay_omega), ite_eq_right (by lay_omega),
    ite_eq_right (by lay_omega), ite_eq_left rfl]

theorem honestAt_fc : honestAt Tz st c r zw R (layOf Tz st).fc = fOf (zcH Tz st c r zw).finalC := by
  unfold honestAt; dsimp only
  rw [ite_eq_right (by lay_omega), ite_eq_right (by lay_omega), ite_eq_right (by lay_omega),
    ite_eq_right (by lay_omega), ite_eq_right (by lay_omega), ite_eq_left rfl]

theorem honestAt_lc (j t : ℕ) (hj : j < st.c.kLog - 6) (ht : t < 2) :
    honestAt Tz st c r zw R ((layOf Tz st).lc j t) =
      if t = 0 then (lcH Tz st c r zw j).1 else (lcH Tz st c r zw j).2 := by
  unfold honestAt; dsimp only
  rw [ite_eq_right (by lay_omega), ite_eq_right (by lay_omega), ite_eq_right (by lay_omega),
    ite_eq_right (by lay_omega), ite_eq_right (by lay_omega), ite_eq_right (by lay_omega), ite_eq_left (by lay_omega),
    show ((layOf Tz st).lc j t - (layOf Tz st).fc - 1) / 2 = j by lay_omega,
    show ((layOf Tz st).lc j t - (layOf Tz st).fc - 1) % 2 = t by lay_omega]

theorem honestAt_zp (i : ℕ) (hi : i < 64) :
    honestAt Tz st c r zw R ((layOf Tz st).zp i) = zpH Tz st c r zw i := by
  unfold honestAt; dsimp only
  rw [ite_eq_right (by lay_omega), ite_eq_right (by lay_omega), ite_eq_right (by lay_omega),
    ite_eq_right (by lay_omega), ite_eq_right (by lay_omega), ite_eq_right (by lay_omega), ite_eq_right (by lay_omega),
    ite_eq_left (by lay_omega), show (layOf Tz st).zp i - (layOf Tz st).zp 0 = i by lay_omega]

/-- The pads on the layout's slots, `len = nE + 4`. -/
theorem len_eq (hE : (layOf Tz st).hab = nE) : (layOf Tz st).len = nE + 4 := by lay_omega

/-- **The masked zerocheck is the honest one under the pads** (`Masks`), on its first `n ≤ m − 6` rounds. -/
theorem masks_honest (hE : (layOf Tz st).hab = nE) (n : ℕ) (hn : n ≤ st.m - 6) :
    ZerocheckMasked.Masks (layOf Tz st) (arrOf (padsAt (len_eq Tz st hE) hp)) (zcH Tz st c r zw)
      (zcOf Tz st (honestMasked Tz st c r zw R hp)) n := by
  have hpad : ∀ k (hk : k < nE), FlockLevel3.GF128.of (arrOf (padsAt (len_eq Tz st hE) hp))[k]! = hp ⟨k, by omega⟩ :=
    fun k hk => by rw [show FlockLevel3.GF128.of _ = fOf _ from rfl, arrOf_get _ k (by rw [len_eq Tz st hE]; omega)]; rfl
  have hv : ∀ k (hk : k < nE) (x : F128), honestAt Tz st c r zw R k = fOf x →
      (msgAt (honestMasked Tz st c r zw R hp) k).val = x + (arrOf (padsAt (len_eq Tz st hE) hp))[k]! := by
    intro k hk x hx
    apply Zerocheck.eq_of_of
    rw [Zerocheck.of_val, FlockLevel3.of_add, hpad k hk, msgAt_honest Tz st c r zw R hp k hk, hx]
  refine ⟨fun i hi => ?_, fun i hi => ?_, fun j hj => ?_, ?_, ?_, ?_⟩
  · rw [zcOf, Zerocheck.get_map_range _ hi]
    exact hv _ (by rw [← hE]; lay_omega) _ (honestAt_r1ab Tz st c r zw R i hi)
  · rw [zcOf, Zerocheck.get_map_range _ hi]
    exact hv _ (by rw [← hE]; lay_omega) _ (honestAt_r1c Tz st c r zw R i hi)
  · rw [zcOf, Zerocheck.get_map_range _ (by omega : j < st.m - 6)]
    exact Prod.ext (hv _ (by rw [← hE]; lay_omega) _ (honestAt_ml Tz st c r zw R j 0 (by omega) (by omega)))
      (hv _ (by rw [← hE]; lay_omega) _ (honestAt_ml Tz st c r zw R j 1 (by omega) (by omega)))
  · exact hv _ (by rw [← hE]; lay_omega) _ (honestAt_fa Tz st c r zw R)
  · exact hv _ (by rw [← hE]; lay_omega) _ (honestAt_fb Tz st c r zw R)
  · exact hv _ (by rw [← hE]; lay_omega) _ (honestAt_fc Tz st c r zw R)

theorem size_rowCEval_form (lay : Zk.PadLayout) (zc : ZerocheckProof) (z : F128) :
    ((ZerocheckMasked.computedC lay zc z).add (ZerocheckMasked.fcL lay zc)).a.size = lay.len := by
  simp [Zk.Lin.add, ZerocheckMasked.size_computedC, ZerocheckMasked.fcL, Zk.Lin.masked]

theorem size_rowAB_form (lay : Zk.PadLayout) (zc : ZerocheckProof) (rr rho : Array F128) (z : F128) (n : ℕ) :
    ((ZerocheckMasked.running lay zc rr rho z n).add (ZerocheckMasked.prodL lay zc)).a.size = lay.len := by
  simp [Zk.Lin.add, ZerocheckMasked.size_running, ZerocheckMasked.prodL]

/-- **Row 0 (zerocheck c_eval) vanishes** at the honest pads, at every coin (`c_eval_zero`). -/
theorem row0_honest (hE : (layOf Tz st).hab = nE) (hm : st.m = Tz.S.m) :
    (rowCEval Tz st c r (honestMasked Tz st c r zw R hp)).eval hp = 0 := by
  rw [rowCEval, ofZk_eval (len_eq Tz st hE), linVal_eq_evalL _ _ (size_rowCEval_form _ _ _),
    ZerocheckMasked.c_eval_zero (layOf Tz st) _ (Tz.S.m - 6) (aT Tz zw) (bT Tz zw) (cT Tz zw) (rF Tz st c r)
      (rhoF Tz st c r) (zcF Tz st c r) _ (masks_honest Tz st c r zw R hp hE _ (by omega))]
  exact fOf_zero

/-- **Row 1 (zerocheck final a·b) vanishes** at the honest pads with `h_ab = h_fa·h_fb`, for valid tables, where no
outer coin is 1 (`final_zero`). -/
theorem row1_honest (hE : (layOf Tz st).hab = nE) (hm : st.m = Tz.S.m)
    (hvalid : ∀ i < 2 ^ (Tz.S.m - 6 + 6), aT Tz zw i * bT Tz zw i = cT Tz zw i)
    (hr : ∀ k < Tz.S.m - 6, (rF Tz st c r)[k]! ≠ 1)
    (hab : hp ⟨(layOf Tz st).hab, by omega⟩ =
      hp ⟨(layOf Tz st).fa, by rw [← hE]; lay_omega⟩ * hp ⟨(layOf Tz st).fb, by rw [← hE]; lay_omega⟩) :
    (rowAB Tz st c r (honestMasked Tz st c r zw R hp)).eval hp = 0 := by
  have hs := len_eq Tz st hE
  have hab' : (arrOf (padsAt hs hp))[(layOf Tz st).hab]! =
      (arrOf (padsAt hs hp))[(layOf Tz st).fa]! * (arrOf (padsAt hs hp))[(layOf Tz st).fb]! := by
    apply Zerocheck.eq_of_of
    rw [FlockLevel3.of_mul]
    rw [show FlockLevel3.GF128.of (arrOf (padsAt hs hp))[(layOf Tz st).hab]! = fOf _ from rfl,
      show FlockLevel3.GF128.of (arrOf (padsAt hs hp))[(layOf Tz st).fa]! = fOf _ from rfl,
      show FlockLevel3.GF128.of (arrOf (padsAt hs hp))[(layOf Tz st).fb]! = fOf _ from rfl,
      arrOf_get _ _ (by rw [hs]; omega), arrOf_get _ _ (by rw [hs, ← hE]; lay_omega),
      arrOf_get _ _ (by rw [hs, ← hE]; lay_omega)]
    exact hab
  rw [rowAB, ofZk_eval hs, linVal_eq_evalL _ _ (size_rowAB_form _ _ _ _ _ _), show st.m - 6 = Tz.S.m - 6 by rw [hm],
    ZerocheckMasked.final_zero (layOf Tz st) _ (Tz.S.m - 6) (by simp [layOf, Zk.PadLayout.new, hm]) (aT Tz zw)
      (bT Tz zw) (cT Tz zw) hvalid (rF Tz st c r) (rhoF Tz st c r) hr (zcF Tz st c r) _
      (masks_honest Tz st c r zw R hp hE _ (by omega)) hab']
  exact fOf_zero

/-- A masked value plus its pad is its clear value (characteristic 2), as `Lincheck.pad` reads the pads. -/
theorem masked_add_pad (hE : (layOf Tz st).hab = nE) (k : ℕ) (hk : k < nE) :
    fOf (msgAt (honestMasked Tz st c r zw R hp) k).val + Lincheck.pad (padsAt (len_eq Tz st hE) hp) k =
      honestAt Tz st c r zw R k := by
  have hk' : k < (layOf Tz st).len := by rw [len_eq Tz st hE]; omega
  rw [Lincheck.pad, dite_eq_left hk']
  show msgAt (honestMasked Tz st c r zw R hp) k + hp ⟨k, by omega⟩ = _
  rw [msgAt_honest Tz st c r zw R hp k hk]
  exact (add_assoc _ _ _).trans ((congrArg (_ + ·) (CharTwo.add_self_eq_zero _)).trans (add_zero _))

/-- **Row 2 (lincheck final) vanishes** at the honest pads, and the `ab` claim, valued at the pads' `linVal` of the
replay's `ab` value, holds for the table (`lincheck_complete` on `modelFold`), given that the clear `final_a`, `final_b`
are the true `a`, `b` claims at the zerocheck's coins. -/
theorem row2_honest (hE : (layOf Tz st).hab = nE) (hm : st.m = Tz.S.m) (hk : st.c.kLog = Tz.S.kLog)
    (hw : Tz.S.Satisfies zw)
    (hA : (⟨fOf (zcH Tz st c r zw).finalA, execArith.lagS (fOf (zcF Tz st c r)), vecOf (Tz.S.m - 6) (rhoF Tz st c r)⟩ :
      Model.Claim 𝔽 Tz.S.m).Holds (aVec Tz.S zw))
    (hB : (⟨fOf (zcH Tz st c r zw).finalB, execArith.lagS (fOf (zcF Tz st c r)), vecOf (Tz.S.m - 6) (rhoF Tz st c r)⟩ :
      Model.Claim 𝔽 Tz.S.m).Holds (bVec Tz.S zw)) :
    (rowLC Tz st c r (honestMasked Tz st c r zw R hp)).eval hp = 0 ∧
      (Lincheck.claimOf Tz.S.m (Lincheck.lcAbClaim Tz.S.m Tz.S.kLog (rhoF Tz st c r) (tsF Tz st c r) (c.rest.rPrime r))
        (Lincheck.linVal (padsAt (len_eq Tz st hE) hp) (Lincheck.lcAbValue (layOf Tz st)
          (zPartialOf Tz st (honestMasked Tz st c r zw R hp)) (c.rest.rPrime r)))).Holds zw := by
  have hs := len_eq Tz st hE
  have hrho : (rhoF Tz st c r).size = Tz.S.m - 6 := by simp [rhoF, hm]
  have hts : (tsF Tz st c r).size = Tz.S.kLog - 6 := by simp [tsF, hk]
  have hcomb : (modelFold Tz.S).folded (c.rest.lcAlpha r) (c.rest.lcBeta r) (skipWeights (zcF Tz st c r))
      ((rhoF Tz st c r).extract 0 (Tz.S.kLog - 6)) (tsF Tz st c r) = .ok (combvOf Tz st c r) := by
    rw [← hk]; rfl
  have hfa : (layOf Tz st).fa < nE := by rw [← hE]; lay_omega
  have hfb : (layOf Tz st).fb < nE := by rw [← hE]; lay_omega
  have hA' := masked_add_pad Tz st c r zw R hp hE _ hfa
  have hB' := masked_add_pad Tz st c r zw R hp hE _ hfb
  rw [honestAt_fa] at hA'
  rw [honestAt_fb] at hB'
  have hround : ∀ j < Tz.S.kLog - 6,
      (fOf (lcRoundsOf Tz st (honestMasked Tz st c r zw R hp))[j]!.1 +
          Lincheck.pad (padsAt hs hp) ((layOf Tz st).lc j 0),
        fOf (lcRoundsOf Tz st (honestMasked Tz st c r zw R hp))[j]!.2 +
          Lincheck.pad (padsAt hs hp) ((layOf Tz st).lc j 1)) =
        Lincheck.lcMsg (2 ^ Tz.S.kLog) (Lincheck.lcU Tz.S (zcF Tz st c r) (c.rest.lcAlpha r) (c.rest.lcBeta r)
          (rhoF Tz st c r)) (Lincheck.lcV Tz.S (rhoF Tz st c r) zw) ((tsF Tz st c r).toList.map fOf) j := by
    intro j hj
    have hj' : j < st.c.kLog - 6 := by rw [hk]; exact hj
    rw [lcRoundsOf, Zerocheck.get_map_range _ hj']
    refine Prod.ext ?_ ?_
    · exact (masked_add_pad Tz st c r zw R hp hE _ (by rw [← hE]; lay_omega)).trans
        (honestAt_lc Tz st c r zw R j 0 hj' (by omega))
    · exact (masked_add_pad Tz st c r zw R hp hE _ (by rw [← hE]; lay_omega)).trans
        (honestAt_lc Tz st c r zw R j 1 hj' (by omega))
  have hzp : ∀ i < 64, fOf (zPartialOf Tz st (honestMasked Tz st c r zw R hp))[i]! +
      Lincheck.pad (padsAt hs hp) ((layOf Tz st).zp i) =
        foldAll ((tsF Tz st c r).toList.map fOf) (2 ^ Tz.S.kLog) (Lincheck.lcV Tz.S (rhoF Tz st c r) zw) i := by
    intro i hi
    rw [zPartialOf, Zerocheck.get_map_range _ hi]
    exact (masked_add_pad Tz st c r zw R hp hE _ (by rw [← hE]; lay_omega)).trans (honestAt_zp Tz st c r zw R i hi)
  have hL := Lincheck.lincheck_complete Tz.S (modelFold Tz.S) (modelFold_realizes Tz.S) zw hw (layOf Tz st)
    (padsAt hs hp) (zcF Tz st c r) (c.rest.lcAlpha r) (c.rest.lcBeta r) (c.rest.rPrime r) (rhoF Tz st c r)
    (tsF Tz st c r) hrho hts (zcOf Tz st (honestMasked Tz st c r zw R hp)).finalA
    (zcOf Tz st (honestMasked Tz st c r zw R hp)).finalB (lcRoundsOf Tz st (honestMasked Tz st c r zw R hp))
    (zPartialOf Tz st (honestMasked Tz st c r zw R hp)) (combvOf Tz st c r)
    (by rw [show fOf (zcOf Tz st (honestMasked Tz st c r zw R hp)).finalA + _ = _ from hA']; exact hA)
    (by rw [show fOf (zcOf Tz st (honestMasked Tz st c r zw R hp)).finalB + _ = _ from hB']; exact hB)
    hround hzp hcomb
  refine ⟨?_, hL.2⟩
  have h1 := hL.1
  rw [← hk] at h1
  rw [rowLC, ofZk_eval hs]
  exact h1

/-- **Row 3 (ring switch ab) vanishes** at the pads where the `ab` claim holds at its value and `ŝ_{2r}` is the
honest `shat` at the claim (characteristic 2). -/
theorem row3_honest (hE : (layOf Tz st).hab = nE) (hm : st.m = Tz.S.m) (hk : st.c.kLog = Tz.S.kLog)
    {nC : ℕ} (sh : Fin nC → Fin 128 → 𝔽)
    (hab : (Lincheck.claimOf Tz.S.m (Lincheck.lcAbClaim Tz.S.m Tz.S.kLog (rhoF Tz st c r) (tsF Tz st c r)
        (c.rest.rPrime r)) (Lincheck.linVal (padsAt (len_eq Tz st hE) hp) (Lincheck.lcAbValue (layOf Tz st)
          (zPartialOf Tz st (honestMasked Tz st c r zw R hp)) (c.rest.rPrime r)))).Holds zw)
    (hs : ∀ v : Fin 128, fOf (shArr sh (2 * r.val))[v.val]! = shat (by have := Tz.hm; omega)
      (RingSwitch.claimOf Tz.S.m (Lincheck.lcAbClaim Tz.S.m Tz.S.kLog (rhoF Tz st c r) (tsF Tz st c r)
        (c.rest.rPrime r))) zw v) :
    (rowRsAb Tz st c r (honestMasked Tz st c r zw R hp) sh).eval hp = 0 := by
  have h7 : 7 ≤ Tz.S.m := by have := Tz.hm; omega
  have hcl : abClaim Tz st c r =
      Lincheck.lcAbClaim Tz.S.m Tz.S.kLog (rhoF Tz st c r) (tsF Tz st c r) (c.rest.rPrime r) := by
    simp only [abClaim, hm, hk]
  have hcc : fOf (Zk.claimCheck (Lincheck.lcAbClaim Tz.S.m Tz.S.kLog (rhoF Tz st c r) (tsF Tz st c r)
      (c.rest.rPrime r)) (shArr sh (2 * r.val))) = (RingSwitch.claimOf Tz.S.m (Lincheck.lcAbClaim Tz.S.m Tz.S.kLog
        (rhoF Tz st c r) (tsF Tz st c r) (c.rest.rPrime r))).eval zw := by
    rw [RingSwitch.fOf_claimCheck h7, claim_eval_eq h7]
    exact Finset.sum_congr rfl fun v _ => by rw [show vecOf 128 _ v = _ from hs v]
  rw [rowRsAb, hcl, ofZk_eval (len_eq Tz st hE), Lincheck.linVal_addConst, hcc]
  refine (congrArg (· + _) hab.symm).trans ?_
  exact CharTwo.add_self_eq_zero _

/-- **Row 4 (ring switch c) vanishes** at the honest pads where `ŝ_{2r+1}` is the honest `shat` at the `c` claim: the
clear `final_c` is the claim's value (`finalC_eq_eval`). -/
theorem row4_honest (hE : (layOf Tz st).hab = nE) {nC : ℕ} (sh : Fin nC → Fin 128 → 𝔽)
    (hs : ∀ v : Fin 128, fOf (shArr sh (2 * r.val + 1))[v.val]! = shat (by have := Tz.hm; omega)
      (RingSwitch.claimOf Tz.S.m (cClaim Tz st c r)) zw v) :
    (rowRsC Tz st c r (honestMasked Tz st c r zw R hp) sh).eval hp = 0 := by
  have h7 : 7 ≤ Tz.S.m := by have := Tz.hm; omega
  have hfc : (layOf Tz st).fc < nE := by rw [← hE]; lay_omega
  have hC := ZerocheckMasked.finalC_eq_eval Tz.S.m (six_le Tz) (aT Tz zw) (bT Tz zw) (cT Tz zw) zw
    (fun i => by rw [cT, dite_eq_left i.isLt]) (rF Tz st c r) (rhoF Tz st c r) (zcF Tz st c r)
  have hcc : fOf (Zk.claimCheck (cClaim Tz st c r) (shArr sh (2 * r.val + 1))) =
      (RingSwitch.claimOf Tz.S.m (cClaim Tz st c r)).eval zw := by
    rw [RingSwitch.fOf_claimCheck h7, claim_eval_eq h7]
    exact Finset.sum_congr rfl fun v _ => by rw [show vecOf 128 _ v = _ from hs v]
  rw [rowRsC, ofZk_eval (len_eq Tz st hE), Lincheck.linVal_addConst, Lincheck.linVal_masked,
    show fOf (zcOf Tz st (honestMasked Tz st c r zw R hp)).finalC + _ = _ from
      (masked_add_pad Tz st c r zw R hp hE _ hfc), honestAt_fc, hcc]
  exact (congrArg (· + _) hC).trans (CharTwo.add_self_eq_zero _)

/-- The tables are valid where the witness satisfies the statement (the R1CS in every block). -/
theorem hvalid_of (hw : Tz.S.Satisfies zw) : ∀ i < 2 ^ (Tz.S.m - 6 + 6), aT Tz zw i * bT Tz zw i = cT Tz zw i := by
  intro i hi
  rw [Nat.sub_add_cancel (six_le Tz)] at hi
  rw [aT, bT, cT, dite_eq_left hi, dite_eq_left hi, dite_eq_left hi, ← val_mul_cast]
  refine congrArg (fun x : ZMod 2 => ((x.val : ℕ) : 𝔽)) ?_
  exact (hw.1 ⟨i / 2 ^ Tz.S.kLog, Tz.S.block_index_lt ⟨i, hi⟩⟩ ⟨i % 2 ^ Tz.S.kLog, Nat.mod_lt _ (Nat.two_pow_pos _)⟩).trans
    (congrArg zw (Fin.ext (Nat.div_add_mod' i _)))

/-- No zerocheck coin is 1 where no outer coin is (`Admits`' last clause). -/
theorem rF_ne_one (h : ∀ x ∈ List.ofFn (c.rest.rOuter r), x ≠ 1) : ∀ k : ℕ, (rF Tz st c r)[k]! ≠ 1 :=
  Zerocheck.pinned_append_ne_one (Array.ofFn (c.rest.rOuter r)) fun k hk => by
    rw [getElem!_pos _ k hk, Array.getElem_ofFn]
    exact h _ (List.mem_ofFn.2 ⟨_, rfl⟩)

/-- **Rep `r`'s ten rows vanish** at the honest pads, given the bridges each row reads. -/
theorem rows_honest (hE : (layOf Tz st).hab = nE) (hm : st.m = Tz.S.m) (hk : st.c.kLog = Tz.S.kLog)
    (hw : Tz.S.Satisfies zw) (hr : ∀ k : ℕ, (rF Tz st c r)[k]! ≠ 1)
    (hA : (⟨fOf (zcH Tz st c r zw).finalA, execArith.lagS (fOf (zcF Tz st c r)), vecOf (Tz.S.m - 6) (rhoF Tz st c r)⟩ :
      Model.Claim 𝔽 Tz.S.m).Holds (aVec Tz.S zw))
    (hB : (⟨fOf (zcH Tz st c r zw).finalB, execArith.lagS (fOf (zcF Tz st c r)), vecOf (Tz.S.m - 6) (rhoF Tz st c r)⟩ :
      Model.Claim 𝔽 Tz.S.m).Holds (bVec Tz.S zw))
    {nC : ℕ} (sh : Fin nC → Fin 128 → 𝔽)
    (hs0 : ∀ v : Fin 128, fOf (shArr sh (2 * r.val))[v.val]! = shat (by have := Tz.hm; omega)
      (RingSwitch.claimOf Tz.S.m (Lincheck.lcAbClaim Tz.S.m Tz.S.kLog (rhoF Tz st c r) (tsF Tz st c r)
        (c.rest.rPrime r))) zw v)
    (hs1 : ∀ v : Fin 128, fOf (shArr sh (2 * r.val + 1))[v.val]! = shat (by have := Tz.hm; omega)
      (RingSwitch.claimOf Tz.S.m (cClaim Tz st c r)) zw v)
    (hk0 : Tz.sch.k0 ≤ Tz.S.m - 7) (hL : L = 2 ^ (Tz.S.m - 7 - Tz.sch.k0))
    (hT : openT Tz st c r sh = ∑ y ∈ Finset.range (2 ^ (Tz.S.m - 7)), wordOf zw y * openB Tz st c r y)
    (y' : Fin L → 𝕂)
    (hy : ∀ col : Fin L, y' col = ∑ l : Fin (2 ^ Tz.sch.k0),
      (∏ i : Fin Tz.sch.k0, if l.val.testBit i.val then FlockLevel3.GF256.of (c.rest.lane r i)
        else 1 + FlockLevel3.GF256.of (c.rest.lane r i)) *
      execArith.embed (wordOf zw (l.val + 2 ^ Tz.sch.k0 * col.val)) +
        (rhoK Tz st c r 0 * execArith.embed (R 0 col) + rhoK Tz st c r 1 * execArith.embed (R 1 col)))
    (hab : hp (ix nE (layOf Tz st).hab) = hp (ix nE (layOf Tz st).fa) * hp (ix nE (layOf Tz st).fb))
    (htw : hp (ix nE (layOf Tz st).tw) = hp (ix nE (layOf Tz st).tx) * hp (ix nE (layOf Tz st).ty))
    (k : Fin (7 + 3)) :
    (rowsOf Tz st c r (honestMasked Tz st c r zw R hp) sh (∑ col : Fin L, y' col * bFold Tz st c r col.val)
      (innerEps Tz st c r * hp (ix nE (layOf Tz st).fa) + hp (ix nE (layOf Tz st).tx),
        hp (ix nE (layOf Tz st).fb) + hp (ix nE (layOf Tz st).ty)) k).eval hp = 0 := by
  have h2 := row2_honest Tz st c r zw R hp hE hm hk hw hA hB
  refine Fin.addCases (fun k => ?_) (fun k => ?_) k
  · rw [rowsOf, Fin.append_left]
    fin_cases k
    · exact row0_honest Tz st c r zw R hp hE hm
    · have hab' := hab
      rw [ix_eq (nE := nE) _ (by omega), ix_eq (nE := nE) _ (by rw [← hE]; lay_omega),
        ix_eq (nE := nE) _ (by rw [← hE]; lay_omega)] at hab'
      exact row1_honest Tz st c r zw R hp hE hm (hvalid_of Tz zw hw) (fun k _ => hr k) hab'
    · exact h2.1
    · exact row3_honest Tz st c r zw R hp hE hm hk sh h2.2 hs0
    · exact row4_honest Tz st c r zw R hp hE sh hs1
    · exact laneRows_honest Tz st c r zw R hp hE hk0 hL sh hT y' hy 0
    · exact laneRows_honest Tz st c r zw R hp hE hk0 hL sh hT y' hy 1
  · rw [rowsOf, Fin.append_right]
    exact triple_rows_zero (trOf nE (layOf Tz st)) _ hp hab htw k

end Rows

/-- The pads at a layout index below `nE` are the exposed ones. -/
theorem comp_symm_lt {nE : ℕ} (hf : PadIdx nE → 𝔽) (k : ℕ) (hk : k < nE) :
    (hf ∘ (padEquiv nE).symm) (ix nE k) = hf (.inl ⟨k, hk⟩) := by
  rw [Function.comp_apply]
  refine congrArg hf ((Equiv.symm_apply_eq _).2 (Fin.ext ?_))
  simp [padEquiv, ix, Nat.mod_eq_of_lt (by omega : k < nE + 4)]

/-- The pads at the layout's last four indices are `PadIdx`'s tail, in `tailPerm`'s order. -/
theorem comp_symm_tail {nE : ℕ} (hf : PadIdx nE → 𝔽) (j : ℕ) (hj : j < 4) :
    (hf ∘ (padEquiv nE).symm) (ix nE (nE + j)) = hf (.inr (tailPerm.symm ⟨j, hj⟩)) := by
  rw [Function.comp_apply]
  refine congrArg hf ((Equiv.symm_apply_eq _).2 (Fin.ext ?_))
  simp [padEquiv, ix, Nat.mod_eq_of_lt (by omega : nE + j < nE + 4)]

/-- The layout's triple indices at `hab = nE`. -/
theorem lay_triple {mPts : ℕ} (Tz : TabZK 𝔽 mPts) (st : Stmt) {nE : ℕ} (hE : (layOf Tz st).hab = nE) :
    (layOf Tz st).fa < nE ∧ (layOf Tz st).fb < nE ∧ (layOf Tz st).tx = nE + 1 ∧ (layOf Tz st).ty = nE + 2 ∧
      (layOf Tz st).tw = nE + 3 := by
  subst hE
  exact ⟨by lay_omega, by lay_omega, rfl, rfl, rfl⟩

end Glue

variable {mPts n : ℕ} {Reg : Type}

/-- **The inner relation at the draw** (open, inner lane): at every coin vector the honest prover doesn't refuse, every
valid witness whose registered values are the public ones satisfies `Table.InnerHolds` (§2.4 line 16's ten
constraints). -/
theorem innerOf_holds (I : Inputs) (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
    (y₀ : DrawSetupZK I mPts S₀ (dj S₀)) (S : Finset (Fin n)) (R : Reg) (y : DrawSetupZK I mPts S (dj S)) :
    let C := coinedOf (tablesAtZK (Reg := Reg) dj y₀ S R) y.x.st
    ∀ c, C.Admits c → ∀ w, C.Valid w → (C.T c).tab.sReg w = (C.T c).tab.pubReg → (C.T c).tab.InnerHolds w := by
  intro C c hc w hw hreg ω r
  apply batch_of_rows
  intro k
  have hE : (layOf (tablesAtZK (Reg := Reg) dj y₀ S R) y.x.st).hab = (C.T c).tab.nE := by
    show _ = padLenOf _ _ - 4
    simp only [padLenOf, layOf, Zk.PadLayout.len, Zk.PadLayout.tw]
    omega
  obtain ⟨hfa, hfb, htx, hty, htw⟩ := lay_triple _ _ hE
  have hv : (ZK.Table.View.pre (C.T c).tab ((C.T c).tab.view w ω)).2.1 r =
      honestMasked (tablesAtZK dj y₀ S R) y.x.st c r (zWith y.x.st w ω.1) (ω.2.1.2.1 r)
        ((C.T c).tab.hFull (ω.2.2.1 r) ∘ (padEquiv (C.T c).tab.nE).symm) := funext fun i => by
    show _ + _ = _ + _
    refine congrArg (_ + ·) ?_
    rw [Function.comp_apply, show (padEquiv (C.T c).tab.nE).symm (Fin.castAdd 4 i) = .inl i from
      (Equiv.symm_apply_eq _).2 (Fin.ext (by simp [padEquiv]))]
    rfl
  have hinner : (ZK.Table.View.pre (C.T c).tab ((C.T c).tab.view w ω)).2.2.2.2.1 r =
      (innerEps (tablesAtZK dj y₀ S R) y.x.st c r *
          ((C.T c).tab.hFull (ω.2.2.1 r) ∘ (padEquiv (C.T c).tab.nE).symm)
            (ix (C.T c).tab.nE (layOf (tablesAtZK dj y₀ S R) y.x.st).fa) +
        ((C.T c).tab.hFull (ω.2.2.1 r) ∘ (padEquiv (C.T c).tab.nE).symm)
          (ix (C.T c).tab.nE (layOf (tablesAtZK dj y₀ S R) y.x.st).tx),
       ((C.T c).tab.hFull (ω.2.2.1 r) ∘ (padEquiv (C.T c).tab.nE).symm)
          (ix (C.T c).tab.nE (layOf (tablesAtZK dj y₀ S R) y.x.st).fb) +
        ((C.T c).tab.hFull (ω.2.2.1 r) ∘ (padEquiv (C.T c).tab.nE).symm)
          (ix (C.T c).tab.nE (layOf (tablesAtZK dj y₀ S R) y.x.st).ty)) := by
    rw [comp_symm_lt _ _ hfa, comp_symm_lt _ _ hfb, htx, hty, comp_symm_tail _ 1 (by omega),
      comp_symm_tail _ 2 (by omega)]
    rfl
  have hab : ((C.T c).tab.hFull (ω.2.2.1 r) ∘ (padEquiv (C.T c).tab.nE).symm)
        (ix (C.T c).tab.nE (layOf (tablesAtZK dj y₀ S R) y.x.st).hab) =
      ((C.T c).tab.hFull (ω.2.2.1 r) ∘ (padEquiv (C.T c).tab.nE).symm)
          (ix (C.T c).tab.nE (layOf (tablesAtZK dj y₀ S R) y.x.st).fa) *
        ((C.T c).tab.hFull (ω.2.2.1 r) ∘ (padEquiv (C.T c).tab.nE).symm)
          (ix (C.T c).tab.nE (layOf (tablesAtZK dj y₀ S R) y.x.st).fb) := by
    rw [comp_symm_lt _ _ hfa, comp_symm_lt _ _ hfb,
      show ix (C.T c).tab.nE (layOf (tablesAtZK dj y₀ S R) y.x.st).hab = ix (C.T c).tab.nE ((C.T c).tab.nE + 0) from
        congrArg _ hE, comp_symm_tail _ 0 (by omega)]
    rfl
  have htw' : ((C.T c).tab.hFull (ω.2.2.1 r) ∘ (padEquiv (C.T c).tab.nE).symm)
        (ix (C.T c).tab.nE (layOf (tablesAtZK dj y₀ S R) y.x.st).tw) =
      ((C.T c).tab.hFull (ω.2.2.1 r) ∘ (padEquiv (C.T c).tab.nE).symm)
          (ix (C.T c).tab.nE (layOf (tablesAtZK dj y₀ S R) y.x.st).tx) *
        ((C.T c).tab.hFull (ω.2.2.1 r) ∘ (padEquiv (C.T c).tab.nE).symm)
          (ix (C.T c).tab.nE (layOf (tablesAtZK dj y₀ S R) y.x.st).ty) := by
    rw [htw, htx, hty, comp_symm_tail _ 1 (by omega), comp_symm_tail _ 2 (by omega), comp_symm_tail _ 3 (by omega)]
    rfl
  rw [hv, hinner]
  refine rows_honest (tablesAtZK dj y₀ S R) y.x.st c r (zWith y.x.st w ω.1) (ω.2.1.2.1 r) _ hE ?hm ?hk ?hw
    (rF_ne_one _ _ c r (hc.2.2.2 r)) ?hA ?hB _ ?hs0 ?hs1 ?hk0 ?hL ?hT ((C.T c).tab.y1' w ω.1 ω.2.1.2.1 r)
    (fun _ => rfl) hab htw' k
  -- the executable statement's sizes are the table's, and the schedule's lane split
  case hm => sorry
  case hk => sorry
  case hk0 => sorry
  case hL => sorry
  -- the prover's table `z = (w, u)` satisfies the statement
  case hw => sorry
  -- the clear `final_a`, `final_b` are the `a`, `b` claims at the zerocheck's coins
  case hA => sorry
  case hB => sorry
  -- the masked claims' `ŝ` is the honest `shat` at the replay's claims, and the opening's target is `Σ_y f(y)·b(y)`
  case hs0 => sorry
  case hs1 => sorry
  case hT => sorry

end FlockSoundness.Discharge.Composed
