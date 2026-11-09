module

public import Proofs.Flock.Claims.ZK.Worker
import Mathlib.Algebra.Order.Chebyshev
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Combinatorics.Enumerative.DoubleCounting
import Mathlib.Data.Nat.Bitwise
import Mathlib.Tactic.Positivity.Finset
import Proofs.Flock.Soundness.Assumptions
import Proofs.Flock.Soundness.Game
import Proofs.Flock.Soundness.Rewinding
import Proofs.Flock.Soundness.ZK.Adaptivity
import Proofs.Flock.Soundness.ZK.Blocks
import Proofs.Flock.Soundness.ZK.Hiding
import Proofs.Flock.Soundness.ZK.RealLeaves
import Proofs.Flock.Soundness.ZK.SHVZK
import Proofs.Flock.Soundness.ZK.Session
import Std.Data.TreeMap.Raw.Lemmas

/-!
# Lemma B at values read from what the workers see

`Worker` proves Lemma B at values that read a table's mask slot and level 0's hiding (`Table.Vals`). A worker sees more
before its algebraic phase: level 0's salts, at the opened positions and at the others, and every table's cells, not
only its own. Its values are then a function of what all the tables' workers see (`Table.Seen`), and a masked value
may read randomness sampled after its pad, so Lemma B's sequential masking no longer applies to the switched view.

The chain is reordered instead. With the pads trees' hidden leaves ideal first, a bijection of each table's
randomness that keeps what the workers see moves the switched view at constant values `c` to the honest view
(`fixE`): where `c`'s rows hold it moves the free pads by `padShift`, `μ`, `pad_h` and `pad_μ` so that every message
stays (`swView_fix`), elsewhere it is `shift`. Then level 0's hidden leaves go ideal and Lemma B is the honest prover's.

* `constVals`, `fixAmt`, `fixE`: the constant values, the move of the inner proof's pads and the bijection.
* `realCore_fix`, `swView_fix`, `swViewWith_fix`, `swDig_inl_fix`: the switched view, with its hidden leaves given, and
  its level-0 digests at constant values, at the moved randomness, are the honest prover's.
* `fixP`, `cut_swPre_fixP`: before the last coin the view shows no `C_h`, so a fix map that moves the free pads and `μ`
  alone makes the switched pre-final view, cut, the honest one at every coin vector.
* `swPre_congr`, `Hm96.swViewR_congr`: the switched views read the values at their own mask slot and level 0's hiding
  alone, so values read from what the workers see are constant values on each fiber of what they see.
* `mix_shvzk_hm96`, `fam_sw_shvzk_hm96`: Lemma B in that order, for one table and jointly at `J` tables, whose values
  read every table's seen randomness.
-/

@[expose] public section

namespace FlockSoundness.ZK.Table

open Finset

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

variable {F K W Salt Seed D Hi VL : Type} [Field F] [Field K]

/-- **Constant clear values**: `c` at every witness, mask slot and level 0's hiding. -/
def constVals (T : Table F K W Salt Seed D Hi VL) (c : Fin 2 → Fin T.nE → F) : T.Vals := fun r _ _ _ => c r

/-- `padShift` from the honest values to `c`'s, as the table's own type. -/
def padShiftC (T : Table F K W Salt Seed D Hi VL) (w : W) (c : Fin 2 → Fin T.nE → F) (u : T.Mask) (V : T.Vis) :
    Fin 2 → (Fin T.nE → F) × F × F :=
  T.onVals.padShift (w, T.constVals c) (w, T.vals) u V

theorem padShiftC_apply (T : Table F K W Salt Seed D Hi VL) (w : W) (c : Fin 2 → Fin T.nE → F) (u : T.Mask)
    (V : T.Vis) (r : Fin 2) :
    T.padShiftC w c u V r = (fun i => c r i - T.vals r w u V i, -(T.ζ r * (c r T.ia - T.vals r w u V T.ia)),
      -(c r T.ib - T.vals r w u V T.ib)) :=
  rfl

variable [CharP F 2]

open Classical in
/-- **The move of the inner proof's pads** at `(u, V, h)`: where `c`'s rows hold, with `h' := h + padShift` and
`d := h − h'` on the full pad vectors, `pad_h` by a solution `δ` of `C_p·δ = C_h·d`, `pad_μ` by `−δ/β` and `μ` by
`d/β`; nothing elsewhere. -/
noncomputable def fixAmt (T : Table F K W Salt Seed D Hi VL) (w : W) (c : Fin 2 → Fin T.nE → F) (u : T.Mask)
    (V : T.Vis) (h : Fin 2 → (Fin T.nE → F) × F × F) :
    (Fin 2 → Fin T.nPad → F) × (Fin 2 → Fin T.nPad → F) × (Fin 2 → PadIdx T.nE → F) :=
  if T.onVals.RowsAt (w, T.constVals c) u V then
    (fun r => Function.invFun (fun p : Fin T.nPad → F => fun k => ∑ j, T.Cp r k j * p j) fun k =>
        ∑ i, T.Ch r k i * (T.hFull (h r) i - T.hFull (h r + T.padShiftC w c u V r) i),
      fun r j => -(Function.invFun (fun p : Fin T.nPad → F => fun k => ∑ j, T.Cp r k j * p j) (fun k =>
        ∑ i, T.Ch r k i * (T.hFull (h r) i -
          T.hFull (h r + T.padShiftC w c u V r) i)) j / T.β r),
      fun r i => (T.hFull (h r) i - T.hFull (h r + T.padShiftC w c u V r) i) / T.β r)
  else 0

/-- **The fix map** at constant values `c`: the free pads moved by `padShift`, the inner proof's pads by `fixAmt`, the
mask slot, level 0's hiding and every salt and seed kept. -/
noncomputable def fixE (T : Table F K W Salt Seed D Hi VL) (w : W) (c : Fin 2 → Fin T.nE → F) : T.Rand ≃ T.Rand where
  toFun ω := (ω.1, ω.2.1, ω.2.2.1 + T.padShiftC w c ω.1 ω.2.1,
    ω.2.2.2.1 + (T.fixAmt w c ω.1 ω.2.1 ω.2.2.1).1, ω.2.2.2.2.1 + (T.fixAmt w c ω.1 ω.2.1 ω.2.2.1).2.1,
    ω.2.2.2.2.2.1 + (T.fixAmt w c ω.1 ω.2.1 ω.2.2.1).2.2, ω.2.2.2.2.2.2)
  invFun ω :=
    (ω.1, ω.2.1, ω.2.2.1 - T.padShiftC w c ω.1 ω.2.1,
      ω.2.2.2.1 - (T.fixAmt w c ω.1 ω.2.1 (ω.2.2.1 - T.padShiftC w c ω.1 ω.2.1)).1,
      ω.2.2.2.2.1 -
        (T.fixAmt w c ω.1 ω.2.1 (ω.2.2.1 - T.padShiftC w c ω.1 ω.2.1)).2.1,
      ω.2.2.2.2.2.1 -
        (T.fixAmt w c ω.1 ω.2.1 (ω.2.2.1 - T.padShiftC w c ω.1 ω.2.1)).2.2,
      ω.2.2.2.2.2.2)
  left_inv ω := by
    obtain ⟨u, V, h, padh, padμ, μ, s⟩ := ω
    simp only [add_sub_cancel_right]
  right_inv ω := by
    obtain ⟨u, V, h, padh, padμ, μ, s⟩ := ω
    simp only [sub_add_cancel]

theorem fixE_apply (T : Table F K W Salt Seed D Hi VL) (w : W) (c : Fin 2 → Fin T.nE → F) (ω : T.Rand) :
    T.fixE w c ω = (ω.1, ω.2.1, ω.2.2.1 + T.padShiftC w c ω.1 ω.2.1,
      ω.2.2.2.1 + (T.fixAmt w c ω.1 ω.2.1 ω.2.2.1).1, ω.2.2.2.2.1 + (T.fixAmt w c ω.1 ω.2.1 ω.2.2.1).2.1,
      ω.2.2.2.2.2.1 + (T.fixAmt w c ω.1 ω.2.1 ω.2.2.1).2.2, ω.2.2.2.2.2.2) :=
  rfl

/-- Where `c`'s rows fail, the fix map is `shift`. -/
theorem fixE_of_not_rows (T : Table F K W Salt Seed D Hi VL) (w : W) (c : Fin 2 → Fin T.nE → F) (ω : T.Rand)
    (hr : ¬ T.onVals.RowsAt (w, T.constVals c) ω.1 ω.2.1) :
    T.fixE w c ω = T.onVals.shift (w, T.constVals c) (w, T.vals) ω := by
  have h0 : T.fixAmt w c ω.1 ω.2.1 ω.2.2.1 = 0 := by
    unfold fixAmt
    exact ite_eq_right hr
  rw [fixE_apply, h0]
  simp only [Prod.fst_zero, Prod.snd_zero, add_zero]
  rfl

/-- **The pre-final fix map** at constant values `c`: the free pads moved by `padShift` and `μ` as `fixE` moves it;
the mask slot, level 0's hiding, `pad_h`, `pad_μ` and every salt and seed kept. The pre-final view shows no `C_h`, so it
needs no move of `pad_h` and holds at every coin vector. -/
noncomputable def fixP (T : Table F K W Salt Seed D Hi VL) (w : W) (c : Fin 2 → Fin T.nE → F) : T.Rand ≃ T.Rand where
  toFun ω := (ω.1, ω.2.1, ω.2.2.1 + T.padShiftC w c ω.1 ω.2.1, ω.2.2.2.1, ω.2.2.2.2.1,
    ω.2.2.2.2.2.1 + (T.fixAmt w c ω.1 ω.2.1 ω.2.2.1).2.2, ω.2.2.2.2.2.2)
  invFun ω := (ω.1, ω.2.1, ω.2.2.1 - T.padShiftC w c ω.1 ω.2.1, ω.2.2.2.1, ω.2.2.2.2.1,
    ω.2.2.2.2.2.1 - (T.fixAmt w c ω.1 ω.2.1 (ω.2.2.1 - T.padShiftC w c ω.1 ω.2.1)).2.2, ω.2.2.2.2.2.2)
  left_inv ω := by
    obtain ⟨u, V, h, padh, padμ, μ, s⟩ := ω
    simp only [add_sub_cancel_right]
  right_inv ω := by
    obtain ⟨u, V, h, padh, padμ, μ, s⟩ := ω
    simp only [sub_add_cancel]

/-- Where `c`'s rows fail, the pre-final fix map is `shift`. -/
theorem fixP_of_not_rows (T : Table F K W Salt Seed D Hi VL) (w : W) (c : Fin 2 → Fin T.nE → F) (ω : T.Rand)
    (hr : ¬ T.onVals.RowsAt (w, T.constVals c) ω.1 ω.2.1) :
    T.fixP w c ω = T.onVals.shift (w, T.constVals c) (w, T.vals) ω := by
  have h0 : T.fixAmt w c ω.1 ω.2.1 ω.2.2.1 = 0 := by
    unfold fixAmt
    exact ite_eq_right hr
  show (ω.1, ω.2.1, ω.2.2.1 + T.padShiftC w c ω.1 ω.2.1, ω.2.2.2.1, ω.2.2.2.2.1,
    ω.2.2.2.2.2.1 + (T.fixAmt w c ω.1 ω.2.1 ω.2.2.1).2.2, ω.2.2.2.2.2.2) = _
  rw [h0]
  simp only [Prod.snd_zero, add_zero]
  rfl

omit [CharP F 2] in
/-- The `δ` of `fixAmt` solves `C_p·δ = C_h·d`. -/
theorem fixAmt_spec (T : Table F K W Salt Seed D Hi VL) (hpads : T.PadsOnto) (r : Fin 2) (d : PadIdx T.nE → F) :
    (fun k => ∑ j, T.Cp r k j *
        Function.invFun (fun p : Fin T.nPad → F => fun k => ∑ j, T.Cp r k j * p j) (fun k => ∑ i, T.Ch r k i * d i) j)
      = fun k => ∑ i, T.Ch r k i * d i :=
  Function.invFun_eq (hpads r _)

omit [CharP F 2] in
/-- `padShift` reads `p`'s clear values at its own mask slot and level 0's hiding alone. -/
theorem onVals_padShift_congr (T : Table F K W Salt Seed D Hi VL) (w : W) (v v' : T.Vals) (q : W × T.Vals)
    (u : T.Mask) (V : T.Vis) (h : ∀ r, v r w u V = v' r w u V) :
    T.onVals.padShift (w, v) q u V = T.onVals.padShift (w, v') q u V := by
  have h' : ∀ r, T.onVals.vals r (w, v) u V = T.onVals.vals r (w, v') u V := h
  funext r
  simp only [padShift, h']

/-- Whether the rows hold reads the clear values at the mask slot and level 0's hiding alone. -/
theorem onVals_rowsAt_congr (T : Table F K W Salt Seed D Hi VL) (w : W) (v v' : T.Vals) (u : T.Mask) (V : T.Vis)
    (h : ∀ r, v r w u V = v' r w u V) : T.onVals.RowsAt (w, v) u V ↔ T.onVals.RowsAt (w, v') u V := by
  unfold RowsAt
  refine forall_congr' fun ω => imp_congr_right fun hu => imp_congr_right fun hV => ?_
  subst hu hV
  rw [T.onVals_view_congr w v v' ω h]

omit [CharP F 2] in
theorem onVals_shift_congr (T : Table F K W Salt Seed D Hi VL) (w : W) (v v' : T.Vals) (q : W × T.Vals)
    (ω : T.Rand) (h : ∀ r, v r w ω.1 ω.2.1 = v' r w ω.1 ω.2.1) :
    T.onVals.shift (w, v) q ω = T.onVals.shift (w, v') q ω := by
  unfold shift
  rw [T.onVals_padShift_congr w v v' q ω.1 ω.2.1 h]

/-- **The switched pre-final view reads the clear values at its own mask slot and level 0's hiding alone.** -/
theorem swPre_congr (T : Table F K W Salt Seed D Hi VL) (w : W) (v v' : T.Vals) (ω : T.Rand)
    (h : ∀ r, v r w ω.1 ω.2.1 = v' r w ω.1 ω.2.1) : T.swPre w v ω = T.swPre w v' ω := by
  have hc := T.onVals_rowsAt_congr w v v' ω.1 ω.2.1 h
  unfold swPre
  by_cases hr : T.onVals.RowsAt (w, v) ω.1 ω.2.1
  · rw [ite_eq_left hr, ite_eq_left (hc.1 hr), T.onVals_preView_congr w v v' ω h]
  · rw [ite_eq_right hr, ite_eq_right (mt hc.2 hr), T.onVals_shift_congr w v v' _ ω h]

variable [AddCommGroup Salt] [AddCommGroup Seed] [AddCommGroup Hi]

set_option smartUnfolding false in
/-- The view's components at a table's witness paired with any clear values: the table's, with `v`'s values masked. -/
theorem onVals_realCore (T : Table F K W Salt Seed D Hi VL) (w : W) (v : T.Vals) (ω : T.Rand) :
    T.onVals.realCore (w, v) ω = ((T.nu w + T.maskHom ω.1,
      (fun q l => ∑ c, T.lanes w ω.1 l c * T.E q c, fun r => T.y1 w ω.1 r, 0) + T.mask1 ω.2.1,
      (fun r => (v r w ω.1 ω.2.1, 0, 0)) + T.mask2 ω.2.2.1,
      (fun r k => ∑ i, T.Ch r k i * T.hFull (ω.2.2.1 r) i, 0, fun r => T.hFull (ω.2.2.1 r), 0) +
        T.mask3 ω.2.2.2) : T.Core) :=
  rfl

/-- **Where `c`'s rows hold, the fix map keeps every component of the view** (Lemma B's sampling order): the masked
values and `ρ_in`, `σ_in` by `padShift`, `C_h` by `C_p·δ = C_h·d`, `Y₂` and `Y₁` by `−δ/β` and `d/β`. -/
theorem realCore_fix (T : Table F K W Salt Seed D Hi VL) (hpads : T.PadsOnto) (hβ : ∀ r, T.β r ≠ 0) (w : W)
    (c : Fin 2 → Fin T.nE → F) (ω : T.Rand) (hr : T.onVals.RowsAt (w, T.constVals c) ω.1 ω.2.1) :
    T.onVals.realCore (w, T.constVals c) ω = T.realCore w (T.fixE w c ω) := by
  obtain ⟨u, V, h, padh, padμ, μ, s⟩ := ω
  dsimp only at hr
  rw [onVals_realCore, fixE_apply]
  simp only [realCore, fixAmt, ite_eq_left hr]
  refine Prod.ext rfl (Prod.ext rfl (Prod.ext ?_ ?_))
  · funext r
    simp only [Pi.add_apply, mask2_apply, padShiftC_apply, constVals, Prod.mk_add_mk, Prod.fst_add, Prod.snd_add]
    refine Prod.ext ?_ (Prod.ext ?_ ?_)
    · funext i
      simp only [Pi.add_apply]
      ring
    · ring
    · ring
  · simp only [mask3_apply, Prod.mk_add_mk]
    refine Prod.ext ?_ (Prod.ext ?_ (Prod.ext ?_ rfl))
    · funext r k
      have hs := congrFun (T.fixAmt_spec hpads r fun i =>
        T.hFull (h r) i - T.hFull (h r + T.padShiftC w c u V r) i) k
      simp only [Pi.add_apply, mul_add, mul_sub, Finset.sum_add_distrib, Finset.sum_sub_distrib] at hs ⊢
      linear_combination -hs
    · funext r j
      have := hβ r
      simp only [Pi.add_apply]
      field_simp
      ring
    · funext r i
      have := hβ r
      simp only [Pi.add_apply]
      field_simp
      ring

omit [CharP F 2] [AddCommGroup Salt] [AddCommGroup Seed] [AddCommGroup Hi] in
set_option smartUnfolding false in
theorem toBase_psi (T : Table F K W Salt Seed D Hi VL) (κ : T.Core) : T.toBase (T.onVals.psi κ) = T.psi κ :=
  rfl

/-- **The switched view at constant values is the honest view at the moved randomness**: where `c`'s rows hold by
`realCore_fix` (both are `psi` of their components), elsewhere because the fix map is `shift`. -/
theorem swView_fix (T : Table F K W Salt Seed D Hi VL) (hW : ∀ r, Function.Bijective fun p : F × F => T.Wr r p.1 p.2)
    (hpads : T.PadsOnto) (hβ : ∀ r, T.β r ≠ 0) (w : W) (hreg : T.sReg w = T.pubReg) (hinner : T.InnerHolds w)
    (c : Fin 2 → Fin T.nE → F) (ω : T.Rand) :
    T.toBase (T.onVals.swView (w, T.constVals c) (w, T.vals) ω) = T.view w (T.fixE w c ω) := by
  by_cases hr : T.onVals.RowsAt (w, T.constVals c) ω.1 ω.2.1
  · simp only [swView, hr, ↓reduceIte]
    rw [← T.onVals.psi_realCore_at hW hβ (w, T.constVals c) hreg ω (hr ω rfl rfl), T.realCore_fix hpads hβ w c ω hr,
      toBase_psi, T.psi_realCore hW hβ w hreg hinner]
  · simp only [swView, hr, ↓reduceIte]
    rw [T.fixE_of_not_rows w c ω hr]
    exact T.toBase_view w _

omit [AddCommGroup Salt] [AddCommGroup Seed] [AddCommGroup Hi] in
open Classical in
set_option smartUnfolding false in
/-- **The switched pre-final view at constant values, cut, is the honest one at the pre-final fix map**, at every coin
vector: the masked values by `padShift`, `ρ_in`, `σ_in` kept by it, and `Y₁` by `μ`'s move wherever it is sent, which
is where `β ≠ 0` (`Admits`). -/
theorem cut_swPre_fixP (T : Table F K W Salt Seed D Hi VL) (S : T.Sent) (hS : T.Admits S) (w : W)
    (c : Fin 2 → Fin T.nE → F) (ω : T.Rand) :
    T.cut S (T.swPre w (T.constVals c) ω) = T.cut S (T.preView w (T.fixP w c ω)) := by
  by_cases hr : T.onVals.RowsAt (w, T.constVals c) ω.1 ω.2.1
  · rw [swPre, ite_eq_left hr]
    obtain ⟨u, V, h, padh, padμ, μ, s⟩ := ω
    dsimp only at hr
    have hA : T.fixAmt w c u V h = _ := ite_eq_left hr
    refine Cut.ext rfl rfl ?_ rfl rfl rfl ?_ ?_ rfl rfl rfl rfl
    · funext r i
      show c r i + (h r).1 i = T.vals r w u V i + (h r + T.padShiftC w c u V r).1 i
      rw [padShiftC_apply, Prod.fst_add, Pi.add_apply]
      ring
    · funext r
      show T.innerMsg r (h r) = T.innerMsg r (h r + T.padShiftC w c u V r)
      rw [padShiftC_apply]
      simp only [innerMsg, Prod.fst_add, Prod.snd_add, Pi.add_apply]
      refine Prod.ext ?_ ?_ <;> ring
    · funext r
      show (if S.fin r then some (fun i => T.hFull (h r) i + T.β r * μ r i) else none) =
        if S.fin r then some (fun i => T.hFull (h r + T.padShiftC w c u V r) i +
          T.β r * (μ r i + (T.fixAmt w c u V h).2.2 r i)) else none
      by_cases hf : S.fin r
      · have hβ := hS.2.2 r hf
        rw [ite_eq_left hf, ite_eq_left hf, hA]
        refine congrArg some (funext fun i => ?_)
        field_simp
        ring
      · rw [ite_eq_right hf, ite_eq_right hf]
  · rw [swPre, ite_eq_right hr, T.fixP_of_not_rows w c ω hr]

/-- **What a worker sees of a table's randomness** before its algebraic phase: the mask slot, level 0's hiding, and
level 0's salts at the opened positions and at the others. -/
abbrev Seen (T : Table F K W Salt Seed D Hi VL) : Type :=
  T.Mask × T.Vis × (Fin T.nS → Salt) × (Fin T.nU → Salt)

/-- What a worker sees of the randomness `ω` and level 0's hidden leaves' salts `ys`. -/
def seenOf (T : Table F K W Salt Seed D Hi VL) (ω : T.Rand) (ys : Fin T.nU → Salt) : T.Seen :=
  (ω.1, ω.2.1, ω.2.2.2.2.2.2.1, ys)

omit [AddCommGroup Salt] [AddCommGroup Seed] [AddCommGroup Hi] in
/-- The fix map keeps what the workers see. -/
theorem seenOf_fixE (T : Table F K W Salt Seed D Hi VL) (w : W) (c : Fin 2 → Fin T.nE → F) (ω : T.Rand)
    (ys : Fin T.nU → Salt) : T.seenOf (T.fixE w c ω) ys = T.seenOf ω ys :=
  rfl

section Hm96

variable {B C : Type} [AddCommGroup B]


/-- **The switched view with its hidden leaves given, at constant values**, is the honest one at the moved
randomness. -/
theorem swViewWith_fix (T : Table F K W Salt (B × Salt) D Hi VL)
    (hW : ∀ r, Function.Bijective fun p : F × F => T.Wr r p.1 p.2) (hpads : T.PadsOnto) (hβ : ∀ r, T.β r ≠ 0) (w : W)
    (hreg : T.sReg w = T.pubReg) (hinner : T.InnerHolds w) (c : Fin 2 → Fin T.nE → F) (ω : T.Rand) (L : T.Hid → D) :
    T.toBase (T.onVals.swViewWith (w, T.constVals c) (w, T.vals) ω L) = T.viewWith w (T.fixE w c ω) L := by
  have h := T.swView_fix hW hpads hβ w hreg hinner c ω
  by_cases hr : T.onVals.RowsAt (w, T.constVals c) ω.1 ω.2.1
  · simp only [swView, hr, ↓reduceIte] at h
    simp only [swViewWith, hr, ↓reduceIte, viewWith]
    rw [← h]
    rfl
  · simp only [swView, hr, ↓reduceIte] at h
    simp only [swViewWith, hr, ↓reduceIte, viewWith]
    rw [← h]
    rfl

omit [AddCommGroup Salt] [AddCommGroup Hi] in
/-- **The switched view's level-0 digests are the honest ones at the moved randomness**: they read the mask slot and
level 0's hiding, which the fix map and `shift` keep. -/
theorem swDig_inl_fix (T : Table F K W Salt (B × Salt) D Hi VL) (M : T.Hm96 C) (w : W) (c : Fin 2 → Fin T.nE → F)
    (ω : T.Rand) (q : Fin T.nU) :
    M.onVals.swDig (w, T.constVals c) (w, T.vals) ω (.inl q) = M.dig w (T.fixE w c ω) (.inl q) := by
  unfold Hm96.swDig
  split_ifs <;> rfl

omit [CharP F 2] [AddCommGroup Salt] [AddCommGroup Hi] in
/-- The hidden leaves' digests read no clear value. -/
theorem Hm96.onVals_dig_congr {T : Table F K W Salt (B × Salt) D Hi VL} (M : T.Hm96 C) (w : W) (v v' : T.Vals)
    (ω : T.Rand) : M.onVals.dig (w, v) ω = M.onVals.dig (w, v') ω := by
  funext i
  rcases i with q | ⟨r, k⟩ <;> rfl

omit [AddCommGroup B] [AddCommGroup Salt] [AddCommGroup Hi] in
theorem onVals_viewWith_congr (T : Table F K W Salt (B × Salt) D Hi VL) (w : W) (v v' : T.Vals) (ω : T.Rand)
    (L : T.Hid → D) (h : ∀ r, v r w ω.1 ω.2.1 = v' r w ω.1 ω.2.1) :
    T.onVals.viewWith (w, v) ω L = T.onVals.viewWith (w, v') ω L := by
  simp only [viewWith, T.onVals_view_congr w v v' ω h]

omit [AddCommGroup Salt] [AddCommGroup Hi] in
/-- **The switched view with real leaves reads the clear values at its own mask slot and level 0's hiding alone.** -/
theorem Hm96.swViewR_congr {T : Table F K W Salt (B × Salt) D Hi VL} (M : T.Hm96 C) (w : W) (v v' : T.Vals)
    (ω : T.Rand) (ys : T.Hid → Salt) (h : ∀ r, v r w ω.1 ω.2.1 = v' r w ω.1 ω.2.1) :
    M.onVals.swViewR (w, v) (w, T.vals) ω ys = M.onVals.swViewR (w, v') (w, T.vals) ω ys := by
  have hc := T.onVals_rowsAt_congr w v v' ω.1 ω.2.1 h
  rw [Hm96.swViewR_eq M.onVals (w, v) (w, T.vals) ω ys, Hm96.swViewR_eq M.onVals (w, v') (w, T.vals) ω ys]
  by_cases hr : T.onVals.RowsAt (w, v) ω.1 ω.2.1
  · rw [ite_eq_left hr, ite_eq_left (hc.1 hr)]
    unfold Hm96.viewR
    rw [M.onVals_dig_congr w v v' ω]
    exact T.onVals_viewWith_congr w v v' ω _ h
  · rw [ite_eq_right hr, ite_eq_right (mt hc.2 hr), T.onVals_shift_congr w v v' _ ω h]

/-- **The honest view with level 0's hidden leaves real and the pads trees' ideal**: real hm96 leaves of level 0's
columns under the salts `ys`, ideal leaves of the seeds `ωu.2` in the pads trees. -/
noncomputable def Hm96.viewMix {T : Table F K W Salt (B × Salt) D Hi VL} (M : T.Hm96 C) (w : W) (ωu : T.Rand × (Fin 2 × Fin T.nPU → B × Salt))
    (ys : Fin T.nU → Salt) : T.View :=
  T.viewWith w ωu.1 (Sum.elim (fun q => M.H (M.dig w ωu.1 (.inl q) + M.My (ys q), M.cs (ys q))) fun i => T.ideal (ωu.2 i))

/-- The hidden leaves' seeds split into level 0's and the pads trees'. -/
def hidSplit (T : Table F K W Salt (B × Salt) D Hi VL) (X : Type) :
    (T.Rand × (Fin 2 × Fin T.nPU → X)) × (Fin T.nU → X) ≃ T.Rand × (T.Hid → X) where
  toFun p := (p.1.1, Sum.elim p.2 p.1.2)
  invFun q := ((q.1, fun i => q.2 (.inr i)), fun q' => q.2 (.inl q'))
  left_inv _ := rfl
  right_inv q := Prod.ext rfl (funext fun i => by rcases i with q' | i <;> rfl)

end Hm96

section Mix

open Game

variable {B C : Type} [AddCommGroup B] [Fintype F] [DecidableEq F] [Fintype K] [DecidableEq K] [Fintype Salt]
  [DecidableEq Salt] [Fintype B] [DecidableEq B] [Fintype Hi] [DecidableEq Hi]

/-- **Lemma B with level 0's hidden leaves real and the pads trees' ideal**, for one table: every event has
probabilities within `(n_U + |Hid|)·δ₁` of each other under the honest prover with those leaves and under `S_shvzk`
with real leaves (`ideal_leaves_swap_fintype` on level 0's leaves, then `table_shvzk_hm96`'s steps). -/
theorem mix_shvzk_hm96 (T : Table F K W Salt (B × Salt) D Hi VL) (M : T.Hm96 C) {δ₁ : ENNReal}
    (hT1 : Claims.Hm96Hiding M.My M.cs δ₁) (hrank : T.Rank) (hpad : T.PadOnto)
    (hW : ∀ r, Function.Bijective fun p : F × F => T.Wr r p.1 p.2) (hpads : T.PadsOnto) (hβ : ∀ r, T.β r ≠ 0)
    (w : W) (hreg : T.sReg w = T.pubReg) (hinner : T.InnerHolds w) (w0 : W) (E : T.View → Prop) :
    prCoin (fun p : (T.Rand × (Fin 2 × Fin T.nPU → B × Salt)) × (Fin T.nU → Salt) => E (M.viewMix w p.1 p.2))
        ≤ prCoin (fun p : T.SimRand × (T.Hid → Salt) => E (M.simR w0 p.1 p.2))
          + ((T.nU : ENNReal) + Fintype.card T.Hid) * δ₁
      ∧ prCoin (fun p : T.SimRand × (T.Hid → Salt) => E (M.simR w0 p.1 p.2))
        ≤ prCoin (fun p : (T.Rand × (Fin 2 × Fin T.nPU → B × Salt)) × (Fin T.nU → Salt) => E (M.viewMix w p.1 p.2))
          + ((T.nU : ENNReal) + Fintype.card T.Hid) * δ₁ := by
  have : Nonempty Salt := ⟨0⟩
  have : Nonempty T.Rand := ⟨0⟩
  have : Nonempty T.SimRand := ⟨0⟩
  have hS := (T.table_shvzk hrank hpad hW hpads hβ w hreg hinner w0).prCoin_eq E
  have hv : prCoin (fun p : (T.Rand × (Fin 2 × Fin T.nPU → B × Salt)) × (Fin T.nU → B × Salt) =>
        E (T.viewWith w p.1.1 (Sum.elim (fun q => M.H ((p.2 q).1, M.cs (p.2 q).2)) fun i => T.ideal (p.1.2 i))))
      = prCoin (fun ω => E (T.view w ω)) := by
    rw [← T.prCoin_viewWith_ideal w E, ← prCoin_congr_equiv (T.hidSplit (B × Salt))]
    congr 1
    funext p
    refine congrArg (fun L => E (T.viewWith w p.1.1 L)) (funext fun i => ?_)
    rcases i with q | i
    · exact (M.ideal_eq (p.2 q).1 (p.2 q).2).symm
    · rfl
  have hs : prCoin (fun p : T.SimRand × (T.Hid → B × Salt) =>
        E (T.simWith w0 p.1 fun i => M.H ((p.2 i).1, M.cs (p.2 i).2)))
      = prCoin (fun σ => E (T.sim w0 σ)) := by
    rw [← T.prCoin_simWith_ideal w0 E]
    congr 1
    funext p
    congr 2
    funext i
    exact (M.ideal_eq (p.2 i).1 (p.2 i).2).symm
  have v1 := ideal_leaves_swap_fintype M.My M.cs hT1 M.H
    (fun (ωu : T.Rand × (Fin 2 × Fin T.nPU → B × Salt)) q => M.dig w ωu.1 (.inl q))
    (fun ωu L => E (T.viewWith w ωu.1 (Sum.elim L fun i => T.ideal (ωu.2 i))))
  have v2 := real_leaves_swap_fintype M.My M.cs hT1 M.H
    (fun (ωu : T.Rand × (Fin 2 × Fin T.nPU → B × Salt)) q => M.dig w ωu.1 (.inl q))
    (fun ωu L => E (T.viewWith w ωu.1 (Sum.elim L fun i => T.ideal (ωu.2 i))))
  have s1 := ideal_leaves_swap_fintype M.My M.cs hT1 M.H (M.digS w0) (fun σ L => E (T.simWith w0 σ L))
  have s2 := real_leaves_swap_fintype M.My M.cs hT1 M.H (M.digS w0) (fun σ L => E (T.simWith w0 σ L))
  beta_reduce at v1 v2 s1 s2
  rw [hv] at v1 v2
  rw [hs] at s1 s2
  simp only [Fintype.card_fin] at v1 v2
  constructor
  · calc prCoin (fun p : (T.Rand × (Fin 2 × Fin T.nPU → B × Salt)) × (Fin T.nU → Salt) => E (M.viewMix w p.1 p.2))
        ≤ prCoin (fun ω => E (T.view w ω)) + T.nU * δ₁ := v1
      _ = prCoin (fun σ => E (T.sim w0 σ)) + T.nU * δ₁ := by rw [hS]
      _ ≤ (prCoin (fun p : T.SimRand × (T.Hid → Salt) => E (M.simR w0 p.1 p.2)) + Fintype.card T.Hid * δ₁)
          + T.nU * δ₁ := add_le_add s2 le_rfl
      _ = _ := by ring
  · calc prCoin (fun p : T.SimRand × (T.Hid → Salt) => E (M.simR w0 p.1 p.2))
        ≤ prCoin (fun σ => E (T.sim w0 σ)) + Fintype.card T.Hid * δ₁ := s1
      _ = prCoin (fun ω => E (T.view w ω)) + Fintype.card T.Hid * δ₁ := by rw [hS]
      _ ≤ (prCoin (fun p : (T.Rand × (Fin 2 × Fin T.nPU → B × Salt)) × (Fin T.nU → Salt) =>
            E (M.viewMix w p.1 p.2)) + T.nU * δ₁) + Fintype.card T.Hid * δ₁ := add_le_add v2 le_rfl
      _ = _ := by ring

end Mix

end FlockSoundness.ZK.Table

namespace FlockSoundness.ZK

open Game

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

section Joint

variable {J : ℕ} {F K Salt B D C : Type} {W Hi VL : Fin J → Type} [Field F] [Field K] [CharP F 2]
  [AddCommGroup Salt] [AddCommGroup B] [∀ j, AddCommGroup (Hi j)]

/-- What the workers see of every table's randomness. -/
def famSeen (T : ∀ j, Table F K (W j) Salt (B × Salt) D (Hi j) (VL j)) (p : ∀ j, (T j).Rand × ((T j).Hid → Salt)) :
    ∀ j, (T j).Seen :=
  fun j => (T j).seenOf (p j).1 fun q => (p j).2 (.inl q)

/-- **The tables' switched views at values read from what every table's workers see** (`vc`), with real leaves. -/
noncomputable def famSw (T : ∀ j, Table F K (W j) Salt (B × Salt) D (Hi j) (VL j)) (M : ∀ j, (T j).Hm96 C)
    (w : ∀ j, W j) (vc : (∀ j, (T j).Seen) → ∀ j, Fin 2 → Fin (T j).nE → F)
    (p : ∀ j, (T j).Rand × ((T j).Hid → Salt)) : ∀ j, (T j).View :=
  fun j => (T j).toBase ((M j).onVals.swViewR (w j, (T j).constVals (vc (famSeen T p) j)) (w j, (T j).vals) (p j).1
    (p j).2)

/-- The randomness with the pads trees' hidden leaves ideal: each table's randomness and pads seeds, and level 0's
hidden salts. -/
abbrev MixRand (T : ∀ j, Table F K (W j) Salt (B × Salt) D (Hi j) (VL j)) : Type :=
  ∀ j, ((T j).Rand × (Fin 2 × Fin (T j).nPU → B × Salt)) × (Fin (T j).nU → Salt)

/-- What the workers see, on that randomness. -/
def mixSeen (T : ∀ j, Table F K (W j) Salt (B × Salt) D (Hi j) (VL j)) (q : MixRand T) : ∀ j, (T j).Seen :=
  fun j => (T j).seenOf (q j).1.1 (q j).2

/-- **The switched views with the pads trees' hidden leaves ideal.** -/
noncomputable def famSwMix (T : ∀ j, Table F K (W j) Salt (B × Salt) D (Hi j) (VL j)) (M : ∀ j, (T j).Hm96 C)
    (w : ∀ j, W j) (vc : (∀ j, (T j).Seen) → ∀ j, Fin 2 → Fin (T j).nE → F) (q : MixRand T) : ∀ j, (T j).View :=
  fun j => (T j).toBase ((T j).onVals.swViewWith (w j, (T j).constVals (vc (mixSeen T q) j)) (w j, (T j).vals)
    (q j).1.1 (Sum.elim (fun q' => (M j).H ((M j).onVals.swDig (w j, (T j).constVals (vc (mixSeen T q) j))
      (w j, (T j).vals) (q j).1.1 (.inl q') + (M j).My ((q j).2 q'), (M j).cs ((q j).2 q')))
      fun i => (T j).ideal ((q j).1.2 i)))

/-- **The fix map at every table**, at the values read from what the workers see, which it keeps. -/
noncomputable def famFix (T : ∀ j, Table F K (W j) Salt (B × Salt) D (Hi j) (VL j)) (w : ∀ j, W j)
    (vc : (∀ j, (T j).Seen) → ∀ j, Fin 2 → Fin (T j).nE → F) : MixRand T ≃ MixRand T where
  toFun q j := (((T j).fixE (w j) (vc (mixSeen T q) j) (q j).1.1, (q j).1.2), (q j).2)
  invFun q j := ((((T j).fixE (w j) (vc (mixSeen T q) j)).symm (q j).1.1, (q j).1.2), (q j).2)
  left_inv q := funext fun j => Prod.ext (Prod.ext (((T j).fixE (w j) (vc (mixSeen T q) j)).symm_apply_apply _) rfl) rfl
  right_inv q := funext fun j => Prod.ext (Prod.ext (((T j).fixE (w j) (vc (mixSeen T q) j)).apply_symm_apply _) rfl) rfl

/-- **With the pads trees' leaves ideal, the switched views are the honest prover's on the fixed randomness**
(`swViewWith_fix`, `swDig_inl_fix`). -/
theorem famSwMix_fix (T : ∀ j, Table F K (W j) Salt (B × Salt) D (Hi j) (VL j)) (M : ∀ j, (T j).Hm96 C)
    (hW : ∀ j r, Function.Bijective fun p : F × F => (T j).Wr r p.1 p.2) (hpads : ∀ j, (T j).PadsOnto)
    (hβ : ∀ j r, (T j).β r ≠ 0) (w : ∀ j, W j) (hreg : ∀ j, (T j).sReg (w j) = (T j).pubReg)
    (hinner : ∀ j, (T j).InnerHolds (w j)) (vc : (∀ j, (T j).Seen) → ∀ j, Fin 2 → Fin (T j).nE → F) (q : MixRand T) :
    famSwMix T M w vc q = fun j => (M j).viewMix (w j) (famFix T w vc q j).1 (famFix T w vc q j).2 := by
  funext j
  rw [famSwMix]
  refine Eq.trans ((T j).swViewWith_fix (hW j) (hpads j) (hβ j) (w j) (hreg j) (hinner j) (vc (mixSeen T q) j)
    (q j).1.1 _) ?_
  refine congrArg ((T j).viewWith (w j) _) (funext fun i => ?_)
  rcases i with q' | i
  · exact congrArg (fun d => (M j).H (d + (M j).My ((q j).2 q'), (M j).cs ((q j).2 q')))
      ((T j).swDig_inl_fix (M j) (w j) (vc (mixSeen T q) j) (q j).1.1 q')
  · rfl

/-- The joint randomness split into what stays (each table's randomness and level 0's hidden salts) and the pads trees'
hidden salts, indexed by table. -/
def padsSplit (T : ∀ j, Table F K (W j) Salt (B × Salt) D (Hi j) (VL j)) (X : Type) :
    (∀ j, (T j).Rand × ((T j).Hid → X)) ≃
      (∀ j, (T j).Rand × (Fin (T j).nU → X)) × ((Σ j, Fin 2 × Fin (T j).nPU) → X) where
  toFun p := (fun j => ((p j).1, fun q => (p j).2 (.inl q)), fun s => (p s.1).2 (.inr s.2))
  invFun a j := ((a.1 j).1, Sum.elim (a.1 j).2 fun i => a.2 ⟨j, i⟩)
  left_inv p := funext fun j => Prod.ext rfl (funext fun i => by rcases i with q | i <;> rfl)
  right_inv _ := rfl

/-- The pads trees' seeds regrouped by table. -/
def padsJoin (T : ∀ j, Table F K (W j) Salt (B × Salt) D (Hi j) (VL j)) :
    (∀ j, (T j).Rand × (Fin (T j).nU → Salt)) × ((Σ j, Fin 2 × Fin (T j).nPU) → B × Salt) ≃ MixRand T where
  toFun b j := (((b.1 j).1, fun i => b.2 ⟨j, i⟩), (b.1 j).2)
  invFun q := (fun j => ((q j).1.1, (q j).2), fun s => (q s.1).1.2 s.2)
  left_inv _ := rfl
  right_inv _ := rfl

omit [CharP F 2] [AddCommGroup Salt] [∀ j, AddCommGroup (Hi j)] in
theorem leaf_common {j : Fin J} {T : Table F K (W j) Salt (B × Salt) D (Hi j) (VL j)} (M : T.Hm96 C) (My : Salt → B)
    (cs : Salt → C) (Hh : B × C → D) (hMy : M.My = My) (hcs : M.cs = cs) (hH : M.H = Hh) (d : B) (y : Salt) :
    M.H (d + M.My y, M.cs y) = Hh (d + My y, cs y) := by
  rw [hMy, hcs, hH]

omit [CharP F 2] [AddCommGroup Salt] [∀ j, AddCommGroup (Hi j)] in
theorem ideal_common {j : Fin J} {T : Table F K (W j) Salt (B × Salt) D (Hi j) (VL j)} (M : T.Hm96 C) (cs : Salt → C)
    (Hh : B × C → D) (hcs : M.cs = cs) (hH : M.H = Hh) (u : B × Salt) : Hh (u.1, cs u.2) = T.ideal u := by
  rw [← hcs, ← hH]
  exact (M.ideal_eq u.1 u.2).symm

variable [Fintype F] [DecidableEq F] [Fintype K] [DecidableEq K] [Fintype Salt] [DecidableEq Salt] [Fintype B]
  [DecidableEq B] [∀ j, Fintype (Hi j)] [∀ j, DecidableEq (Hi j)]

omit [DecidableEq F] [Fintype K] [DecidableEq K] [DecidableEq Salt] [DecidableEq B] [∀ j, DecidableEq (Hi j)] in
/-- **The pads trees' hidden leaves go ideal at every table at once**, at `δ₁` per leaf (`ideal_leaves_swap_fintype` on
all of them): the values read level 0's salts, never the pads trees'. -/
theorem famSw_pads_swap (T : ∀ j, Table F K (W j) Salt (B × Salt) D (Hi j) (VL j)) (M : ∀ j, (T j).Hm96 C)
    (My : Salt → B) (cs : Salt → C) (Hh : B × C → D) (hMy : ∀ j, (M j).My = My) (hcs : ∀ j, (M j).cs = cs)
    (hH : ∀ j, (M j).H = Hh) {δ₁ : ENNReal} (hT1 : Claims.Hm96Hiding My cs δ₁) (w : ∀ j, W j)
    (vc : (∀ j, (T j).Seen) → ∀ j, Fin 2 → Fin (T j).nE → F) (E : (∀ j, (T j).View) → Prop) :
    prCoin (fun p : ∀ j, (T j).Rand × ((T j).Hid → Salt) => E (famSw T M w vc p))
        ≤ prCoin (fun q : MixRand T => E (famSwMix T M w vc q)) + ((∑ j, 2 * (T j).nPU : ℕ) : ENNReal) * δ₁
      ∧ prCoin (fun q : MixRand T => E (famSwMix T M w vc q))
        ≤ prCoin (fun p : ∀ j, (T j).Rand × ((T j).Hid → Salt) => E (famSw T M w vc p))
          + ((∑ j, 2 * (T j).nPU : ℕ) : ENNReal) * δ₁ := by
  have : Nonempty Salt := ⟨0⟩
  have : Nonempty (∀ j, (T j).Rand × (Fin (T j).nU → Salt)) := ⟨fun _ => 0⟩
  let cva : (∀ j, (T j).Rand × (Fin (T j).nU → Salt)) → ∀ j, Fin 2 → Fin (T j).nE → F := fun a =>
    vc fun k => (T k).seenOf (a k).1 (a k).2
  let x : (∀ j, (T j).Rand × (Fin (T j).nU → Salt)) → (Σ j, Fin 2 × Fin (T j).nPU) → B := fun a s =>
    (M s.1).onVals.swDig (w s.1, (T s.1).constVals (cva a s.1)) (w s.1, (T s.1).vals) (a s.1).1 (.inr s.2)
  let G : (∀ j, (T j).Rand × (Fin (T j).nU → Salt)) → ((Σ j, Fin 2 × Fin (T j).nPU) → D) → Prop := fun a L =>
    E fun j => (T j).toBase ((T j).onVals.swViewWith (w j, (T j).constVals (cva a j)) (w j, (T j).vals) (a j).1
      (Sum.elim (fun q' => (M j).H ((M j).onVals.swDig (w j, (T j).constVals (cva a j)) (w j, (T j).vals) (a j).1
        (.inl q') + (M j).My ((a j).2 q'), (M j).cs ((a j).2 q'))) fun i => L ⟨j, i⟩))
  have v1 := ideal_leaves_swap_fintype My cs hT1 Hh x G
  have v2 := real_leaves_swap_fintype My cs hT1 Hh x G
  have hl : prCoin (fun p : ∀ j, (T j).Rand × ((T j).Hid → Salt) => E (famSw T M w vc p))
      = prCoin (fun b : (∀ j, (T j).Rand × (Fin (T j).nU → Salt)) × ((Σ j, Fin 2 × Fin (T j).nPU) → Salt) =>
          G b.1 fun s => Hh (x b.1 s + My (b.2 s), cs (b.2 s))) := by
    rw [← prCoin_congr_equiv (padsSplit T Salt)]
    congr 1
    funext p
    refine congrArg E (funext fun j => ?_)
    refine congrArg (fun L => (T j).toBase ((T j).onVals.swViewWith _ _ _ L)) (funext fun i => ?_)
    rcases i with q' | i
    · rfl
    · exact leaf_common (M j) My cs Hh (hMy j) (hcs j) (hH j) _ _
  have hr : prCoin (fun b : (∀ j, (T j).Rand × (Fin (T j).nU → Salt)) × ((Σ j, Fin 2 × Fin (T j).nPU) → B × Salt) =>
          G b.1 fun s => Hh ((b.2 s).1, cs (b.2 s).2))
      = prCoin (fun q : MixRand T => E (famSwMix T M w vc q)) := by
    rw [← prCoin_congr_equiv (padsJoin T)]
    congr 1
    funext b
    refine congrArg E (funext fun j => ?_)
    refine congrArg (fun L => (T j).toBase ((T j).onVals.swViewWith _ _ _ L)) (funext fun i => ?_)
    rcases i with q' | i
    · rfl
    · exact ideal_common (M j) cs Hh (hcs j) (hH j) _
  have hc : Fintype.card (Σ j, Fin 2 × Fin (T j).nPU) = ∑ j, 2 * (T j).nPU := by
    simp only [Fintype.card_sigma, Fintype.card_prod, Fintype.card_fin]
  rw [hl, ← hr, ← hc]
  exact ⟨v1, v2⟩

/-- **Lemma B at values read from what every table's workers see, with real leaves**: at `J` tables, at a fixed
non-degenerate coin vector of each, for witnesses `w` carrying the public region data whose honest pads satisfy the inner
constraint, and for any values `vc` read from what every table's workers see (`Table.Seen`: the mask slots, level 0's
hiding and level 0's salts), the tables' switched views and their `S_shvzk` with real leaves on any dummies are within
`Σ_j 2·|Hid_j|·δ₁` both ways. The pads trees' hidden leaves go ideal first (`famSw_pads_swap`); then the fix map, a
bijection that keeps what the workers see, makes the switched views the honest prover's (`famSwMix_fix`), whose tables
are independent, each within `(n_U + |Hid|)·δ₁` of its `S_shvzk` (`mix_shvzk_hm96`, `prCoin_pi_close`). -/
theorem fam_sw_shvzk_hm96 (T : ∀ j, Table F K (W j) Salt (B × Salt) D (Hi j) (VL j)) (M : ∀ j, (T j).Hm96 C)
    (My : Salt → B) (cs : Salt → C) (Hh : B × C → D) (hMy : ∀ j, (M j).My = My) (hcs : ∀ j, (M j).cs = cs)
    (hH : ∀ j, (M j).H = Hh) {δ₁ : ENNReal} (hT1 : Claims.Hm96Hiding My cs δ₁) (hrank : ∀ j, (T j).Rank)
    (hpad : ∀ j, (T j).PadOnto) (hW : ∀ j r, Function.Bijective fun p : F × F => (T j).Wr r p.1 p.2)
    (hpads : ∀ j, (T j).PadsOnto) (hβ : ∀ j r, (T j).β r ≠ 0) (w : ∀ j, W j)
    (hreg : ∀ j, (T j).sReg (w j) = (T j).pubReg) (hinner : ∀ j, (T j).InnerHolds (w j))
    (vc : (∀ j, (T j).Seen) → ∀ j, Fin 2 → Fin (T j).nE → F) (w0 : ∀ j, W j) (E : (∀ j, (T j).View) → Prop) :
    prCoin (fun p : ∀ j, (T j).Rand × ((T j).Hid → Salt) => E (famSw T M w vc p))
        ≤ prCoin (fun p : ∀ j, (T j).SimRand × ((T j).Hid → Salt) => E fun j => (M j).simR (w0 j) (p j).1 (p j).2)
          + ∑ j, 2 * Fintype.card (T j).Hid * δ₁
      ∧ prCoin (fun p : ∀ j, (T j).SimRand × ((T j).Hid → Salt) => E fun j => (M j).simR (w0 j) (p j).1 (p j).2)
        ≤ prCoin (fun p : ∀ j, (T j).Rand × ((T j).Hid → Salt) => E (famSw T M w vc p))
          + ∑ j, 2 * Fintype.card (T j).Hid * δ₁ := by
  have : Nonempty Salt := ⟨0⟩
  have : ∀ j, Nonempty (((T j).Rand × (Fin 2 × Fin (T j).nPU → B × Salt)) × (Fin (T j).nU → Salt)) :=
    fun _ => ⟨0⟩
  have : ∀ j, Nonempty ((T j).SimRand × ((T j).Hid → Salt)) := fun _ => ⟨0⟩
  obtain ⟨a1, a2⟩ := famSw_pads_swap T M My cs Hh hMy hcs hH hT1 w vc E
  have hb : prCoin (fun q : MixRand T => E (famSwMix T M w vc q))
      = prCoin (fun q : MixRand T => E fun j => (M j).viewMix (w j) (q j).1 (q j).2) := by
    rw [← prCoin_congr_equiv (famFix T w vc) fun q => E fun j => (M j).viewMix (w j) (q j).1 (q j).2]
    congr 1
    funext q
    rw [famSwMix_fix T M hW hpads hβ w hreg hinner vc q]
  have one : ∀ j (E' : (T j).View → Prop),
      prCoin (fun p : ((T j).Rand × (Fin 2 × Fin (T j).nPU → B × Salt)) × (Fin (T j).nU → Salt) =>
          E' ((M j).viewMix (w j) p.1 p.2))
        ≤ prCoin (fun p : (T j).SimRand × ((T j).Hid → Salt) => E' ((M j).simR (w0 j) p.1 p.2))
          + (((T j).nU : ENNReal) + Fintype.card (T j).Hid) * δ₁
      ∧ prCoin (fun p : (T j).SimRand × ((T j).Hid → Salt) => E' ((M j).simR (w0 j) p.1 p.2))
        ≤ prCoin (fun p : ((T j).Rand × (Fin 2 × Fin (T j).nPU → B × Salt)) × (Fin (T j).nU → Salt) =>
            E' ((M j).viewMix (w j) p.1 p.2)) + (((T j).nU : ENNReal) + Fintype.card (T j).Hid) * δ₁ := by
    intro j E'
    have hT1j : Claims.Hm96Hiding (M j).My (M j).cs δ₁ := by
      rw [hMy j, hcs j]
      exact hT1
    exact (T j).mix_shvzk_hm96 (M j) hT1j (hrank j) (hpad j) (hW j) (hpads j) (hβ j) (w j) (hreg j) (hinner j)
      (w0 j) E'
  have p1 := prCoin_pi_close
    (Ω := fun j => ((T j).Rand × (Fin 2 × Fin (T j).nPU → B × Salt)) × (Fin (T j).nU → Salt))
    (Ω' := fun j => (T j).SimRand × ((T j).Hid → Salt)) (V := fun j => (T j).View)
    (fun j p => (M j).viewMix (w j) p.1 p.2) (fun j p => (M j).simR (w0 j) p.1 p.2)
    (fun j => (((T j).nU : ENNReal) + Fintype.card (T j).Hid) * δ₁) (fun j E' => (one j E').1) E
  have p2 := prCoin_pi_close
    (Ω := fun j => (T j).SimRand × ((T j).Hid → Salt))
    (Ω' := fun j => ((T j).Rand × (Fin 2 × Fin (T j).nPU → B × Salt)) × (Fin (T j).nU → Salt))
    (V := fun j => (T j).View)
    (fun j p => (M j).simR (w0 j) p.1 p.2) (fun j p => (M j).viewMix (w j) p.1 p.2)
    (fun j => (((T j).nU : ENNReal) + Fintype.card (T j).Hid) * δ₁) (fun j E' => (one j E').2) E
  have hsum : ((∑ j, 2 * (T j).nPU : ℕ) : ENNReal) * δ₁ + ∑ j, (((T j).nU : ENNReal) + Fintype.card (T j).Hid) * δ₁
      = ∑ j, 2 * Fintype.card (T j).Hid * δ₁ := by
    rw [Nat.cast_sum, Finset.sum_mul, ← Finset.sum_add_distrib]
    refine Finset.sum_congr rfl fun j _ => ?_
    simp only [Table.Hid, Fintype.card_sum, Fintype.card_prod, Fintype.card_fin]
    push_cast
    ring
  rw [hb] at a1 a2
  constructor
  · calc _ ≤ _ := a1
      _ ≤ _ := add_le_add p1 le_rfl
      _ = _ := by rw [add_assoc, add_comm (∑ j, _), hsum]
  · calc _ ≤ _ := p2
      _ ≤ _ := add_le_add a2 le_rfl
      _ = _ := by rw [add_assoc, hsum]

end Joint

end FlockSoundness.ZK
