import Proofs.Flock.Soundness.Refine.Final

/-!
# The batched basis refines the model's (PROTOCOL.md §12.4)

`basis_eq`: the executable's `basis sws q e` at `y < 2^e` is the model's `bHat` of the batched basis `b` (R4's
`OpenRel.b`) at `q ‖ bits(y)`.

The executable evaluates each claim's term with a width-128 branching program over bit vectors. `applyMap (mulColumns g)`
is multiplication by `g` on a vector of bits (`applyMap_mul`). After the prefix coordinates, and then the suffix
coordinates read as `0`/`1` points, the state is `Σ_w eq(point, w)·bits(eq(z, w))` (`bfcState`). The final
contraction with `eq(r'')` is the model's `phi` (§12.4).
-/

namespace FlockSoundness.Refine

open Flock FlockLevel3 Game Transcript Model Accounting

local notation "𝔽" => FlockLevel3.GF128
local notation "𝕂" => FlockLevel3.GF256

/-! ## Monomials and the multiplication columns -/

theorem w128_monomial (u : ℕ) (hu : u < 128) :
    w128 (F128.monomial u).lo (F128.monomial u).hi = 1 <<< u := by
  unfold F128.monomial
  split_ifs with h
  · simp only [w128, UInt64.toNat_zero, mul_zero, zero_add]
    rw [UInt64.toNat_shiftLeft, toNat_toUInt64 u h, Nat.mod_eq_of_lt h, Nat.shiftLeft_eq, Nat.shiftLeft_eq]
    simp only [UInt64.toNat_one, one_mul]
    exact Nat.mod_eq_of_lt (Nat.pow_lt_pow_right (by norm_num) h)
  · simp only [w128, UInt64.toNat_zero, add_zero]
    rw [UInt64.toNat_shiftLeft, toNat_toUInt64 (u - 64) (by omega), Nat.mod_eq_of_lt (by omega : u - 64 < 64),
      Nat.shiftLeft_eq, Nat.shiftLeft_eq]
    simp only [UInt64.toNat_one, one_mul]
    rw [Nat.mod_eq_of_lt (Nat.pow_lt_pow_right (by norm_num) (by omega)), ← pow_add]
    congr 1
    omega

theorem toAdj_monomial (u : ℕ) (hu : u < 128) : toAdj (F128.monomial u) = AdjoinRoot.root G ^ u := by
  rw [toAdj, F128.poly, w128_monomial u hu, natPoly_shiftLeft, natPoly_one, mul_one, map_pow, AdjoinRoot.mk_X]

/-- `x^u` is the executable's `monomial u` below the field's degree. -/
theorem fOf_monomial (u : ℕ) (hu : u < 128) : fOf (F128.monomial 1) ^ u = fOf (F128.monomial u) := by
  induction u with
  | zero =>
      rw [pow_zero]
      exact GF128.one_eq.trans (congrArg GF128.of (toAdj_injective (by
        rw [toAdj_one, toAdj_monomial 0 (by norm_num), pow_zero])))
  | succ u ih =>
      rw [pow_succ, ih (by omega)]
      show GF128.of (F128.monomial u) * GF128.of (F128.monomial 1) = _
      rw [GF128.mul_eq]
      exact congrArg GF128.of (toAdj_injective (by
        rw [toAdj_mul, toAdj_monomial u (by omega), toAdj_monomial 1 (by norm_num), toAdj_monomial (u + 1) hu,
          pow_one, pow_succ]))

theorem id_forIn_yield {α σ : Type} (f : α → σ → Id (ForInStep σ)) (g : σ → α → σ)
    (hf : ∀ a s, f a s = .yield (g s a)) : ∀ (l : List α) (s : σ), (forIn l s f : Id σ) = l.foldl g s
  | [], _ => rfl
  | a :: l, s => by
      rw [List.forIn_cons, hf]
      exact id_forIn_yield f g hf l (g s a)

theorem mulColumns_fold (x : F128) : ∀ (k : ℕ) (a : ℕ) (out : Array F128) (v : F128),
    (((List.range' a k).foldl (fun r _ => (r.1.push r.2, r.2 * x)) (out, v)).1.size = out.size + k) ∧
    (∀ i < out.size, ((List.range' a k).foldl (fun r _ => (r.1.push r.2, r.2 * x)) (out, v)).1[i]! = out[i]!) ∧
    ∀ i < k, fOf ((List.range' a k).foldl (fun r _ => (r.1.push r.2, r.2 * x)) (out, v)).1[out.size + i]! =
      fOf v * fOf x ^ i
  | 0, _, out, v => ⟨rfl, fun _ _ => rfl, fun _ hi => absurd hi (Nat.not_lt_zero _)⟩
  | k + 1, a, out, v => by
      rw [List.range'_succ, List.foldl_cons]
      obtain ⟨h1, h2, h3⟩ := mulColumns_fold x k (a + 1) (out.push v) (v * x)
      refine ⟨by rw [h1, Array.size_push]; omega, fun i hi => ?_, fun i hi => ?_⟩
      · rw [h2 i (by simp; omega), getElem!_pos _ i (by simp; omega), getElem!_pos out i hi, Array.getElem_push_lt]
      · rcases i with _ | i
        · simp only [Nat.add_zero, pow_zero, mul_one]
          rw [h2 out.size (by simp), getElem!_pos _ out.size (by simp), Array.getElem_push_eq]
        · have := h3 i (by omega)
          rw [Array.size_push, show out.size + 1 + i = out.size + (i + 1) by omega, fOf_mul, mul_assoc,
            ← pow_succ'] at this
          exact this

/-- The executable's columns of "multiply by `g`" are `g·x^u`. -/
theorem mulColumns_get (g : F128) (u : ℕ) (hu : u < 128) :
    fOf (mulColumns g)[u]! = fOf g * fOf (F128.monomial u) := by
  unfold mulColumns
  simp only [Id.run, Std.Legacy.Range.forIn_eq_forIn_range', range_size, List.forIn_pure_yield_eq_foldl, pure_bind]
  have := (mulColumns_fold (F128.monomial 1) 128 0 (Array.mkEmpty 128) g).2.2 u hu
  rw [show (Array.mkEmpty 128 : Array F128).size = 0 from rfl, Nat.zero_add] at this
  rw [← fOf_monomial u hu]
  exact this

/-! ## `applyMap`: a matrix over the columns' bits -/

/-- One column's contribution: add `su` at every bit `l` of the column `c`. -/
def colAdd (c : F128) (su : F256) (out : Array F256) : Array F256 :=
  (List.range' 0 128).foldl (fun out l => if c.bit l then out.set! l (out[l]! + su) else out) out

theorem colAdd_get : ∀ (L : List ℕ) (c : F128) (su : F256) (out : Array F256) (l : ℕ), l < out.size → L.Nodup →
    (L.foldl (fun out l => if c.bit l then out.set! l (out[l]! + su) else out) out).size = out.size ∧
    gOf (L.foldl (fun out l => if c.bit l then out.set! l (out[l]! + su) else out) out)[l]! =
      gOf out[l]! + if l ∈ L ∧ c.bit l = true then gOf su else 0
  | [], c, su, out, l, _, _ => by simp
  | l' :: L, c, su, out, l, hl, hnd => by
      rw [List.foldl_cons]
      have hnd' := (List.nodup_cons.1 hnd)
      have hsz : (if c.bit l' then out.set! l' (out[l']! + su) else out).size = out.size := by
        split_ifs <;> simp
      obtain ⟨h1, h2⟩ := colAdd_get L c su _ l (by rw [hsz]; exact hl) hnd'.2
      refine ⟨h1.trans hsz, ?_⟩
      rw [h2]
      by_cases hll : l' = l
      · subst hll
        have hnot : l' ∉ L := hnd'.1
        by_cases hb : c.bit l' = true
        · simp only [hb, ite_true, Array.set!_eq_setIfInBounds]
          rw [getElem!_pos _ l' (by simp; omega), Array.getElem_setIfInBounds hl, getElem!_pos out l' hl]
          simp [hnot, gOf_add]
        · simp [hb, hnot]
      · have hget : (if c.bit l' then out.set! l' (out[l']! + su) else out)[l]! = out[l]! := by
          split_ifs
          · rw [Array.set!_eq_setIfInBounds, getElem!_pos _ l (by simp; omega), Array.getElem_setIfInBounds hl,
              getElem!_pos out l hl]
            simp [hll]
          · rfl
        rw [hget]
        simp [hll, Ne.symm hll]

theorem isZero_gOf {a : F256} (h : (a.c0.isZero && a.c1.isZero) = true) : gOf a = 0 := by
  simp only [F128.isZero, Bool.and_eq_true, beq_iff_eq] at h
  obtain ⟨⟨h1, h2⟩, h3, h4⟩ := h
  have : a = 0 := by
    cases a with | mk c0 c1 => cases c0; cases c1; simp_all; rfl
  rw [this, gOf_zero]

/-- **`applyMap` is a matrix** (§12.4): entry `l` of `applyMap cols s` is the sum of the `s[u]` whose column has bit
`l`. -/
theorem applyMap_get (cols : Array F128) (s : Array F256) (l : ℕ) (hl : l < 128) :
    gOf (applyMap cols s)[l]! = ∑ u ∈ Finset.range 128, if cols[u]!.bit l then gOf s[u]! else 0 := by
  unfold applyMap
  simp only [Id.run, Std.Legacy.Range.forIn_eq_forIn_range', range_size]
  have hinner : ∀ (c : F128) (su : F256) (out : Array F256),
      (forIn (List.range' 0 128) out (fun l out => if c.bit l = true then pure (ForInStep.yield (out.set! l (out[l]! + su)))
        else pure (ForInStep.yield out)) : Id (Array F256)) = colAdd c su out := by
    intro c su out
    exact id_forIn_yield _ _ (fun l out => by split_ifs <;> rfl) _ _
  rw [id_forIn_yield _ (fun out u => if (s[u]!.c0.isZero && s[u]!.c1.isZero) = true then out
      else colAdd cols[u]! s[u]! out) (fun u out => by
        split_ifs with h
        · rfl
        · show (forIn (List.range' 0 128) out _ >>= fun x => pure (ForInStep.yield x) : Id _) = _
          rw [hinner]
          rfl)]
  have hgen : ∀ (L : List ℕ) (out : Array F256), out.size = 128 → L.Nodup →
      (L.foldl (fun out u => if (s[u]!.c0.isZero && s[u]!.c1.isZero) = true then out
        else colAdd cols[u]! s[u]! out) out).size = 128 ∧
      gOf (L.foldl (fun out u => if (s[u]!.c0.isZero && s[u]!.c1.isZero) = true then out
        else colAdd cols[u]! s[u]! out) out)[l]! =
        gOf out[l]! + (L.map fun u => if cols[u]!.bit l then gOf s[u]! else 0).sum := by
    intro L
    induction L with
    | nil => intro out h _; simp [h]
    | cons u L ih =>
        intro out hsz hnd
        rw [List.foldl_cons]
        have hstep : ((if (s[u]!.c0.isZero && s[u]!.c1.isZero) = true then out
            else colAdd cols[u]! s[u]! out).size = 128) ∧
            gOf (if (s[u]!.c0.isZero && s[u]!.c1.isZero) = true then out else colAdd cols[u]! s[u]! out)[l]! =
              gOf out[l]! + if cols[u]!.bit l then gOf s[u]! else 0 := by
          by_cases hz : (s[u]!.c0.isZero && s[u]!.c1.isZero) = true
          · rw [ite_eq_left hz, isZero_gOf hz]
            exact ⟨hsz, by split_ifs <;> simp⟩
          · rw [ite_eq_right hz]
            obtain ⟨h1, h2⟩ := colAdd_get (List.range' 0 128) cols[u]! s[u]! out l (by omega) List.nodup_range'
            refine ⟨h1.trans hsz, ?_⟩
            unfold colAdd
            rw [h2]
            simp [List.mem_range', hl]
        obtain ⟨ih1, ih2⟩ := ih _ hstep.1 (List.nodup_cons.1 hnd).2
        refine ⟨ih1, ?_⟩
        rw [ih2, hstep.2, List.map_cons, List.sum_cons, add_assoc]
  obtain ⟨-, h⟩ := hgen (List.range' 0 128) (Array.replicate 128 0) (by simp) List.nodup_range'
  rw [bind_pure, h, getElem!_pos _ l (by simp; omega), Array.getElem_replicate, gOf_zero, zero_add, ← List.range_eq_range',
    FlockLevel3.list_sum_range]

/-! ## Bits in `K`, and multiplication on bit vectors -/

/-- Bit `u` of a field element, as `0` or `1` in `K`. -/
noncomputable def bK (a : 𝔽) (u : ℕ) : 𝕂 :=
  if h : u < 128 then ZMod.castHom (dvd_refl 2) 𝕂 (unpack a ⟨u, h⟩) else 0

theorem bK_add (a b : 𝔽) (u : ℕ) : bK (a + b) u = bK a u + bK b u := by
  unfold bK
  split_ifs
  · rw [unpack_add, Pi.add_apply, map_add]
  · rw [add_zero]

theorem bK_sum {ι : Type} (s : Finset ι) (f : ι → 𝔽) (u : ℕ) : bK (∑ i ∈ s, f i) u = ∑ i ∈ s, bK (f i) u := by
  classical
  induction s using Finset.induction_on with
  | empty =>
      simp only [Finset.sum_empty]
      unfold bK
      split_ifs
      · rw [show (0 : 𝔽) = fOf 0 from fOf_zero.symm, unpack_fOf]
        simp [F128.bit, show (0 : F128) = ⟨0, 0⟩ from rfl]
      · rfl
  | insert a s ha ih => rw [Finset.sum_insert ha, Finset.sum_insert ha, bK_add, ih]

theorem bK_fOf (a : F128) (u : ℕ) (hu : u < 128) : bK (fOf a) u = if a.bit u then 1 else 0 := by
  unfold bK
  rw [dif_pos hu, unpack_fOf]
  split_ifs <;> simp

theorem bK_ite (c : Prop) [Decidable c] (a : 𝔽) (u : ℕ) : bK (if c then a else 0) u = if c then bK a u else 0 := by
  split_ifs
  · rfl
  · unfold bK
    split_ifs
    · rw [show (0 : 𝔽) = fOf 0 from fOf_zero.symm, unpack_fOf]
      simp [F128.bit, show (0 : F128) = ⟨0, 0⟩ from rfl]
    · rfl

/-- **Multiplication on bit vectors** (§12.4): the bits of `g·a` are the columns `g·x^u` of the bits of `a`. -/
theorem bK_mul (g : F128) (a : 𝔽) (l : ℕ) (hl : l < 128) :
    ∑ u ∈ Finset.range 128, (if (mulColumns g)[u]!.bit l then bK a u else 0) = bK (fOf g * a) l := by
  have ha : a = ∑ v : Fin 128, if unpack a v = 1 then fOf (F128.monomial v) else 0 := by
    rw [← pack_eq_sum, pack_unpack]
  have hms : fOf g * a = ∑ v : Fin 128, fOf g * (if unpack a v = 1 then fOf (F128.monomial v) else 0) := by
    conv_lhs => rw [ha]
    exact Finset.mul_sum _ _ _
  rw [hms, bK_sum, ← Fin.sum_univ_eq_sum_range (fun u => if (mulColumns g)[u]!.bit l then bK a u else 0) 128]
  refine Finset.sum_congr rfl fun v _ => ?_
  rw [mul_ite, mul_zero, bK_ite, ← mulColumns_get g v v.isLt, bK_fOf _ l hl]
  unfold bK
  rw [dif_pos v.isLt]
  rcases (show ∀ z : ZMod 2, z = 0 ∨ z = 1 by decide) (unpack a v) with h0 | h1
  · rw [h0]; simp
  · rw [h1]; simp

/-! ## The branching program's invariant -/

/-- The state after `J` coordinates of the point `P`: `Σ_w eq(P↾J, w)·bits(eq(z↾J, w))`. -/
noncomputable def invS (P : List 𝕂) (z : Array F128) (J : ℕ) (u : ℕ) : 𝕂 :=
  ∑ w ∈ Finset.range (2 ^ J), Model.eqList (P.take J) w * bK (Model.eqAt (vecOf J z) w) u

theorem eqList_append_one (L : List 𝕂) (a : 𝕂) (w : ℕ) :
    Model.eqList (L ++ [a]) w = Model.eqList L w * (if w.testBit L.length then a else 1 + a) := by
  unfold Model.eqList
  rw [List.zipIdx_append, List.map_append, List.prod_append]
  simp

theorem eqList_take_succ (P : List 𝕂) (J : ℕ) (hJ : J < P.length) (w : ℕ) :
    Model.eqList (P.take (J + 1)) w =
      Model.eqList (P.take J) w * (if w.testBit J then P[J] else 1 + P[J]) := by
  rw [List.take_succ, List.getElem?_eq_getElem hJ, Option.toList_some, eqList_append_one,
    List.length_take, Nat.min_eq_left hJ.le]

theorem eqList_two_pow_add (L : List 𝕂) (w : ℕ) (hw : w < 2 ^ L.length) :
    Model.eqList L (2 ^ L.length + w) = Model.eqList L w := by
  rw [eqList_eq, eqList_eq]
  refine Finset.prod_congr rfl fun i hi => ?_
  rw [Finset.mem_range] at hi
  rw [Nat.testBit_two_pow_add_gt hi]

theorem eqAt_succ (z : Array F128) (J w : ℕ) :
    Model.eqAt (vecOf (J + 1) z) w =
      Model.eqAt (vecOf J z) w * (if w.testBit J then fOf z[J]! else 1 + fOf z[J]!) := by
  unfold Model.eqAt
  rw [Fin.prod_univ_castSucc]
  rfl

theorem eqAt_two_pow_add (z : Array F128) (J w : ℕ) :
    Model.eqAt (vecOf J z) (2 ^ J + w) = Model.eqAt (vecOf J z) w := by
  unfold Model.eqAt
  refine Finset.prod_congr rfl fun i _ => ?_
  rw [Nat.testBit_two_pow_add_gt i.isLt]

/-- The multiplication columns act on the invariant coordinate-wise in `w`. -/
theorem mat_invS (P : List 𝕂) (z : Array F128) (J : ℕ) (S : ℕ → 𝕂) (hS : ∀ u < 128, S u = invS P z J u)
    (c : F128) (l : ℕ) (hl : l < 128) :
    ∑ u ∈ Finset.range 128, (if (mulColumns c)[u]!.bit l then S u else 0) =
      ∑ w ∈ Finset.range (2 ^ J), Model.eqList (P.take J) w * bK (fOf c * Model.eqAt (vecOf J z) w) l := by
  have h1 : ∑ u ∈ Finset.range 128, (if (mulColumns c)[u]!.bit l then S u else 0) =
      ∑ u ∈ Finset.range 128, ∑ w ∈ Finset.range (2 ^ J),
        Model.eqList (P.take J) w * (if (mulColumns c)[u]!.bit l then bK (Model.eqAt (vecOf J z) w) u else 0) := by
    refine Finset.sum_congr rfl fun u hu => ?_
    rw [Finset.mem_range] at hu
    rw [hS u hu, invS]
    split_ifs <;> simp
  rw [h1, Finset.sum_comm]
  refine Finset.sum_congr rfl fun w _ => ?_
  rw [← bK_mul c _ l hl]
  exact (Finset.mul_sum _ _ _).symm

theorem char2 (E z : 𝔽) : E * (1 + z) + E * z = E := by
  rw [mul_add, mul_one, add_assoc, CharTwo.add_self_eq_zero, add_zero]

/-- **One coordinate** (§12.4): the invariant after `J + 1` coordinates, from the one after `J`. -/
theorem invS_succ (P : List 𝕂) (z : Array F128) (J : ℕ) (hJ : J < P.length) (l : ℕ) :
    invS P z (J + 1) l = ∑ w ∈ Finset.range (2 ^ J), Model.eqList (P.take J) w *
      (bK ((1 + fOf z[J]!) * Model.eqAt (vecOf J z) w) l + P[J] * bK (Model.eqAt (vecOf J z) w) l) := by
  unfold invS
  have hlen : (P.take J).length = J := by simp; omega
  rw [pow_succ, mul_two, Finset.sum_range_add, ← Finset.sum_add_distrib]
  refine Finset.sum_congr rfl fun w hw => ?_
  rw [Finset.mem_range] at hw
  have hE : Model.eqList (P.take J) (2 ^ J + w) = Model.eqList (P.take J) w := by
    have := eqList_two_pow_add (P.take J) w (by rw [hlen]; exact hw)
    rwa [hlen] at this
  rw [eqList_take_succ P J hJ, eqList_take_succ P J hJ, eqAt_succ, eqAt_succ, Nat.testBit_two_pow_add_eq,
    Nat.testBit_lt_two_pow hw, eqAt_two_pow_add, hE]
  simp only [Bool.not_false, Bool.false_eq_true, ite_true, ite_false]
  have hsplit : bK (Model.eqAt (vecOf J z) w) l =
      bK (Model.eqAt (vecOf J z) w * (1 + fOf z[J]!)) l + bK (Model.eqAt (vecOf J z) w * fOf z[J]!) l := by
    rw [← bK_add, char2]
  rw [hsplit, mul_comm (1 + fOf z[J]!)]
  ring

/-! ## The executable's loops -/

/-- A prefix step (§12.4): `s ← M_{1+z_J}·s + q_J·s`. -/
def preStep (z : Array F128) (q : Array F256) (s : Array F256) (j : ℕ) : Array F256 :=
  (Array.range 128).map fun l => (applyMap (mulColumns (1 + z[j]!)) s)[l]! + q[j]! * s[l]!

theorem preStep_inv (P : List 𝕂) (z : Array F128) (q : Array F256) (J : ℕ) (hJ : J < P.length)
    (hP : P[J] = gOf q[J]!) (s : Array F256) (hs : ∀ u < 128, gOf s[u]! = invS P z J u) :
    ∀ l < 128, gOf (preStep z q s J)[l]! = invS P z (J + 1) l := by
  intro l hl
  unfold preStep
  rw [getElem!_range_map _ hl, gOf_add, gOf_mul, applyMap_get _ _ _ hl,
    mat_invS P z J (fun u => gOf s[u]!) hs _ l hl, hs l hl, invS_succ P z J hJ, ← hP, invS]
  calc _ = ∑ w ∈ Finset.range (2 ^ J), (Model.eqList (P.take J) w *
        bK (fOf (1 + z[J]!) * Model.eqAt (vecOf J z) w) l +
        P[J] * (Model.eqList (P.take J) w * bK (Model.eqAt (vecOf J z) w) l)) := by
        rw [Finset.sum_add_distrib]
        exact congrArg (_ + ·) (Finset.mul_sum _ _ _)
    _ = _ := Finset.sum_congr rfl fun w _ => by rw [fOf_add, fOf_one]; ring

/-- A suffix step (§12.4): the coordinate is a bit of `y`, so the step is `M_{z_J}` or `M_{1+z_J}`. -/
theorem sufStep_inv (P : List 𝕂) (z : Array F128) (J : ℕ) (hJ : J < P.length) (bit : Bool)
    (hP : P[J] = if bit then 1 else 0) (t : Array F256) (ht : ∀ u < 128, gOf t[u]! = invS P z J u) :
    ∀ l < 128, gOf (applyMap (if bit then mulColumns z[J]! else mulColumns (1 + z[J]!)) t)[l]! =
      invS P z (J + 1) l := by
  intro l hl
  rw [applyMap_get _ _ _ hl, invS_succ P z J hJ, hP]
  cases bit with
  | false =>
      simp only [Bool.false_eq_true, ite_false, zero_mul, add_zero]
      rw [mat_invS P z J (fun u => gOf t[u]!) ht _ l hl]
      refine Finset.sum_congr rfl fun w _ => ?_
      rw [fOf_add, fOf_one]
  | true =>
      simp only [ite_true, one_mul]
      rw [mat_invS P z J (fun u => gOf t[u]!) ht _ l hl]
      refine Finset.sum_congr rfl fun w _ => ?_
      rw [← bK_add]
      congr 2
      rw [add_mul, one_mul, add_comm, ← add_assoc, CharTwo.add_self_eq_zero, zero_add]

theorem one_bit : ∀ u : Fin 128, (1 : F128).bit u.val = decide (u.val = 0) := by decide

theorem invS_zero (P : List 𝕂) (z : Array F128) (u : ℕ) (hu : u < 128) :
    invS P z 0 u = if u = 0 then 1 else 0 := by
  unfold invS
  rw [pow_zero, Finset.sum_range_one]
  have h1 : Model.eqList (P.take 0) 0 = 1 := by simp [Model.eqList]
  have h2 : Model.eqAt (vecOf 0 z) 0 = fOf 1 := by simp [Model.eqAt, fOf_one]
  rw [h1, h2, one_mul, bK_fOf 1 u hu, one_bit ⟨u, hu⟩]
  simp

theorem start_get (u : ℕ) (hu : u < 128) :
    gOf ((Array.replicate 128 (0 : F256)).set! 0 1)[u]! = if u = 0 then 1 else 0 := by
  rw [Array.set!_eq_setIfInBounds, getElem!_pos _ u (by simp; omega), Array.getElem_setIfInBounds (by simp; omega)]
  split_ifs with h1 h2 h2
  · exact gOf_one
  · exact absurd h1.symm h2
  · exact absurd h2.symm h1
  · rw [Array.getElem_replicate, gOf_zero]

theorem preLoop_inv (P : List 𝕂) (z : Array F128) (q : Array F256) (hq : q.size ≤ P.length)
    (hP : ∀ j < q.size, P.getD j 0 = gOf q[j]!) :
    ∀ (J : ℕ), J ≤ q.size → ∀ u < 128,
      gOf ((List.range' 0 J).foldl (preStep z q) ((Array.replicate 128 (0 : F256)).set! 0 1))[u]! = invS P z J u := by
  intro J
  induction J with
  | zero => intro _ u hu; rw [List.range'_zero, List.foldl_nil, start_get u hu, invS_zero P z u hu]
  | succ J ih =>
      intro hJ u hu
      rw [List.range'_concat, List.foldl_append, List.foldl_cons, List.foldl_nil,
        show 0 + 1 * J = J by omega]
      refine preStep_inv P z q J (by omega) ?_ _ (ih (by omega)) u hu
      rw [← hP J (by omega), List.getD_eq_getElem _ _ (by omega)]

theorem sufLoop_inv (P : List 𝕂) (z : Array F128) (Q e y : ℕ) (hP : P.length = Q + e)
    (hbits : ∀ i < e, P.getD (Q + i) 0 = if y.testBit i then 1 else 0) (pre : Array F256)
    (hpre : ∀ u < 128, gOf pre[u]! = invS P z Q u) :
    ∀ (i : ℕ), i ≤ e → ∀ u < 128,
      gOf ((List.range' 0 i).foldl (fun t i => applyMap (if ((y >>> i) % 2 == 1) = true then
        ((Array.range e).map fun i => mulColumns z[Q + i]!)[i]! else
        ((Array.range e).map fun i => mulColumns (1 + z[Q + i]!))[i]!) t) pre)[u]! = invS P z (Q + i) u := by
  intro i
  induction i with
  | zero => intro _ u hu; exact hpre u hu
  | succ i ih =>
      intro hi u hu
      rw [List.range'_concat, List.foldl_append, List.foldl_cons, List.foldl_nil, show 0 + 1 * i = i by omega,
        getElem!_range_map _ (by omega : i < e), getElem!_range_map _ (by omega : i < e), bit_eq]
      have h := sufStep_inv P z (Q + i) (by omega) (y.testBit i)
        (by rw [← hbits i (by omega), List.getD_eq_getElem _ _ (by omega)]) _ (ih (by omega)) u hu
      rw [show Q + (i + 1) = Q + i + 1 by omega]
      rw [← h]

theorem castHom_embed (b : ZMod 2) : ZMod.castHom (dvd_refl 2) 𝕂 b = execArith.embed ((b.val : ℕ) : 𝔽) := by
  rcases (show ∀ z : ZMod 2, z = 0 ∨ z = 1 by decide) b with h | h <;> subst h <;> simp

/-- **The contraction with `eq(r'')`** (§12.4): the program's last step is the model's `phi` on each weight. -/
theorem contract_eq (t : Array F256) (rpp : Array F128) (hrpp : rpp.size = 7) (P : List 𝕂) (z : Array F128) (N : ℕ)
    (ht : ∀ u < 128, gOf t[u]! = invS P z N u) :
    gOf ((Array.range 128).foldl (fun acc u => acc + t[u]!.smul (eqTable rpp)[u]!) 0) =
      ∑ w ∈ Finset.range (2 ^ N), Model.eqList (P.take N) w *
        execArith.embed (Model.phi execArith (vecOf 7 rpp) (Model.eqAt (vecOf N z) w)) := by
  rw [range_foldl, gOf_foldl_add, gOf_zero, zero_add, FlockLevel3.list_sum_range]
  have h1 : ∀ u ∈ Finset.range 128, gOf (t[u]!.smul (eqTable rpp)[u]!) =
      ∑ w ∈ Finset.range (2 ^ N), Model.eqList (P.take N) w *
        (bK (Model.eqAt (vecOf N z) w) u * execArith.embed (Model.eqAt (vecOf 7 rpp) u)) := by
    intro u hu
    rw [Finset.mem_range] at hu
    rw [gOf_smul, ht u hu, eqTable_vecOf rpp 7 hrpp u (by norm_num; omega), invS]
    exact (Finset.sum_mul _ _ _).trans (Finset.sum_congr rfl fun w _ => by rw [mul_assoc]; rfl)
  rw [Finset.sum_congr rfl h1, Finset.sum_comm]
  refine Finset.sum_congr rfl fun w _ => ?_
  refine (Finset.mul_sum _ _ _).symm.trans (congrArg (_ * ·) ?_)
  unfold Model.phi
  rw [map_sum]
  refine (Fin.sum_univ_eq_sum_range (fun u => bK (Model.eqAt (vecOf N z) w) u *
    execArith.embed (Model.eqAt (vecOf 7 rpp) u)) 128).symm.trans (Finset.sum_congr rfl fun u _ => ?_)
  rw [map_mul, bK, dif_pos u.isLt, castHom_embed]
  rfl

theorem push_fold_get {α : Type} [Inhabited α] (V : ℕ → α) : ∀ (k a : ℕ) (out : Array α),
    ((List.range' a k).foldl (fun out y => out.push (V y)) out).size = out.size + k ∧
    ∀ i < k, ((List.range' a k).foldl (fun out y => out.push (V y)) out)[out.size + i]! = V (a + i)
  | 0, _, out => ⟨rfl, fun _ hi => absurd hi (Nat.not_lt_zero _)⟩
  | k + 1, a, out => by
      rw [List.range'_succ, List.foldl_cons]
      obtain ⟨h1, h2⟩ := push_fold_get V k (a + 1) (out.push (V a))
      refine ⟨by rw [h1, Array.size_push]; omega, fun i hi => ?_⟩
      rcases i with _ | i
      · have hk : ∀ (k a : ℕ) (out : Array α) (j : ℕ), j < out.size →
            ((List.range' a k).foldl (fun out y => out.push (V y)) out)[j]! = out[j]! := by
          intro k
          induction k with
          | zero => intro a out j _; rfl
          | succ k ih =>
              intro a out j hj
              rw [List.range'_succ, List.foldl_cons, ih _ _ j (by simp; omega), getElem!_pos _ j (by simp; omega),
                getElem!_pos out j hj, Array.getElem_push_lt]
        simp only [Nat.add_zero]
        rw [hk k (a + 1) (out.push (V a)) out.size (by simp), getElem!_pos _ out.size (by simp),
          Array.getElem_push_eq]
      · have := h2 i (by omega)
        rw [Array.size_push, show out.size + 1 + i = out.size + (i + 1) by omega,
          show a + 1 + i = a + (i + 1) by omega] at this
        exact this

theorem push_get_empty {α : Type} [Inhabited α] (V : ℕ → α) (c k i : ℕ) (hi : i < k) :
    ((List.range' 0 k).foldl (fun out y => out.push (V y)) (Array.mkEmpty c))[i]! = V i := by
  have := (push_fold_get V k 0 (Array.mkEmpty c)).2 i hi
  rw [show (Array.mkEmpty c : Array α).size = 0 from rfl, Nat.zero_add] at this
  exact this

theorem bfc_get (sw : Switched) (q : Array F256) (e : ℕ) (hrpp : sw.rpp.size = 7) (y : ℕ) (hy : y < 2 ^ e) :
    gOf (basisForClaim sw q e)[y]! = ∑ w ∈ Finset.range (2 ^ (q.size + e)),
      Model.eqList (q.toList.map gOf ++ bitsK e y) w *
        execArith.embed (Model.phi execArith (vecOf 7 sw.rpp) (Model.eqAt (vecOf (q.size + e) sw.z) w)) := by
  unfold basisForClaim
  simp only [Id.run, Std.Legacy.Range.forIn_eq_forIn_range', range_size, List.forIn_pure_yield_eq_foldl, pure_bind]
  change gOf (List.foldl _ (Array.mkEmpty (2 ^ e)) (List.range' 0 (2 ^ e)))[y]! = _
  rw [push_get_empty _ _ _ _ hy]
  have hlen : (q.toList.map gOf ++ bitsK e y).length = q.size + e := by simp [bitsK]
  have hpre := preLoop_inv (q.toList.map gOf ++ bitsK e y) sw.z q (by rw [hlen]; omega)
    (fun j hj => by rw [List.getD_append _ _ _ _ (by simpa using hj), getD_map_gOf _ _ hj]) q.size le_rfl
  have hsuf := sufLoop_inv (q.toList.map gOf ++ bitsK e y) sw.z q.size e y hlen
    (fun i hi => by
      rw [List.getD_append_right _ _ _ _ (by simp), List.length_map, Array.length_toList, Nat.add_sub_cancel_left,
        bitsK, List.getD_eq_getElem _ _ (by simpa using hi), List.getElem_map, List.getElem_range])
    _ hpre e le_rfl
  refine (contract_eq _ sw.rpp hrpp (q.toList.map gOf ++ bitsK e y) sw.z (q.size + e) hsuf).trans ?_
  rw [List.take_of_length_le (by rw [hlen])]

theorem basis_fold_get (q : Array F256) (e : ℕ) {y : ℕ} (hy : y < 2 ^ e) : ∀ (L : List Switched) (init : Array F256),
    (L.foldl (fun acc sw => (Array.range (2 ^ e)).map fun y =>
      acc[y]! + (basisForClaim sw q e)[y]!.smul sw.gamma) init)[y]! =
      L.foldl (fun acc sw => acc + (basisForClaim sw q e)[y]!.smul sw.gamma) init[y]!
  | [], _ => rfl
  | sw :: L, init => by
      rw [List.foldl_cons, List.foldl_cons, basis_fold_get q e hy L, getElem!_range_map _ hy]

theorem list_sum_array {α : Type} [Inhabited α] (a : Array α) (f : α → 𝕂) :
    (a.toList.map f).sum = ∑ k ∈ Finset.range a.size, f a[k]! := by
  rw [← FlockLevel3.list_sum_range]
  congr 1
  refine List.ext_getElem (by simp) fun i h₁ h₂ => ?_
  simp only [List.getElem_map, Array.getElem_toList, List.getElem_range]
  rw [getElem!_pos a i (by simpa using h₁)]

/-- **The batched basis** (§12.4): the executable's `basis sws q e` at `y < 2^e` is the model's `bHat` of the opening's
batched basis (R4's `OpenRel.b`) at `q ‖ bits(y)`, for every prefix `q` with `|q| + e = m − 7`. -/
theorem basis_eq {m n : ℕ} {T : F128} {sws : Array Switched} {op : Model.OpenOut 𝔽} (hop : OpenRel m n T sws op)
    (q : Array F256) (e : ℕ) (hqe : q.size + e = m - 7) (y : ℕ) (hy : y < 2 ^ e) :
    gOf (basis sws q e)[y]! = Model.bHat execArith (m - 7) op.b (q.toList.map gOf ++ bitsK e y) := by
  unfold basis
  rw [← Array.foldl_toList, basis_fold_get q e hy, gOf_foldl_add, getElem!_pos _ y (by simpa using hy),
    Array.getElem_replicate, gOf_zero, zero_add, list_sum_array, hop.size]
  unfold Model.bHat
  have hrhs : ∀ w ∈ Finset.range (2 ^ (m - 7)), execArith.embed (op.b w) *
      Model.eqList (q.toList.map gOf ++ bitsK e y) w = ∑ k ∈ Finset.range n,
        execArith.embed (fOf (sws[k]!).gamma) * (Model.eqList (q.toList.map gOf ++ bitsK e y) w *
          execArith.embed (Model.phi execArith (vecOf 7 (sws[k]!).rpp) (Model.eqAt (vecOf (m - 7) (sws[k]!).z) w))) := by
    intro w _
    rw [hop.b w, map_sum, ← Fin.sum_univ_eq_sum_range (fun k => execArith.embed (fOf (sws[k]!).gamma) *
      (Model.eqList (q.toList.map gOf ++ bitsK e y) w *
        execArith.embed (Model.phi execArith (vecOf 7 (sws[k]!).rpp)
          (Model.eqAt (vecOf (m - 7) (sws[k]!).z) w)))) n]
    refine (Finset.sum_mul _ _ _).trans (Finset.sum_congr rfl fun k _ => ?_)
    rw [map_mul]
    ring
  rw [Finset.sum_congr rfl hrhs, Finset.sum_comm]
  refine Finset.sum_congr rfl fun k hk => ?_
  rw [Finset.mem_range] at hk
  rw [gOf_smul, bfc_get _ q e (hop.rpp_size k hk) y hy, hqe, mul_comm]
  exact Finset.mul_sum _ _ _

/-- **Ligerito on the opening's output refines the model's** (§12.4, §13): when the executable's Ligerito accepts on
ring switching's `T` and `basis sws`, the model's compiled Ligerito on the opening's output `op` (R4's `OpenRel`),
played with the proof's messages and the rounds consumed, accepts at the openings `o`. -/
theorem ligerito_accepts_opening {D : Type} (ms : MerkleScheme) (sch : Flock.Schedule) (T : F128)
    (sws : Array Switched) (n : ℕ) (op : Model.OpenOut 𝔽) (hop : OpenRel sch.m n T sws op) (cap : Array ByteArray)
    (p : LigeritoProof) (r : ℕ) (capD : Array ByteArray → Cap D) (o : Opens 𝔽 D) (hdim : DimSeam sch)
    (hlv : ∀ i < sch.levels.size, LevelSeam (sch.levels.getD i default))
    (hα1 : ∀ i < sch.levels.size, ceilLog2 (sch.levels.getD i default).q ≠ 1)
    (ho : ∀ ℓ < sch.levels.size, ∀ j < (sch.levels.getD ℓ default).q,
      (o r ℓ j).1 = ((openAt p (sch.levels.size - 1) ℓ).1)[j]!.toList.map fOf)
    {t t' : Transcript} (hwf : RoundsWF t.rounds)
    (h : Flock.ligerito ms sch T cap p (basis sws) t = .ok ((), t')) :
    ∃ pend : PendL 𝔽 D, t'.rounds = t.rounds ∧ t.k ≤ t'.k ∧
      Plays fOf (Model.ligeritoC execArith r (schedOf sch) op.T op.b) (ligMsgs p capD sch)
        (consumed t.rounds t.k t'.k) pend ∧
      pend.caps = (List.range (sch.levels.size - 1)).map (fun j => capD p.recursiveCaps[j]!) ∧
      (∀ req ∈ pend.reqs, ReqOK ms sch cap p req) ∧ pend.verdict o = true := by
  rw [hop.T]
  exact ligerito_accepts ms sch T cap p (basis sws) r op.b capD o hdim hlv hα1 ho
    (fun orig y horig hy => basis_eq hop orig sch.ext horig y hy) hwf h

end FlockSoundness.Refine
