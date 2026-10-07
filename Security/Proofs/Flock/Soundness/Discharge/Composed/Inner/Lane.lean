import Proofs.Flock.Soundness.Discharge.Composed.Inner
import Proofs.Flock.Soundness.LigeritoRes

/-!
# `innerOf`'s lane phase: level 0's two lane-phase rows vanish at an honest prover's pads

Rep `r`'s rows `ligerito lane phase (c0)`, `(c1)` (`rowsOf`'s rows 5 and 6: `limbRows` of `laneRow`), on the honest
masked values (`honestMasked`: `honestAt` plus the exposed pads) and the honest restart `T' = Σ_c y₁'(c)·b̄(c)`, given the
ring switch's target `T = Σ_y f(y)·b(y)` (`hT`, at the opening's batched basis `openB`). No coin is excluded: the lane
rounds don't divide, so these hold at every coin vector, an outer coin `1` included.

* `laneRun_eval`: after `j ≤ k₀` lane rounds the claim's value at the pads is `Σ_y f_j(y)·B_j(y)` (`laneF`, `laneB`), from
  `T + Σ_i β_i·y_i = Σ_y f(y)·B(y)` and one honest round at a time (`lane_round`).
* `laneF_k0`: `f` folded by every lane challenge is `Σ_l eq(r̄, l)·f(l + 2^k₀·c)`, the folded lanes `y₁`.
* `laneRow_eval`, `laneRows_honest`: the `K` row's value is `0`, and so are both its coordinate rows.
-/

namespace FlockSoundness.Discharge.Composed

open Flock Model ZkSession ZkView ZK Refine

local notation "𝔽" => FlockLevel3.GF128
local notation "𝕂" => FlockLevel3.GF256

/-- **A `K` row's coordinate rows vanish where it does** (the converse of `eval_eq_zero_of_limbRows`). -/
theorem limbRows_eval_eq_zero {np : ℕ} (row : LinK 𝕂 np) (h : Fin np → 𝔽)
    (h0 : LinK.eval execArith h row = 0) (k : Fin 2) : (limbRows execArith row k).eval h = 0 := by
  have key : LinK.eval execArith h row =
      execArith.ofLimbs (∑ i, (coords execArith (row.1 i)).1 * h i + (coords execArith row.2).1)
        (∑ i, (coords execArith (row.1 i)).2 * h i + (coords execArith row.2).2) := by
    rw [← ofLimbs_add, ← ofLimbs_sum, ofLimbs_coords execArith execArith_correct]
    simp only [LinK.eval, LinearMap.coe_mk, AddHom.coe_mk]
    congr 1
    refine Finset.sum_congr rfl fun i _ => ?_
    rw [← ofLimbs_mul_embed, ofLimbs_coords execArith execArith_correct, mul_comm]
  rw [key, ← ofLimbs_zero execArith] at h0
  obtain ⟨h₀, h₁⟩ := ofLimbs_inj execArith execArith_correct h0
  fin_cases k
  · simpa [limbRows, Lin.eval] using h₀
  · simpa [limbRows, Lin.eval] using h₁

/-- **One honest lane round** (characteristic 2): the round's message `(q(0), q₂)` of `q(X) = Σ_y f(X, y)·B(X, y)`, read
from the claim `Σ_x f(x)·B(x)` (`from(msg, s)`), is the sum of the tables folded at `t`. -/
theorem lane_round {K : Type} [Field K] [CharP K 2] (n : ℕ) (f B : ℕ → K) (t : K) :
    (∑ y ∈ Finset.range (2 ^ n), f (2 * y) * B (2 * y)) +
        t * ((∑ x ∈ Finset.range (2 ^ (n + 1)), f x * B x) +
          ∑ y ∈ Finset.range (2 ^ n), (f (2 * y + 1) + f (2 * y)) * (B (2 * y + 1) + B (2 * y))) +
        t ^ 2 * ∑ y ∈ Finset.range (2 ^ n), (f (2 * y + 1) + f (2 * y)) * (B (2 * y + 1) + B (2 * y)) =
      ∑ y ∈ Finset.range (2 ^ n), foldLo t f y * foldLo t B y := by
  rw [sum_range_pairs, ← Finset.sum_add_distrib]
  simp only [Finset.mul_sum, ← Finset.sum_add_distrib]
  refine Finset.sum_congr rfl fun y _ => ?_
  have h2 : (2 : K) = 0 := CharTwo.two_eq_zero
  simp only [foldLo]
  linear_combination (t * f (2 * y + 1) * B (2 * y + 1)) * h2

/-- Folding by a list of challenges, one more at the end. -/
theorem foldLoL_append_single {K : Type} [Field K] :
    ∀ (l : List K) (t : K) (u : ℕ → K), foldLoL (l ++ [t]) u = foldLo t (foldLoL l u)
  | [], _, _ => rfl
  | a :: l, t, u => by rw [List.cons_append, foldLoL, foldLoL, foldLoL_append_single l t (foldLo a u)]

/-- Swapping a weighted double sum (stated over any commutative semiring: `Finset.mul_sum` doesn't rewrite at
`GF128`/`GF256`'s instances, but they unify with it). -/
theorem sum_mul_sum_swap {K ι κ : Type} [CommSemiring K] (s : Finset ι) (t : Finset κ) (a : ι → K)
    (e : ι → κ → K) (w : κ → K) :
    ∑ i ∈ s, a i * ∑ y ∈ t, e i y * w y = ∑ y ∈ t, w y * ∑ i ∈ s, a i * e i y := by
  simp only [Finset.mul_sum]
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun y _ => Finset.sum_congr rfl fun i _ => ?_
  ring

theorem mul_sum_assoc {K ι : Type} [CommSemiring K] (s : Finset ι) (a : K) (f g : ι → K) :
    a * ∑ x ∈ s, f x * g x = ∑ x ∈ s, a * f x * g x := by
  simp only [Finset.mul_sum, mul_assoc]

section Lane

variable {mPts : ℕ} (Tz : TabZK 𝔽 mPts) (st : Stmt) (c : CoinsZK (RestOf Tz st)) (r : Fin 2)
  (zw : Witness Tz.S.m) {nE L : ℕ} (R : Fin 2 → Fin L → 𝔽) (hp : Fin (nE + 4) → 𝔽)

/-- **The honest masked values**: each exposed value's clear value (`honestAt`) plus its pad. -/
noncomputable def honestMasked : Fin nE → 𝔽 := fun i => honestAt Tz st c r zw R i.val + hp (Fin.castAdd 4 i)

/-- A masked value unmasked by its pad is the clear value (characteristic 2). -/
theorem msgAt_add_pad (k : ℕ) (hk : k < nE) :
    msgAt (honestMasked Tz st c r zw R hp) k + hp (ix nE k) = honestAt Tz st c r zw R k := by
  have hix : ix nE k = Fin.castAdd 4 ⟨k, hk⟩ := Fin.ext (by simp [ix, Nat.mod_eq_of_lt (by omega : k < nE + 4)])
  simp only [msgAt, dite_eq_left hk, honestMasked, hix]
  rw [add_assoc, CharTwo.add_self_eq_zero, add_zero]

/-- A masked `K` message at the pads: its clear coordinates. -/
theorem eval_mkK (i₀ i₁ : ℕ) (h₀ : i₀ < nE) (h₁ : i₁ < nE) :
    LinK.eval execArith hp (mkK (honestMasked Tz st c r zw R hp) i₀ i₁) =
      execArith.ofLimbs (honestAt Tz st c r zw R i₀) (honestAt Tz st c r zw R i₁) := by
  simp only [mkK, map_add, eval_cK, eval_padK]
  rw [ofLimbs_add, msgAt_add_pad Tz st c r zw R hp i₀ h₀, msgAt_add_pad Tz st c r zw R hp i₁ h₁]

/-! ### The layout's level-0 indices -/


theorem lgOod_lt (hE : (layOf Tz st).hab = nE) (i : ℕ) (hi : i < (layOf Tz st).nOod0) :
    (layOf Tz st).lgOod i < nE := by
  subst hE; simp only [layOf, Zk.PadLayout.new, Zk.PadLayout.lgOod, Zk.PadLayout.hab] at hi ⊢; omega

theorem lgMsg_lt (hE : (layOf Tz st).hab = nE) (j t : ℕ) (hj : j < Tz.sch.k0) (ht : t < 4) :
    (layOf Tz st).lgMsg j t < nE := by
  subst hE; simp only [layOf, Zk.PadLayout.new, Zk.PadLayout.lgMsg, Zk.PadLayout.hab]; omega

theorem lgE_lt (hE : (layOf Tz st).hab = nE) (i t : ℕ) (hi : i < 2) (ht : t < 2) :
    (layOf Tz st).lgE i t < nE := by
  subst hE; simp only [layOf, Zk.PadLayout.new, Zk.PadLayout.lgE, Zk.PadLayout.hab]; omega

/-! ### The honest values at level 0's indices -/

/-- The out-of-domain value: `y₀ = Σ_y eq(z₀, y)·f(y)`. -/
theorem honestAt_lgOod (i : ℕ) (hi : i < (layOf Tz st).nOod0) :
    honestAt Tz st c r zw R ((layOf Tz st).lgOod i) =
      ∑ y ∈ Finset.range (2 ^ (Tz.S.m - 7)), eqAt (oodPoint Tz st c r i) y * wordOf zw y := by
  unfold honestAt
  dsimp only
  rw [ite_eq_right (by (simp only [layOf, Zk.PadLayout.new, Zk.PadLayout.fa, Zk.PadLayout.fb, Zk.PadLayout.fc, Zk.PadLayout.lc, Zk.PadLayout.zp, Zk.PadLayout.lgOod, Zk.PadLayout.lgMsg, Zk.PadLayout.lgE, Zk.PadLayout.hab] at *; omega)), ite_eq_right (by (simp only [layOf, Zk.PadLayout.new, Zk.PadLayout.fa, Zk.PadLayout.fb, Zk.PadLayout.fc, Zk.PadLayout.lc, Zk.PadLayout.zp, Zk.PadLayout.lgOod, Zk.PadLayout.lgMsg, Zk.PadLayout.lgE, Zk.PadLayout.hab] at *; omega)), ite_eq_right (by (simp only [layOf, Zk.PadLayout.new, Zk.PadLayout.fa, Zk.PadLayout.fb, Zk.PadLayout.fc, Zk.PadLayout.lc, Zk.PadLayout.zp, Zk.PadLayout.lgOod, Zk.PadLayout.lgMsg, Zk.PadLayout.lgE, Zk.PadLayout.hab] at *; omega)),
    ite_eq_right (by (simp only [layOf, Zk.PadLayout.new, Zk.PadLayout.fa, Zk.PadLayout.fb, Zk.PadLayout.fc, Zk.PadLayout.lc, Zk.PadLayout.zp, Zk.PadLayout.lgOod, Zk.PadLayout.lgMsg, Zk.PadLayout.lgE, Zk.PadLayout.hab] at *; omega)), ite_eq_right (by (simp only [layOf, Zk.PadLayout.new, Zk.PadLayout.fa, Zk.PadLayout.fb, Zk.PadLayout.fc, Zk.PadLayout.lc, Zk.PadLayout.zp, Zk.PadLayout.lgOod, Zk.PadLayout.lgMsg, Zk.PadLayout.lgE, Zk.PadLayout.hab] at *; omega)), ite_eq_right (by (simp only [layOf, Zk.PadLayout.new, Zk.PadLayout.fa, Zk.PadLayout.fb, Zk.PadLayout.fc, Zk.PadLayout.lc, Zk.PadLayout.zp, Zk.PadLayout.lgOod, Zk.PadLayout.lgMsg, Zk.PadLayout.lgE, Zk.PadLayout.hab] at *; omega)),
    ite_eq_right (by (simp only [layOf, Zk.PadLayout.new, Zk.PadLayout.fa, Zk.PadLayout.fb, Zk.PadLayout.fc, Zk.PadLayout.lc, Zk.PadLayout.zp, Zk.PadLayout.lgOod, Zk.PadLayout.lgMsg, Zk.PadLayout.lgE, Zk.PadLayout.hab] at *; omega)), ite_eq_right (by (simp only [layOf, Zk.PadLayout.new, Zk.PadLayout.fa, Zk.PadLayout.fb, Zk.PadLayout.fc, Zk.PadLayout.lc, Zk.PadLayout.zp, Zk.PadLayout.lgOod, Zk.PadLayout.lgMsg, Zk.PadLayout.lgE, Zk.PadLayout.hab] at *; omega)),
    ite_eq_left (by (simp only [layOf, Zk.PadLayout.new, Zk.PadLayout.fa, Zk.PadLayout.fb, Zk.PadLayout.fc, Zk.PadLayout.lc, Zk.PadLayout.zp, Zk.PadLayout.lgOod, Zk.PadLayout.lgMsg, Zk.PadLayout.lgE, Zk.PadLayout.hab] at *; omega)), show (layOf Tz st).lgOod i - (layOf Tz st).lgOod 0 = i by (simp only [layOf, Zk.PadLayout.new, Zk.PadLayout.fa, Zk.PadLayout.fb, Zk.PadLayout.fc, Zk.PadLayout.lc, Zk.PadLayout.zp, Zk.PadLayout.lgOod, Zk.PadLayout.lgMsg, Zk.PadLayout.lgE, Zk.PadLayout.hab] at *; omega)]

/-- Lane round `j`'s message coordinates. -/
theorem honestAt_lgMsg (j t : ℕ) (hj : j < Tz.sch.k0) (ht : t < 4) :
    honestAt Tz st c r zw R ((layOf Tz st).lgMsg j t) =
      let m := laneMsg Tz st c r zw j
      let x := if t < 2 then m.1 else m.2
      if t % 2 = 0 then (coords execArith x).1 else (coords execArith x).2 := by
  unfold honestAt
  dsimp only
  rw [ite_eq_right (by (simp only [layOf, Zk.PadLayout.new, Zk.PadLayout.fa, Zk.PadLayout.fb, Zk.PadLayout.fc, Zk.PadLayout.lc, Zk.PadLayout.zp, Zk.PadLayout.lgOod, Zk.PadLayout.lgMsg, Zk.PadLayout.lgE, Zk.PadLayout.hab] at *; omega)), ite_eq_right (by (simp only [layOf, Zk.PadLayout.new, Zk.PadLayout.fa, Zk.PadLayout.fb, Zk.PadLayout.fc, Zk.PadLayout.lc, Zk.PadLayout.zp, Zk.PadLayout.lgOod, Zk.PadLayout.lgMsg, Zk.PadLayout.lgE, Zk.PadLayout.hab] at *; omega)), ite_eq_right (by (simp only [layOf, Zk.PadLayout.new, Zk.PadLayout.fa, Zk.PadLayout.fb, Zk.PadLayout.fc, Zk.PadLayout.lc, Zk.PadLayout.zp, Zk.PadLayout.lgOod, Zk.PadLayout.lgMsg, Zk.PadLayout.lgE, Zk.PadLayout.hab] at *; omega)),
    ite_eq_right (by (simp only [layOf, Zk.PadLayout.new, Zk.PadLayout.fa, Zk.PadLayout.fb, Zk.PadLayout.fc, Zk.PadLayout.lc, Zk.PadLayout.zp, Zk.PadLayout.lgOod, Zk.PadLayout.lgMsg, Zk.PadLayout.lgE, Zk.PadLayout.hab] at *; omega)), ite_eq_right (by (simp only [layOf, Zk.PadLayout.new, Zk.PadLayout.fa, Zk.PadLayout.fb, Zk.PadLayout.fc, Zk.PadLayout.lc, Zk.PadLayout.zp, Zk.PadLayout.lgOod, Zk.PadLayout.lgMsg, Zk.PadLayout.lgE, Zk.PadLayout.hab] at *; omega)), ite_eq_right (by (simp only [layOf, Zk.PadLayout.new, Zk.PadLayout.fa, Zk.PadLayout.fb, Zk.PadLayout.fc, Zk.PadLayout.lc, Zk.PadLayout.zp, Zk.PadLayout.lgOod, Zk.PadLayout.lgMsg, Zk.PadLayout.lgE, Zk.PadLayout.hab] at *; omega)),
    ite_eq_right (by (simp only [layOf, Zk.PadLayout.new, Zk.PadLayout.fa, Zk.PadLayout.fb, Zk.PadLayout.fc, Zk.PadLayout.lc, Zk.PadLayout.zp, Zk.PadLayout.lgOod, Zk.PadLayout.lgMsg, Zk.PadLayout.lgE, Zk.PadLayout.hab] at *; omega)), ite_eq_right (by (simp only [layOf, Zk.PadLayout.new, Zk.PadLayout.fa, Zk.PadLayout.fb, Zk.PadLayout.fc, Zk.PadLayout.lc, Zk.PadLayout.zp, Zk.PadLayout.lgOod, Zk.PadLayout.lgMsg, Zk.PadLayout.lgE, Zk.PadLayout.hab] at *; omega)), ite_eq_right (by (simp only [layOf, Zk.PadLayout.new, Zk.PadLayout.fa, Zk.PadLayout.fb, Zk.PadLayout.fc, Zk.PadLayout.lc, Zk.PadLayout.zp, Zk.PadLayout.lgOod, Zk.PadLayout.lgMsg, Zk.PadLayout.lgE, Zk.PadLayout.hab] at *; omega)),
    ite_eq_left (by (simp only [layOf, Zk.PadLayout.new, Zk.PadLayout.fa, Zk.PadLayout.fb, Zk.PadLayout.fc, Zk.PadLayout.lc, Zk.PadLayout.zp, Zk.PadLayout.lgOod, Zk.PadLayout.lgMsg, Zk.PadLayout.lgE, Zk.PadLayout.hab] at *; omega)),
    show (layOf Tz st).lgMsg j t - (layOf Tz st).lgMsg 0 0 = 4 * j + t by (simp only [layOf, Zk.PadLayout.new, Zk.PadLayout.fa, Zk.PadLayout.fb, Zk.PadLayout.fc, Zk.PadLayout.lc, Zk.PadLayout.zp, Zk.PadLayout.lgOod, Zk.PadLayout.lgMsg, Zk.PadLayout.lgE, Zk.PadLayout.hab] at *; omega),
    show (4 * j + t) / 4 = j by omega, show (4 * j + t) % 4 = t by omega]

theorem honestAt_lgMsg_fst (j : ℕ) (hj : j < Tz.sch.k0) :
    execArith.ofLimbs (honestAt Tz st c r zw R ((layOf Tz st).lgMsg j 0))
      (honestAt Tz st c r zw R ((layOf Tz st).lgMsg j 1)) = (laneMsg Tz st c r zw j).1 := by
  rw [honestAt_lgMsg Tz st c r zw R j 0 hj (by omega), honestAt_lgMsg Tz st c r zw R j 1 hj (by omega)]
  simpa using ofLimbs_coords execArith execArith_correct (laneMsg Tz st c r zw j).1

theorem honestAt_lgMsg_snd (j : ℕ) (hj : j < Tz.sch.k0) :
    execArith.ofLimbs (honestAt Tz st c r zw R ((layOf Tz st).lgMsg j 2))
      (honestAt Tz st c r zw R ((layOf Tz st).lgMsg j 3)) = (laneMsg Tz st c r zw j).2 := by
  rw [honestAt_lgMsg Tz st c r zw R j 2 hj (by omega), honestAt_lgMsg Tz st c r zw R j 3 hj (by omega)]
  simpa using ofLimbs_coords execArith execArith_correct (laneMsg Tz st c r zw j).2

/-- The extra lanes' sums `e_i = Σ_c R_i(c)·b̄(c)`. -/
theorem honestAt_lgE (i : Fin 2) :
    execArith.ofLimbs (honestAt Tz st c r zw R ((layOf Tz st).lgE i 0))
      (honestAt Tz st c r zw R ((layOf Tz st).lgE i 1)) =
      ∑ col : Fin L, execArith.embed (R i col) * bFold Tz st c r col.val := by
  have hi := i.isLt
  have key : ∀ t < 2, honestAt Tz st c r zw R ((layOf Tz st).lgE i t) =
      let e := ∑ col : Fin L, execArith.embed (seqOf R i.val col) * bFold Tz st c r col.val
      if t % 2 = 0 then (coords execArith e).1 else (coords execArith e).2 := by
    intro t ht
    unfold honestAt
    dsimp only
    rw [ite_eq_right (by (simp only [layOf, Zk.PadLayout.new, Zk.PadLayout.fa, Zk.PadLayout.fb, Zk.PadLayout.fc, Zk.PadLayout.lc, Zk.PadLayout.zp, Zk.PadLayout.lgOod, Zk.PadLayout.lgMsg, Zk.PadLayout.lgE, Zk.PadLayout.hab] at *; omega)), ite_eq_right (by (simp only [layOf, Zk.PadLayout.new, Zk.PadLayout.fa, Zk.PadLayout.fb, Zk.PadLayout.fc, Zk.PadLayout.lc, Zk.PadLayout.zp, Zk.PadLayout.lgOod, Zk.PadLayout.lgMsg, Zk.PadLayout.lgE, Zk.PadLayout.hab] at *; omega)), ite_eq_right (by (simp only [layOf, Zk.PadLayout.new, Zk.PadLayout.fa, Zk.PadLayout.fb, Zk.PadLayout.fc, Zk.PadLayout.lc, Zk.PadLayout.zp, Zk.PadLayout.lgOod, Zk.PadLayout.lgMsg, Zk.PadLayout.lgE, Zk.PadLayout.hab] at *; omega)),
      ite_eq_right (by (simp only [layOf, Zk.PadLayout.new, Zk.PadLayout.fa, Zk.PadLayout.fb, Zk.PadLayout.fc, Zk.PadLayout.lc, Zk.PadLayout.zp, Zk.PadLayout.lgOod, Zk.PadLayout.lgMsg, Zk.PadLayout.lgE, Zk.PadLayout.hab] at *; omega)), ite_eq_right (by (simp only [layOf, Zk.PadLayout.new, Zk.PadLayout.fa, Zk.PadLayout.fb, Zk.PadLayout.fc, Zk.PadLayout.lc, Zk.PadLayout.zp, Zk.PadLayout.lgOod, Zk.PadLayout.lgMsg, Zk.PadLayout.lgE, Zk.PadLayout.hab] at *; omega)), ite_eq_right (by (simp only [layOf, Zk.PadLayout.new, Zk.PadLayout.fa, Zk.PadLayout.fb, Zk.PadLayout.fc, Zk.PadLayout.lc, Zk.PadLayout.zp, Zk.PadLayout.lgOod, Zk.PadLayout.lgMsg, Zk.PadLayout.lgE, Zk.PadLayout.hab] at *; omega)),
      ite_eq_right (by (simp only [layOf, Zk.PadLayout.new, Zk.PadLayout.fa, Zk.PadLayout.fb, Zk.PadLayout.fc, Zk.PadLayout.lc, Zk.PadLayout.zp, Zk.PadLayout.lgOod, Zk.PadLayout.lgMsg, Zk.PadLayout.lgE, Zk.PadLayout.hab] at *; omega)), ite_eq_right (by (simp only [layOf, Zk.PadLayout.new, Zk.PadLayout.fa, Zk.PadLayout.fb, Zk.PadLayout.fc, Zk.PadLayout.lc, Zk.PadLayout.zp, Zk.PadLayout.lgOod, Zk.PadLayout.lgMsg, Zk.PadLayout.lgE, Zk.PadLayout.hab] at *; omega)), ite_eq_right (by (simp only [layOf, Zk.PadLayout.new, Zk.PadLayout.fa, Zk.PadLayout.fb, Zk.PadLayout.fc, Zk.PadLayout.lc, Zk.PadLayout.zp, Zk.PadLayout.lgOod, Zk.PadLayout.lgMsg, Zk.PadLayout.lgE, Zk.PadLayout.hab] at *; omega)),
      ite_eq_right (by (simp only [layOf, Zk.PadLayout.new, Zk.PadLayout.fa, Zk.PadLayout.fb, Zk.PadLayout.fc, Zk.PadLayout.lc, Zk.PadLayout.zp, Zk.PadLayout.lgOod, Zk.PadLayout.lgMsg, Zk.PadLayout.lgE, Zk.PadLayout.hab] at *; omega)), ite_eq_left (by (simp only [layOf, Zk.PadLayout.new, Zk.PadLayout.fa, Zk.PadLayout.fb, Zk.PadLayout.fc, Zk.PadLayout.lc, Zk.PadLayout.zp, Zk.PadLayout.lgOod, Zk.PadLayout.lgMsg, Zk.PadLayout.lgE, Zk.PadLayout.hab] at *; omega)),
      show (layOf Tz st).lgE i t - (layOf Tz st).lgE 0 0 = 2 * i.val + t by (simp only [layOf, Zk.PadLayout.new, Zk.PadLayout.fa, Zk.PadLayout.fb, Zk.PadLayout.fc, Zk.PadLayout.lc, Zk.PadLayout.zp, Zk.PadLayout.lgOod, Zk.PadLayout.lgMsg, Zk.PadLayout.lgE, Zk.PadLayout.hab] at *; omega),
      show (2 * i.val + t) / 2 = i.val by omega, show (2 * i.val + t) % 2 = t by omega]
    interval_cases t <;> rfl
  rw [key 0 (by omega), key 1 (by omega)]
  have hR : seqOf R i.val = R i := by simp [seqOf, i.isLt]
  simpa [hR] using ofLimbs_coords execArith execArith_correct
    (∑ col : Fin L, execArith.embed (R i col) * bFold Tz st c r col.val)

/-! ### The lane rounds -/

/-- The claim before the lane rounds: `T + Σ_i β_i·y_i = Σ_y f(y)·B(y)`. -/
theorem laneRun_zero_eval (hE : (layOf Tz st).hab = nE) {nC : ℕ} (sh : Fin nC → Fin 128 → 𝔽)
    (hT : openT Tz st c r sh = ∑ y ∈ Finset.range (2 ^ (Tz.S.m - 7)), wordOf zw y * openB Tz st c r y) :
    LinK.eval execArith hp (laneRun Tz st c r (honestMasked Tz st c r zw R hp) sh 0) =
      ∑ y ∈ Finset.range (2 ^ (Tz.S.m - 7)), laneF Tz st c r zw 0 y * laneB Tz st c r 0 y := by
  simp only [laneRun, map_add, map_sum, map_smul, eval_cK, eval_padF, smul_eq_mul]
  have hood : ∀ i : Fin (layOf Tz st).nOod0,
      execArith.embed (msgAt (honestMasked Tz st c r zw R hp) ((layOf Tz st).lgOod i)) +
        execArith.embed (hp (ix nE ((layOf Tz st).lgOod i))) =
      execArith.embed (∑ y ∈ Finset.range (2 ^ (Tz.S.m - 7)), eqAt (oodPoint Tz st c r i.val) y * wordOf zw y) := by
    intro i
    rw [← map_add, msgAt_add_pad Tz st c r zw R hp _ (lgOod_lt Tz st hE i i.isLt),
      honestAt_lgOod Tz st c r zw R i i.isLt]
  simp only [hood, hT, laneF, laneB, laneB0, map_sum, map_mul, mul_add, Finset.sum_add_distrib]
  rw [add_right_inj]
  exact sum_mul_sum_swap _ _ _ _ _

/-- **After `j ≤ k₀` lane rounds** the claim's value at the pads is `Σ_y f_j(y)·B_j(y)`. -/
theorem laneRun_eval (hE : (layOf Tz st).hab = nE) (hk : Tz.sch.k0 ≤ Tz.S.m - 7) {nC : ℕ}
    (sh : Fin nC → Fin 128 → 𝔽)
    (hT : openT Tz st c r sh = ∑ y ∈ Finset.range (2 ^ (Tz.S.m - 7)), wordOf zw y * openB Tz st c r y) :
    ∀ j ≤ Tz.sch.k0, LinK.eval execArith hp (laneRun Tz st c r (honestMasked Tz st c r zw R hp) sh j) =
      ∑ y ∈ Finset.range (2 ^ (Tz.S.m - 7 - j)), laneF Tz st c r zw j y * laneB Tz st c r j y
  | 0, _ => laneRun_zero_eval Tz st c r zw R hp hE sh hT
  | j + 1, hj => by
      have ih := laneRun_eval hE hk sh hT j (by omega)
      have hj' : j < Tz.sch.k0 := by omega
      simp only [laneRun, QuadL.eval, fromMsgL, map_add, map_smul, smul_eq_mul]
      rw [eval_mkK Tz st c r zw R hp _ _ (lgMsg_lt Tz st hE j 0 hj' (by omega)) (lgMsg_lt Tz st hE j 1 hj' (by omega)),
        eval_mkK Tz st c r zw R hp _ _ (lgMsg_lt Tz st hE j 2 hj' (by omega)) (lgMsg_lt Tz st hE j 3 hj' (by omega)),
        honestAt_lgMsg_fst Tz st c r zw R j hj', honestAt_lgMsg_snd Tz st c r zw R j hj', ih]
      have hn : Tz.S.m - 7 - j = Tz.S.m - 7 - j - 1 + 1 := by omega
      rw [hn, show Tz.S.m - 7 - (j + 1) = Tz.S.m - 7 - j - 1 by omega]
      simp only [laneMsg]
      exact lane_round _ _ _ _

/-! ### The folded lanes -/

/-- The lane challenges `r̄_0 … r̄_{j−1}`. -/
def laneChs (j : ℕ) : List 𝕂 := (List.range j).map (laneCh Tz st c r)

theorem laneF_eq_foldLoL (j : ℕ) : laneF Tz st c r zw j = foldLoL (laneChs Tz st c r j) (laneF Tz st c r zw 0) := by
  induction j with
  | zero => rfl
  | succ j ih =>
      rw [laneChs, List.range_succ, List.map_append, List.map_singleton, foldLoL_append_single, ← laneChs, ← ih]
      rfl

/-- `eq(r̄, l)` over the lane challenges: `eqLane`'s product. -/
theorem eqList_laneChs (l : ℕ) :
    eqList (laneChs Tz st c r Tz.sch.k0) l = ∏ i : Fin Tz.sch.k0,
      if l.testBit i.val then FlockLevel3.GF256.of (c.rest.lane r i)
      else 1 + FlockLevel3.GF256.of (c.rest.lane r i) := by
  rw [eqList, prod_map_zipIdx, laneChs, List.length_map, List.length_range, ← Fin.prod_univ_eq_prod_range]
  refine Finset.prod_congr rfl fun i _ => ?_
  have hget : ((List.range Tz.sch.k0).map (laneCh Tz st c r)).getD i.val 0 = laneCh Tz st c r i.val := by
    simp [List.getD_eq_getElem?_getD, i.isLt]
  simp only [hget, zero_add, laneCh, seqOf, dite_eq_left i.isLt]

/-- **`f` folded by every lane challenge** is the `eq(r̄, ·)`-weighted sum of lane `l`'s word `l + 2^k₀·c`. -/
theorem laneF_k0 (col : ℕ) :
    laneF Tz st c r zw Tz.sch.k0 col = ∑ l : Fin (2 ^ Tz.sch.k0),
      (∏ i : Fin Tz.sch.k0, if l.val.testBit i.val then FlockLevel3.GF256.of (c.rest.lane r i)
        else 1 + FlockLevel3.GF256.of (c.rest.lane r i)) *
      execArith.embed (wordOf zw (l.val + 2 ^ Tz.sch.k0 * col)) := by
  rw [laneF_eq_foldLoL, foldLoL_eq_sum, laneChs, List.length_map, List.length_range, ← laneChs,
    ← Fin.sum_univ_eq_sum_range]
  refine Finset.sum_congr rfl fun l _ => ?_
  rw [eqList_laneChs]
  rfl

/-! ### The rows -/

/-- **Level 0's lane-phase row vanishes** at the honest pads: the claim after the `k₀` lane rounds is
`Σ_c y₁(c)·b̄(c)`, and `y₁' = y₁ + ρ₀·R₀ + ρ₁·R₁` gives `T' = claim + ρ₀·e₀ + ρ₁·e₁`. -/
theorem laneRow_eval (hE : (layOf Tz st).hab = nE) (hk : Tz.sch.k0 ≤ Tz.S.m - 7)
    (hL : L = 2 ^ (Tz.S.m - 7 - Tz.sch.k0)) {nC : ℕ} (sh : Fin nC → Fin 128 → 𝔽)
    (hT : openT Tz st c r sh = ∑ y ∈ Finset.range (2 ^ (Tz.S.m - 7)), wordOf zw y * openB Tz st c r y)
    (y' : Fin L → 𝕂)
    (hy : ∀ col : Fin L, y' col = ∑ l : Fin (2 ^ Tz.sch.k0),
      (∏ i : Fin Tz.sch.k0, if l.val.testBit i.val then FlockLevel3.GF256.of (c.rest.lane r i)
        else 1 + FlockLevel3.GF256.of (c.rest.lane r i)) *
      execArith.embed (wordOf zw (l.val + 2 ^ Tz.sch.k0 * col.val)) +
        (rhoK Tz st c r 0 * execArith.embed (R 0 col) + rhoK Tz st c r 1 * execArith.embed (R 1 col))) :
    LinK.eval execArith hp (laneRow Tz st c r (honestMasked Tz st c r zw R hp) sh
      (∑ col : Fin L, y' col * bFold Tz st c r col.val)) = 0 := by
  simp only [laneRow, map_add, map_sub, map_sum, map_smul, smul_eq_mul, eval_cK]
  rw [laneRun_eval Tz st c r zw R hp hE hk sh hT Tz.sch.k0 le_rfl]
  have he : ∀ i : Fin 2, LinK.eval execArith hp (mkK (honestMasked Tz st c r zw R hp)
      ((layOf Tz st).lgE i 0) ((layOf Tz st).lgE i 1)) =
      ∑ col : Fin L, execArith.embed (R i col) * bFold Tz st c r col.val := fun i => by
    rw [eval_mkK Tz st c r zw R hp _ _ (lgE_lt Tz st hE i 0 i.isLt (by omega)) (lgE_lt Tz st hE i 1 i.isLt (by omega)),
      honestAt_lgE]
  simp only [he, hy, add_mul, Finset.sum_add_distrib, Fin.sum_univ_two]
  subst hL
  rw [← Fin.sum_univ_eq_sum_range (fun y => laneF Tz st c r zw Tz.sch.k0 y * laneB Tz st c r Tz.sch.k0 y)]
  simp only [laneF_k0, bFold]
  rw [sub_eq_zero]
  congr 2 <;> exact mul_sum_assoc _ _ _ _

/-- **Both lane-phase rows vanish** at the honest pads (`rowsOf`'s rows 5 and 6). -/
theorem laneRows_honest (hE : (layOf Tz st).hab = nE) (hk : Tz.sch.k0 ≤ Tz.S.m - 7)
    (hL : L = 2 ^ (Tz.S.m - 7 - Tz.sch.k0)) {nC : ℕ} (sh : Fin nC → Fin 128 → 𝔽)
    (hT : openT Tz st c r sh = ∑ y ∈ Finset.range (2 ^ (Tz.S.m - 7)), wordOf zw y * openB Tz st c r y)
    (y' : Fin L → 𝕂)
    (hy : ∀ col : Fin L, y' col = ∑ l : Fin (2 ^ Tz.sch.k0),
      (∏ i : Fin Tz.sch.k0, if l.val.testBit i.val then FlockLevel3.GF256.of (c.rest.lane r i)
        else 1 + FlockLevel3.GF256.of (c.rest.lane r i)) *
      execArith.embed (wordOf zw (l.val + 2 ^ Tz.sch.k0 * col.val)) +
        (rhoK Tz st c r 0 * execArith.embed (R 0 col) + rhoK Tz st c r 1 * execArith.embed (R 1 col)))
    (k : Fin 2) :
    (limbRows execArith (laneRow Tz st c r (honestMasked Tz st c r zw R hp) sh
      (∑ col : Fin L, y' col * bFold Tz st c r col.val)) k).eval hp = 0 :=
  limbRows_eval_eq_zero _ _ (laneRow_eval Tz st c r zw R hp hE hk hL sh hT y' hy) k

end Lane

end FlockSoundness.Discharge.Composed
