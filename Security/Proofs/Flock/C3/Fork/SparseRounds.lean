import Proofs.Flock.Soundness.RepSound
import Proofs.Flock.Soundness.Model.Holo
import Mathlib.Tactic.LinearCombination
import Mathlib.Tactic.Push

/-!
# The sparse phase's rounds, one coin at a time

The two coin bounds `sparse_sound` (`C3/Fork/Sparse.lean`) chains, and the algebra they rest on:

* `value_drawFin_le`: Schwartz–Zippel per `drawFin` coin. A nonzero vector's multilinear extension at `n` fresh coins,
  drawn one at a time, is zero with probability at most `n/|F|`: each coin folds the lowest bit, and a nonzero pair
  `(u(2b), u(2b+1))` folds to zero at one coin at most.
* `value_degRounds_le`: `degRounds`' round step. A round's polynomial sums to the claim by construction
  (`g(0) + g(1) = claim` in characteristic 2), so on a false claim it differs from the true one, of degree `k + 1` in the
  round's variable (`DegLe`), and the next claim is false except with probability `(k + 1)/|F|`.

`cube G` is `G`'s sum over `{0,1}^n` (bit `j` of `b` is coordinate `j`, as in `eqAt`), and a function of degree one in
each variable is its multilinear extension (`sum_eqAt_mul`).
-/

namespace FlockC3.Fork.SparseCount

open FlockSoundness
open Game Model Polynomial
open scoped ENNReal

section Poly

variable {F : Type} [Field F]

/-- `g` is a polynomial function of degree at most `d`. -/
def PolyLe (d : ℕ) (g : F → F) : Prop := ∃ P : F[X], P.natDegree ≤ d ∧ ∀ x, g x = P.eval x

theorem PolyLe.congr {d : ℕ} {f g : F → F} (h : PolyLe d f) (hfg : ∀ x, f x = g x) : PolyLe d g := by
  obtain ⟨P, hP, he⟩ := h
  exact ⟨P, hP, fun x => (hfg x).symm.trans (he x)⟩

theorem PolyLe.mono {d d' : ℕ} {f : F → F} (h : PolyLe d f) (hd : d ≤ d') : PolyLe d' f := by
  obtain ⟨P, hP, he⟩ := h
  exact ⟨P, hP.trans hd, he⟩

theorem PolyLe.const (d : ℕ) (c : F) : PolyLe d fun _ => c :=
  ⟨C c, (natDegree_C c).le.trans (Nat.zero_le d), fun _ => (eval_C).symm⟩

theorem PolyLe.id : PolyLe 1 fun x : F => x :=
  ⟨X, natDegree_X_le, fun _ => (eval_X).symm⟩

theorem PolyLe.add {d : ℕ} {f g : F → F} (hf : PolyLe d f) (hg : PolyLe d g) : PolyLe d fun x => f x + g x := by
  obtain ⟨P, hP, he⟩ := hf
  obtain ⟨Q, hQ, he'⟩ := hg
  exact ⟨P + Q, (natDegree_add_le P Q).trans (max_le hP hQ), fun x => by rw [eval_add, ← he, ← he']⟩

theorem PolyLe.mul {d₁ d₂ : ℕ} {f g : F → F} (hf : PolyLe d₁ f) (hg : PolyLe d₂ g) :
    PolyLe (d₁ + d₂) fun x => f x * g x := by
  obtain ⟨P, hP, he⟩ := hf
  obtain ⟨Q, hQ, he'⟩ := hg
  exact ⟨P * Q, natDegree_mul_le.trans (add_le_add hP hQ), fun x => by rw [eval_mul, ← he, ← he']⟩

theorem PolyLe.const_mul {d : ℕ} {f : F → F} (c : F) (hf : PolyLe d f) : PolyLe d fun x => c * f x :=
  ((PolyLe.const 0 c).mul hf).mono (by omega)

theorem PolyLe.sum {ι : Type} (s : Finset ι) {d : ℕ} {f : ι → F → F} (h : ∀ i ∈ s, PolyLe d (f i)) :
    PolyLe d fun x => ∑ i ∈ s, f i x := by
  classical
  induction s using Finset.induction_on with
  | empty => exact (PolyLe.const d 0).congr fun x => by rw [Finset.sum_empty]
  | insert a s ha ih =>
    refine ((h a (Finset.mem_insert_self a s)).add (ih fun i hi => h i (Finset.mem_insert_of_mem hi))).congr
      fun x => ?_
    rw [Finset.sum_insert ha]

theorem PolyLe.prod {ι : Type} (s : Finset ι) {d : ι → ℕ} {f : ι → F → F} (h : ∀ i ∈ s, PolyLe (d i) (f i)) :
    PolyLe (∑ i ∈ s, d i) fun x => ∏ i ∈ s, f i x := by
  classical
  induction s using Finset.induction_on with
  | empty => exact (PolyLe.const _ 1).congr fun x => by rw [Finset.prod_empty]
  | insert a s ha ih =>
    rw [Finset.sum_insert ha]
    refine ((h a (Finset.mem_insert_self a s)).mul (ih fun i hi => h i (Finset.mem_insert_of_mem hi))).congr
      fun x => ?_
    rw [Finset.prod_insert ha]

/-- In characteristic 2 a polynomial function of degree one is its line through `0` and `1`. -/
theorem PolyLe.affine [CharP F 2] {g : F → F} (hg : PolyLe 1 g) (y : F) : (1 + y) * g 0 + y * g 1 = g y := by
  obtain ⟨P, hP, he⟩ := hg
  rw [he, he, he, eq_X_add_C_of_natDegree_le_one hP]
  simp only [eval_add, eval_mul, eval_C, eval_X]
  linear_combination (y * P.coeff 0) * Rounds.two_eq_zero (F := F)

/-- `G` has degree at most `d` in each variable. -/
def DegLe {n : ℕ} (d : ℕ) (G : (Fin n → F) → F) : Prop :=
  ∀ (v : Fin n → F) (j : Fin n), PolyLe d fun x => G (Function.update v j x)

theorem DegLe.cons {n d : ℕ} {G : (Fin (n + 1) → F) → F} (hG : DegLe d G) (r : F) :
    DegLe d fun w => G (Fin.cons r w) := fun v j =>
  (hG (Fin.cons r v) j.succ).congr fun x => by rw [← Fin.cons_update]

theorem DegLe.head {n d : ℕ} {G : (Fin (n + 1) → F) → F} (hG : DegLe d G) (w : Fin n → F) :
    PolyLe d fun x => G (Fin.cons x w) :=
  (hG (Fin.cons 0 w) 0).congr fun x => by rw [Fin.update_cons_zero]

theorem polyLe_update {n : ℕ} (v : Fin n → F) (j i : Fin n) :
    PolyLe (if i = j then 1 else 0) fun x => Function.update v j x i := by
  by_cases h : i = j
  · subst h
    rw [ite_eq_left rfl]
    exact PolyLe.id.congr fun x => by rw [Function.update_self]
  · rw [ite_eq_right h]
    exact (PolyLe.const 0 (v i)).congr fun x => by rw [Function.update_of_ne h]

theorem polyLe_eqAt {n : ℕ} (v : Fin n → F) (j : Fin n) (b : ℕ) :
    PolyLe 1 fun x => eqAt (Function.update v j x) b := by
  have h := PolyLe.prod (Finset.univ : Finset (Fin n)) (d := fun i => if i = j then 1 else 0)
    (f := fun i x => if b.testBit i then Function.update v j x i else 1 + Function.update v j x i) fun i _ => by
      by_cases hb : b.testBit i
      · exact (polyLe_update v j i).congr fun x => by rw [ite_eq_left hb]
      · exact ((PolyLe.const _ 1).add (polyLe_update v j i)).congr fun x => by rw [ite_eq_right hb]
  simp only [Finset.sum_ite_eq', Finset.mem_univ, ite_true] at h
  exact h

theorem sum_ite_le_one (s : Finset ℕ) (a : ℕ) : ∑ i ∈ s, (if i = a then 1 else 0) ≤ 1 := by
  rw [Finset.sum_ite_eq']
  split_ifs <;> omega

/-! ## The Boolean cube -/

/-- The Boolean point of `b`'s low `n` bits. -/
def bitsOf (n b : ℕ) : Fin n → F := fun j => if b.testBit j then 1 else 0

theorem bitsOf_two_mul_add (n b c : ℕ) (hc : c < 2) :
    (bitsOf (n + 1) (2 * b + c) : Fin (n + 1) → F) = Fin.cons (if c = 1 then 1 else 0) (bitsOf n b) := by
  funext j
  refine Fin.cases ?_ (fun i => ?_) j
  · have hb : (2 * b + c).testBit 0 = decide (c = 1) := by
      rw [Nat.testBit_zero]; congr 1; apply propext; omega
    simp only [bitsOf, Fin.cons_zero, Fin.val_zero, hb]
    by_cases hc1 : c = 1 <;> simp [hc1]
  · have hb : (2 * b + c).testBit (i.val + 1) = b.testBit i.val := by
      rw [Nat.testBit_succ]; congr 1; omega
    simp only [bitsOf, Fin.cons_succ, Fin.val_succ, hb]

/-- `G`'s sum over `{0,1}^n`. -/
def cube {n : ℕ} (G : (Fin n → F) → F) : F := ∑ b ∈ Finset.range (2 ^ n), G (bitsOf n b)

theorem cube_succ {n : ℕ} (G : (Fin (n + 1) → F) → F) :
    cube G = ∑ b ∈ Finset.range (2 ^ n), (G (Fin.cons 0 (bitsOf n b)) + G (Fin.cons 1 (bitsOf n b))) := by
  unfold cube
  rw [sum_range_pairs]
  refine Finset.sum_congr rfl fun b _ => ?_
  have h0 := bitsOf_two_mul_add (F := F) n b 0 (by norm_num)
  have h1 := bitsOf_two_mul_add (F := F) n b 1 (by norm_num)
  rw [add_zero, ite_eq_right (show (0 : ℕ) ≠ 1 by norm_num)] at h0
  rw [ite_eq_left rfl] at h1
  rw [h0, h1]

theorem eqAt_bitsOf [CharP F 2] {n b b₀ : ℕ} (hb : b < 2 ^ n) (hb₀ : b₀ < 2 ^ n) :
    eqAt (bitsOf n b₀ : Fin n → F) b = if b = b₀ then 1 else 0 :=
  eqAt_bool b b₀ hb hb₀

/-- At a Boolean point the equality weights pick one entry. -/
theorem sum_eqAt_bitsOf [CharP F 2] {n N : ℕ} (hN : N ≤ 2 ^ n) (b₀ : ℕ) (hb₀ : b₀ < N) (g : Fin N → F) :
    ∑ s : Fin N, eqAt (bitsOf n b₀ : Fin n → F) s.val * g s = g ⟨b₀, hb₀⟩ := by
  rw [Finset.sum_eq_single ⟨b₀, hb₀⟩ (fun s _ hs => ?_) (fun h => absurd (Finset.mem_univ _) h)]
  · rw [eqAt_bitsOf (lt_of_lt_of_le hb₀ hN) (lt_of_lt_of_le hb₀ hN), ite_eq_left rfl, one_mul]
  · rw [eqAt_bitsOf (lt_of_lt_of_le s.isLt hN) (lt_of_lt_of_le hb₀ hN),
      ite_eq_right (fun h => hs (Fin.ext h)), zero_mul]

/-- **A function of degree one in each variable is its multilinear extension.** -/
theorem sum_eqAt_mul [CharP F 2] : ∀ {n : ℕ} (G : (Fin n → F) → F), DegLe 1 G → ∀ ζ : Fin n → F,
    ∑ b ∈ Finset.range (2 ^ n), eqAt ζ b * G (bitsOf n b) = G ζ
  | 0, G, _, ζ => by
      rw [pow_zero, Finset.sum_range_one, show (bitsOf 0 0 : Fin 0 → F) = ζ from funext fun j => j.elim0]
      simp [eqAt]
  | n + 1, G, hG, ζ => by
      have ih := sum_eqAt_mul (fun w => G (Fin.cons (ζ 0) w)) (hG.cons (ζ 0)) (Fin.tail ζ)
      rw [Fin.cons_self_tail] at ih
      rw [← ih, sum_range_pairs]
      refine Finset.sum_congr rfl fun b _ => ?_
      have h0 := bitsOf_two_mul_add (F := F) n b 0 (by norm_num)
      have h1 := bitsOf_two_mul_add (F := F) n b 1 (by norm_num)
      rw [add_zero, ite_eq_right (show (0 : ℕ) ≠ 1 by norm_num)] at h0
      rw [ite_eq_left rfl] at h1
      have ha := (hG.head (bitsOf n b)).affine (ζ 0)
      beta_reduce at ha
      rw [h0, h1, eqAt_two_mul, eqAt_two_mul_add_one, ← ha]
      ring

theorem cube_eq_fin {n N : ℕ} (hN : N = 2 ^ n) (G : (Fin n → F) → F) :
    cube G = ∑ b : Fin N, G (bitsOf n b.val) := by
  subst hN
  rw [cube, Finset.sum_range]

theorem sum_fin_eqAt_mul [CharP F 2] {n N : ℕ} (hN : N = 2 ^ n) (G : (Fin n → F) → F) (hG : DegLe 1 G)
    (ζ : Fin n → F) : ∑ s : Fin N, eqAt ζ s.val * G (bitsOf n s.val) = G ζ := by
  subst hN
  rw [← sum_eqAt_mul G hG ζ, Finset.sum_range]

end Poly

/-! ## The phase's vectors, `Φ`'s degree, and a registered claim's point -/

section Vectors

variable {F : Type} [Field F]

theorem PolyLe.mul_const {d : ℕ} {f : F → F} (c : F) (hf : PolyLe d f) : PolyLe d fun x => f x * c :=
  (hf.mul (PolyLe.const 0 c)).mono (by omega)

/-- `sparsePhi`'s degree in a variable its arguments are polynomial functions of. -/
theorem polyLe_sparsePhi (ℓ : ℕ) (α : F) (x y q : F → ℕ → F) (e : ℕ) (dx dy : ℕ → ℕ)
    (hq₀ : PolyLe e fun t => q t (2 * ℓ)) (hq₁ : PolyLe e fun t => q t (2 * ℓ + 1))
    (hx : ∀ i < ℓ, PolyLe (dx i) fun t => x t i) (hqx : ∀ i < ℓ, PolyLe (dx i) fun t => q t i)
    (hy : ∀ i < ℓ, PolyLe (dy i) fun t => y t i) (hqy : ∀ i < ℓ, PolyLe (dy i) fun t => q t (ℓ + i)) :
    PolyLe (e + ∑ i ∈ Finset.range ℓ, (dx i + dy i)) fun t => sparsePhi ℓ α (x t) (y t) (q t) := by
  unfold sparsePhi
  refine ((hq₀.const_mul α).add hq₁).mul (PolyLe.prod _ fun i hi => ?_)
  have hi := Finset.mem_range.mp hi
  exact (((PolyLe.const _ 1).add (hx i hi)).add (hqx i hi)).mul
    (((PolyLe.const _ 1).add (hy i hi)).add (hqy i hi))

theorem polyLe_cat6 {m : ℕ} (a : Fin 6 → F) (j : Fin 6) (b : Fin m → F) (i : ℕ) :
    PolyLe (if i = j.val then 1 else 0) fun x => cat6 (Function.update a j x) b i := by
  by_cases hi : i < 6
  · have hd : (if i = j.val then 1 else 0) = (if (⟨i, hi⟩ : Fin 6) = j then 1 else 0) := by
      simp only [Fin.ext_iff]
    rw [hd]
    exact (polyLe_update a j ⟨i, hi⟩).congr fun x => by simp [cat6, vget, hi]
  · exact (PolyLe.const _ (vget b (i - 6))).congr fun x => by simp [cat6, hi]

/-- `Φ` reads its planes `q` at `0, …, 2ℓ + 1` only. -/
theorem sparsePhi_congr (ℓ : ℕ) (α : F) (x y : ℕ → F) {q q' : ℕ → F} (h : ∀ p ≤ 2 * ℓ + 1, q p = q' p) :
    sparsePhi ℓ α x y q = sparsePhi ℓ α x y q' := by
  unfold sparsePhi
  rw [h (2 * ℓ) (by omega), h (2 * ℓ + 1) le_rfl]
  congr 1
  refine Finset.prod_congr rfl fun j hj => ?_
  have := Finset.mem_range.mp hj
  rw [h j (by omega), h (ℓ + j) (by omega)]

theorem vget_val {N : ℕ} (u : Fin N → F) (q : Fin N) : vget u q.val = u q := by
  simp [vget]

theorem vget_fun {N : ℕ} (g : ℕ → F) {p : ℕ} (hp : p < N) : vget (fun q : Fin N => g q.val) p = g p := by
  simp [vget, hp]

theorem sum_range_vget_sub {n N : ℕ} (hN : N = 2 ^ n) (ζ : Fin n → F) (u w : Fin N → F) :
    ∑ b ∈ Finset.range (2 ^ n), eqAt ζ b * (vget u b - vget w b) =
      ∑ q : Fin N, eqAt ζ q.val * u q - ∑ q : Fin N, eqAt ζ q.val * w q := by
  subst hN
  rw [← Finset.sum_sub_distrib, Finset.sum_range]
  refine Finset.sum_congr rfl fun q _ => ?_
  rw [vget_val, vget_val, mul_sub]

theorem exists_vget_ne {n N : ℕ} (hN : N = 2 ^ n) {u w : Fin N → F} (h : u ≠ w) :
    ∃ b < 2 ^ n, vget u b - vget w b ≠ 0 := by
  subst hN
  by_contra hall
  push Not at hall
  exact h (funext fun q => by simpa [vget_val, sub_eq_zero] using hall q.val q.isLt)

theorem sum_range_mul {M : Type} [AddCommMonoid M] (m n : ℕ) (f : ℕ → M) :
    ∑ a ∈ Finset.range (m * n), f a = ∑ i ∈ Finset.range n, ∑ j ∈ Finset.range m, f (j + m * i) := by
  induction n with
  | zero => simp
  | succ n ih =>
    rw [Nat.mul_succ, Finset.sum_range_add, ih, Finset.sum_range_succ]
    congr 1
    exact Finset.sum_congr rfl fun j _ => by rw [add_comm]

theorem lt_two_pow_pLog (sh : SparseShape) {p : ℕ} (hp : p ≤ 2 * sh.ell + 1) : p < 2 ^ sh.pLog :=
  lt_of_le_of_lt hp Nat.lt_log2_self

/-- A registered claim's point is zero past the cells, in the padding as past `mReg`. -/
theorem coord_eq_zero {sh : SparseShape} (c : RegClaim F sh) {j : ℕ} (hj : sh.nu + sh.pLog + sh.tLog ≤ j) :
    c.coord j = 0 := by
  simp only [RegClaim.coord, vget]
  rw [ite_eq_right (by omega), ite_eq_right (by omega), dite_eq_right (by omega)]

/-- Coordinates past `M` that are zero don't change `eqAt` on `{0, …, 2^M - 1}`. -/
theorem eqAt_cut {N : ℕ} (M : ℕ) (hM : M ≤ N) (g : ℕ → F) (hg : ∀ j, M ≤ j → g j = 0) (a : ℕ)
    (ha : a < 2 ^ M) : eqAt (fun j : Fin N => g j.val) a = eqAt (fun j : Fin M => g j.val) a := by
  have h := eqAt_split_at M (N - M) (by omega) (fun j : Fin N => g j.val) a 0 ha
  rw [mul_zero, add_zero] at h
  have h1 : eqAt (fun j : Fin (N - M) => g (M + j.val)) 0 = 1 := by
    unfold eqAt
    refine Finset.prod_eq_one fun j _ => ?_
    simp [hg (M + j.val) (by omega)]
  rw [h]
  exact (congrArg (eqAt (fun j : Fin M => g j.val) a * ·) h1).trans (mul_one _)

/-- A registered claim's skip weights and point are `eqAt` of its coordinates. -/
theorem coord_split {sh : SparseShape} (c : RegClaim F sh) (a : ℕ) (ha : a < 2 ^ sh.mReg) :
    eqAt (fun j : Fin 6 => c.coord j.val) (a % 64) *
        eqAt (fun i : Fin (sh.mReg - 6) => c.coord (i.val + 6)) (a / 64) =
      eqAt (fun j : Fin sh.mReg => c.coord j.val) a := by
  have h := eqAt_split_at 6 (sh.mReg - 6) rfl (fun j : Fin (6 + (sh.mReg - 6)) => c.coord j.val) (a % 64)
    (a / 64) (Nat.mod_lt _ (by norm_num))
  rw [show a % 64 + 2 ^ 6 * (a / 64) = a by
    rw [show (2 : ℕ) ^ 6 = 64 by norm_num]; exact Nat.mod_add_div a 64] at h
  rw [← eqAt_cut (N := 6 + (sh.mReg - 6)) sh.mReg (by omega) (fun j => c.coord j)
    (fun j hj => coord_eq_zero c (by unfold SparseShape.mReg at hj; omega)) a ha, h]
  congr 1
  congr 1
  funext i
  exact congrArg _ (Nat.add_comm _ _)

/-- Past the cells a registered claim's point weighs nothing: a row of the padding has a bit set where the point is
zero (`coord_eq_zero`). -/
theorem eqAt_coord_high {sh : SparseShape} (c : RegClaim F sh) {a : ℕ}
    (ha : 2 ^ (sh.nu + sh.pLog + sh.tLog) ≤ a) (ha' : a < 2 ^ sh.mReg) :
    eqAt (fun j : Fin sh.mReg => c.coord j.val) a = 0 := by
  have hCm : sh.nu + sh.pLog + sh.tLog ≤ sh.mReg := by unfold SparseShape.mReg; omega
  have hpos : 0 < 2 ^ (sh.nu + sh.pLog + sh.tLog) := Nat.pos_of_ne_zero (pow_ne_zero _ two_ne_zero)
  have h := eqAt_split_at (sh.nu + sh.pLog + sh.tLog) (sh.mReg - (sh.nu + sh.pLog + sh.tLog)) (by omega)
    (fun j : Fin sh.mReg => c.coord j.val) (a % 2 ^ (sh.nu + sh.pLog + sh.tLog))
    (a / 2 ^ (sh.nu + sh.pLog + sh.tLog)) (Nat.mod_lt _ hpos)
  rw [Nat.mod_add_div] at h
  rw [h]
  refine mul_eq_zero_of_right _ ?_
  have hb0 : 0 < a / 2 ^ (sh.nu + sh.pLog + sh.tLog) := Nat.div_pos ha hpos
  have hlt : a / 2 ^ (sh.nu + sh.pLog + sh.tLog) < 2 ^ (sh.mReg - (sh.nu + sh.pLog + sh.tLog)) := by
    rw [Nat.div_lt_iff_lt_mul hpos, ← pow_add, Nat.sub_add_cancel hCm]
    exact ha'
  have hbit : ∃ j : Fin (sh.mReg - (sh.nu + sh.pLog + sh.tLog)),
      (a / 2 ^ (sh.nu + sh.pLog + sh.tLog)).testBit j.val = true := by
    by_contra hall
    push Not at hall
    refine (Nat.pos_iff_ne_zero.1 hb0) (Nat.eq_of_testBit_eq fun i => ?_)
    rw [Nat.zero_testBit]
    by_cases hi : i < sh.mReg - (sh.nu + sh.pLog + sh.tLog)
    · simpa using hall ⟨i, hi⟩
    · exact Nat.testBit_eq_false_of_lt (lt_of_lt_of_le hlt (Nat.pow_le_pow_right (by norm_num) (by omega)))
  obtain ⟨j, hj⟩ := hbit
  unfold eqAt
  refine Finset.prod_eq_zero (Finset.mem_univ j) ?_
  rw [if_pos hj]
  exact coord_eq_zero c (Nat.le_add_right _ _)

/-- A registered claim's point, split into the entries', the planes' and the type's. -/
theorem coord_eqAt {sh : SparseShape} (c : RegClaim F sh) (k p t : ℕ) (hk : k < 2 ^ sh.nu)
    (hp : p < 2 ^ sh.pLog) (ht : t < 2 ^ sh.tLog) :
    eqAt (fun j : Fin sh.mReg => c.coord j.val) (k + 2 ^ sh.nu * (p + 2 ^ sh.pLog * t)) =
      eqAt c.rk k * (eqAt c.zetaP p * eqAt c.rho t) := by
  have key : ∀ {a m b n : ℕ}, a < m → b < n → a + m * b < m * n := by
    intro a m b n ha hb
    calc a + m * b < m + m * b := by omega
      _ = m * (b + 1) := by ring
      _ ≤ m * n := Nat.mul_le_mul (le_refl m) (by omega)
  rw [eqAt_cut (N := sh.mReg) (sh.nu + (sh.pLog + sh.tLog)) (by unfold SparseShape.mReg; omega) c.coord
      (fun j hj => coord_eq_zero c (by omega)) (k + 2 ^ sh.nu * (p + 2 ^ sh.pLog * t))
      (by rw [pow_add, pow_add]; exact key hk (key hp ht)),
    eqAt_split_at sh.nu (sh.pLog + sh.tLog) rfl _ k _ hk, eqAt_split_at sh.pLog sh.tLog rfl _ p t hp]
  congr 1
  · congr 1; funext j; simp [RegClaim.coord, vget]
  · congr 1
    · congr 1; funext j; simp [RegClaim.coord, vget]
    · congr 1; funext j; simp [RegClaim.coord, vget]

end Vectors

/-! ## The coin bounds -/

section Coins

variable {F : Type} [Field F] [Fintype F] {β : Type}

theorem value_bind_le_of_forall {α : Type} (p : β → Prop) (g : Game α) (f : α → Game β) (ε : ℝ≥0∞)
    (hf : ∀ a, value p (f a) ≤ ε) : value p (g >>= f) ≤ ε := by
  induction g with
  | ret a => exact hf a
  | send M k ih =>
      show value p (.send M fun m => k m >>= f) ≤ ε
      rw [value_send]
      exact iSup_le ih
  | coin C k ih =>
      show value p (.coin C fun c => k c >>= f) ≤ ε
      rw [value_coin]
      exact (avg_mono ih).trans (avg_const ε).le

theorem value_bind_le' {α : Type} (p : β → Prop) (g : Game α) (f : α → Game β) (D : α → Prop) (ε : ℝ≥0∞)
    (hf : ∀ a, D a → value p (f a) ≤ ε) : value p (g >>= f) ≤ value (fun a => ¬ D a) g + ε :=
  value_bind_le g f p D ε hf

theorem value_pure_eq_zero {α : Type} {p : α → Prop} {a : α} (h : ¬ p a) : value p (pure a) = 0 := by
  classical
  rw [pure_eq, value_ret, ite_eq_right h]

theorem value_draw_bind_le (p : β → Prop) (C : Type) [Fintype C] [Nonempty C] (f : C → Game β) (bad : C → Prop)
    (ε ε' : ℝ≥0∞) (hbad : prCoin bad ≤ ε) (hgood : ∀ c, ¬ bad c → value p (f c) ≤ ε') :
    value p (draw C >>= f) ≤ ε + ε' :=
  value_coin_le p C f bad ε ε' hbad hgood

omit [Field F] in
theorem prCoin_eq_le (c₀ : F) {bad : F → Prop} (h : ∀ c, bad c → c = c₀) :
    prCoin bad ≤ 1 / Fintype.card F := by
  classical
  refine (prCoin_mono h).trans ?_
  rw [prCoin_eq_card]
  gcongr
  exact_mod_cast Finset.card_le_one.mpr fun a ha b hb => by
    simp only [Finset.mem_filter] at ha hb
    exact ha.2.trans hb.2.symm

/-- **Schwartz–Zippel per `drawFin` coin**: a nonzero vector's multilinear extension at `n` fresh coins is zero with
probability at most `n/|F|`. -/
theorem value_drawFin_le (p : β → Prop) : ∀ (n : ℕ) (u : ℕ → F), (∃ b < 2 ^ n, u b ≠ 0) →
    ∀ (k : (Fin n → F) → Game β) (ε : ℝ≥0∞),
      (∀ ζ, ∑ b ∈ Finset.range (2 ^ n), eqAt ζ b * u b ≠ 0 → value p (k ζ) ≤ ε) →
        value p (drawFin n >>= k) ≤ (n : ℝ≥0∞) / Fintype.card F + ε
  | 0, u, hu, k, ε, hk => by
      obtain ⟨b, hb, hub⟩ := hu
      obtain rfl : b = 0 := by rw [pow_zero] at hb; omega
      have h0 : ∑ b ∈ Finset.range (2 ^ 0), eqAt (Fin.elim0 : Fin 0 → F) b * u b ≠ 0 := by
        simpa [eqAt] using hub
      show value p (k Fin.elim0) ≤ _
      exact (hk _ h0).trans le_add_self
  | n + 1, u, hu, k, ε, hk => by
      classical
      obtain ⟨b₀, hb₀, hub₀⟩ := hu
      have hsum : ∀ (c : F) (ζ : Fin n → F),
          ∑ b ∈ Finset.range (2 ^ (n + 1)), eqAt (Fin.cons c ζ : Fin (n + 1) → F) b * u b =
            ∑ b ∈ Finset.range (2 ^ n), eqAt ζ b * ((1 + c) * u (2 * b) + c * u (2 * b + 1)) := by
        intro c ζ
        rw [sum_range_pairs]
        refine Finset.sum_congr rfl fun b _ => ?_
        rw [eqAt_two_mul, eqAt_two_mul_add_one, Fin.cons_zero, Fin.tail_cons]
        ring
      -- the one coin that folds the pair at `b₀ / 2` to zero
      set a := u (2 * (b₀ / 2)) with ha
      set a' := u (2 * (b₀ / 2) + 1) with ha'
      have hb₂ : b₀ / 2 < 2 ^ n := by rw [pow_succ] at hb₀; omega
      have hbad : ∀ c, (∀ b < 2 ^ n, (1 + c) * u (2 * b) + c * u (2 * b + 1) = 0) → c = -a / (a + a') := by
        intro c hc
        have h1 := hc (b₀ / 2) hb₂
        rw [← ha, ← ha'] at h1
        have hne : a + a' ≠ 0 := by
          intro hz
          have ha0 : a = 0 := by linear_combination h1 - c * hz
          have ha'0 : a' = 0 := by linear_combination hz - ha0
          rcases Nat.mod_two_eq_zero_or_one b₀ with hm | hm
          · exact hub₀ (by rw [show b₀ = 2 * (b₀ / 2) by omega]; exact ha0)
          · exact hub₀ (by rw [show b₀ = 2 * (b₀ / 2) + 1 by omega]; exact ha'0)
        rw [eq_div_iff hne]
        linear_combination h1
      have hgame : drawFin (n + 1) >>= k = (draw F >>= fun c => drawFin n >>= fun ζ => k (Fin.cons c ζ)) := by
        rw [drawFin.eq_2]
        simp only [bind_eq, bind_assoc, pure_eq, bind_ret]
      rw [hgame]
      refine (value_draw_bind_le p F _ (fun c => ∀ b < 2 ^ n, (1 + c) * u (2 * b) + c * u (2 * b + 1) = 0)
        (1 / Fintype.card F) ((n : ℝ≥0∞) / Fintype.card F + ε) (prCoin_eq_le _ hbad) fun c hc => ?_).trans
          (le_of_eq ?_)
      · push Not at hc
        exact value_drawFin_le p n _ hc _ ε fun ζ hζ => hk _ (by rw [hsum]; exact hζ)
      · rw [← add_assoc, ENNReal.div_add_div_same, Nat.cast_succ, add_comm (1 : ℝ≥0∞)]

/-- `value_drawFin_le` with a disjunct `E` that, when it holds, bounds the rest by itself. -/
theorem value_drawFin_le_or (p : β → Prop) (E : Prop) (n : ℕ) (u : ℕ → F) (hu : E ∨ ∃ b < 2 ^ n, u b ≠ 0)
    (k : (Fin n → F) → Game β) (ε : ℝ≥0∞)
    (hk : ∀ ζ, E ∨ ∑ b ∈ Finset.range (2 ^ n), eqAt ζ b * u b ≠ 0 → value p (k ζ) ≤ ε) :
    value p (drawFin n >>= k) ≤ (n : ℝ≥0∞) / Fintype.card F + ε := by
  by_cases hE : E
  · exact (value_bind_le_of_forall p _ k ε fun ζ => hk ζ (Or.inl hE)).trans le_add_self
  · exact value_drawFin_le p n u (hu.resolve_left hE) k ε fun ζ h => hk ζ (Or.inr h)

/-- A round's polynomial `c₀ + c₁X + Σ_i c_{i+2}X^{i+2}` with `c₁ = claim + Σ_i c_{i+2}` (`degNext`). -/
noncomputable def degPoly {k : ℕ} (claim : F) (c : F × (Fin k → F)) : F[X] :=
  C c.1 + C (claim + ∑ i, c.2 i) * X + ∑ i, C (c.2 i) * X ^ (i.val + 2)

omit [Fintype F] in
theorem degNext_eq {k : ℕ} (claim : F) (c : F × (Fin k → F)) (r : F) :
    degNext claim c r = (degPoly claim c).eval r := by
  simp only [degNext, degPoly, eval_add, eval_mul, eval_C, eval_X, eval_pow, eval_finsetSum]

omit [Fintype F] in
theorem degPoly_natDegree_le {k : ℕ} (claim : F) (c : F × (Fin k → F)) : (degPoly claim c).natDegree ≤ k + 1 := by
  unfold degPoly
  refine (natDegree_add_le _ _).trans (max_le ((natDegree_add_le _ _).trans (max_le ?_ ?_)) ?_)
  · exact (natDegree_C _).le.trans (Nat.zero_le _)
  · exact ((natDegree_C_mul_le _ _).trans natDegree_X_le).trans (by omega)
  · exact natDegree_sum_le_of_forall_le _ _ fun i _ => (natDegree_C_mul_X_pow_le _ _).trans (by omega)

omit [Fintype F] in
/-- The sum rule (C1): `g(0) + g(1) = claim` in characteristic 2. -/
theorem degPoly_sum [CharP F 2] {k : ℕ} (claim : F) (c : F × (Fin k → F)) :
    (degPoly claim c).eval 0 + (degPoly claim c).eval 1 = claim := by
  rw [← degNext_eq, ← degNext_eq]
  simp only [degNext, mul_zero, add_zero, mul_one, one_pow, ne_eq, add_eq_zero, OfNat.ofNat_ne_zero, and_false,
    not_false_eq_true, zero_pow, Finset.sum_const_zero]
  linear_combination (c.1 + ∑ i, c.2 i) * Rounds.two_eq_zero (F := F)

/-- **`degRounds` on a false claim**: if `claim` is not `G`'s sum over the cube, the rounds end at a claim equal to `G`
at their coins with probability at most `n(k + 1)/|F|`. -/
theorem value_degRounds_le [CharP F 2] (k : ℕ) : ∀ (n : ℕ) (G : (Fin n → F) → F), DegLe (k + 1) G →
    ∀ claim : F, claim ≠ cube G →
      value (fun o : (Fin n → F) × F => o.2 = G o.1) (degRounds k n claim) ≤
        ((n * (k + 1) : ℕ) : ℝ≥0∞) / Fintype.card F
  | 0, G, _, claim, hc => by
      rw [degRounds.eq_1]
      refine (value_pure_eq_zero ?_).le.trans zero_le
      intro h
      apply hc
      unfold cube
      rw [pow_zero, Finset.sum_range_one]
      exact h.trans (congrArg G (funext fun j => j.elim0))
  | n + 1, G, hG, claim, hc => by
      classical
      choose P hPd hPe using fun b : ℕ => hG.head (bitsOf n b)
      have hTe : ∀ x, (∑ b ∈ Finset.range (2 ^ n), P b).eval x =
          ∑ b ∈ Finset.range (2 ^ n), G (Fin.cons x (bitsOf n b)) := fun x => by
        rw [eval_finsetSum]; exact Finset.sum_congr rfl fun b _ => (hPe b x).symm
      have hTd : (∑ b ∈ Finset.range (2 ^ n), P b).natDegree ≤ k + 1 :=
        natDegree_sum_le_of_forall_le _ _ fun b _ => hPd b
      have hle : ((k + 1 : ℕ) : ℝ≥0∞) / Fintype.card F + ((n * (k + 1) : ℕ) : ℝ≥0∞) / Fintype.card F =
          (((n + 1) * (k + 1) : ℕ) : ℝ≥0∞) / Fintype.card F := by
        rw [ENNReal.div_add_div_same]; congr 1; push_cast; ring
      rw [degRounds.eq_2, ← hle]
      refine value_bind_le_of_forall _ _ _ _ fun c => ?_
      refine value_draw_bind_le _ F _ (fun r => (degPoly claim c).eval r = (∑ b ∈ Finset.range (2 ^ n), P b).eval r)
        _ _ ?_ fun r hr => ?_
      · refine Game.prCoin_eval_eq_eval_le _ _ (fun hPT => hc ?_) (k + 1) (degPoly_natDegree_le _ _) hTd
        rw [cube_succ, ← degPoly_sum claim c, hPT, hTe, hTe, Finset.sum_add_distrib]
      · refine (value_bind_le' _ _ _ (fun x => x.2 ≠ G (Fin.cons r x.1)) 0 ?_).trans ?_
        · rintro ⟨rs, out⟩ hx
          exact (value_pure_eq_zero hx).le
        · rw [add_zero]
          refine (value_mono (fun x hx => not_not.mp hx) _).trans ?_
          refine value_degRounds_le k n _ (hG.cons r) _ fun hcl => hr ?_
          rw [← degNext_eq, hcl, hTe]
          rfl

/-- `value_degRounds_le` at a bind, with a disjunct `E` that, when it holds, bounds the rest by itself. -/
theorem value_degRounds_bind_le [CharP F 2] (p : β → Prop) (E : Prop) (k n : ℕ) (G : (Fin n → F) → F)
    (hG : DegLe (k + 1) G) (claim : F) (hc : E ∨ claim ≠ cube G) (f : (Fin n → F) × F → Game β) (ε : ℝ≥0∞)
    (hf : ∀ rs out, E ∨ out ≠ G rs → value p (f (rs, out)) ≤ ε) :
    value p (degRounds k n claim >>= f) ≤ ((n * (k + 1) : ℕ) : ℝ≥0∞) / Fintype.card F + ε := by
  by_cases hE : E
  · exact (value_bind_le_of_forall p _ f ε fun x => hf x.1 x.2 (Or.inl hE)).trans le_add_self
  · refine (value_bind_le' p _ f (fun x => x.2 ≠ G x.1) ε fun x hx => hf x.1 x.2 (Or.inr hx)).trans
      (add_le_add ?_ le_rfl)
    exact (value_mono (fun x hx => not_not.mp hx) _).trans
      (value_degRounds_le k n G hG claim (hc.resolve_left hE))

end Coins

end FlockC3.Fork.SparseCount
