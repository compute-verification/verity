import Mathlib

/-!
# Relative security for matmul PoUW: the definition, the masking algebra, the reduction's shape

Neekon Vafa's definition (`note:20261015T1810Z-draft-relative-security-def`) in an abstract machine model, so that the
construction and proof workers' results land here as theorems or as named hypotheses.

* The machine model (`Machine`, `MatMachine`): algorithms from inputs to outputs, their semantics with the coins as
  part of the input, and an abstract cost `time` with axioms: nonnegativity, monotonicity (running `f` first, or beside
  another algorithm, costs at least `f`), subadditivity of sequential and side-by-side composition, free routing, and
  the cost of a linear map (an `a × b` by `b × c` product costs at most `κ a b c`).
* The definition (`Protocol`, `Complete`, `Secure`, `Efficient`, `RelativelySecure`) with `ε`, `δ`, `γ`, `η`;
  probabilities are uniform counts over the finite σ and coins (`prob`).
* The masking algebra, proved over any ring (`mask_peel`, `lowRank_peel`), exact over `ZMod (2^32)` and over any
  field, and recovering the integer product from its s32 residue when the entries are below `2^31`
  (`int8_product_exact`); a saturating int8 accumulator breaks the peel (`sat8_breaks_peel`).
* The reduction's shape (`reduction`): run the extractor (M's run and its committed blocks), keep the proof, peel and
  correct; `time_reduction_le` is `time M' ≤ time E + time peel + time correction`, and `secure_of_extraction` gives
  relative security from the extraction step, a named hypothesis (`Extraction`), for full-rank and low-rank masks.
-/

namespace Pouw.RelativeSecurity

open Finset

section Probability

/-- The probability of an event under the uniform distribution on a finite type. -/
noncomputable def prob {Ω : Type*} [Fintype Ω] (p : Ω → Prop) : ℝ := by
  classical exact ((univ.filter p).card : ℝ) / (Fintype.card Ω : ℝ)

theorem prob_nonneg {Ω : Type*} [Fintype Ω] (p : Ω → Prop) : 0 ≤ prob p := by
  unfold prob
  positivity

theorem prob_le_one {Ω : Type*} [Fintype Ω] (p : Ω → Prop) : prob p ≤ 1 := by
  classical
  unfold prob
  rcases Nat.eq_zero_or_pos (Fintype.card Ω) with h | h
  · simp [h]
  · rw [div_le_one (by exact_mod_cast h)]
    exact_mod_cast card_filter_le _ _

theorem prob_mono {Ω : Type*} [Fintype Ω] {p q : Ω → Prop} (h : ∀ ω, p ω → q ω) : prob p ≤ prob q := by
  classical
  unfold prob
  apply div_le_div_of_nonneg_right _ (by positivity)
  exact_mod_cast card_le_card fun ω hω => by
    simp only [mem_filter, mem_univ, true_and] at hω ⊢
    exact h ω hω

end Probability

section Machine

/-- A machine model: algorithms from inputs `I` to outputs `O` (coins are part of the input), what they compute, and
their cost. The axioms are the properties of `time` the reduction uses. -/
structure Machine where
  /-- The algorithms from `I` to `O`. -/
  Alg : Type → Type → Type
  /-- What an algorithm computes. -/
  eval : ∀ {I O : Type}, Alg I O → I → O
  /-- Its cost (an abstract `time`: priced work, depth, …). -/
  time : ∀ {I O : Type}, Alg I O → ℝ
  /-- Run `f`, then `g` on its output. -/
  seq : ∀ {I O P : Type}, Alg I O → Alg O P → Alg I P
  /-- Run `f` and `g` on the same input. -/
  pair : ∀ {I O P : Type}, Alg I O → Alg I P → Alg I (O × P)
  /-- The projections (routing). -/
  fst : ∀ {I J : Type}, Alg (I × J) I
  snd : ∀ {I J : Type}, Alg (I × J) J
  eval_seq : ∀ {I O P : Type} (f : Alg I O) (g : Alg O P) (x : I), eval (seq f g) x = eval g (eval f x)
  eval_pair : ∀ {I O P : Type} (f : Alg I O) (g : Alg I P) (x : I), eval (pair f g) x = (eval f x, eval g x)
  eval_fst : ∀ {I J : Type} (x : I × J), eval (fst (I := I) (J := J)) x = x.1
  eval_snd : ∀ {I J : Type} (x : I × J), eval (snd (I := I) (J := J)) x = x.2
  /-- Costs are nonnegative. -/
  time_nonneg : ∀ {I O : Type} (f : Alg I O), 0 ≤ time f
  /-- Subadditivity of sequential composition. -/
  time_seq_le : ∀ {I O P : Type} (f : Alg I O) (g : Alg O P), time (seq f g) ≤ time f + time g
  /-- Monotonicity: running `f` first costs at least `f`. (Not `g`: `g`'s worst case may lie outside `f`'s range.) -/
  le_time_seq : ∀ {I O P : Type} (f : Alg I O) (g : Alg O P), time f ≤ time (seq f g)
  /-- Subadditivity of side-by-side composition. -/
  time_pair_le : ∀ {I O P : Type} (f : Alg I O) (g : Alg I P), time (pair f g) ≤ time f + time g
  /-- Monotonicity: running `f` beside `g` costs at least each. -/
  le_time_pair_left : ∀ {I O P : Type} (f : Alg I O) (g : Alg I P), time f ≤ time (pair f g)
  le_time_pair_right : ∀ {I O P : Type} (f : Alg I O) (g : Alg I P), time g ≤ time (pair f g)
  /-- Routing is free. -/
  time_fst : ∀ {I J : Type}, time (fst (I := I) (J := J)) = 0
  time_snd : ∀ {I J : Type}, time (snd (I := I) (J := J)) = 0

/-- `n × m` matrices over `R`, indexed by `Fin`. -/
abbrev Mat (R : Type) (a b : ℕ) : Type := Matrix (Fin a) (Fin b) R

/-- A machine model with matrix arithmetic over a ring `R`: the cost of a linear map, `X ↦ X Y` for an `a × b` by
`b × c` product, is at most `κ a b c` (`κ` the price of a multiply-accumulate), and an entrywise addition or
subtraction of `a × b` matrices costs at most `κadd a b`. -/
structure MatMachine (R : Type) [Ring R] extends Machine where
  κ : ℝ
  κadd : ℝ
  mul : ∀ a b c : ℕ, Alg (Mat R a b × Mat R b c) (Mat R a c)
  add : ∀ a b : ℕ, Alg (Mat R a b × Mat R a b) (Mat R a b)
  sub : ∀ a b : ℕ, Alg (Mat R a b × Mat R a b) (Mat R a b)
  eval_mul : ∀ (a b c : ℕ) (x : Mat R a b × Mat R b c), eval (mul a b c) x = x.1 * x.2
  eval_add : ∀ (a b : ℕ) (x : Mat R a b × Mat R a b), eval (add a b) x = x.1 + x.2
  eval_sub : ∀ (a b : ℕ) (x : Mat R a b × Mat R a b), eval (sub a b) x = x.1 - x.2
  /-- The cost of a linear map. -/
  time_mul_le : ∀ a b c : ℕ, time (mul a b c) ≤ κ * ((a : ℝ) * b * c)
  time_add_le : ∀ a b : ℕ, time (add a b) ≤ κadd * ((a : ℝ) * b)
  time_sub_le : ∀ a b : ℕ, time (sub a b) ≤ κadd * ((a : ℝ) * b)

namespace Machine

variable (Mc : Machine)

/-- Run `f` and `g` on the same input and combine their outputs with `h`. -/
def lift2 {I O P Q : Type} (f : Mc.Alg I O) (g : Mc.Alg I P) (h : Mc.Alg (O × P) Q) : Mc.Alg I Q :=
  Mc.seq (Mc.pair f g) h

theorem eval_lift2 {I O P Q : Type} (f : Mc.Alg I O) (g : Mc.Alg I P) (h : Mc.Alg (O × P) Q) (x : I) :
    Mc.eval (Mc.lift2 f g h) x = Mc.eval h (Mc.eval f x, Mc.eval g x) := by
  rw [lift2, Mc.eval_seq, Mc.eval_pair]

theorem time_lift2_le {I O P Q : Type} (f : Mc.Alg I O) (g : Mc.Alg I P) (h : Mc.Alg (O × P) Q) :
    Mc.time (Mc.lift2 f g h) ≤ Mc.time f + Mc.time g + Mc.time h :=
  (Mc.time_seq_le _ _).trans (by linarith [Mc.time_pair_le f g])

theorem time_seq_eq_zero {I O P : Type} {f : Mc.Alg I O} {g : Mc.Alg O P} (hf : Mc.time f = 0)
    (hg : Mc.time g = 0) : Mc.time (Mc.seq f g) = 0 :=
  le_antisymm ((Mc.time_seq_le f g).trans (by rw [hf, hg, add_zero])) (Mc.time_nonneg _)

end Machine

end Machine

section Definition

/-- A proof-of-useful-work protocol in a machine model, for instances `Inst` with answers `Out`. -/
structure Protocol (Mc : Machine) (Inst Out : Type) where
  /-- The verifier's random strings σ, `λ` bits. -/
  Seed : Type
  [instSeed : Fintype Seed]
  lam : ℕ
  card_seed : Fintype.card Seed = 2 ^ lam
  /-- Solve's coins `r`. -/
  Coins : Type
  [instCoins : Fintype Coins]
  /-- Proofs `π`, at most `10 λ` bits. -/
  Proof : Type
  proofBits : Proof → ℕ
  proofBits_le : ∀ π, proofBits π ≤ 10 * lam
  /-- `C` answers `x` (for matmul: `A · B = C` on the exact grid). -/
  correct : Inst → Out → Prop
  /-- `Solve(σ, x; r) → (π, C)`. -/
  solve : Mc.Alg (Seed × Inst × Coins) (Proof × Out)
  /-- `Verify(σ, π) → b`, deterministic and not priced. -/
  verify : Seed → Proof → Bool

attribute [instance] Protocol.instSeed Protocol.instCoins

variable {Mc : Machine} {Inst Out : Type}

/-- An algorithm with inputs `(σ, x; r)` answers `x` correctly with a proof that verifies, with probability at least
`1 - η` over σ and its coins, for every instance. -/
def SolvesAll (P : Protocol Mc Inst Out) {Ω : Type} [Fintype Ω] (M' : Mc.Alg (P.Seed × Inst × Ω) (P.Proof × Out))
    (η : ℝ) : Prop :=
  ∀ x : Inst, 1 - η ≤ prob fun ω : P.Seed × Ω =>
    P.correct x (Mc.eval M' (ω.1, x, ω.2)).2 ∧ P.verify ω.1 (Mc.eval M' (ω.1, x, ω.2)).1 = true

/-- The probability that an adversary `M(σ; r)` is accepted. -/
noncomputable def acceptance (P : Protocol Mc Inst Out) {Ω : Type} [Fintype Ω] (M : Mc.Alg (P.Seed × Ω) P.Proof) :
    ℝ :=
  prob fun ω : P.Seed × Ω => P.verify ω.1 (Mc.eval M ω) = true

/-- **Completeness**: Solve answers every instance with a proof that verifies, with probability at least `1 - η`. -/
def Complete (P : Protocol Mc Inst Out) (η : ℝ) : Prop :=
  SolvesAll P P.solve η

/-- **Security**: every adversary `M` accepted with probability at least `δ` gives an `M'` that costs at most
`(1 + ε) time M` and answers every instance with a proof that verifies, with probability at least `1 - η`. -/
def Secure (P : Protocol Mc Inst Out) (ε δ η : ℝ) : Prop :=
  ∀ (Ω : Type) [Fintype Ω] (M : Mc.Alg (P.Seed × Ω) P.Proof), δ ≤ acceptance P M →
    ∃ (Ω' : Type) (_ : Fintype Ω') (M' : Mc.Alg (P.Seed × Inst × Ω') (P.Proof × Out)),
      Mc.time M' ≤ (1 + ε) * Mc.time M ∧ SolvesAll P M' η

/-- **Efficiency**: Solve costs at most `(1 + γ)` times the baseline (`NaiveMatmul`). -/
def Efficient (P : Protocol Mc Inst Out) (γ : ℝ) (naive : Mc.Alg Inst Out) : Prop :=
  Mc.time P.solve ≤ (1 + γ) * Mc.time naive

/-- **Relative security** with parameters `ε, δ, γ, η` against a baseline. -/
def RelativelySecure (P : Protocol Mc Inst Out) (ε δ γ η : ℝ) (naive : Mc.Alg Inst Out) : Prop :=
  Complete P η ∧ Secure P ε δ η ∧ Efficient P γ naive

/-- The matmul problem: instances `(A, B)` of `n × n` matrices over `R`, answered by `C = A · B`. -/
def IsMatmul {R : Type} [Ring R] {n : ℕ} {Mc : Machine} (P : Protocol Mc (Mat R n n × Mat R n n) (Mat R n n)) :
    Prop :=
  ∀ x C, P.correct x C ↔ x.1 * x.2 = C

theorem secure_mono (P : Protocol Mc Inst Out) {ε ε' δ δ' η η' : ℝ} (h : Secure P ε δ η) (hε : ε ≤ ε')
    (hδ : δ ≤ δ') (hη : η ≤ η') : Secure P ε' δ' η' := by
  intro Ω _ M hM
  obtain ⟨Ω', _, M', ht, hs⟩ := h Ω M (hδ.trans hM)
  refine ⟨Ω', inferInstance, M', ht.trans ?_, fun x => le_trans (by linarith) (hs x)⟩
  exact mul_le_mul_of_nonneg_right (by linarith) (Mc.time_nonneg M)

theorem complete_mono (P : Protocol Mc Inst Out) {η η' : ℝ} (h : Complete P η) (hη : η ≤ η') : Complete P η' :=
  fun x => le_trans (by linarith) (h x)

end Definition

section Masking

variable {α : Type*} [Ring α] {m k p r r' : Type*} [Fintype k] [Fintype r] [Fintype r']

/-- **The full-rank peel**: `A · B = (A + R)(B + S) − R (B + S) − A · S`, over any ring (in particular any commutative
ring, `ZMod (2^32)` and any field). -/
theorem mask_peel (A R : Matrix m k α) (B S : Matrix k p α) :
    A * B = (A + R) * (B + S) - R * (B + S) - A * S := by
  rw [Matrix.add_mul, Matrix.mul_add]
  abel

/-- **The low-rank peel**: with masks `U Vᵀ` (rank `r`) and `W Zᵀ` (rank `r'`), the masked product minus the two
corrections, each a product through an inner dimension `r` or `r'`, is `A · B`:
`(A + U Vᵀ)(B + W Zᵀ) − U (Vᵀ (B + W Zᵀ)) − (A W) Zᵀ = A · B`. -/
theorem lowRank_peel (A : Matrix m k α) (B : Matrix k p α) (U : Matrix m r α) (Vt : Matrix r k α)
    (W : Matrix k r' α) (Zt : Matrix r' p α) :
    (A + U * Vt) * (B + W * Zt) - U * (Vt * (B + W * Zt)) - (A * W) * Zt = A * B := by
  rw [Matrix.add_mul, Matrix.mul_add A, Matrix.mul_assoc U, Matrix.mul_assoc A]
  abel

/-- The full-rank peel in s32 arithmetic (`ZMod (2^32)`), where it is exact. -/
theorem mask_peel_zmod32 (A R : Matrix m k (ZMod (2 ^ 32))) (B S : Matrix k p (ZMod (2 ^ 32))) :
    A * B = (A + R) * (B + S) - R * (B + S) - A * S :=
  mask_peel A R B S

/-- The full-rank peel over a field, where it is also exact. -/
theorem mask_peel_field {F : Type*} [Field F] (A R : Matrix m k F) (B S : Matrix k p F) :
    A * B = (A + R) * (B + S) - R * (B + S) - A * S :=
  mask_peel A R B S

end Masking

section Exactness

/-- An integer is its s32 residue's signed representative when `-2^31 < z ≤ 2^31`. -/
theorem valMinAbs_intCast_of_bounds {z : ℤ} (hlo : -2 ^ 31 < z) (hhi : z ≤ 2 ^ 31) :
    ((z : ZMod (2 ^ 32))).valMinAbs = z := by
  rw [ZMod.valMinAbs_spec]
  refine ⟨rfl, ?_, ?_⟩ <;> push_cast <;> omega

variable {m k p : Type*} [Fintype k]

/-- The full-rank peel of integer matrices `A`, `B` computed in s32 (`ZMod (2^32)`) with masks `R`, `S`:
`(A + R)(B + S) − R (B + S) − A · S`. -/
def peel32 (A : Matrix m k ℤ) (B : Matrix k p ℤ) (R : Matrix m k (ZMod (2 ^ 32)))
    (S : Matrix k p (ZMod (2 ^ 32))) : Matrix m p (ZMod (2 ^ 32)) :=
  let A' := A.map (Int.castRingHom (ZMod (2 ^ 32)))
  let B' := B.map (Int.castRingHom (ZMod (2 ^ 32)))
  (A' + R) * (B' + S) - R * (B' + S) - A' * S

/-- **s32 recovers the integer product**: for integer matrices whose product has entries in `(-2^31, 2^31]`, the
peel computed in s32, with any masks, has the integer product as its signed representative. -/
theorem s32_peel_recovers (A : Matrix m k ℤ) (B : Matrix k p ℤ) (R : Matrix m k (ZMod (2 ^ 32)))
    (S : Matrix k p (ZMod (2 ^ 32)))
    (hb : ∀ i j, -2 ^ 31 < (A * B) i j ∧ (A * B) i j ≤ 2 ^ 31) (i : m) (j : p) :
    (peel32 A B R S i j).valMinAbs = (A * B) i j := by
  have h : peel32 A B R S = (A * B).map (Int.castRingHom (ZMod (2 ^ 32))) := by
    rw [Matrix.map_mul]
    exact (mask_peel _ R _ S).symm
  rw [h, Matrix.map_apply]
  exact valMinAbs_intCast_of_bounds (hb i j).1 (hb i j).2

/-- int8 by int8 products of inner dimension at most `2^17` lie in `(-2^31, 2^31]`, so s32 accumulation is exact. -/
theorem int8_product_bounds (A : Matrix m k ℤ) (B : Matrix k p ℤ) (hk : Fintype.card k ≤ 2 ^ 17)
    (hA : ∀ i l, -128 ≤ A i l ∧ A i l ≤ 127) (hB : ∀ l j, -128 ≤ B l j ∧ B l j ≤ 127) (i : m) (j : p) :
    -2 ^ 31 < (A * B) i j ∧ (A * B) i j ≤ 2 ^ 31 := by
  have hterm : ∀ l, -16256 ≤ A i l * B l j ∧ A i l * B l j ≤ 16384 := by
    intro l
    obtain ⟨a1, a2⟩ := hA i l
    obtain ⟨b1, b2⟩ := hB l j
    constructor <;> nlinarith
  have hk' : (Fintype.card k : ℤ) ≤ 2 ^ 17 := by exact_mod_cast hk
  rw [Matrix.mul_apply]
  constructor
  · have : ∑ l, (-16256 : ℤ) ≤ ∑ l, A i l * B l j := Finset.sum_le_sum fun l _ => (hterm l).1
    simp only [Finset.sum_const, Finset.card_univ, nsmul_eq_mul] at this
    linarith
  · have : ∑ l, A i l * B l j ≤ ∑ l, (16384 : ℤ) := Finset.sum_le_sum fun l _ => (hterm l).2
    simp only [Finset.sum_const, Finset.card_univ, nsmul_eq_mul] at this
    linarith

/-- **int8 with s32 accumulation is exact** for inner dimension at most `2^17`: the s32 peel of int8 matrices, with
any masks, recovers the integer product. -/
theorem int8_product_exact (A : Matrix m k ℤ) (B : Matrix k p ℤ) (hk : Fintype.card k ≤ 2 ^ 17)
    (hA : ∀ i l, -128 ≤ A i l ∧ A i l ≤ 127) (hB : ∀ l j, -128 ≤ B l j ∧ B l j ≤ 127)
    (R : Matrix m k (ZMod (2 ^ 32))) (S : Matrix k p (ZMod (2 ^ 32))) (i : m) (j : p) :
    (peel32 A B R S i j).valMinAbs = (A * B) i j :=
  s32_peel_recovers A B R S (int8_product_bounds A B hk hA hB) i j

/-- Saturating int8 arithmetic (clamped to `[-128, 127]`), the kind of non-exact accumulator the peel needs to avoid. -/
def sat8 (z : ℤ) : ℤ := max (-128) (min 127 z)

/-- **A non-exact arithmetic breaks the peel**: masking `a = 100` by `r = 100` in saturating int8 and peeling the
mask off gives `27`, not `100`. -/
theorem sat8_breaks_peel : sat8 (sat8 (100 + 100) - 100) ≠ 100 := by
  decide

end Exactness

end Pouw.RelativeSecurity
