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

/-! Cost bounds in goal-directed form, for `apply_rules`. -/

theorem time_seq_le_of {I O P : Type} {f : Mc.Alg I O} {g : Mc.Alg O P} {a b : ℝ} (hf : Mc.time f ≤ a)
    (hg : Mc.time g ≤ b) : Mc.time (Mc.seq f g) ≤ a + b :=
  (Mc.time_seq_le f g).trans (add_le_add hf hg)

theorem time_pair_le_of {I O P : Type} {f : Mc.Alg I O} {g : Mc.Alg I P} {a b : ℝ} (hf : Mc.time f ≤ a)
    (hg : Mc.time g ≤ b) : Mc.time (Mc.pair f g) ≤ a + b :=
  (Mc.time_pair_le f g).trans (add_le_add hf hg)

theorem time_fst_le {I J : Type} : Mc.time (Mc.fst (I := I) (J := J)) ≤ 0 := Mc.time_fst.le

theorem time_snd_le {I J : Type} : Mc.time (Mc.snd (I := I) (J := J)) ≤ 0 := Mc.time_snd.le

end Machine

/-- The axioms are satisfiable with nontrivial prices: algorithms are functions annotated with a cost, composition
adds costs, routing is free, and an `a × b` by `b × c` product costs exactly `a b c`. (This model has no floor: any
function is an algorithm. A floor needs a syntactic model, RS-MODEL's.) -/
def costed : Machine where
  Alg I O := (I → O) × ℕ
  eval f := f.1
  time f := f.2
  seq f g := (g.1 ∘ f.1, f.2 + g.2)
  pair f g := (fun x => (f.1 x, g.1 x), f.2 + g.2)
  fst := (Prod.fst, 0)
  snd := (Prod.snd, 0)
  eval_seq _ _ _ := rfl
  eval_pair _ _ _ := rfl
  eval_fst _ := rfl
  eval_snd _ := rfl
  time_nonneg _ := Nat.cast_nonneg _
  time_seq_le _ _ := by push_cast; rfl
  le_time_seq _ g := by push_cast; linarith [Nat.cast_nonneg (α := ℝ) g.2]
  time_pair_le _ _ := by push_cast; rfl
  le_time_pair_left _ g := by push_cast; linarith [Nat.cast_nonneg (α := ℝ) g.2]
  le_time_pair_right f _ := by push_cast; linarith [Nat.cast_nonneg (α := ℝ) f.2]
  time_fst := by simp
  time_snd := by simp

/-- `costed` with matrix arithmetic over a ring: `κ = κadd = 1`. -/
def costedMat (R : Type) [Ring R] : MatMachine R where
  toMachine := costed
  κ := 1
  κadd := 1
  mul a b c := (fun x => x.1 * x.2, a * b * c)
  add a b := (fun x => x.1 + x.2, a * b)
  sub a b := (fun x => x.1 - x.2, a * b)
  eval_mul _ _ _ _ := rfl
  eval_add _ _ _ := rfl
  eval_sub _ _ _ := rfl
  time_mul_le a b c := by simp [costed]
  time_add_le a b := by simp [costed]
  time_sub_le a b := by simp [costed]

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

section Reduction

namespace Machine

variable (Mc : Machine)

/-- **The reduction** `M' = M ++ Q`: run the extractor `E` (M's own program on the masked instance; its proof and the
blocks read off its wires), keep the proof, and run the peel and then the correction on the blocks. -/
def reduction {I Pf Bl Mid O : Type} (E : Mc.Alg I (Pf × Bl)) (peel : Mc.Alg Bl Mid) (corr : Mc.Alg Mid O) :
    Mc.Alg I (Pf × O) :=
  Mc.seq E (Mc.pair Mc.fst (Mc.seq Mc.snd (Mc.seq peel corr)))

theorem eval_reduction {I Pf Bl Mid O : Type} (E : Mc.Alg I (Pf × Bl)) (peel : Mc.Alg Bl Mid)
    (corr : Mc.Alg Mid O) (x : I) :
    Mc.eval (Mc.reduction E peel corr) x = ((Mc.eval E x).1, Mc.eval corr (Mc.eval peel (Mc.eval E x).2)) := by
  simp only [reduction, Mc.eval_seq, Mc.eval_pair, Mc.eval_fst, Mc.eval_snd]

/-- The reduction costs at most the extractor, the peel and the correction. -/
theorem time_reduction_le {I Pf Bl Mid O : Type} (E : Mc.Alg I (Pf × Bl)) (peel : Mc.Alg Bl Mid)
    (corr : Mc.Alg Mid O) : Mc.time (Mc.reduction E peel corr) ≤ Mc.time E + Mc.time peel + Mc.time corr := by
  have h1 := Mc.time_seq_le E (Mc.pair Mc.fst (Mc.seq Mc.snd (Mc.seq peel corr)))
  have h2 := Mc.time_pair_le (Mc.fst (I := Pf) (J := Bl)) (Mc.seq Mc.snd (Mc.seq peel corr))
  have h3 := Mc.time_seq_le (Mc.snd (I := Pf) (J := Bl)) (Mc.seq peel corr)
  have h4 := Mc.time_seq_le peel corr
  have h5 := Mc.time_fst (I := Pf) (J := Bl)
  have h6 := Mc.time_snd (I := Pf) (J := Bl)
  unfold reduction
  linarith

/-- **The cost inequality**: with an extractor that costs at most `M`,
`time M' ≤ time M + time peel + time correction`. -/
theorem time_reduction_le_of_le {I Pf Bl Mid O J : Type} {E : Mc.Alg I (Pf × Bl)} {M : Mc.Alg J Pf}
    (hE : Mc.time E ≤ Mc.time M) (peel : Mc.Alg Bl Mid) (corr : Mc.Alg Mid O) :
    Mc.time (Mc.reduction E peel corr) ≤ Mc.time M + Mc.time peel + Mc.time corr := by
  linarith [Mc.time_reduction_le E peel corr]

end Machine

variable {Mc : Machine} {Inst Out : Type}

/-- **Extraction** (a named hypothesis, RS-EXT's): from an adversary `M`, an extractor `E` that costs at most `M` (it
runs M's own program on the masked instance and reads blocks off its wires) and, for every instance `x`, with
probability at least `1 - η` over σ and its coins, outputs a proof that verifies and blocks valid for `x`. -/
def Extraction (P : Protocol Mc Inst Out) {Bl : Type} (valid : Inst → Bl → Prop) {Ω : Type} [Fintype Ω]
    (M : Mc.Alg (P.Seed × Ω) P.Proof) (η : ℝ) : Prop :=
  ∃ (Ω' : Type) (_ : Fintype Ω') (E : Mc.Alg (P.Seed × Inst × Ω') (P.Proof × Bl)),
    Mc.time E ≤ Mc.time M ∧ ∀ x, 1 - η ≤ prob fun ω : P.Seed × Ω' =>
      valid x (Mc.eval E (ω.1, x, ω.2)).2 ∧ P.verify ω.1 (Mc.eval E (ω.1, x, ω.2)).1 = true

/-- **The floor** (a named hypothesis, the floor agenda's: Conjecture A, Theorem W): every algorithm that answers every
instance correctly with probability at least `1 - η` over its coins costs at least `T` (`T_mm(n)` for matmul). -/
def Floor (Mc : Machine) {Inst Out : Type} (correct : Inst → Out → Prop) (T η : ℝ) : Prop :=
  ∀ (Ω : Type) [Fintype Ω] (N : Mc.Alg (Inst × Ω) Out),
    (∀ x, 1 - η ≤ prob fun ω : Ω => correct x (Mc.eval N (x, ω))) → T ≤ Mc.time N

/-- An algorithm that solves every instance (with a proof) is above the floor: drop σ into its coins and forget
the proof, which routing does for free. -/
theorem Floor.le_time (P : Protocol Mc Inst Out) {T η : ℝ} (h : Floor Mc P.correct T η) {Ω : Type} [Fintype Ω]
    {M' : Mc.Alg (P.Seed × Inst × Ω) (P.Proof × Out)} (hM' : SolvesAll P M' η) : T ≤ Mc.time M' := by
  let route : Mc.Alg (Inst × P.Seed × Ω) (P.Seed × Inst × Ω) :=
    Mc.pair (Mc.seq Mc.snd Mc.fst) (Mc.pair Mc.fst (Mc.seq Mc.snd Mc.snd))
  have hroute : ∀ y, Mc.eval route y = (y.2.1, y.1, y.2.2) := fun y => by
    simp only [route, Mc.eval_seq, Mc.eval_pair, Mc.eval_fst, Mc.eval_snd]
  have hT := h (P.Seed × Ω) (Mc.seq (Mc.seq route M') Mc.snd) fun x => (hM' x).trans (prob_mono fun ω hω => by
    rw [Mc.eval_seq, Mc.eval_seq, hroute, Mc.eval_snd]
    exact hω.1)
  have r1 := Mc.time_seq_le (Mc.seq route M') (Mc.snd (I := P.Proof) (J := Out))
  have r2 := Mc.time_seq_le route M'
  have r3 := Mc.time_pair_le (Mc.seq (Mc.snd (I := Inst) (J := P.Seed × Ω)) Mc.fst)
    (Mc.pair Mc.fst (Mc.seq (Mc.snd (I := Inst) (J := P.Seed × Ω)) Mc.snd))
  have r4 := Mc.time_pair_le (Mc.fst (I := Inst) (J := P.Seed × Ω))
    (Mc.seq (Mc.snd (I := Inst) (J := P.Seed × Ω)) Mc.snd)
  have r5 := Mc.time_seq_le (Mc.snd (I := Inst) (J := P.Seed × Ω)) (Mc.fst (I := P.Seed) (J := Ω))
  have r6 := Mc.time_seq_le (Mc.snd (I := Inst) (J := P.Seed × Ω)) (Mc.snd (I := P.Seed) (J := Ω))
  have z1 := Mc.time_fst (I := Inst) (J := P.Seed × Ω)
  have z2 := Mc.time_snd (I := Inst) (J := P.Seed × Ω)
  have z3 := Mc.time_fst (I := P.Seed) (J := Ω)
  have z4 := Mc.time_snd (I := P.Seed) (J := Ω)
  have z5 := Mc.time_snd (I := P.Proof) (J := Out)
  linarith

/-- With extraction, the reduction solves every instance and costs at most `time M + time peel + time corr`, when the
peel and the correction answer `x` from blocks valid for `x`. -/
theorem exists_reduction (P : Protocol Mc Inst Out) {Bl Mid : Type} {valid : Inst → Bl → Prop}
    (peel : Mc.Alg Bl Mid) (corr : Mc.Alg Mid Out)
    (hpeel : ∀ x b, valid x b → P.correct x (Mc.eval corr (Mc.eval peel b))) {η : ℝ} {Ω : Type} [Fintype Ω]
    {M : Mc.Alg (P.Seed × Ω) P.Proof} (hext : Extraction P valid M η) :
    ∃ (Ω' : Type) (_ : Fintype Ω') (M' : Mc.Alg (P.Seed × Inst × Ω') (P.Proof × Out)),
      Mc.time M' ≤ Mc.time M + Mc.time peel + Mc.time corr ∧ SolvesAll P M' η := by
  obtain ⟨Ω', _, E, hE, hsucc⟩ := hext
  refine ⟨Ω', inferInstance, Mc.reduction E peel corr, Mc.time_reduction_le_of_le hE peel corr,
    fun x => (hsucc x).trans (prob_mono fun ω hω => ?_)⟩
  rw [Mc.eval_reduction]
  exact ⟨hpeel x _ hω.1, hω.2⟩

/-- **Security from extraction**, given that the peel and the correction cost at most `ε time M` for every adversary
`M` accepted with probability at least `δ`. -/
theorem secure_of_extraction (P : Protocol Mc Inst Out) {Bl Mid : Type} {valid : Inst → Bl → Prop}
    (peel : Mc.Alg Bl Mid) (corr : Mc.Alg Mid Out)
    (hpeel : ∀ x b, valid x b → P.correct x (Mc.eval corr (Mc.eval peel b))) {ε δ η : ℝ}
    (hext : ∀ (Ω : Type) [Fintype Ω] (M : Mc.Alg (P.Seed × Ω) P.Proof), δ ≤ acceptance P M →
      Extraction P valid M η)
    (hcost : ∀ (Ω : Type) [Fintype Ω] (M : Mc.Alg (P.Seed × Ω) P.Proof), δ ≤ acceptance P M →
      Mc.time peel + Mc.time corr ≤ ε * Mc.time M) :
    Secure P ε δ η := by
  intro Ω _ M hM
  obtain ⟨Ω', _, M', ht, hs⟩ := exists_reduction P peel corr hpeel (hext Ω M hM)
  exact ⟨Ω', inferInstance, M', by linarith [hcost Ω M hM], hs⟩

/-- **Security from extraction and the floor**: if the peel and the correction cost at most `q < T`, where `T` is the
floor, then `ε = q / (T - q)`. (`M'` solves every instance, so `T ≤ time M' ≤ time M + q`.) -/
theorem secure_of_extraction_floor (P : Protocol Mc Inst Out) {Bl Mid : Type} {valid : Inst → Bl → Prop}
    (peel : Mc.Alg Bl Mid) (corr : Mc.Alg Mid Out)
    (hpeel : ∀ x b, valid x b → P.correct x (Mc.eval corr (Mc.eval peel b))) {δ η T q : ℝ}
    (hext : ∀ (Ω : Type) [Fintype Ω] (M : Mc.Alg (P.Seed × Ω) P.Proof), δ ≤ acceptance P M →
      Extraction P valid M η)
    (hfloor : Floor Mc P.correct T η) (hq : Mc.time peel + Mc.time corr ≤ q) (hqT : q < T) :
    Secure P (q / (T - q)) δ η := by
  intro Ω _ M hM
  obtain ⟨Ω', _, M', ht, hs⟩ := exists_reduction P peel corr hpeel (hext Ω M hM)
  refine ⟨Ω', inferInstance, M', ?_, hs⟩
  have hTM := hfloor.le_time P hs
  have hTq : 0 < T - q := sub_pos.2 hqT
  have hq0 : 0 ≤ q := (add_nonneg (Mc.time_nonneg _) (Mc.time_nonneg _)).trans hq
  set ε := q / (T - q) with hε
  have hε0 : 0 ≤ ε := div_nonneg hq0 hTq.le
  have hεq : ε * (T - q) = q := div_mul_cancel₀ q hTq.ne'
  have hM : ε * (T - q) ≤ ε * Mc.time M := mul_le_mul_of_nonneg_left (by linarith) hε0
  linarith

/-- **Security is a relative floor** (RS-DEF's finding (i)), one direction: if some algorithm `M₀` solves every
instance and costs at most `(1 + ε) time M` for every adversary `M` accepted with probability at least `δ`, the
protocol is secure. No reduction from `M` is needed. -/
theorem secure_of_solver (P : Protocol Mc Inst Out) {ε δ η : ℝ} {Ω₀ : Type} [Fintype Ω₀]
    {M₀ : Mc.Alg (P.Seed × Inst × Ω₀) (P.Proof × Out)} (hM₀ : SolvesAll P M₀ η)
    (hcost : ∀ (Ω : Type) [Fintype Ω] (M : Mc.Alg (P.Seed × Ω) P.Proof), δ ≤ acceptance P M →
      Mc.time M₀ ≤ (1 + ε) * Mc.time M) :
    Secure P ε δ η :=
  fun Ω _ M hM => ⟨Ω₀, inferInstance, M₀, hcost Ω M hM, hM₀⟩

/-- **Security is a relative floor**, the other direction: under security, every adversary accepted with probability
at least `δ` costs at least `T / (1 + ε)`, for `T` the floor. -/
theorem floor_le_of_secure (P : Protocol Mc Inst Out) {ε δ η T : ℝ} (h : Secure P ε δ η)
    (hfloor : Floor Mc P.correct T η) {Ω : Type} [Fintype Ω] (M : Mc.Alg (P.Seed × Ω) P.Proof)
    (hM : δ ≤ acceptance P M) : T ≤ (1 + ε) * Mc.time M := by
  obtain ⟨Ω', _, M', ht, hs⟩ := h Ω M hM
  exact (hfloor.le_time P hs).trans ht

end Reduction

section MatmulPeel

namespace MatMachine

variable {R : Type} [Ring R] (Mc : MatMachine R)

/-- Multiply the outputs of `f` and `g`. -/
def mul₂ {I : Type} {a b c : ℕ} (f : Mc.Alg I (Mat R a b)) (g : Mc.Alg I (Mat R b c)) : Mc.Alg I (Mat R a c) :=
  Mc.lift2 f g (Mc.mul a b c)

/-- Subtract the output of `g` from that of `f`. -/
def sub₂ {I : Type} {a b : ℕ} (f g : Mc.Alg I (Mat R a b)) : Mc.Alg I (Mat R a b) :=
  Mc.lift2 f g (Mc.sub a b)

theorem eval_mul₂ {I : Type} {a b c : ℕ} (f : Mc.Alg I (Mat R a b)) (g : Mc.Alg I (Mat R b c)) (x : I) :
    Mc.eval (Mc.mul₂ f g) x = Mc.eval f x * Mc.eval g x := by
  rw [mul₂, Machine.eval_lift2, Mc.eval_mul]

theorem eval_sub₂ {I : Type} {a b : ℕ} (f g : Mc.Alg I (Mat R a b)) (x : I) :
    Mc.eval (Mc.sub₂ f g) x = Mc.eval f x - Mc.eval g x := by
  rw [sub₂, Machine.eval_lift2, Mc.eval_sub]

theorem time_mul₂_le_of {I : Type} {a b c : ℕ} {f : Mc.Alg I (Mat R a b)} {g : Mc.Alg I (Mat R b c)} {u v : ℝ}
    (hf : Mc.time f ≤ u) (hg : Mc.time g ≤ v) :
    Mc.time (Mc.mul₂ f g) ≤ u + v + Mc.κ * ((a : ℝ) * b * c) :=
  (Machine.time_lift2_le _ _ _ _).trans (by linarith [Mc.time_mul_le a b c])

theorem time_sub₂_le_of {I : Type} {a b : ℕ} {f g : Mc.Alg I (Mat R a b)} {u v : ℝ} (hf : Mc.time f ≤ u)
    (hg : Mc.time g ≤ v) : Mc.time (Mc.sub₂ f g) ≤ u + v + Mc.κadd * ((a : ℝ) * b) :=
  (Machine.time_lift2_le _ _ _ _).trans (by linarith [Mc.time_sub_le a b])

/-- The correction `(C', X, Y) ↦ C' − X − Y`. -/
def corr (n : ℕ) : Mc.Alg (Mat R n n × Mat R n n × Mat R n n) (Mat R n n) :=
  Mc.sub₂ (Mc.sub₂ Mc.fst (Mc.seq Mc.snd Mc.fst)) (Mc.seq Mc.snd Mc.snd)

theorem eval_corr (n : ℕ) (b : Mat R n n × Mat R n n × Mat R n n) :
    Mc.eval (Mc.corr n) b = b.1 - b.2.1 - b.2.2 := by
  simp only [corr, eval_sub₂, Mc.eval_seq, Mc.eval_fst, Mc.eval_snd]

/-- The correction costs two subtractions. -/
theorem time_corr_le (n : ℕ) : Mc.time (Mc.corr n) ≤ 2 * Mc.κadd * (n : ℝ) ^ 2 := by
  apply le_trans
  · unfold corr
    apply_rules [time_mul₂_le_of, time_sub₂_le_of, Machine.time_pair_le_of, Machine.time_seq_le_of,
      Machine.time_fst_le, Machine.time_snd_le]
  · linarith

/-- The full-rank peel's blocks `(C', A, Y, R, S)`: the committed product, the instance's `A`, M's masked
`Y = B + S`, and the masks. -/
abbrev FullBlocks (R : Type) (n : ℕ) : Type := Mat R n n × Mat R n n × Mat R n n × Mat R n n × Mat R n n

/-- Blocks valid for the instance `(A, B)`: `A` is the instance's, `Y = B + S`, and `C' = (A + R) Y`. -/
def ValidFull {n : ℕ} (x : Mat R n n × Mat R n n) (b : FullBlocks R n) : Prop :=
  b.2.1 = x.1 ∧ b.2.2.1 = x.2 + b.2.2.2.2 ∧ b.1 = (x.1 + b.2.2.2.1) * b.2.2.1

/-- **The full-rank peel**: `(C', A, Y, R, S) ↦ (C', R Y, A S)`, two `n × n × n` products. -/
def peelFull (n : ℕ) : Mc.Alg (FullBlocks R n) (Mat R n n × Mat R n n × Mat R n n) :=
  Mc.pair Mc.fst
    (Mc.pair (Mc.mul₂ (Mc.seq Mc.snd (Mc.seq Mc.snd (Mc.seq Mc.snd Mc.fst))) (Mc.seq Mc.snd (Mc.seq Mc.snd Mc.fst)))
      (Mc.mul₂ (Mc.seq Mc.snd Mc.fst) (Mc.seq Mc.snd (Mc.seq Mc.snd (Mc.seq Mc.snd Mc.snd)))))

theorem eval_peelFull (n : ℕ) (b : FullBlocks R n) :
    Mc.eval (Mc.peelFull n) b = (b.1, b.2.2.2.1 * b.2.2.1, b.2.1 * b.2.2.2.2) := by
  simp only [peelFull, eval_mul₂, Mc.eval_pair, Mc.eval_seq, Mc.eval_fst, Mc.eval_snd]

/-- The full-rank peel and correction answer the instance from valid blocks: `C' − R Y − A S = A B`. -/
theorem peelFull_correct {n : ℕ} (x : Mat R n n × Mat R n n) (b : FullBlocks R n) (hb : ValidFull x b) :
    x.1 * x.2 = Mc.eval (Mc.corr n) (Mc.eval (Mc.peelFull n) b) := by
  obtain ⟨C', A, Y, Rm, S⟩ := b
  obtain ⟨hA, hY, hC⟩ := hb
  dsimp only at hA hY hC
  subst hA hY hC
  rw [eval_peelFull, eval_corr]
  exact mask_peel x.1 Rm x.2 S

/-- The full-rank peel costs two `n × n × n` products. -/
theorem time_peelFull_le (n : ℕ) : Mc.time (Mc.peelFull n) ≤ 2 * Mc.κ * (n : ℝ) ^ 3 := by
  apply le_trans
  · unfold peelFull
    apply_rules [time_mul₂_le_of, time_sub₂_le_of, Machine.time_pair_le_of, Machine.time_seq_le_of,
      Machine.time_fst_le, Machine.time_snd_le]
  · linarith

/-- The low-rank peel's blocks `(C', A, Y, U, Vᵀ, W, Zᵀ)`: the committed product, the instance's `A`, M's masked
`Y = B + W Zᵀ`, and the factors of the rank-`r` masks `U Vᵀ` and `W Zᵀ`. -/
abbrev LowBlocks (R : Type) (n r : ℕ) : Type :=
  Mat R n n × Mat R n n × Mat R n n × Mat R n r × Mat R r n × Mat R n r × Mat R r n

/-- Blocks valid for the instance `(A, B)`: `A` is the instance's, `Y = B + W Zᵀ`, and `C' = (A + U Vᵀ) Y`. -/
def ValidLow {n r : ℕ} (x : Mat R n n × Mat R n n) (b : LowBlocks R n r) : Prop :=
  b.2.1 = x.1 ∧ b.2.2.1 = x.2 + b.2.2.2.2.2.1 * b.2.2.2.2.2.2 ∧ b.1 = (x.1 + b.2.2.2.1 * b.2.2.2.2.1) * b.2.2.1

/-- **The low-rank peel**: `(C', A, Y, U, Vᵀ, W, Zᵀ) ↦ (C', U (Vᵀ Y), (A W) Zᵀ)`, four products through the inner
dimension `r`. -/
def peelLow (n r : ℕ) : Mc.Alg (LowBlocks R n r) (Mat R n n × Mat R n n × Mat R n n) :=
  Mc.pair Mc.fst
    (Mc.pair
      (Mc.mul₂ (Mc.seq Mc.snd (Mc.seq Mc.snd (Mc.seq Mc.snd Mc.fst)))
        (Mc.mul₂ (Mc.seq Mc.snd (Mc.seq Mc.snd (Mc.seq Mc.snd (Mc.seq Mc.snd Mc.fst))))
          (Mc.seq Mc.snd (Mc.seq Mc.snd Mc.fst))))
      (Mc.mul₂
        (Mc.mul₂ (Mc.seq Mc.snd Mc.fst)
          (Mc.seq Mc.snd (Mc.seq Mc.snd (Mc.seq Mc.snd (Mc.seq Mc.snd (Mc.seq Mc.snd Mc.fst))))))
        (Mc.seq Mc.snd (Mc.seq Mc.snd (Mc.seq Mc.snd (Mc.seq Mc.snd (Mc.seq Mc.snd Mc.snd)))))))

theorem eval_peelLow (n r : ℕ) (b : LowBlocks R n r) :
    Mc.eval (Mc.peelLow n r) b =
      (b.1, b.2.2.2.1 * (b.2.2.2.2.1 * b.2.2.1), (b.2.1 * b.2.2.2.2.2.1) * b.2.2.2.2.2.2) := by
  simp only [peelLow, eval_mul₂, Mc.eval_pair, Mc.eval_seq, Mc.eval_fst, Mc.eval_snd]

/-- The low-rank peel and correction answer the instance from valid blocks:
`C' − U (Vᵀ Y) − (A W) Zᵀ = A B`. -/
theorem peelLow_correct {n r : ℕ} (x : Mat R n n × Mat R n n) (b : LowBlocks R n r) (hb : ValidLow x b) :
    x.1 * x.2 = Mc.eval (Mc.corr n) (Mc.eval (Mc.peelLow n r) b) := by
  obtain ⟨C', A, Y, U, Vt, W, Zt⟩ := b
  obtain ⟨hA, hY, hC⟩ := hb
  dsimp only at hA hY hC
  subst hA hY hC
  rw [eval_peelLow, eval_corr]
  exact (lowRank_peel x.1 x.2 U Vt W Zt).symm

/-- The low-rank peel costs four products through the inner dimension `r`: `4 κ n² r`. -/
theorem time_peelLow_le (n r : ℕ) : Mc.time (Mc.peelLow n r) ≤ 4 * Mc.κ * (n : ℝ) ^ 2 * r := by
  apply le_trans
  · unfold peelLow
    apply_rules (maxDepth := 200) [time_mul₂_le_of, time_sub₂_le_of, Machine.time_pair_le_of, Machine.time_seq_le_of,
      Machine.time_fst_le, Machine.time_snd_le]
  · linarith

variable {n : ℕ} {P : Protocol Mc.toMachine (Mat R n n × Mat R n n) (Mat R n n)}

/-- **Relative security of matmul PoUW with full-rank masks**, from extraction (named hypothesis `Extraction`) and
the floor (named hypothesis `Floor`) above the peel's and the correction's cost `q = 2κ n³ + 2κadd n²`:
`ε = q / (T − q)`. -/
theorem secure_fullRank (hmat : IsMatmul P) {δ η T : ℝ}
    (hext : ∀ (Ω : Type) [Fintype Ω] (M : Mc.Alg (P.Seed × Ω) P.Proof), δ ≤ acceptance P M →
      Extraction P ValidFull M η)
    (hfloor : Floor Mc.toMachine P.correct T η) (hT : 2 * Mc.κ * (n : ℝ) ^ 3 + 2 * Mc.κadd * (n : ℝ) ^ 2 < T) :
    Secure P ((2 * Mc.κ * (n : ℝ) ^ 3 + 2 * Mc.κadd * (n : ℝ) ^ 2) /
      (T - (2 * Mc.κ * (n : ℝ) ^ 3 + 2 * Mc.κadd * (n : ℝ) ^ 2))) δ η :=
  secure_of_extraction_floor P (Mc.peelFull n) (Mc.corr n)
    (fun x b hb => (hmat x _).2 (Mc.peelFull_correct x b hb)) hext hfloor
    (by linarith [Mc.time_peelFull_le n, Mc.time_corr_le n]) hT

/-- **Relative security of matmul PoUW with rank-`r` masks**, from extraction and the floor above
`q = 4κ n² r + 2κadd n²`: `ε = q / (T − q)`, which is `O(r / n)` against a floor of order `n³`. -/
theorem secure_lowRank (hmat : IsMatmul P) (r : ℕ) {δ η T : ℝ}
    (hext : ∀ (Ω : Type) [Fintype Ω] (M : Mc.Alg (P.Seed × Ω) P.Proof), δ ≤ acceptance P M →
      Extraction P (ValidLow (r := r)) M η)
    (hfloor : Floor Mc.toMachine P.correct T η)
    (hT : 4 * Mc.κ * (n : ℝ) ^ 2 * r + 2 * Mc.κadd * (n : ℝ) ^ 2 < T) :
    Secure P ((4 * Mc.κ * (n : ℝ) ^ 2 * r + 2 * Mc.κadd * (n : ℝ) ^ 2) /
      (T - (4 * Mc.κ * (n : ℝ) ^ 2 * r + 2 * Mc.κadd * (n : ℝ) ^ 2))) δ η :=
  secure_of_extraction_floor P (Mc.peelLow n r) (Mc.corr n)
    (fun x b hb => (hmat x _).2 (Mc.peelLow_correct x b hb)) hext hfloor
    (by linarith [Mc.time_peelLow_le n r, Mc.time_corr_le n]) hT

end MatMachine

end MatmulPeel

end Pouw.RelativeSecurity
