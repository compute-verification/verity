import Mathlib.Data.Fintype.BigOperators
import Mathlib.Data.Fintype.Fin
import Mathlib.Logic.Equiv.Fin.Basic
import Mathlib.Tactic.NormNum
import Definitions.Pous.Guarantees

/-!
# Sanity theorems: the definitions are not vacuous

* `honest_sim_accepted`, `honest_seq_accepted`: storing `C` (`B·ℓ` bits) and answering by lookup, with no query,
  is accepted with probability 1 in both reveal modes. The game can be passed.
* `raw_sim_accepted`, `raw_seq_accepted`: storing `m` raw blocks passes a `k`-block audit with probability at least
  `(m/B)^k`. The probability space and the uniform challenges are what they should be.
* `k_ge_86`: consequently, no scheme with `19 ∣ B` meets the requirement with `ε = 0` and fewer than 86 challenges.
* `plain_short_coins_insecure`: in the plain model, if the coins fit in the state then an adversary is accepted
  with probability 1. `W` and `pp` really are free, and security is not automatic.
-/

namespace Pous

open scoped ENNReal

variable {M : IdealModel}

/-! ### Uniform probability -/

section pr

variable {α β : Type} [Fintype α] [Fintype β]

theorem pr_def (P : α → Prop) [DecidablePred P] :
    pr P = ((Finset.univ.filter P).card : ℝ≥0∞) / Fintype.card α := by
  unfold pr; congr

theorem pr_eq_one [Nonempty α] {P : α → Prop} (h : ∀ x, P x) : pr P = 1 := by
  classical
  rw [pr_def, Finset.filter_true_of_mem fun x _ => h x, Finset.card_univ]
  exact ENNReal.div_self (by exact_mod_cast Fintype.card_ne_zero) (ENNReal.natCast_ne_top _)

theorem pr_mono {P Q : α → Prop} (h : ∀ x, P x → Q x) : pr P ≤ pr Q := by
  classical
  rw [pr_def, pr_def]
  gcongr
  exact h _

theorem pr_snd [Nonempty α] (P : β → Prop) : pr (fun x : α × β => P x.2) = pr P := by
  classical
  rw [pr_def, pr_def]
  have : (Finset.univ.filter fun x : α × β => P x.2) = Finset.univ ×ˢ Finset.univ.filter P := by
    ext x; simp
  rw [this, Finset.card_product, Finset.card_univ, Fintype.card_prod, Nat.cast_mul, Nat.cast_mul]
  exact ENNReal.mul_div_mul_left _ _ (by exact_mod_cast Fintype.card_ne_zero) (ENNReal.natCast_ne_top _)

/-- `k` uniform indices all fall below `m`. -/
theorem pr_all_lt (k B m : ℕ) (hm : m ≤ B) :
    pr (fun I : Fin k → Fin B => ∀ j, (I j : ℕ) < m) = ((m : ℝ≥0∞) / B) ^ k := by
  classical
  rw [pr_def]
  have : (Finset.univ.filter fun I : Fin k → Fin B => ∀ j, (I j : ℕ) < m) =
      Fintype.piFinset fun _ => Finset.univ.filter fun i : Fin B => (i : ℕ) < m := by
    ext I; simp [Fintype.mem_piFinset]
  rw [this, Fintype.card_piFinset, Finset.prod_const, Finset.card_univ, Fintype.card_fin, Fin.card_filter_val_lt,
    min_eq_right hm, Fintype.card_fun, Fintype.card_fin, Fintype.card_fin, div_eq_mul_inv, div_eq_mul_inv, mul_pow,
    Nat.cast_pow, Nat.cast_pow, ENNReal.inv_pow]

end pr

/-! ### Encodings of blocks as bits -/

/-- Pad an `a`-bit string with zeros to `b ≥ a` bits. -/
def pad {a b : ℕ} (x : Bits a) : Bits b := fun i => if hi : (i : ℕ) < a then x ⟨i, hi⟩ else false

/-- The first `a` bits of a `b`-bit string. -/
def trunc {a b : ℕ} (h : a ≤ b) (x : Bits b) : Bits a := fun i => x (Fin.castLE h i)

theorem trunc_pad {a b : ℕ} (h : a ≤ b) (x : Bits a) : trunc h (pad x) = x :=
  funext fun i => by simp [trunc, pad]

theorem ret_bounded {α : Type} (a : α) (D Q : ℕ) : (OracleProg.ret a : OracleProg M α).Bounded D Q :=
  fun _ => ⟨Nat.zero_le _, Nat.zero_le _⟩

/-- A sequential responder that never changes its state answers `is` by `g σ`. -/
theorem seqAnswers_const {ℓ S B : ℕ} (f : M.Qry → M.Ans) (g : Bits S → Fin B → Bits ℓ) :
    ∀ (j : ℕ) (is : List (Fin B)) (σ : Bits S),
      seqAnswers f (fun _ σ i => .ret (g σ i, σ)) j is σ = is.map (g σ)
  | _, [], _ => rfl
  | j, i :: is, σ => by
      simp only [seqAnswers, OracleProg.run, List.map_cons, seqAnswers_const f g (j + 1) is σ]

/-! ### The honest prover -/

/-- The honest prover stores `C` whole. -/
def honestPre (sch : Scheme M) : Preprocessor sch (sch.B * sch.ℓ) :=
  fun _ _ C _ => codeEquiv _ _ C

/-- It answers all indices by lookup, with no query. -/
def honestSim (sch : Scheme M) (k : ℕ) : SimResponder sch (sch.B * sch.ℓ) 0 0 k where
  prog _ _ σ I := .ret fun j => (codeEquiv _ _).symm σ (I j)
  bounded _ _ _ _ := ret_bounded _ _ _

/-- Sequentially, it answers each index by lookup and keeps its state. -/
def honestSeq (sch : Scheme M) : SeqResponder sch (sch.B * sch.ℓ) 0 0 where
  step _ _ _ σ i := .ret ((codeEquiv _ _).symm σ i, σ)
  bounded _ _ _ _ _ := ret_bounded _ _ _

theorem honest_sim_accepted (sch : Scheme M) (k : ℕ) (W : Bits sch.n) :
    pr (simAccepts sch W (honestPre sch) (honestSim sch k)) = 1 :=
  pr_eq_one fun x => by simp [simAccepts, honestPre, honestSim, OracleProg.run]

theorem honest_seq_accepted (sch : Scheme M) (k : ℕ) (W : Bits sch.n) :
    pr (seqAccepts sch k W (honestPre sch) (honestSeq sch)) = 1 :=
  pr_eq_one fun x => by
    simp only [seqAccepts, honestSeq, honestPre]
    rw [seqAnswers_const _ (fun σ i => (codeEquiv _ _).symm σ i), List.map_ofFn]
    congr 1
    funext j
    simp

/-! ### Storing `m` raw blocks -/

section raw

variable (sch : Scheme M) {m S : ℕ} (h : m * sch.ℓ ≤ S) (hm : m ≤ sch.B)

/-- Keep the first `m` blocks raw, padded to `S` bits. -/
def rawPre : Preprocessor sch S :=
  fun _ _ C _ => pad (codeEquiv m sch.ℓ fun b => C (Fin.castLE hm b))

/-- Return a stored block, and zeros for any other index. -/
def rawAnswer (σ : Bits S) (i : Fin sch.B) : Bits sch.ℓ :=
  if hi : (i : ℕ) < m then (codeEquiv m sch.ℓ).symm (trunc h σ) ⟨i, hi⟩ else fun _ => false

def rawSim (D Q k : ℕ) : SimResponder sch S D Q k where
  prog _ _ σ I := .ret fun j => rawAnswer sch h σ (I j)
  bounded _ _ _ _ := ret_bounded _ _ _

def rawSeq (D Q : ℕ) : SeqResponder sch S D Q where
  step _ _ _ σ i := .ret (rawAnswer sch h σ i, σ)
  bounded _ _ _ _ _ := ret_bounded _ _ _

theorem rawAnswer_eq (W : Bits sch.n) (pp : Bits sch.t) (C : sch.Code) (ω : M.Ω) (i : Fin sch.B)
    (hi : (i : ℕ) < m) : rawAnswer sch h (rawPre sch hm W pp C ω) i = C i := by
  simp only [rawAnswer, hi, dite_true, rawPre, trunc_pad, Equiv.symm_apply_apply]
  rfl

theorem raw_sim_accepted (D Q k : ℕ) (W : Bits sch.n) :
    ((m : ℝ≥0∞) / sch.B) ^ k ≤ pr (simAccepts sch W (rawPre sch hm) (rawSim sch h D Q k)) := by
  rw [← pr_all_lt k sch.B m hm, ← pr_snd (α := sch.Coins), ← pr_snd (α := M.Ω)]
  refine pr_mono fun x hx => ?_
  simp only [simAccepts, rawSim, OracleProg.run]
  funext j
  exact rawAnswer_eq sch h hm _ _ _ _ _ (hx j)

theorem raw_seq_accepted (D Q k : ℕ) (W : Bits sch.n) :
    ((m : ℝ≥0∞) / sch.B) ^ k ≤ pr (seqAccepts sch k W (rawPre sch hm) (rawSeq sch h D Q)) := by
  rw [← pr_all_lt k sch.B m hm, ← pr_snd (α := sch.Coins), ← pr_snd (α := M.Ω)]
  refine pr_mono fun x hx => ?_
  simp only [seqAccepts, rawSeq]
  rw [seqAnswers_const _ (fun σ i => rawAnswer sch h σ i), List.map_ofFn]
  congr 1
  funext j
  exact rawAnswer_eq sch h hm _ _ _ _ _ (hx j)

end raw

/-! ### The requirement forces `k ≥ 86` -/

theorem rho_pow_85 : (1 / 100 : ℚ) < (18 / 19) ^ 85 := by norm_num

/-- If the requirement holds with `ε = 0` and `19 ∣ B`, then `k ≥ 86`: storing `18/19` of the blocks raw passes
with probability `(18/19)^k`. -/
theorem k_ge_86 (sch : Scheme M) (reveal : Reveal) (D Q k : ℕ) (h19 : 19 ∣ sch.B)
    (hsec : AuditSecure sch reveal (sch.stateBound Params.ρ) D Q k (ENNReal.ofReal Params.δ) 0) : 86 ≤ k := by
  obtain ⟨c, hc⟩ := h19
  have hc0 : c ≠ 0 := by rintro rfl; simp at hc; exact absurd hc sch.B_pos.ne'
  have hm : 18 * c ≤ sch.B := by omega
  have hS : 18 * c * sch.ℓ ≤ sch.stateBound Params.ρ := by
    have : Params.ρ * ((sch.B * sch.ℓ : ℕ) : ℚ) = ((18 * c * sch.ℓ : ℕ) : ℚ) := by
      rw [hc, Params.ρ]; push_cast; ring
    simp only [Scheme.stateBound, this, Nat.floor_natCast, le_refl]
  have hpow : ((18 / 19 : ℝ≥0∞)) ^ k ≤ ENNReal.ofReal Params.δ := by
    have hratio : ((18 * c : ℕ) : ℝ≥0∞) / sch.B = 18 / 19 := by
      rw [hc]; push_cast
      exact ENNReal.mul_div_mul_right _ _ (by exact_mod_cast hc0) (ENNReal.natCast_ne_top _)
    have key : ((((18 * c : ℕ)) : ℝ≥0∞) / sch.B) ^ k ≤ ENNReal.ofReal Params.δ + 0 := by
      cases reveal with
      | simultaneous =>
          exact (raw_sim_accepted sch hS hm D Q k fun _ => false).trans (hsec _ _ _)
      | sequential =>
          exact (raw_seq_accepted sch hS hm D Q k fun _ => false).trans (hsec _ _ _)
    rwa [hratio, add_zero] at key
  have hreal : ((18 / 19 : ℝ)) ^ k ≤ 1 / 100 := by
    have := ENNReal.toReal_mono ENNReal.ofReal_ne_top hpow
    rw [ENNReal.toReal_pow, ENNReal.toReal_div, ENNReal.toReal_ofReal (by norm_num [Params.δ])] at this
    simpa [Params.δ] using this
  by_contra hk
  have hq : ((18 / 19 : ℝ)) ^ 85 ≤ (18 / 19) ^ k :=
    pow_le_pow_of_le_one (by norm_num) (by norm_num) (by omega)
  have h85 : (1 / 100 : ℝ) < (18 / 19) ^ 85 := by norm_num
  exact absurd (h85.trans_le (hq.trans hreal)) (lt_irrefl _)

/-! ### The plain model: short coins break every scheme -/

/-- Coins consistent with `(W, pp, C)`, when there are any. -/
noncomputable def coinGuess (sch : Scheme IdealModel.plain) (W : Bits sch.n) (pp : Bits sch.t) (C : sch.Code) :
    sch.Coins :=
  open Classical in if h : ∃ r, sch.enc W () r = (pp, C) then h.choose else Classical.arbitrary _

/-- In the plain model, if the coins fit in `S` bits, an `S`-bit adversary that stores consistent coins and
re-encodes is accepted with probability 1, for every `W` and `k`. -/
theorem plain_short_coins_insecure (sch : Scheme IdealModel.plain) (S k : ℕ)
    (hS : Fintype.card sch.Coins ≤ 2 ^ S) :
    ∃ (A₁ : Preprocessor sch S) (A₂ : SimResponder sch S 0 0 k), ∀ W, pr (simAccepts sch W A₁ A₂) = 1 := by
  obtain ⟨e⟩ := Function.Embedding.nonempty_of_card_le (β := Bits S) (by simpa using hS)
  refine ⟨fun W pp C _ => e (coinGuess sch W pp C),
    ⟨fun W _ σ I => .ret fun j => (sch.enc W () (Function.invFun e σ)).2 (I j), fun _ _ _ _ => ret_bounded _ _ _⟩,
    fun W => pr_eq_one fun x => ?_⟩
  obtain ⟨ω, r, I⟩ := x
  have hex : ∃ r', sch.enc W () r' = ((sch.enc W ω r).1, (sch.enc W ω r).2) := ⟨r, rfl⟩
  simp only [simAccepts, OracleProg.run, coinGuess, hex, dite_true, Function.leftInverse_invFun e.injective _]
  rw [hex.choose_spec]

/-! ### Degenerate requirements are excluded -/

/-- The statement red team's scheme: `W` sits in `pp`, `C` is 133 bits of fresh noise, and decoding ignores `C`. -/
def junk : Scheme IdealModel.plain where
  n := 2660
  t := 2660
  B := 1
  ℓ := 133
  B_pos := Nat.one_pos
  Coins := Bits 133
  enc W _ r := (W, fun _ => r)
  dec pp _ _ := pp

theorem junk_correct : junk.Correct := fun _ _ _ => rfl

theorem junk_spaceBound : junk.SpaceBound Params.expansion := by
  simp only [Scheme.SpaceBound, junk, Params.expansion]; norm_num

/-- `junk` is correct and within the space bound, and its audit is hard to pass, but `|C| < |W|`, so it does not
meet the requirement. -/
theorem junk_not_meets (reveal : Reveal) (D Q k : ℕ) (ε : ℝ≥0∞) : ¬ Meets junk reveal D Q k ε :=
  fun h => absurd h.2.2.1 (by simp [Scheme.CodeHoldsW, junk])

/-- `ε` above `2^-128` never meets the requirement; in particular `ε = 1`, which would make the audit vacuous. -/
theorem not_meets_of_lt_eps (sch : Scheme M) (reveal : Reveal) (D Q k : ℕ) {ε : ℝ≥0∞} (hε : Params.εMax < ε) :
    ¬ Meets sch reveal D Q k ε :=
  fun h => absurd h.1 (not_le.2 hε)

theorem not_meets_eps_one (sch : Scheme M) (reveal : Reveal) (D Q k : ℕ) : ¬ Meets sch reveal D Q k 1 :=
  not_meets_of_lt_eps sch reveal D Q k
    (pow_lt_one₀ zero_le (ENNReal.inv_lt_one.2 ENNReal.one_lt_two) (by norm_num [Params.secParam]))

end Pous
