import Mathlib.NumberTheory.LegendreSymbol.Basic
import Definitions.Pous.P2.Cost
import Proofs.Pous.P2.Domain

/-!
# P2 in the ring model: the cost bound, the labels, and correctness

The cost model is `Security/Definitions/Pous/P2/Cost.lean`. This file proves:

* `ringBounded_of_bounded`: `Bounded D Q → RingBounded D Q Q` (every round counts at most once, every query at most
  once in each count), so `MeetsOn` at `(D, Q, Q)` implies `Meets` at `(D, Q)` (`meets_of_meetsOn`);
* `sqE_bijective`: under an odd labeling, P2's parity-signed square is a bijection of `Z/p` when `p` is a prime
  `≡ 3 mod 4` (`−1` is a non-residue, and `x`, `−x` have labels of opposite parity);
* `p2ring_correct`: `decBlock` inverts `encBlock` on every primitive, so `p2ring` is `Correct`;
* `complete`: `p2ring` is correct, holds `W`, is within the space bound, decodes a block with two multiplication
  rounds, two key queries and three ring queries, and storing `C` whole passes the audit with no query.
-/

namespace PousP2Ring

open Pous PousP2p
open scoped ENNReal

/-! ### Programs: sequencing and the ring cost -/

section OracleProg

variable {B w p : ℕ} [NeZero p] {α β : Type}

/-- Run `P`, then `g` on its output. -/
def pbind : OracleProg (ringRO B w p) α → (α → OracleProg (ringRO B w p) β) → OracleProg (ringRO B w p) β
  | .ret a, g => g a
  | .round qs k, g => .round qs fun as => pbind (k as) g

variable (f : (ringRO B w p).Qry → (ringRO B w p).Ans)

theorem run_pbind (P : OracleProg (ringRO B w p) α) (g : α → OracleProg (ringRO B w p) β) :
    (pbind P g).run f = (g (P.run f)).run f := by
  induction P with
  | ret a => rfl
  | round qs k ih => exact ih _

theorem mulDepth_pbind (P : OracleProg (ringRO B w p) α) (g : α → OracleProg (ringRO B w p) β) :
    (pbind P g).mulDepth f = P.mulDepth f + (g (P.run f)).mulDepth f := by
  induction P with
  | ret a => simp [pbind, OracleProg.mulDepth, OracleProg.run]
  | round qs k ih => simp only [pbind, OracleProg.mulDepth, OracleProg.run, ih]; omega

theorem keyWork_pbind (P : OracleProg (ringRO B w p) α) (g : α → OracleProg (ringRO B w p) β) :
    (pbind P g).keyWork f = P.keyWork f + (g (P.run f)).keyWork f := by
  induction P with
  | ret a => simp [pbind, OracleProg.keyWork, OracleProg.run]
  | round qs k ih => simp only [pbind, OracleProg.keyWork, OracleProg.run, ih]; omega

theorem ringWork_pbind (P : OracleProg (ringRO B w p) α) (g : α → OracleProg (ringRO B w p) β) :
    (pbind P g).ringWork f = P.ringWork f + (g (P.run f)).ringWork f := by
  induction P with
  | ret a => simp [pbind, OracleProg.ringWork, OracleProg.run]
  | round qs k ih => simp only [pbind, OracleProg.ringWork, OracleProg.run, ih]; omega

theorem ringBounded_pbind {D₁ Q₁ L₁ D₂ Q₂ L₂ : ℕ} {P : OracleProg (ringRO B w p) α} {g : α → OracleProg (ringRO B w p) β}
    (hP : P.RingBounded D₁ Q₁ L₁) (hg : ∀ a, (g a).RingBounded D₂ Q₂ L₂) :
    (pbind P g).RingBounded (D₁ + D₂) (Q₁ + Q₂) (L₁ + L₂) := fun ω => by
  rw [mulDepth_pbind, keyWork_pbind, ringWork_pbind]
  obtain ⟨h1, h2, h3⟩ := hP ω
  obtain ⟨h4, h5, h6⟩ := hg (P.run ((ringRO B w p).answer ω)) ω
  omega

theorem mulDepth_progMap (g : α → β) (P : OracleProg (ringRO B w p) α) :
    (PousThm1.progMap g P).mulDepth f = P.mulDepth f := by
  induction P with
  | ret a => rfl
  | round qs k ih => simp only [PousThm1.progMap, OracleProg.mulDepth, ih]

theorem keyWork_progMap (g : α → β) (P : OracleProg (ringRO B w p) α) :
    (PousThm1.progMap g P).keyWork f = P.keyWork f := by
  induction P with
  | ret a => rfl
  | round qs k ih => simp only [PousThm1.progMap, OracleProg.keyWork, ih]

theorem ringWork_progMap (g : α → β) (P : OracleProg (ringRO B w p) α) :
    (PousThm1.progMap g P).ringWork f = P.ringWork f := by
  induction P with
  | ret a => rfl
  | round qs k ih => simp only [PousThm1.progMap, OracleProg.ringWork, ih]

theorem ringBounded_progMap {D Q L : ℕ} (g : α → β) {P : OracleProg (ringRO B w p) α} (h : P.RingBounded D Q L) :
    (PousThm1.progMap g P).RingBounded D Q L := fun ω => by
  rw [mulDepth_progMap, keyWork_progMap, ringWork_progMap]
  exact h ω

theorem ringBounded_ret (a : α) (D Q L : ℕ) : (OracleProg.ret a : OracleProg (ringRO B w p) α).RingBounded D Q L :=
  fun _ => ⟨Nat.zero_le _, Nat.zero_le _, Nat.zero_le _⟩

theorem mulDepth_le_depth (P : OracleProg (ringRO B w p) α) : P.mulDepth f ≤ P.depth f := by
  induction P with
  | ret a => simp [OracleProg.mulDepth, OracleProg.depth]
  | round qs k ih =>
      simp only [OracleProg.mulDepth, OracleProg.depth]
      have := ih (qs.map f)
      split <;> omega

theorem keyWork_le_work (P : OracleProg (ringRO B w p) α) : P.keyWork f ≤ P.work f := by
  induction P with
  | ret a => simp [OracleProg.keyWork, OracleProg.work]
  | round qs k ih =>
      simp only [OracleProg.keyWork, OracleProg.work]
      exact Nat.add_le_add (List.countP_le_length) (ih _)

theorem ringWork_le_work (P : OracleProg (ringRO B w p) α) : P.ringWork f ≤ P.work f := by
  induction P with
  | ret a => simp [OracleProg.ringWork, OracleProg.work]
  | round qs k ih =>
      simp only [OracleProg.ringWork, OracleProg.work]
      exact Nat.add_le_add (List.countP_le_length) (ih _)

/-- **The old bound is a special case**: a `(D, Q)`-bounded program is `(D, Q, Q)`-bounded in the ring model. -/
theorem ringBounded_of_bounded {D Q : ℕ} {P : OracleProg (ringRO B w p) α} (h : P.Bounded D Q) :
    P.RingBounded D Q Q := fun ω =>
  ⟨(mulDepth_le_depth _ P).trans (h ω).1, (keyWork_le_work _ P).trans (h ω).2,
    (ringWork_le_work _ P).trans (h ω).2⟩

end OracleProg

/-- Security against every `(D, Q, Q)`-bounded responder is security against every `(D, Q)`-bounded one. -/
theorem auditSecure_of_auditSecureOn {B w p : ℕ} [NeZero p] (sch : Scheme (ringRO B w p)) (reveal : Reveal)
    (S D Q k : ℕ) (δ ε : ℝ≥0∞) (h : AuditSecureOn sch reveal S D Q Q k δ ε) :
    AuditSecure sch reveal S D Q k δ ε := by
  cases reveal with
  | simultaneous =>
      intro W A₁ A₂
      exact h W A₁ ⟨A₂.prog, fun W pp σ I => ringBounded_of_bounded (A₂.bounded W pp σ I)⟩
  | sequential =>
      intro W A₁ A₂
      exact h W A₁ ⟨A₂.step, fun W pp j σ i => ringBounded_of_bounded (A₂.bounded W pp j σ i)⟩

/-- **`MeetsOn` at `(D, Q, Q)` implies `Meets` at `(D, Q)`**, in either reveal mode, for every scheme in the ring
model: the ring bound only enlarges the class of responders the requirement quantifies over. -/
theorem meets_of_meetsOn {B w p : ℕ} [NeZero p] (sch : Scheme (ringRO B w p)) (reveal : Reveal) (D Q k : ℕ)
    (ε : ℝ≥0∞) (h : MeetsOn sch reveal D Q Q k ε) : Meets sch reveal D Q k ε :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, auditSecure_of_auditSecureOn sch reveal _ D Q k _ ε h.2.2.2.2⟩

/-! ### Labels -/

section Labels

variable {B w p : ℕ} [NeZero p] (φ : OddLabeling p)

theorem bitsVal_lab (hp2 : p ≤ 2 ^ w) (x : ZMod p) : bitsVal (lab w φ x) = φ.1 x :=
  bitsVal_bitsOfNat ((φ.1 x).isLt.trans_le hp2)

theorem lab_lt (hp2 : p ≤ 2 ^ w) (x : ZMod p) : bitsVal (lab w φ x) < p := by
  rw [bitsVal_lab φ hp2]
  exact (φ.1 x).isLt

theorem eltD_of_lt {a : Bits w} (h : bitsVal a < p) : eltD φ a = φ.1.symm ⟨bitsVal a, h⟩ := by
  simp [eltD, h]

theorem eltD_lab (hp2 : p ≤ 2 ^ w) (x : ZMod p) : eltD φ (lab w φ x) = x := by
  rw [eltD_of_lt φ (lab_lt φ hp2 x)]
  have : (⟨bitsVal (lab w φ x), lab_lt φ hp2 x⟩ : Fin p) = φ.1 x := Fin.ext (bitsVal_lab φ hp2 x)
  rw [this, Equiv.symm_apply_apply]

theorem lab_eltD {a : Bits w} (h : bitsVal a < p) : lab w φ (eltD φ a) = a := by
  rw [eltD_of_lt φ h, lab, Equiv.apply_symm_apply]
  exact bitsOfNat_bitsVal a

theorem lab_injective (hp2 : p ≤ 2 ^ w) : Function.Injective (lab w φ) := fun x y h => by
  have := congrArg (eltD φ) h
  rwa [eltD_lab φ hp2, eltD_lab φ hp2] at this

theorem lab_symm (τ : Fin p) : lab w φ (φ.1.symm τ) = bitsOfNat w τ := by
  simp [lab]

/-- An odd labeling fixes `0` when `p` is odd. -/
theorem lab_zero (hodd : p % 2 = 1) : φ.1 0 = 0 := by
  have h := φ.2 0
  rw [neg_zero] at h
  have hv := congrArg Fin.val h
  rw [Fin.val_neg] at hv
  split_ifs at hv with h0
  · exact h0
  · have := (φ.1 0).isLt
    omega

theorem lab_eq_zero_iff (hodd : p % 2 = 1) {x : ZMod p} : φ.1 x = 0 ↔ x = 0 :=
  ⟨fun h => φ.1.injective (h.trans (lab_zero φ hodd).symm), fun h => h ▸ lab_zero φ hodd⟩

theorem val_lab_neg (hodd : p % 2 = 1) {x : ZMod p} (hx : x ≠ 0) : (φ.1 (-x) : ℕ) = p - φ.1 x := by
  rw [φ.2 x, Fin.val_neg, ite_eq_right ((lab_eq_zero_iff φ hodd).not.2 hx)]

/-- `x` and `−x` have labels of opposite parity, for `x ≠ 0` and `p` odd. -/
theorem parity_neg (hodd : p % 2 = 1) {x : ZMod p} (hx : x ≠ 0) : (φ.1 (-x) : ℕ) % 2 ≠ (φ.1 x : ℕ) % 2 := by
  rw [val_lab_neg φ hodd hx]
  have := (φ.1 x).isLt
  omega

theorem negL_lab (hp2 : p ≤ 2 ^ w) (hodd : p % 2 = 1) (x : ZMod p) : negL p (lab w φ x) = lab w φ (-x) := by
  unfold negL
  rw [bitsVal_lab φ hp2]
  by_cases hx : x = 0
  · subst hx
    simp [lab_zero φ hodd]
  · have h0 : (φ.1 x : ℕ) ≠ 0 := fun h => hx ((lab_eq_zero_iff φ hodd).1 (Fin.ext h))
    rw [ite_eq_right h0, lab, val_lab_neg φ hodd hx]

theorem sqFix_lab (hp2 : p ≤ 2 ^ w) (hodd : p % 2 = 1) (x : ZMod p) :
    sqFix p (lab w φ x) (lab w φ (x * x)) = lab w φ (sqE φ x) := by
  unfold sqFix sqE
  rw [bitsVal_lab φ hp2]
  split_ifs
  · exact negL_lab φ hp2 hodd _
  · rfl

omit [NeZero p] in
theorem sigmaK_lt {K r : Bits w} (h : bitsVal r < p) : bitsVal (sigmaK p K r) < p := by
  unfold sigmaK
  split_ifs with h'
  · exact h'
  · exact h

omit [NeZero p] in
/-- `σ_K` is an involution on the labels below `p`. -/
theorem sigmaK_sigmaK {K r : Bits w} (h : bitsVal r < p) : sigmaK p K (sigmaK p K r) = r := by
  have hx : (fun i => xor ((fun i => xor (r i) (K i) : Bits w) i) (K i) : Bits w) = r :=
    funext fun i => by simp
  by_cases h' : bitsVal (fun i => xor (r i) (K i) : Bits w) < p
  · have e : sigmaK p K r = fun i => xor (r i) (K i) := ite_eq_left h'
    rw [e, sigmaK, hx, ite_eq_left h]
  · have e : sigmaK p K r = r := ite_eq_right h'
    rw [e, e]

end Labels

/-! ### The model's answers on valid labels -/

section Answers

variable {B w p : ℕ} [NeZero p] (ω : (ringRO B w p).Ω)

theorem answer_mul (hp2 : p ≤ 2 ^ w) (x y : ZMod p) :
    (ringRO B w p).answer ω (RingQry.mul (lab w ω.1 x) (lab w ω.1 y)) = lab w ω.1 (x * y) := by
  change ringOp ω.1 (· * ·) _ _ = _
  rw [ringOp, ite_eq_left ⟨lab_lt _ hp2 x, lab_lt _ hp2 y⟩, eltD_lab _ hp2, eltD_lab _ hp2]

theorem answer_add (hp2 : p ≤ 2 ^ w) (x y : ZMod p) :
    (ringRO B w p).answer ω (RingQry.add (lab w ω.1 x) (lab w ω.1 y)) = lab w ω.1 (x + y) := by
  change ringOp ω.1 (· + ·) _ _ = _
  rw [ringOp, ite_eq_left ⟨lab_lt _ hp2 x, lab_lt _ hp2 y⟩, eltD_lab _ hp2, eltD_lab _ hp2]

theorem answer_one : (ringRO B w p).answer ω RingQry.one = lab w ω.1 1 := rfl

theorem answer_mask (j : Fin B) : (ringRO B w p).answer ω (RingQry.key j .mask) = (ω.2 j).1 := rfl

theorem answer_offset (j : Fin B) :
    (ringRO B w p).answer ω (RingQry.key j .offset) = lab w ω.1 (ω.1.1.symm (ω.2 j).2) :=
  (lab_symm ω.1 _).symm

end Answers

/-! ### The parity-signed square is a bijection -/

section Square

variable {p : ℕ} [NeZero p]

omit [NeZero p] in
theorem odd_of_mod_four (h4 : p % 4 = 3) : p % 2 = 1 := by omega

theorem sqE_injective (hp : p.Prime) (h4 : p % 4 = 3) (φ : OddLabeling p) : Function.Injective (sqE φ) := by
  have := Fact.mk hp
  have hodd := odd_of_mod_four h4
  have hns : ∀ x y : ZMod p, x ≠ 0 → y * y = -(x * x) → False := by
    intro x y hx h
    have hsq : IsSquare (-1 : ZMod p) := ⟨y * x⁻¹, by
      rw [show y * x⁻¹ * (y * x⁻¹) = (y * y) * (x * x)⁻¹ by ring, h, neg_mul,
        mul_inv_cancel₀ (mul_ne_zero hx hx)]⟩
    exact (ZMod.exists_sq_eq_neg_one_iff.1 hsq) h4
  have key : ∀ a b : ZMod p, a * a = b * b → (φ.1 a : ℕ) % 2 = (φ.1 b : ℕ) % 2 → a = b := by
    intro a b hab hpar
    rcases mul_self_eq_mul_self_iff.1 hab with h | h
    · exact h
    · by_cases hb : b = 0
      · rw [h, hb, neg_zero]
      · exact absurd (h ▸ hpar) (parity_neg φ hodd hb)
  have zero : ∀ a : ZMod p, a * a = 0 → a = 0 := fun a h => mul_self_eq_zero.1 h
  intro a b h
  unfold sqE at h
  split_ifs at h with ha hb hb
  · exact key a b (neg_inj.1 h) (by omega)
  · by_cases ha0 : a = 0
    · subst ha0
      rw [mul_zero, neg_zero] at h
      exact (zero b h.symm).symm
    · exact (hns a b ha0 h.symm).elim
  · by_cases hb0 : b = 0
    · subst hb0
      rw [mul_zero, neg_zero] at h
      exact zero a h
    · exact (hns b a hb0 h).elim
  · exact key a b h (by omega)

theorem sqE_surjective (hp : p.Prime) (h4 : p % 4 = 3) (φ : OddLabeling p) : Function.Surjective (sqE φ) :=
  Finite.surjective_of_injective (sqE_injective hp h4 φ)

theorem sqE_rhoE (hp : p.Prime) (h4 : p % 4 = 3) (φ : OddLabeling p) (y : ZMod p) : sqE φ (rhoE φ y) = y :=
  Function.invFun_eq (sqE_surjective hp h4 φ y)

/-- `ρ` is the inverse of `sqE`: any `x` with `sqE x = y` is `ρ(y)`. -/
theorem rhoE_eq (hp : p.Prime) (h4 : p % 4 = 3) (φ : OddLabeling p) {x y : ZMod p} (h : sqE φ x = y) :
    rhoE φ y = x :=
  sqE_injective hp h4 φ ((sqE_rhoE hp h4 φ y).trans h.symm)

end Square

/-! ### Correctness -/

section Correct

variable {B w p b : ℕ} [NeZero p]

theorem nthAns_zero {w : ℕ} (a : Bits w) (as : List (Bits w)) : nthAns (a :: as) 0 = a := rfl
theorem nthAns_succ {w : ℕ} (a : Bits w) (as : List (Bits w)) (i : ℕ) : nthAns (a :: as) (i + 1) = nthAns as i := rfl

/-- `decBlock`'s run, unfolded: `u = σ_K(sqFix(c, c²))`, then the label of `sqFix(u, u²) + (−τ)`. -/
theorem run_decBlock (f : (ringRO B w p).Qry → (ringRO B w p).Ans) (j : Fin B) (c : Bits w) :
    (decBlock B w p b j c).run f =
      bitsOfNat b (bitsVal (f (RingQry.add
        (sqFix p (sigmaK p (f (RingQry.key j .mask)) (sqFix p c (f (RingQry.mul c c))))
          (f (RingQry.mul (sigmaK p (f (RingQry.key j .mask)) (sqFix p c (f (RingQry.mul c c))))
            (sigmaK p (f (RingQry.key j .mask)) (sqFix p c (f (RingQry.mul c c)))))))
        (negL p (f (RingQry.key j .offset)))))) := rfl

/-- `decBlock` on a valid label `lab c`: the label of `sqE(σ_K(sqE c)) − t`, read mod `2^b`. -/
theorem decBlock_lab (hp2 : p ≤ 2 ^ w) (hodd : p % 2 = 1) (ω : (ringRO B w p).Ω) (j : Fin B) (c : ZMod p) :
    (decBlock B w p b j (lab w ω.1 c)).run ((ringRO B w p).answer ω) =
      bitsOfNat b (bitsVal (lab w ω.1
        (sqE ω.1 (eltD ω.1 (sigmaK p (ω.2 j).1 (lab w ω.1 (sqE ω.1 c)))) + -ω.1.1.symm (ω.2 j).2))) := by
  rw [run_decBlock, answer_mask, answer_mul ω hp2, sqFix_lab ω.1 hp2 hodd, answer_offset,
    negL_lab ω.1 hp2 hodd]
  have hu : bitsVal (sigmaK p (ω.2 j).1 (lab w ω.1 (sqE ω.1 c))) < p := sigmaK_lt (lab_lt ω.1 hp2 _)
  generalize sigmaK p (ω.2 j).1 (lab w ω.1 (sqE ω.1 c)) = u at hu ⊢
  have hu' : u = lab w ω.1 (eltD ω.1 u) := (lab_eltD ω.1 hu).symm
  conv_lhs => rw [hu']
  rw [answer_mul ω hp2, sqFix_lab ω.1 hp2 hodd, answer_add ω hp2]

/-- **`decBlock` inverts `encBlock`** on every primitive, for a payload label below `p`. -/
theorem decBlock_encBlock (hp : p.Prime) (h4 : p % 4 = 3) (hp2 : p ≤ 2 ^ w) (ω : (ringRO B w p).Ω) (j : Fin B)
    {v : Bits w} (hv : bitsVal v < p) :
    (decBlock B w p b j (encBlock ω.1 (ω.2 j).1 (ω.2 j).2 v)).run ((ringRO B w p).answer ω) =
      bitsOfNat b (bitsVal v) := by
  have hodd := odd_of_mod_four h4
  have hs : bitsVal (sigmaK p (ω.2 j).1 (lab w ω.1 (rhoE ω.1 (eltD ω.1 v + ω.1.1.symm (ω.2 j).2)))) < p :=
    sigmaK_lt (lab_lt ω.1 hp2 _)
  rw [encBlock, decBlock_lab hp2 hodd, sqE_rhoE hp h4, lab_eltD ω.1 hs, sigmaK_sigmaK (lab_lt ω.1 hp2 _),
    eltD_lab ω.1 hp2, sqE_rhoE hp h4, add_neg_cancel_right, lab_eltD ω.1 hv]

theorem p2ring_correct [NeZero B] (hp : p.Prime) (h4 : p % 4 = 3) (hb : 2 ^ b ≤ p) (hp2 : p ≤ 2 ^ w) :
    (p2ring B w p b).Correct := by
  intro W ω r
  change codeEquiv B b (fun j => (decBlock B w p b j
    (encBlock ω.1 (ω.2 j).1 (ω.2 j).2 (pad B w b W j))).run ((ringRO B w p).answer ω)) = W
  have h : ∀ j, (decBlock B w p b j (encBlock ω.1 (ω.2 j).1 (ω.2 j).2 (pad B w b W j))).run
      ((ringRO B w p).answer ω) = (codeEquiv B b).symm W j := by
    intro j
    rw [decBlock_encBlock hp h4 hp2 ω j (pad_lt hb hp2 W j)]
    exact (congrArg (bitsOfNat b) (bitsVal_pad (hb.trans hp2) W j)).trans (bitsOfNat_bitsVal _)
  simp only [h]
  exact (codeEquiv B b).apply_symm_apply W

theorem p2ring_codeHoldsW [NeZero B] (hbw : b ≤ w) : (p2ring B w p b).CodeHoldsW :=
  Nat.mul_le_mul_left B hbw

theorem p2ring_spaceBound [NeZero B] (hsp : 20 * w ≤ 21 * b) : (p2ring B w p b).SpaceBound Params.expansion := by
  change (((B * w + 0 : ℕ)) : ℚ) ≤ 21 / 20 * ((B * b : ℕ) : ℚ)
  rw [Nat.add_zero]
  push_cast
  have h : (20 : ℚ) * w ≤ 21 * b := by exact_mod_cast hsp
  have hB : (0 : ℚ) ≤ B := Nat.cast_nonneg _
  nlinarith

/-- One block's decoding: two rounds with a multiplication, two key queries, three ring queries. -/
theorem decBlock_bounded (j : Fin B) (c : Bits w) : (decBlock B w p b j c).RingBounded 2 2 3 := fun _ =>
  ⟨le_of_eq rfl, le_of_eq rfl, le_of_eq rfl⟩

/-- **Completeness (non-vacuity) of P2 in the ring model.** For every prime `p ≡ 3 mod 4` with `2^b ≤ p ≤ 2^w` and
`20·w ≤ 21·b`: `p2ring B w p b` is correct, holds `W` and is within the space bound; decoding one block
(`decBlock`, the program `dec` runs on every block) is `RingBounded 2 2 3`: two rounds that contain a
multiplication, two key queries and three ring queries, on every primitive; and storing `C` whole and answering by
lookup (`honestPre`, no query) passes the sequential audit with probability 1 for every `k`. -/
theorem complete (B w p b : ℕ) [NeZero B] [NeZero p] (hp : p.Prime) (h4 : p % 4 = 3) (hb : 2 ^ b ≤ p)
    (hp2 : p ≤ 2 ^ w) (hsp : 20 * w ≤ 21 * b) :
    (p2ring B w p b).Correct ∧ (p2ring B w p b).CodeHoldsW ∧ (p2ring B w p b).SpaceBound Params.expansion ∧
      (∀ (j : Fin B) (c : Bits w), (decBlock B w p b j c).RingBounded 2 2 3) ∧
      ∃ A₂ : SeqResponderOn (p2ring B w p b) ((p2ring B w p b).B * (p2ring B w p b).ℓ) 0 0 0,
        ∀ k W, pr (seqAcceptsOn (p2ring B w p b) k W (honestPre _) A₂) = 1 :=
  ⟨p2ring_correct hp h4 hb hp2, p2ring_codeHoldsW (le_of_two_pow_le (hb.trans hp2)), p2ring_spaceBound hsp,
    decBlock_bounded, ⟨(honestSeq _).step, fun _ _ _ _ _ => ringBounded_ret _ 0 0 0⟩,
    fun k W => honest_seq_accepted (p2ring B w p b) k W⟩

end Correct

end PousP2Ring
