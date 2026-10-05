import ArkLib.Data.CodingTheory.JohnsonBound.Family
import FlockSoundness.Protocol.Defs

/-!
# The list size: at most `L₀` codewords near any level-0 table

Every phase of `rep_sound` pays a union bound over the codewords within the level-0 proximity
radius of the committed table (`closeMsgs`). This file bounds their number by `listBound₀ =
100·2^r` at rate `2^-r` (`closeMsgs_card_le`), from ArkLib's Johnson bound for interleaved
Reed–Solomon codes (`irs_lambda_le_johnson_mds`: at radius `1 − √ρ − η`, at most `1/(2ηρ)`
codewords). The bridge:

* a table, read position first (`tr`), is an interleaved word; an encoded message is a codeword of
  `irsCode` on the level's evaluation points, because each row is the evaluation of a polynomial of
  degree below `2^cols` (the novel basis, `Arith.Correct.xhat_poly`);
* "close" (`Close`: every lane agrees on a `1 − γ` fraction of positions) is relative block
  Hamming distance at most `γ`;
* encoding is injective (a nonzero polynomial of degree below `2^cols ≤ 2^d` cannot vanish on `2^d`
  distinct points), so distinct close messages are distinct close codewords.
-/

namespace FlockSoundness

open Model Accounting
open scoped ENNReal

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]

/-- The level's evaluation points, as an embedding. -/
def domainEmb (A : Arith F K) (hA : A.Correct) (d : ℕ) (hd : d ≤ 64) : Fin (2 ^ d) ↪ F :=
  ⟨fun q => A.omega q.val, fun q q' h => Fin.ext (hA.omega_injective d hd q.isLt q'.isLt h)⟩

theorem domainEmb_apply (A : Arith F K) (hA : A.Correct) (d : ℕ) (hd : d ≤ 64) (q : Fin (2 ^ d)) :
    domainEmb A hA d hd q = A.omega q.val := rfl

/-- A table read position first: an interleaved word. -/
def tr {lanes d : ℕ} (T : Table F lanes d) : Fin (2 ^ d) → Fin (2 ^ lanes) → F := fun q ℓ => T ℓ q

/-- The novel-basis polynomials of `Arith.Correct.xhat_poly`. -/
noncomputable def xP (A : Arith F K) (hA : A.Correct) (L : ℕ) (hL : L ≤ 40) :
    Fin (2 ^ L) → Polynomial F :=
  (hA.xhat_poly L hL).choose

theorem xP_natDegree (A : Arith F K) (hA : A.Correct) (L : ℕ) (hL : L ≤ 40) (c : Fin (2 ^ L)) :
    (xP A hA L hL c).natDegree < 2 ^ L := (hA.xhat_poly L hL).choose_spec.1 c

theorem xP_linearIndependent (A : Arith F K) (hA : A.Correct) (L : ℕ) (hL : L ≤ 40) :
    LinearIndependent F (xP A hA L hL) := (hA.xhat_poly L hL).choose_spec.2.1

theorem xP_eval (A : Arith F K) (hA : A.Correct) (L : ℕ) (hL : L ≤ 40) (c : Fin (2 ^ L)) (ω : F) :
    A.Xhat c.val L ω = (xP A hA L hL c).eval ω := (hA.xhat_poly L hL).choose_spec.2.2 c ω

/-- The polynomial of one message row. -/
noncomputable def rowPoly (A : Arith F K) (hA : A.Correct) {L : ℕ} (hL : L ≤ 40)
    (row : Fin (2 ^ L) → F) : Polynomial F :=
  ∑ c, Polynomial.C (row c) * xP A hA L hL c

theorem rowPoly_eval (A : Arith F K) (hA : A.Correct) {L : ℕ} (hL : L ≤ 40) (row : Fin (2 ^ L) → F)
    (q : ℕ) : (rowPoly A hA hL row).eval (A.omega q) = A.encodeAt L row q := by
  simp only [rowPoly, Polynomial.eval_finset_sum, Polynomial.eval_mul, Polynomial.eval_C,
    Arith.encodeAt, xP_eval A hA L hL]

theorem rowPoly_natDegree (A : Arith F K) (hA : A.Correct) {L : ℕ} (hL : L ≤ 40)
    (row : Fin (2 ^ L) → F) : (rowPoly A hA hL row).natDegree < 2 ^ L := by
  have h : (rowPoly A hA hL row).natDegree ≤ 2 ^ L - 1 := by
    refine Polynomial.natDegree_sum_le_of_forall_le _ _ fun c _ => ?_
    refine (Polynomial.natDegree_C_mul_le _ _).trans ?_
    have := xP_natDegree A hA L hL c
    omega
  have : 0 < 2 ^ L := Nat.two_pow_pos L
  omega


/-! ## The list size -/

/-- A `fast100` schedule is for the statement's `m`. -/
theorem fast100_m : ∀ m s, fast100 m = some s → s.m = m := by
  intro m s h
  have hm : ∀ m', fast100 m' ≠ none → 22 ≤ m' ∧ m' ≤ 35 := by
    intro m' hm'
    by_contra hc
    apply hm'
    unfold fast100
    split <;> simp_all
  obtain ⟨h1, h2⟩ := hm m (by simp [h])
  interval_cases m <;> (simp only [fast100, Option.some.injEq] at h; subst h; rfl)

end FlockSoundness
