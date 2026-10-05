import Proofs.Flock.Soundness.LigeritoFolds

/-!
# Ligerito's weights: the algebra of the final check (PROTOCOL.md §13.4, §13.6)

The final check evaluates the claim's weight at the fixed challenges through closed forms. They
all come from one identity: summing a product of per-bit factors against `eq(p, ·)` over the
hypercube gives the product of the factors' extensions (`sum_eqList_prod`). Two corollaries:
* the split coordinate's weight `(b ? u : 1)` extends to `cff(c) = 1 + c·(1 + u)`
  (`sum_eqList_split`);
* the novel-basis monomial `c ↦ X̂_c(ω)` extends to `Π_i (1 + p_i·(1 + Ŵ_i(ω)))`, the consistency
  weight of §13.4 (`sum_eqList_Xhat`).
-/

namespace FlockSoundness

open Game Model Accounting
open scoped ENNReal

/-- **Sums of products over the hypercube.** Against `eq(p, ·)`, a product of per-bit factors sums
to the product of the factors' multilinear extensions at `p`. -/
theorem sum_eqList_prod {L : Type} [CommRing L] :
    ∀ (p : List L) (f : ℕ → Bool → L),
      ∑ c ∈ Finset.range (2 ^ p.length), eqList p c * ∏ i ∈ Finset.range p.length,
          f i (c.testBit i) =
        ∏ i ∈ Finset.range p.length, ((1 + p.getD i 0) * f i false + p.getD i 0 * f i true)
  | [], f => by simp [eqList]
  | a :: l, f => by
      rw [List.length_cons, sum_range_pairs, Finset.prod_range_succ']
      simp only [List.getD_cons_zero, List.getD_cons_succ]
      rw [← sum_eqList_prod l (fun i b => f (i + 1) b), Finset.sum_mul]
      refine Finset.sum_congr rfl fun y _ => ?_
      rw [eqList_cons, eqList_cons, Finset.prod_range_succ', Finset.prod_range_succ']
      have h0 : (2 * y).testBit 0 = false := by simp [Nat.testBit_zero]
      have h1 : (2 * y + 1).testBit 0 = true := by simp [Nat.testBit_zero]
      have hs : ∀ i, (2 * y).testBit (i + 1) = y.testBit i := fun i => by
        rw [Nat.testBit_succ]; congr 1; omega
      have hs' : ∀ i, (2 * y + 1).testBit (i + 1) = y.testBit i := fun i => by
        rw [Nat.testBit_succ]; congr 1; omega
      simp only [h0, h1, hs, hs', show 2 * y / 2 = y by omega, show (2 * y + 1) / 2 = y by omega,
        Bool.false_eq_true, if_false, if_true]
      ring

/-- **The split coordinate.** The weight `(b ? u : 1)` of a split table extends to
`cff(c) = 1 + c·(1 + u)` (§13.6). -/
theorem sum_eqList_split {F K : Type} [Field F] [Field K] (A : Arith F K) (c : K) :
    ∑ b ∈ Finset.range 2, eqList [c] b * (if b.testBit 0 then A.u else 1) = cff A c := by
  have := sum_eqList_prod [c] (fun _ bit => if bit then A.u else 1)
  simp only [List.length_singleton, pow_one, Finset.range_one, Finset.prod_singleton,
    List.getD_cons_zero, Bool.false_eq_true, if_false, if_true] at this
  rw [this, cff]
  ring

/-- **The consistency weight** (§13.4). The novel-basis monomial `c ↦ X̂_c(ω)` over `L` variables
extends, at a point `p` of length `L`, to `Π_i (1 + p_i·(1 + Ŵ_i(ω)))`. -/
theorem sum_eqList_Xhat {F K : Type} [Field F] [Field K] (A : Arith F K) (p : List K) (ω : F) :
    ∑ c ∈ Finset.range (2 ^ p.length), eqList p c * A.embed (A.Xhat c p.length ω) =
      ∏ i ∈ Finset.range p.length, (1 + p.getD i 0 * (1 + A.embed (A.What i ω))) := by
  have := sum_eqList_prod p (fun i bit => if bit then A.embed (A.What i ω) else 1)
  have hX : ∀ c, A.embed (A.Xhat c p.length ω) =
      ∏ i ∈ Finset.range p.length, (if c.testBit i then A.embed (A.What i ω) else 1) := by
    intro c
    rw [Arith.Xhat, map_prod, ← Fin.prod_univ_eq_prod_range
      (fun i => if c.testBit i then A.embed (A.What i ω) else 1)]
    refine Finset.prod_congr rfl fun i _ => ?_
    split_ifs <;> simp
  simp_rw [hX, this]
  refine Finset.prod_congr rfl fun i _ => ?_
  simp only [Bool.false_eq_true, if_false, if_true]
  ring

end FlockSoundness
