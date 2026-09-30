import FlockSoundness.Assumptions

/-!
# Zero knowledge: T1, an hm96 leaf swapped for an ideal one

`docs/zk-proof-public.md` §4.1's T1: take any experiment in which a leaf's salt `y` is fresh, used for nothing else and
never revealed, while its digest `x` may depend on everything else the experiment samples. Replacing the leaf by an ideal
one changes the probability of any event by at most `δ₁`.

An hm96 leaf is a function of the pair `(b, c) = (x ⊕ M·y, H(sp ‖ y))`, and an ideal one of `(U, H(sp ‖ y))` with `U` fresh
and uniform. `Assumptions.Hm96Hiding` bounds the distance of `(M·y, c)` from `(U, c)` by `δ₁`. XOR by `x` is a bijection
that maps uniform to uniform, so for each value of the other randomness the two leaves are `δ₁` apart, and averaging keeps
the bound. Here XOR is the addition of the group `B`, so the proof holds for any group of digests.

`Table.view` has every leaf the final message doesn't open ideal (world `W₁`). T1 applied once per such leaf, `N_hid` on
each side, is Lemma B's `2·δ₁·N_hid`; that hybrid over the leaves is on paper.
-/

namespace FlockSoundness.ZK

open Game
open scoped ENNReal

theorem avg_prod_split {C D : Type} [Fintype C] [Fintype D] (f : C × D → ℝ≥0∞) :
    avg f = avg fun c => avg fun d => f (c, d) := by
  unfold avg
  rw [Fintype.sum_prod_type, Fintype.card_prod, Nat.cast_mul,
    ENNReal.mul_inv (Or.inr (ENNReal.natCast_ne_top _)) (Or.inl (ENNReal.natCast_ne_top _))]
  refine Finset.sum_congr rfl fun c _ => ?_
  rw [Finset.mul_sum]
  refine Finset.sum_congr rfl fun d _ => ?_
  ring

theorem prCoin_prod_split {C D : Type} [Fintype C] [Fintype D] (P : C → D → Prop) :
    prCoin (fun p : C × D => P p.1 p.2) = avg fun c => prCoin fun d => P c d := by
  unfold prCoin
  exact avg_prod_split _

theorem prCoin_equiv {C : Type} [Fintype C] (e : C ≃ C) (P : C → Prop) : prCoin (fun c => P (e c)) = prCoin P := by
  unfold prCoin avg
  exact Fintype.sum_equiv e _ _ (fun _ => rfl)

/-- **T1** (§4.1): in an experiment on uniform `(ω, y)` whose leaf is `H(x(ω) + M·y, c(y))`, replacing the leaf by the ideal
`H(U, c(y))` with a fresh uniform `U` changes the probability of any event `G` by at most `δ₁`. The digest `x` may depend on
anything in `ω`; the salt `y` is used only by this leaf. -/
theorem ideal_leaf_swap {Ω Y B C D : Type} [Fintype Ω] [Nonempty Ω] [Fintype Y] [AddCommGroup B] [Fintype B]
    (My : Y → B) (c : Y → C) {δ₁ : ℝ≥0∞} (hT1 : Assumptions.Hm96Hiding My c δ₁) (H : B × C → D) (x : Ω → B)
    (G : Ω → D → Prop) :
    prCoin (fun p : Ω × Y => G p.1 (H (x p.1 + My p.2, c p.2)))
      ≤ prCoin (fun p : Ω × (B × Y) => G p.1 (H (p.2.1, c p.2.2))) + δ₁ := by
  rw [prCoin_prod_split (fun ω y => G ω (H (x ω + My y, c y))),
    prCoin_prod_split (fun ω (u : B × Y) => G ω (H (u.1, c u.2)))]
  calc avg (fun ω => prCoin fun y => G ω (H (x ω + My y, c y)))
      ≤ avg (fun ω => prCoin (fun u : B × Y => G ω (H (u.1, c u.2))) + δ₁) := by
        refine avg_mono fun ω => ?_
        have h := hT1 fun bc => G ω (H (x ω + bc.1, bc.2))
        have e : prCoin (fun u : B × Y => G ω (H (x ω + u.1, c u.2))) = prCoin fun u : B × Y => G ω (H (u.1, c u.2)) :=
          prCoin_equiv ((Equiv.addLeft (x ω)).prodCongr (Equiv.refl Y)) (fun u : B × Y => G ω (H (u.1, c u.2)))
        exact h.trans (le_of_eq (by rw [e]))
    _ = avg (fun ω => prCoin fun u : B × Y => G ω (H (u.1, c u.2))) + δ₁ := by
        rw [avg_add, avg_const]

end FlockSoundness.ZK
