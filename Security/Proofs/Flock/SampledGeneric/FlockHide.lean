module

public import Proofs.Flock.SampledGeneric.Defs
public import Proofs.Flock.Soundness.ZK.RealLeaves
public import Proofs.Flock.Soundness.Discharge.ZkCoins.Leaf

/-!
# hm96's hiding commitment as a `Hider`

The trace is a digest at each of `N = |ι|` hiding leaves, the prover's coins a salt per leaf, and leaf `i` of the
commitment to `X` under the salts `y` is `H(X i + My (y i), c (y i))`: hm96's `H(p_leaf ‖ x ⊕ Φy ‖ H(p_salt ‖ y))`, with
XOR the addition of the group `B` (`hm96Hider`).

`Claims.Hm96Hiding My c δ₁` bounds one leaf's distance from the ideal leaf `H(U, c y)`, one way. `Hider.Hiding` compares
the commitments to two traces, so each leaf costs `δ₁` from `X`'s leaf to the ideal and `δ₁` back to `X'`'s:

* `hm96Hider_commit_le`: on every event, the commitments to `X` and `X'` differ by at most `2·N·δ₁` (the hybrid over the
  leaves, `ideal_leaves_swap_fintype` then `real_leaves_swap_fintype`);
* `Hider.pr_andThen`: a post-processed commitment's probability is the average over the coins of the post-processing's;
* `hm96Hider_hiding`: `Hiding (2·N·δ₁)`, against every post-processing (`avg_le_of_prCoin` lifts the events to the
  post-processing's `[0,1]`-valued probabilities);
* `hm96Hider_hiding_card_fails`: `Hiding (N·δ₁)` does not follow from `Hm96Hiding` at `δ₁`. One leaf with a constant
  mask satisfies `Hm96Hiding` at `δ₁ = 1/2`, yet its commitment is the trace itself, `1` apart between two traces, so
  the factor `2` is tight.
-/

@[expose] public section

namespace FlockSoundness.SampledGeneric

open Game ZK Discharge.ZkCoins
open scoped ENNReal

/-- A post-processed commitment's probability: the average, over the prover's coins, of the post-processing's
probability at the commitment they give. -/
theorem Hider.pr_andThen {Tr Rnd Com W : Type} [Fintype Rnd] [Nonempty Rnd] (H : Hider Tr Rnd Com) (X : Tr)
    (f : Com → Exp W) (E : W → Prop) :
    ((H.exp X).andThen f).pr E = avg fun r => (f (H.commit X r)).pr E := rfl

theorem prCoin_unit_prod {Ω : Type} [Fintype Ω] (P : Ω → Prop) :
    prCoin (fun p : Unit × Ω => P p.2) = prCoin P :=
  (prCoin_prod_split fun (_ : Unit) (y : Ω) => P y).trans (avg_const _)

variable {ι Y B C D : Type} [Fintype ι] [DecidableEq ι] [Fintype Y] [Nonempty Y] [AddCommGroup B] [Fintype B]

/-- **hm96's hiding commitment to a trace of `N = |ι|` digests**: the prover's coins are a salt per leaf, and leaf `i`
is `H(X i + My (y i), c (y i))`, with `My` the pinned key's mask of a salt and `c` the salt's hash. -/
def hm96Hider (My : Y → B) (c : Y → C) (H : B × C → D) : Hider (ι → B) (ι → Y) (ι → D) where
  commit X y i := H (X i + My (y i), c (y i))

/-- **The hybrid over the leaves**: on every event, the commitments to two traces differ by at most `2·N·δ₁`, `N·δ₁`
from `X`'s leaves to ideal ones and `N·δ₁` from those to `X'`'s. -/
theorem hm96Hider_commit_le (My : Y → B) (c : Y → C) {δ₁ : ℝ≥0∞} (hT1 : Claims.Hm96Hiding My c δ₁)
    (H : B × C → D) (X X' : ι → B) (P : (ι → D) → Prop) :
    prCoin (fun y => P ((hm96Hider My c H).commit X y)) ≤
      prCoin (fun y => P ((hm96Hider My c H).commit X' y)) + 2 * Fintype.card ι * δ₁ := by
  show prCoin (fun y : ι → Y => P fun i => H (X i + My (y i), c (y i))) ≤
    prCoin (fun y : ι → Y => P fun i => H (X' i + My (y i), c (y i))) + 2 * Fintype.card ι * δ₁
  calc prCoin (fun y : ι → Y => P fun i => H (X i + My (y i), c (y i)))
      = prCoin (fun p : Unit × (ι → Y) => P fun i => H (X i + My (p.2 i), c (p.2 i))) :=
        (prCoin_unit_prod fun y : ι → Y => P fun i => H (X i + My (y i), c (y i))).symm
    _ ≤ prCoin (fun p : Unit × (ι → B × Y) => P fun i => H ((p.2 i).1, c (p.2 i).2)) + Fintype.card ι * δ₁ :=
        ideal_leaves_swap_fintype My c hT1 H (Ω := Unit) (fun _ => X) (fun _ L => P L)
    _ ≤ (prCoin (fun p : Unit × (ι → Y) => P fun i => H (X' i + My (p.2 i), c (p.2 i))) + Fintype.card ι * δ₁) +
          Fintype.card ι * δ₁ :=
        add_le_add (real_leaves_swap_fintype My c hT1 H (Ω := Unit) (fun _ => X') (fun _ L => P L)) le_rfl
    _ = prCoin (fun y : ι → Y => P fun i => H (X' i + My (y i), c (y i))) + 2 * Fintype.card ι * δ₁ := by
        rw [prCoin_unit_prod fun y : ι → Y => P fun i => H (X' i + My (y i), c (y i)), add_assoc, ← two_mul,
          ← mul_assoc]

/-- **hm96's commitment hides at `2·N·δ₁`**: against every post-processing, the commitments to any two traces are
`2·N·δ₁`-close, from `Claims.Hm96Hiding` at `δ₁` per leaf. -/
theorem hm96Hider_hiding (My : Y → B) (c : Y → C) {δ₁ : ℝ≥0∞} (hT1 : Claims.Hm96Hiding My c δ₁) (H : B × C → D) :
    (hm96Hider (ι := ι) My c H).Hiding (2 * Fintype.card ι * δ₁) := by
  classical
  have key : ∀ (W : Type) (X X' : ι → B) (f : (ι → D) → Exp W) (E : W → Prop),
      ((hm96Hider (ι := ι) My c H).exp X |>.andThen f).pr E ≤
        ((hm96Hider (ι := ι) My c H).exp X' |>.andThen f).pr E + 2 * Fintype.card ι * δ₁ := by
    intro W X X' f E
    rw [Hider.pr_andThen, Hider.pr_andThen]
    exact avg_le_of_prCoin ((hm96Hider My c H).commit X) ((hm96Hider My c H).commit X')
      (Finset.univ.image ((hm96Hider My c H).commit X) ∪ Finset.univ.image ((hm96Hider My c H).commit X'))
      (fun y => Finset.mem_union_left _ (Finset.mem_image_of_mem _ (Finset.mem_univ y)))
      (fun y => Finset.mem_union_right _ (Finset.mem_image_of_mem _ (Finset.mem_univ y)))
      (hm96Hider_commit_le My c hT1 H X X') (fun d => (f d).pr E) fun d => prob_le_one _ _ _
  exact fun W X X' f E => ⟨key W X X' f E, key W X' X f E⟩

/-- A constant mask hides at `δ₁ = 1/2`: the point `0` is `1/2` from uniform on `ZMod 2`. -/
theorem hm96Hiding_const_half : Claims.Hm96Hiding (fun _ : Unit => (0 : ZMod 2)) (fun _ : Unit => ()) (1 / 2) := by
  intro E
  by_cases hE : E (0, ())
  · have hc : (Fintype.card (ZMod 2 × Unit) : ℝ≥0∞) = 2 := by simp [ZMod.card]
    have hsum : (2 : ℝ≥0∞)⁻¹ ≤ Claims.PrENNReal (fun p : ZMod 2 × Unit => E (p.1, ())) := by
      unfold Claims.PrENNReal
      calc (2 : ℝ≥0∞)⁻¹
          = (Fintype.card (ZMod 2 × Unit) : ℝ≥0∞)⁻¹ * @ite _ (E (0, ())) (Classical.propDecidable _) 1 0 := by
            rw [hc, ite_eq_left hE, mul_one]
        _ ≤ _ := Finset.single_le_sum (f := fun p : ZMod 2 × Unit => (Fintype.card (ZMod 2 × Unit) : ℝ≥0∞)⁻¹ *
              @ite _ (E (p.1, ())) (Classical.propDecidable _) 1 0) (fun _ _ => zero_le) (Finset.mem_univ (0, ()))
    calc Claims.PrENNReal (fun _ : Unit => E (0, ()))
        = 1 := by simp [Claims.PrENNReal, hE]
      _ = 2⁻¹ + 1 / 2 := by rw [one_div, ENNReal.inv_two_add_inv_two]
      _ ≤ _ := add_le_add hsum le_rfl
  · calc Claims.PrENNReal (fun _ : Unit => E (0, ()))
        = 0 := by simp [Claims.PrENNReal, hE]
      _ ≤ _ := zero_le

/-- **`Hiding (N·δ₁)` does not follow from `Hm96Hiding` at `δ₁`**: one leaf with a constant mask satisfies
`Hm96Hiding` at `δ₁ = 1/2`, but its commitment is the trace itself, so on the event that the leaf is `0` the
commitments to `0` and `1` are `1` apart, more than `N·δ₁ = 1/2`. -/
theorem hm96Hider_hiding_card_fails :
    Claims.Hm96Hiding (fun _ : Unit => (0 : ZMod 2)) (fun _ : Unit => ()) (1 / 2) ∧
      ¬ (hm96Hider (ι := Unit) (fun _ : Unit => (0 : ZMod 2)) (fun _ : Unit => ()) id).Hiding
        (Fintype.card Unit * (1 / 2)) := by
  refine ⟨hm96Hiding_const_half, fun h => ?_⟩
  have e : ∀ x : ZMod 2,
      (((hm96Hider (ι := Unit) (fun _ : Unit => (0 : ZMod 2)) (fun _ : Unit => ()) id).exp fun _ => x).andThen
        fun d => ⟨.ret d, PUnit.unit⟩).pr (fun d => (d ()).1 = 0) = if x = 0 then 1 else 0 := by
    intro x
    rw [Hider.pr_andThen]
    by_cases hx : x = 0 <;> simp [avg, Exp.pr, prob, hm96Hider, hx]
  have h0 := (h (Unit → ZMod 2 × Unit) (fun _ => 0) (fun _ => 1) (fun d => ⟨.ret d, PUnit.unit⟩)
    fun d => (d ()).1 = 0).1
  have h10 : (1 : ZMod 2) ≠ 0 := by decide
  rw [e, e, ite_eq_left rfl, ite_eq_right h10, zero_add, Fintype.card_unit, Nat.cast_one, one_mul] at h0
  exact absurd h0 (not_le.2 (ENNReal.half_lt_self one_ne_zero ENNReal.one_ne_top))

end FlockSoundness.SampledGeneric
