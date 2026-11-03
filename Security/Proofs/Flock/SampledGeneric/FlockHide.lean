module

public import Proofs.Flock.SampledGeneric.Defs
public import Proofs.Flock.Soundness.ZK.RealLeaves

/-!
# hm96's leaf commitment hides, in the generic interface (`Hider.Hiding`)

`GenericSampledZK` and `GenericSampledZKMul` take the commitment's hiding against every post-processing
(`Hider.Hiding`). Here hm96's leaf commitment to a vector of digests (leaf `i` is `H(x_i + M·y_i, c(y_i))` at its own
salt `y_i`) is `2·N·δ₁`-hiding over `N` leaves, from `Claims.Hm96Hiding M c δ₁`:

* `Hider.pr_andThen_exp`: post-processing a commitment is an average over the prover's coins;
* `avg_test_le`: on finite uniform coins, a bound on every event of two random variables is a bound on every test with
  values in `[0, 1]` (take the event that the first variable's value is likelier than the second's);
* `hm96_event_le`: the commitments to two vectors are within `2·N·δ₁` on every event, through ideal leaves, which read
  no digest (`ZK.ideal_leaves_swap_fintype`, then `ZK.real_leaves_swap_fintype`), as `Session.session_shvzk_hm96` pays
  `2·|Hid|·δ₁`;
* `hm96Hider_hiding`.
-/

@[expose] public section

namespace FlockSoundness.SampledGeneric

open Game
open scoped ENNReal

/-- Post-processing a commitment is the average, over the prover's coins, of the post-processing at each commitment. -/
theorem Hider.pr_andThen_exp {Tr Rnd Com W : Type} [Fintype Rnd] [Nonempty Rnd] (H : Hider Tr Rnd Com) (Xv : Tr)
    (f : Com → Exp W) (E : W → Prop) :
    ((H.exp Xv).andThen f).pr E = avg fun r => (f (H.commit Xv r)).pr E :=
  rfl

open Classical in
/-- An average of a function of `a` is a sum over any finite set holding `a`'s values, each weighted by its
probability. -/
theorem avg_comp_eq_sum {Ω C : Type} [Fintype Ω] (a : Ω → C) (S : Finset C) (hS : ∀ ω, a ω ∈ S)
    (f : C → ℝ≥0∞) : avg (fun ω => f (a ω)) = ∑ c ∈ S, f c * prCoin fun ω => a ω = c := by
  simp only [prCoin, avg, Finset.mul_sum]
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun ω _ => ?_
  rw [Finset.sum_eq_single_of_mem (a ω) (hS ω) fun c _ hc => by simp [Ne.symm hc]]
  simp [mul_comm]

open Classical in
/-- The probability that `x` lands in `P` is the sum over `P` of the probability of each value. -/
theorem prCoin_mem_eq_sum {Ω C : Type} [Fintype Ω] (x : Ω → C) (P : Finset C) :
    prCoin (fun ω => x ω ∈ P) = ∑ c ∈ P, prCoin fun ω => x ω = c := by
  simp only [prCoin, avg]
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun ω _ => ?_
  rw [← Finset.mul_sum]
  congr 1
  by_cases hx : x ω ∈ P
  · rw [Finset.sum_eq_single_of_mem (x ω) hx fun c _ hc => by simp [Ne.symm hc]]
    simp [hx]
  · rw [Finset.sum_eq_zero fun c hc => by simp [show x ω ≠ c from fun h => hx (by rw [h]; exact hc)]]
    simp [hx]

/-- A test with values in `[0, 1]` gains at most the excess mass where `pa` exceeds `pb`. -/
theorem sum_mul_le_filter {C : Type} (S : Finset C) (g pa pb : C → ℝ≥0∞) (hg : ∀ c, g c ≤ 1) :
    ∑ c ∈ S, g c * pa c ≤ ∑ c ∈ S, g c * pb c + ∑ c ∈ S.filter (fun c => pb c < pa c), (pa c - pb c) := by
  rw [Finset.sum_filter, ← Finset.sum_add_distrib]
  refine Finset.sum_le_sum fun c _ => ?_
  split_ifs with hq
  · calc g c * pa c = g c * pb c + g c * (pa c - pb c) := by rw [← mul_add, add_tsub_cancel_of_le hq.le]
      _ ≤ g c * pb c + (pa c - pb c) := add_le_add le_rfl (mul_le_of_le_one_left zero_le (hg c))
  · rw [add_zero]
    gcongr
    exact not_lt.1 hq

open Classical in
/-- **Events to tests**: if every event of `a` is at most `δ` likelier than the same event of `b`, every test with
values in `[0, 1]` has expectation at most `δ` more at `a` than at `b`. -/
theorem avg_test_le {Ω₁ Ω₂ C : Type} [Fintype Ω₁] [Fintype Ω₂] [Nonempty Ω₂] (a : Ω₁ → C) (b : Ω₂ → C)
    {δ : ℝ≥0∞} (h : ∀ G : C → Prop, prCoin (fun ω => G (a ω)) ≤ prCoin (fun ω => G (b ω)) + δ)
    (g : C → ℝ≥0∞) (hg : ∀ c, g c ≤ 1) :
    avg (fun ω => g (a ω)) ≤ avg (fun ω => g (b ω)) + δ := by
  obtain ⟨S, ha, hb⟩ : ∃ S : Finset C, (∀ ω, a ω ∈ S) ∧ ∀ ω, b ω ∈ S :=
    ⟨Finset.univ.image a ∪ Finset.univ.image b,
      fun ω => Finset.mem_union_left _ (Finset.mem_image_of_mem a (Finset.mem_univ ω)),
      fun ω => Finset.mem_union_right _ (Finset.mem_image_of_mem b (Finset.mem_univ ω))⟩
  obtain ⟨pa, hpa⟩ : ∃ pa : C → ℝ≥0∞, ∀ c, pa c = prCoin fun ω => a ω = c := ⟨_, fun _ => rfl⟩
  obtain ⟨pb, hpb⟩ : ∃ pb : C → ℝ≥0∞, ∀ c, pb c = prCoin fun ω => b ω = c := ⟨_, fun _ => rfl⟩
  have ea : avg (fun ω => g (a ω)) = ∑ c ∈ S, g c * pa c := by
    rw [avg_comp_eq_sum a S ha g]
    simp only [hpa]
  have eb : avg (fun ω => g (b ω)) = ∑ c ∈ S, g c * pb c := by
    rw [avg_comp_eq_sum b S hb g]
    simp only [hpb]
  rw [ea, eb]
  refine (sum_mul_le_filter S g pa pb hg).trans (add_le_add le_rfl ?_)
  obtain ⟨P, hP⟩ : ∃ P : Finset C, P = S.filter fun c => pb c < pa c := ⟨_, rfl⟩
  rw [← hP]
  have hev : prCoin (fun ω => a ω ∈ P) ≤ prCoin (fun ω => b ω ∈ P) + δ := h (· ∈ P)
  rw [prCoin_mem_eq_sum a P, prCoin_mem_eq_sum b P] at hev
  simp only [← hpa, ← hpb] at hev
  have h1 : ∑ c ∈ P, (pa c - pb c) + ∑ c ∈ P, pb c = ∑ c ∈ P, pa c := by
    rw [← Finset.sum_add_distrib]
    refine Finset.sum_congr rfl fun c hc => tsub_add_cancel_of_le ?_
    rw [hP] at hc
    exact (Finset.mem_filter.1 hc).2.le
  have hfin : ∑ c ∈ P, pb c ≠ ⊤ := ENNReal.sum_ne_top.2 fun c _ => by
    rw [hpb]
    exact ZK.prCoin_ne_top _
  refine (ENNReal.add_le_add_iff_right hfin).1 ?_
  rw [h1, add_comm δ]
  exact hev

/-- **hm96's leaves** for the digests `Xv` at the salts `y`: leaf `i` is `H(Xv_i + M·y_i, c(y_i))`. -/
def hm96Leaves {ι Y B C D : Type} [Add B] (My : Y → B) (c : Y → C) (H : B × C → D) (Xv : ι → B) (y : ι → Y) :
    ι → D :=
  fun i => H (Xv i + My (y i), c (y i))

/-- **hm96's leaf commitment** as a commitment to digest vectors under the prover's salts. -/
def hm96Hider {ι Y B C D : Type} [Add B] (My : Y → B) (c : Y → C) (H : B × C → D) : Hider (ι → B) (ι → Y) (ι → D) :=
  ⟨hm96Leaves My c H⟩

/-- **The commitments to two digest vectors are within `2·N·δ₁` on every event**, through ideal leaves. -/
theorem hm96_event_le {ι Y B C D : Type} [Fintype Y] [Nonempty Y] [AddCommGroup B] [Fintype B] [Fintype ι]
    [DecidableEq ι] (My : Y → B) (c : Y → C) {δ₁ : ℝ≥0∞} (hT1 : Claims.Hm96Hiding My c δ₁) (H : B × C → D)
    (G : (ι → D) → Prop) (Xv Xw : ι → B) :
    prCoin (fun y : ι → Y => G (hm96Leaves My c H Xv y)) ≤
      prCoin (fun y : ι → Y => G (hm96Leaves My c H Xw y)) + 2 * Fintype.card ι * δ₁ := by
  have e : ∀ Xu : ι → B, prCoin (fun p : Unit × (ι → Y) => G fun i => H (Xu i + My (p.2 i), c (p.2 i))) =
      prCoin fun y : ι → Y => G (hm96Leaves My c H Xu y) := fun Xu =>
    ZK.prCoin_congr_equiv (Equiv.punitProd (ι → Y)) (fun y => G (hm96Leaves My c H Xu y))
  have h1 : prCoin (fun p : Unit × (ι → Y) => G fun i => H (Xv i + My (p.2 i), c (p.2 i))) ≤
      prCoin (fun p : Unit × (ι → B × Y) => G fun i => H ((p.2 i).1, c (p.2 i).2)) + Fintype.card ι * δ₁ :=
    ZK.ideal_leaves_swap_fintype My c hT1 H (fun _ => Xv) (fun _ L => G L)
  have h2 : prCoin (fun p : Unit × (ι → B × Y) => G fun i => H ((p.2 i).1, c (p.2 i).2)) ≤
      prCoin (fun p : Unit × (ι → Y) => G fun i => H (Xw i + My (p.2 i), c (p.2 i))) + Fintype.card ι * δ₁ :=
    ZK.real_leaves_swap_fintype My c hT1 H (fun _ => Xw) (fun _ L => G L)
  rw [← e Xv, ← e Xw, mul_assoc, two_mul, ← add_assoc]
  exact h1.trans (add_le_add h2 le_rfl)

/-- **hm96's leaf commitment is `2·N·δ₁`-hiding** against every post-processing, from `Claims.Hm96Hiding M c δ₁`. -/
theorem hm96Hider_hiding {ι Y B C D : Type} [Fintype Y] [Nonempty Y] [AddCommGroup B] [Fintype B] [Fintype ι]
    [DecidableEq ι] (My : Y → B) (c : Y → C) {δ₁ : ℝ≥0∞} (hT1 : Claims.Hm96Hiding My c δ₁) (H : B × C → D) :
    (hm96Hider (ι := ι) My c H).Hiding (2 * Fintype.card ι * δ₁) := by
  intro W Xv Xw f E
  have key : ∀ Xa Xb : ι → B, avg (fun y => (f (hm96Leaves My c H Xa y)).pr E) ≤
      avg (fun y => (f (hm96Leaves My c H Xb y)).pr E) + 2 * Fintype.card ι * δ₁ := fun Xa Xb =>
    avg_test_le (hm96Leaves My c H Xa) (hm96Leaves My c H Xb) (fun G => hm96_event_le My c hT1 H G Xa Xb)
      (fun cm => (f cm).pr E) fun cm => prob_le_one E (f cm).g (f cm).s
  rw [Hider.pr_andThen_exp, Hider.pr_andThen_exp]
  exact ⟨key Xv Xw, key Xw Xv⟩

end FlockSoundness.SampledGeneric
