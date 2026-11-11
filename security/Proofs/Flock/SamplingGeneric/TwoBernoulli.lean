module

public import Proofs.Flock.SamplingGeneric.Two
public import Proofs.Flock.Claims.Sampling.EndToEndDeploy
import Proofs.Flock.Claims.Sampling.Draws

/-!
# The deployed first stage is below Bernoulli(Q/N)

The deployed two-stage design's first stage samples a uniform `Q`-subset of the population's `N` replay blocks and every
input replay block (`slotDesign`'s `⟨[subset Q, all]⟩` on `Slots.layout`'s two classes). The integrity profile reads
that level as `Bernoulli(Q/N)`: it reports `∏_ρ (1 − Q/N + (Q/N)·e_ρ)`, each replay block caught independently. That
bounds `P̄_a` (`Design.Two.Pa`), so `M̄_δ` is inside the profile's confidence set and the guarantee at `M̄_δ` is one at
the profile's.

* `Design.Two.Pa_le_prod`: **a first stage below independent catches**. When the first stage misses each set `T` of
  replay blocks with probability at most `∏_{ρ∈T} (1 − q_ρ)`, `P̄_a ≤ ∏_ρ (1 − q_ρ + q_ρ·e_ρ)` (`Finset.prod_add`, each
  weight `∏_{ρ∈T}(1 − e_ρ)·∏_{ρ∉T} e_ρ` nonnegative).
* `Design.Two.subsetInputs_Pa_le`, `Design.Two.Pa_le_bernoulli`: the deployed first stage is below independent
  catches at `q_ρ = Q/N` on the population (a uniform `Q`-subset misses `T` with probability
  `C(N − |T|, Q)/C(N, Q) ≤ (1 − Q/N)^{|T|}`, `choose_ratio_le_pow`) and `1` on the input class (`all` misses no set that
  meets it), so `P̄_a ≤ ∏_{ρ<N} (1 − Q/N + (Q/N)·e_ρ) · ∏_{ρ≥N} e_ρ`.
* `Design.Two.Pa_le_bernoulli_pop`, `Design.Two.M_subset_bernoulli`: the profile's product drops the input replay
  blocks' escapes (each at most `1`), and `M̄_δ` is inside `{fp : δ < ∏_{ρ<N} (1 − Q/N + (Q/N)·e_ρ)}`.
-/

@[expose] public section

namespace Claims.Sampling

open Finset
open Claims.Sampling.EndToEndDraw (SubsetInputsDesign)

/-- **A first stage below independent catches**: when the first stage misses each set `T` of replay blocks with
probability at most `∏_{ρ∈T} (1 − q_ρ)`, `P̄_a ≤ ∏_ρ (1 − q_ρ + q_ρ·e_ρ)`. -/
theorem Design.Two.Pa_le_prod (d : Design.Two) (Pt : Partition) (fp : ℕ → ℕ → ℕ) (q : ℕ → ℝ)
    (hT : ∀ T ⊆ range d.part.blocks.length, d.replay.Pa d.part T ≤ ∏ ρ ∈ T, (1 - q ρ)) :
    d.Pa Pt fp ≤ ∏ ρ ∈ range d.part.blocks.length, (1 - q ρ + q ρ * d.esc Pt fp ρ) := by
  have hprod : ∏ ρ ∈ range d.part.blocks.length, (1 - q ρ + q ρ * d.esc Pt fp ρ) =
      ∏ ρ ∈ range d.part.blocks.length, ((1 - q ρ) * (1 - d.esc Pt fp ρ) + d.esc Pt fp ρ) :=
    prod_congr rfl fun ρ _ => by ring
  rw [hprod, prod_add]
  unfold Design.Two.Pa
  refine sum_le_sum fun T hT' => ?_
  rw [prod_mul_distrib, mul_assoc]
  exact mul_le_mul_of_nonneg_right (hT T (mem_powerset.1 hT'))
    (mul_nonneg (prod_nonneg fun ρ _ => sub_nonneg.2 (d.esc_le_one Pt fp ρ))
      (prod_nonneg fun ρ _ => d.esc_nonneg Pt fp ρ))

/-- **The deployed first stage is below independent catches**: a first stage sampling a uniform `Q`-subset of the
population's `N` replay blocks and every input replay block misses each set `T` of replay blocks with probability at
most `∏_{ρ∈T} (1 − q_ρ)`, `q_ρ = Q/N` on the population and `1` on the inputs. -/
theorem Design.Two.subsetInputs_Pa_le (d : Design.Two) {Q : ℕ} (hd : SubsetInputsDesign d.replay Q)
    {c ic : List (List (List ℕ))} (hP : d.part.classes = [c, ic]) (hQ : Q ≤ c.flatten.length) :
    ∀ T ⊆ range d.part.blocks.length, d.replay.Pa d.part T ≤
      ∏ ρ ∈ T, (1 - if ρ < c.flatten.length then (Q : ℝ) / c.flatten.length else 1) := by
  set N := c.flatten.length
  set I := ic.flatten.length
  have hlen : d.part.blocks.length = N + I := EndToEndDraw.blocks_length_two hP
  intro T hT
  rw [hlen] at hT
  by_cases hw : window N I T = ∅
  · have hT' := EndToEndDraw.subset_range_of_window hT hw
    rw [EndToEndDraw.Pa_subsetInputs_of_window hd hP hT hw]
    have : ∏ ρ ∈ T, (1 - if ρ < N then (Q : ℝ) / N else 1) = (1 - (Q : ℝ) / N) ^ T.card := by
      rw [← prod_const]
      exact prod_congr rfl fun ρ hρ => by simp [mem_range.1 (hT' hρ)]
    rw [this]
    refine choose_ratio_le_pow hQ ((card_le_card hT').trans_eq (card_range N)) fun hN => ?_
    rw [div_mul_cancel₀ _ (by exact_mod_cast hN.ne')]
  · rw [EndToEndDraw.Pa_subsetInputs hd hP, EndToEndDraw.all_Pa_of_ne hw, mul_zero]
    obtain ⟨u, hu⟩ := nonempty_iff_ne_empty.2 hw
    obtain ⟨v, hv, -⟩ := mem_image.1 hu
    rw [prod_eq_zero (mem_filter.1 hv).1 (by simp [not_lt.2 (mem_filter.1 hv).2.1])]

/-- **The deployed first stage is below Bernoulli(Q/N)**: at a first stage sampling a uniform `Q`-subset of the
population's `N` replay blocks and every input replay block, `P̄_a ≤ ∏_{ρ<N} (1 − Q/N + (Q/N)·e_ρ) · ∏_{ρ≥N} e_ρ`. -/
theorem Design.Two.Pa_le_bernoulli (d : Design.Two) (Pt : Partition) (fp : ℕ → ℕ → ℕ) {Q : ℕ}
    (hd : SubsetInputsDesign d.replay Q) {c ic : List (List (List ℕ))} (hP : d.part.classes = [c, ic])
    (hQ : Q ≤ c.flatten.length) :
    d.Pa Pt fp ≤ (∏ ρ ∈ range c.flatten.length,
        (1 - (Q : ℝ) / c.flatten.length + (Q : ℝ) / c.flatten.length * d.esc Pt fp ρ)) *
      ∏ ρ ∈ Ico c.flatten.length (c.flatten.length + ic.flatten.length), d.esc Pt fp ρ := by
  have h := d.Pa_le_prod Pt fp (fun ρ => if ρ < c.flatten.length then (Q : ℝ) / c.flatten.length else 1)
    (d.subsetInputs_Pa_le hd hP hQ)
  set N := c.flatten.length
  set I := ic.flatten.length
  have hlen : d.part.blocks.length = N + I := EndToEndDraw.blocks_length_two hP
  rw [hlen, range_eq_Ico, ← prod_Ico_consecutive _ (Nat.zero_le N) (Nat.le_add_right N I), ← range_eq_Ico] at h
  refine h.trans (le_of_eq ?_)
  congr 1
  · exact prod_congr rfl fun ρ hρ => by simp [mem_range.1 hρ]
  · exact prod_congr rfl fun ρ hρ => by simp [not_lt.2 (mem_Ico.1 hρ).1]

/-- **The profile's product**: the input replay blocks' escapes dropped, `P̄_a ≤ ∏_{ρ<N} (1 − Q/N + (Q/N)·e_ρ)`. -/
theorem Design.Two.Pa_le_bernoulli_pop (d : Design.Two) (Pt : Partition) (fp : ℕ → ℕ → ℕ) {Q : ℕ}
    (hd : SubsetInputsDesign d.replay Q) {c ic : List (List (List ℕ))} (hP : d.part.classes = [c, ic])
    (hQ : Q ≤ c.flatten.length) :
    d.Pa Pt fp ≤ ∏ ρ ∈ range c.flatten.length,
      (1 - (Q : ℝ) / c.flatten.length + (Q : ℝ) / c.flatten.length * d.esc Pt fp ρ) := by
  have hp : (Q : ℝ) / c.flatten.length ≤ 1 := div_le_one_of_le₀ (by exact_mod_cast hQ) (Nat.cast_nonneg _)
  refine (d.Pa_le_bernoulli Pt fp hd hP hQ).trans (mul_le_of_le_one_right (prod_nonneg fun ρ _ => ?_)
    (prod_le_one₀ (fun ρ _ => d.esc_nonneg Pt fp ρ) fun ρ _ => d.esc_le_one Pt fp ρ))
  have := d.esc_nonneg Pt fp ρ
  nlinarith [div_nonneg (Nat.cast_nonneg (α := ℝ) Q) (Nat.cast_nonneg (α := ℝ) c.flatten.length)]

/-- **`M̄_δ` is inside the profile's confidence set**: a footprint with `P̄_a > δ` has
`∏_{ρ<N} (1 − Q/N + (Q/N)·e_ρ) > δ`. -/
theorem Design.Two.M_subset_bernoulli (d : Design.Two) (Pt : Partition) {Q : ℕ} (hd : SubsetInputsDesign d.replay Q)
    {c ic : List (List (List ℕ))} (hP : d.part.classes = [c, ic]) (hQ : Q ≤ c.flatten.length) (δ : ℝ) :
    d.M Pt δ ⊆ {fp | δ < ∏ ρ ∈ range c.flatten.length,
      (1 - (Q : ℝ) / c.flatten.length + (Q : ℝ) / c.flatten.length * d.esc Pt fp ρ)} :=
  fun fp hfp => lt_of_lt_of_le hfp (d.Pa_le_bernoulli_pop Pt fp hd hP hQ)

end Claims.Sampling
