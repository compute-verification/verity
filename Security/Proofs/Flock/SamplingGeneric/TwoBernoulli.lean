module

public import Proofs.Flock.SamplingGeneric.Two
public import Proofs.Flock.Claims.Sampling.EndToEndDeploy
import Proofs.Flock.Claims.Sampling.Draws

/-!
# The deployed first stage is below Bernoulli(Q/N)

The deployed two-stage design's first stage samples a uniform `Q`-subset of the population's `N` replay blocks
(`slotDesign`'s `⟨[subset Q]⟩` on `Slots.fileLayout`'s one class, each input gate in the block of the unit that reads it
first). The integrity profile reads that level as `Bernoulli(Q/N)`: it reports `∏_ρ (1 − Q/N + (Q/N)·e_ρ)`, each replay
block caught independently. That bounds `P̄_a` (`Design.Two.Pa`), so `M̄_δ` is inside the profile's confidence set and
the guarantee at `M̄_δ` is one at the profile's.

* `Design.Two.Pa_le_prod`: **a first stage below independent catches**. When the first stage misses each set `T` of
  replay blocks with probability at most `∏_{ρ∈T} (1 − q_ρ)`, `P̄_a ≤ ∏_ρ (1 − q_ρ + q_ρ·e_ρ)` (`Finset.prod_add`, each
  weight `∏_{ρ∈T}(1 − e_ρ)·∏_{ρ∉T} e_ρ` nonnegative).
* `Design.Two.subset_Pa_le`, `Design.Two.Pa_le_bernoulli`: the deployed first stage is below independent catches at
  `q_ρ = Q/N` (a uniform `Q`-subset misses `T` with probability `C(N − |T|, Q)/C(N, Q) ≤ (1 − Q/N)^{|T|}`,
  `choose_ratio_le_pow`), so `P̄_a ≤ ∏_{ρ<N} (1 − Q/N + (Q/N)·e_ρ)`.
* `Design.Two.M_subset_bernoulli`: `M̄_δ` is inside `{fp : δ < ∏_{ρ<N} (1 − Q/N + (Q/N)·e_ρ)}`.
-/

@[expose] public section

namespace Claims.Sampling

open Finset
open Claims.Sampling.EndToEndDraw (SubsetDesign)

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
population's `N` replay blocks misses each set `T` of replay blocks with probability at most `(1 − Q/N)^{|T|}`. -/
theorem Design.Two.subset_Pa_le (d : Design.Two) {Q : ℕ} (hd : SubsetDesign d.replay Q)
    {c : List (List (List ℕ))} (hP : d.part.classes = [c]) (hQ : Q ≤ c.flatten.length) :
    ∀ T ⊆ range d.part.blocks.length, d.replay.Pa d.part T ≤ ∏ ρ ∈ T, (1 - (Q : ℝ) / c.flatten.length) := by
  set N := c.flatten.length
  have hlen : d.part.blocks.length = N := EndToEndDraw.blocks_length hP
  intro T hT
  rw [hlen] at hT
  rw [EndToEndDraw.Pa_subset hd hP, hlen,
    EndToEndDraw.window_zero (n := N) (B := T) fun v hv => mem_range.1 (hT hv), prod_const]
  refine choose_ratio_le_pow hQ ((card_le_card hT).trans_eq (card_range N)) fun hN => ?_
  rw [div_mul_cancel₀ _ (by exact_mod_cast hN.ne')]

/-- **The deployed first stage is below Bernoulli(Q/N)**: at a first stage sampling a uniform `Q`-subset of the
population's `N` replay blocks, `P̄_a ≤ ∏_{ρ<N} (1 − Q/N + (Q/N)·e_ρ)`. -/
theorem Design.Two.Pa_le_bernoulli (d : Design.Two) (Pt : Partition) (fp : ℕ → ℕ → ℕ) {Q : ℕ}
    (hd : SubsetDesign d.replay Q) {c : List (List (List ℕ))} (hP : d.part.classes = [c])
    (hQ : Q ≤ c.flatten.length) :
    d.Pa Pt fp ≤ ∏ ρ ∈ range c.flatten.length,
      (1 - (Q : ℝ) / c.flatten.length + (Q : ℝ) / c.flatten.length * d.esc Pt fp ρ) := by
  have h := d.Pa_le_prod Pt fp (fun _ => (Q : ℝ) / c.flatten.length) (d.subset_Pa_le hd hP hQ)
  rwa [EndToEndDraw.blocks_length hP] at h

/-- **`M̄_δ` is inside the profile's confidence set**: a footprint with `P̄_a > δ` has
`∏_{ρ<N} (1 − Q/N + (Q/N)·e_ρ) > δ`. -/
theorem Design.Two.M_subset_bernoulli (d : Design.Two) (Pt : Partition) {Q : ℕ} (hd : SubsetDesign d.replay Q)
    {c : List (List (List ℕ))} (hP : d.part.classes = [c]) (hQ : Q ≤ c.flatten.length) (δ : ℝ) :
    d.M Pt δ ⊆ {fp | δ < ∏ ρ ∈ range c.flatten.length,
      (1 - (Q : ℝ) / c.flatten.length + (Q : ℝ) / c.flatten.length * d.esc Pt fp ρ)} :=
  fun fp hfp => lt_of_lt_of_le hfp (d.Pa_le_bernoulli Pt fp hd hP hQ)

end Claims.Sampling
