module

public import Definitions.Claims.Guarantees.SamplingTwo
public import Proofs.Flock.SamplingGeneric.TwoBernoulli

/-!
# A footprint in `M̄_δ` marks fewer than `K` replay blocks incorrect

The cap a two-stage audit puts on what one accepted run can hide. A footprint marks a replay block incorrect when one of
its cells holds an incorrect proof block (`Design.Two.wrongBlocks`). The second stage samples each cell `(ρ, t)` of
`n_{ρ,t}` proof blocks at `min(k_t, n_{ρ,t})`; when that is at least `r₂·n_{ρ,t}` in every cell, an incorrect replay
block escapes with probability at most `1 − r₂`, since a cell with one incorrect proof block escapes with probability
`1 − min(k, n)/n` (`escCell_one`) and more incorrect ones only lower it (`escCell_anti`).

* `Design.Two.Pa_le_pow`: at a first stage below independent catches at rates `q_ρ ≥ r₁` (`Design.Two.Pa_le_prod`),
  `P̄_a ≤ (1 − r₁·r₂)^{|wrongBlocks|}`.
* `Claims.SecurityProofs.CapTwo`: the same at the two rates as the lock states them, `Claims.Guarantees.CapTwo`.
* `Design.Two.card_wrongBlocks_lt`: so a footprint in `M̄_δ` marks fewer than every `K` with `(1 − r₁·r₂)^K ≤ δ`.
* `Design.Two.card_wrongBlocks_lt_subsetInputs`: the same at the deployed first stage, a uniform `Q`-subset of the
  population's `N` replay blocks and every input replay block (`Design.Two.subsetInputs_Pa_le`), at `r₁ = Q/N`.
* `footprint_le_card`: a footprint counts at most its cell, as the cap reads it.
-/

@[expose] public section

namespace Claims.Sampling

open Finset
open Claims.Sampling.EndToEndDraw (SubsetInputsDesign)

/-- A footprint counts at most its cell's proof blocks. -/
theorem footprint_le_card (H : Msg → Msg) (Cm : CommitmentScheme Bool) (s : AuditStatement) (Pt Pρ : Partition)
    (w : Wit) (ι : ℕ → ℕ → Bool) (ρ t : ℕ) : footprint H Cm s Pt Pρ w ι ρ t ≤ (cell Pt Pρ ρ t).card := by
  classical
  exact card_filter_le _ _

/-- **A cell with one incorrect proof block** escapes with probability `1 − min(k, n)/n`. -/
theorem escCell_one (k : ℕ) {n : ℕ} (hn : 0 < n) : escCell n k 1 = 1 - ((min k n : ℕ) : ℝ) / (n : ℝ) := by
  obtain ⟨m, rfl⟩ : ∃ m, n = m + 1 := ⟨n - 1, by omega⟩
  have hk : min k (m + 1) ≤ m + 1 := min_le_right _ _
  have hC : (0 : ℝ) < ((m + 1).choose (min k (m + 1)) : ℝ) := by exact_mod_cast Nat.choose_pos hk
  have hm : (0 : ℝ) < ((m + 1 : ℕ) : ℝ) := by positivity
  have key : ((m.choose (min k (m + 1)) : ℕ) : ℝ) * ((m + 1 : ℕ) : ℝ) =
      ((m + 1).choose (min k (m + 1)) : ℝ) * (((m + 1 : ℕ) : ℝ) - ((min k (m + 1) : ℕ) : ℝ)) := by
    rw [← Nat.cast_sub hk]
    exact_mod_cast Nat.choose_mul_succ_eq m (min k (m + 1))
  unfold escCell
  rw [Nat.add_sub_cancel, one_sub_div hm.ne', div_eq_div_iff hC.ne' hm.ne', key]
  ring

/-- A replay block's escape is at most each of its cells'. -/
theorem Design.Two.esc_le_escCell (d : Design.Two) (Pt : Partition) (fp : ℕ → ℕ → ℕ) (ρ : ℕ) {t : ℕ}
    (ht : t < Pt.strata.length) :
    d.esc Pt fp ρ ≤ escCell (cell Pt d.part ρ t).card (d.k.getD t 0) (fp ρ t) := by
  unfold Design.Two.esc
  rw [← mul_prod_erase (range Pt.strata.length)
    (fun t => escCell (cell Pt d.part ρ t).card (d.k.getD t 0) (fp ρ t)) (mem_range.2 ht)]
  exact mul_le_of_le_one_right (escCell_nonneg _ _ _)
    (prod_le_one₀ (fun _ _ => escCell_nonneg _ _ _) fun _ _ => escCell_le_one _ _ _)

/-- **An incorrect replay block's factor**: caught by the first stage at `q ≥ r₁`, with each of its cells sampled at
`r₂` or more, a replay block with an incorrect proof block contributes at most `1 − r₁·r₂` to `P̄_a`'s bound. -/
theorem Design.Two.factor_le (d : Design.Two) (Pt : Partition) (fp : ℕ → ℕ → ℕ) {ρ : ℕ} {q r₁ r₂ : ℝ}
    (hq : r₁ ≤ q) (hq0 : 0 ≤ q) (hr₂ : 0 ≤ r₂) (hcell : ∀ t, fp ρ t ≤ (cell Pt d.part ρ t).card)
    (hrate : ∀ t < Pt.strata.length, 0 < (cell Pt d.part ρ t).card →
      r₂ * (cell Pt d.part ρ t).card ≤ ((min (d.k.getD t 0) (cell Pt d.part ρ t).card : ℕ) : ℝ))
    (hw : ∃ t < Pt.strata.length, 0 < fp ρ t) :
    1 - q + q * d.esc Pt fp ρ ≤ 1 - r₁ * r₂ := by
  obtain ⟨t, ht, hpos⟩ := hw
  have hn : 0 < (cell Pt d.part ρ t).card := lt_of_lt_of_le hpos (hcell t)
  have hnR : (0 : ℝ) < (cell Pt d.part ρ t).card := by exact_mod_cast hn
  have he : d.esc Pt fp ρ ≤ 1 - r₂ := by
    refine (d.esc_le_escCell Pt fp ρ ht).trans ((escCell_anti _ _ hpos).trans ?_)
    rw [escCell_one _ hn, sub_le_sub_iff_left, le_div_iff₀ hnR]
    exact hrate t ht hn
  nlinarith [mul_le_mul_of_nonneg_left he hq0, mul_nonneg (sub_nonneg.2 hq) hr₂]

/-- **The acceptance probability at a footprint is at most `(1 − r₁·r₂)^m`**, `m` the replay blocks it marks
incorrect: at a first stage that misses each set `T` of replay blocks with probability at most `∏_{ρ∈T} (1 − q_ρ)`,
each `q_ρ` in `[r₁, 1]`, and a second stage sampling each nonempty cell `(ρ, t)` at `min(k_t, n_{ρ,t}) ≥ r₂·n_{ρ,t}`,
at a footprint that counts at most each cell. -/
theorem Design.Two.Pa_le_pow (d : Design.Two) (Pt : Partition) (q : ℕ → ℝ) {r₁ r₂ : ℝ}
    (hT : ∀ T ⊆ range d.part.blocks.length, d.replay.Pa d.part T ≤ ∏ ρ ∈ T, (1 - q ρ))
    (hq : ∀ ρ < d.part.blocks.length, r₁ ≤ q ρ ∧ q ρ ≤ 1) (hr₁ : 0 ≤ r₁) (hr₂ : 0 ≤ r₂)
    (hrate : ∀ ρ < d.part.blocks.length, ∀ t < Pt.strata.length, 0 < (cell Pt d.part ρ t).card →
      r₂ * (cell Pt d.part ρ t).card ≤ ((min (d.k.getD t 0) (cell Pt d.part ρ t).card : ℕ) : ℝ))
    {fp : ℕ → ℕ → ℕ} (hcell : ∀ ρ t, fp ρ t ≤ (cell Pt d.part ρ t).card) :
    d.Pa Pt fp ≤ (1 - r₁ * r₂) ^ (d.wrongBlocks Pt fp).card := by
  have hsub : d.wrongBlocks Pt fp ⊆ range d.part.blocks.length := filter_subset _ _
  refine (d.Pa_le_prod Pt fp q hT).trans ?_
  calc ∏ ρ ∈ range d.part.blocks.length, (1 - q ρ + q ρ * d.esc Pt fp ρ)
      ≤ ∏ ρ ∈ range d.part.blocks.length, (if ρ ∈ d.wrongBlocks Pt fp then 1 - r₁ * r₂ else 1) := by
        refine prod_le_prod₀ (fun ρ hρ => ?_) fun ρ hρ => ?_
        · obtain ⟨h1, h2⟩ := hq ρ (mem_range.1 hρ)
          nlinarith [d.esc_nonneg Pt fp ρ, mul_nonneg (le_trans hr₁ h1) (d.esc_nonneg Pt fp ρ)]
        · obtain ⟨h1, h2⟩ := hq ρ (mem_range.1 hρ)
          split_ifs with hw
          · exact d.factor_le Pt fp h1 (le_trans hr₁ h1) hr₂ (hcell ρ) (hrate ρ (mem_range.1 hρ))
              (mem_filter.1 hw).2
          · nlinarith [mul_le_mul_of_nonneg_left (d.esc_le_one Pt fp ρ) (le_trans hr₁ h1)]
    _ = (1 - r₁ * r₂) ^ (d.wrongBlocks Pt fp).card := by
        rw [prod_ite_mem, inter_eq_right.2 hsub, prod_const]

/-- **A footprint in `M̄_δ` marks fewer than `K` replay blocks incorrect**: at a first stage that misses each set `T`
of replay blocks with probability at most `∏_{ρ∈T} (1 − q_ρ)`, each `q_ρ` in `[r₁, 1]`, and a second stage sampling
each cell `(ρ, t)` at `min(k_t, n_{ρ,t}) ≥ r₂·n_{ρ,t}`, a footprint with `P̄_a > δ` that counts at most each cell
marks fewer than `K` replay blocks incorrect whenever `(1 − r₁·r₂)^K ≤ δ`. -/
theorem Design.Two.card_wrongBlocks_lt (d : Design.Two) (Pt : Partition) (q : ℕ → ℝ) {r₁ r₂ δ : ℝ} {K : ℕ}
    (hT : ∀ T ⊆ range d.part.blocks.length, d.replay.Pa d.part T ≤ ∏ ρ ∈ T, (1 - q ρ))
    (hq : ∀ ρ < d.part.blocks.length, r₁ ≤ q ρ ∧ q ρ ≤ 1) (hr₁ : 0 ≤ r₁) (hr₁' : r₁ ≤ 1) (hr₂ : 0 ≤ r₂)
    (hr₂' : r₂ ≤ 1)
    (hrate : ∀ ρ < d.part.blocks.length, ∀ t < Pt.strata.length, 0 < (cell Pt d.part ρ t).card →
      r₂ * (cell Pt d.part ρ t).card ≤ ((min (d.k.getD t 0) (cell Pt d.part ρ t).card : ℕ) : ℝ))
    (hK : (1 - r₁ * r₂) ^ K ≤ δ) {fp : ℕ → ℕ → ℕ} (hfp : fp ∈ d.M Pt δ)
    (hcell : ∀ ρ t, fp ρ t ≤ (cell Pt d.part ρ t).card) :
    (d.wrongBlocks Pt fp).card < K := by
  refine Nat.lt_of_not_le fun hge => ?_
  have hδ : δ < d.Pa Pt fp := hfp
  have hx0 : 0 ≤ 1 - r₁ * r₂ := by nlinarith [mul_nonneg (sub_nonneg.2 hr₁') hr₂]
  have hx1 : 1 - r₁ * r₂ ≤ 1 := by nlinarith [mul_nonneg hr₁ hr₂]
  have hpow : (1 - r₁ * r₂) ^ (d.wrongBlocks Pt fp).card ≤ (1 - r₁ * r₂) ^ K := pow_le_pow_of_le_one hx0 hx1 hge
  linarith [d.Pa_le_pow Pt q hT hq hr₁ hr₂ hrate hcell]

/-- **The cap at the deployed first stage**: a uniform `Q`-subset of the population's `N` replay blocks and every input
replay block, with each cell sampled at `min(k_t, n_{ρ,t}) ≥ r₂·n_{ρ,t}`: a footprint in `M̄_δ` that counts at most
each cell marks fewer than `K` replay blocks incorrect whenever `(1 − (Q/N)·r₂)^K ≤ δ`. -/
theorem Design.Two.card_wrongBlocks_lt_subsetInputs (d : Design.Two) (Pt : Partition) {Q : ℕ}
    (hd : SubsetInputsDesign d.replay Q) {c ic : List (List (List ℕ))} (hP : d.part.classes = [c, ic])
    (hQ : Q ≤ c.flatten.length) {r₂ δ : ℝ} {K : ℕ} (hr₂ : 0 ≤ r₂) (hr₂' : r₂ ≤ 1)
    (hrate : ∀ ρ < d.part.blocks.length, ∀ t < Pt.strata.length, 0 < (cell Pt d.part ρ t).card →
      r₂ * (cell Pt d.part ρ t).card ≤ ((min (d.k.getD t 0) (cell Pt d.part ρ t).card : ℕ) : ℝ))
    (hK : (1 - (Q : ℝ) / c.flatten.length * r₂) ^ K ≤ δ) {fp : ℕ → ℕ → ℕ} (hfp : fp ∈ d.M Pt δ)
    (hcell : ∀ ρ t, fp ρ t ≤ (cell Pt d.part ρ t).card) :
    (d.wrongBlocks Pt fp).card < K := by
  have hr₁ : (0 : ℝ) ≤ (Q : ℝ) / c.flatten.length := by positivity
  have hr₁' : (Q : ℝ) / c.flatten.length ≤ 1 := div_le_one_of_le₀ (by exact_mod_cast hQ) (Nat.cast_nonneg _)
  refine d.card_wrongBlocks_lt Pt _ (d.subsetInputs_Pa_le hd hP hQ) (fun ρ _ => ?_) hr₁ hr₁' hr₂ hr₂' hrate hK hfp
    hcell
  split_ifs
  · exact ⟨le_rfl, hr₁'⟩
  · exact ⟨hr₁', le_rfl⟩

end Claims.Sampling

/-- **The per-window cap** (`Claims.Guarantees.CapTwo`): `Design.Two.Pa_le_pow` at the constant first-stage rate. -/
theorem Claims.SecurityProofs.CapTwo : Claims.Guarantees.CapTwo := by
  intro H Cm s Pt d r₁ r₂ h₁ h₂ hr₁ hr₁' hr₂ w ι
  refine d.Pa_le_pow Pt (fun _ => r₁) (fun T hT => ?_) (fun _ _ => ⟨le_rfl, hr₁'⟩) hr₁ hr₂ h₂
    (Claims.Sampling.footprint_le_card H Cm s Pt d.part w ι)
  show d.replay.Pa d.part T ≤ ∏ _ρ ∈ T, (1 - r₁)
  rw [Finset.prod_const]
  exact h₁ T hT
