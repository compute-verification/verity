module

public import Definitions.Claims.Guarantees.SamplingTwo
public import Definitions.Claims.SamplingTwoReg
public import Proofs.Flock.SamplingGeneric.TwoBernoulli
import Proofs.Flock.Claims.Sampling.Extract

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
* `Design.Two.firstRate_subsetInputs`: that first stage catches each replay block at rate `Q/N`, the `FirstRate` the
  cap reads.
* `footprint_le_card`: a footprint counts at most its cell, as the cap reads it.
* `wrongAtBoundary`: the replay blocks wrong at their boundary, which hold an incorrect proof block at every interior.
  They read only the boundary trace, which serving commits before any coin. Every interior's `wrongBlocks` holds them
  (`wrongAtBoundary_subset`), and some interior's are exactly them (`exists_wrongBlocks_eq`), so they are the fewest
  replay blocks any interior marks incorrect.
* `sum_lt_of_le_prod`: the period's joint cap. Windows whose misses `≤ x^{m}` multiply to at least `δ` mark fewer than
  `K` replay blocks incorrect in all whenever `x^K < δ`, however many windows the period has (`prod_le_pow_sum`).
* `wrongAllAt`: at boundary values, over the grid of replay blocks and strata, each cell's incorrect proof blocks at its
  replay block's interior; its share of a cell is that cell's `wrongInAt` (`inter_cell_wrongAllAt`). The committed
  form's incorrect set is `wrongAllAt` at the committed traces (`CompiledTwo.FregTwo_eq`), and reads no second-stage
  coin, since the committed interior doesn't (`CompiledTwo.FregTwo_forkOn`, `CompiledTwo.iregVal_forkOn`).
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

/-- **The deployed first stage catches each replay block at rate `Q/N`** (`Design.Two.FirstRate`): a uniform
`Q`-subset of the population's `N` replay blocks and every input replay block (`Design.Two.subsetInputs_Pa_le`). -/
theorem Design.Two.firstRate_subsetInputs (d : Design.Two) {Q : ℕ} (hd : SubsetInputsDesign d.replay Q)
    {c ic : List (List (List ℕ))} (hP : d.part.classes = [c, ic]) (hQ : Q ≤ c.flatten.length) :
    d.FirstRate ((Q : ℝ) / c.flatten.length) := by
  intro T hT
  have hr : (0 : ℝ) ≤ 1 - (Q : ℝ) / c.flatten.length :=
    sub_nonneg.2 (div_le_one_of_le₀ (by exact_mod_cast hQ) (Nat.cast_nonneg _))
  refine (d.subsetInputs_Pa_le hd hP hQ T hT).trans ?_
  rw [← prod_const]
  refine prod_le_prod₀ (fun ρ _ => ?_) (fun ρ _ => ?_) <;> split_ifs
  · exact hr
  · simp
  · exact le_rfl
  · simpa using hr

open Classical in
/-- **The replay blocks wrong at their boundary** under the boundary trace `w`: each holds an incorrect proof block at
every interior, so an interior committed after the first stage's coins can't hide it from the second stage. It reads
only `w`'s boundary values. -/
noncomputable def wrongAtBoundary (H : Msg → Msg) (Cm : CommitmentScheme Bool) (s : AuditStatement) (Pt Pρ : Partition)
    (w : Wit) : Finset ℕ :=
  (range Pρ.blocks.length).filter fun ρ =>
    ∀ i : ℕ → Bool, ∃ t < Pt.strata.length, (wrongIn H Cm s Pt Pρ w ρ t i).Nonempty

/-- **Every interior marks the replay blocks wrong at their boundary**: at any interior values, each is among the
footprint's `wrongBlocks`. -/
theorem wrongAtBoundary_subset (H : Msg → Msg) (Cm : CommitmentScheme Bool) (s : AuditStatement) (Pt : Partition)
    (d : Design.Two) (w : Wit) (ι : ℕ → ℕ → Bool) :
    wrongAtBoundary H Cm s Pt d.part w ⊆ d.wrongBlocks Pt (footprint H Cm s Pt d.part w ι) := by
  intro ρ hρ
  simp only [wrongAtBoundary, mem_filter] at hρ
  obtain ⟨t, ht, hne⟩ := hρ.2 (ι ρ)
  simp only [Design.Two.wrongBlocks, mem_filter]
  exact ⟨hρ.1, t, ht, card_pos.2 hne⟩

/-- **Some interior marks only them**: at each replay block not wrong at its boundary, an interior with no incorrect
proof block makes the footprint's `wrongBlocks` exactly the replay blocks wrong at their boundary. -/
theorem exists_wrongBlocks_eq (H : Msg → Msg) (Cm : CommitmentScheme Bool) (s : AuditStatement) (Pt : Partition)
    (d : Design.Two) (w : Wit) :
    ∃ ι : ℕ → ℕ → Bool, d.wrongBlocks Pt (footprint H Cm s Pt d.part w ι) = wrongAtBoundary H Cm s Pt d.part w := by
  have key : ∀ ρ, ∃ i : ℕ → Bool, ρ ∉ wrongAtBoundary H Cm s Pt d.part w → ρ < d.part.blocks.length →
      ∀ t < Pt.strata.length, ¬ (wrongIn H Cm s Pt d.part w ρ t i).Nonempty := by
    intro ρ
    by_cases h : ∃ i : ℕ → Bool, ∀ t < Pt.strata.length, ¬ (wrongIn H Cm s Pt d.part w ρ t i).Nonempty
    · obtain ⟨i, hi⟩ := h
      exact ⟨i, fun _ _ => hi⟩
    · refine ⟨fun _ => false, fun hn hlt => absurd ?_ hn⟩
      simp only [wrongAtBoundary, mem_filter, mem_range]
      refine ⟨hlt, fun i => ?_⟩
      by_contra hc
      exact h ⟨i, fun t ht hne => hc ⟨t, ht, hne⟩⟩
  choose ι hι using key
  refine ⟨ι, ?_⟩
  ext ρ
  refine ⟨fun hρ => ?_, fun hρ => wrongAtBoundary_subset H Cm s Pt d w ι hρ⟩
  by_contra hn
  simp only [Design.Two.wrongBlocks, mem_filter, mem_range] at hρ
  obtain ⟨hlt, t, ht, hpos⟩ := hρ
  exact hι ρ hn hlt t ht (card_pos.1 hpos)

/-- **The windows' misses multiply to at most `x` to the period's incorrect replay blocks**: when each window `a` of a
period misses with probability at most `x^{m(a)}`, `0 ≤ x`, their product is at most `x^{Σ m}`. -/
theorem prod_le_pow_sum {α : Type} (l : List α) (pa : α → ENNReal) (m : α → ℕ) {x : ℝ} (hx : 0 ≤ x)
    (hpa : ∀ a ∈ l, pa a ≤ ENNReal.ofReal (x ^ m a)) : (l.map pa).prod ≤ ENNReal.ofReal (x ^ (l.map m).sum) := by
  induction l with
  | nil => simp
  | cons a l ih =>
    simp only [List.map_cons, List.prod_cons, List.sum_cons, pow_add,
      ENNReal.ofReal_mul (pow_nonneg hx _)]
    exact mul_le_mul' (hpa a List.mem_cons_self) (ih fun b hb => hpa b (List.mem_cons_of_mem a hb))

/-- **The period's joint cap**: windows whose misses `≤ x^{m}` multiply to at least `δ`, as a period's confidence set
holds them, mark fewer than `K` replay blocks incorrect in all whenever `x^K < δ` and `x ≤ 1`, however many windows the
period has. -/
theorem sum_lt_of_le_prod {α : Type} (l : List α) (pa : α → ENNReal) (m : α → ℕ) {x δ : ℝ} {K : ℕ} (hx : 0 ≤ x)
    (hx1 : x ≤ 1) (hK : x ^ K < δ) (hpa : ∀ a ∈ l, pa a ≤ ENNReal.ofReal (x ^ m a))
    (hin : ENNReal.ofReal δ ≤ (l.map pa).prod) : (l.map m).sum < K := by
  by_contra hS
  have hle : x ^ (l.map m).sum ≤ x ^ K := pow_le_pow_of_le_one hx hx1 (Nat.le_of_not_lt hS)
  have hlt : ENNReal.ofReal (x ^ (l.map m).sum) < ENNReal.ofReal δ :=
    ENNReal.ofReal_lt_ofReal_iff'.2 ⟨hle.trans_lt hK, (pow_nonneg hx K).trans_lt hK⟩
  exact absurd (hin.trans (prod_le_pow_sum l pa m hx hpa)) (not_le.2 hlt)

open Classical in
theorem wrongInAt_subset_cell (s : AuditStatement) (Pt Pρ : Partition) (val : ℕ → Option Bool) (ρ t : ℕ)
    (ι : ℕ → Bool) : wrongInAt s Pt Pρ val ρ t ι ⊆ cell Pt Pρ ρ t :=
  filter_subset _ _

theorem footprintAt_le_card (s : AuditStatement) (Pt Pρ : Partition) (val : ℕ → Option Bool) (ι : ℕ → ℕ → Bool)
    (ρ t : ℕ) : footprintAt s Pt Pρ val ι ρ t ≤ (cell Pt Pρ ρ t).card :=
  card_le_card (wrongInAt_subset_cell s Pt Pρ val ρ t (ι ρ))

/-- **The incorrect proof blocks at boundary values `val` and interior values `ι`** over the grid of replay blocks and
strata: each cell's at its replay block's interior. -/
noncomputable def wrongAllAt (s : AuditStatement) (Pt Pρ : Partition) (val : ℕ → Option Bool) (ι : ℕ → ℕ → Bool) :
    Finset ℕ :=
  (range Pρ.blocks.length ×ˢ range Pt.strata.length).biUnion fun p => wrongInAt s Pt Pρ val p.1 p.2 (ι p.1)

theorem wrongAllAt_subset (s : AuditStatement) (Pt Pρ : Partition) (val : ℕ → Option Bool) (ι : ℕ → ℕ → Bool) :
    wrongAllAt s Pt Pρ val ι ⊆ range Pt.blocks.length := by
  intro v hv
  obtain ⟨p, _, hvp⟩ := mem_biUnion.1 hv
  exact cell_subset Pt Pρ p.1 p.2 (wrongInAt_subset_cell s Pt Pρ val p.1 p.2 _ hvp)

/-- On the grid, a cell's share of `wrongAllAt` is its incorrect proof blocks at its replay block's interior. -/
theorem inter_cell_wrongAllAt (s : AuditStatement) (Pt Pρ : Partition) (val : ℕ → Option Bool) (ι : ℕ → ℕ → Bool)
    {ρ t : ℕ} (hρ : ρ < Pρ.blocks.length) (ht : t < Pt.strata.length) :
    wrongAllAt s Pt Pρ val ι ∩ cell Pt Pρ ρ t = wrongInAt s Pt Pρ val ρ t (ι ρ) := by
  ext v
  rw [mem_inter]
  constructor
  · rintro ⟨hv, hvc⟩
    obtain ⟨p, _, hvp⟩ := mem_biUnion.1 hv
    obtain ⟨h1, h2⟩ := cell_disj (wrongInAt_subset_cell s Pt Pρ val p.1 p.2 _ hvp) hvc
    subst h1
    subst h2
    exact hvp
  · intro hv
    exact ⟨mem_biUnion.2 ⟨(ρ, t), mem_product.2 ⟨mem_range.2 hρ, mem_range.2 ht⟩, hv⟩,
      wrongInAt_subset_cell s Pt Pρ val ρ t _ hv⟩

end Claims.Sampling

/-- **The per-window cap** (`Claims.Guarantees.CapTwo`): `Design.Two.Pa_le_pow` at the constant first-stage rate. -/
theorem Claims.SecurityProofs.CapTwo : Claims.Guarantees.CapTwo := by
  intro H Cm s Pt d r₁ r₂ h₁ h₂ hr₁ hr₁' hr₂ w ι
  refine d.Pa_le_pow Pt (fun _ => r₁) (fun T hT => ?_) (fun _ _ => ⟨le_rfl, hr₁'⟩) hr₁ hr₂ h₂
    (Claims.Sampling.footprint_le_card H Cm s Pt d.part w ι)
  show d.replay.Pa d.part T ≤ ∏ _ρ ∈ T, (1 - r₁)
  rw [Finset.prod_const]
  exact h₁ T hT

namespace Claims.SamplingGeneric

open Claims
open Claims.Sampling (AuditStatement Partition InScopeTwo forkOn forkOn_forkOn wrongAllAt)

namespace CompiledTwo

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}

/-- On a statement that parses and whose partitions evaluate, the committed incorrect set is `wrongAllAt` at the
committed boundary trace and interiors. -/
theorem FregTwo_eq (H : Msg → Msg) (L : CompiledTwo Coin A) {Ω : Type} (P : Prover Ω) {x : Msg}
    {s : AuditStatement} {Pt : Partition} {d : Sampling.Design.Two} (hs : L.parse x = some s)
    (hPt : L.partition s = some Pt) (hd : L.two s = some d) (ω : Ω) (r : Fin (L.V.k x) → Coin) :
    L.FregTwo H P x ω r = wrongAllAt s Pt d.part (L.regVal H P x ω) (L.iregVal H P x ω r) := by
  simp only [FregTwo, hs, hPt, hd, wrongAllAt]

/-- **The committed interior reads no second-stage coin.** -/
theorem iregVal_forkOn (H : Msg → Msg) (L : CompiledTwo Coin A) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω)
    (r e : Fin (L.V.k x) → Coin) : L.iregVal H P x ω (forkOn (L.stage2 x) r e) = L.iregVal H P x ω r := by
  funext ρ g
  simp only [iregVal, forkOn_forkOn]

/-- **The committed incorrect set reads no second-stage coin.** -/
theorem FregTwo_forkOn (H : Msg → Msg) (L : CompiledTwo Coin A) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω)
    (r e : Fin (L.V.k x) → Coin) : L.FregTwo H P x ω (forkOn (L.stage2 x) r e) = L.FregTwo H P x ω r := by
  simp only [FregTwo, iregVal_forkOn]

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem inScopeTwo_of (L : CompiledTwo Coin A) {x : Msg} {s : AuditStatement} {Pt : Partition}
    {d : Sampling.Design.Two} (hx : L.Scope x) (hs : L.parse x = some s) (hPt : L.partition s = some Pt)
    (hd : L.two s = some d) : InScopeTwo Pt d s := by
  obtain ⟨s', Pt', d', hs', hPt', hd', hin⟩ := hx
  rw [hs] at hs'
  cases hs'
  rw [hPt] at hPt'
  cases hPt'
  rw [hd] at hd'
  cases hd'
  exact hin

end CompiledTwo

end Claims.SamplingGeneric
