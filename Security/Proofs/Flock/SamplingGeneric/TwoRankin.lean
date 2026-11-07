module

public import Definitions.Claims.SamplingTwoRankin
public import Proofs.Flock.SamplingGeneric.TwoServed
public import Proofs.Flock.SamplingGeneric.TwoBernoulli

/-!
# Rankin's bound on the two-stage count

`AdmissibleTwoRankin`, `AdmissibleTwoRankinPatterns` and `AdmissibleTwoRankinServed`
(`Definitions.Claims.SamplingTwoRankin`): at a first stage below independent catches at rates `q_ρ`, the admissible
sets' count, their patterns' count and the served-output cap's sum are at most `δ^{−θ}` times a product over the
replay blocks.

* `stratumOf_lt`, `replayOf_lt`, `mem_blocksIn`, `blocksIn_disjoint`, `biUnion_inter_blocksIn`: a proof block of `Pt`
  lies in replay block `ρ`'s proof blocks (`blocksIn`) exactly when its replay block is `ρ`; distinct replay blocks'
  are disjoint, and in scope they cover every proof block, so a set `T` of proof blocks is the union of its pieces
  `T ∩ blocksIn ρ`.
* `bitsOf_eq_sum`, `esc_inter_blocksIn`, `passes_inter_blocksIn`: `b(T)` is the sum of its pieces', and replay block
  `ρ`'s escape at `T`'s counts reads only `T`'s piece in `ρ`.
* `Design.Two.passes_nonneg`, `Pa_le_passes`: at rates `q_ρ ≤ 1` each factor is nonnegative, and `P̄_a` is at most
  their product (`Design.Two.Pa_le_prod`).
* `one_le_rankin`: **Rankin's step**, `1[δ < P] ≤ (P/δ)^θ` at `θ ≥ 0`.
* `weight_le_rankin`, `admissibleTwo_rankin_weight`: **Rankin's bound at a weight that factors**. An admissible `T`
  whose weight is at most `∏_ρ w_ρ(T ∩ blocksIn ρ)` weighs at most `δ^{−θ}·∏_ρ w_ρ(T ∩ blocksIn ρ)·passes^θ`;
  `T ↦ (T ∩ blocksIn ρ)_ρ` is injective on the admissible sets, into the product of the replay blocks' powersets, over
  which the sum of the product is the product of the sums (`Finset.prod_univ_sum`).
* `admissibleTwo_rankin`: **`AdmissibleTwoRankin`**, at `w_ρ(U) = 2^{b(U)}`.
* `admissibleTwo_rankin_patterns`: **`AdmissibleTwoRankinPatterns`**. With each request's proof blocks in one replay
  block `ρ_r`, a set's pattern is its pieces' patterns, each request's read in `ρ_r` (`blocksTo_inter_blocksIn`); a
  pattern weighs at most `δ^{−θ}` times the product of its pieces' weights and their groups' largest `passes^θ`, and the
  map to its pieces' patterns is injective.
* `min_sum_le_sum_min`, `sum_min_inter_le`, `servedMin_le_sum`: with distinct requests reading disjoint proof blocks,
  the served-output cap's exponent at `T` is at most `Σ_ρ min(b(T ∩ ρ), S_ρ)`: split each `b(T_r)` over the replay
  blocks, and in each, the requests' pieces are disjoint subsets of `T ∩ ρ` and their served bits at most `S_ρ`.
* `admissibleTwo_rankin_served`: **`AdmissibleTwoRankinServed`**, at `w_ρ(U) = 2^{min(b(U), S_ρ)}`.
* `bernoulli_Pa_le`: a Bernoulli first stage `bernoulli:NUM/DEN` on a statement in scope misses each set `T` of replay
  blocks with probability `(1 − NUM/DEN)^{|T|}`, so it meets the rates' hypothesis at `q_ρ = NUM/DEN ≤ 1`. The deployed
  first stage meets it at `Q/N` and `1` (`Design.Two.subsetInputs_Pa_le`).
-/

@[expose] public section

namespace Claims.Sampling

open Finset

theorem stratumIdx_lt : ∀ (ns : List ℕ) {v : ℕ}, v < ns.sum → stratumIdx ns v < ns.length
  | [], v, h => by simp at h
  | n :: ns, v, h => by
    unfold stratumIdx
    split_ifs with hv
    · simp
    · have := stratumIdx_lt ns (v := v - n) (by simp only [List.sum_cons] at h; omega)
      simp only [List.length_cons]
      omega

/-- Every proof block's stratum is one of `Pt`'s strata. -/
theorem stratumOf_lt {P : Partition} {v : ℕ} (hv : v < P.blocks.length) : P.stratumOf v < P.strata.length := by
  have h := stratumIdx_lt (P.strata.map List.length) (v := v) (by rw [← List.length_flatten]; exact hv)
  simpa [Partition.stratumOf] using h

/-- A proof block's replay block is one of the replay partition's blocks. -/
theorem replayOf_lt {Pt Pρ : Partition} {v ρ : ℕ} (h : replayOf Pt Pρ v = some ρ) : ρ < Pρ.blocks.length := by
  unfold replayOf at h
  exact List.mem_range.1 (List.mem_of_find?_eq_some h)

theorem mem_blocksIn {Pt Pρ : Partition} {ρ v : ℕ} :
    v ∈ blocksIn Pt Pρ ρ ↔ v < Pt.blocks.length ∧ replayOf Pt Pρ v = some ρ := by
  unfold blocksIn
  simp only [mem_biUnion, mem_range]
  constructor
  · rintro ⟨t, -, hv⟩
    exact ⟨mem_range.1 (cell_subset _ _ _ _ hv), replayOf_of_mem_cell hv⟩
  · rintro ⟨hv, hρ⟩
    exact ⟨_, stratumOf_lt hv, mem_filter.2 ⟨mem_range.2 hv, hρ, rfl⟩⟩

theorem blocksIn_disjoint {Pt Pρ : Partition} {ρ ρ' : ℕ} (h : ρ ≠ ρ') :
    Disjoint (blocksIn Pt Pρ ρ) (blocksIn Pt Pρ ρ') :=
  disjoint_left.2 fun _ hv hv' => h (Option.some.inj ((mem_blocksIn.1 hv).2.symm.trans (mem_blocksIn.1 hv').2))

/-- **The pieces cover**: in scope, a set of proof blocks is the union of its pieces in the replay blocks. -/
theorem biUnion_inter_blocksIn {Pt Pρ : Partition} (hcov : ∀ v < Pt.blocks.length, (replayOf Pt Pρ v).isSome)
    {T : Finset ℕ} (hT : T ⊆ range Pt.blocks.length) :
    (univ : Finset (Fin Pρ.blocks.length)).biUnion (fun i => T ∩ blocksIn Pt Pρ i) = T := by
  ext v
  simp only [mem_biUnion, mem_univ, true_and, mem_inter]
  constructor
  · rintro ⟨_, hv, _⟩
    exact hv
  · intro hv
    have hvN := mem_range.1 (hT hv)
    obtain ⟨ρ, hρ⟩ := Option.isSome_iff_exists.1 (hcov v hvN)
    exact ⟨⟨ρ, replayOf_lt hρ⟩, hv, mem_blocksIn.2 ⟨hvN, hρ⟩⟩

theorem bitsOf_eq_sum (s : AuditStatement) {Pt Pρ : Partition}
    (hcov : ∀ v < Pt.blocks.length, (replayOf Pt Pρ v).isSome) {T : Finset ℕ} (hT : T ⊆ range Pt.blocks.length) :
    bitsOf s Pt T = ∑ i : Fin Pρ.blocks.length, bitsOf s Pt (T ∩ blocksIn Pt Pρ i) := by
  conv_lhs => rw [← biUnion_inter_blocksIn hcov hT]
  unfold bitsOf
  rw [sum_biUnion]
  intro i _ j _ hij
  exact (blocksIn_disjoint fun h => hij (Fin.ext h)).mono inter_subset_right inter_subset_right

theorem esc_inter_blocksIn (d : Design.Two) (Pt : Partition) (T : Finset ℕ) (ρ : ℕ) :
    d.esc Pt (counts Pt d.part (T ∩ blocksIn Pt d.part ρ)) ρ = d.esc Pt (counts Pt d.part T) ρ := by
  unfold Design.Two.esc
  refine prod_congr rfl fun t ht => ?_
  unfold counts
  rw [inter_assoc, inter_eq_right.2 (show cell Pt d.part ρ t ⊆ blocksIn Pt d.part ρ from
    subset_biUnion_of_mem (cell Pt d.part ρ) ht)]

theorem passes_inter_blocksIn (d : Design.Two) (Pt : Partition) (q : ℝ) (T : Finset ℕ) (ρ : ℕ) :
    d.passes Pt q (counts Pt d.part (T ∩ blocksIn Pt d.part ρ)) ρ = d.passes Pt q (counts Pt d.part T) ρ := by
  unfold Design.Two.passes
  rw [esc_inter_blocksIn]

theorem Design.Two.passes_nonneg (d : Design.Two) (Pt : Partition) {q : ℝ} (hq : q ≤ 1) (fp : ℕ → ℕ → ℕ) (ρ : ℕ) :
    0 ≤ d.passes Pt q fp ρ := by
  unfold Design.Two.passes
  have h0 := d.esc_nonneg Pt fp ρ
  have h1 := d.esc_le_one Pt fp ρ
  nlinarith [mul_nonneg (sub_nonneg.2 hq) (sub_nonneg.2 h1)]

theorem Design.Two.passes_le_one (d : Design.Two) (Pt : Partition) {q : ℝ} (hq : 0 ≤ q) (fp : ℕ → ℕ → ℕ) (ρ : ℕ) :
    d.passes Pt q fp ρ ≤ 1 := by
  unfold Design.Two.passes
  have h1 := d.esc_le_one Pt fp ρ
  nlinarith [mul_nonneg hq (sub_nonneg.2 h1)]

/-- `P̄_a` is at most the product of the replay blocks' factors `1 − q_ρ·(1 − e_ρ)`. -/
theorem Pa_le_passes (d : Design.Two) (Pt : Partition) (fp : ℕ → ℕ → ℕ) (q : ℕ → ℝ)
    (hq : ∀ T ⊆ range d.part.blocks.length, d.replay.Pa d.part T ≤ ∏ ρ ∈ T, (1 - q ρ)) :
    d.Pa Pt fp ≤ ∏ ρ ∈ range d.part.blocks.length, d.passes Pt (q ρ) fp ρ :=
  (d.Pa_le_prod Pt fp q hq).trans_eq (prod_congr rfl fun ρ _ => by unfold Design.Two.passes; ring)

/-- **Rankin's step**: `1[δ < P] ≤ (P/δ)^θ` at `θ ≥ 0`. -/
theorem one_le_rankin {δ θ P : ℝ} (hδ : 0 < δ) (hθ : 0 ≤ θ) (h : δ < P) : 1 ≤ δ ^ (-θ) * P ^ θ := by
  have h1 : 0 < δ ^ θ := Real.rpow_pos_of_pos hδ θ
  rw [Real.rpow_neg hδ.le, inv_mul_eq_div, le_div_iff₀ h1, one_mul]
  exact Real.rpow_le_rpow hδ.le h.le hθ

/-- An admissible set's weight, at most the product of its pieces' weights `w_ρ(U)`, is at most `δ^{−θ}` times its
pieces' terms `w_ρ(U)·passes(U)^θ`. -/
theorem weight_le_rankin {Pt : Partition} {d : Design.Two} {q : ℕ → ℝ} {δ θ : ℝ}
    (hq : ∀ T ⊆ range d.part.blocks.length, d.replay.Pa d.part T ≤ ∏ ρ ∈ T, (1 - q ρ)) (hq1 : ∀ ρ, q ρ ≤ 1)
    (hδ : 0 < δ) (hθ : 0 ≤ θ) {w : ℕ → Finset ℕ → ℝ} (hw : ∀ ρ U, 0 ≤ w ρ U) {x : ℝ} {T : Finset ℕ}
    (hT : T ∈ admissibleTwo Pt d δ) (hx : x ≤ ∏ i : Fin d.part.blocks.length, w i (T ∩ blocksIn Pt d.part i)) :
    x ≤ δ ^ (-θ) * ∏ i : Fin d.part.blocks.length,
      (w i (T ∩ blocksIn Pt d.part i) * d.passes Pt (q i) (counts Pt d.part (T ∩ blocksIn Pt d.part i)) i ^ θ) := by
  classical
  have hP : δ < ∏ ρ ∈ range d.part.blocks.length, d.passes Pt (q ρ) (counts Pt d.part T) ρ :=
    lt_of_lt_of_le (mem_filter.1 hT).2 (Pa_le_passes d Pt _ q hq)
  have hprod : (∏ i : Fin d.part.blocks.length, w i (T ∩ blocksIn Pt d.part i)) *
      (∏ ρ ∈ range d.part.blocks.length, d.passes Pt (q ρ) (counts Pt d.part T) ρ) ^ θ =
      ∏ i : Fin d.part.blocks.length, (w i (T ∩ blocksIn Pt d.part i) *
        d.passes Pt (q i) (counts Pt d.part (T ∩ blocksIn Pt d.part i)) i ^ θ) := by
    rw [← Real.finsetProd_rpow (range _) (fun ρ => d.passes Pt (q ρ) (counts Pt d.part T) ρ)
        (fun ρ _ => d.passes_nonneg Pt (hq1 ρ) _ _), ← Fin.prod_univ_eq_prod_range, ← prod_mul_distrib]
    exact prod_congr rfl fun i _ => by rw [passes_inter_blocksIn]
  have h0 : 0 ≤ ∏ i : Fin d.part.blocks.length, w i (T ∩ blocksIn Pt d.part i) := prod_nonneg fun i _ => hw i _
  rw [← hprod]
  calc x ≤ (∏ i : Fin d.part.blocks.length, w i (T ∩ blocksIn Pt d.part i)) * 1 := by rw [mul_one]; exact hx
    _ ≤ (∏ i : Fin d.part.blocks.length, w i (T ∩ blocksIn Pt d.part i)) * (δ ^ (-θ) *
          (∏ ρ ∈ range d.part.blocks.length, d.passes Pt (q ρ) (counts Pt d.part T) ρ) ^ θ) :=
        mul_le_mul_of_nonneg_left (one_le_rankin hδ hθ hP) h0
    _ = _ := by ring

/-- **Rankin's bound at a weight that factors over the replay blocks**: at rates `q_ρ ≤ 1` the first stage meets, a
weight `W` on the admissible sets at most the product `∏_ρ w_ρ(T ∩ blocksIn ρ)` of nonnegative weights of their pieces
sums to at most `δ^{−θ} ∏_ρ Σ_{U ⊆ ρ} w_ρ(U)·passes(U)^θ`. -/
theorem admissibleTwo_rankin_weight {Pt : Partition} {d : Design.Two} {q : ℕ → ℝ} {δ θ : ℝ}
    (hcov : ∀ v < Pt.blocks.length, (replayOf Pt d.part v).isSome)
    (hq : ∀ T ⊆ range d.part.blocks.length, d.replay.Pa d.part T ≤ ∏ ρ ∈ T, (1 - q ρ)) (hq1 : ∀ ρ, q ρ ≤ 1)
    (hδ : 0 < δ) (hθ : 0 ≤ θ) {w : ℕ → Finset ℕ → ℝ} (hw : ∀ ρ U, 0 ≤ w ρ U) {W : Finset ℕ → ℝ}
    (hW : ∀ T ∈ admissibleTwo Pt d δ, W T ≤ ∏ i : Fin d.part.blocks.length, w i (T ∩ blocksIn Pt d.part i)) :
    ∑ T ∈ admissibleTwo Pt d δ, W T ≤ δ ^ (-θ) * ∏ ρ ∈ range d.part.blocks.length,
      ∑ U ∈ (blocksIn Pt d.part ρ).powerset, w ρ U * d.passes Pt (q ρ) (counts Pt d.part U) ρ ^ θ := by
  classical
  let f : Fin d.part.blocks.length → Finset ℕ → ℝ := fun i U =>
    w i U * d.passes Pt (q i) (counts Pt d.part U) i ^ θ
  have hf0 : ∀ i U, 0 ≤ f i U := fun i U =>
    mul_nonneg (hw i U) (Real.rpow_nonneg (d.passes_nonneg Pt (hq1 i) _ _) θ)
  let Φ : Finset ℕ → Fin d.part.blocks.length → Finset ℕ := fun T i => T ∩ blocksIn Pt d.part i
  have hinj : Set.InjOn Φ (admissibleTwo Pt d δ) := by
    intro T hT T' hT' h
    rw [← biUnion_inter_blocksIn hcov (mem_powerset.1 (mem_filter.1 hT).1),
      ← biUnion_inter_blocksIn hcov (mem_powerset.1 (mem_filter.1 hT').1)]
    exact biUnion_congr rfl fun i _ => congr_fun h i
  have hpi : (admissibleTwo Pt d δ).image Φ ⊆
      Fintype.piFinset fun i : Fin d.part.blocks.length => (blocksIn Pt d.part i).powerset := by
    intro x hx
    obtain ⟨T, -, rfl⟩ := mem_image.1 hx
    exact Fintype.mem_piFinset.2 fun i => mem_powerset.2 inter_subset_right
  calc ∑ T ∈ admissibleTwo Pt d δ, W T
      ≤ ∑ T ∈ admissibleTwo Pt d δ, δ ^ (-θ) * ∏ i, f i (Φ T i) :=
        sum_le_sum fun T hT => weight_le_rankin hq hq1 hδ hθ hw hT (hW T hT)
    _ = δ ^ (-θ) * ∑ x ∈ (admissibleTwo Pt d δ).image Φ, ∏ i, f i (x i) := by
        rw [sum_image hinj, mul_sum]
    _ ≤ δ ^ (-θ) * ∑ x ∈ Fintype.piFinset (fun i : Fin d.part.blocks.length => (blocksIn Pt d.part i).powerset),
          ∏ i, f i (x i) :=
        mul_le_mul_of_nonneg_left (sum_le_sum_of_subset_of_nonneg hpi fun x _ _ => prod_nonneg fun i _ => hf0 i (x i))
          (Real.rpow_nonneg hδ.le _)
    _ = δ ^ (-θ) * ∏ ρ ∈ range d.part.blocks.length,
          ∑ U ∈ (blocksIn Pt d.part ρ).powerset, w ρ U * d.passes Pt (q ρ) (counts Pt d.part U) ρ ^ θ := by
        rw [← prod_univ_sum, ← Fin.prod_univ_eq_prod_range]

/-- **`AdmissibleTwoRankin`**. -/
theorem admissibleTwo_rankin : AdmissibleTwoRankin := by
  intro s Pt d q δ θ hs hq hq1 hδ hθ
  classical
  push_cast
  exact admissibleTwo_rankin_weight hs.2.2.2.1 hq hq1 hδ hθ (w := fun _ U => (2 : ℝ) ^ bitsOf s Pt U)
    (fun _ _ => pow_nonneg zero_le_two _) fun T hT => le_of_eq (by
      show (2 : ℝ) ^ bitsOf s Pt T = ∏ i : Fin d.part.blocks.length, (2 : ℝ) ^ bitsOf s Pt (T ∩ blocksIn Pt d.part i)
      rw [bitsOf_eq_sum s hs.2.2.2.1 (mem_powerset.1 (mem_filter.1 hT).1), prod_pow_eq_pow_sum])

theorem mem_blocksTo {C : Prog} {Pt : Partition} {R : Type} (req : Fin C.outs.length → R) {T : Finset ℕ} {r : R}
    {v : ℕ} : v ∈ blocksTo C Pt req T r ↔ v ∈ T ∧ ∃ g ∈ Pt.blocks.getD v [],
      C.Boundary (Pt.blocks.getD v []) g = true ∧ ∃ i, req i = r ∧ C.Reads C.outs[i] g := by
  classical
  unfold blocksTo
  exact mem_filter

theorem blocksTo_subset (C : Prog) (Pt : Partition) {R : Type} (req : Fin C.outs.length → R) (T : Finset ℕ) (r : R) :
    blocksTo C Pt req T r ⊆ T := by
  classical
  unfold blocksTo
  exact filter_subset _ _

theorem blocksTo_mono {C : Prog} {Pt : Partition} {R : Type} (req : Fin C.outs.length → R) {T T' : Finset ℕ}
    (h : T ⊆ T') (r : R) : blocksTo C Pt req T r ⊆ blocksTo C Pt req T' r := by
  classical
  intro v hv
  unfold blocksTo at hv ⊢
  exact mem_filter.2 ⟨h (mem_filter.1 hv).1, (mem_filter.1 hv).2⟩

/-- **A request reads in its replay block**: when every proof block request `r` reads lies in replay block `ρ_r`, a
set's proof blocks that `r` reads are its piece's in `ρ_r`, and its piece in any other replay block holds none. -/
theorem blocksTo_inter_blocksIn {s : AuditStatement} {Pt Pρ : Partition} {R : Type}
    (req : Fin s.prog.outs.length → R) {ρr : R → ℕ}
    (hρr : ∀ r, ∀ v ∈ blocksTo s.prog Pt req (range Pt.blocks.length) r, replayOf Pt Pρ v = some (ρr r))
    {T : Finset ℕ} (hT : T ⊆ range Pt.blocks.length) (ρ : ℕ) (r : R) :
    blocksTo s.prog Pt req (T ∩ blocksIn Pt Pρ ρ) r = if ρ = ρr r then blocksTo s.prog Pt req T r else ∅ := by
  classical
  have hto : ∀ v ∈ blocksTo s.prog Pt req T r, replayOf Pt Pρ v = some (ρr r) := fun v hv =>
    hρr r v (blocksTo_mono req hT r hv)
  ext v
  unfold blocksTo
  split_ifs with h
  · simp only [mem_filter, mem_inter]
    constructor
    · rintro ⟨⟨hv, -⟩, hr⟩
      exact ⟨hv, hr⟩
    · rintro ⟨hv, hr⟩
      refine ⟨⟨hv, mem_blocksIn.2 ⟨mem_range.1 (hT hv), ?_⟩⟩, hr⟩
      rw [h]
      exact hto v (mem_filter.2 ⟨hv, hr⟩)
  · simp only [mem_filter, mem_inter, notMem_empty, iff_false]
    rintro ⟨⟨hv, hB⟩, hr⟩
    exact h (Option.some.inj ((mem_blocksIn.1 hB).2.symm.trans (hto v (mem_filter.2 ⟨hv, hr⟩))))

theorem lightOf_empty (s : AuditStatement) (Pt : Partition) (n : ℕ) : lightOf s Pt n ∅ = none := by
  simp [lightOf]

/-- **`AdmissibleTwoRankinPatterns`**. -/
theorem admissibleTwo_rankin_patterns : AdmissibleTwoRankinPatterns := by
  intro s Pt d q δ θ Rq _ _ req hs hq hq1 hδ hθ hal
  classical
  choose ρr hρr using hal
  let pat : Finset ℕ → Rq → Option (Option (Finset ℕ)) := fun U r =>
    lightOf s Pt (servedBits req r) (blocksTo s.prog Pt req U r)
  let wt : (Rq → Option (Option (Finset ℕ))) → ℕ := fun c => ∑ r, lightBits s Pt (servedBits req r) (c r)
  let ψ : (Rq → Option (Option (Finset ℕ))) → Fin d.part.blocks.length → Rq → Option (Option (Finset ℕ)) :=
    fun p i r => if (i : ℕ) = ρr r then p r else none
  let cls : Fin d.part.blocks.length → (Rq → Option (Option (Finset ℕ))) → Finset (Finset ℕ) := fun i c =>
    (blocksIn Pt d.part i).powerset.filter fun U => pat U = c
  let g : Fin d.part.blocks.length → (Rq → Option (Option (Finset ℕ))) → ℝ := fun i c =>
    (2 : ℝ) ^ wt c * ⨆ U : cls i c, d.passes Pt (q i) (counts Pt d.part U.1) i ^ θ
  have hg0 : ∀ i c, 0 ≤ g i c := fun i c =>
    mul_nonneg (by positivity) (Real.iSup_nonneg fun U => Real.rpow_nonneg (d.passes_nonneg Pt (hq1 i) _ _) θ)
  have hpat : ∀ T ⊆ range Pt.blocks.length, ∀ i : Fin d.part.blocks.length,
      pat (T ∩ blocksIn Pt d.part i) = ψ (pat T) i := by
    intro T hT i
    funext r
    simp only [pat, ψ]
    rw [blocksTo_inter_blocksIn req hρr hT]
    split_ifs <;> simp [lightOf_empty]
  have hnone : ∀ T ⊆ range Pt.blocks.length, ∀ r, ¬ ρr r < d.part.blocks.length → pat T r = none := by
    intro T hT r hr
    have he : blocksTo s.prog Pt req T r = ∅ := eq_empty_of_forall_notMem fun v hv =>
      hr (replayOf_lt (hρr r v (blocksTo_mono req hT r hv)))
    simp only [pat, he]
    exact lightOf_empty s Pt _
  have hwt : ∀ T ⊆ range Pt.blocks.length, wt (pat T) = ∑ i, wt (ψ (pat T) i) := by
    intro T hT
    simp only [wt, ψ]
    rw [sum_comm]
    refine sum_congr rfl fun r _ => ?_
    have hl : ∀ i : Fin d.part.blocks.length,
        lightBits s Pt (servedBits req r) (if (i : ℕ) = ρr r then pat T r else none) =
          if (i : ℕ) = ρr r then lightBits s Pt (servedBits req r) (pat T r) else 0 := fun i => by
      split_ifs <;> rfl
    rw [sum_congr rfl fun i _ => hl i,
      Fin.sum_univ_eq_sum_range (fun i => if i = ρr r then lightBits s Pt (servedBits req r) (pat T r) else 0),
      sum_ite_eq']
    split_ifs with h
    · rfl
    · simp only [hnone T hT r fun h' => h (mem_range.2 h'), lightBits]
  have key : ∀ p ∈ (admissibleTwo Pt d δ).image pat, (2 : ℝ) ^ wt p ≤ δ ^ (-θ) * ∏ i, g i (ψ p i) := by
    intro p hp
    obtain ⟨T, hT, rfl⟩ := mem_image.1 hp
    obtain ⟨hTp, hTM⟩ := mem_filter.1 hT
    have hTN := mem_powerset.1 hTp
    have hP : δ < ∏ ρ ∈ range d.part.blocks.length, d.passes Pt (q ρ) (counts Pt d.part T) ρ :=
      lt_of_lt_of_le hTM (Pa_le_passes d Pt _ q hq)
    have hw : (2 : ℝ) ^ wt (pat T) = ∏ i, (2 : ℝ) ^ wt (ψ (pat T) i) := by
      rw [hwt T hTN, prod_pow_eq_pow_sum]
    have hpow : (∏ ρ ∈ range d.part.blocks.length, d.passes Pt (q ρ) (counts Pt d.part T) ρ) ^ θ =
        ∏ i : Fin d.part.blocks.length, d.passes Pt (q i) (counts Pt d.part (T ∩ blocksIn Pt d.part i)) i ^ θ := by
      rw [← Real.finsetProd_rpow (range _) (fun ρ => d.passes Pt (q ρ) (counts Pt d.part T) ρ)
        (fun ρ _ => d.passes_nonneg Pt (hq1 ρ) _ _), ← Fin.prod_univ_eq_prod_range]
      exact prod_congr rfl fun i _ => by rw [passes_inter_blocksIn]
    have hle : ∀ i, (2 : ℝ) ^ wt (ψ (pat T) i) *
        d.passes Pt (q i) (counts Pt d.part (T ∩ blocksIn Pt d.part i)) i ^ θ ≤ g i (ψ (pat T) i) := by
      intro i
      refine mul_le_mul_of_nonneg_left ?_ (by positivity)
      have hmem : T ∩ blocksIn Pt d.part i ∈ cls i (ψ (pat T) i) :=
        mem_filter.2 ⟨mem_powerset.2 inter_subset_right, hpat T hTN i⟩
      exact le_ciSup (f := fun U : cls i (ψ (pat T) i) => d.passes Pt (q i) (counts Pt d.part U.1) i ^ θ)
        (Set.finite_range _).bddAbove ⟨_, hmem⟩
    calc (2 : ℝ) ^ wt (pat T) = 2 ^ wt (pat T) * 1 := (mul_one _).symm
      _ ≤ 2 ^ wt (pat T) * (δ ^ (-θ) *
            (∏ ρ ∈ range d.part.blocks.length, d.passes Pt (q ρ) (counts Pt d.part T) ρ) ^ θ) :=
          mul_le_mul_of_nonneg_left (one_le_rankin hδ hθ hP) (by positivity)
      _ = δ ^ (-θ) * (2 ^ wt (pat T) *
            (∏ ρ ∈ range d.part.blocks.length, d.passes Pt (q ρ) (counts Pt d.part T) ρ) ^ θ) := by ring
      _ = δ ^ (-θ) * ∏ i, ((2 : ℝ) ^ wt (ψ (pat T) i) *
            d.passes Pt (q i) (counts Pt d.part (T ∩ blocksIn Pt d.part i)) i ^ θ) := by
          rw [hw, hpow, prod_mul_distrib]
      _ ≤ δ ^ (-θ) * ∏ i, g i (ψ (pat T) i) :=
          mul_le_mul_of_nonneg_left (prod_le_prod₀ (fun i _ => mul_nonneg (by positivity)
            (Real.rpow_nonneg (d.passes_nonneg Pt (hq1 i) _ _) θ)) fun i _ => hle i) (Real.rpow_nonneg hδ.le _)
  have hinj : Set.InjOn ψ ((admissibleTwo Pt d δ).image pat) := by
    intro p hp p' hp' h
    obtain ⟨T, hT, rfl⟩ := mem_image.1 hp
    obtain ⟨T', hT', rfl⟩ := mem_image.1 hp'
    funext r
    by_cases hr : ρr r < d.part.blocks.length
    · have := congr_fun (congr_fun h ⟨ρr r, hr⟩) r
      simpa [ψ] using this
    · rw [hnone T (mem_powerset.1 (mem_filter.1 hT).1) r hr, hnone T' (mem_powerset.1 (mem_filter.1 hT').1) r hr]
  have hpi : ((admissibleTwo Pt d δ).image pat).image ψ ⊆
      Fintype.piFinset fun i : Fin d.part.blocks.length => (blocksIn Pt d.part i).powerset.image pat := by
    intro x hx
    obtain ⟨p, hp, rfl⟩ := mem_image.1 hx
    obtain ⟨T, hT, rfl⟩ := mem_image.1 hp
    exact Fintype.mem_piFinset.2 fun i => mem_image.2
      ⟨T ∩ blocksIn Pt d.part i, mem_powerset.2 inter_subset_right, hpat T (mem_powerset.1 (mem_filter.1 hT).1) i⟩
  push_cast
  show ∑ p ∈ (admissibleTwo Pt d δ).image pat, (2 : ℝ) ^ wt p ≤ _
  calc ∑ p ∈ (admissibleTwo Pt d δ).image pat, (2 : ℝ) ^ wt p
      ≤ ∑ p ∈ (admissibleTwo Pt d δ).image pat, δ ^ (-θ) * ∏ i, g i (ψ p i) := sum_le_sum key
    _ = δ ^ (-θ) * ∑ x ∈ ((admissibleTwo Pt d δ).image pat).image ψ, ∏ i, g i (x i) := by
        rw [sum_image hinj, mul_sum]
    _ ≤ δ ^ (-θ) * ∑ x ∈ Fintype.piFinset
          (fun i : Fin d.part.blocks.length => (blocksIn Pt d.part i).powerset.image pat), ∏ i, g i (x i) :=
        mul_le_mul_of_nonneg_left (sum_le_sum_of_subset_of_nonneg hpi fun x _ _ => prod_nonneg fun i _ => hg0 i (x i))
          (Real.rpow_nonneg hδ.le _)
    _ = δ ^ (-θ) * ∏ ρ ∈ range d.part.blocks.length, rankinFactorPatterns s Pt d req (q ρ) θ ρ := by
        rw [← prod_univ_sum, ← Fin.prod_univ_eq_prod_range]
        rfl

theorem min_sum_le_sum_min {ι : Type*} (I : Finset ι) (a : ι → ℕ) (c : ℕ) :
    min (∑ i ∈ I, a i) c ≤ ∑ i ∈ I, min (a i) c := by
  classical
  induction I using Finset.induction_on with
  | empty => simp
  | @insert j I hj ih =>
    rw [sum_insert hj, sum_insert hj]
    omega

/-- **A replay block's served bits**: when distinct requests read disjoint proof blocks, the requests' terms
`min(b(T_r ∩ ρ), |r|)` sum to at most `min(b(T ∩ ρ), S)`, at a cap `S` on the served bits of the requests that read in
`ρ`. -/
theorem sum_min_inter_le {s : AuditStatement} {Pt Pρ : Partition} {R : Type} [Fintype R] [DecidableEq R]
    {req : Fin s.prog.outs.length → R}
    (hdisj : ∀ r r', r ≠ r' → Disjoint (blocksTo s.prog Pt req (range Pt.blocks.length) r)
      (blocksTo s.prog Pt req (range Pt.blocks.length) r'))
    {T : Finset ℕ} (hT : T ⊆ range Pt.blocks.length) (ρ S : ℕ)
    (hS : ∑ r ∈ univ.filter (fun r => (blocksTo s.prog Pt req (blocksIn Pt Pρ ρ) r).Nonempty),
      servedBits req r ≤ S) :
    ∑ r, min (bitsOf s Pt (blocksTo s.prog Pt req T r ∩ blocksIn Pt Pρ ρ)) (servedBits req r) ≤
      min (bitsOf s Pt (T ∩ blocksIn Pt Pρ ρ)) S := by
  have hF : ∀ r, r ∉ univ.filter (fun r => (blocksTo s.prog Pt req (blocksIn Pt Pρ ρ) r).Nonempty) →
      blocksTo s.prog Pt req T r ∩ blocksIn Pt Pρ ρ = ∅ := by
    intro r hr
    refine eq_empty_of_forall_notMem fun v hv => hr (mem_filter.2 ⟨mem_univ r, v, ?_⟩)
    obtain ⟨hvT, hvB⟩ := mem_inter.1 hv
    exact (mem_blocksTo req).2 ⟨hvB, ((mem_blocksTo req).1 hvT).2⟩
  have hsub : ∑ r ∈ univ.filter (fun r => (blocksTo s.prog Pt req (blocksIn Pt Pρ ρ) r).Nonempty),
      min (bitsOf s Pt (blocksTo s.prog Pt req T r ∩ blocksIn Pt Pρ ρ)) (servedBits req r) =
      ∑ r, min (bitsOf s Pt (blocksTo s.prog Pt req T r ∩ blocksIn Pt Pρ ρ)) (servedBits req r) :=
    sum_subset (subset_univ _) fun r _ hr => by simp [hF r hr, bitsOf]
  rw [← hsub]
  refine le_min ?_ ((sum_le_sum fun r _ => min_le_right _ _).trans hS)
  calc ∑ r ∈ univ.filter (fun r => (blocksTo s.prog Pt req (blocksIn Pt Pρ ρ) r).Nonempty),
        min (bitsOf s Pt (blocksTo s.prog Pt req T r ∩ blocksIn Pt Pρ ρ)) (servedBits req r)
      ≤ ∑ r, bitsOf s Pt (blocksTo s.prog Pt req T r ∩ blocksIn Pt Pρ ρ) :=
        (sum_le_sum fun r _ => min_le_left _ _).trans (sum_le_sum_of_subset (subset_univ _))
    _ = bitsOf s Pt (univ.biUnion fun r => blocksTo s.prog Pt req T r ∩ blocksIn Pt Pρ ρ) := by
        unfold bitsOf
        rw [sum_biUnion]
        intro r _ r' _ hrr'
        exact (hdisj r r' hrr').mono (inter_subset_left.trans (blocksTo_mono req hT r))
          (inter_subset_left.trans (blocksTo_mono req hT r'))
    _ ≤ bitsOf s Pt (T ∩ blocksIn Pt Pρ ρ) := by
        unfold bitsOf
        exact sum_le_sum_of_subset (biUnion_subset.2 fun r _ =>
          inter_subset_inter (blocksTo_subset _ _ _ _ _) subset_rfl)

/-- **The served-output cap's exponent factors over the replay blocks**: in scope, when distinct requests read
disjoint proof blocks and `S_ρ` bounds the served bits of the requests that read in replay block `ρ`, the exponent
`min(b(T), Σ_{r : T_r ≠ ∅} min(b(T_r), |r|))` is at most `Σ_ρ min(b(T ∩ ρ), S_ρ)`. -/
theorem servedMin_le_sum {s : AuditStatement} {Pt : Partition} {d : Design.Two} (hs : InScopeTwo Pt d s) {R : Type}
    [Fintype R] [DecidableEq R] {req : Fin s.prog.outs.length → R} {S : ℕ → ℕ}
    (hdisj : ∀ r r', r ≠ r' → Disjoint (blocksTo s.prog Pt req (range Pt.blocks.length) r)
      (blocksTo s.prog Pt req (range Pt.blocks.length) r'))
    (hS : ∀ ρ, ∑ r ∈ univ.filter (fun r => (blocksTo s.prog Pt req (blocksIn Pt d.part ρ) r).Nonempty),
      servedBits req r ≤ S ρ)
    {T : Finset ℕ} (hT : T ⊆ range Pt.blocks.length) :
    min (bitsOf s Pt T) (∑ r ∈ univ.filter (fun r => (blocksTo s.prog Pt req T r).Nonempty),
        min (bitsOf s Pt (blocksTo s.prog Pt req T r)) (servedBits req r)) ≤
      ∑ i : Fin d.part.blocks.length, min (bitsOf s Pt (T ∩ blocksIn Pt d.part i)) (S i) := by
  refine (min_le_right _ _).trans ?_
  calc ∑ r ∈ univ.filter (fun r => (blocksTo s.prog Pt req T r).Nonempty),
        min (bitsOf s Pt (blocksTo s.prog Pt req T r)) (servedBits req r)
      ≤ ∑ r, min (bitsOf s Pt (blocksTo s.prog Pt req T r)) (servedBits req r) :=
        sum_le_sum_of_subset (subset_univ _)
    _ ≤ ∑ r, ∑ i : Fin d.part.blocks.length,
          min (bitsOf s Pt (blocksTo s.prog Pt req T r ∩ blocksIn Pt d.part i)) (servedBits req r) :=
        sum_le_sum fun r _ => by
          rw [bitsOf_eq_sum s hs.2.2.2.1 ((blocksTo_subset s.prog Pt req T r).trans hT)]
          exact min_sum_le_sum_min _ _ _
    _ = ∑ i : Fin d.part.blocks.length, ∑ r,
          min (bitsOf s Pt (blocksTo s.prog Pt req T r ∩ blocksIn Pt d.part i)) (servedBits req r) := sum_comm
    _ ≤ _ := sum_le_sum fun i _ => sum_min_inter_le hdisj hT i (S i) (hS i)

/-- **`AdmissibleTwoRankinServed`**. -/
theorem admissibleTwo_rankin_served : AdmissibleTwoRankinServed := by
  intro s Pt d q δ θ Rq _ _ req S hs hq hq1 hδ hθ hdisj hS
  classical
  push_cast
  refine admissibleTwo_rankin_weight hs.2.2.2.1 hq hq1 hδ hθ (w := fun ρ U => (2 : ℝ) ^ min (bitsOf s Pt U) (S ρ))
    (fun _ _ => pow_nonneg zero_le_two _) fun T hT => ?_
  show (2 : ℝ) ^ _ ≤ ∏ i : Fin d.part.blocks.length, (2 : ℝ) ^ min (bitsOf s Pt (T ∩ blocksIn Pt d.part i)) (S i)
  rw [prod_pow_eq_pow_sum]
  exact pow_le_pow_right₀ one_le_two (servedMin_le_sum hs hdisj hS (mem_powerset.1 (mem_filter.1 hT).1))

/-- **A Bernoulli first stage meets the rates' hypothesis**: on a statement in scope whose first stage is
`bernoulli:NUM/DEN`, the first stage misses each set `T` of replay blocks with probability `∏_{ρ∈T} (1 − NUM/DEN)`, and
the rate is at most `1`. -/
theorem bernoulli_Pa_le {s : AuditStatement} {Pt : Partition} {d : Design.Two} (hs : InScopeTwo Pt d s) {num den : ℕ}
    (hb : d.replay.rules = [.bernoulli num den]) :
    (∀ T ⊆ range d.part.blocks.length, d.replay.Pa d.part T ≤ ∏ _ρ ∈ T, (1 - (num : ℝ) / den)) ∧
      (num : ℝ) / den ≤ 1 := by
  obtain ⟨hlen, hwf⟩ := hs.2.2.2.2.1
  rw [hb] at hlen hwf
  obtain ⟨c, hc⟩ : ∃ c, d.part.classes = [c] := by
    rcases hcl : d.part.classes with _ | ⟨c, _ | ⟨c', cs⟩⟩
    · rw [hcl] at hlen
      simp at hlen
    · exact ⟨c, rfl⟩
    · rw [hcl] at hlen
      simp at hlen
  have hwf1 : 0 < den ∧ num ≤ den := hwf (.bernoulli num den, c) (by simp [hc])
  refine ⟨fun T hT => ?_, div_le_one_of_le₀ (by exact_mod_cast hwf1.2) (Nat.cast_nonneg _)⟩
  rw [EndToEndDraw.blocks_length hc] at hT
  have hPa : d.replay.Pa d.part T =
      (OneStage.bernoulli num den).Pa (strataSizes c) (window 0 c.flatten.length T) * 1 := by
    unfold Design.Pa
    rw [hb, hc]
    rfl
  rw [hPa, mul_one, EndToEndDraw.window_zero fun v hv => mem_range.1 (hT hv)]
  show (((den - num : ℕ) : ℝ) / den) ^ T.card ≤ _
  rw [prod_const, Nat.cast_sub hwf1.2, sub_div, div_self (by exact_mod_cast hwf1.1.ne')]

end Claims.Sampling
