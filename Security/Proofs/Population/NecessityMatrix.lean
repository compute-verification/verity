import Proofs.Population.NecessitySkeleton
import Proofs.Population.Transfer
import Mathlib.Data.Fintype.Powerset

/-!
# The policy × property matrix: what each policy needs beyond tier 1, and what each property buys

Tier 1 (`NecessitySkeleton`) gives the certificate; a policy is inferable from it exactly when a bound on its
amplification is known (`inference_iff`). These are the matrix's cells that a theorem settles, one section per
policy. Each cell is a sufficiency (the property bounds the amplification) or a counterexample (without it the
amplification is unbounded).

* **Integrity.** `integrity_inferred` (tier 1 alone, `π ∝ E`). `integrity_inferred_off`, **what an always-proved set
  buys**: the cone-egress bound is needed only off `A`, so a massive value that fans out costs nothing when it is
  always proved.
* **Control.** `card_egressWidth_le`, **control at the public widths**: the runs whose faulty proof blocks write at most
  `M` slots in all show at most `#{B : Σ_B w ≤ M} · |V|^M` egress values, with no bound on any unit's width;
  `card_widthSets_le` counts those sets when every unit writes. `control_inferred`: drawing by width, `w(u) ≤ C · π(u)`,
  an accepted run lies in that set at `M ≥ C ε` but with probability at most `(1 − ε)^k`. `wide_egress`, **what bounded
  width buys**: under the count draw one faulty proof block writing `N` slots shows `2^N` values, so a width bound is
  what the count draw needs, and the width-weighted draw needs none.
* **Skippable work.** `fan_skips`, **the late cone is necessary**: in a checked population of `N + 1` unit-weight units,
  one faulty proof block (`x := false`) makes every unit's work skippable with no unit lucky, `N + 1` skipped at wrong
  weight `1`, which is the cone `F = N + 1` (`fan_down`). `late_operand` and `early_operand`, **commit before the work
  coin is necessary**: an operand chosen after the coin `c` zeroes `(x + c) · b` at every coin, and one committed before
  it does so at one coin in `|𝔽|`, the luck rate. `skipped_work_le_wt_off`: the cone bound `F` is needed only off the
  always-proved set.
* **Weight isolation.** `isolation_of_range`, **range restriction suffices**: if a public bound `R u` holds every slot
  unit `u` reads in every admitted run, and no secret is upstream of the egress in `rangeFlow D R`, leakage is at most
  control. Addresses may be computed and corrupted inside their range: uncorruptible addressing (Theorem O1) is the
  case `R` = the padding. Necessity is `DynPopulation.leak_of_moved_read`: a corruptible address that moves a read onto
  a slot no honest run reads leaks that slot through one faulty proof block.
-/

namespace Population.Iop

open Finset

variable {U : Type}

/-! ## Integrity and work: what an always-proved set buys -/

section AlwaysProved

variable {S : Type} (P : Dataflow U S)

/-- **Integrity, inferred, with the cone bound needed only off the always-proved set**: drawing by cone egress off `A`
(`E(u) ≤ C · π(u)` for `u ∉ A`, anything on `A`), an accepted run differs from the honest run on more than `C ε` egress
slots with probability at most `(1 − ε)^k`. -/
theorem integrity_inferred_off [Fintype U] (h : P.Checked) {V : Type} (f : P.Fns V) (a : S → V) {honest : S → V}
    (hh : ∀ s ∈ P.input, honest s = a s) (hc : ∀ u, P.Correct f honest u) {π : U → ℝ} (hπ : Dist π)
    (A : Finset U) (k : ℕ) {ε : ℝ} (hε : ε ≤ 1) (O : Finset S) {C : ℝ} (hC : 0 ≤ C)
    (hE : ∀ u, u ∉ A → ((P.coneEgress O {u}).card : ℝ) ≤ C * π u) (σ : S → V) (hσ : ∀ s ∈ P.input, σ s = a s) :
    prob π k (fun d => Accepts (corr P f) A d σ ∧ C * ε < (damaged O σ honest).card) ≤ (1 - ε) ^ k := by
  classical
  let c : U → ℝ := fun u => if u ∈ A then 0 else ((P.coneEgress O {u}).card : ℝ)
  have hamp : AmpLe (corr P f) A c
      (fun τ => if ∀ s ∈ P.input, τ s = a s then ((damaged O τ honest).card : ℝ) else 0) 1 := by
    intro τ hd
    rw [one_mul, bad_corr]
    have hsum : ∑ u ∈ P.wrong f τ, c u = ∑ u ∈ P.wrong f τ, ((P.coneEgress O {u}).card : ℝ) :=
      sum_congr rfl fun u hu => ite_eq_right (disjoint_left.1 hd hu)
    rw [hsum]
    dsimp only
    split_ifs with hτ
    · exact_mod_cast card_damaged_le P h f a hh hc hτ O
    · exact sum_nonneg fun _ _ => Nat.cast_nonneg _
  have hc' : ∀ u, c u ≤ C * π u := by
    intro u
    by_cases hu : u ∈ A
    · simp only [c, ite_eq_left hu]
      exact mul_nonneg hC (hπ.nonneg u)
    · simp only [c, ite_eq_right hu]
      exact hE u hu
  have := inferable_of_amp π k hπ zero_le_one hC hamp hc' hε σ
  rw [ite_eq_left hσ, one_mul] at this
  exact this

/-- **The skipped work, with the cone bound needed only off the always-proved set**: when no always-proved unit is
wrong, `F` need bound only the cones rooted off `A`. -/
theorem skipped_work_le_wt_off [Fintype U] [DecidableEq U] (h : P.Checked) {V : Type} (f : P.Fns V) (a : S → V)
    {honest σ : S → V} (hh : ∀ s ∈ P.input, honest s = a s) (hc : ∀ u, P.Correct f honest u)
    (hσ : ∀ s ∈ P.input, σ s = a s) {K L : Finset U} (hl : P.SkipsLucky f honest σ K L) (wt : U → ℕ) (F : ℕ)
    (A : Finset U) (hA : Disjoint (P.wrong f σ) A) (hF : ∀ u, u ∉ A → ∑ v ∈ P.down {u}, wt v ≤ F * wt u) :
    ∑ u ∈ univ \ K, wt u ≤ F * ∑ u ∈ P.wrong f σ, wt u + ∑ u ∈ L, wt u := by
  calc ∑ u ∈ univ \ K, wt u ≤ ∑ u ∈ P.down (P.wrong f σ) ∪ L, wt u :=
        sum_le_sum_of_subset (P.skipped_sub h f a hh hc hσ hl)
    _ ≤ ∑ u ∈ P.down (P.wrong f σ), wt u + ∑ u ∈ L, wt u := Dataflow.sum_union_le _ _ _
    _ ≤ F * ∑ u ∈ P.wrong f σ, wt u + ∑ u ∈ L, wt u := by
        refine Nat.add_le_add_right ?_ _
        calc ∑ u ∈ P.down (P.wrong f σ), wt u
            ≤ ∑ u ∈ (P.wrong f σ).biUnion fun u => P.down {u}, wt u :=
              sum_le_sum_of_subset (P.down_subset_biUnion _)
          _ ≤ ∑ u ∈ P.wrong f σ, ∑ v ∈ P.down {u}, wt v := Dataflow.sum_biUnion_le _ _ _
          _ ≤ ∑ u ∈ P.wrong f σ, F * wt u := sum_le_sum fun u hu => hF u (disjoint_left.1 hA hu)
          _ = F * ∑ u ∈ P.wrong f σ, wt u := (mul_sum _ _ _).symm

end AlwaysProved

/-! ## Control: the public widths -/

section Control

variable {S : Type} (P : Dataflow U S)

/-- **Control at the public widths**: the runs whose faulty proof blocks write at most `M` slots in all show at most
`#{B : Σ_B w ≤ M} · |V|^M` egress values. No unit's width is bounded. -/
theorem card_egressWidth_le [Fintype U] [Fintype S] (hc : P.Covered) (ha : P.Acyclic) {V : Type} [Fintype V]
    [Nonempty V] (f : P.Fns V) (a : S → V) {X O : Finset S} (hX : ∀ s ∈ X, s ∉ P.up O) (M : ℕ) :
    (egressWidth P f a X O M).card ≤
      (univ.filter fun B : Finset U => ∑ u ∈ B, (P.writes u).card ≤ M).card * Fintype.card V ^ M := by
  classical
  set small := univ.filter fun B : Finset U => ∑ u ∈ B, (P.writes u).card ≤ M with hsmall
  have hsub : egressWidth P f a X O M ⊆ small.biUnion fun B => P.egress f a X O B := by
    intro e he
    simp only [egressWidth, mem_image, mem_filter, mem_univ, true_and] at he
    obtain ⟨σ, ⟨hσa, hσM⟩, rfl⟩ := he
    simp only [mem_biUnion]
    refine ⟨P.wrong f σ, by simpa [small] using hσM, ?_⟩
    simp only [Dataflow.egress, mem_image, mem_filter, mem_univ, true_and]
    exact ⟨σ, ⟨hσa, subset_rfl⟩, rfl⟩
  have hB : ∀ B ∈ small, (P.egress f a X O B).card ≤ Fintype.card V ^ M := by
    intro B hB
    simp only [small, mem_filter, mem_univ, true_and] at hB
    refine (P.card_egress_le hc ha f a hX).trans (Nat.pow_le_pow_right Fintype.card_pos ?_)
    exact card_biUnion_le.trans hB
  calc (egressWidth P f a X O M).card ≤ (small.biUnion fun B => P.egress f a X O B).card := card_le_card hsub
    _ ≤ ∑ B ∈ small, (P.egress f a X O B).card := card_biUnion_le
    _ ≤ ∑ _B ∈ small, Fintype.card V ^ M := sum_le_sum hB
    _ = small.card * Fintype.card V ^ M := by rw [sum_const, smul_eq_mul]

/-- When every unit writes, the wrong sets within the budget `M` have at most `M` units: at most
`Σ_{j ≤ M} C(|U|, j)` of them. -/
theorem card_widthSets_le [Fintype U] (hw : ∀ u, 1 ≤ (P.writes u).card) (M : ℕ) :
    (univ.filter fun B : Finset U => ∑ u ∈ B, (P.writes u).card ≤ M).card ≤
      ∑ j ∈ range (M + 1), (Fintype.card U).choose j := by
  classical
  have hsub : (univ.filter fun B : Finset U => ∑ u ∈ B, (P.writes u).card ≤ M) ⊆
      (range (M + 1)).biUnion fun j => powersetCard j univ := by
    intro B hB
    simp only [mem_filter, mem_univ, true_and] at hB
    simp only [mem_biUnion, mem_range, mem_powersetCard]
    refine ⟨B.card, Nat.lt_succ_of_le ?_, subset_univ _, rfl⟩
    calc B.card = ∑ _u ∈ B, 1 := by simp
      _ ≤ ∑ u ∈ B, (P.writes u).card := sum_le_sum fun u _ => hw u
      _ ≤ M := hB
  refine (card_le_card hsub).trans (card_biUnion_le.trans (le_of_eq (sum_congr rfl fun j _ => ?_)))
  rw [card_powersetCard, card_univ]

/-- **Control, inferred at the public widths**: drawing by write width (`w(u) ≤ C · π(u)`), an accepted run's egress
lies outside the width-`M` egress values, for any `M ≥ C ε`, with probability at most `(1 − ε)^k`. With
`card_egressWidth_le`, control is at most `log #{B : Σ_B w ≤ M} + M log |V|` bits. -/
theorem control_inferred [Fintype U] [Fintype S] {V : Type} [Fintype V] (f : P.Fns V) (a : S → V) (X O : Finset S)
    {π : U → ℝ} (hπ : Dist π) (A : Finset U) (k : ℕ) {ε : ℝ} (hε : ε ≤ 1) {C : ℝ} (hC : 0 ≤ C)
    (hw : ∀ u, ((P.writes u).card : ℝ) ≤ C * π u) (M : ℕ) (hM : C * ε ≤ M) (σ : S → V) (hσ : P.Holds a X σ) :
    prob π k (fun d => Accepts (corr P f) A d σ ∧ (fun o : O => σ o) ∉ egressWidth P f a X O M) ≤ (1 - ε) ^ k := by
  classical
  refine (prob_mono π k hπ.nonneg
    (E' := fun d => Accepts (corr P f) A d σ ∧ ¬ Certified (corr P f) π A ε σ)
    fun d hd => ⟨hd.1, fun hcert => hd.2 ?_⟩).trans (certificate π k hπ (corr P f) A hε σ)
  have hle : ((∑ u ∈ P.wrong f σ, (P.writes u).card : ℕ) : ℝ) ≤ M := by
    push_cast
    calc ∑ u ∈ P.wrong f σ, ((P.writes u).card : ℝ) ≤ ∑ u ∈ P.wrong f σ, C * π u :=
          sum_le_sum fun u _ => hw u
      _ = C * mass π (P.wrong f σ) := by rw [mass, mul_sum]
      _ ≤ C * ε := mul_le_mul_of_nonneg_left hcert.2 hC
      _ ≤ M := hM
  simp only [egressWidth, mem_image, mem_filter, mem_univ, true_and]
  exact ⟨σ, ⟨hσ, by exact_mod_cast hle⟩, rfl⟩

end Control

section Wide

/-- One unit, reading nothing, writing all `N` slots. -/
def wide (N : ℕ) : Dataflow Unit (Fin N) where
  input := ∅
  reads _ := ∅
  writes _ := univ

/-- It writes `false` everywhere. -/
def wideFns (N : ℕ) : (wide N).Fns Bool := fun _ _ _ => false

/-- **What bounded width buys**: one faulty proof block, of count cost `1`, shows every one of `2^N` egress values. The
count draw's amplification for control is at least `N` bits per faulty proof block, unbounded without a width bound; the
width draw charges this unit `N`. -/
theorem wide_egress (N : ℕ) (a : Fin N → Bool) :
    ((wide N).egress (wideFns N) a ∅ univ {()}).card = 2 ^ N := by
  classical
  have h : (wide N).egress (wideFns N) a ∅ univ {()} = univ := by
    refine eq_univ_of_forall fun e => ?_
    simp only [Dataflow.egress, mem_image, mem_filter, mem_univ, true_and]
    refine ⟨fun s => e ⟨s, mem_univ s⟩, ⟨fun s hs => by simp [wide] at hs, fun u _ => by simp⟩, rfl⟩
  rw [h, card_univ]
  simp

end Wide

/-! ## Skippable work: the late cone, and the operand before the coin -/

section Fan

/-- The fan's slots: `inl u` is unit `u`'s output (`inl none` is `x`, `inl (some i)` is `yᵢ`), `inr i` the input `bᵢ`. -/
abbrev Fs (N : ℕ) := Option (Fin N) ⊕ Fin N

/-- **The fan**: unit `none` writes `x` from nothing, and unit `some i` writes `yᵢ` from `x` and `bᵢ`. -/
def fan (N : ℕ) : Dataflow (Option (Fin N)) (Fs N) where
  input := univ.image Sum.inr
  reads
    | none => ∅
    | some i => {Sum.inl none, Sum.inr i}
  writes u := {Sum.inl u}

/-- `x := true`, and `yᵢ := x ∧ bᵢ`: honestly `yᵢ = bᵢ`, the work. -/
def fanFns (N : ℕ) : (fan N).Fns Bool
  | none => fun _ _ => true
  | some i => fun r _ => r ⟨Sum.inl none, by simp [fan]⟩ && r ⟨Sum.inr i, by simp [fan]⟩

/-- The honest run on the inputs `b`. -/
def fanHonest {N : ℕ} (b : Fin N → Bool) : Fs N → Bool := Sum.elim (fun u => u.elim true b) b

/-- **The skipping run**: `x := false`, so every `yᵢ = false` without reading `bᵢ`. -/
def fanSkip {N : ℕ} (b : Fin N → Bool) : Fs N → Bool := Sum.elim (fun _ => false) b

theorem mem_wrong_iff {S V : Type} [Fintype U] {P : Dataflow U S} {f : P.Fns V} {σ : S → V} {u : U} :
    u ∈ P.wrong f σ ↔ ¬ P.Correct f σ u := by
  classical
  simp [Dataflow.wrong]

theorem fan_correct_none {N : ℕ} {σ : Fs N → Bool} :
    (fan N).Correct (fanFns N) σ none ↔ σ (Sum.inl none) = true := by
  constructor
  · intro h
    exact h ⟨Sum.inl none, mem_singleton_self _⟩
  · intro h ⟨s, hs⟩
    have : s = Sum.inl none := by simpa [fan] using hs
    subst this
    exact h

theorem fan_correct_some {N : ℕ} {σ : Fs N → Bool} {i : Fin N} :
    (fan N).Correct (fanFns N) σ (some i) ↔ σ (Sum.inl (some i)) = (σ (Sum.inl none) && σ (Sum.inr i)) := by
  constructor
  · intro h
    exact h ⟨Sum.inl (some i), mem_singleton_self _⟩
  · intro h ⟨s, hs⟩
    have : s = Sum.inl (some i) := by simpa [fan] using hs
    subst this
    exact h

theorem fan_dep {N : ℕ} {u v : Option (Fin N)} (h : (fan N).Dep u v) : u = none ∧ v ≠ none := by
  obtain ⟨s, hs, hr⟩ := h
  have hs' : s = Sum.inl u := by simpa [fan] using hs
  subst hs'
  rcases v with _ | i
  · simp [fan] at hr
  · simp [fan] at hr
    exact ⟨hr, by simp⟩

/-- The fan is a checked population. -/
theorem fan_checked (N : ℕ) : (fan N).Checked := by
  refine ⟨fun s => ?_, fun u hu => ?_⟩
  · rcases s with u | i
    · refine ⟨fun h => by simp [fan] at h, fun _ => ⟨u, by simp [fan], fun v hv => ?_⟩⟩
      simpa [fan, eq_comm] using hv
    · exact ⟨fun _ v => by simp [fan], fun h => (h (by simp [fan])).elim⟩
  · obtain ⟨w, hw, hwu⟩ := Relation.TransGen.head'_iff.1 hu
    obtain ⟨rfl, hw0⟩ := fan_dep hw
    rcases Relation.ReflTransGen.cases_head hwu with h | ⟨c, hc, -⟩
    · exact hw0 h
    · exact hw0 (fan_dep hc).1

theorem fan_honest_correct {N : ℕ} (b : Fin N → Bool) (u : Option (Fin N)) :
    (fan N).Correct (fanFns N) (fanHonest b) u := by
  rcases u with _ | i
  · exact fan_correct_none.2 rfl
  · exact fan_correct_some.2 (by simp [fanHonest])

theorem fan_down (N : ℕ) : (fan N).down {none} = univ := by
  refine eq_univ_of_forall fun v => (fan N).mem_down.2 ⟨none, mem_singleton_self _, ?_⟩
  rcases v with _ | i
  · exact .refl
  · exact .single ⟨Sum.inl none, mem_singleton_self _, by simp [fan]⟩

/-- **The late cone is necessary**: on any inputs `b`, the skipping run holds the inputs, its one faulty proof block is
`x`, no unit is lucky (`SkipsLucky` with nothing computed and nothing lucky), and its outputs are all `false`, read from
no `bᵢ`. All `N + 1` units' work is skipped at wrong weight `1`, which is exactly `x`'s cone. -/
theorem fan_skips {N : ℕ} (b : Fin N → Bool) :
    (∀ s ∈ (fan N).input, fanSkip b s = fanHonest b s) ∧
      (fan N).wrong (fanFns N) (fanSkip b) = {none} ∧
      (fan N).SkipsLucky (fanFns N) (fanHonest b) (fanSkip b) ∅ ∅ ∧
      (∀ i, fanSkip b (Sum.inl (some i)) = false) ∧
      (univ \ ∅ : Finset (Option (Fin N))).card = N + 1 ∧ ((fan N).down {none}).card = N + 1 := by
  classical
  refine ⟨fun s hs => ?_, ?_, fun u _ hr hc => ?_, fun _ => rfl, by simp, by rw [fan_down]; simp⟩
  · obtain ⟨i, -, rfl⟩ := mem_image.1 hs
    rfl
  · ext u
    rw [mem_wrong_iff, mem_singleton]
    rcases u with _ | i
    · simp [fan_correct_none, fanSkip]
    · simp [fan_correct_some, fanSkip]
  · rcases u with _ | i
    · simp [fan_correct_none, fanSkip] at hc
    · have := hr (Sum.inl none) (by simp [fan])
      simp [fanSkip, fanHonest] at this

end Fan

section Operand

/-- **An operand chosen after the work coin skips the work**: `x := −c` zeroes `(x + c) · b` for every `b`. -/
theorem late_operand {R : Type} [Ring R] (c b : R) : (-c + c) * b = 0 := by simp

/-- **An operand committed before the work coin skips it at one coin in `|𝔽|`**: for `b ≠ 0`, exactly one coin `c`
zeroes `(x + c) · b`. The luck rate is `1 / |𝔽|`. -/
theorem early_operand {𝔽 : Type} [Field 𝔽] [Fintype 𝔽] [DecidableEq 𝔽] (x b : 𝔽) (hb : b ≠ 0) :
    (univ.filter fun c : 𝔽 => (x + c) * b = 0).card = 1 := by
  rw [card_eq_one]
  refine ⟨-x, ?_⟩
  ext c
  simp only [mem_filter, mem_univ, true_and, mem_singleton, mul_eq_zero, hb, or_false]
  rw [eq_neg_iff_add_eq_zero, add_comm]

end Operand

/-! ## Weight isolation: range restriction -/

section Isolation

variable {S V : Type} (D : DynPopulation U S V)

theorem padOver_reads_sub [Fintype S] {C : Set (S → V)} {R : U → Finset S} (hR : ∀ σ ∈ C, ∀ u, D.reads u σ ⊆ R u)
    (u : U) : (D.padOver C).reads u ⊆ R u := by
  intro s hs
  simp only [DynPopulation.padOver, mem_filter, mem_univ, true_and] at hs
  obtain ⟨σ, hσ, hs⟩ := hs
  exact hR σ hσ u hs

theorem padOver_up_sub [Fintype S] {C : Set (S → V)} {R : U → Finset S} (hR : ∀ σ ∈ C, ∀ u, D.reads u σ ⊆ R u)
    (O : Finset S) : (D.padOver C).up O ⊆ (rangeFlow D R).up O := by
  rintro s ⟨o, ho, hso⟩
  exact ⟨o, ho, Relation.ReflTransGen.mono (fun _ _ ⟨u, hr, hw⟩ => ⟨u, padOver_reads_sub D hR u hr, hw⟩) _ _ hso⟩

/-- **Isolation by range restriction**: if a public bound `R u` holds every slot unit `u` reads in every admitted run
(`C`), and no secret slot is upstream of the egress in `R`'s static dataflow, the leakage is at most the control.
Addresses may be computed and corrupted, as long as they stay in range. -/
theorem isolation_of_range [Fintype U] [Fintype S] [DecidableEq S] [Fintype V] [DecidableEq V] [Inhabited V]
    (hsa : D.SingleAssignment) {step : U → ℕ} (ht : D.TimeOrdered step) {C : Set (S → V)} {R : U → Finset S}
    (hR : ∀ σ ∈ C, ∀ u, D.reads u σ ⊆ R u) {X O : Finset S} (a : S → V)
    (hX : ∀ s ∈ X, s ∉ (rangeFlow D R).up O) (Cl : Finset (S → V)) (m w : ℕ) (hw : ∀ u, (D.writes u).card ≤ w)
    (σ : (S → V) → S → V) (hσC : ∀ s, σ s ∈ C)
    (hσ : ∀ s, ∀ t ∈ D.input, σ s t = if t ∈ X then s t else a t) (hm : ∀ s, (D.wrong (σ s)).card ≤ m) :
    (Cl.image fun s (o : O) => σ s o).card ≤
      (∑ j ∈ range (m + 1), (Fintype.card U).choose j) * Fintype.card V ^ (m * w) :=
  D.card_leak_le_of_not_up_pad hsa ht a (fun s hs hup => hX s hs (padOver_up_sub D hR O hup)) Cl m w hw σ hσC hσ hm

end Isolation

end Population.Iop
