module

public import Definitions.Claims.SamplingTwoReg
public import Proofs.Flock.SamplingGeneric.TwoCap
import Proofs.Flock.Claims.Prob
import Proofs.Flock.Claims.Sampling.KS
import Proofs.Flock.Claims.Sampling.Wait
import Proofs.Flock.SamplingGeneric.Reg
import Proofs.Flock.SamplingGeneric.Two
import Proofs.Flock.SamplingGeneric.TwoReg

/-! The two-stage committed form's miss, threshold form and headline, at any boundary values. -/

@[expose] public section

namespace Claims.Sampling

open Finset

open Classical in
theorem exists_best_interiorAt (s : AuditStatement) (Pt : Partition) (d : Design.Two) (val : ℕ → Option Bool)
    {ρ : ℕ} (hρ : ρ < d.part.blocks.length) :
    ∃ ι : ℕ → Bool, ∀ ι' : ℕ → Bool,
      d.esc Pt (footprintAt s Pt d.part val fun _ => ι') ρ ≤ d.esc Pt (footprintAt s Pt d.part val fun _ => ι) ρ := by
  have hf : ∀ ι' : ℕ → Bool, d.esc Pt (footprintAt s Pt d.part val fun _ => ι') ρ =
      d.esc Pt (fun ρ t => ((incorrectAllAt s Pt d.part val fun _ => ι') ∩ cell Pt d.part ρ t).card) ρ := by
    intro ι'
    refine prod_congr rfl fun t ht => ?_
    show escCell _ _ (incorrectInAt s Pt d.part val ρ t ι').card =
      escCell _ _ ((incorrectAllAt s Pt d.part val fun _ => ι') ∩ cell Pt d.part ρ t).card
    rw [inter_cell_incorrectAllAt s Pt d.part val _ hρ (mem_range.1 ht)]
  have hA : ∀ ι' : ℕ → Bool, (incorrectAllAt s Pt d.part val fun _ => ι') ∈
      (range Pt.blocks.length).powerset.filter fun wa =>
        ∃ ι'' : ℕ → Bool, (incorrectAllAt s Pt d.part val fun _ => ι'') = wa :=
    fun ι' => mem_filter.2 ⟨mem_powerset.2 (incorrectAllAt_subset s Pt d.part val _), ι', rfl⟩
  obtain ⟨wa, hwa, hmax⟩ := ((range Pt.blocks.length).powerset.filter fun wa =>
      ∃ ι'' : ℕ → Bool, (incorrectAllAt s Pt d.part val fun _ => ι'') = wa).exists_max_image
    (fun wa => d.esc Pt (fun ρ t => (wa ∩ cell Pt d.part ρ t).card) ρ) ⟨_, hA fun _ => false⟩
  obtain ⟨ι, hι⟩ := (mem_filter.1 hwa).2
  refine ⟨ι, fun ι' => ?_⟩
  rw [hf, hf, hι]
  exact hmax _ (hA ι')

theorem esc_footprintAt_congr (s : AuditStatement) (Pt : Partition) (d : Design.Two) (val : ℕ → Option Bool)
    {ι ι' : ℕ → ℕ → Bool} {ρ : ℕ} (h : ι ρ = ι' ρ) :
    d.esc Pt (footprintAt s Pt d.part val ι) ρ = d.esc Pt (footprintAt s Pt d.part val ι') ρ := by
  simp only [Design.Two.esc, footprintAt, h]

theorem exists_best_interiorsAt (s : AuditStatement) (Pt : Partition) (d : Design.Two) (val : ℕ → Option Bool) :
    ∃ ι : ℕ → ℕ → Bool, ∀ ρ < d.part.blocks.length, ∀ ι' : ℕ → ℕ → Bool,
      d.esc Pt (footprintAt s Pt d.part val ι') ρ ≤ d.esc Pt (footprintAt s Pt d.part val ι) ρ := by
  classical
  choose! b hb using fun ρ (hρ : ρ < d.part.blocks.length) => exists_best_interiorAt s Pt d val hρ
  refine ⟨b, fun ρ hρ ι' => ?_⟩
  have h1 : d.esc Pt (footprintAt s Pt d.part val ι') ρ = d.esc Pt (footprintAt s Pt d.part val fun _ => ι' ρ) ρ :=
    esc_footprintAt_congr s Pt d val rfl
  have h2 : d.esc Pt (footprintAt s Pt d.part val b) ρ = d.esc Pt (footprintAt s Pt d.part val fun _ => b ρ) ρ :=
    esc_footprintAt_congr s Pt d val rfl
  rw [h1, h2]
  exact hb ρ hρ (ι' ρ)

open Classical in
theorem incorrectAtBoundaryAt_subset (s : AuditStatement) (Pt : Partition) (d : Design.Two) (val : ℕ → Option Bool)
    (ι : ℕ → ℕ → Bool) : incorrectAtBoundaryAt s Pt d.part val ⊆ d.incorrectBlocks Pt (footprintAt s Pt d.part val ι) := by
  intro ρ hρ
  obtain ⟨hρl, hw⟩ := mem_filter.1 hρ
  obtain ⟨t, ht, hne⟩ := hw (ι ρ)
  exact mem_filter.2 ⟨hρl, t, ht, card_pos.2 hne⟩

end Claims.Sampling

namespace Claims.SamplingGeneric

open Claims Finset
open Claims.Sampling (AuditStatement Partition InScopeTwo forkOn cell escCell footprintAt incorrectAtBoundaryAt
  incorrectAllAt inter_cell_incorrectAllAt footprintAt_le_card incorrectAtBoundaryAt_subset exists_best_interiorsAt)

namespace CompiledTwo

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}

theorem pr_missTwo_fork_le (H : Msg → Msg) (L : CompiledTwo Coin A) (hS : L.Sessions H) {x : Msg}
    {s : AuditStatement} {Pt : Partition} {d : Sampling.Design.Two} (hs : L.parse x = some s)
    (hPt : L.partition s = some Pt) (hd : L.two s = some d) (hin : InScopeTwo Pt d s) {Ω : Type} (P : Prover Ω)
    (ω : Ω) (val : ℕ → Option Bool) (ι : ℕ → ℕ → Bool)
    (hι : ∀ ρ < d.part.blocks.length, ∀ ι' : ℕ → ℕ → Bool,
      d.esc Pt (footprintAt s Pt d.part val ι') ρ ≤ d.esc Pt (footprintAt s Pt d.part val ι) ρ)
    (r : Fin (L.V.k x) → Coin) :
    Pr (fun e : Fin (L.V.k x) → Coin => ∀ u ∈ L.checked x (forkOn (L.secondStage x) r e),
        u ∉ incorrectAllAt s Pt d.part val (L.iregVal H P x ω (forkOn (L.secondStage x) r e))) ≤
      ∏ ρ ∈ (range d.part.blocks.length).filter (· ∈ L.replaySample x r),
        d.esc Pt (footprintAt s Pt d.part val ι) ρ := by
  simp only [iregVal_forkOn]
  refine (hS.secondSamples x s Pt d hs hPt hd hin r (incorrectAllAt s Pt d.part val (L.iregVal H P x ω r))).trans ?_
  refine prod_le_prod₀ (fun _ _ => prod_nonneg fun _ _ => Claims.Sampling.escCell_nonneg _ _ _) fun ρ hρ => ?_
  have hρl : ρ < d.part.blocks.length := mem_range.1 (mem_filter.1 hρ).1
  refine le_trans (le_of_eq ?_) (hι ρ hρl (L.iregVal H P x ω r))
  simp only [Sampling.Design.Two.esc]
  refine prod_congr rfl fun t ht => ?_
  show _ = escCell _ _ (Claims.Sampling.incorrectInAt s Pt d.part val ρ t (L.iregVal H P x ω r ρ)).card
  rw [inter_cell_incorrectAllAt s Pt d.part _ _ hρl (mem_range.1 ht)]

theorem pr_missTwo_le_Pa (H : Msg → Msg) (L : CompiledTwo Coin A) (hS : L.Sessions H) {x : Msg}
    {s : AuditStatement} {Pt : Partition} {d : Sampling.Design.Two} (hs : L.parse x = some s)
    (hPt : L.partition s = some Pt) (hd : L.two s = some d) (hin : InScopeTwo Pt d s) {Ω : Type} (P : Prover Ω)
    (ω : Ω) (val : ℕ → Option Bool) (ι : ℕ → ℕ → Bool)
    (hι : ∀ ρ < d.part.blocks.length, ∀ ι' : ℕ → ℕ → Bool,
      d.esc Pt (footprintAt s Pt d.part val ι') ρ ≤ d.esc Pt (footprintAt s Pt d.part val ι) ρ) :
    Pr (fun r : Fin (L.V.k x) → Coin =>
        ∀ u ∈ L.checked x r, u ∉ incorrectAllAt s Pt d.part val (L.iregVal H P x ω r)) ≤
      d.Pa Pt (footprintAt s Pt d.part val ι) := by
  rw [Claims.SamplingGeneric.Pr_forkOn (L.secondStage x)]
  exact (Ex_mono fun r => pr_missTwo_fork_le H L hS hs hPt hd hin P ω val ι hι r).trans
    (Claims.Sampling.Ex_esc_le d Pt _ (L.replaySample x) (hS.replaySamples x s Pt d hs hPt hd hin))

theorem pr_missTwo_le_pow (H : Msg → Msg) (L : CompiledTwo Coin A) (hS : L.Sessions H) {r₁ r₂ : ℝ} (hr₁ : 0 ≤ r₁)
    (hr₁' : r₁ ≤ 1) (hr₂ : 0 ≤ r₂) (hr₂' : r₂ ≤ 1) {x : Msg} {s : AuditStatement} {Pt : Partition}
    {d : Sampling.Design.Two} (hx : L.Scope x) (hs : L.parse x = some s) (hPt : L.partition s = some Pt)
    (hd : L.two s = some d) (h1 : d.FirstRate r₁) (h2 : d.SecondRate Pt r₂) {Ω : Type} (P : Prover Ω) (ω : Ω)
    (val : ℕ → Option Bool) :
    Pr (fun r : Fin (L.V.k x) → Coin =>
        ∀ u ∈ L.checked x r, u ∉ incorrectAllAt s Pt d.part val (L.iregVal H P x ω r)) ≤
      (1 - r₁ * r₂) ^ (incorrectAtBoundaryAt s Pt d.part val).card := by
  have hin := L.inScopeTwo_of hx hs hPt hd
  obtain ⟨ι, hι⟩ := exists_best_interiorsAt s Pt d val
  have hx0 : 0 ≤ 1 - r₁ * r₂ := by nlinarith
  have hx1 : 1 - r₁ * r₂ ≤ 1 := sub_le_self _ (mul_nonneg hr₁ hr₂)
  calc Pr (fun r : Fin (L.V.k x) → Coin =>
        ∀ u ∈ L.checked x r, u ∉ incorrectAllAt s Pt d.part val (L.iregVal H P x ω r))
      ≤ d.Pa Pt (footprintAt s Pt d.part val ι) := pr_missTwo_le_Pa H L hS hs hPt hd hin P ω val ι hι
    _ ≤ (1 - r₁ * r₂) ^ (d.incorrectBlocks Pt (footprintAt s Pt d.part val ι)).card :=
        d.Pa_le_pow Pt (fun _ => r₁) (fun T hT => (h1 T hT).trans_eq (prod_const _).symm)
          (fun _ _ => ⟨le_rfl, hr₁'⟩) hr₁ hr₂ h2 (footprintAt_le_card s Pt d.part _ ι)
    _ ≤ (1 - r₁ * r₂) ^ (incorrectAtBoundaryAt s Pt d.part val).card :=
        pow_le_pow_of_le_one hx0 hx1 (card_le_card (incorrectAtBoundaryAt_subset s Pt d _ ι))

theorem missTwo (H : Msg → Msg) (L : CompiledTwo Coin A) (hS : L.Sessions H) {r₁ r₂ : ℝ} (hr₁ : 0 ≤ r₁)
    (hr₁' : r₁ ≤ 1) (hr₂ : 0 ≤ r₂) (hr₂' : r₂ ≤ 1) : L.MissTwo H r₁ r₂ := by
  intro x s Pt d hx hs hPt hd h1 h2 Ω P ω
  simp only [FregTwo_eq H L P hs hPt hd]
  exact pr_missTwo_le_pow H L hS hr₁ hr₁' hr₂ hr₂' hx hs hPt hd h1 h2 P ω (L.regVal H P x ω)

theorem threshold_of_event (H : Msg → Msg) (L : CompiledTwo Coin A) (hS : L.Sessions H) {x : Msg}
    {s : AuditStatement} {Pt : Partition} {d : Sampling.Design.Two} (hs : L.parse x = some s)
    (hPt : L.partition s = some Pt) (hd : L.two s = some d) (hin : InScopeTwo Pt d s) {Ω : Type} [Fintype Ω]
    [Nonempty Ω] (P : Prover Ω) (val : Ω → ℕ → Option Bool) {ε : ℝ}
    (hev : Pr (fun p : Ω × (Fin (L.V.k x) → Coin) => (run H P L.V x p.1 p.2).out = .acc ∧
      ∃ u ∈ L.checked x p.2, u ∈ incorrectAllAt s Pt d.part (val p.1) (L.iregVal H P x p.1 p.2)) ≤ ε) {δ : ℝ}
    (hδ : 0 ≤ δ) :
    Pr (fun p : Ω × (Fin (L.V.k x) → Coin) => (run H P L.V x p.1 p.2).out = .acc ∧
      ∀ ι : ℕ → ℕ → Bool, footprintAt s Pt d.part (val p.1) ι ∉ d.M Pt δ) ≤ δ + ε := by
  rw [add_comm]
  refine Claims.Sampling.Pr_or_le
    (q := fun p : Ω × (Fin (L.V.k x) → Coin) => (run H P L.V x p.1 p.2).out = .acc ∧
      ∃ u ∈ L.checked x p.2, u ∈ incorrectAllAt s Pt d.part (val p.1) (L.iregVal H P x p.1 p.2))
    (r := fun p : Ω × (Fin (L.V.k x) → Coin) =>
      (∀ ι : ℕ → ℕ → Bool, footprintAt s Pt d.part (val p.1) ι ∉ d.M Pt δ) ∧
        ∀ u ∈ L.checked x p.2, u ∉ incorrectAllAt s Pt d.part (val p.1) (L.iregVal H P x p.1 p.2))
    (fun p hp => ?_) hev ?_
  · by_cases hu : ∃ u ∈ L.checked x p.2, u ∈ incorrectAllAt s Pt d.part (val p.1) (L.iregVal H P x p.1 p.2)
    · exact Or.inl ⟨hp.1, hu⟩
    · exact Or.inr ⟨hp.2, fun u hu' hF => hu ⟨u, hu', hF⟩⟩
  · rw [Claims.Sampling.Pr_prod_eq_Ex]
    refine (Ex_mono fun ω => ?_).trans_eq (Ex_const δ)
    by_cases hM : ∀ ι : ℕ → ℕ → Bool, footprintAt s Pt d.part (val ω) ι ∉ d.M Pt δ
    · obtain ⟨ι, hι⟩ := exists_best_interiorsAt s Pt d (val ω)
      have hPa : d.Pa Pt (footprintAt s Pt d.part (val ω) ι) ≤ δ := by
        have h := hM ι
        simp only [Sampling.Design.Two.M, Set.mem_ofPred_eq, not_lt] at h
        exact h
      exact (Pr_mono fun r (hr : (∀ ι : ℕ → ℕ → Bool, footprintAt s Pt d.part (val ω) ι ∉ d.M Pt δ) ∧
          ∀ u ∈ L.checked x r, u ∉ incorrectAllAt s Pt d.part (val ω) (L.iregVal H P x ω r)) => hr.2).trans
        ((pr_missTwo_le_Pa H L hS hs hPt hd hin P ω (val ω) ι hι).trans hPa)
    · exact ((Pr_mono (q := fun _ => False) fun r (hr : (∀ ι : ℕ → ℕ → Bool,
          footprintAt s Pt d.part (val ω) ι ∉ d.M Pt δ) ∧
          ∀ u ∈ L.checked x r, u ∉ incorrectAllAt s Pt d.part (val ω) (L.iregVal H P x ω r)) => hM hr.1).trans_eq
        Pr_false).trans hδ

theorem ksRegTwo_of_event (H : Msg → Msg) (L : CompiledTwo Coin A) (hS : L.Sessions H)
    (hev : ∀ x, L.Scope x → ∀ (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω),
      Pr (fun p : Ω × (Fin (L.V.k x) → Coin) => (run H P L.V x p.1 p.2).out = .acc ∧
        ∃ u ∈ L.checked x p.2, u ∈ L.FregTwo H P x p.1 p.2) ≤ L.εregTwo H x P) :
    L.KSRegTwo H := by
  intro x hx Ω _ _ P
  refine ⟨hev x hx Ω P, fun s Pt d hs hPt hd δ hδ => ?_⟩
  have h := hev x hx Ω P
  simp only [FregTwo_eq H L P hs hPt hd] at h
  exact threshold_of_event H L hS hs hPt hd (L.inScopeTwo_of hx hs hPt hd) P (L.regVal H P x) h hδ

theorem headline_of_event (H : Msg → Msg) (L : CompiledTwo Coin A) (hS : L.Sessions H) {x : Msg}
    {s : AuditStatement} {Pt : Partition} {d : Sampling.Design.Two} (hs : L.parse x = some s)
    (hPt : L.partition s = some Pt) (hd : L.two s = some d) (hin : InScopeTwo Pt d s) {Ω : Type} [Fintype Ω]
    [Nonempty Ω] (P : Prover Ω) (val : Ω → ℕ → Option Bool) {ε : ℝ}
    (hev : Pr (fun p : Ω × (Fin (L.V.k x) → Coin) => (run H P L.V x p.1 p.2).out = .acc ∧
      ∃ u ∈ L.checked x p.2, u ∈ incorrectAllAt s Pt d.part (val p.1) (L.iregVal H P x p.1 p.2)) ≤ ε) (pb : Ω → ℝ)
    (hpb : ∀ ω ι, d.Pa Pt (footprintAt s Pt d.part (val ω) ι) ≤ pb ω) :
    Pr (fun p : Ω × (Fin (L.V.k x) → Coin) => (run H P L.V x p.1 p.2).out = .acc) ≤ Ex pb + ε := by
  rw [add_comm]
  refine Claims.Sampling.Pr_or_le
    (q := fun p : Ω × (Fin (L.V.k x) → Coin) => (run H P L.V x p.1 p.2).out = .acc ∧
      ∃ u ∈ L.checked x p.2, u ∈ incorrectAllAt s Pt d.part (val p.1) (L.iregVal H P x p.1 p.2))
    (r := fun p : Ω × (Fin (L.V.k x) → Coin) =>
      ∀ u ∈ L.checked x p.2, u ∉ incorrectAllAt s Pt d.part (val p.1) (L.iregVal H P x p.1 p.2))
    (fun p hp => ?_) hev ?_
  · by_cases hu : ∃ u ∈ L.checked x p.2, u ∈ incorrectAllAt s Pt d.part (val p.1) (L.iregVal H P x p.1 p.2)
    · exact Or.inl ⟨hp, hu⟩
    · exact Or.inr fun u hu' hF => hu ⟨u, hu', hF⟩
  · rw [Claims.Sampling.Pr_prod_eq_Ex]
    refine Ex_mono fun ω => ?_
    obtain ⟨ι, hι⟩ := exists_best_interiorsAt s Pt d (val ω)
    exact (pr_missTwo_le_Pa H L hS hs hPt hd hin P ω (val ω) ι hι).trans (hpb ω ι)

theorem headlineTwo_of (H : Msg → Msg) (L : CompiledTwo Coin A) (hS : L.Sessions H) (hks : L.KSRegTwo H) :
    L.HeadlineTwo H := by
  intro x s Pt d hx hs hPt hd Ω _ _ P pb hpb
  have h := (hks x hx Ω P).1
  simp only [FregTwo_eq H L P hs hPt hd] at h
  exact headline_of_event H L hS hs hPt hd (L.inScopeTwo_of hx hs hPt hd) P (L.regVal H P x) h pb hpb

end CompiledTwo

/-- **The two-stage compiler's committed form**: `CompiledTwo.ksRegTwo_of_event` at the event's bound
(`CompiledTwo.eventTwo_le`). -/
theorem compilerKSRegTwo : CompilerKSRegTwo := by
  intro Coin _ _ _ A L H hA hbind hS
  exact CompiledTwo.ksRegTwo_of_event H L hS (CompiledTwo.eventTwo_le hA hbind hS)

/-- **The two-stage miss bound**: `CompiledTwo.missTwo`. -/
theorem compilerMissTwo : CompilerMissTwo := by
  intro Coin _ _ _ A L H hS r₁ r₂ hr₁ hr₁' hr₂ hr₂'
  exact CompiledTwo.missTwo H L hS hr₁ hr₁' hr₂ hr₂'

/-- **The two-stage headline from the committed form**: `CompiledTwo.headlineTwo_of`. -/
theorem compilerHeadlineTwo : CompilerHeadlineTwo := by
  intro Coin _ _ _ A L H hS hks
  exact CompiledTwo.headlineTwo_of H L hS hks

end Claims.SamplingGeneric
