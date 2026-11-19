module

public import Definitions.Claims.SamplingLevels
public import Proofs.Core.LevelsProofBlocks
public import Proofs.Core.Game.Basic
public import Proofs.Flock.Claims.Prob
public import Proofs.Flock.Claims.Sampling.Session

/-!
# The one statement over levels at one level and at two

`Definitions/Claims/SamplingLevels.lean` states SPOT's audit at any number of levels with core's levels as its design.
Here its relation and claims are read at one level and at two, where core's levels realize the claims designs.

* `levM_down`: the confidence set at levels is closed under subsets.
* `exists_incorrectLev_eq_faulty`, `rlev_one_iff`: with no top partition the incorrect set at the best interiors is
  `faulty`, so at one level that misses as the statement's design does (`Verity.Levels.MissesAsDesign`) the relation
  at levels is `RelS`.
* `incorrectLev_inter_cell`, `footprint_eq_counts`, `rlev_two_iff`: with the top partition `d.part` the incorrect set
  meets each cell in that cell's incorrect proof blocks, so its counts per cell are the footprint, and at levels that
  miss as `Design.Two.Pa` at the counts the relation at levels is `RTwo`'s.
* `Compiled.liftLev`: a compiled verifier at one level, its levels the ones `LevelsRealizeOne` gives. It meets the
  contract at levels where it meets `Compiled.Sessions` (`Compiled.sessions_liftLev`), and its claims at levels give
  `Compiled.KS` and `Compiled.KSReg` (`Compiled.ks_of_liftLev`, `Compiled.ksReg_of_liftLev`).
* `CompiledTwo.liftLev`: a compiled two-stage verifier at two levels, its levels the ones `LevelsRealizeTwo` gives,
  its replay partition the top partition and its replay sample the first level's (`CompiledTwo.upperOf`). It meets the
  contract at levels where it meets `CompiledTwo.Sessions` (`CompiledTwo.sessions_liftLev`), and its claims at levels
  give `CompiledTwo.KS` and `CompiledTwo.KSRegTwo` (`CompiledTwo.ks_of_liftLev`, `CompiledTwo.ksRegTwo_of_liftLev`).
* `compilerKS_of_lev`, `compilerKSReg_of_lev`, `compilerKSTwo_of_lev`, `compilerKSRegTwo_of_lev`: so
  `CompilerKSLev` and `CompilerKSRegLev` give `CompilerKS` and `CompilerKSReg` where one level realizes the claims
  designs, and `CompilerKSTwo` and `CompilerKSRegTwo` where two levels realize the two-stage designs.
-/

@[expose] public section

namespace Claims.Sampling

open Verity

variable {Pt : Partition}

theorem levM_down {L : Verity.Levels (Fin Pt.blocks.length)} {δ : ℝ} {B B' : Finset ℕ} (h : B ⊆ B')
    (hB' : B' ∈ levM Pt L δ) : B ∈ levM Pt L δ := by
  refine lt_of_lt_of_le hB' (ENNReal.toReal_mono (ne_top_of_le_ne_top ENNReal.one_ne_top
    (SamplingDesign.missProb_le_one _ _)) ?_)
  exact Game.prCoin_mono fun ω hω => Finset.disjoint_of_subset_right (Finset.image_subset_image h) hω

theorem faulty_subset_incorrectLev (H : Msg → Msg) (Cm : CommitmentScheme Bool) (s : AuditStatement) (w : Wit)
    (ι : ℕ → ℕ → Bool) : faulty H Cm s Pt w ⊆ incorrectLev s Pt none (w.val H Cm s) ι := by
  intro v hv
  simp only [faulty, Faulty, Finset.mem_filter] at hv
  simp only [incorrectLev, Finset.mem_filter]
  refine ⟨hv.1, fun ⟨u, ρ, r, hu, hρ, hr, hc⟩ => hv.2 ⟨u, hu, ι ρ, ?_⟩⟩
  cases hρ
  have : r = u := Option.some_injective _ (hr.symm.trans hu)
  subst this
  exact hc

theorem exists_incorrectLev_eq_faulty (H : Msg → Msg) (Cm : CommitmentScheme Bool) (s : AuditStatement) (w : Wit) :
    ∃ ι, incorrectLev s Pt none (w.val H Cm s) ι = faulty H Cm s Pt w := by
  classical
  let ι : ℕ → ℕ → Bool := fun v =>
    if h : ∃ ι' : ℕ → Bool, ∃ u, Pt.blocks[v]? = some u ∧
        s.prog.Correct (s.prog.refIn u (w.val H Cm s) ι') u then h.choose else fun _ => true
  refine ⟨ι, Finset.Subset.antisymm (fun v hv => ?_) (faulty_subset_incorrectLev H Cm s w ι)⟩
  simp only [incorrectLev, Finset.mem_filter] at hv
  simp only [faulty, Faulty, Finset.mem_filter]
  refine ⟨hv.1, fun ⟨u, hu, ι', hc⟩ => hv.2 ?_⟩
  have h : ∃ ι' : ℕ → Bool, ∃ u, Pt.blocks[v]? = some u ∧
      s.prog.Correct (s.prog.refIn u (w.val H Cm s) ι') u := ⟨ι', u, hu, hc⟩
  obtain ⟨u', hu', hc'⟩ := h.choose_spec
  refine ⟨u', v, u', hu', rfl, hu', ?_⟩
  simp only [ι, dif_pos h]
  exact hc'

/-- **At one level the relation at levels is `RelS`**, where the level misses as the statement's design does. -/
theorem rlev_one_iff {H : Msg → Msg} {Cm : CommitmentScheme Bool} {s : AuditStatement}
    {L : Verity.Levels (Fin Pt.blocks.length)} (hm : L.MissesAsDesign s.design Pt (slot Pt)) (δ : ℝ) (w : Wit) :
    (∃ ι, incorrectLev s Pt none (w.val H Cm s) ι ∈ levM Pt L δ) ↔ RelS H Cm δ s Pt w := by
  rw [Verity.Levels.MissesAsDesign.relS_iff hm]
  constructor
  · rintro ⟨ι, hι⟩
    exact levM_down (faulty_subset_incorrectLev H Cm s w ι) hι
  · intro h
    obtain ⟨ι, hι⟩ := exists_incorrectLev_eq_faulty (Pt := Pt) H Cm s w
    exact ⟨ι, hι ▸ h⟩

theorem incorrectLev_subset_range (s : AuditStatement) (top : Option Partition) (val : ℕ → Option Bool)
    (ι : ℕ → ℕ → Bool) : incorrectLev s Pt top val ι ⊆ Finset.range Pt.blocks.length := by
  classical
  exact Finset.filter_subset _ _

theorem mem_incorrectLev {s : AuditStatement} {top : Option Partition} {val : ℕ → Option Bool}
    {ι : ℕ → ℕ → Bool} {v : ℕ} : v ∈ incorrectLev s Pt top val ι ↔ v < Pt.blocks.length ∧
      ¬ ∃ u ρ r, Pt.blocks[v]? = some u ∧ topOf Pt top v = some ρ ∧ (top.getD Pt).blocks[ρ]? = some r ∧
        s.prog.Correct (s.prog.refIn r val (ι ρ)) u := by
  classical
  unfold incorrectLev
  rw [Finset.mem_filter, Finset.mem_range]

/-- With the top partition `Q`, the incorrect set meets each cell in that cell's incorrect proof blocks. -/
theorem incorrectLev_inter_cell (s : AuditStatement) (Q : Partition) (val : ℕ → Option Bool) (ι : ℕ → ℕ → Bool)
    (ρ t : ℕ) : incorrectLev s Pt (some Q) val ι ∩ cell Pt Q ρ t = incorrectInAt s Pt Q val ρ t (ι ρ) := by
  classical
  ext v
  rw [Finset.mem_inter, mem_incorrectLev]
  unfold incorrectInAt cell
  simp only [Finset.mem_filter, Finset.mem_range, topOf, Option.getD_some]
  constructor
  · rintro ⟨⟨-, hv⟩, hc⟩
    refine ⟨hc, fun ⟨u, r, hu, hr, hcr⟩ => hv ⟨u, ρ, r, hu, hc.2.1, hr, hcr⟩⟩
  · rintro ⟨hc, hv⟩
    refine ⟨⟨hc.1, fun ⟨u, ρ', r, hu, hρ', hr, hcr⟩ => hv ⟨u, r, hu, ?_, ?_⟩⟩, hc⟩
    · have : ρ' = ρ := Option.some_injective _ (hρ'.symm.trans hc.2.1)
      exact this ▸ hr
    · have : ρ' = ρ := Option.some_injective _ (hρ'.symm.trans hc.2.1)
      exact this ▸ hcr

/-- With the top partition `d.part`, the incorrect set's counts per cell are the footprint. -/
theorem footprint_eq_counts (H : Msg → Msg) (Cm : CommitmentScheme Bool) (s : AuditStatement) (Q : Partition)
    (w : Wit) (ι : ℕ → ℕ → Bool) :
    footprint H Cm s Pt Q w ι = counts Pt Q (incorrectLev s Pt (some Q) (w.val H Cm s) ι) := by
  funext ρ t
  unfold counts
  rw [incorrectLev_inter_cell]
  rfl

/-- With the top partition `Q`, the incorrect set's counts per cell are the footprint at boundary values `val`. -/
theorem footprintAt_eq_counts (s : AuditStatement) (Q : Partition) (val : ℕ → Option Bool) (ι : ℕ → ℕ → Bool) :
    footprintAt s Pt Q val ι = counts Pt Q (incorrectLev s Pt (some Q) val ι) := by
  funext ρ t
  unfold counts footprintAt
  rw [incorrectLev_inter_cell]

/-- At levels that miss as `Design.Two.Pa` at the counts, the incorrect set at levels is in `M_δ` exactly when its
footprint is in `M̄_δ`. -/
theorem incorrectLev_mem_levM_iff {s : AuditStatement} {d : Design.Two} {L : Verity.Levels (Fin Pt.blocks.length)}
    (hm : ∀ B ⊆ Finset.range Pt.blocks.length,
      (L.design.missProb (B.image (slot Pt))).toReal = d.Pa Pt (counts Pt d.part B)) (δ : ℝ)
    (val : ℕ → Option Bool) (ι : ℕ → ℕ → Bool) :
    incorrectLev s Pt (some d.part) val ι ∈ levM Pt L δ ↔ footprintAt s Pt d.part val ι ∈ d.M Pt δ := by
  show δ < _ ↔ δ < _
  rw [hm _ (incorrectLev_subset_range _ _ _ _), footprintAt_eq_counts]

theorem cell_eq_empty {Q : Partition} {ρ : ℕ} (h : ¬ (topBlock Pt Q ρ).Nonempty) (t : ℕ) : cell Pt Q ρ t = ∅ := by
  classical
  refine Finset.eq_empty_of_forall_notMem fun v hv => h ?_
  unfold cell at hv
  rw [Finset.mem_filter, Finset.mem_range] at hv
  refine ⟨⟨v, hv.1⟩, ?_⟩
  unfold topBlock
  rw [Finset.mem_filter]
  exact ⟨Finset.mem_univ _, hv.2.1⟩

theorem topBlock_injOn {Q : Partition} {ρ ρ' : ℕ} (h : (topBlock Pt Q ρ).Nonempty)
    (he : topBlock Pt Q ρ = topBlock Pt Q ρ') : ρ = ρ' := by
  classical
  obtain ⟨v, hv⟩ := h
  have hv' := he ▸ hv
  unfold topBlock at hv hv'
  rw [Finset.mem_filter] at hv hv'
  exact Option.some_injective _ (hv.2.symm.trans hv'.2)

/-- **At two levels the relation at levels is `RTwo`'s**, where the levels miss as `Design.Two.Pa` at the counts. -/
theorem rlev_two_iff {H : Msg → Msg} {Cm : CommitmentScheme Bool} {s : AuditStatement} {d : Design.Two}
    {L : Verity.Levels (Fin Pt.blocks.length)}
    (hm : ∀ B ⊆ Finset.range Pt.blocks.length,
      (L.design.missProb (B.image (slot Pt))).toReal = d.Pa Pt (counts Pt d.part B)) (δ : ℝ) (w : Wit) :
    (∃ ι, incorrectLev s Pt (some d.part) (w.val H Cm s) ι ∈ levM Pt L δ) ↔
      ∃ ι, footprint H Cm s Pt d.part w ι ∈ d.M Pt δ := by
  refine exists_congr fun ι => ?_
  show δ < _ ↔ δ < _
  rw [hm _ (incorrectLev_subset_range _ _ _ _), footprint_eq_counts]

end Claims.Sampling

namespace Claims

open Claims.Sampling

theorem prBad_congr {Coin W : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {H : Msg → Msg} {V : Verifier Coin} {R R' : Rel W}
    {E : Extractor Coin W V} {x : Msg} {Ω : Type} [Fintype Ω] (P : Prover Ω) (h : ∀ w, R x w ↔ R' x w) :
    PrBad H V R E x P = PrBad H V R' E x P := by
  have he : ∀ o, R.ext x o ↔ R'.ext x o := fun o => by cases o <;> simp [Rel.ext, h]
  unfold PrBad
  simp only [he]

theorem ksLinOn_of {Coin W : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {H : Msg → Msg} {V : Verifier Coin} {S S' : Msg → Prop}
    {R R' : Rel W} {E : Extractor Coin W V} {κ γ β : Msg → ℝ} {Fe : FinderFamily} (hS : ∀ x, S x → S' x)
    (hR : ∀ x, S x → ∀ w, R x w ↔ R' x w) (h : KSLinOn H V S' R' E κ γ β Fe) : KSLinOn H V S R E κ γ β Fe := by
  intro x hx Ω _ _ P
  rw [prBad_congr P (hR x hx)]
  exact h x (hS x hx) Ω P

end Claims

namespace Claims.SamplingGeneric

open Claims
open Claims.Sampling

namespace Compiled

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}

open Classical in
/-- **`L` at one level**: its levels on a statement in scope the one level `LevelsRealizeOne` gives, no top
partition, no sample above the last. -/
noncomputable def liftLev (L : Compiled Coin A) : CompiledLev Coin A where
  toCompiled := L
  levels s Pt :=
    if h : InScope Pt s ∧ ∃ Lv : Verity.Levels (Fin Pt.blocks.length),
        TopIs Pt Lv none ∧ Lv.SlotsAreBlocks Pt (slot Pt) ∧ Lv.MissesAsDesign s.design Pt (slot Pt) then
      some h.2.choose
    else none
  top _ := none
  upper _ _ _ := []
  stages _ _ := ∅

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem liftLev_levels {L : Compiled Coin A} {s : AuditStatement} {Pt : Partition}
    {Lv : Verity.Levels (Fin Pt.blocks.length)} (h : L.liftLev.levels s Pt = some Lv) :
    InScope Pt s ∧ TopIs Pt Lv none ∧ Lv.SlotsAreBlocks Pt (slot Pt) ∧ Lv.MissesAsDesign s.design Pt (slot Pt) := by
  unfold liftLev at h
  dsimp only at h
  split_ifs at h with hc
  cases h
  exact ⟨hc.1, hc.2.choose_spec⟩

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem liftLev_levels_isSome (hreal : LevelsRealizeOne) (L : Compiled Coin A) {s : AuditStatement}
    {Pt : Partition} (hs : InScope Pt s) : ∃ Lv, L.liftLev.levels s Pt = some Lv := by
  refine ⟨_, dif_pos ⟨hs, hreal s Pt hs⟩⟩

theorem scope_liftLev (hreal : LevelsRealizeOne) {L : Compiled Coin A} {x : Msg} (h : L.Scope x) :
    L.liftLev.Scope x := by
  obtain ⟨s, Pt, hp, hq, hs⟩ := h
  obtain ⟨Lv, hLv⟩ := L.liftLev_levels_isSome hreal hs
  obtain ⟨-, htop, hslots, -⟩ := liftLev_levels hLv
  exact ⟨s, Pt, Lv, hp, hq, hLv, hs.1, hs.2.1, hslots, htop⟩

theorem scope_of_liftLev {L : Compiled Coin A} {x : Msg} (h : L.liftLev.Scope x) : L.Scope x := by
  obtain ⟨s, Pt, Lv, hp, hq, hLv, -⟩ := h
  exact ⟨s, Pt, hp, hq, (liftLev_levels hLv).1⟩

/-- **`L` at one level meets the contract at levels** where it meets `Compiled.Sessions`. -/
theorem sessions_liftLev (hreal : LevelsRealizeOne) {H : Msg → Msg} {L : Compiled Coin A} (S : L.Sessions H) :
    L.liftLev.Sessions H where
  samples x s Pt Lv hp hq hLv _ := by
    obtain ⟨hs, htop, -, hm⟩ := liftLev_levels hLv
    have h1 : Lv.levels.length = 1 := htop
    refine ⟨fun _ B hB => ?_, fun l l' rest he => ?_, fun j l' _ hj => ?_, fun lm _ h2 => ?_⟩
    · obtain ⟨esc, h0, hle, hEx⟩ := S.draws x s Pt hp hq hs B hB
      exact ⟨esc, h0, hle, (hm B hB).symm ▸ hEx⟩
    · rw [he] at h1
      simp at h1
    · omega
    · omega
  unions x _ _ _ _ _ _ _ := S.unions x
  slots := S.slots
  shared := S.shared
  keys := S.keys
  refuses x Ω P ω r hacc := scope_liftLev hreal (S.refuses x Ω P ω r hacc)
  interior x Ω P ω r e ρ := ⟨S.interior x Ω P ω r e ρ, fun _ =>
    congrArg (fun r' => L.iroots x (run H P L.V x ω r').seen ρ) (forkOn_empty (Coin := Coin) r e)⟩
  fixed := S.fixed
  runs := S.runs
  opens x s Pt Lv Ω Pv ω r u xS vS w hp hq hLv _ hacc hu hsess hR := by
    obtain ⟨b, hb, hc⟩ := S.opens x s Pt Ω Pv ω r u xS vS w hp hq (liftLev_levels hLv).1 hacc hu hsess hR
    exact ⟨b, u, b, hb, rfl, hb, hc⟩

theorem rlev_liftLev_iff (hreal : LevelsRealizeOne) {H : Msg → Msg} {L : Compiled Coin A} {x : Msg}
    (hx : L.Scope x) (δ : ℝ) (w : Wit) :
    R H L.commit L.parse L.partition δ x w ↔
      RLev H L.liftLev.commit L.liftLev.parse L.liftLev.partition L.liftLev.levels L.liftLev.top δ x w := by
  constructor
  · rintro ⟨s, Pt, hp, hq, hR⟩
    obtain ⟨s', Pt', hp', hq', hs⟩ := hx
    rw [hp] at hp'
    cases hp'
    rw [hq] at hq'
    cases hq'
    obtain ⟨Lv, hLv⟩ := L.liftLev_levels_isSome hreal hs
    exact ⟨s, Pt, Lv, hp, hq, hLv, (rlev_one_iff (liftLev_levels hLv).2.2.2 δ w).2 hR⟩
  · rintro ⟨s, Pt, Lv, hp, hq, hLv, hι⟩
    exact ⟨s, Pt, hp, hq, (rlev_one_iff (liftLev_levels hLv).2.2.2 δ w).1 hι⟩

/-- **KS at levels at one level gives `Compiled.KS`.** -/
theorem ks_of_liftLev (hreal : LevelsRealizeOne) {H : Msg → Msg} {L : Compiled Coin A} (h : L.liftLev.KS H) :
    L.KS H := fun δ hδ =>
  ksLinOn_of (fun _ hx => scope_liftLev hreal hx) (fun _ hx w => rlev_liftLev_iff hreal hx δ w) (h δ hδ)

/-- **The committed form at levels at one level gives `Compiled.KSReg`.** -/
theorem ksReg_of_liftLev (hreal : LevelsRealizeOne) {H : Msg → Msg} {L : Compiled Coin A}
    (h : L.liftLev.KSRegLev H) : L.KSReg H := by
  intro x hx Ω _ _ P
  obtain ⟨s, Pt, hp, hq, hs⟩ := hx
  obtain ⟨h1, h2⟩ := h x (scope_liftLev hreal ⟨s, Pt, hp, hq, hs⟩) Ω P
  have hε : L.liftLev.εregLev H x P = L.εreg H x P := by
    unfold CompiledLev.εregLev
    have : (L.liftLev.parse x >>= L.liftLev.top) = none := by
      show (L.parse x >>= fun _ => none) = none
      cases L.parse x <;> rfl
    rw [this]
    rfl
  have hF : ∀ ω r, L.liftLev.FregLev H P x ω r = L.Freg H P x ω := by
    intro ω r
    unfold CompiledLev.FregLev
    show (match L.parse x with
      | none => ∅
      | some s => match L.partition s, (none : Option Partition) with
        | some _, none => L.Freg H P x ω
        | some Pt, some Q => _
        | none, _ => ∅) = _
    rw [hp]
    dsimp only
    rw [hq]
  refine ⟨?_, fun s' hp' δ hδ => ?_⟩
  · rw [← hε]
    refine le_trans (le_of_eq (Pr_congr fun p => ?_)) h1
    rw [hF]
    exact Iff.rfl
  · rw [hp] at hp'
    cases hp'
    obtain ⟨Lv, hLv⟩ := L.liftLev_levels_isSome hreal hs
    obtain ⟨-, -, -, hm⟩ := liftLev_levels hLv
    rw [← hε]
    refine le_trans (Pr_mono fun p ⟨hacc, hnot⟩ => ⟨hacc, fun ι hι => hnot ?_⟩) (h2 s Pt Lv hp hq hLv δ hδ)
    have hpart : L.part s = Pt := by simp [Compiled.part, hq]
    rw [hpart]
    have hsub : L.Freg H P x p.1 ⊆ incorrectLev s Pt none (L.regVal H P x p.1) ι := by
      intro v hv
      simp only [Compiled.Freg, hp, hpart, Finset.mem_filter] at hv
      simp only [incorrectLev, Finset.mem_filter]
      refine ⟨hv.1, fun ⟨u, ρ, r, hu, hρ, hr, hc⟩ => hv.2 ⟨u, hu, L.regVal H P x p.1, ι ρ, fun _ _ => rfl, ?_⟩⟩
      cases hρ
      have : r = u := Option.some_injective _ (hr.symm.trans hu)
      subst this
      exact hc
    have hrange : L.Freg H P x p.1 ⊆ Finset.range Pt.blocks.length :=
      hsub.trans (incorrectLev_subset_range _ _ _ _)
    show δ < s.design.Pa Pt _
    rw [← hm _ hrange]
    exact levM_down hsub hι

end Compiled


namespace CompiledTwo

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}

open Classical in
/-- **The replay sample `rs` as the first level's sample**: the nonempty top blocks of its replay blocks. -/
noncomputable def upperOf (Pt : Partition) (d : Design.Two) (rs : List ℕ) : Finset (Finset (Fin Pt.blocks.length)) :=
  ((Finset.range d.part.blocks.length).filter fun ρ => ρ ∈ rs ∧ (topBlock Pt d.part ρ).Nonempty).image
    (topBlock Pt d.part)

open Classical in
/-- **`L` at two levels**: its levels on a statement in scope the two levels `LevelsRealizeTwo` gives, its replay
partition the top partition, its replay sample the first level's. -/
noncomputable def liftLev (L : CompiledTwo Coin A) : CompiledLev Coin A where
  toCompiled := L.toCompiled
  levels s Pt :=
    match L.two s with
    | some d =>
      if h : InScopeTwo Pt d s ∧ ∃ Lv : Verity.Levels (Fin Pt.blocks.length), d.RealizedBy Pt Lv then
        some h.2.choose
      else none
    | none => none
  top s := (L.two s).map Design.Two.part
  upper x Pt r :=
    match L.parse x >>= L.two with
    | some d => [upperOf Pt d (L.replaySample x r)]
    | none => []
  stages _ _ := ∅

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem liftLev_levels {L : CompiledTwo Coin A} {s : AuditStatement} {Pt : Partition}
    {Lv : Verity.Levels (Fin Pt.blocks.length)} (h : L.liftLev.levels s Pt = some Lv) :
    ∃ d, L.two s = some d ∧ InScopeTwo Pt d s ∧ d.RealizedBy Pt Lv := by
  unfold liftLev at h
  dsimp only at h
  revert h
  cases hd : L.two s with
  | none => intro h; cases h
  | some d =>
    intro h
    dsimp only at h
    split_ifs at h with hc
    cases h
    exact ⟨d, rfl, hc.1, hc.2.choose_spec⟩

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem liftLev_levels_isSome (hreal : LevelsRealizeTwo) (L : CompiledTwo Coin A) {s : AuditStatement}
    {Pt : Partition} {d : Design.Two} (hd : L.two s = some d) (hs : InScopeTwo Pt d s) :
    ∃ Lv, L.liftLev.levels s Pt = some Lv := by
  classical
  have hc : InScopeTwo Pt d s ∧ ∃ Lv : Verity.Levels (Fin Pt.blocks.length), d.RealizedBy Pt Lv :=
    ⟨hs, hreal s Pt d hs⟩
  refine ⟨hc.2.choose, ?_⟩
  show (match L.two s with
    | some d =>
      if h : InScopeTwo Pt d s ∧ ∃ Lv : Verity.Levels (Fin Pt.blocks.length), d.RealizedBy Pt Lv then
        some h.2.choose
      else none
    | none => none) = _
  rw [hd]
  exact dif_pos hc

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem top_liftLev {L : CompiledTwo Coin A} {s : AuditStatement} {d : Design.Two} (hd : L.two s = some d) :
    L.liftLev.top s = some d.part := by
  show (L.two s).map Design.Two.part = _
  rw [hd]
  rfl

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem upper_liftLev {L : CompiledTwo Coin A} {x : Msg} {s : AuditStatement} {d : Design.Two}
    (hp : L.parse x = some s) (hd : L.two s = some d) (Pt : Partition) (r : Fin (L.liftLev.V.k x) → Coin) :
    L.liftLev.upper x Pt r = [upperOf Pt d (L.replaySample x r)] := by
  have h : (L.parse x >>= L.two) = some d := by rw [hp]; exact hd
  show (match L.parse x >>= L.two with
    | some d => [upperOf Pt d (L.replaySample x r)]
    | none => []) = _
  rw [h]

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem scope_liftLev (hreal : LevelsRealizeTwo) {L : CompiledTwo Coin A} {x : Msg} (h : L.Scope x) :
    L.liftLev.Scope x := by
  obtain ⟨s, Pt, d, hp, hq, hd, hs⟩ := h
  obtain ⟨Lv, hLv⟩ := L.liftLev_levels_isSome hreal hd hs
  obtain ⟨d', hd', -, htop, hslots, -⟩ := liftLev_levels hLv
  rw [hd] at hd'
  cases hd'
  refine ⟨s, Pt, Lv, hp, hq, hLv, hs.1, hs.2.1, hslots, ?_⟩
  rw [top_liftLev hd]
  exact htop

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem scope_of_liftLev {L : CompiledTwo Coin A} {x : Msg} (h : L.liftLev.Scope x) : L.Scope x := by
  obtain ⟨s, Pt, Lv, hp, hq, hLv, -⟩ := h
  obtain ⟨d, hd, hs, -⟩ := liftLev_levels hLv
  exact ⟨s, Pt, d, hp, hq, hd, hs⟩

theorem escCell_empty (k : ℕ) (X : Finset ℕ) : escCell (∅ : Finset ℕ).card k (X ∩ ∅).card = 1 := by
  simp [escCell]

/-- **`L` at two levels meets the contract at levels** where it meets `CompiledTwo.Sessions`. -/
theorem sessions_liftLev (hreal : LevelsRealizeTwo) {H : Msg → Msg} {L : CompiledTwo Coin A} (S : L.Sessions H) :
    L.liftLev.Sessions H where
  samples x s Pt Lv hp hq hLv _ := by
    classical
    obtain ⟨d, hd, hs, -, -, -, l₁, l₂, hL, h₁, h₂⟩ := liftLev_levels hLv
    have hup := upper_liftLev hp hd Pt
    have hhd : ∀ a : Finset (Finset (Fin Pt.blocks.length)), [a].headD ∅ = a := fun _ => rfl
    have hld : ∀ a : Finset (Finset (Fin Pt.blocks.length)), [a].getLastD ∅ = a := fun _ => rfl
    refine ⟨fun h1 => absurd h1 (by rw [hL]; simp), fun l l' rest he T => ?_,
      fun j l' _ hj => absurd hj (by rw [hL]; simp), fun lm hlm _ r X => ?_⟩
    · rw [hL] at he
      obtain ⟨rfl, -⟩ := List.cons.inj he
      simp only [hup, hhd]
      refine le_trans (Pr_mono fun r hdis ρ hρ hρT => ?_)
        (le_trans (S.replaySamples x s Pt d hp hq hd hs _ (Finset.filter_subset _ _)) (h₁ T))
      rw [Finset.mem_filter] at hρT
      refine Finset.disjoint_left.1 hdis (Finset.mem_image.2 ⟨ρ, Finset.mem_filter.2 ⟨hρT.1, hρ, hρT.2.1⟩, rfl⟩)
        hρT.2.2
    · have hlm' : lm = l₂ := by
        rw [hL] at hlm
        simpa using hlm.symm
      subst hlm'
      simp only [hup, hld]
      refine le_trans (S.secondSamples x s Pt d hp hq hd hs r X) ?_
      unfold upperOf
      rw [Finset.prod_image fun a ha b hb hab =>
        topBlock_injOn (Finset.mem_filter.1 ha).2.2 hab]
      rw [← Finset.prod_subset (s₁ := (Finset.range d.part.blocks.length).filter fun ρ =>
          ρ ∈ L.replaySample x r ∧ (topBlock Pt d.part ρ).Nonempty)
        (fun ρ hρ => by
          rw [Finset.mem_filter] at hρ ⊢
          exact ⟨hρ.1, hρ.2.1⟩)
        (fun ρ hρ hρ' => by
          rw [Finset.mem_filter] at hρ hρ'
          have hne : ¬ (topBlock Pt d.part ρ).Nonempty := fun h => hρ' ⟨hρ.1, hρ.2, h⟩
          refine Finset.prod_eq_one fun t _ => ?_
          rw [cell_eq_empty hne t]
          exact escCell_empty _ _)]
      refine Finset.prod_le_prod₀ ?_ ?_
      · intro ρ _
        refine Finset.prod_nonneg fun t _ => ?_
        unfold escCell
        positivity
      · intro ρ hρ
        rw [Finset.mem_filter, Finset.mem_range] at hρ
        exact h₂ ρ hρ.1 hρ.2.2 X
  unions x s Pt Lv hp hq hLv _ := by
    classical
    obtain ⟨d, hd, hs, -⟩ := liftLev_levels hLv
    obtain ⟨hf, hi⟩ := S.unions x s Pt d hp hq hd hs
    rw [top_liftLev hd]
    refine ⟨hf, fun Ω P ω r e hacc => le_trans ?_ (hi Ω P ω r e hacc)⟩
    rw [upper_liftLev hp hd Pt r]
    show ((upperOf Pt d (L.replaySample x r)).biUnion id).card ≤ _
    refine Finset.card_le_card_of_injOn Fin.val ?_ fun a _ b _ h => Fin.ext h
    intro v hv
    obtain ⟨Y, hY, hvY⟩ := Finset.mem_biUnion.1 hv
    obtain ⟨ρ, hρ, rfl⟩ := Finset.mem_image.1 hY
    rw [Finset.mem_filter] at hρ
    simp only [id, topBlock, Finset.mem_filter, Finset.mem_univ, true_and] at hvY
    unfold sampleBlocks
    rw [Finset.mem_coe, Finset.mem_filter, Finset.mem_range]
    exact ⟨v.2, ρ, hρ.2.1, hvY⟩
  slots := S.slots
  shared := S.shared
  keys := S.keys
  refuses x Ω P ω r hacc := scope_liftLev hreal (S.refuses x Ω P ω r hacc)
  interior x Ω P ω r e ρ := ⟨S.interior x Ω P ω r e ρ, fun _ =>
    congrArg (fun r' => L.iroots x (run H P L.V x ω r').seen ρ) (forkOn_empty (Coin := Coin) r e)⟩
  fixed := S.fixed
  runs := S.runs
  opens x s Pt Lv Ω Pv ω r u xS vS w hp hq hLv _ hacc hu hsess hR := by
    obtain ⟨d, hd, hs, -⟩ := liftLev_levels hLv
    obtain ⟨b, ρ, rb, hb, hρ, hrb, hc⟩ := S.opens x s Pt d Ω Pv ω r u xS vS w hp hq hd hs hacc hu hsess hR
    refine ⟨b, ρ, rb, hb, ?_, ?_, hc⟩
    · rw [top_liftLev hd]
      exact hρ
    · rw [top_liftLev hd]
      exact hrb

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem rlev_liftLev_iff (hreal : LevelsRealizeTwo) {H : Msg → Msg} {L : CompiledTwo Coin A} {x : Msg}
    (hx : L.Scope x) (δ : ℝ) (w : Wit) :
    RTwo H L.commit L.parse L.partition L.two δ x w ↔
      RLev H L.liftLev.commit L.liftLev.parse L.liftLev.partition L.liftLev.levels L.liftLev.top δ x w := by
  constructor
  · rintro ⟨s, Pt, d, hp, hq, hd, hι⟩
    obtain ⟨s', Pt', d', hp', hq', hd', hs⟩ := hx
    rw [hp] at hp'
    cases hp'
    rw [hq] at hq'
    cases hq'
    rw [hd] at hd'
    cases hd'
    obtain ⟨Lv, hLv⟩ := L.liftLev_levels_isSome hreal hd hs
    obtain ⟨d', hd', -, -, -, hm, -⟩ := liftLev_levels hLv
    rw [hd] at hd'
    cases hd'
    refine ⟨s, Pt, Lv, hp, hq, hLv, ?_⟩
    rw [top_liftLev hd]
    exact (rlev_two_iff hm δ w).2 hι
  · rintro ⟨s, Pt, Lv, hp, hq, hLv, hι⟩
    obtain ⟨d, hd, -, -, -, hm, -⟩ := liftLev_levels hLv
    rw [top_liftLev hd] at hι
    exact ⟨s, Pt, d, hp, hq, hd, (rlev_two_iff hm δ w).1 hι⟩

/-- **KS at levels at two levels gives `CompiledTwo.KS`.** -/
theorem ks_of_liftLev (hreal : LevelsRealizeTwo) {H : Msg → Msg} {L : CompiledTwo Coin A} (h : L.liftLev.KS H) :
    L.KS H := fun δ hδ =>
  ksLinOn_of (fun _ hx => scope_liftLev hreal hx) (fun _ hx w => rlev_liftLev_iff hreal hx δ w) (h δ hδ)

/-- **The committed form at levels at two levels gives `CompiledTwo.KSRegTwo`.** -/
theorem ksRegTwo_of_liftLev (hreal : LevelsRealizeTwo) {H : Msg → Msg} {L : CompiledTwo Coin A}
    (h : L.liftLev.KSRegLev H) : L.KSRegTwo H := by
  intro x hx Ω _ _ P
  obtain ⟨s, Pt, d, hp, hq, hd, hs⟩ := hx
  obtain ⟨h1, h2⟩ := h x (scope_liftLev hreal ⟨s, Pt, d, hp, hq, hd, hs⟩) Ω P
  have ht : (L.liftLev.parse x >>= L.liftLev.top) = some d.part := by
    show (L.parse x >>= fun s => (L.two s).map Design.Two.part) = _
    rw [hp]
    show (L.two s).map Design.Two.part = _
    rw [hd]
    rfl
  have hε : L.liftLev.εregLev H x P = L.εregTwo H x P := by
    unfold CompiledLev.εregLev
    rw [ht]
    rfl
  have hF : ∀ ω r, L.liftLev.FregLev H P x ω r = L.FregTwo H P x ω r := by
    intro ω r
    have htop : L.liftLev.top s = some d.part := top_liftLev hd
    unfold CompiledLev.FregLev CompiledTwo.FregTwo
    show (match L.parse x with
      | none => ∅
      | some s => match L.partition s, L.liftLev.top s with
        | some _, none => _
        | some Pt, some Q => _
        | none, _ => ∅) = (match L.parse x with
      | none => ∅
      | some s => match L.partition s, L.two s with
        | some Pt, some d => _
        | _, _ => ∅)
    rw [hp]
    dsimp only
    rw [hq, htop, hd]
    rfl
  refine ⟨?_, fun s' Pt' d' hp' hq' hd' δ hδ => ?_⟩
  · rw [← hε]
    refine le_trans (le_of_eq (Pr_congr fun p => ?_)) h1
    rw [hF]
    exact Iff.rfl
  · rw [hp] at hp'
    cases hp'
    rw [hq] at hq'
    cases hq'
    rw [hd] at hd'
    cases hd'
    obtain ⟨Lv, hLv⟩ := L.liftLev_levels_isSome hreal hd hs
    obtain ⟨d', hd', -, -, -, hm, -⟩ := liftLev_levels hLv
    rw [hd] at hd'
    cases hd'
    rw [← hε]
    refine le_trans (le_of_eq (Pr_congr fun p => ?_)) (h2 s Pt Lv hp hq hLv δ hδ)
    rw [top_liftLev hd]
    exact and_congr Iff.rfl (forall_congr' fun ι => not_congr (incorrectLev_mem_levM_iff hm δ _ ι).symm)

end CompiledTwo

/-- **`CompilerKSLev` at one level is `CompilerKS`**, where one level realizes the claims designs. -/
theorem compilerKS_of_lev (hreal : LevelsRealizeOne) (h : CompilerKSLev) : CompilerKS :=
  fun Coin _ _ _ A L H hA hB hS => Compiled.ks_of_liftLev hreal (h Coin A L.liftLev H hA hB
    (Compiled.sessions_liftLev hreal hS))

/-- **`CompilerKSRegLev` at one level is `CompilerKSReg`**, where one level realizes the claims designs. -/
theorem compilerKSReg_of_lev (hreal : LevelsRealizeOne) (h : CompilerKSRegLev) : CompilerKSReg :=
  fun Coin _ _ _ A L H hA hB hS => Compiled.ksReg_of_liftLev hreal (h Coin A L.liftLev H hA hB
    (Compiled.sessions_liftLev hreal hS))


/-- **`CompilerKSLev` at two levels is `CompilerKSTwo`**, where two levels realize the two-stage designs. -/
theorem compilerKSTwo_of_lev (hreal : LevelsRealizeTwo) (h : CompilerKSLev) : CompilerKSTwo :=
  fun Coin _ _ _ A L H hA hB hS => CompiledTwo.ks_of_liftLev hreal (h Coin A L.liftLev H hA hB
    (CompiledTwo.sessions_liftLev hreal hS))

/-- **`CompilerKSRegLev` at two levels is `CompilerKSRegTwo`**, where two levels realize the two-stage designs. -/
theorem compilerKSRegTwo_of_lev (hreal : LevelsRealizeTwo) (h : CompilerKSRegLev) : CompilerKSRegTwo :=
  fun Coin _ _ _ A L H hA hB hS => CompiledTwo.ksRegTwo_of_liftLev hreal (h Coin A L.liftLev H hA hB
    (CompiledTwo.sessions_liftLev hreal hS))

end Claims.SamplingGeneric
