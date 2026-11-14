module

public import Definitions.Claims.SamplingTwoServedX
public import Proofs.Flock.SamplingGeneric.TwoCount
public import Proofs.Flock.SamplingGeneric.TwoRegX
import Proofs.Flock.Claims.Prob

/-!
# What the two-stage audit's committed trace serves

`CompiledTwo.exfil_of_ks` counts what an accepted run serves at the extractor's witness. This file gives the count
at any trace, and so at the boundary trace read at the statement (`Compiled.regValX`), which is fixed before the
sample: `CompiledTwo.ExfilRegTwoX` from the committed form (`CompiledTwo.KSRegTwoX`).

Nothing of the determinism lemma is redone: `boundary_eq` and `outs_eq` read two traces, and `glue`, `wrongOf` and
`exits` read any values. What `TwoCount` states at a witness's openings is stated here at a trace:

* `inter_cell_wrongOfAt`, `counts_wrongOfAt`, `wrongOf_mem_admissibleAt`: the incorrect proof blocks of a trace at
  interiors `ι` are its footprint's, cell by cell, so a footprint in `M̄_δ` makes them admissible.
* `servedAtX`, `traceServed_eq_of`, `card_servedAtX_le`: the traces with inputs `a` whose incorrect set is `T` serve
  at most `2^{|exits T|}` outputs.
* `card_influenceTwoAt_le`: **the count**, `|influenceTwoAt| ≤ Σ_{T admissible} 2^{b(T)}` on a statement in scope.
* `influenceTwo_subset_at`: a witness's values are a trace, so `influenceTwo` is within `influenceTwoAt` and
  `card_influenceTwo_le` is the count's case at the witnesses.
* `CompiledTwo.exfilRegTwoX_of_ks`, `compilerExfilRegTwoX`: **`ExfilRegTwoX` from `KSRegTwoX`**. A trace whose
  footprint is in `M̄_δ` at some interiors serves an output of the influence set at its own inputs, so the event is
  within `KSRegTwoX`'s.
-/

@[expose] public section

namespace Claims.Sampling

open Finset

/-! ## The incorrect set at interiors -/

/-- **The incorrect proof blocks of a trace at interiors `ι` are its footprint's, cell by cell**
(`inter_cell_wrongOf` at a trace). -/
theorem inter_cell_wrongOfAt {s : AuditStatement} {Pt Pρ : Partition} (hP : Pt.Partitions s.prog)
    (hn : Pρ.gates.Nodup) (val : ℕ → Option Bool) (ι : ℕ → ℕ → Bool) (ρ t : ℕ) :
    wrongOf s.prog Pt (glue s.prog Pρ val ι) ∩ cell Pt Pρ ρ t = wrongInAt s Pt Pρ val ρ t (ι ρ) := by
  classical
  ext v
  unfold wrongOf wrongInAt
  simp only [mem_inter, mem_filter]
  constructor
  · rintro ⟨⟨_, hnc⟩, hc⟩
    refine ⟨hc, ?_⟩
    rintro ⟨u, r, hu, hr, hcor⟩
    obtain ⟨u', r', hu', hr', hur⟩ := cell_blocks hc
    obtain rfl := Option.some.inj (hu'.symm.trans hu)
    obtain rfl := Option.some.inj (hr'.symm.trans hr)
    exact hnc (by rw [getD_of_getElem? hu']; exact (glue_correct_iff hP hn hu' hr' hur).2 hcor)
  · rintro ⟨hc, hnc⟩
    refine ⟨⟨cell_subset Pt Pρ ρ t hc, fun hcor => hnc ?_⟩, hc⟩
    obtain ⟨u, r, hu, hr, hur⟩ := cell_blocks hc
    rw [getD_of_getElem? hu] at hcor
    exact ⟨u, r, hu, hr, (glue_correct_iff hP hn hu hr hur).1 hcor⟩

theorem counts_wrongOfAt {s : AuditStatement} {Pt Pρ : Partition} (hP : Pt.Partitions s.prog) (hn : Pρ.gates.Nodup)
    (val : ℕ → Option Bool) (ι : ℕ → ℕ → Bool) :
    counts Pt Pρ (wrongOf s.prog Pt (glue s.prog Pρ val ι)) = footprintAt s Pt Pρ val ι := by
  funext ρ t
  exact congrArg card (inter_cell_wrongOfAt hP hn val ι ρ t)

/-- **The incorrect set of a trace at interiors whose footprint is in `M̄_δ` is admissible**
(`wrongOf_mem_admissible` at a trace). -/
theorem wrongOf_mem_admissibleAt {s : AuditStatement} {Pt : Partition} {d : Design.Two} (hP : Pt.Partitions s.prog)
    (hn : d.part.gates.Nodup) {val : ℕ → Option Bool} {ι : ℕ → ℕ → Bool} {δ : ℝ}
    (hM : footprintAt s Pt d.part val ι ∈ d.M Pt δ) :
    wrongOf s.prog Pt (glue s.prog d.part val ι) ∈ admissibleTwo Pt d δ := by
  classical
  unfold admissibleTwo
  refine mem_filter.2 ⟨mem_powerset.2 fun v hv => ?_, ?_⟩
  · unfold wrongOf at hv
    exact (mem_filter.1 hv).1
  · rw [counts_wrongOfAt hP hn val ι]
    exact hM

/-! ## The count -/

open Classical in
/-- What the traces with inputs `a` whose incorrect set at some interiors is `T` serve (`servedAt` at a trace). -/
noncomputable def servedAtX (s : AuditStatement) (Pt Pρ : Partition) (a : ℕ → Option Bool) (T : Finset ℕ) :
    Finset (Fin s.prog.outs.length → Bool) :=
  univ.filter fun y => ∃ (val : ℕ → Option Bool) (ι : ℕ → ℕ → Bool), traceInputs s val = a ∧
    wrongOf s.prog Pt (glue s.prog Pρ val ι) = T ∧ traceServed s val = y

/-- Two traces at interiors with the incorrect set `T`, the same inputs and the same bits at `T`'s exits serve the
same outputs (`served_eq_of` at a trace). -/
theorem traceServed_eq_of {s : AuditStatement} {Pt Pρ : Partition} (hC : s.prog.Topological)
    (hP : Pt.Partitions s.prog) {val val' : ℕ → Option Bool} {ι ι' : ℕ → ℕ → Bool} {T : Finset ℕ}
    (hw : wrongOf s.prog Pt (glue s.prog Pρ val ι) = T) (hw' : wrongOf s.prog Pt (glue s.prog Pρ val' ι') = T)
    (ha : traceInputs s val = traceInputs s val')
    (hT : ∀ g ∈ exits s.prog Pt T,
      (glue s.prog Pρ val ι g).getD false = (glue s.prog Pρ val' ι' g).getD false) :
    traceServed s val = traceServed s val' := by
  classical
  have hcor : ∀ tr : ℕ → Option Bool, wrongOf s.prog Pt tr = T →
      ∀ v < Pt.blocks.length, v ∉ T → s.prog.Correct tr (Pt.blocks.getD v []) := by
    intro tr htr v hv hvT
    by_contra hc
    refine hvT (htr ▸ ?_)
    unfold wrongOf
    exact mem_filter.2 ⟨mem_range.2 hv, hc⟩
  have hin : ∀ g, s.prog.gates[g]? = some .input → glue s.prog Pρ val ι g = glue s.prog Pρ val' ι' g := by
    intro g hg
    have h1 := congrFun ha g
    unfold traceInputs at h1
    rw [if_pos hg, if_pos hg] at h1
    rw [glue_of_boundary fun r => Prog.boundary_of_input r hg, glue_of_boundary fun r => Prog.boundary_of_input r hg]
    exact h1
  have hTb : ∀ v ∈ T, ∀ g ∈ Pt.blocks.getD v [], s.prog.Boundary (Pt.blocks.getD v []) g = true →
      (glue s.prog Pρ val ι g).getD false = (glue s.prog Pρ val' ι' g).getD false :=
    fun v hv g hg hb => hT g (mem_biUnion.2 ⟨v, hv, List.mem_toFinset.2 (List.mem_filter.2 ⟨hg, hb⟩)⟩)
  funext i
  have ho : s.prog.outs[i] ∈ s.prog.outs := List.getElem_mem i.isLt
  have hglue : ∀ (tv : ℕ → Option Bool) (κ : ℕ → ℕ → Bool), s.prog.read tv s.prog.outs[i] =
      s.prog.read (glue s.prog Pρ tv κ) s.prog.outs[i] := fun tv κ =>
    s.prog.read_congr (glue_of_boundary fun r => Prog.boundary_of_out r ho).symm
  show (s.prog.read val s.prog.outs[i]).getD false = (s.prog.read val' s.prog.outs[i]).getD false
  rw [hglue val ι, hglue val' ι']
  exact outs_eq hC hP (hcor _ hw) (hcor _ hw') hin hTb _ ho

/-- **A set `T` of incorrect proof blocks serves at most `2^{|exits T|}` outputs** on fixed inputs
(`card_servedAt_le` at a trace). -/
theorem card_servedAtX_le {s : AuditStatement} {Pt Pρ : Partition} (hC : s.prog.Topological)
    (hP : Pt.Partitions s.prog) (a : ℕ → Option Bool) (T : Finset ℕ) :
    (servedAtX s Pt Pρ a T).card ≤ 2 ^ (exits s.prog Pt T).card := by
  classical
  have hspec : ∀ y ∈ servedAtX s Pt Pρ a T, ∃ (val : ℕ → Option Bool) (ι : ℕ → ℕ → Bool),
      traceInputs s val = a ∧ wrongOf s.prog Pt (glue s.prog Pρ val ι) = T ∧ traceServed s val = y :=
    fun y hy => (mem_filter.1 hy).2
  let f : (Fin s.prog.outs.length → Bool) → (exits s.prog Pt T → Bool) := fun y =>
    if h : ∃ (val : ℕ → Option Bool) (ι : ℕ → ℕ → Bool), traceInputs s val = a ∧
        wrongOf s.prog Pt (glue s.prog Pρ val ι) = T ∧ traceServed s val = y then
      fun g => (glue s.prog Pρ (Classical.choose h) (Classical.choose (Classical.choose_spec h)) g).getD false
    else fun _ => false
  calc (servedAtX s Pt Pρ a T).card ≤ (univ : Finset (exits s.prog Pt T → Bool)).card := by
        refine card_le_card_of_injOn f (fun _ _ => mem_coe.2 (mem_univ _)) ?_
        intro y hy y' hy' hf
        have h := hspec y hy
        have h' := hspec y' hy'
        simp only [f, h, h', ↓reduceDIte] at hf
        obtain ⟨ha, hw, hsw⟩ := Classical.choose_spec (Classical.choose_spec h)
        obtain ⟨ha', hw', hsw'⟩ := Classical.choose_spec (Classical.choose_spec h')
        rw [← hsw, ← hsw']
        exact traceServed_eq_of hC hP hw hw' (ha.trans ha'.symm) fun g hg => congrFun hf ⟨g, hg⟩
    _ = 2 ^ (exits s.prog Pt T).card := by rw [card_univ, Fintype.card_fun, Fintype.card_bool, Fintype.card_coe]

/-- **The exfiltration count at a trace's footprint**: on a statement in scope, at any input values, the traces whose
footprint is in `M̄_δ` at some interiors serve at most `Σ_{T admissible} 2^{b(T)}` outputs (`card_influenceTwo_le` at
a trace). -/
theorem card_influenceTwoAt_le {s : AuditStatement} {Pt : Partition} {d : Design.Two} (hs : InScopeTwo Pt d s)
    (δ : ℝ) (a : ℕ → Option Bool) :
    (influenceTwoAt s Pt d δ a).card ≤ ∑ T ∈ admissibleTwo Pt d δ, 2 ^ bitsOf s Pt T := by
  classical
  have hsub : influenceTwoAt s Pt d δ a ⊆ (admissibleTwo Pt d δ).biUnion (servedAtX s Pt d.part a) := by
    intro y hy
    unfold influenceTwoAt at hy
    obtain ⟨val, ha, ⟨ι, hM⟩, hwy⟩ := (mem_filter.1 hy).2
    refine mem_biUnion.2 ⟨_, wrongOf_mem_admissibleAt hs.2.1 hs.2.2.1 hM, ?_⟩
    unfold servedAtX
    exact mem_filter.2 ⟨mem_univ _, val, ι, ha, rfl, hwy⟩
  calc (influenceTwoAt s Pt d δ a).card ≤ ∑ T ∈ admissibleTwo Pt d δ, (servedAtX s Pt d.part a T).card :=
        (card_le_card hsub).trans card_biUnion_le
    _ ≤ ∑ T ∈ admissibleTwo Pt d δ, 2 ^ bitsOf s Pt T :=
        sum_le_sum fun T _ => (card_servedAtX_le hs.1 hs.2.1 a T).trans
          (Nat.pow_le_pow_right (by norm_num) (card_exits_le s Pt T))

/-- **A witness's values are a trace**: the influence set at the witnesses is within the traces'. -/
theorem influenceTwo_subset_at (H : Msg → Msg) (Cm : CommitmentScheme Bool) (s : AuditStatement) (Pt : Partition)
    (d : Design.Two) (δ : ℝ) (a : ℕ → Option Bool) :
    influenceTwo H Cm s Pt d δ a ⊆ influenceTwoAt s Pt d δ a := by
  classical
  intro y hy
  unfold influenceTwo at hy
  obtain ⟨w, ha, hM, hwy⟩ := (mem_filter.1 hy).2
  unfold influenceTwoAt
  exact mem_filter.2 ⟨mem_univ _, w.val H Cm s, ha, hM, hwy⟩

end Claims.Sampling

namespace Claims.SamplingGeneric

open Claims Finset
open Claims.Sampling (influenceTwoAt card_influenceTwoAt_le)

namespace CompiledTwo

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}

/-- **`ExfilRegTwoX` from `KSRegTwoX`**: an accepted run whose committed trace serves an output outside the influence
set at its own inputs has that trace's footprint outside `M̄_δ` at every interior, which is `KSRegTwoX`'s event; the
count is `card_influenceTwoAt_le`. -/
theorem exfilRegTwoX_of_ks {H : Msg → Msg} {L : CompiledTwo Coin A} (h : L.KSRegTwoX H) : L.ExfilRegTwoX H := by
  classical
  refine ⟨h, fun x hx Ω _ _ P s Pt d hs hPt hd δ hδ => ?_,
    fun x s Pt d a _ _ _ hin δ => card_influenceTwoAt_le hin δ a⟩
  refine le_trans (Pr_mono fun p hp => ⟨hp.1, fun ι hM => hp.2 ?_⟩) ((h x hx Ω P).2 s Pt d hs hPt hd δ hδ)
  unfold influenceTwoAt
  exact mem_filter.2 ⟨mem_univ _, L.regValX H P x, rfl, ⟨ι, hM⟩, rfl⟩

end CompiledTwo

/-- **The two-stage compiler bounds what the committed trace serves**: at `CompilerKSRegTwoX`'s premises, every
compiled two-stage verifier meets `CompiledTwo.ExfilRegTwoX`. -/
theorem compilerExfilRegTwoX (Coin : Type) [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (A : Arg Coin)
    (L : CompiledTwo Coin A) (H : Msg → Msg) (hA : A.KS H) (hB : L.commit.Binds H) (hS : L.Sessions H) :
    L.ExfilRegTwoX H :=
  CompiledTwo.exfilRegTwoX_of_ks (compilerKSRegTwoX Coin A L H hA hB hS)

end Claims.SamplingGeneric
