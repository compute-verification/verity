module

public import Definitions.Claims.Sampling
public import Definitions.Claims.SamplingTwo
public import Proofs.Flock.Claims.Witnesses.SamplingTwoWork
import Proofs.Flock.SamplingBoundary
import Proofs.Flock.SamplingGeneric.Two
import Proofs.Flock.SamplingGeneric.TwoCount

/-! Serve-time interiors recover compute credit at the proof-block grain, which interiors committed after the replay sample do not; the knowledge soundness of a compiler that opens each sampled replay block's interior against serve-time roots is not proved here. -/

@[expose] public section

namespace Claims.Sampling

def Prog.Checked (C : Prog) (u : List ℕ) (g : ℕ) : Prop :=
  (g ∈ u ∧ C.Boundary u g = true) ∨ (g ∉ u ∧ ∃ g' ∈ u, ∃ x : BGate, C.gates[g']? = some x ∧ g ∈ x.reads)

def OpensChecked (H : Msg → Msg) (Cm : CommitmentScheme Bool) (s : AuditStatement) (Pt Pρ : Partition) (w : Wit)
    (ι : ℕ → ℕ → Bool) : Prop :=
  ∀ (ρ : ℕ) (r : List ℕ), Pρ.blocks[ρ]? = some r → ∀ g ∈ r, s.prog.Boundary r g = false →
    (∃ (v : ℕ) (u : List ℕ), Pt.blocks[v]? = some u ∧ (∀ g' ∈ u, g' ∈ r) ∧ s.prog.Checked u g) →
      w.val H Cm s g = some (ι ρ g)

theorem faulty_iff_boundaryFaulty {H : Msg → Msg} {Cm : CommitmentScheme Bool} {s : AuditStatement} {P : Partition}
    {w : Wit} {v : ℕ} : Faulty H Cm s P w v ↔ BoundaryFaulty H Cm s P w v := by
  unfold Faulty BoundaryFaulty
  exact not_congr ⟨fun ⟨u, hu, ι, h⟩ => ⟨u, hu, (Boundary.core s.prog u _ ι).1 h⟩,
    fun ⟨u, hu, h⟩ => ⟨u, hu, fun _ => false, (Boundary.core s.prog u _ _).2 h⟩⟩

theorem correct_refIn_iff (C : Prog) {r u : List ℕ} {val : ℕ → Option Bool} {ι : ℕ → Bool}
    (hA : ∀ g ∈ u, C.Boundary u g = true → C.refIn r val ι g = val g)
    (hB : ∀ g ∈ u, ∀ x : BGate, C.gates[g]? = some x → ∀ i ∈ x.reads, i ∉ u → C.refIn r val ι i = val i) :
    C.Correct (C.refIn r val ι) u ↔ C.Correct val u := by
  rw [← Boundary.core C u (C.refIn r val ι) ι, ← Boundary.core C u val ι]
  refine C.correct_congr (fun g hg => ?_) fun g hg x hx i hi hiu => ?_
  · cases hb : C.Boundary u g
    · simp [Prog.refIn, hg, hb]
    · rw [Boundary.ref_boundary C hb, Boundary.ref_boundary C hb]
      exact hA g hg hb
  · rw [Boundary.ref_out C hiu, Boundary.ref_out C hiu]
    exact hB g hg x hx i hi hiu

theorem incorrectIn_eq_inter_faulty {H : Msg → Msg} {Cm : CommitmentScheme Bool} {s : AuditStatement}
    {Pt Pρ : Partition} {w : Wit} {ρ t : ℕ} {ι : ℕ → Bool}
    (hι : ∀ r : List ℕ, Pρ.blocks[ρ]? = some r → ∀ g ∈ r, s.prog.Boundary r g = false →
      (∃ (v : ℕ) (u : List ℕ), Pt.blocks[v]? = some u ∧ (∀ g' ∈ u, g' ∈ r) ∧ s.prog.Checked u g) →
        w.val H Cm s g = some (ι g)) :
    incorrectIn H Cm s Pt Pρ w ρ t ι = cell Pt Pρ ρ t ∩ faulty H Cm s Pt w := by
  have hiff : ∀ v ∈ cell Pt Pρ ρ t, ∀ u r, Pt.blocks[v]? = some u → Pρ.blocks[ρ]? = some r →
      (s.prog.Correct (s.prog.refIn r (w.val H Cm s) ι) u ↔ s.prog.Correct (w.val H Cm s) u) := by
    intro v hv u r hu hr
    obtain ⟨u', r', hu', hr', hur⟩ := cell_blocks hv
    rw [hu, Option.some.injEq] at hu'
    rw [hr, Option.some.injEq] at hr'
    subst hu' hr'
    have key : ∀ g, (∃ (v : ℕ) (u : List ℕ), Pt.blocks[v]? = some u ∧ (∀ g' ∈ u, g' ∈ r) ∧ s.prog.Checked u g) →
        s.prog.refIn r (w.val H Cm s) ι g = w.val H Cm s g := fun g hc => by
      by_cases hgr : g ∈ r
      · cases hb : s.prog.Boundary r g
        · rw [hι r hr g hgr hb hc]
          simp [Prog.refIn, hgr, hb]
        · exact Prog.refIn_of_boundary hb
      · exact Prog.refIn_of_not_mem hgr
    exact correct_refIn_iff s.prog (fun g hg hb => key g ⟨v, u, hu, hur, Or.inl ⟨hg, hb⟩⟩)
      fun g hg x hx i hi hiu => key i ⟨v, u, hu, hur, Or.inr ⟨hiu, g, hg, x, hx, hi⟩⟩
  ext v
  unfold incorrectIn faulty
  simp only [Finset.mem_filter, Finset.mem_inter, faulty_iff_boundaryFaulty, BoundaryFaulty]
  constructor
  · rintro ⟨hv, hn⟩
    refine ⟨hv, cell_subset _ _ _ _ hv, ?_⟩
    rintro ⟨u, hu, hc⟩
    obtain ⟨_, r, _, hr, -⟩ := cell_blocks hv
    exact hn ⟨u, r, hu, hr, (hiff v hv u r hu hr).2 hc⟩
  · rintro ⟨hv, -, hf⟩
    refine ⟨hv, ?_⟩
    rintro ⟨u, r, hu, hr, hc⟩
    exact hf ⟨u, hu, (hiff v hv u r hu hr).1 hc⟩

theorem footprint_eq_counts_faulty {H : Msg → Msg} {Cm : CommitmentScheme Bool} {s : AuditStatement}
    {Pt Pρ : Partition} {w : Wit} {ι : ℕ → ℕ → Bool} (hι : OpensChecked H Cm s Pt Pρ w ι) :
    footprint H Cm s Pt Pρ w ι = counts Pt Pρ (faulty H Cm s Pt w) := by
  unfold OpensChecked at hι
  funext ρ t
  unfold footprint counts
  rw [incorrectIn_eq_inter_faulty (hι ρ), Finset.inter_comm]

theorem counts_faulty_mem {H : Msg → Msg} {Cm : CommitmentScheme Bool} {s : AuditStatement} {Pt : Partition}
    {d : Design.Two} {w : Wit} {δ : ℝ} {ι : ℕ → ℕ → Bool} (hι : OpensChecked H Cm s Pt d.part w ι)
    (h : footprint H Cm s Pt d.part w ι ∈ d.M Pt δ) : counts Pt d.part (faulty H Cm s Pt w) ∈ d.M Pt δ := by
  rwa [footprint_eq_counts_faulty hι] at h

end Claims.Sampling

namespace Claims.Witness.SamplingTwoCredit

open Claims.Sampling Claims.Witness.SamplingGeneric

theorem notNotPρ_blocks : notNotPρ.blocks = [[0], [1, 2]] := rfl

theorem notNotTwo_Pa (fp : ℕ → ℕ → ℕ) :
    notNotTwo.Pa notNotPt fp = notNotTwo.esc notNotPt fp 0 * notNotTwo.esc notNotPt fp 1 := by
  unfold Design.Two.Pa
  have hl : notNotTwo.part.blocks.length = 2 := by decide
  rw [hl, Finset.sum_eq_single_of_mem ∅ (Finset.empty_mem_powerset _)]
  · have hw : window 0 2 (∅ : Finset ℕ) = ∅ := by decide
    rw [notNotTwo_replay_Pa]
    simp [hw, Finset.prod_range_succ]
  · intro T hT hne
    obtain ⟨v, hv⟩ := Finset.nonempty_iff_ne_empty.2 hne
    have hv2 : v < 2 := Finset.mem_range.1 (Finset.mem_powerset.1 hT hv)
    have hw : window 0 2 T ≠ ∅ := by
      intro hw
      have hvw : v ∈ window 0 2 T := by
        simp [window, hv]
        omega
      rw [hw] at hvw
      simp at hvw
    rw [notNotTwo_replay_Pa]
    simp [hw]

theorem notNotTwo_esc (fp : ℕ → ℕ → ℕ) (ρ : ℕ) :
    notNotTwo.esc notNotPt fp ρ = escCell (cell notNotPt notNotPρ ρ 0).card 1 (fp ρ 0) := by
  unfold Design.Two.esc
  have hs : notNotPt.strata.length = 1 := by decide
  rw [hs, Finset.prod_range_one]
  rfl

theorem counts_notNot_empty : counts notNotPt notNotPρ ∅ = fun _ _ => 0 := by
  funext ρ t
  simp [counts]

theorem fullW_opensChecked (H : Msg → Msg) :
    OpensChecked H notNotCm notNotStmt notNotPt notNotPρ fullW fun _ _ => false := by
  unfold OpensChecked
  intro ρ r hr g hg hb _
  rw [notNotPρ_blocks] at hr
  match ρ, hr with
  | 0, hr =>
    obtain rfl : [0] = r := Option.some.inj hr
    obtain rfl := List.mem_singleton.1 hg
    exact absurd hb (by decide)
  | 1, hr =>
    obtain rfl : [1, 2] = r := Option.some.inj hr
    rcases List.mem_cons.1 hg with rfl | hg
    · exact (fullW_val H).2.1
    · obtain rfl := List.mem_singleton.1 hg
      exact absurd hb (by decide)
  | _ + 2, hr => simp at hr

theorem lateW_not_opensChecked (H : Msg → Msg) (ι : ℕ → ℕ → Bool) :
    ¬ OpensChecked H notNotCm notNotStmt notNotPt notNotPρ lateW ι := by
  intro h
  unfold OpensChecked at h
  have hu : ∀ g' ∈ [1], g' ∈ [1, 2] := fun g' hg' => by
    obtain rfl := List.mem_singleton.1 hg'
    decide
  have h1 := h 1 [1, 2] rfl 1 (by decide) (by decide)
    ⟨1, [1], rfl, hu, Or.inl ⟨List.mem_singleton_self 1, by decide⟩⟩
  rw [(lateW_val H).2.1] at h1
  cases h1

theorem lateW_served_Pa (H : Msg → Msg) :
    notNotTwo.Pa notNotPt (counts notNotPt notNotPρ (faulty H notNotCm notNotStmt notNotPt lateW)) = 0 := by
  rw [notNotTwo_Pa, notNotTwo_esc _ 1, lateW_faulty]
  have h1 : (cell notNotPt notNotPρ 1 0).card = 2 := by decide
  have h2 : counts notNotPt notNotPρ {1, 2} 1 0 = 2 := by decide
  have h3 : escCell 2 1 2 = 0 := by
    unfold escCell
    rw [show Nat.choose (2 - 2) (min 1 2) = 0 by decide]
    simp
  rw [h1, h2, h3, mul_zero]

/-- **Serve-time interiors recover compute credit** at `notNotTwo`, where `RTwo` takes both traces: `fullW`'s
incorrect set's counts are in `M̄_δ` at every `δ < 1`; no interior agrees with `lateW`, which skipped gate `1`, and its
incorrect set's counts are in no `M̄_δ` at `δ ≥ 0`. -/
theorem served_credited (H : Msg → Msg) :
    OpensChecked H notNotCm notNotStmt notNotPt notNotPρ fullW (fun _ _ => false) ∧
      (∀ ι, ¬ OpensChecked H notNotCm notNotStmt notNotPt notNotPρ lateW ι) ∧
      (∀ δ : ℝ, δ < 1 →
        counts notNotPt notNotPρ (faulty H notNotCm notNotStmt notNotPt fullW) ∈ notNotTwo.M notNotPt δ) ∧
      ∀ δ : ℝ, 0 ≤ δ →
        counts notNotPt notNotPρ (faulty H notNotCm notNotStmt notNotPt lateW) ∉ notNotTwo.M notNotPt δ := by
  refine ⟨fullW_opensChecked H, lateW_not_opensChecked H, fun δ hδ => ?_, fun δ hδ hM => ?_⟩
  · show δ < notNotTwo.Pa notNotPt (counts notNotPt notNotPρ (faulty H notNotCm notNotStmt notNotPt fullW))
    rw [fullW_faulty, counts_notNot_empty, notNotTwo_Pa_zero]
    exact hδ
  · have h : δ < notNotTwo.Pa notNotPt (counts notNotPt notNotPρ (faulty H notNotCm notNotStmt notNotPt lateW)) :=
      hM
    rw [lateW_served_Pa] at h
    linarith

end Claims.Witness.SamplingTwoCredit
