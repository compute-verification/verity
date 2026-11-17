module

public import Proofs.Properties.Verity.Sampled.Target
public import Proofs.Flock.SamplingGeneric.Guarantees

/-! The two-stage subprotocol's claim with replay blocks as its units. -/

@[expose] public section

namespace Verity.SampledTwo

open Claims
open scoped ENNReal

variable {ι : Type} {d : ResourceAccounting.Device ι} {α : Type} [Fintype α]

def ReplayCorrect (H : Claims.Msg → Claims.Msg) (Cm : Claims.Sampling.CommitmentScheme Bool) (s : Claims.Sampling.AuditStatement) (Pt : Claims.Sampling.Partition)
    (dd : Claims.Sampling.Design.Two) (w : Claims.Sampling.Wit) (ρ : ℕ) : Prop :=
  ∃ ιρ : ℕ → Bool, ∀ t < Pt.strata.length, Claims.Sampling.incorrectIn H Cm s Pt dd.part w ρ t ιρ = ∅

open Classical in
noncomputable def incorrectReplay (H : Claims.Msg → Claims.Msg) (Cm : Claims.Sampling.CommitmentScheme Bool) (s : Claims.Sampling.AuditStatement) (Pt : Claims.Sampling.Partition)
    (dd : Claims.Sampling.Design.Two) (w : Claims.Sampling.Wit) : Finset ℕ :=
  (Finset.range dd.part.blocks.length).filter fun ρ => ¬ ReplayCorrect H Cm s Pt dd w ρ

def CapsAt (dd : Claims.Sampling.Design.Two) (Pt : Claims.Sampling.Partition) (m δ : ℝ) : Prop :=
  ∀ fp : ℕ → ℕ → ℕ, m < ((dd.incorrectBlocks Pt fp).card : ℝ) → dd.Pa Pt fp ≤ δ

theorem incorrectReplay_subset (H : Claims.Msg → Claims.Msg) (Cm : Claims.Sampling.CommitmentScheme Bool) (s : Claims.Sampling.AuditStatement) (Pt : Claims.Sampling.Partition)
    (dd : Claims.Sampling.Design.Two) (w : Claims.Sampling.Wit) (ι' : ℕ → ℕ → Bool) :
    incorrectReplay H Cm s Pt dd w ⊆ dd.incorrectBlocks Pt (Claims.Sampling.footprint H Cm s Pt dd.part w ι') := by
  classical
  intro ρ hρ
  simp only [incorrectReplay, Finset.mem_filter] at hρ
  obtain ⟨hlt, hnc⟩ := hρ
  simp only [Claims.Sampling.Design.Two.incorrectBlocks, Finset.mem_filter]
  refine ⟨hlt, ?_⟩
  by_contra hno
  apply hnc
  refine ⟨ι' ρ, fun t ht => ?_⟩
  have h0 : ¬ 0 < Claims.Sampling.footprint H Cm s Pt dd.part w ι' ρ t := fun hp => hno ⟨t, ht, hp⟩
  have : (Claims.Sampling.incorrectIn H Cm s Pt dd.part w ρ t (ι' ρ)).card = 0 := Nat.eq_zero_of_not_pos h0
  exact Finset.card_eq_zero.1 this

theorem not_rTwo_of_incorrect (H : Claims.Msg → Claims.Msg) (Cm : Claims.Sampling.CommitmentScheme Bool) (parse : Claims.Msg → Option Claims.Sampling.AuditStatement)
    (partition : Claims.Sampling.AuditStatement → Option Claims.Sampling.Partition) (two : Claims.Sampling.AuditStatement → Option Claims.Sampling.Design.Two) (δ m : ℝ) (x : Claims.Msg)
    (s : Claims.Sampling.AuditStatement) (Pt : Claims.Sampling.Partition) (dd : Claims.Sampling.Design.Two) (hs : parse x = some s) (hP : partition s = some Pt)
    (hd : two s = some dd) (hcap : CapsAt dd Pt m δ) (w : Claims.Sampling.Wit)
    (hw : m < ((incorrectReplay H Cm s Pt dd w).card : ℝ)) :
    ¬ Claims.Sampling.RTwo H Cm parse partition two δ x w := by
  rintro ⟨s', Pt', d', hs', hP', hd', ι', hι⟩
  rw [hs] at hs'
  cases hs'
  rw [hP] at hP'
  cases hP'
  rw [hd] at hd'
  cases hd'
  have hsub := incorrectReplay_subset H Cm s Pt dd w ι'
  have hcard : ((incorrectReplay H Cm s Pt dd w).card : ℝ) ≤
      ((dd.incorrectBlocks Pt (Claims.Sampling.footprint H Cm s Pt dd.part w ι')).card : ℝ) := by
    exact_mod_cast Finset.card_le_card hsub
  have hle := hcap _ (lt_of_lt_of_le hw hcard)
  exact absurd hι (not_lt.2 hle)

/-- A bijection of the units with the replay blocks under which each unit unbound in a witness's transcript
is a replay block not correct at it; the instance owes it at the two-stage parameters. -/
def UnitsAreReplay (V : Vocab d α) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray) (H : Claims.Msg → Claims.Msg)
    (Cm : Claims.Sampling.CommitmentScheme Bool) (s : Claims.Sampling.AuditStatement) (Pt : Claims.Sampling.Partition) (dd : Claims.Sampling.Design.Two)
    (readX : Claims.Sampling.Wit → Verity.Circuit.Trace (V.circuitOf θ F)) : Prop :=
  ∃ e : Fin (V.units θ F) ↪ ℕ, ∀ (w : Claims.Sampling.Wit) (u : Fin (V.units θ F)),
    u ∈ ResourceAccounting.unbound (V.partsOf θ F) (V.Opens θ F c) (readX w) → e u ∈ incorrectReplay H Cm s Pt dd w

theorem holds_of_rTwo (V : Vocab d α) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray) (H : Claims.Msg → Claims.Msg)
    (Cm : Claims.Sampling.CommitmentScheme Bool) (parse : Claims.Msg → Option Claims.Sampling.AuditStatement)
    (partition : Claims.Sampling.AuditStatement → Option Claims.Sampling.Partition) (two : Claims.Sampling.AuditStatement → Option Claims.Sampling.Design.Two) (δ : ℝ) (x : Claims.Msg)
    (s : Claims.Sampling.AuditStatement) (Pt : Claims.Sampling.Partition) (dd : Claims.Sampling.Design.Two) (hs : parse x = some s) (hP : partition s = some Pt)
    (hd : two s = some dd) (readX : Claims.Sampling.Wit → Verity.Circuit.Trace (V.circuitOf θ F))
    (hU : UnitsAreReplay V θ F c H Cm s Pt dd readX)
    (hcap : CapsAt dd Pt ((V.certOf θ).ε * V.units θ F) δ) (w : Claims.Sampling.Wit)
    (hR : Claims.Sampling.RTwo H Cm parse partition two δ x w) :
    ResourceAccounting.IntegrityHolds (V.partsOf θ F) (V.Opens θ F c) (readX w) (V.certOf θ).ε := by
  classical
  obtain ⟨e, he⟩ := hU
  have hle : ((incorrectReplay H Cm s Pt dd w).card : ℝ) ≤ (V.certOf θ).ε * V.units θ F := by
    by_contra hlt
    exact not_rTwo_of_incorrect H Cm parse partition two δ _ x s Pt dd hs hP hd hcap w (not_le.1 hlt) hR
  have hmap : (ResourceAccounting.unbound (V.partsOf θ F) (V.Opens θ F c) (readX w)).map e ⊆
      incorrectReplay H Cm s Pt dd w := by
    intro ρ hρ
    obtain ⟨u, hu, rfl⟩ := Finset.mem_map.1 hρ
    exact he w u hu
  have hc := Finset.card_le_card hmap
  rw [Finset.card_map] at hc
  unfold ResourceAccounting.IntegrityHolds
  exact le_trans (by exact_mod_cast hc) hle

end Verity.SampledTwo
