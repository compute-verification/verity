module

public import Proofs.Properties.Verity.Sampled.Target
public import Proofs.Flock.SamplingGeneric.Guarantees

/-! The two-stage subprotocol's claim counted in inner blocks, beside `Claim`'s count in replay blocks. -/

@[expose] public section

namespace Verity.SampledTwo

open Claims
open scoped ENNReal

variable {ι : Type} {d : ResourceAccounting.Device ι} {α : Type} [Fintype α]

def grid (dd : Claims.Sampling.Design.Two) (Pt : Claims.Sampling.Partition) : Finset (ℕ × ℕ) :=
  Finset.range dd.part.blocks.length ×ˢ Finset.range Pt.strata.length

def innerCount (dd : Claims.Sampling.Design.Two) (Pt : Claims.Sampling.Partition) (fp : ℕ → ℕ → ℕ) : ℕ :=
  ∑ p ∈ grid dd Pt, fp p.1 p.2

def CellCapsAt (dd : Claims.Sampling.Design.Two) (Pt : Claims.Sampling.Partition) (m δ : ℝ) : Prop :=
  ∀ fp : ℕ → ℕ → ℕ, (∀ ρ t, fp ρ t ≤ (Claims.Sampling.cell Pt dd.part ρ t).card) →
    m < (innerCount dd Pt fp : ℝ) → dd.Pa Pt fp ≤ δ

open Classical in
noncomputable def incorrectInner (H : Claims.Msg → Claims.Msg) (Cm : Claims.Sampling.CommitmentScheme Bool)
    (s : Claims.Sampling.AuditStatement) (Pt : Claims.Sampling.Partition) (dd : Claims.Sampling.Design.Two)
    (w : Claims.Sampling.Wit) : Finset ℕ :=
  (Finset.range Pt.blocks.length).filter fun v => ∀ ι' : ℕ → ℕ → Bool,
    ∃ p ∈ grid dd Pt, v ∈ Claims.Sampling.incorrectIn H Cm s Pt dd.part w p.1 p.2 (ι' p.1)

theorem footprint_le_cell (H : Claims.Msg → Claims.Msg) (Cm : Claims.Sampling.CommitmentScheme Bool)
    (s : Claims.Sampling.AuditStatement) (Pt : Claims.Sampling.Partition) (dd : Claims.Sampling.Design.Two)
    (w : Claims.Sampling.Wit) (ι' : ℕ → ℕ → Bool) (ρ t : ℕ) :
    Claims.Sampling.footprint H Cm s Pt dd.part w ι' ρ t ≤ (Claims.Sampling.cell Pt dd.part ρ t).card := by
  classical
  unfold Claims.Sampling.footprint Claims.Sampling.incorrectIn
  exact Finset.card_filter_le _ _

theorem incorrectInner_card_le (H : Claims.Msg → Claims.Msg) (Cm : Claims.Sampling.CommitmentScheme Bool)
    (s : Claims.Sampling.AuditStatement) (Pt : Claims.Sampling.Partition) (dd : Claims.Sampling.Design.Two)
    (w : Claims.Sampling.Wit) (ι' : ℕ → ℕ → Bool) :
    (incorrectInner H Cm s Pt dd w).card ≤
      innerCount dd Pt (Claims.Sampling.footprint H Cm s Pt dd.part w ι') := by
  classical
  have hsub : incorrectInner H Cm s Pt dd w ⊆
      (grid dd Pt).biUnion fun p => Claims.Sampling.incorrectIn H Cm s Pt dd.part w p.1 p.2 (ι' p.1) := by
    intro v hv
    simp only [incorrectInner, Finset.mem_filter] at hv
    obtain ⟨p, hp, hvp⟩ := hv.2 ι'
    exact Finset.mem_biUnion.2 ⟨p, hp, hvp⟩
  unfold innerCount Claims.Sampling.footprint
  exact le_trans (Finset.card_le_card hsub) Finset.card_biUnion_le

theorem not_rTwo_of_incorrectInner (H : Claims.Msg → Claims.Msg) (Cm : Claims.Sampling.CommitmentScheme Bool)
    (parse : Claims.Msg → Option Claims.Sampling.AuditStatement)
    (partition : Claims.Sampling.AuditStatement → Option Claims.Sampling.Partition)
    (two : Claims.Sampling.AuditStatement → Option Claims.Sampling.Design.Two) (δ m : ℝ) (x : Claims.Msg)
    (s : Claims.Sampling.AuditStatement) (Pt : Claims.Sampling.Partition) (dd : Claims.Sampling.Design.Two)
    (hs : parse x = some s) (hP : partition s = some Pt) (hd : two s = some dd) (hcap : CellCapsAt dd Pt m δ)
    (w : Claims.Sampling.Wit) (hw : m < ((incorrectInner H Cm s Pt dd w).card : ℝ)) :
    ¬ Claims.Sampling.RTwo H Cm parse partition two δ x w := by
  rintro ⟨s', Pt', d', hs', hP', hd', ι', hι⟩
  rw [hs] at hs'
  cases hs'
  rw [hP] at hP'
  cases hP'
  rw [hd] at hd'
  cases hd'
  have hcard : ((incorrectInner H Cm s Pt dd w).card : ℝ) ≤
      (innerCount dd Pt (Claims.Sampling.footprint H Cm s Pt dd.part w ι') : ℝ) := by
    exact_mod_cast incorrectInner_card_le H Cm s Pt dd w ι'
  have hle := hcap _ (footprint_le_cell H Cm s Pt dd w ι') (lt_of_lt_of_le hw hcard)
  exact absurd hι (not_lt.2 hle)

/-- One to one, each unit unbound in a witness's transcript to an inner block incorrect at every interior;
the instance owes it at the two-stage parameters. -/
def UnitsAreInner (V : Vocab d α) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray) (H : Claims.Msg → Claims.Msg)
    (Cm : Claims.Sampling.CommitmentScheme Bool) (s : Claims.Sampling.AuditStatement) (Pt : Claims.Sampling.Partition)
    (dd : Claims.Sampling.Design.Two) (readX : Claims.Sampling.Wit → Verity.Circuit.Trace (V.circuitOf θ F)) : Prop :=
  ∃ e : Fin (V.units θ F) ↪ ℕ, ∀ (w : Claims.Sampling.Wit) (u : Fin (V.units θ F)),
    u ∈ ResourceAccounting.unbound (V.partsOf θ F) (V.Opens θ F c) (readX w) → e u ∈ incorrectInner H Cm s Pt dd w

theorem holds_of_rTwo_inner (V : Vocab d α) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (H : Claims.Msg → Claims.Msg) (Cm : Claims.Sampling.CommitmentScheme Bool)
    (parse : Claims.Msg → Option Claims.Sampling.AuditStatement)
    (partition : Claims.Sampling.AuditStatement → Option Claims.Sampling.Partition)
    (two : Claims.Sampling.AuditStatement → Option Claims.Sampling.Design.Two) (δ : ℝ) (x : Claims.Msg)
    (s : Claims.Sampling.AuditStatement) (Pt : Claims.Sampling.Partition) (dd : Claims.Sampling.Design.Two)
    (hs : parse x = some s) (hP : partition s = some Pt) (hd : two s = some dd)
    (readX : Claims.Sampling.Wit → Verity.Circuit.Trace (V.circuitOf θ F))
    (hU : UnitsAreInner V θ F c H Cm s Pt dd readX)
    (hcap : CellCapsAt dd Pt ((V.certOf θ).ε * V.units θ F) δ) (w : Claims.Sampling.Wit)
    (hR : Claims.Sampling.RTwo H Cm parse partition two δ x w) :
    ResourceAccounting.IntegrityHolds (V.partsOf θ F) (V.Opens θ F c) (readX w) (V.certOf θ).ε := by
  classical
  obtain ⟨e, he⟩ := hU
  have hle : ((incorrectInner H Cm s Pt dd w).card : ℝ) ≤ (V.certOf θ).ε * V.units θ F := by
    by_contra hlt
    exact not_rTwo_of_incorrectInner H Cm parse partition two δ _ x s Pt dd hs hP hd hcap w (not_le.1 hlt) hR
  have hmap : (ResourceAccounting.unbound (V.partsOf θ F) (V.Opens θ F c) (readX w)).map e ⊆
      incorrectInner H Cm s Pt dd w := by
    intro v hv
    obtain ⟨u, hu, rfl⟩ := Finset.mem_map.1 hv
    exact he w u hu
  have hc := Finset.card_le_card hmap
  rw [Finset.card_map] at hc
  unfold ResourceAccounting.IntegrityHolds
  exact le_trans (by exact_mod_cast hc) hle

end Verity.SampledTwo
