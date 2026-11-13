module

public import Proofs.Properties.Verity.Sampled.Target
public import Proofs.Flock.SamplingGeneric.Guarantees

/-!
# The two-stage leg's claim, at inner blocks

`Claim` counts the claim in outer blocks (replay blocks): an outer block with one incorrect inner block counts whole,
and the first stage sampling every outer block still leaves the second stage's miss of one inner block in each. This
module counts it in inner blocks (the proof blocks of `Pt`, each in one cell `(ρ, t)`), beside `Claim`'s count and
replacing nothing.

* **The inner blocks a footprint marks incorrect** (`grid`, `innerCount`): the sum of the footprint over the cells of
  every outer block and stratum, the grid `footBits` multiplies over.
* **The design caps `m` inner blocks at `δ`** (`CellCapsAt`): every footprint that fits its cells and marks more than
  `m` inner blocks incorrect is accepted with probability at most `δ` (`Design.Two.Pa`).
* **An inner block incorrect at every interior** (`wrongInner`): at any interiors, one per outer block, it is incorrect
  in its cell. Their count is at most the footprint's at any interiors (`wrongInner_card_le`).
* **The relation fails past the cap** (`not_rTwo_of_wrongInner`): a witness with more than `m` inner blocks incorrect at
  every interior is not in the two-stage relation at `δ`.
* **The claim at the inner blocks** (`UnitsAreInner`, `holds_of_rTwo_inner`): where the vocabulary's units map one to
  one into the inner blocks, and a unit unbound in the transcript a witness is read as maps to an inner block
  incorrect at every interior, a witness in the relation at a `δ` that caps the claim's share makes the claim hold on
  its transcript.

Nothing imports this module.
-/

@[expose] public section

namespace Verity.SampledTwo

open Claims
open scoped ENNReal

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

/-- **The cells**: each outer block of the design's replay partition with each stratum of `Pt`. -/
def grid (dd : Claims.Sampling.Design.Two) (Pt : Claims.Sampling.Partition) : Finset (ℕ × ℕ) :=
  Finset.range dd.part.blocks.length ×ˢ Finset.range Pt.strata.length

/-- **The inner blocks a footprint marks incorrect**: its sum over the cells. -/
def innerCount (dd : Claims.Sampling.Design.Two) (Pt : Claims.Sampling.Partition) (fp : ℕ → ℕ → ℕ) : ℕ :=
  ∑ p ∈ grid dd Pt, fp p.1 p.2

/-- **The design caps `m` inner blocks at `δ`**: every footprint at most its cells' sizes that marks more than `m`
inner blocks incorrect has acceptance probability at most `δ`. -/
def CellCapsAt (dd : Claims.Sampling.Design.Two) (Pt : Claims.Sampling.Partition) (m δ : ℝ) : Prop :=
  ∀ fp : ℕ → ℕ → ℕ, (∀ ρ t, fp ρ t ≤ (Claims.Sampling.cell Pt dd.part ρ t).card) →
    m < (innerCount dd Pt fp : ℝ) → dd.Pa Pt fp ≤ δ

open Classical in
/-- **The inner blocks incorrect at every interior**: at any interiors, one per outer block, some cell holds the inner
block among its incorrect ones. -/
noncomputable def wrongInner (H : Claims.Msg → Claims.Msg) (Cm : Claims.Sampling.CommitmentScheme Bool)
    (s : Claims.Sampling.AuditStatement) (Pt : Claims.Sampling.Partition) (dd : Claims.Sampling.Design.Two)
    (w : Claims.Sampling.Wit) : Finset ℕ :=
  (Finset.range Pt.blocks.length).filter fun v => ∀ ι' : ℕ → ℕ → Bool,
    ∃ p ∈ grid dd Pt, v ∈ Claims.Sampling.wrongIn H Cm s Pt dd.part w p.1 p.2 (ι' p.1)

/-- A footprint fits its cells. -/
theorem footprint_le_cell (H : Claims.Msg → Claims.Msg) (Cm : Claims.Sampling.CommitmentScheme Bool)
    (s : Claims.Sampling.AuditStatement) (Pt : Claims.Sampling.Partition) (dd : Claims.Sampling.Design.Two)
    (w : Claims.Sampling.Wit) (ι' : ℕ → ℕ → Bool) (ρ t : ℕ) :
    Claims.Sampling.footprint H Cm s Pt dd.part w ι' ρ t ≤ (Claims.Sampling.cell Pt dd.part ρ t).card := by
  classical
  unfold Claims.Sampling.footprint Claims.Sampling.wrongIn
  exact Finset.card_filter_le _ _

/-- The inner blocks incorrect at every interior are at most the footprint's count at any interiors. -/
theorem wrongInner_card_le (H : Claims.Msg → Claims.Msg) (Cm : Claims.Sampling.CommitmentScheme Bool)
    (s : Claims.Sampling.AuditStatement) (Pt : Claims.Sampling.Partition) (dd : Claims.Sampling.Design.Two)
    (w : Claims.Sampling.Wit) (ι' : ℕ → ℕ → Bool) :
    (wrongInner H Cm s Pt dd w).card ≤
      innerCount dd Pt (Claims.Sampling.footprint H Cm s Pt dd.part w ι') := by
  classical
  have hsub : wrongInner H Cm s Pt dd w ⊆
      (grid dd Pt).biUnion fun p => Claims.Sampling.wrongIn H Cm s Pt dd.part w p.1 p.2 (ι' p.1) := by
    intro v hv
    simp only [wrongInner, Finset.mem_filter] at hv
    obtain ⟨p, hp, hvp⟩ := hv.2 ι'
    exact Finset.mem_biUnion.2 ⟨p, hp, hvp⟩
  unfold innerCount Claims.Sampling.footprint
  exact le_trans (Finset.card_le_card hsub) Finset.card_biUnion_le

/-- **The relation fails past the cap**: where the design caps `m` inner blocks at `δ`, a witness with more than `m`
inner blocks incorrect at every interior is not in the two-stage relation at `δ`. -/
theorem not_rTwo_of_wrongInner (H : Claims.Msg → Claims.Msg) (Cm : Claims.Sampling.CommitmentScheme Bool)
    (parse : Claims.Msg → Option Claims.Sampling.AuditStatement)
    (partition : Claims.Sampling.AuditStatement → Option Claims.Sampling.Partition)
    (two : Claims.Sampling.AuditStatement → Option Claims.Sampling.Design.Two) (δ m : ℝ) (x : Claims.Msg)
    (s : Claims.Sampling.AuditStatement) (Pt : Claims.Sampling.Partition) (dd : Claims.Sampling.Design.Two)
    (hs : parse x = some s) (hP : partition s = some Pt) (hd : two s = some dd) (hcap : CellCapsAt dd Pt m δ)
    (w : Claims.Sampling.Wit) (hw : m < ((wrongInner H Cm s Pt dd w).card : ℝ)) :
    ¬ Claims.Sampling.RTwo H Cm parse partition two δ x w := by
  rintro ⟨s', Pt', d', hs', hP', hd', ι', hι⟩
  rw [hs] at hs'
  cases hs'
  rw [hP] at hP'
  cases hP'
  rw [hd] at hd'
  cases hd'
  have hcard : ((wrongInner H Cm s Pt dd w).card : ℝ) ≤
      (innerCount dd Pt (Claims.Sampling.footprint H Cm s Pt dd.part w ι') : ℝ) := by
    exact_mod_cast wrongInner_card_le H Cm s Pt dd w ι'
  have hle := hcap _ (footprint_le_cell H Cm s Pt dd w ι') (lt_of_lt_of_le hw hcard)
  exact absurd hι (not_lt.2 hle)

/-- **The vocabulary's units map into the inner blocks** at the statement `s`, its partitions and the reading `readX`
of a witness as the vocabulary's transcript: one to one, each unit unbound in a witness's transcript to an inner block
incorrect at every interior. The instance owes it at the two-stage parameters. -/
def UnitsAreInner (V : Vocab d α) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray) (H : Claims.Msg → Claims.Msg)
    (Cm : Claims.Sampling.CommitmentScheme Bool) (s : Claims.Sampling.AuditStatement) (Pt : Claims.Sampling.Partition)
    (dd : Claims.Sampling.Design.Two) (readX : Claims.Sampling.Wit → Verity.Circuit.Trace (V.circuitOf θ F)) : Prop :=
  ∃ e : Fin (V.units θ F) ↪ ℕ, ∀ (w : Claims.Sampling.Wit) (u : Fin (V.units θ F)),
    u ∈ Accounting.unbound (V.partsOf θ F) (V.Opens θ F c) (readX w) → e u ∈ wrongInner H Cm s Pt dd w

/-- **The claim holds on a witness the relation takes**: where the units map into the inner blocks and the design caps
the claim's share `ε · n` of inner blocks at `δ`, a witness in the two-stage relation at `δ` makes the claim hold on
its transcript. -/
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
    Accounting.IntegrityHolds (V.partsOf θ F) (V.Opens θ F c) (readX w) (V.certOf θ).ε := by
  classical
  obtain ⟨e, he⟩ := hU
  have hle : ((wrongInner H Cm s Pt dd w).card : ℝ) ≤ (V.certOf θ).ε * V.units θ F := by
    by_contra hlt
    exact not_rTwo_of_wrongInner H Cm parse partition two δ _ x s Pt dd hs hP hd hcap w (not_le.1 hlt) hR
  have hmap : (Accounting.unbound (V.partsOf θ F) (V.Opens θ F c) (readX w)).map e ⊆
      wrongInner H Cm s Pt dd w := by
    intro v hv
    obtain ⟨u, hu, rfl⟩ := Finset.mem_map.1 hv
    exact he w u hu
  have hc := Finset.card_le_card hmap
  rw [Finset.card_map] at hc
  unfold Accounting.IntegrityHolds
  exact le_trans (by exact_mod_cast hc) hle

end Verity.SampledTwo
