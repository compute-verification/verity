module

public import Proofs.Properties.Verity.Sampled.Target
public import Proofs.Flock.SamplingGeneric.Guarantees

/-!
# The two-stage leg's claim, at replay blocks

A proof block's interior values are committed only after the first stage's sample, so at the two-stage form the
claim's units are the replay blocks: a replay block is bound when some interior values make each of its proof blocks
correct on the boundary values the commitment opens.

* **A correct replay block** (`ReplayCorrect`, `wrongReplay`): at some interiors, no cell of it has an incorrect
  proof block (`Claims.Sampling.wrongIn`). `wrongIn` reads every value the replay block's boundary doesn't commit from
  the interiors, and every other from what the witness opens, none where it opens none; `Prog.Correct` asks every value
  a proof block reads from outside it and every value its boundary commits to be opened. So a replay block whose
  boundary the witness leaves unopened at any position is not correct, as `Accounting.unbound` counts a unit with an
  unopened value.
* **The design's cap at a count** (`CapsAt`): every footprint that marks more than `m` replay blocks incorrect is
  accepted with probability at most `δ` (`Design.Two.Pa`).
* **The relation fails past the cap** (`not_rTwo_of_wrong`): a witness with more than `m` replay blocks not correct is
  not in the two-stage relation at `δ`, at any interiors.
* **The claim at the replay blocks** (`UnitsAreReplay`, `holds_of_rTwo`): where the vocabulary's units are the replay
  blocks and a unit unbound in the transcript a witness is read as is a replay block not correct at it, a witness in
  the relation at a `δ` that caps the claim's share makes the claim hold on its transcript.

Nothing imports this module.
-/

@[expose] public section

namespace Verity.SampledTwo

open Claims
open scoped ENNReal

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

/-- **Replay block `ρ` is correct at the witness `w`**: at some interiors, no cell of `ρ` in a stratum of `Pt` has an
incorrect proof block. -/
def ReplayCorrect (H : Claims.Msg → Claims.Msg) (Cm : Claims.Sampling.CommitmentScheme Bool) (s : Claims.Sampling.AuditStatement) (Pt : Claims.Sampling.Partition)
    (dd : Claims.Sampling.Design.Two) (w : Claims.Sampling.Wit) (ρ : ℕ) : Prop :=
  ∃ ιρ : ℕ → Bool, ∀ t < Pt.strata.length, Claims.Sampling.wrongIn H Cm s Pt dd.part w ρ t ιρ = ∅

open Classical in
/-- **The replay blocks not correct at `w`**. -/
noncomputable def wrongReplay (H : Claims.Msg → Claims.Msg) (Cm : Claims.Sampling.CommitmentScheme Bool) (s : Claims.Sampling.AuditStatement) (Pt : Claims.Sampling.Partition)
    (dd : Claims.Sampling.Design.Two) (w : Claims.Sampling.Wit) : Finset ℕ :=
  (Finset.range dd.part.blocks.length).filter fun ρ => ¬ ReplayCorrect H Cm s Pt dd w ρ

/-- **The design caps `m` replay blocks at `δ`**: every footprint marking more than `m` replay blocks incorrect has
acceptance probability at most `δ`. -/
def CapsAt (dd : Claims.Sampling.Design.Two) (Pt : Claims.Sampling.Partition) (m δ : ℝ) : Prop :=
  ∀ fp : ℕ → ℕ → ℕ, m < ((dd.wrongBlocks Pt fp).card : ℝ) → dd.Pa Pt fp ≤ δ

/-- Every replay block not correct at `w` is one the footprint at any interiors marks incorrect. -/
theorem wrongReplay_subset (H : Claims.Msg → Claims.Msg) (Cm : Claims.Sampling.CommitmentScheme Bool) (s : Claims.Sampling.AuditStatement) (Pt : Claims.Sampling.Partition)
    (dd : Claims.Sampling.Design.Two) (w : Claims.Sampling.Wit) (ι' : ℕ → ℕ → Bool) :
    wrongReplay H Cm s Pt dd w ⊆ dd.wrongBlocks Pt (Claims.Sampling.footprint H Cm s Pt dd.part w ι') := by
  classical
  intro ρ hρ
  simp only [wrongReplay, Finset.mem_filter] at hρ
  obtain ⟨hlt, hnc⟩ := hρ
  simp only [Claims.Sampling.Design.Two.wrongBlocks, Finset.mem_filter]
  refine ⟨hlt, ?_⟩
  by_contra hno
  apply hnc
  refine ⟨ι' ρ, fun t ht => ?_⟩
  have h0 : ¬ 0 < Claims.Sampling.footprint H Cm s Pt dd.part w ι' ρ t := fun hp => hno ⟨t, ht, hp⟩
  have : (Claims.Sampling.wrongIn H Cm s Pt dd.part w ρ t (ι' ρ)).card = 0 := Nat.eq_zero_of_not_pos h0
  exact Finset.card_eq_zero.1 this

/-- **The relation fails past the cap**: where the design caps `m` replay blocks at `δ`, a witness with more than `m`
replay blocks not correct is not in the two-stage relation at `δ`. -/
theorem not_rTwo_of_wrong (H : Claims.Msg → Claims.Msg) (Cm : Claims.Sampling.CommitmentScheme Bool) (parse : Claims.Msg → Option Claims.Sampling.AuditStatement)
    (partition : Claims.Sampling.AuditStatement → Option Claims.Sampling.Partition) (two : Claims.Sampling.AuditStatement → Option Claims.Sampling.Design.Two) (δ m : ℝ) (x : Claims.Msg)
    (s : Claims.Sampling.AuditStatement) (Pt : Claims.Sampling.Partition) (dd : Claims.Sampling.Design.Two) (hs : parse x = some s) (hP : partition s = some Pt)
    (hd : two s = some dd) (hcap : CapsAt dd Pt m δ) (w : Claims.Sampling.Wit)
    (hw : m < ((wrongReplay H Cm s Pt dd w).card : ℝ)) :
    ¬ Claims.Sampling.RTwo H Cm parse partition two δ x w := by
  rintro ⟨s', Pt', d', hs', hP', hd', ι', hι⟩
  rw [hs] at hs'
  cases hs'
  rw [hP] at hP'
  cases hP'
  rw [hd] at hd'
  cases hd'
  have hsub := wrongReplay_subset H Cm s Pt dd w ι'
  have hcard : ((wrongReplay H Cm s Pt dd w).card : ℝ) ≤
      ((dd.wrongBlocks Pt (Claims.Sampling.footprint H Cm s Pt dd.part w ι')).card : ℝ) := by
    exact_mod_cast Finset.card_le_card hsub
  have hle := hcap _ (lt_of_lt_of_le hw hcard)
  exact absurd hι (not_lt.2 hle)

/-- **The vocabulary's units are the replay blocks** at the statement `s`, its partitions and the reading `readX` of a
witness as the vocabulary's transcript: a bijection of the units with the replay blocks under which each unit unbound
in a witness's transcript is a replay block not correct at it. The instance owes it at the two-stage parameters. -/
def UnitsAreReplay (V : Vocab d α) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray) (H : Claims.Msg → Claims.Msg)
    (Cm : Claims.Sampling.CommitmentScheme Bool) (s : Claims.Sampling.AuditStatement) (Pt : Claims.Sampling.Partition) (dd : Claims.Sampling.Design.Two)
    (readX : Claims.Sampling.Wit → Verity.Circuit.Trace (V.circuitOf θ F)) : Prop :=
  ∃ e : Fin (V.units θ F) ↪ ℕ, ∀ (w : Claims.Sampling.Wit) (u : Fin (V.units θ F)),
    u ∈ Accounting.unbound (V.partsOf θ F) (V.Opens θ F c) (readX w) → e u ∈ wrongReplay H Cm s Pt dd w

/-- **The claim holds on a witness the relation takes**: where the units are the replay blocks and the design caps the
claim's share `ε · n` at `δ`, a witness in the two-stage relation at `δ` makes the claim hold on its transcript. -/
theorem holds_of_rTwo (V : Vocab d α) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray) (H : Claims.Msg → Claims.Msg)
    (Cm : Claims.Sampling.CommitmentScheme Bool) (parse : Claims.Msg → Option Claims.Sampling.AuditStatement)
    (partition : Claims.Sampling.AuditStatement → Option Claims.Sampling.Partition) (two : Claims.Sampling.AuditStatement → Option Claims.Sampling.Design.Two) (δ : ℝ) (x : Claims.Msg)
    (s : Claims.Sampling.AuditStatement) (Pt : Claims.Sampling.Partition) (dd : Claims.Sampling.Design.Two) (hs : parse x = some s) (hP : partition s = some Pt)
    (hd : two s = some dd) (readX : Claims.Sampling.Wit → Verity.Circuit.Trace (V.circuitOf θ F))
    (hU : UnitsAreReplay V θ F c H Cm s Pt dd readX)
    (hcap : CapsAt dd Pt ((V.certOf θ).ε * V.units θ F) δ) (w : Claims.Sampling.Wit)
    (hR : Claims.Sampling.RTwo H Cm parse partition two δ x w) :
    Accounting.IntegrityHolds (V.partsOf θ F) (V.Opens θ F c) (readX w) (V.certOf θ).ε := by
  classical
  obtain ⟨e, he⟩ := hU
  have hle : ((wrongReplay H Cm s Pt dd w).card : ℝ) ≤ (V.certOf θ).ε * V.units θ F := by
    by_contra hlt
    exact not_rTwo_of_wrong H Cm parse partition two δ _ x s Pt dd hs hP hd hcap w (not_le.1 hlt) hR
  have hmap : (Accounting.unbound (V.partsOf θ F) (V.Opens θ F c) (readX w)).map e ⊆
      wrongReplay H Cm s Pt dd w := by
    intro ρ hρ
    obtain ⟨u, hu, rfl⟩ := Finset.mem_map.1 hρ
    exact he w u hu
  have hc := Finset.card_le_card hmap
  rw [Finset.card_map] at hc
  unfold Accounting.IntegrityHolds
  exact le_trans (by exact_mod_cast hc) hle

end Verity.SampledTwo
