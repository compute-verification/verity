module

public import Proofs.Properties.Verity.SampledTwo.CouplesWired

/-!
# `UnitsWired` in two parts

`UnitsAreReplay` asks for a labelling of the vocabulary's units by replay blocks under which each unit unbound in a
witness's transcript is a replay block not correct at it. `unitsAreReplay_of` gives it from two parts:

* the labelling `e`, injective, into the design's replay blocks (`e u < dd.part.blocks.length`), a premise on the
  held copies and the registered circuit alone, since the session's design does not read the statement
  (`SpotFlock.two`);
* the reading: a replay block correct at a witness leaves its unit bound in the witness's transcript under `readX`,
  correct and opened by the commitment at each of its committed wires.

`unitsWired_of` is the same at what the session's verifier reads (`UnitsWired`).
-/

@[expose] public section

namespace Verity.SampledTwo

open Claims

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

/-- **The units are the replay blocks, from a labelling into the design's replay blocks and the reading.** -/
theorem unitsAreReplay_of {V : Vocab d α} {θ : ByteArray} {F : Array ByteArray} {c : ByteArray}
    {H : Claims.Msg → Claims.Msg} {Cm : Claims.Sampling.CommitmentScheme Bool} {s : Claims.Sampling.AuditStatement}
    {Pt : Claims.Sampling.Partition} {dd : Claims.Sampling.Design.Two}
    {readX : Claims.Sampling.Wit → Verity.Circuit.Trace (V.circuitOf θ F)} (e : Fin (V.units θ F) ↪ ℕ)
    (hlen : ∀ u, e u < dd.part.blocks.length)
    (hread : ∀ w u, ReplayCorrect H Cm s Pt dd w (e u) →
      u ∉ Accounting.unbound (V.partsOf θ F) (V.Opens θ F c) (readX w)) :
    UnitsAreReplay V θ F c H Cm s Pt dd readX := by
  refine ⟨e, fun w u hu => ?_⟩
  simp only [wrongReplay, Finset.mem_filter, Finset.mem_range]
  exact ⟨hlen u, fun hc => hread w u hc hu⟩

/-- **`UnitsWired` from a labelling into the session's design and the reading**, at what the session's verifier
reads of `x`. -/
theorem unitsWired_of {V : Vocab d α} {A : Live₂} {x : Msg} {θ : ByteArray} {F : Array ByteArray} {c : ByteArray}
    {readX : Claims.Sampling.Wit → Verity.Circuit.Trace (V.circuitOf θ F)} (e : Fin (V.units θ F) ↪ ℕ)
    (hlen : ∀ s dd, A.L.two s = some dd → ∀ u, e u < dd.part.blocks.length)
    (hread : ∀ s Pt dd, A.L.parse x = some s → A.L.partition s = some Pt → A.L.two s = some dd →
      ∀ w u, ReplayCorrect FlockDeploy.sha512 A.L.commit s Pt dd w (e u) →
        u ∉ Accounting.unbound (V.partsOf θ F) (V.Opens θ F c) (readX w)) :
    UnitsWired V A x θ F c readX :=
  fun s Pt dd hs hPt hdd => unitsAreReplay_of e (hlen s dd hdd) (hread s Pt dd hs hPt hdd)

end Verity.SampledTwo
