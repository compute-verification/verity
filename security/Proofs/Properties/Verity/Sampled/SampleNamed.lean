module

public import Proofs.Properties.Verity.Sampled.RecordDraw
public import Proofs.Properties.Composition

/-! v2 reads the sample's coins of one family and one digest. -/

@[expose] public section

namespace Verity.Sampled

open Verity.Game
open Properties.Challenger (Draws)

variable {ι : Type} {d : ResourceAccounting.Device ι} {α : Type} [Fintype α]

/-- **The sample's family at the parameters `θ`**: the challenger's family v2's parameters in `θ` name (`ParamsAt`,
`paramsOf`), or the empty string where they name none. -/
noncomputable def labelAt (θ : ByteArray) : String :=
  match paramsOf θ with
  | some p => (p.base.challenger.map (·.2)).getD ""
  | none => ""

theorem labelAt_eq {θ : ByteArray} {p : _root_.Spot.OneStageZk.Params} {key} {l : String}
    (hp : ParamsAt θ p) (hc : p.base.challenger = some (key, l)) : labelAt θ = l := by
  simp only [labelAt, paramsOf_eq hp, hc, Option.map_some, Option.getD_some]

theorem sample_label {k : Verity.Registration.Composite} {s : Verity.Sent} {a b e : ByteArray}
    {p : _root_.Spot.OneStageZk.Params} {c : _root_.Spot.OneStage.Commitment}
    {ev : _root_.Spot.OneStageZk.Evidence} {ξ : ByteArray} {dr : Draws}
    {p₀ : _root_.Spot.OneStageZk.Params} (hS : Verity.SameAccepts k s) (hcase : V2Case k s a b e)
    (hacc : V2Accepts (_root_.Core.Blobs.ofArray s.blobs) a b e p c ev ξ dr) (hθ : ParamsAt s.params p₀) :
    dr.label = labelAt s.params := by
  obtain ⟨hp, -, -, -, hdr, -⟩ := hacc
  obtain rfl : p = p₀ := Sum.inl.inj (params_of_case hS hcase hp hθ)
  obtain ⟨key, hch, -⟩ := source_of hdr
  exact (labelAt_eq hθ hch).symm

theorem sampleNamed_layout {V : Vocab d α} {k : Verity.Registration.Composite} {θ : ByteArray}
    {F : Array ByteArray} {c : ByteArray} {A : Live} (hA : SetsUp V θ F A) {σ : Strategy (V.game θ F c)}
    {after : ByteArray} :
    ∀ x : CoinTape (V.game θ F c) σ, V.Verdict k .sampled (CoinTape.run (V.game θ F c) σ x) →
      Binding.Logged Binding.readsV2 (CoinTape.run (V.game θ F c) σ x) →
      SampledDigest k (CoinTape.run (V.game θ F c) σ x).sent after →
        SampleNamed k (CoinTape.run (V.game θ F c) σ x).sent (labelAt θ) after := by
  intro x hv _ hd a b e p c' ev ξ dr hcase hacc
  obtain ⟨p₀, hp₀, -⟩ := hA.params
  have hpe : (CoinTape.run (V.game θ F c) σ x).sent.params = θ := (play_ends V θ F c σ x).1
  have hθ : ParamsAt (CoinTape.run (V.game θ F c) σ x).sent.params p₀ := by
    rw [hpe]
    exact hp₀
  exact ⟨(sample_label hv.2.1 hcase hacc hθ).trans (congrArg labelAt hpe), hd a b e p c' ev ξ dr hcase hacc⟩

end Verity.Sampled
