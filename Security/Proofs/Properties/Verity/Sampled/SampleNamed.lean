module

public import Proofs.Properties.Verity.Sampled.RecordDraw
public import Proofs.Properties.Composition

/-!
# v2 reads the sample's coins of one family and one digest

The record's sample (`recordDraw_firstStop`) and the sample-coupled coin map (`sampleMap`) take a family `label` and a
digest `after` fixed by the parameters, the committed files, the commitment and the strategy, before any coin, and v2's
sample in every play the sampled verdict accepts at them (`SampleNamed`). v2 samples under the challenger its parameters
name, at the coins of its family released after the registration's digest (`source_of`):

* **The family** (`draw_label`, `labelAt`): the parameters' challenger family, a function of the parameters `θ` alone,
  since v2's case reads the parameters `θ` names for it (`params_of_case`).
* **The digest**: sampled proofs' entry in the components' digests that the prover's commitment binds (`draw_after`),
  fixed only up to the cut at the commitment, where the play's digests are the most likely ones (`SampledDigest`,
  `afterAt`, bounded by a SHA-512 collision on `bind`'s messages).

So `SampleNamed` holds at the family `labelAt θ` and the digest `after` on every accepted play that meets the clause
that v2 samples at `after` (`sampleNamed_layout`, the form `movesReach_layout`'s `hN` takes on the coupling's cut, where
`after` is the strategy's `afterAt`). The logged reading isn't needed.
-/

@[expose] public section

namespace Verity.Sampled

open Verity.Game
open Properties.Challenger (Draws)

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

/-! ## The family -/

open Classical in
/-- **The sample's family at the parameters `θ`**: the challenger's family v2's parameters in `θ` name (`ParamsAt`),
or the empty string where they name none. -/
noncomputable def labelAt (θ : ByteArray) : String :=
  if h : ∃ l p key, ParamsAt θ p ∧ p.base.challenger = some (key, l) then h.choose else ""

theorem labelAt_eq {θ : ByteArray} {p : _root_.Spot.OneStageZk.Params} {key} {l : String}
    (hp : ParamsAt θ p) (hc : p.base.challenger = some (key, l)) : labelAt θ = l := by
  have h : ∃ l p key, ParamsAt θ p ∧ p.base.challenger = some (key, l) := ⟨l, p, key, hp, hc⟩
  rw [labelAt, dite_eq_left h]
  obtain ⟨p', key', hp', hc'⟩ := h.choose_spec
  rw [ParamsAt.unique hp' hp, hc] at hc'
  exact (Prod.mk.inj (Option.some.inj hc')).2.symm

/-- **v2 samples at the parameters' family**: where composite `k` accepted what the prover sent and v2's case in it
accepts at the sample `dr`, `dr`'s family is the one v2's parameters in what was sent name. -/
theorem draw_label {k : Verity.Registration.Composite} {s : Verity.Sent} {a b e : ByteArray}
    {p : _root_.Spot.OneStageZk.Params} {c : _root_.Spot.OneStage.Commitment}
    {ev : _root_.Spot.OneStageZk.Evidence} {ξ : ByteArray} {dr : Draws}
    {p₀ : _root_.Spot.OneStageZk.Params} (hS : Verity.SameAccepts k s) (hcase : V2Case k s a b e)
    (hacc : V2Accepts (_root_.Core.Blobs.ofArray s.blobs) a b e p c ev ξ dr) (hθ : ParamsAt s.params p₀) :
    dr.label = labelAt s.params := by
  obtain ⟨hp, -, -, -, hdr, -⟩ := hacc
  obtain rfl : p = p₀ := Sum.inl.inj (params_of_case hS hcase hp hθ)
  obtain ⟨key, hch, -⟩ := source_of hdr
  exact (labelAt_eq hθ hch).symm

/-! ## At the cut -/

/-- **`movesReach_layout`'s `hN`** at the family `labelAt θ` and a digest `after`, on the plays that meet the clause
that v2 samples at `after` (`SampledDigest`, the cut's at `afterAt`); it reads nothing of the logged reading. -/
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
  exact ⟨(draw_label hv.2.1 hcase hacc hθ).trans (congrArg labelAt hpe), hd a b e p c' ev ξ dr hcase hacc⟩

end Verity.Sampled
