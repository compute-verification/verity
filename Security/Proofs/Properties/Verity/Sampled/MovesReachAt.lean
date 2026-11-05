module

public import Proofs.Properties.Verity.Sampled.MovesLayout
public import Proofs.Properties.Verity.Sampled.SampleNamed

/-!
# `MovesReach` at the deployment

`couplingTarget_assembled_src`'s `hM` at the deployment's constructions (`movesReachAt`): the readings and the decoders
at the session's layout (`readingAt`, `decodersAt`), the sample-coupled coin map under `fieldEnc` at the family v2's
parameters name (`labelAt`), the digest the strategy's accepted plays bind (`afterAt`) and the session's bytes as
`MAX_COINS` fresh frames (`srcCoinsAt`), the deployed session's sample (`deployedLb`).

At these, `movesReach_layout`'s premises on the sample close: its coins fit `MAX_COINS` frames, and `srcCoinsAt` reads
the session's bytes (`bytesOf_srcCoinsAt`) wherever the session reads `MAX_COINS` frames' bytes (`hL`, which
`couplingTarget_assembled_src` takes too); and v2 reads the sample's coins of that family and digest wherever the
strategy's accepted plays bind the commitment to one list of digests (`sampleNamed_layout`, at `OpensOnce`). What is
left is `movesReachAt`'s premises, each named there.
-/

@[expose] public section

namespace Verity.Sampled

open Verity.Game

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

/-- **The sample's coins at `m` frames read the session's bytes** where the session reads `m` frames' bytes
(`bytesOf_srcCoins`): `movesReach_layout`'s `hsrc` at `srcCoinsAt`. -/
theorem bytesOf_srcCoinsAt {m : ℕ} {A : Live} (hL : A.Lb 0 = m * 64) (ω : A.Ω) :
    _root_.Properties.SampledProofs.bytesOf (A.src ω 0) =
      _root_.Properties.SampledProofs.bytesOf (coinBytes m 64 (srcCoinsAt m A ω) 0) := by
  rw [srcCoinsAt, dite_eq_left hL]
  exact bytesOf_srcCoins A hL ω

/-- **`MovesReach` at the deployment**: `couplingTarget_assembled_src`'s `hM` at `fieldEnc`, the readings at the
session's layout with the record each commitment's plays reveal and its digest (`record`, `recordDigest`), the decoders
at that layout (`reg`), the family `labelAt θ`, the digest `afterAt` and the sample's coins `srcCoinsAt MAX_COINS`. It
holds wherever:

* `A` is the session v2 sets up (`hA`), v2's parameters name the vocabulary's digests (`hDg`) and the session reads
  `MAX_COINS` frames' bytes (`hL`), as `couplingTarget_assembled_src` takes them;
* v2's read shows every coin of the sample's family (`hR`, `DrawReadAll` at `coinsRead`), v2 holds the record's draw
  to its own (`hH`, `DrawHeld`), and `norm` fixes the session's draw files (`hfix`);
* each strategy's accepted plays bind the commitment to one list of digests (`hOnce`, `OpensOnce`);
* v2's order refusal passes at every accepted play (`hord`, `OrderHeld`);
* the device's log opens only the record's rounds at every accepted play whose log v2 read is the device's (`hO`,
  `OpenedRecorded`). -/
theorem movesReachAt {V : Vocab d α} {k : Verity.Registration.Composite} {θ : ByteArray} {F : Array ByteArray}
    {A : Live} (reg : Option Claims.Msg → A.Reg) (record : ByteArray → Exchange → Option Claims.Msg)
    (recordDigest : ByteArray → Claims.Msg) (hA : SetsUp V θ F A) (hDg : DigestsAt V θ)
    (hL : A.Lb 0 = _root_.SampledProofs.Challenger.MAX_COINS * 64)
    (hR : DrawReadAll k coinsRead) (hH : DrawHeld k) (hfix : ∀ S, Flock.norm (A.dj S) = A.dj S)
    (hOnce : ∀ c σ, OpensOnce V k θ F c σ)
    (hord : ∀ c (σ : Strategy (V.game θ F c)) (x : CoinTape (V.game θ F c) σ),
      V.Verdict k .sampled (CoinTape.run (V.game θ F c) σ x) → OrderHeld (CoinTape.run (V.game θ F c) σ x).sent)
    (hO : ∀ c (σ : Strategy (V.game θ F c)) (x : CoinTape (V.game θ F c) σ),
      V.Verdict k .sampled (CoinTape.run (V.game θ F c) σ x) →
      Binding.Logged Binding.readsV2 (CoinTape.run (V.game θ F c) σ x) →
      ∀ text Jr, sentRecord (CoinTape.run (V.game θ F c) σ x).sent = some text →
        Flock.JsonParse.parse text = .ok Jr → OpenedRecorded Jr (CoinTape.run (V.game θ F c) σ x).log)
    (c : ByteArray) (σ : Strategy (V.game θ F c)) :
    MovesReach V θ F c k (Binding.Logged Binding.readsV2) A (readingAt A.tables A.layout (record c) (recordDigest c)) σ
      (decodersAt reg A.layout)
      (sampleMap fieldEnc V θ F c A (readingAt A.tables A.layout (record c) (recordDigest c)) σ (labelAt θ)
        (afterAt V k θ F c σ) (srcCoinsAt _root_.SampledProofs.Challenger.MAX_COINS A))
      (linkEmpty fieldEnc (finish (V.signer θ) θ F c σ.1)) (roundEmpty fieldEnc (finish (V.signer θ) θ F c σ.1)) :=
  movesReach_layout (label := labelAt θ) (after := afterAt V k θ F c σ)
    (κ := srcCoinsAt _root_.SampledProofs.Challenger.MAX_COINS A) hA hDg hR hH hfix reg (record c) (recordDigest c)
    rfl rfl le_rfl (bytesOf_srcCoinsAt hL) rfl (sampleNamed_layout hA (hOnce c σ)) (hord c σ) (hO c σ)

end Verity.Sampled
