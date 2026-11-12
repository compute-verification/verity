module

public import Proofs.Properties.Verity.Sampled.MovesLayout
public import Proofs.Properties.Verity.Sampled.SampleNamed

/-!
# `MovesReach` at the deployment

`couplingTarget_assembled_src`'s `hM` at the deployment's constructions (`movesReachAt`), at the coupling's cut (v2's
read of the log is the device's, and v2 samples at the strategy's digest `afterAt`, `SampledDigest`): the readings and
the decoders at the session's layout (`readingAt`, `decodersAt`), the sample-coupled coin map under `fieldEnc` at the
family v2's parameters name (`labelAt`), the digest `afterAt` and the session's bytes as `MAX_COINS` fresh frames
(`srcCoinsAt`), the deployed session's sample (`deployedLb`).

At these, `movesReach_layout`'s premises on the sample close: its coins fit `MAX_COINS` frames, and `srcCoinsAt` reads
the session's bytes (`bytesOf_srcCoinsAt`) wherever the session reads `MAX_COINS` frames' bytes (`hL`, which
`couplingTarget_assembled_src` takes too); and v2 reads the sample's coins of that family and digest at the cut
(`sampleNamed_layout`). What is left is `movesReachAt`'s premises, each named there.

The coupling at the deployment (`couplingTargetAt`) is `couplingTarget_assembled_src` there, at the deployed device's
history (`Device.dev`): the device printed each play's log (`Device.printsLog_dev`), it signed what each play's
signatures are on wherever the vocabulary's signer is the deployment's (`Device.signsPlay_dev`), and the session's
writer, `wrZP`, keeps an accepted transcript's view (`writesView_wrZP`). What is left is that signer, the stops' term
(`hε`) and `movesReachAt`'s premises.
-/

@[expose] public section

namespace Verity.Sampled

open Verity.Game

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

local notation "𝔽" => FlockLevel3.GF128

/-- **The sample's coins at `m` frames read the session's bytes** where the session reads `m` frames' bytes
(`bytesOf_srcCoins`): `movesReach_layout`'s `hsrc` at `srcCoinsAt`. -/
theorem bytesOf_srcCoinsAt {m : ℕ} {A : Live} (hL : A.Lb 0 = m * 64) (ω : A.Ω) :
    _root_.Properties.Spot.bytesOf (A.src ω 0) =
      _root_.Properties.Spot.bytesOf (coinBytes m 64 (srcCoinsAt m A ω) 0) := by
  rw [srcCoinsAt, dite_eq_left hL]
  exact bytesOf_srcCoins A hL ω

/-- **`MovesReach` at the deployment**: `couplingTarget_assembled_src`'s `hM` at `fieldEnc`, at the coupling's cut,
the readings at the session's layout with the record each commitment's plays reveal and its digest (`record`,
`recordDigest`), the decoders at that layout (`reg`), the family `labelAt θ`, the digest `afterAt` and the sample's
coins `srcCoinsAt MAX_COINS`. It holds wherever:

* `A` is the session v2 sets up (`hA`), v2's parameters name the vocabulary's digests (`hDg`) and the session reads
  `MAX_COINS` frames' bytes (`hL`), as `couplingTarget_assembled_src` takes them;
* the device's log opens only the record's rounds at every accepted play whose log v2 read is the device's (`hO`,
  `OpenedRecorded`). -/
theorem movesReachAt {V : Vocab d α} {k : Verity.Registration.Composite} {θ : ByteArray} {F : Array ByteArray}
    {A : Live} (reg : Option Claims.Msg → A.Reg) (record : ByteArray → Exchange → Option Claims.Msg)
    (recordDigest : ByteArray → Claims.Msg) (hA : SetsUp V θ F A) (hDg : DigestsAt V θ)
    (hL : A.Lb 0 = _root_.Spot.Challenger.MAX_COINS * 64)
    (hO : ∀ c (σ : Strategy (V.game θ F c)) (x : CoinTape (V.game θ F c) σ),
      V.Verdict k .sampled (CoinTape.run (V.game θ F c) σ x) →
      Binding.Logged Binding.readsV2 (CoinTape.run (V.game θ F c) σ x) →
      ∀ text Jr, sentRecord (CoinTape.run (V.game θ F c) σ x).sent = some text →
        Flock.JsonParse.parse text = .ok Jr → OpenedRecorded Jr (CoinTape.run (V.game θ F c) σ x).log)
    (c : ByteArray) (σ : Strategy (V.game θ F c)) :
    MovesReach V θ F c k (fun o => Binding.Logged Binding.readsV2 o ∧ SampledDigest k o.sent (afterAt V k θ F c σ)) A
      (readingAt A.tables A.layout (record c) (recordDigest c)) σ (decodersAt reg A.layout)
      (sampleMap fieldEnc V θ F c A (readingAt A.tables A.layout (record c) (recordDigest c)) σ (labelAt θ)
        (afterAt V k θ F c σ) (srcCoinsAt _root_.Spot.Challenger.MAX_COINS A))
      (linkEmpty fieldEnc (finish (V.signer θ) θ F c σ.1)) (roundEmpty fieldEnc (finish (V.signer θ) θ F c σ.1)) :=
  movesReach_layout (label := labelAt θ) (after := afterAt V k θ F c σ)
    (κ := srcCoinsAt _root_.Spot.Challenger.MAX_COINS A) hA hDg reg (record c) (recordDigest c)
    rfl rfl le_rfl (bytesOf_srcCoinsAt hL) rfl (sampleNamed_layout hA) (hO c σ)

/-- **The coupling at the deployment**: `couplingTarget_assembled_src` at the deployed device's history
(`Device.dev`) and `movesReachAt`'s readings, decoders, family, digest and coins, with its `hM` by `movesReachAt`.
It holds wherever the vocabulary's signer at `θ` is the deployment's (`hD`), the stops' term is at most the hiding term
(`hε`), and `movesReachAt`'s premises hold. -/
theorem couplingTargetAt {V : Vocab d α} {k : Verity.Registration.Composite} (K : Constructions V)
    (P : Device.Deployment) {θ : ByteArray} {F : Array ByteArray} {A : Live}
    (reg : (A : Live) → Option Claims.Msg → A.Reg)
    (record : Live → ByteArray → Array ByteArray → ByteArray → Exchange → Option Claims.Msg)
    (recordDigest : Live → ByteArray → Array ByteArray → ByteArray → Claims.Msg)
    (hD : V.signer θ = P.signer V.families θ)
    (hε : ∀ c σ, couplesErr V θ F c A (readingAt A.tables A.layout (record A θ F c) (recordDigest A θ F c)) σ
      (decodersAt (reg A) A.layout)
      (sampleMap fieldEnc V θ F c A (readingAt A.tables A.layout (record A θ F c) (recordDigest A θ F c)) σ
        (labelAt θ) (afterAt V k θ F c σ) (srcCoinsAt _root_.Spot.Challenger.MAX_COINS A))
      (fun q _ s => openGap (finish (V.signer θ) θ F c σ.1) q.g' (roundBits fieldEnc : (Fin s.n → 𝔽) → List Bool) +
        openEmpty (finish (V.signer θ) θ F c σ.1) q.g' (roundBits fieldEnc : (Fin s.n → 𝔽) → List Bool))
      (fun q => openGap (finish (V.signer θ) θ F c σ.1) q.g' (linkBits fieldEnc : (Fin 2 → Fin A.mP → 𝔽) → List Bool) +
        openEmpty (finish (V.signer θ) θ F c σ.1) q.g' (linkBits fieldEnc : (Fin 2 → Fin A.mP → 𝔽) → List Bool)) ≤
        A.hidingTerm)
    (hA : SetsUp V θ F A) (hDg : DigestsAt V θ) (hL : A.Lb 0 = _root_.Spot.Challenger.MAX_COINS * 64)
    (hO : ∀ c (σ : Strategy (V.game θ F c)) (x : CoinTape (V.game θ F c) σ),
      V.Verdict k .sampled (CoinTape.run (V.game θ F c) σ x) →
      Binding.Logged Binding.readsV2 (CoinTape.run (V.game θ F c) σ x) →
      ∀ text Jr, sentRecord (CoinTape.run (V.game θ F c) σ x).sent = some text →
        Flock.JsonParse.parse text = .ok Jr → OpenedRecorded Jr (CoinTape.run (V.game θ F c) σ x).log) :
    CouplingTarget V k (K.assembled fieldEnc (Device.dev (P.setupOf V.families)) k
      (fun A θ F c => readingAt A.tables A.layout (record A θ F c) (recordDigest A θ F c))
      (fun A _ _ _ _ => decodersAt (reg A) A.layout) (fun θ _ _ _ => labelAt θ) (afterAt V k)
      (srcCoinsAt _root_.Spot.Challenger.MAX_COINS)) A θ F :=
  couplingTarget_assembled_src fieldEnc V k K (Device.dev (P.setupOf V.families))
    (fun A θ F c => readingAt A.tables A.layout (record A θ F c) (recordDigest A θ F c))
    (fun A _ _ _ _ => decodersAt (reg A) A.layout) (fun θ _ _ _ => labelAt θ)
    _root_.Spot.Challenger.MAX_COINS A θ F (fun o _ => Device.printsLog_dev _ o)
    (fun c => Device.signsPlay_dev V P θ F c hD) hL hε hA hDg (writesView_wrZP A rfl)
    (movesReachAt (reg A) (record A θ F) (recordDigest A θ F) hA hDg hL hO)

end Verity.Sampled
