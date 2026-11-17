module

public import Proofs.Properties.Verity.Sampled.MovesLayout
public import Proofs.Properties.Verity.Sampled.SampleNamed

/-! `MovesReach` at the deployment. -/

@[expose] public section

namespace Verity.Sampled

open Verity.Game

variable {ι : Type} {d : ResourceAccounting.Device ι} {α : Type} [Fintype α]

local notation "𝔽" => FlockLevel3.GF128

theorem bytesOf_srcCoinsAt {m : ℕ} {A : Live} (hL : A.Lb 0 = m * 64) (ω : A.Ω) :
    _root_.Properties.Spot.bytesOf (A.src ω 0) =
      _root_.Properties.Spot.bytesOf (coinBytes m 64 (srcCoinsAt m A ω) 0) := by
  rw [srcCoinsAt, dite_eq_left hL]
  exact bytesOf_srcCoins A hL ω

theorem movesReachAt {V : Vocab d α} {T : PinnedTables} {k : Verity.Registration.Composite} {θ : ByteArray}
    {F : Array ByteArray} {A : Live} (reg : Option Claims.Msg → A.Reg)
    (record : ByteArray → Exchange → Option Claims.Msg)
    (recordDigest : ByteArray → Claims.Msg) (hA : SetsUp V θ F A) (hDg : DigestsAt V T θ)
    (hL : A.Lb 0 = _root_.Spot.Challenger.MAX_COINS * 64)
    (hO : ∀ c (σ : Strategy (V.game θ F c)) (x : CoinTape (V.game θ F c) σ),
      V.Verdict k .sampled (CoinTape.run (V.game θ F c) σ x) →
      Binding.Logged Binding.readsV2 (CoinTape.run (V.game θ F c) σ x) →
      ∀ text Jr, sentRecord (CoinTape.run (V.game θ F c) σ x).sent = some text →
        Flock.JsonParse.parse text = .ok Jr → OpenedRecorded Jr (CoinTape.run (V.game θ F c) σ x).log)
    (c : ByteArray) (σ : Strategy (V.game θ F c)) :
    MovesReach V θ F c k
      (fun o =>
        (Binding.Logged Binding.readsV2 o ∧ TablesSent T o.sent) ∧ SampledDigest k o.sent (afterAt V k θ F c σ)) A
      (readingAt A.tables A.layout (record c) (recordDigest c)) σ (decodersAt reg A.layout)
      (sampleMap fieldEnc V θ F c A (readingAt A.tables A.layout (record c) (recordDigest c)) σ (labelAt θ)
        (afterAt V k θ F c σ) (srcCoinsAt _root_.Spot.Challenger.MAX_COINS A))
      (linkEmpty fieldEnc (finish (V.signer θ) θ F c σ.1)) (roundEmpty fieldEnc (finish (V.signer θ) θ F c σ.1)) :=
  movesReach_layout (label := labelAt θ) (after := afterAt V k θ F c σ)
    (κ := srcCoinsAt _root_.Spot.Challenger.MAX_COINS A) hA hDg reg (record c) (recordDigest c)
    rfl rfl le_rfl (bytesOf_srcCoinsAt hL) rfl (sampleNamed_layout hA) (hO c σ)

end Verity.Sampled
