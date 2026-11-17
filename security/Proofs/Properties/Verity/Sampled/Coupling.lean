module

public import Proofs.Properties.Verity.Sampled.Messages

/-! SPOT's coupling: the target off the binding and the logged coupling. -/

@[expose] public section

namespace Verity.Sampled

open Verity.Game
open scoped ENNReal

variable {ι : Type} {d : ResourceAccounting.Device ι} {α : Type} [Fintype α]

/-- **The subprotocol's constructions with the messages' two-continuation finders at the readings `rd`** (`msgFinder`,
`recordFinder`): `K`'s coins, projection, read finder, forger and bound digests' finder. -/
noncomputable def Constructions.withMessages {V : Vocab d α} (K : Constructions V) (rd : Readings) :
    Constructions V :=
  { K with msgFinder := Verity.Sampled.msgFinder V rd, recordFinder := Verity.Sampled.recordFinder V rd,
           msg_calls := msgFinder_calls V rd, record_calls := recordFinder_calls V rd }

variable (V : Vocab d α) (k : Verity.Registration.Composite)

def LoggedBound (K : Constructions V) (P : Exchange → Prop) (θ : ByteArray) (F : Array ByteArray) : Prop :=
  ∀ (c : ByteArray) (σ : Game.Strategy (V.game θ F c)),
    Game.prob (fun o => V.Verdict k .sampled o ∧ ¬ P o) (V.game θ F c) σ ≤
      ENNReal.ofReal ((K.readFinder θ F c σ).Pr sha512Msg) + ENNReal.ofReal ((K.forger θ F c σ).Pr Verity.lmsVerify)

def DigestBound (K : Constructions V) (A : Live) (θ : ByteArray) (F : Array ByteArray)
    (Q : (c : ByteArray) → Game.Strategy (V.game θ F c) → Exchange → Prop) : Prop :=
  ∀ (c : ByteArray) (σ : Game.Strategy (V.game θ F c)),
    Game.prob (fun o => V.Verdict k .sampled o ∧ ¬ Q c σ o) (V.game θ F c) σ ≤
      ENNReal.ofReal (Real.sqrt ((K.bindFinder A θ F c σ).Pr sha512Msg))

def LoggedCoupling (rd : Readings) (K : Constructions V) (A : Live) (θ : ByteArray) (F : Array ByteArray)
    (P : (c : ByteArray) → Game.Strategy (V.game θ F c) → Exchange → Prop) : Prop :=
  ∀ (c : ByteArray) (σ : Game.Strategy (V.game θ F c)),
    ENNReal.ofReal (Claims.Pr fun x : CoinTape (V.game θ F c) σ =>
        V.Verdict k .sampled (CoinTape.run (V.game θ F c) σ x) ∧ NoMiss V θ F c (rd A θ F c) σ x ∧
          P c σ (CoinTape.run (V.game θ F c) σ x)) ≤
      FlockSoundness.Game.prob (fun o => FlockSoundness.Discharge.ZkHidden.LiveAcceptsZKP A.I A.own o)
          (K.live A θ F c σ).game (K.project A θ F c σ) +
        A.hidingTerm

theorem noMiss_le_of_logged (rd : Readings) (K : Constructions V) (A : Live) (P : Exchange → Prop) (θ : ByteArray)
    (F : Array ByteArray) (Q : (c : ByteArray) → Game.Strategy (V.game θ F c) → Exchange → Prop)
    (ha : LoggedBound V k K P θ F) (hq : DigestBound V k K A θ F Q)
    (hb : LoggedCoupling V k rd K A θ F fun c σ o => P o ∧ Q c σ o) (c : ByteArray)
    (σ : Game.Strategy (V.game θ F c)) :
    ENNReal.ofReal (Claims.Pr fun x : CoinTape (V.game θ F c) σ =>
        V.Verdict k .sampled (CoinTape.run (V.game θ F c) σ x) ∧ NoMiss V θ F c (rd A θ F c) σ x) ≤
      FlockSoundness.Game.prob (fun o => FlockSoundness.Discharge.ZkHidden.LiveAcceptsZKP A.I A.own o)
          (K.live A θ F c σ).game (K.project A θ F c σ) +
        (ENNReal.ofReal (Real.sqrt ((K.bindFinder A θ F c σ).Pr sha512Msg)) + A.hidingTerm +
          ENNReal.ofReal ((K.readFinder θ F c σ).Pr sha512Msg) +
          ENNReal.ofReal ((K.forger θ F c σ).Pr Verity.lmsVerify)) := by
  have hsplit : Claims.Pr (fun x : CoinTape (V.game θ F c) σ =>
        V.Verdict k .sampled (CoinTape.run (V.game θ F c) σ x) ∧ NoMiss V θ F c (rd A θ F c) σ x) ≤
      Claims.Pr (fun x : CoinTape (V.game θ F c) σ =>
          V.Verdict k .sampled (CoinTape.run (V.game θ F c) σ x) ∧ NoMiss V θ F c (rd A θ F c) σ x ∧
            (P (CoinTape.run (V.game θ F c) σ x) ∧ Q c σ (CoinTape.run (V.game θ F c) σ x))) +
        (Claims.Pr (fun x : CoinTape (V.game θ F c) σ =>
            V.Verdict k .sampled (CoinTape.run (V.game θ F c) σ x) ∧ ¬ P (CoinTape.run (V.game θ F c) σ x)) +
          Claims.Pr (fun x : CoinTape (V.game θ F c) σ =>
            V.Verdict k .sampled (CoinTape.run (V.game θ F c) σ x) ∧
              ¬ Q c σ (CoinTape.run (V.game θ F c) σ x))) := by
    refine (Claims.Pr_mono fun x hx => ?_).trans ((pr_or_le _ _).trans (add_le_add le_rfl (pr_or_le _ _)))
    by_cases hp : P (CoinTape.run (V.game θ F c) σ x)
    · by_cases hQ : Q c σ (CoinTape.run (V.game θ F c) σ x)
      · exact Or.inl ⟨hx.1, hx.2, hp, hQ⟩
      · exact Or.inr (Or.inr ⟨hx.1, hQ⟩)
    · exact Or.inr (Or.inl ⟨hx.1, hp⟩)
  have hnot := ha c σ
  rw [prob_eq_ofReal_pr] at hnot
  have hnq := hq c σ
  rw [prob_eq_ofReal_pr] at hnq
  refine ((ENNReal.ofReal_le_ofReal hsplit).trans
    (ENNReal.ofReal_add_le.trans (add_le_add le_rfl ENNReal.ofReal_add_le))).trans ?_
  refine (add_le_add (hb c σ) (add_le_add hnot hnq)).trans_eq ?_
  ring

theorem couplingTarget_of_logged (rd : Readings) (K : Constructions V) (A : Live) (P : Exchange → Prop)
    (θ : ByteArray) (F : Array ByteArray) (Q : (c : ByteArray) → Game.Strategy (V.game θ F c) → Exchange → Prop)
    (ha : LoggedBound V k K P θ F) (hq : DigestBound V k K A θ F Q)
    (hb : LoggedCoupling V k rd K A θ F fun c σ o => P o ∧ Q c σ o) :
    CouplingTarget V k (K.withMessages rd) A θ F :=
  couplingTarget_of_noMiss V k rd (K.withMessages rd) rfl rfl A θ F
    (noMiss_le_of_logged V k rd (K.withMessages rd) A P θ F Q ha hq hb)

end Verity.Sampled
