module

public import Proofs.Properties.Verity.Sampled.Messages

/-!
# Sampled proofs' coupling: the target off the binding and the logged coupling

`CouplingTarget` at the leg's constructions with the messages' two-continuation finders (`Constructions.withMessages`,
whose finders are `msgFinder` and `recordFinder`, so `couplingTarget_of_noMiss` applies by `rfl`), from three bounds on
the plays the sampled verdict accepts, cut by a property `P` of the play, the one the check's read of the device's log
is held to (the binding's `Logged`: the log the check read is the device's), and by a property `Q` of the play at each
commitment and strategy, the one the digest v2's sample answers is held to (the strategy's own, `afterAt`):

* **(a), the binding** (`LoggedBound`): an accepted play `P` doesn't hold of is at most as likely as the read finder's
  collision plus the forger's forgery. It is the binding's bound (`Binding.logged_le_withBinding`, from its pointwise
  `Binds` and the deployed LMS parameters of the parameters' key), taken here as a hypothesis: whether the game's device
  signatures let `Binds` hold is open.
* **(a'), the digest** (`DigestBound`): an accepted play `Q` doesn't hold of is at most as likely as the square root of
  the bound digests' two-continuation finder's success (`Constructions.bindFinder`; `digestBound_of` at the digest's
  clause).
* **(b), the logged coupling** (`LoggedCoupling`): an accepted play that the cut holds of, on a tape that misses at no
  round and not at the record (`NoMiss`), is at most as likely as the live audit's acceptance at the projected
  strategy, plus the hiding term. Its pieces are the sample, the rounds and the messages at their cuts, at the projection
  the constructions give.

The three meet by the union bound on the tape (`noMiss_le_of_logged`): an accepted play that misses nowhere either is
one `P` and `Q` hold of, or is one `P` doesn't hold of, or one `Q` doesn't.
-/

@[expose] public section

namespace Verity.Sampled

open Verity.Game
open scoped ENNReal

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

/-- **The leg's constructions with the messages' two-continuation finders at the readings `rd`** (`msgFinder`,
`recordFinder`): `K`'s coins, projection, read finder, forger and bound digests' finder. -/
noncomputable def Constructions.withMessages {V : Vocab d α} (K : Constructions V) (rd : Readings) :
    Constructions V :=
  { K with msgFinder := Verity.Sampled.msgFinder V rd, recordFinder := Verity.Sampled.recordFinder V rd,
           msg_calls := msgFinder_calls V rd, record_calls := recordFinder_calls V rd }

variable (V : Vocab d α) (k : Verity.Registration.Composite)

/-- **(a), the binding, at `P`**: at the parameters `θ` and the committed files `F`, a play of the period's game whose
sampled verdict at `k` accepts and that `P` doesn't hold of is at most as likely as the read finder's collision plus the
forger's forgery. -/
def LoggedBound (K : Constructions V) (P : Exchange → Prop) (θ : ByteArray) (F : Array ByteArray) : Prop :=
  ∀ (c : ByteArray) (σ : Game.Strategy (V.game θ F c)),
    Game.prob (fun o => V.Verdict k .sampled o ∧ ¬ P o) (V.game θ F c) σ ≤
      ENNReal.ofReal ((K.readFinder θ F c σ).Pr sha512Msg) + ENNReal.ofReal ((K.forger θ F c σ).Pr Verity.lmsVerify)

/-- **(a'), the digest, at `Q`**: at the session `A`, the parameters `θ` and the committed files `F`, a play of the
period's game whose sampled verdict at `k` accepts and that `Q` at its commitment and strategy doesn't hold of is at
most as likely as the square root of the bound digests' two-continuation finder's success. -/
def DigestBound (K : Constructions V) (A : Live) (θ : ByteArray) (F : Array ByteArray)
    (Q : (c : ByteArray) → Game.Strategy (V.game θ F c) → Exchange → Prop) : Prop :=
  ∀ (c : ByteArray) (σ : Game.Strategy (V.game θ F c)),
    Game.prob (fun o => V.Verdict k .sampled o ∧ ¬ Q c σ o) (V.game θ F c) σ ≤
      ENNReal.ofReal (Real.sqrt ((K.bindFinder A θ F c σ).Pr sha512Msg))

/-- **(b), the logged coupling, at the cut `P`**: at the session `A`, the parameters `θ` and the committed files `F`, a
play whose sampled verdict at `k` accepts and that `P` at its commitment and strategy holds of, on a tape that misses at
no round and not at the record at the readings `rd` (`NoMiss`), is at most as likely as the live audit's plays, at the
session widened by the projection's coins and the strategy projected there, that flock-verify `--zk` accepts at `own`,
plus the hiding term. -/
def LoggedCoupling (rd : Readings) (K : Constructions V) (A : Live) (θ : ByteArray) (F : Array ByteArray)
    (P : (c : ByteArray) → Game.Strategy (V.game θ F c) → Exchange → Prop) : Prop :=
  ∀ (c : ByteArray) (σ : Game.Strategy (V.game θ F c)),
    ENNReal.ofReal (Claims.Pr fun x : CoinTape (V.game θ F c) σ =>
        V.Verdict k .sampled (CoinTape.run (V.game θ F c) σ x) ∧ NoMiss V θ F c (rd A θ F c) σ x ∧
          P c σ (CoinTape.run (V.game θ F c) σ x)) ≤
      FlockSoundness.Game.prob (fun o => FlockSoundness.Discharge.ZkHidden.LiveAcceptsZKP A.I A.own o)
          (K.live A θ F c σ).game (K.project A θ F c σ) +
        A.hidingTerm

/-- **The accepted plays that miss nowhere, off the binding, the digest and the logged coupling**: each is one `P` and
`Q` hold of, bounded by (b), or one `P` doesn't hold of, bounded by (a), or one `Q` doesn't, bounded by (a'). This is
the hypothesis of `couplingTarget_of_noMiss`. -/
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

/-- **The coupling, off the binding, the digest and the logged coupling**: at the constructions with the messages'
finders at the readings `rd`, `CouplingTarget` holds at the session `A` wherever (a) holds at one property `P` of the
play, (a') at one property `Q` at each commitment and strategy, and (b) at the two. -/
theorem couplingTarget_of_logged (rd : Readings) (K : Constructions V) (A : Live) (P : Exchange → Prop)
    (θ : ByteArray) (F : Array ByteArray) (Q : (c : ByteArray) → Game.Strategy (V.game θ F c) → Exchange → Prop)
    (ha : LoggedBound V k K P θ F) (hq : DigestBound V k K A θ F Q)
    (hb : LoggedCoupling V k rd K A θ F fun c σ o => P o ∧ Q c σ o) :
    CouplingTarget V k (K.withMessages rd) A θ F :=
  couplingTarget_of_noMiss V k rd (K.withMessages rd) rfl rfl A θ F
    (noMiss_le_of_logged V k rd (K.withMessages rd) A P θ F Q ha hq hb)

end Verity.Sampled
