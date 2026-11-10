module

public import Proofs.Properties.Verity.Sampled.Messages

/-!
# Sampled proofs' coupling: the target off the binding and the logged coupling

`CouplingTarget` at the leg's constructions with the messages' two-continuation finders (`Constructions.withMessages`,
whose finders are `msgFinder` and `recordFinder`, so `couplingTarget_of_noMiss` applies by `rfl`), from two bounds on
the plays the sampled verdict accepts, cut by a property `P` of the play, the one the check's read of the device's log
is held to (the binding's `Logged`: the log the check read is the device's):

* **(a), the binding** (`LoggedBound`): an accepted play `P` doesn't hold of is at most as likely as the read finder's
  collision plus the forger's forgery. It is the binding's bound (`Binding.logged_le_withBinding`, from its pointwise
  `Binds` and the deployed LMS parameters of the parameters' key), taken here as a hypothesis: whether the game's device
  signatures let `Binds` hold is open.
* **(b), the logged coupling** (`LoggedCoupling`): an accepted play that `P` holds of, on a tape that misses at no
  round and not at the record (`NoMiss`), is at most as likely as the live audit's acceptance at the projected
  strategy, plus the hiding term. Its pieces are the sample, the rounds and the messages at their cuts, at the projection
  the constructions give.

The two meet by the union bound on the tape (`noMiss_le_of_logged`): an accepted play that misses nowhere either is
one `P` holds of or is one it doesn't.
-/

@[expose] public section

namespace Verity.Sampled

open Verity.Game
open scoped ENNReal

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

/-- **The leg's constructions with the messages' two-continuation finders at the readings `rd`** (`msgFinder`,
`recordFinder`): `K`'s coins, projection, read finder and forger. -/
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

/-- **(b), the logged coupling, at `P`**: at the session `A`, the parameters `θ` and the committed files `F`, a play
whose sampled verdict at `k` accepts and that `P` holds of, on a tape that misses at no round and not at the record at
the readings `rd` (`NoMiss`), is at most as likely as the live audit's plays, at the session widened by the projection's
coins and the strategy projected there, that flock-verify `--zk` accepts at `own`, plus the hiding term. -/
def LoggedCoupling (rd : Readings) (K : Constructions V) (A : Live) (P : Exchange → Prop) (θ : ByteArray)
    (F : Array ByteArray) : Prop :=
  ∀ (c : ByteArray) (σ : Game.Strategy (V.game θ F c)),
    ENNReal.ofReal (Claims.Pr fun x : CoinTape (V.game θ F c) σ =>
        V.Verdict k .sampled (CoinTape.run (V.game θ F c) σ x) ∧ NoMiss V θ F c (rd A θ F c) σ x ∧
          P (CoinTape.run (V.game θ F c) σ x)) ≤
      FlockSoundness.Game.prob (fun o => FlockSoundness.Discharge.ZkHidden.LiveAcceptsZKP A.I A.own o)
          (K.live A θ F c σ).game (K.project A θ F c σ) +
        A.hidingTerm

/-- **The accepted plays that miss nowhere, off the binding and the logged coupling**: each is one `P` holds of, bounded
by (b), or one it doesn't, bounded by (a). This is the hypothesis of `couplingTarget_of_noMiss`. -/
theorem noMiss_le_of_logged (rd : Readings) (K : Constructions V) (A : Live) (P : Exchange → Prop) (θ : ByteArray)
    (F : Array ByteArray) (ha : LoggedBound V k K P θ F) (hb : LoggedCoupling V k rd K A P θ F) (c : ByteArray)
    (σ : Game.Strategy (V.game θ F c)) :
    ENNReal.ofReal (Claims.Pr fun x : CoinTape (V.game θ F c) σ =>
        V.Verdict k .sampled (CoinTape.run (V.game θ F c) σ x) ∧ NoMiss V θ F c (rd A θ F c) σ x) ≤
      FlockSoundness.Game.prob (fun o => FlockSoundness.Discharge.ZkHidden.LiveAcceptsZKP A.I A.own o)
          (K.live A θ F c σ).game (K.project A θ F c σ) +
        (A.hidingTerm + ENNReal.ofReal ((K.readFinder θ F c σ).Pr sha512Msg) +
          ENNReal.ofReal ((K.forger θ F c σ).Pr Verity.lmsVerify)) := by
  have hsplit : Claims.Pr (fun x : CoinTape (V.game θ F c) σ =>
        V.Verdict k .sampled (CoinTape.run (V.game θ F c) σ x) ∧ NoMiss V θ F c (rd A θ F c) σ x) ≤
      Claims.Pr (fun x : CoinTape (V.game θ F c) σ =>
          V.Verdict k .sampled (CoinTape.run (V.game θ F c) σ x) ∧ NoMiss V θ F c (rd A θ F c) σ x ∧
            P (CoinTape.run (V.game θ F c) σ x)) +
        Claims.Pr (fun x : CoinTape (V.game θ F c) σ =>
          V.Verdict k .sampled (CoinTape.run (V.game θ F c) σ x) ∧ ¬ P (CoinTape.run (V.game θ F c) σ x)) := by
    refine (Claims.Pr_mono fun x hx => ?_).trans (pr_or_le _ _)
    by_cases hp : P (CoinTape.run (V.game θ F c) σ x)
    · exact Or.inl ⟨hx.1, hx.2, hp⟩
    · exact Or.inr ⟨hx.1, hp⟩
  have hnot := ha c σ
  rw [prob_eq_ofReal_pr] at hnot
  refine ((ENNReal.ofReal_le_ofReal hsplit).trans ENNReal.ofReal_add_le).trans ?_
  refine (add_le_add (hb c σ) hnot).trans_eq ?_
  ring

/-- **The coupling, off the binding and the logged coupling**: at the constructions with the messages' finders at the
readings `rd`, `CouplingTarget` holds at the session `A` wherever (a) and (b) hold at one property `P` of the play. -/
theorem couplingTarget_of_logged (rd : Readings) (K : Constructions V) (A : Live) (P : Exchange → Prop)
    (θ : ByteArray) (F : Array ByteArray) (ha : LoggedBound V k K P θ F) (hb : LoggedCoupling V k rd K A P θ F) :
    CouplingTarget V k (K.withMessages rd) A θ F :=
  couplingTarget_of_noMiss V k rd (K.withMessages rd) rfl rfl A θ F
    (noMiss_le_of_logged V k rd (K.withMessages rd) A P θ F ha hb)

end Verity.Sampled
