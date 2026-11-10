module

public import Proofs.Properties.Verity.Sampled.Window.Reach

/-!
# Sampled proofs' leg over a window: the leg from the walk's premises

`spotSound_of_window` gives SPOT's target at a window of several members from the coupling at each window
(`WindowCouplingTarget`); the projection through the members' sessions in order gives the coupling from the coin map's
premise and the decoders' (`windowCouplingTarget_of_logged`), and the decoders' premise holds a member at a time
(`windowDecodes_of_members`). Put together:

* **The leg from the two premises** (`windowSpotSound_of_logged`): at the constructions with the window's projection
  and the messages' finders, from the coin map's premise (`WindowCouples`) and the decoders' (`WindowDecodes`) at
  every strategy.
* **The leg, a member at a time** (`windowSpotSound_of_members`): the decoders' premise at each member's session
  (`MemberDecodes`): each member's session's trace, in the members' order, writes a transcript flock-verify `--zk`
  accepts wherever the play v2 accepts is on a tape through the trace's end.
-/

@[expose] public section

namespace Verity.Sampled

open Verity.Game

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

variable (V : Vocab d α) (k : Verity.Registration.Composite)

/-- **SPOT's leg over a window of several members, from the walk's two premises**: soundness's vocabulary at
`.sampled` reading, at the deployment's windows `S`, the constructions with the window's projection and the messages'
finders at the readings `rd`, the members' decoders `D` and the coin maps `M` (`WindowDeployed`), with `S` covering
the one-stage form (`WindowCovers`), meets SPOT's target at composite `k` wherever, at each window of `S`, (a) an
accepted play `P` doesn't hold of is at most as likely as the read finder's collision plus the forger's forgery, (a')
one `Q` doesn't hold of is at most as likely as the square root of the bound digests' finder's success, and, at each
commitment and strategy off its held-off events, the coin map's premise (`WindowCouples`) and the decoders'
(`WindowDecodes`) hold. -/
theorem windowSpotSound_of_logged (rd : WindowReadings) (K : WindowConstructions V)
    (D : (W : Window) → (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) → Strategy (V.game θ F c) →
      ℕ → Decoders W.rounds ByteArray)
    (M : (W : Window) → (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) → (σ : Strategy (V.game θ F c)) →
      WindowCoinMap (V.game θ F c) σ W.Ω W.rounds)
    {S : ByteArray → Array ByteArray → Option Window} (hD : WindowDeployed V (K.withProject V rd D M) S)
    (hCov : WindowCovers V k S) (P : Exchange → Prop)
    (Q : (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) → Strategy (V.game θ F c) → Exchange → Prop)
    (BL : (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) → Strategy (V.game θ F c) → WindowLinkHold)
    (B : (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) → Strategy (V.game θ F c) → WindowRoundHold)
    (ha : ∀ (θ : ByteArray) (F : Array ByteArray) (c : ByteArray) (σ : Strategy (V.game θ F c)),
      prob (fun o => V.Verdict k .sampled o ∧ ¬ P o) (V.game θ F c) σ ≤
        ENNReal.ofReal ((K.readFinder θ F c σ).Pr sha512Msg) +
          ENNReal.ofReal ((K.forger θ F c σ).Pr Verity.lmsVerify))
    (hq : ∀ θ F W, S θ F = some W → ∀ (c : ByteArray) (σ : Strategy (V.game θ F c)),
      prob (fun o => V.Verdict k .sampled o ∧ ¬ Q θ F c σ o) (V.game θ F c) σ ≤
        ENNReal.ofReal (Real.sqrt ((K.bindFinder W θ F c σ).Pr sha512Msg)))
    (hC : ∀ θ F W, S θ F = some W → ∀ (c : ByteArray) (σ : Strategy (V.game θ F c)),
      WindowCouples V θ F c W (V.cost θ F c σ) (rd W θ F c) σ (D W θ F c σ) (M W θ F c σ) (BL θ F c σ) (B θ F c σ))
    (hDec : ∀ θ F W, S θ F = some W → ∀ (c : ByteArray) (σ : Strategy (V.game θ F c)),
      WindowDecodes V θ F c k (fun o => P o ∧ Q θ F c σ o) W (V.cost θ F c σ) (rd W θ F c) σ (D W θ F c σ)
        (M W θ F c σ) (BL θ F c σ) (B θ F c σ)) :
    V.SpotSound k :=
  spotSound_of_window V k hD hCov fun θ F W hW =>
    windowCouplingTarget_of_logged V k rd K D M W P θ F (Q θ F) (BL θ F) (B θ F) (ha θ F) (hq θ F W hW)
      (hC θ F W hW) (hDec θ F W hW)

/-- **SPOT's leg over a window of several members, a member at a time**: `windowSpotSound_of_logged` with the
decoders' premise at each member's session (`MemberDecodes`), at any invariant `I` of the position a member's session
starts at. At each window of `S`, each commitment and each strategy: the coin map's premise over the window
(`WindowCouples`), and for each member in the members' order, its session's trace from where the one before it ended
writes a transcript flock-verify `--zk` accepts at the member's `own` when the member has a proof block in the sample,
wherever a tape through the trace's end plays a play v2's sampled verdict accepts that misses nowhere and `P` and `Q`
hold of. The bound is `spotSound_of_window`'s: the sample's own miss and the members' terms summed
(`Window.sessionDelta`, `JointShareV5`'s), a two-continuation term per round of every member's session, the record's,
the bound digests', the hiding term, the read finder's and the forger's. -/
theorem windowSpotSound_of_members (rd : WindowReadings) (K : WindowConstructions V)
    (D : (W : Window) → (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) → Strategy (V.game θ F c) →
      ℕ → Decoders W.rounds ByteArray)
    (M : (W : Window) → (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) → (σ : Strategy (V.game θ F c)) →
      WindowCoinMap (V.game θ F c) σ W.Ω W.rounds)
    {S : ByteArray → Array ByteArray → Option Window} (hD : WindowDeployed V (K.withProject V rd D M) S)
    (hCov : WindowCovers V k S) (P : Exchange → Prop)
    (Q : (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) → Strategy (V.game θ F c) → Exchange → Prop)
    (BL : (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) → Strategy (V.game θ F c) → WindowLinkHold)
    (B : (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) → Strategy (V.game θ F c) → WindowRoundHold)
    (I : (W : Window) → (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) → (σ : Strategy (V.game θ F c)) →
      (W.extend (M W θ F c σ).widening).L.Ω → ℕ → Pos (V.game θ F c) σ Claims.Msg (Fin W.rounds) → Prop)
    (ha : ∀ (θ : ByteArray) (F : Array ByteArray) (c : ByteArray) (σ : Strategy (V.game θ F c)),
      prob (fun o => V.Verdict k .sampled o ∧ ¬ P o) (V.game θ F c) σ ≤
        ENNReal.ofReal ((K.readFinder θ F c σ).Pr sha512Msg) +
          ENNReal.ofReal ((K.forger θ F c σ).Pr Verity.lmsVerify))
    (hq : ∀ θ F W, S θ F = some W → ∀ (c : ByteArray) (σ : Strategy (V.game θ F c)),
      prob (fun o => V.Verdict k .sampled o ∧ ¬ Q θ F c σ o) (V.game θ F c) σ ≤
        ENNReal.ofReal (Real.sqrt ((K.bindFinder W θ F c σ).Pr sha512Msg)))
    (hC : ∀ θ F W, S θ F = some W → ∀ (c : ByteArray) (σ : Strategy (V.game θ F c)),
      WindowCouples V θ F c W (V.cost θ F c σ) (rd W θ F c) σ (D W θ F c σ) (M W θ F c σ) (BL θ F c σ) (B θ F c σ))
    (hm : ∀ θ F W, S θ F = some W → ∀ (c : ByteArray) (σ : Strategy (V.game θ F c)),
      MemberDecodes V θ F c k (fun o => P o ∧ Q θ F c σ o) W (V.cost θ F c σ) (rd W θ F c) σ (D W θ F c σ)
        (M W θ F c σ) (BL θ F c σ) (B θ F c σ) (I W θ F c σ)) :
    V.SpotSound k :=
  windowSpotSound_of_logged V k rd K D M hD hCov P Q BL B ha hq hC fun θ F W hW c σ =>
    windowDecodes_of_members V θ F c k (fun o => P o ∧ Q θ F c σ o) W (V.cost θ F c σ) (rd W θ F c) σ (D W θ F c σ)
      (M W θ F c σ) (BL θ F c σ) (B θ F c σ) (I W θ F c σ) (hm θ F W hW c σ)

end Verity.Sampled
