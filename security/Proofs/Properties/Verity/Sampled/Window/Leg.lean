module

public import Proofs.Properties.Verity.Sampled.Window.Logged
public import Proofs.Properties.Verity.Sampled.Window.Charge

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
* **The leg at the window's one sample** (`windowSpotSound_of_sample`): the walk, the sample and the members' steps
  in one statement. The coin maps are the sample-coupled ones (`windowSampleMaps`), whose premise holds off the
  openings' empty classes at their counts, within the hiding term at every strategy, wherever the window's coins are
  uniform (`windowCouplingTarget_of_sample_at`), so nothing is left of the coin map; and each member's step is taken
  at a position that keeps the play's start and stops where `StopsAt` says (`MemberSteps`,
  `memberDecodes_of_stops`).
-/

@[expose] public section

namespace Verity.Sampled

open Verity.Game
open FlockSoundness.Discharge.Tables (SubmitJ LiveOutJ)
open scoped ENNReal

local notation "𝔽" => FlockLevel3.GF128

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
    (D : (W : Window) → (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) → (σ : Strategy (V.game θ F c)) →
      WindowDecoders V θ F c W σ)
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
decoders' premise at each member's session (`MemberDecodes`), at any invariant `I` of the position the members'
traces end at and the position a member's session starts at. At each window of `S`, each commitment and each strategy: the coin map's premise over the window
(`WindowCouples`), and for each member with a proof block in the sample, in the members' order, its session's trace
from where the one before it ended writes a transcript flock-verify `--zk` accepts at the member's `own`, wherever a
tape through the trace's end plays a play v2's sampled verdict accepts that misses nowhere and `P` and `Q` hold of; a
member with none has no session, and the next member's starts from the same position. The bound is
`spotSound_of_window`'s: the sample's own miss and the members' terms summed (`Window.sessionDelta`,
`JointShareV5`'s), a two-continuation term per round of every member's session, the record's, the bound digests',
the hiding term, the read finder's and the forger's. -/
theorem windowSpotSound_of_members (rd : WindowReadings) (K : WindowConstructions V)
    (D : (W : Window) → (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) → (σ : Strategy (V.game θ F c)) →
      WindowDecoders V θ F c W σ)
    (M : (W : Window) → (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) → (σ : Strategy (V.game θ F c)) →
      WindowCoinMap (V.game θ F c) σ W.Ω W.rounds)
    {S : ByteArray → Array ByteArray → Option Window} (hD : WindowDeployed V (K.withProject V rd D M) S)
    (hCov : WindowCovers V k S) (P : Exchange → Prop)
    (Q : (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) → Strategy (V.game θ F c) → Exchange → Prop)
    (BL : (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) → Strategy (V.game θ F c) → WindowLinkHold)
    (B : (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) → Strategy (V.game θ F c) → WindowRoundHold)
    (I : (W : Window) → (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) → (σ : Strategy (V.game θ F c)) →
      (W.extend (M W θ F c σ).widening).L.Ω → Pos (V.game θ F c) σ Claims.Msg (Fin W.rounds) → ℕ →
        Pos (V.game θ F c) σ Claims.Msg (Fin W.rounds) → Prop)
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

/-- **SPOT's leg over a window of several members, at the window's one sample**: the walk through the members'
sessions in order, the one sample's coins and each member's step, composed onto `JointShareV5`'s bound
(`spotSound_of_window`). Soundness's vocabulary at `.sampled` reading, at the deployment's windows `S`, the
constructions with the window's projection at the readings `rd`, the members' decoders `D` and the sample-coupled
coin maps (`windowSampleMaps`, at the family `label` answering `after` and the window's coins `κ`), with `S` covering
the one-stage form, meets SPOT's target at composite `k` wherever, at each window of `S`:

* the window's coins `κ` are uniform (`hκ`; A3 at the bytes the sample reads gives it, `Window.avg_byteCoins`);
* (a) an accepted play `P` doesn't hold of is at most as likely as the read finder's collision plus the forger's
  forgery (`ha`), and (a') one `Q` doesn't hold of at most as likely as the square root of the bound digests'
  finder's success (`hq`);
* each round's reading picks only openings (`hR`);
* the members' steps hold off the openings' empty classes at their counts (`hm`, `MemberSteps` at `linkEmptyAt` and
  `roundEmptyAt`): the session's trace of each member with a proof block in the sample, in the members' order, from a
  position that keeps the play's start and whose stop is where `StopsAt` says, writes a transcript flock-verify `--zk`
  accepts at the member's `own`, wherever a tape through the trace's end plays a play v2's sampled verdict accepts,
  that misses at no round and not at the record, and that `P` and `Q` hold of. A member with no proof block in the
  sample has no statement and no session in the play: the walk consumes no stop for it, and the next member's session
  starts from the same position.

The stops' term takes no premise: over every member's stops the openings' gaps and empty classes at their counts are
within the hiding term at every strategy (`windowErr_sampleMap_le_at`). The member's transcript carries the proof
files its decoders give at the position its trace ends at (`PosDecoders.files`). The bound is
`spotSound_of_window`'s: the sample's own miss and the members' terms summed (`Window.sessionDelta`), a
two-continuation term per round of every member's session, the record's, the bound digests', the hiding term, the
read finder's and the forger's. -/
theorem windowSpotSound_of_sample (e : 𝔽 ≃ (Fin 128 → Bool)) (rd : WindowReadings) (K : WindowConstructions V)
    (D : (W : Window) → (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) → (σ : Strategy (V.game θ F c)) →
      WindowDecoders V θ F c W σ)
    (label : (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) → Strategy (V.game θ F c) → String)
    (after : (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) → Strategy (V.game θ F c) → ByteArray) {nc : ℕ}
    (κ : (W : Window) → W.Ω → Fin nc → Fin 512 → Bool) {S : ByteArray → Array ByteArray → Option Window}
    (hD : WindowDeployed V (K.withProject V rd D (windowSampleMaps e V rd label after κ)) S)
    (hCov : WindowCovers V k S) (P : Exchange → Prop)
    (Q : (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) → Strategy (V.game θ F c) → Exchange → Prop)
    (I : (W : Window) → (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) → (σ : Strategy (V.game θ F c)) →
      (W.extend (windowSampleMaps e V rd label after κ W θ F c σ).widening).L.Ω →
        Pos (V.game θ F c) σ Claims.Msg (Fin W.rounds) → ℕ → Pos (V.game θ F c) σ Claims.Msg (Fin W.rounds) → Prop)
    (hκ : ∀ θ F W, S θ F = some W → ∀ H : (Fin nc → Fin 512 → Bool) → ℝ≥0∞, avg (fun ω => H (κ W ω)) = avg H)
    (ha : ∀ θ F W, S θ F = some W → ∀ (c : ByteArray) (σ : Strategy (V.game θ F c)),
      prob (fun o => V.Verdict k .sampled o ∧ ¬ P o) (V.game θ F c) σ ≤
        ENNReal.ofReal ((K.readFinder θ F c σ).Pr sha512Msg) +
          ENNReal.ofReal ((K.forger θ F c σ).Pr Verity.lmsVerify))
    (hq : ∀ θ F W, S θ F = some W → ∀ (c : ByteArray) (σ : Strategy (V.game θ F c)),
      prob (fun o => V.Verdict k .sampled o ∧ ¬ Q θ F c σ o) (V.game θ F c) σ ≤
        ENNReal.ofReal (Real.sqrt ((K.bindFinder W θ F c σ).Pr sha512Msg)))
    (hR : ∀ θ F W, S θ F = some W → ∀ (c : ByteArray) j pre fr, (rd W θ F c).round j pre fr →
      ∃ i s n, fr = .opening i s n)
    (hm : ∀ θ F W, S θ F = some W → ∀ (c : ByteArray) (σ : Strategy (V.game θ F c)),
      MemberSteps V θ F c k (fun o => P o ∧ Q θ F c σ o) W (V.cost θ F c σ) (rd W θ F c) σ (D W θ F c σ)
        (windowSampleMaps e V rd label after κ W θ F c σ) (fun _ _ => linkEmptyAt e (finish (V.signer θ) θ F c σ.1))
        (fun _ => roundEmptyAt e (finish (V.signer θ) θ F c σ.1)) (I W θ F c σ)) :
    V.SpotSound k :=
  spotSound_of_window V k hD hCov fun θ F W hW =>
    windowCouplingTarget_of_sample_at e V k rd K D label after κ W P θ F (Q θ F) (hκ θ F W hW) (ha θ F W hW)
      (hq θ F W hW) fun c σ =>
        windowDecodes_of_members V θ F c k (fun o => P o ∧ Q θ F c σ o) W (V.cost θ F c σ) (rd W θ F c) σ
          (D W θ F c σ) (windowSampleMaps e V rd label after κ W θ F c σ) _ _ _
          (memberDecodes_of_stops V θ F c k (fun o => P o ∧ Q θ F c σ o) W (V.cost θ F c σ) (rd W θ F c) σ
            (D W θ F c σ) (windowSampleMaps e V rd label after κ W θ F c σ)
            (fun _ _ => linkEmptyAt e (finish (V.signer θ) θ F c σ.1))
            (fun _ => roundEmptyAt e (finish (V.signer θ) θ F c σ.1)) (I W θ F c σ) (hR θ F W hW c)
            (hm θ F W hW c σ))

end Verity.Sampled
