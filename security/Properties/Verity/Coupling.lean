module

public import Properties.Verity.Subprotocols
public import Definitions.Accounting.JointPeriod

/-!
# The causal game's coins, flipped up front: each subprotocol's bound in the coins model carried to the play

In the causal game (`Verity.play`) the device flips its coins as the prover drives it, round by round: each frame the
prover sends picks what the device samples next, after every coin before it. A subprotocol's guarantee is stated over coins
flipped up front: `Accounting.PeriodRun.Sound` bounds, at every value of the other coins, the probability over the audit's
own coins. This module states the coupling between the two and the bounds it carries.

* **The coins flipped up front** (`UpFrontAll fs D`): every coin of the coin commitment's families `fs` (`UpFront fs`),
  family `j`'s coin `n` the one the device releases when the family goes from `n + 1` coins left to `n`
  (`UpFront.read`, `coinsLeft`), and the device's sample at the close under its signer `D` (`finish`: each coin's salt
  and the coins it committed to and didn't release), which no release reads. The coin trees' coins are not among them:
  what a tree samples depends on the shape and nonce the prover sends at its `Hello`, so they are resolved from the
  device's log so far (`TreeCoins`).
* **The play at those coins** (`outUp fs D θ F c ρ σ t`): the play of `σ` with each family coin read from `t`, each coin
  tree's from `ρ` and the close's sample from `t`: the prover's request, the device's log, the bytes the close leaves to
  the device, the sample and the device's signatures, and the evidence `σ` sends after it (`outK`). Each family coin is
  read once: a family's coins left only go down. A frame the device refuses, a tree frame past its coin-tree lines or a
  `Hello` of a spec it doesn't sample among them, ends its releases, as in the play.
* **The coupling**, which every guarantee here rests on and which the proofs tree proves (`play_couples`): for every
  event, the play's probability at `σ` lies between its probabilities over the coins flipped up front at two resolutions
  of the coin trees' coins. An event that reads no coin tree's coin has the same probability in both.
* **Each subprotocol's bound over the coins flipped up front** (`Vocab.UpFrontFor acc k a`, `Vocab.UpFrontBound k a`): at
  every resolution of the coin trees' coins, the probability over the coins flipped up front that `acc` holds while `a`'s
  claim fails is at most `a`'s error at the strategy. It gives `a`'s target at `acc` in the play (`CoinsCouple`), and so
  `Vocab.ComponentBound` at `acc = ComponentAccepts`.
* **From a period guarantee** (`TargetOfPeriods`): for the subprotocol that decides `Audit` `b`, a run of the coins model
  (`Accounting.PeriodRun`) whose coins are in bijection with the coins flipped up front (`Vocab.PeriodCoupling`), whose
  verdict `acc` refines and whose claim refines `b`'s in the play, meeting `PeriodRun.Sound` within the published `δ`,
  with the rest of the event (a log out of place, the coupling's break) bounded by the forger's and the finder's terms
  (`Vocab.PeriodCoupling.BreakBound`; together `Vocab.PeriodsCouple`), gives `b`'s target at `acc` in the play. At
  `acc = ComponentAccepts` it is `Vocab.ComponentBound`, which is then no longer a hypothesis of
  `SoundnessOfComponents` for that subprotocol.
* **The wipe** (`WipeTarget`): the wipe's guarantee (`Verity.Wipe.SurvivorBound`) is over its own sample, and its coupling
  to the play (`Vocab.WipeCoupling`) is `Verity.Wipe.OfRun.Preserves`, the wipe's verdict refining `acc` and the
  survivor error within the wipe's; with them it gives the wipe's target at `acc`. `Preserves` reads the foreign
  content as uniform on the play.
* **Soundness** (`SoundnessOfUpFront`): at every composite, each subprotocol's bound at its component, over the coins
  flipped up front, or as its target (`Vocab.SubprotocolBound`), gives `Vocab.SoundnessAt`.

A subprotocol whose claim reads the coin trees' coins (sampled proofs) is bounded at every resolution of them only if a
prover choosing them can't break it, so its bound stays at its component (`Vocab.ComponentBound`) or its target
(`Vocab.Target`), proved in the play itself. The inference-only policy's claim reads no coin, so its bound is its own.
-/

@[expose] public section

namespace Verity

open Accounting Core
open scoped ENNReal

/-- **The coin commitment's coins, flipped up front**: for each of the families `fs`, in order, each of its coins, 512
bits. A family's coin `n` is the one the device releases when the family goes from `n + 1` coins left to `n`. -/
def UpFront : List (String × ℕ) → Type
  | [] => PUnit
  | (_, n) :: fs => (Fin n → Fin 512 → Bool) × UpFront fs

instance UpFront.instFintype : (fs : List (String × ℕ)) → Fintype (UpFront fs)
  | [] => inferInstanceAs (Fintype PUnit)
  | (_, n) :: fs => have := UpFront.instFintype fs; inferInstanceAs (Fintype ((Fin n → Fin 512 → Bool) × UpFront fs))

instance UpFront.instNonempty : (fs : List (String × ℕ)) → Nonempty (UpFront fs)
  | [] => inferInstanceAs (Nonempty PUnit)
  | (_, n) :: fs => have := UpFront.instNonempty fs; inferInstanceAs (Nonempty ((Fin n → Fin 512 → Bool) × UpFront fs))

/-- **Every coin the play samples apart from the coin trees', sampled up front**: the coin commitment's families' coins and
the device's sample at the close under the signer `D` (`finish`), which no release reads. -/
abbrev UpFrontAll (fs : List (String × ℕ)) (D : Signer) : Type := UpFront fs × (Fin D.draws → Bool)

/-- **The coins left of the family `l`** in `fs`: of the first family labelled `l`, the one `takeCoin` takes from; `0`
where no family is. -/
def coinsLeft : List (String × ℕ) → String → ℕ
  | [], _ => 0
  | (l', n) :: fs, l => if l' = l then n else coinsLeft fs l

/-- **Coin `n` of the family `l`, sampled up front**: coin `n` of the first family of `fs` labelled `l`; all zero where
no family is or the family has no coin `n`. -/
def UpFront.read : {fs : List (String × ℕ)} → UpFront fs → String → ℕ → Fin 512 → Bool
  | [], _, _, _ => fun _ => false
  | (l', c) :: _, t, l, n =>
    if l' = l then (if h : n < c then t.1 ⟨n, h⟩ else fun _ => false) else UpFront.read t.2 l n

/-- **The coin trees' coins, resolved from the device's log**: at a coin tree's `Hello`, after the releases `pre`, the
sample for its shape and nonce; at an opening of the `i`-th tree for its stream `s`'s first `n` coins, after the releases
`pre` and with the bytes `b`, a sample among those with the leaf the tree committed to at the round it opens. -/
structure TreeCoins where
  hello : List Release → (spec : Flock.CoinTree.Spec) → ByteArray → TreeSample spec
  opening : List Release → (i s n : ℕ) → ByteArray → (T : OpenTree) → (s' : Fin T.spec.streams.size) →
    (r : Fin (2 ^ T.spec.roundsLog)) → TreeOpenable T.nonce T.spec T.sample s' r

/-- A strategy for `G` relabelled and then continued by `f` is a strategy for `G` continued by `f` after the
relabelling: the same message at every node. -/
def Game.Strategy.assocMap {α β γ : Type} (h : α → β) (f : β → Game γ) :
    (G : Game α) → Game.Strategy ((G.map h).bind f) → Game.Strategy (G.bind fun a => f (h a))
  | .ret _, s => s
  | .send _ k, s => ⟨s.1, Game.Strategy.assocMap h f (k s.1) s.2⟩
  | @Game.coin _ _ _ _ k, s => fun c => Game.Strategy.assocMap h f (k c) (s c)

/-- A strategy for an `if` continued by `f`, where the condition holds: a strategy for its first branch continued by
`f`. -/
def Game.Strategy.ifPos {β γ : Type} {c : Prop} [h : Decidable c] {A B : Game β} {f : β → Game γ} (hc : c)
    (σ : Game.Strategy ((if c then A else B).bind f)) : Game.Strategy (A.bind f) :=
  match h, σ with
  | isTrue _, σ => σ
  | isFalse hn, _ => absurd hc hn

/-- A strategy for an `if` continued by `f`, where the condition fails: a strategy for its second branch continued by
`f`. -/
def Game.Strategy.ifNeg {β γ : Type} {c : Prop} [h : Decidable c] {A B : Game β} {f : β → Game γ} (hc : ¬ c)
    (σ : Game.Strategy ((if c then A else B).bind f)) : Game.Strategy (B.bind f) :=
  match h, σ with
  | isTrue hp, _ => absurd hp hc
  | isFalse _, σ => σ

/-- **Where the play of a strategy goes at the coins flipped up front**: from the device's state (the families' coins
left `fs`, the trees open `ts`, at most `k` frames more) after the releases `pre`, with each family coin read from `t`
(the coins of the families `fs₀`) and each coin tree's from `ρ`, the device's log from there and the strategy left for
`f` of it. -/
def outK (fs₀ : List (String × ℕ)) (t : UpFront fs₀) (ρ : TreeCoins) {β : Type} :
    (k : ℕ) → (fs : List (String × ℕ)) → (ts : List OpenTree) → (pre : List Release) →
    (f : List Release → Game β) → Game.Strategy ((releases k fs ts).bind f) →
    (log : List Release) × Game.Strategy (f log)
  | 0, _, _, _, _, σ => ⟨[], σ⟩
  | _ + 1, _, _, _, _, ⟨none, σ⟩ => ⟨[], σ⟩
  | k + 1, fs, ts, pre, f, ⟨some (.fresh l, b), σ⟩ => by
    dsimp only at σ
    exact match takeCoin l fs, σ with
      | none, σ => ⟨[], σ⟩
      | some fs', σ =>
        let x := t.read l (coinsLeft fs' l)
        let r : Release := ⟨.fresh l, b, List.ofFn x⟩
        let o := outK fs₀ t ρ k fs' ts (pre ++ [r]) (fun log => f (r :: log))
          (Game.Strategy.assocMap (r :: ·) f _ (σ x))
        ⟨r :: o.1, o.2⟩
  | k + 1, fs, ts, pre, f, ⟨some (.hello spec, ν), σ⟩ => by
    dsimp only at σ
    exact if hc : treeLines ts < maxTreeLines ∧ TreeSpecOk spec then
      let σ := Game.Strategy.ifPos hc σ
      let x := ρ.hello pre spec ν
      let r : Release := ⟨.hello spec, ν, bitsOfBytes (treeRoot ν spec x ++ treeKey x.1)⟩
      let o := outK fs₀ t ρ k fs (ts ++ [⟨ν, spec, x, fun _ => 0⟩]) (pre ++ [r]) (fun log => f (r :: log))
        (Game.Strategy.assocMap (r :: ·) f _ (σ x))
      ⟨r :: o.1, o.2⟩
    else ⟨[], Game.Strategy.ifNeg hc σ⟩
  | k + 1, fs, ts, pre, f, ⟨some (.opening i s n, b), σ⟩ => by
    dsimp only at σ
    refine if hc : treeLines ts < maxTreeLines then ?_ else ⟨[], Game.Strategy.ifNeg hc σ⟩
    match ts[i]?, Game.Strategy.ifPos hc σ with
    | none, σ => exact ⟨[], σ⟩
    | some T, σ =>
      dsimp only at σ
      match T.next s n, σ with
      | none, σ => exact ⟨[], σ⟩
      | some (s', r'), σ =>
        exact
          let d := ρ.opening pre i s n b T s' r'
          let r : Release := ⟨.opening i s n, b, TreeDraw.opened d.1 n⟩
          let o := outK fs₀ t ρ k fs (ts.set i (T.bump s')) (pre ++ [r]) (fun log => f (r :: log))
            (Game.Strategy.assocMap (r :: ·) f _ (σ d))
          ⟨r :: o.1, o.2⟩

/-- **The play of `σ` at the coins flipped up front `t`**, with the coin trees' coins resolved by `ρ`: the parameters, the
committed files and the commitment, the prover's request, the device's log, the bytes the prover's close leaves to the
device, the device's sample at the close (`t`'s) and its signatures under `D`, and the evidence `σ` sends after it. -/
def outUp (fs : List (String × ℕ)) (D : Signer) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (ρ : TreeCoins) (σ : Game.Strategy (play fs D θ F c)) (t : UpFrontAll fs D) : Exchange :=
  let o := outK fs t.1 ρ (frames fs) fs [] [] (finish D θ F c σ.1) σ.2
  let e := (o.2.2 t.2).1
  ⟨θ, F, c, σ.1, o.1, o.2.1, List.ofFn t.2, D.signatures σ.1 o.1 o.2.1 (List.ofFn t.2), e.1, e.2⟩

namespace Vocab

variable {ι : Type} {d : Device ι} {α : Type} [Fintype α] (V : Vocab d α)

/-- **Subprotocol `a`'s bound at composite `k` at the verdict `acc`, over the coins flipped up front**: at all `θ`, `F`
and `c`, for every strategy and every resolution of the coin trees' coins, the probability over the coins flipped up front
that `acc` holds of the play at them while `a`'s claim fails there is at most `a`'s error at the strategy. -/
def UpFrontFor (acc : Exchange → Prop) (k : Verity.Registration.Composite) (a : Subprotocol) : Prop :=
  ∀ (θ : ByteArray) (F : Array ByteArray) (c : ByteArray) (σ : Game.Strategy (V.game θ F c)) (ρ : TreeCoins),
    Game.prCoin (fun t => acc (outUp (V.families θ) (V.signer θ) θ F c ρ σ t) ∧
        ¬ V.Holds k θ F c σ a (outUp (V.families θ) (V.signer θ) θ F c ρ σ t)) ≤
      V.error a θ F c σ

/-- **Subprotocol `a`'s bound at its compiled component at composite `k`, over the coins flipped up front**: `UpFrontFor`
at `a`'s component's acceptance of its own case in what the prover sent. -/
def UpFrontBound (k : Verity.Registration.Composite) (a : Subprotocol) : Prop :=
  V.UpFrontFor (fun o => ComponentAccepts k a o.sent) k a

/-- **A period run coupled to the play of `σ` at the resolution `ρ`, for the subprotocol that decides `Audit` `b` at
composite `k` and the verdict `acc`**: a run of the coins model on a circuit `C` whose coins are in bijection with the
coins flipped up front (`coins`), at a price, with a break on the coins (`brk`, the events `b`'s finder answers for), such
that where `acc` holds of the play at the coins, every release `b`'s check read is the device's (`Logged`) and the
break is absent, the run's verdict for `b` accepts (`verdict`), and the run's claim for `b` implies the play's
(`world`). -/
structure PeriodCoupling (acc : Exchange → Prop) (k : Verity.Registration.Composite) (b : Audit) (θ : ByteArray)
    (F : Array ByteArray) (c : ByteArray) (σ : Game.Strategy (V.game θ F c)) (ρ : TreeCoins) where
  C : Verity.Circuit
  I : Type
  Ω : I → Type
  [fintypeI : Fintype I]
  [decEqI : DecidableEq I]
  [fintypeΩ : ∀ i, Fintype (Ω i)]
  [nonemptyΩ : ∀ i, Nonempty (Ω i)]
  coins : ((i : I) → Ω i) ≃ UpFrontAll (V.families θ) (V.signer θ)
  run : PeriodRun d α C Ω
  price : Price
  brk : ((i : I) → Ω i) → Prop
  verdict : ∀ ξ, acc (outUp (V.families θ) (V.signer θ) θ F c ρ σ (coins ξ)) →
    V.Logged (.ofAudit b) (outUp (V.families θ) (V.signer θ) θ F c ρ σ (coins ξ)) → ¬ brk ξ → run.accepts b ξ
  world : ∀ ξ, run.Holds (V.certOf θ) price b ξ →
    V.Holds k θ F c σ (.ofAudit b) (outUp (V.families θ) (V.signer θ) θ F c ρ σ (coins ξ))

attribute [instance] PeriodCoupling.fintypeI PeriodCoupling.decEqI PeriodCoupling.fintypeΩ
  PeriodCoupling.nonemptyΩ

/-- **The rest of the event, beyond the run's**: the probability over the coupled run's coins that `acc` holds of the
play at them while a release `b`'s check read is not the device's or the coupling's break holds is at most the
probability that `b`'s forger forges under the compiled LMS verifier plus `b`'s loss at its finder's collision
probability. A release out of place is a forgery of the device's signature over its log, and a break a collision. -/
def PeriodCoupling.BreakBound {acc : Exchange → Prop} {k : Verity.Registration.Composite} {b : Audit} {θ : ByteArray}
    {F : Array ByteArray} {c : ByteArray} {σ : Game.Strategy (V.game θ F c)} {ρ : TreeCoins}
    (q : V.PeriodCoupling acc k b θ F c σ ρ) : Prop :=
  Game.prCoin (fun ξ : (i : q.I) → q.Ω i => acc (outUp (V.families θ) (V.signer θ) θ F c ρ σ (q.coins ξ)) ∧
      (¬ V.Logged (.ofAudit b) (outUp (V.families θ) (V.signer θ) θ F c ρ σ (q.coins ξ)) ∨ q.brk ξ)) ≤
    ENNReal.ofReal ((V.forger (.ofAudit b) θ F c σ).Pr lmsVerify) +
      V.loss (.ofAudit b) θ (ENNReal.ofReal ((V.finder (.ofAudit b) θ F c σ).Pr Verity.sha512Msg))

/-- **The plays at `b`'s subprotocol couple to period runs**: at every `θ`, `F`, `c`, strategy and resolution of the
coin trees' coins there is a period run coupled to the play (`PeriodCoupling`) that meets `b`'s per-period guarantee
(`Accounting.PeriodRun.Sound`) within `b`'s published `δ` at the strategy's calls, and whose rest of the event is bounded
by the forger's and the finder's terms where the finder meets `FinderCR` (`PeriodCoupling.BreakBound`). -/
def PeriodsCouple (acc : Exchange → Prop) (k : Verity.Registration.Composite) (b : Audit) : Prop :=
  ∀ (θ : ByteArray) (F : Array ByteArray) (c : ByteArray) (σ : Game.Strategy (V.game θ F c)) (ρ : TreeCoins),
    ∃ (q : V.PeriodCoupling acc k b θ F c σ ρ) (δ : ℝ), q.run.Sound (V.certOf θ) q.price b δ ∧
      ENNReal.ofReal δ ≤ V.delta θ (.ofAudit b) (V.cost θ F c σ) ∧
      (V.FinderCR (.ofAudit b) θ F c σ → q.BreakBound)

/-- **The plays couple to the wipe's game at the survivor error `ε`, for the verdict `acc`**: every play read at the
wipe's boundary is distributed as the wipe's game samples it at its strategy (`Verity.Wipe.OfRun.Preserves`), `acc`
implies the wipe's verdict on the play (`Verity.Wipe.OfRun.accepts`), and `ε` at the play's moves at the boundary is at
most the wipe's error at the strategy. -/
structure WipeCoupling [Fintype V.Seed] [Nonempty V.Seed] [Fintype V.Spots] [Nonempty V.Spots]
    (acc : Exchange → Prop)
    (ε : (θ : ByteArray) → (F : Array ByteArray) → (S_φ : ℕ) → ((Fin S_φ → Bool) → Device.State d) →
      (Device.State d → (Fin (V.wipe θ F).egress → Bool) → V.Spots → Fin S_φ → Bool) → ℝ≥0∞) : Prop where
  preserves : ∀ θ F c σ, (V.wipeRun θ F c σ).Preserves σ
  verdict : ∀ θ F c σ o, acc o → (V.wipeRun θ F c σ).accepts o
  error : ∀ θ F c σ, ε θ F (V.wipeRun θ F c σ).S_φ (V.wipeRun θ F c σ).hold (V.wipeRun θ F c σ).keep ≤
    V.error .wipe θ F c σ

/-- **Subprotocol `a`'s bound at composite `k`, in one of three forms**: at its compiled component in the play
(`ComponentBound`), at its compiled component over the coins flipped up front (`UpFrontBound`), or as its target in the
play (`Target`, as a piece that proves it at its verdict directly gives it). -/
def SubprotocolBound (k : Verity.Registration.Composite) (a : Subprotocol) : Prop :=
  V.ComponentBound k a ∨ V.UpFrontBound k a ∨ V.Target k a

end Vocab

/-- **A subprotocol's target in the play from its bound over the coins flipped up front.** On any device, for any names
`V` the subprotocols define, any verdict `acc`, composite `k` and subprotocol `a`: if at every resolution of the coin
trees' coins the probability over the coins flipped up front that `acc` holds while `a`'s claim fails is at most `a`'s
error, then so is its probability in the play, at every strategy. At `acc = ComponentAccepts k a` it gives
`Vocab.ComponentBound k a` from `Vocab.UpFrontBound k a`. -/
def CoinsCouple : Prop :=
  ∀ (ι α : Type) [Fintype α] (d : Device ι) (V : Vocab d α) (acc : Exchange → Prop)
    (k : Verity.Registration.Composite) (a : Subprotocol),
    V.UpFrontFor acc k a → V.TargetFor acc k a

/-- **A subprotocol's target in the play from a period guarantee of the coins model.** On any device, for any names
`V`, any verdict `acc`, composite `k` and `Audit` `b`: if at every `θ`, `F`, `c`, strategy and resolution of the coin
trees' coins there is a period run coupled to the play (`Vocab.PeriodCoupling`) that meets `b`'s per-period guarantee
(`Accounting.PeriodRun.Sound`) within `b`'s published `δ` at the strategy's calls, and whose rest of the event is
bounded by the forger's and the finder's terms where the finder meets `FinderCR` (`BreakBound`; `Vocab.PeriodsCouple`),
then `b`'s subprotocol
meets its target at `acc` in the play. At `acc = ComponentAccepts k (.ofAudit b)` it is
`Vocab.ComponentBound k (.ofAudit b)`. -/
def TargetOfPeriods : Prop :=
  ∀ (ι α : Type) [Fintype α] (d : Device ι) (V : Vocab d α) (acc : Exchange → Prop)
    (k : Verity.Registration.Composite) (b : Audit),
    V.PeriodsCouple acc k b → V.TargetFor acc k (.ofAudit b)

/-- **The wipe's target in the play from its survivor bound.** On any device, for any names `V` whose wipe's seeds and
spot reads are finite, any verdict `acc`, composite `k` and survivor error `ε`: if the wipe meets its survivor bound at
the published threshold and `ε` (`Verity.Wipe.SurvivorBound`), every play read at the wipe's boundary is distributed as
the wipe's game samples it at its strategy (`Verity.Wipe.OfRun.Preserves`), `acc` implies the wipe's verdict on the play
(`Verity.Wipe.OfRun.accepts`), and `ε` at the play's moves at the boundary is at most the wipe's error at the strategy
(`Vocab.WipeCoupling`), then the wipe meets its target at `acc` in the play. -/
def WipeTarget : Prop :=
  ∀ (ι α : Type) [Fintype α] (d : Device ι) (V : Vocab d α) [Fintype V.Seed] [Nonempty V.Seed] [Fintype V.Spots]
    [Nonempty V.Spots] (acc : Exchange → Prop) (k : Verity.Registration.Composite)
    (ε : (θ : ByteArray) → (F : Array ByteArray) → (S_φ : ℕ) → ((Fin S_φ → Bool) → Device.State d) →
      (Device.State d → (Fin (V.wipe θ F).egress → Bool) → V.Spots → Fin S_φ → Bool) → ℝ≥0∞),
    (∀ θ F, (V.wipe θ F).SurvivorBound (V.foreignBits θ) (ε θ F)) → V.WipeCoupling acc ε →
    V.TargetFor acc k .wipe

/-- **Soundness at every composite's verifier, from its subprotocols' bounds at their components, over the
coins flipped up front, or as targets.** On any device, for any names `V` and any composite `k`, if each of `k`'s
subprotocols meets its bound at its compiled component in the play (`Vocab.ComponentBound`), over the coins flipped up
front (`Vocab.UpFrontBound`), or as its target in the play (`Vocab.Target`; `Vocab.SubprotocolBound`), then `k`'s compiled
verifier is sound (`Vocab.SoundnessAt`). -/
def SoundnessOfUpFront : Prop :=
  ∀ (ι α : Type) [Fintype α] (d : Device ι) (V : Vocab d α) (k : Verity.Registration.Composite),
    (∀ a ∈ subprotocols k, V.SubprotocolBound k a) → V.SoundnessAt k

end Verity
