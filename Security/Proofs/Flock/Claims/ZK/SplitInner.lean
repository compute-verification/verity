import Proofs.Flock.Claims.ZK.SimInner

/-!
# `InnerSrcE` as five statements

`SimInner` follows from `InnerSrcE` (`simInner_of`): where the phases hybrids 2 and 3 share reach a rep's inner proof,
the cut's source carries what the session absorbs there. Here `InnerSrcE` is five statements at its hypotheses
(`InnerAt`), each proved on its own:

* `InnerSlot`: the shared phases end at the rep's `ε` slot (`innerSlotE`), the round `czExe` reads `ζ` from.
* `InnerTriple`: the cut's triple openings are `ρ`, `σ` at the honest pads, with `ε` the coin at that slot.
* `InnerBatched`: the batched constraint holds at the honest pads, at the batching coins after the slot.
* `InnerTau`: the cut's `τ` commitment is the session's, over the shared phases' constraints. It is two statements
  (`innerTau_of`): `InnerTauVal`, the session's `τ` is the real view's (`View.tau`, which the last message also sends),
  and `InnerTauLeaf`, the cut's commitment is the view's `τ` under the pads' `τ` salt.
* `InnerY`: the cut's `Y` is the session's, where `β` is not `0`.

`innerSrcE_of` assembles them (`InnerSlot` puts the other four at the slot the shared phases end at), and
`simInner_of_split` gives `SimInner`.
-/

namespace Claims.ZKMal

open Claims FlockSoundness.ZK
open FlockSoundness.Discharge
open CoinLayer

namespace Masked

/-- **Table `j`'s rep `r`'s `ε` slot**: round `innerOff 0` after the rep's PIOP rounds, where `czExe` reads `ζ`
(`restAt`'s `zeta`); the batching coins, `β` and the query coins are the next three. -/
abbrev innerSlotE (x : Msg) (j : Fin (jOf x)) (r : Fin 2) : ℕ := repSlot x (2 * j.val + r.val) (innerOff 0)

/-- The inner proof's coins at table `j`'s rep `r`'s slot, as `asksCut` answers the opened coins `c`. -/
abbrev innerCoinsE {x : Msg} (c : Fin (refK x) → FlockDeploy.Coin) (j : Fin (jOf x)) (r : Fin 2) :
    FlockComplete.Replay.InZ.Challenges :=
  FlockComplete.SimInner.cutCoins (fun k i => slotAt c k i) (innerSlotE x j r)

/-- Table `j`'s rep `r` source on the cut. -/
noncomputable abbrev innerSrcRepE (x : Msg) (c : Fin (refK x) → FlockDeploy.Coin) (w : Claims.Flock.Wit) (τ : UE.Tape)
    (j : Fin (jOf x)) (r : Fin 2) : FlockComplete.RepSrc :=
  (srcE x c w τ j.val).rep r.val

/-- **Where the shared phases reach table `j`'s rep `r`'s inner proof** (`InnerSrcE`'s hypotheses): at a live statement
and a witness in the deployed relation, at opened coins `c` and a stop `n` at which no refusal is seen, the phases hybrids
2 and 3 share (`innerPreE`), begun at the rep's first round, return the constraints `cons` after `L` bodies. -/
def InnerAt (x : Msg) (w : Claims.Flock.Wit) (τ : UE.Tape) (c : Fin (refK x) → FlockDeploy.Coin) (n : ℕ)
    (j : Fin (jOf x)) (r : Fin 2) (cons : Array (String × _root_.Flock.Zk.Lin)) (L : ℕ) : Prop :=
  Live x ∧ RZk x w ∧ ¬ Refused x c n ∧
    ∃ s spec, FlockDeploy.setupOf true x = some (s, spec, stsOf x) ∧
    ∃ (dom : ByteArray) (cm0 : FlockComplete.Commit) (ic : FlockComplete.InnerCommit)
      (extras : Array _root_.Flock.Claim),
      (FlockDeploy.tabsOf (stsOf x))[j.val]!.extra (ptsE c spec) = .ok extras ∧
    ∃ l : Array Msg, l.size = repBase x (2 * j.val + r.val) ∧
    ∃ o t' l', ((innerPreE x w τ c n j r dom cm0 extras ic).run (_root_.Flock.Transcript.init #[] dom)).go l =
        (.ok (o, t'), l') ∧ o.2.2.2 = cons ∧ l'.size = L

/-- **The slot count**: where the shared phases reach the inner proof, they have sent up to the rep's `ε` slot. -/
def InnerSlot : Prop :=
  ∀ x w τ c n j r cons L, InnerAt x w τ c n j r cons L → L = innerSlotE x j r

/-- **The triple's openings**: once `ε` is answered, the cut's `ρ`, `σ` are the session's at the honest pads. -/
def InnerTriple : Prop :=
  ∀ x w τ c n j r cons L, InnerAt x w τ c n j r cons L → innerSlotE x j r < n →
    (innerSrcRepE x c w τ j r).rhoIn =
        FlockComplete.Replay.InZ.rho (innerLayE x j) (innerPadsE x τ j r) (innerCoinsE c j r) ∧
      (innerSrcRepE x c w τ j r).sigmaIn = FlockComplete.Replay.InZ.sigma (innerLayE x j) (innerPadsE x τ j r)

/-- **The batched constraint**: once the batching coins are answered, the batch of the shared phases' constraints and the
triple's holds at the honest pads. -/
def InnerBatched : Prop :=
  ∀ x w τ c n j r cons L, InnerAt x w τ c n j r cons L → innerSlotE x j r + 1 < n →
    ((FlockComplete.Replay.InZ.lin (innerLayE x j) (innerPadsE x τ j r) cons (innerCoinsE c j r)).c +
        _root_.Flock.dot (FlockComplete.Replay.InZ.lin (innerLayE x j) (innerPadsE x τ j r) cons (innerCoinsE c j r)).a
          (innerPadsE x τ j r).h).isZero = true

/-- **`τ`'s commitment**: once the batching coins are answered, the cut's is the session's, `τ` over the shared phases'
constraints at the honest pads, committed through SHA-512 under the pads' `τ` salt. -/
def InnerTau : Prop :=
  ∀ x w τ c n j r cons L, InnerAt x w τ c n j r cons L → innerSlotE x j r + 1 < n →
    (innerSrcRepE x c w τ j r).tauCom =
      FlockComplete.Replay.InZ.tcOf _root_.Flock.Sha512.hash
        (FlockComplete.Replay.InZ.tau (innerLayE x j) (innerPadsE x τ j r) cons (innerCoinsE c j r))
        (innerPadsE x τ j r).tauSalt

/-- **Table `j`'s rep `r`'s `τ` in its real view** at the opened coins `c`, on `w`'s table witnesses and the tape `τ`
(`View.tau`, the value `tabSrcView` gives the last message). -/
noncomputable abbrev viewTauE (x : Msg) (c : Fin (refK x) → FlockDeploy.Coin) (w : Claims.Flock.Wit) (τ : UE.Tape)
    (j : Fin (jOf x)) (r : Fin 2) : _root_.Flock.F128 :=
  FlockLevel3.GF128.val (((session (fun _ _ => 0) (tableData PE QE UE)).viewsAt x c (wtOf PE x w) τ j).tau r)

/-- **`τ`'s value**: once the batching coins are answered, the session's `τ` over the shared phases' constraints at the
honest pads is the real view's (`View.tau`, `λ`'s batch against `μ`). The last message's `τ` is the same value. -/
def InnerTauVal : Prop :=
  ∀ x w τ c n j r cons L, InnerAt x w τ c n j r cons L → innerSlotE x j r + 1 < n →
    FlockComplete.Replay.InZ.tau (innerLayE x j) (innerPadsE x τ j r) cons (innerCoinsE c j r) = viewTauE x c w τ j r

/-- **`τ`'s leaf**: the cut's `τ` commitment is the real view's `τ` committed through SHA-512 under the pads' `τ` salt
(the cut's `tauCom` is the view's, `cutV_viewsAt`, and the view's is `leafT` of `τ` and `y_τ`). -/
def InnerTauLeaf : Prop :=
  ∀ x w τ c n j r cons L, InnerAt x w τ c n j r cons L →
    (innerSrcRepE x c w τ j r).tauCom =
      FlockComplete.Replay.InZ.tcOf _root_.Flock.Sha512.hash (viewTauE x c w τ j r) (innerPadsE x τ j r).tauSalt

theorem innerTau_of (hv : InnerTauVal) (hl : InnerTauLeaf) : InnerTau := by
  intro x w τ c n j r cons L hA h
  rw [hl x w τ c n j r cons L hA, hv x w τ c n j r cons L hA h]

/-- **`Y`**: once `β` is answered and is not `0`, the cut's `Y` is the session's at the honest pads. -/
def InnerY : Prop :=
  ∀ x w τ c n j r cons L, InnerAt x w τ c n j r cons L → innerSlotE x j r + 2 < n →
    (slotAt c (innerSlotE x j r + 2) 0).isZero = false →
    (innerSrcRepE x c w τ j r).y = FlockComplete.Replay.InZ.y (innerPadsE x τ j r) (innerCoinsE c j r)

/-- **`InnerSrcE` from the five statements**: the slot count puts the inner proof's coins at the rep's slot, where the
other four give `InnerSrc`'s fields. -/
theorem innerSrcE_of (h1 : InnerSlot) (h2 : InnerTriple) (h3 : InnerBatched) (h4 : InnerTau) (h5 : InnerY) :
    InnerSrcE := by
  intro x hx w hw τ c n hn s spec hs j r dom cm0 ic extras he l hl o t' l' ho
  have hA : InnerAt x w τ c n j r o.2.2.2 l'.size :=
    ⟨hx, hw, hn, s, spec, hs, dom, cm0, ic, extras, he, l, hl, o, t', l', ho, rfl, rfl⟩
  rw [h1 x w τ c n j r _ _ hA]
  exact ⟨h2 x w τ c n j r _ _ hA, h3 x w τ c n j r _ _ hA, h4 x w τ c n j r _ _ hA, h5 x w τ c n j r _ _ hA⟩

/-- **`SimInner` from the five statements** (`innerSrcE_of`, `simInner_of`). -/
theorem simInner_of_split (h1 : InnerSlot) (h2 : InnerTriple) (h3 : InnerBatched) (h4 : InnerTau) (h5 : InnerY) :
    SimInner :=
  simInner_of (innerSrcE_of h1 h2 h3 h4 h5)

end Masked

end Claims.ZKMal
