module

public import Properties.Poub.Remote

/-! What PoUB's verifier gives of the opening sample's premises `Refused` and `Live`, and the release order it cannot
give. -/

@[expose] public section

open Poub

namespace Properties.Poub

/-- Receipt `u` of window `w` (its index among the window's receipts in pass order, `receiptIndex`) has an opening
that passes `checkOpening` under its pass's receipter: some pass `i` and position `p < L_i` with index `u`. -/
def Openable (reg : Registration) (appends : Array Append) (w : WindowRecord) (u : Nat) : Prop :=
  ∃ i p o, i < w.passes.size ∧ p < w.passes[i]!.plan.length ∧
    receiptIndex (w.passes.map (·.plan.length)) i p = u ∧
    checkOpening reg reg.digest appends w (receipterOf reg w i) i p o = none

/-- An accepted window's samples are the sampler's values `us` on its openings key, and each value is an openable
receipt of the window. -/
def AcceptSamplesOpenable : Prop :=
  ∀ (reg : Registration) (appends : Array Append) (w : WindowRecord) (coin : ByteArray) (samples : Nat)
    (openings : Array Opening),
    verifyWindow reg appends w coin samples openings = some .accept →
      ∃ L us, FlockSoundness.Randomness.subset (openingsKey reg.digest w.index coin)
          (min samples (w.passes.map (·.plan.length)).toList.sum) (w.passes.map (·.plan.length)).toList.sum
          [.str "openings"] L = some us ∧
        ∀ u ∈ us, Openable reg appends w u

/-- Window `t`'s samples in a session, `none` until `close t` samples them. -/
def sampledOf (s : Session.State) (t : Nat) : Option (Array (Nat × Nat)) := (s.windows[t]?).bind (·.drawn)

/-- A run from `State.init` that leaves window `t` unsampled reads no coin at `W + t` or above: on coins `coins'` that
agree with `coins` below `W + t`, the same events give the same results and the same state, with `coins'`. -/
def OpeningsUnreadUntilSampled : Prop :=
  ∀ (params : Session.Params) (reg : Registration) (coins appendCoins : Array ByteArray) (notBeforeNs : Nat)
    (events : Array Session.Event) (rs : Array Session.Result) (s : Session.State),
    Session.run (Session.State.init params reg coins appendCoins notBeforeNs) events = .ok (rs, s) →
    ∀ (t : Nat) (coins' : Array ByteArray), 1 ≤ params.windows → sampledOf s t = none →
      (∀ j, j < params.windows + t → coins'[j]! = coins[j]!) →
      Session.run (Session.State.init params reg coins' appendCoins notBeforeNs) events =
        .ok (rs, { s with coins := coins' })

/-- Once a run from `State.init` has sampled window `t` (at `s`), any events after it keep the window's samples, its record
and its openable receipts: at `s'` they are those at `s`. -/
def RecordKeptAfterSample : Prop :=
  ∀ (params : Session.Params) (reg : Registration) (coins appendCoins : Array ByteArray) (notBeforeNs : Nat)
    (events events' : Array Session.Event) (rs rs' : Array Session.Result) (s s' : Session.State),
    Session.run (Session.State.init params reg coins appendCoins notBeforeNs) events = .ok (rs, s) →
    Session.run s events' = .ok (rs', s') →
    ∀ (t : Nat) (d : Array (Nat × Nat)), 1 ≤ params.minPasses → sampledOf s t = some d →
      sampledOf s' t = some d ∧ s'.record t = s.record t ∧
        ∀ u, Openable reg s'.appends (s'.record t) u ↔ Openable reg s.appends (s.record t) u

/-- **The release order**, the deployment's, which the code cannot give: window `t`'s openings coin reaches none of
the events `D` gives a session before the one that samples the window. -/
def ReleaseOrder (params : Session.Params) (reg : Registration) (appendCoins : Array ByteArray) (notBeforeNs : Nat)
    (D : Array ByteArray → Array Session.Event) : Prop :=
  ∀ (coins coins' : Array ByteArray) (t k : Nat) (rs : Array Session.Result) (s : Session.State),
    (∀ j, j < params.windows + t → coins'[j]! = coins[j]!) →
    Session.run (Session.State.init params reg coins appendCoins notBeforeNs) ((D coins).extract 0 k) = .ok (rs, s) →
    sampledOf s t = none → (D coins')[k]? = (D coins)[k]?

/-- **`Live`, as the code states it**, on the events `D` a session sees on its coins: two sessions on coins that agree
below `W + t`, each run to a point where window `t` is sampled, see the same record of it and the same openable
receipts. -/
def LiveUnder (params : Session.Params) (reg : Registration) (appendCoins : Array ByteArray) (notBeforeNs : Nat)
    (D : Array ByteArray → Array Session.Event) : Prop :=
  ∀ (coins coins' : Array ByteArray) (t k k' : Nat) (rs rs' : Array Session.Result) (s s' : Session.State),
    (∀ j, j < params.windows + t → coins'[j]! = coins[j]!) →
    Session.run (Session.State.init params reg coins appendCoins notBeforeNs) ((D coins).extract 0 k) = .ok (rs, s) →
    Session.run (Session.State.init params reg coins' appendCoins notBeforeNs) ((D coins').extract 0 k') =
      .ok (rs', s') →
    (sampledOf s t).isSome → (sampledOf s' t).isSome →
      s'.record t = s.record t ∧
        ∀ u, Openable reg s'.appends (s'.record t) u ↔ Openable reg s.appends (s.record t) u

end Properties.Poub
