module

public import Properties.Poub.Remote.Coins
public import Poub.Shared

/-! PoUB's coin and sample properties for a session that samples its openings `g` windows at a time on one openings
coin. -/

@[expose] public section

open Poub

namespace Properties.Poub.Shared

/-- Sample `c`'s windows, at `g` windows a sample and `W` in all: `c·g … min(c·g + g, W) − 1`, the windows
`Shared.close` samples together. -/
def sampleWindows (g W c : Nat) : List Nat := (List.range (min (c * g + g) W - c * g)).map (c * g + ·)

/-- The lengths of sample `c`'s passes, window by window in order: `Shared.close`'s `lengths`, whose positions number
the sample's receipts. -/
def sampleLengths (g : Nat) (s : Session.State) (c : Nat) : Array Nat :=
  (sampleWindows g s.params.windows c).foldl (fun acc u => acc ++ s.windows[u]!.plans.map (·.length)) #[]

/-- The passes of the windows from `first` up to window `u`, not counting `u`'s: where `u`'s passes start among its
sample's. -/
def passesBefore (s : Session.State) (first u : Nat) : Nat :=
  ((List.range (u - first)).map fun j => s.windows[first + j]!.plans.size).sum

/-- The samples of `sampled` in passes `base … base + n − 1`, renumbered from `base`: a window's share of its sample. -/
def shareOf (sampled : Array (Nat × Nat)) (base n : Nat) : Array (Nat × Nat) :=
  sampled.filterMap fun (i, p) => if base ≤ i && i < base + n then some (i - base, p) else none

/-- The pass lengths of window records `ws`, window by window in order: a sample's numbering of its receipts. -/
def lengthsOf (ws : List WindowRecord) : Array Nat :=
  (ws.flatMap fun w => (w.passes.map (·.plan.length)).toList).toArray

/-- The passes of window records `ws`: with `ws` a sample's first `k` windows, where window `k`'s passes start. -/
def passCount (ws : List WindowRecord) : Nat := (ws.map (·.passes.size)).sum

/-- The receipts of window records `ws`: with `ws` a sample's first `k` windows, where window `k`'s receipts start. -/
def receiptCount (ws : List WindowRecord) : Nat := (ws.map fun w => (w.passes.map (·.plan.length)).toList.sum).sum

/-- Sample `c` (`t / g` of the window `t` it closes at): its values are `Key.subset`'s `min(s, U)`-subset of `[0, U)`
on the openings key of `c` and the coin `coins[W + c]`, `U` the sample's receipts (`sampleLengths`); each value is mapped to
the receipt `(i, p)` of the sample's passes with that number, so they are distinct receipts of the sample, one for each
value; and each window `u` of the sample holds its share of them (`shareOf`), renumbered to its own passes. -/
def SampleIsSubset : Prop :=
  ∀ (g : Nat) (s : Session.State) (t : Int) (now : Nat) (sampled : Array (Nat × Nat)) (s' : Session.State),
    Shared.close g s t now = .ok (.drew sampled, s') →
      ∃ L us, FlockSoundness.Randomness.subset
          (openingsKey s.digest (t.toNat / g) s.coins[s.params.windows + t.toNat / g]!)
          (min s.params.samples (sampleLengths g s (t.toNat / g)).toList.sum) (sampleLengths g s (t.toNat / g)).toList.sum
          [.str "openings"] L = some us ∧
        sampled.size = us.size ∧ us.size = min s.params.samples (sampleLengths g s (t.toNat / g)).toList.sum ∧
        (∀ k (hk : k < sampled.size), sampled[k].1 < (sampleLengths g s (t.toNat / g)).size ∧
          sampled[k].2 < (sampleLengths g s (t.toNat / g))[sampled[k].1]! ∧
          receiptIndex (sampleLengths g s (t.toNat / g)) sampled[k].1 sampled[k].2 = us[k]!) ∧
        (∀ k₁ k₂ (h₁ : k₁ < sampled.size) (h₂ : k₂ < sampled.size), sampled[k₁] = sampled[k₂] → k₁ = k₂) ∧
        ∀ u ∈ sampleWindows g s.params.windows (t.toNat / g),
          sampledOf s' u = some (shareOf sampled (passesBefore s (t.toNat / g * g) u) s.windows[u]!.plans.size)

/-- Sample `c` over the records `ws` of its windows, in order, on the coin `coin`: its values `us` are the sampler's on
its openings key, `U` the windows' receipts (`lengthsOf`). Once window `k`'s check of its share (`shareOf`, from its
passes' start, `passCount`) accepts its openings, for each `k`, or the share is empty (which the close accepts), each
value `v` is an openable receipt of the window `k` it falls in, its number there `v − receiptCount (ws.take k)`. -/
def AcceptSamplesOpenable : Prop :=
  ∀ (reg : Registration) (appends : Array Append) (c : Nat) (coin : ByteArray) (samples : Nat)
    (ws : List WindowRecord) (sampled : Array (Nat × Nat)),
    receiptSample reg.digest c coin samples (lengthsOf ws) = some sampled →
      ∃ L us, FlockSoundness.Randomness.subset (openingsKey reg.digest c coin)
          (min samples (lengthsOf ws).toList.sum) (lengthsOf ws).toList.sum [.str "openings"] L = some us ∧
        ∀ openings : Nat → Array Opening,
          (∀ k (hk : k < ws.length), shareOf sampled (passCount (ws.take k)) ws[k].passes.size = #[] ∨
            checkWindow reg reg.digest appends ws[k] (shareOf sampled (passCount (ws.take k)) ws[k].passes.size)
              (openings k) = .accept) →
          ∀ v ∈ us, ∃ k, ∃ hk : k < ws.length, receiptCount (ws.take k) ≤ v ∧
            Openable reg appends ws[k] (v - receiptCount (ws.take k))

/-- The records of sample `c`'s windows at `s`, in order. -/
def sampleRecords (g : Nat) (s : Session.State) (c : Nat) : List WindowRecord :=
  (sampleWindows g s.params.windows c).map s.record

/-- **The session link**, per sample, at any `g ≥ 1` windows a sample: in a run from `State.init` that has sampled the
first window of sample `c` (one of the session's, `c·g < W`), the sample is `draws` on the openings key of `c` and the
coin `coins[W + c]` over the records `ws` of `c`'s windows (`sampleRecords`), `U` their receipts (`lengthsOf`); window
`k` of it holds its share (`shareOf`, from its passes' start, `passCount`); and once window `k` is accepted, its share
is empty (the close accepts it) or `checkWindow` of its share passes on some openings and the session's appends. -/
def RunSamplesRecords : Prop :=
  ∀ (g : Nat) (params : Session.Params) (reg : Registration) (coins appendCoins : Array ByteArray) (notBeforeNs : Nat)
    (events : Array Session.Event) (rs : Array Session.Result) (s : Session.State),
    Shared.run g (Session.State.init params reg coins appendCoins notBeforeNs) events = .ok (rs, s) →
    ∀ c, 1 ≤ g → 1 ≤ params.minPasses → c * g < params.windows → (sampledOf s (c * g)).isSome →
      ∃ sampled, receiptSample reg.digest c coins[params.windows + c]! params.samples (lengthsOf (sampleRecords g s c)) =
          some sampled ∧
        ∀ k (hk : k < (sampleRecords g s c).length),
          sampledOf s (c * g + k) =
            some (shareOf sampled (passCount ((sampleRecords g s c).take k)) (sampleRecords g s c)[k].passes.size) ∧
          (s.windows[c * g + k]!.verdict = some .accept →
            shareOf sampled (passCount ((sampleRecords g s c).take k)) (sampleRecords g s c)[k].passes.size = #[] ∨
            ∃ openings : Array Opening, checkWindow reg reg.digest s.appends (sampleRecords g s c)[k]
              (shareOf sampled (passCount ((sampleRecords g s c).take k)) (sampleRecords g s c)[k].passes.size)
              openings = .accept)

/-- **A sample's accepted windows open its values** (`RunSamplesRecords` and `AcceptSamplesOpenable` joined): in a run
from `State.init` that has accepted every window of sample `c` (`c·g < W`), the sample's values `us` are the sampler's
on the openings key of `c` and the coin `coins[W + c]`, `U` its windows' receipts, and each value `v` is an openable
receipt, on the session's appends, of the window `k` it falls in, its number there `v − receiptCount (ws.take k)`. -/
def AcceptedSamplesOpenable : Prop :=
  ∀ (g : Nat) (params : Session.Params) (reg : Registration) (coins appendCoins : Array ByteArray) (notBeforeNs : Nat)
    (events : Array Session.Event) (rs : Array Session.Result) (s : Session.State),
    Shared.run g (Session.State.init params reg coins appendCoins notBeforeNs) events = .ok (rs, s) →
    ∀ c, 1 ≤ g → 1 ≤ params.minPasses → c * g < params.windows → (sampledOf s (c * g)).isSome →
      (∀ k, k < (sampleRecords g s c).length → s.windows[c * g + k]!.verdict = some .accept) →
      ∃ L us, FlockSoundness.Randomness.subset (openingsKey reg.digest c coins[params.windows + c]!)
          (min params.samples (lengthsOf (sampleRecords g s c)).toList.sum) (lengthsOf (sampleRecords g s c)).toList.sum
          [.str "openings"] L = some us ∧
        ∀ v ∈ us, ∃ k, ∃ hk : k < (sampleRecords g s c).length, receiptCount ((sampleRecords g s c).take k) ≤ v ∧
          Openable reg s.appends (sampleRecords g s c)[k] (v - receiptCount ((sampleRecords g s c).take k))

/-- A run from `State.init` that leaves sample `c`'s first window unsampled reads no coin at `W + c` or above: on coins
`coins'` that agree with `coins` below `W + c`, the same events give the same results and the same state, with
`coins'`. -/
def OpeningsUnreadUntilSampled : Prop :=
  ∀ (g : Nat) (params : Session.Params) (reg : Registration) (coins appendCoins : Array ByteArray) (notBeforeNs : Nat)
    (events : Array Session.Event) (rs : Array Session.Result) (s : Session.State),
    Shared.run g (Session.State.init params reg coins appendCoins notBeforeNs) events = .ok (rs, s) →
    ∀ (c : Nat) (coins' : Array ByteArray), 1 ≤ params.windows → sampledOf s (c * g) = none →
      (∀ j, j < params.windows + c → coins'[j]! = coins[j]!) →
      Shared.run g (Session.State.init params reg coins' appendCoins notBeforeNs) events =
        .ok (rs, { s with coins := coins' })

/-- Once a run from `State.init` has sampled window `t` (at `s`), any events after it keep the window's samples, its record
and its openable receipts: at `s'` they are those at `s`. -/
def RecordKeptAfterSample : Prop :=
  ∀ (g : Nat) (params : Session.Params) (reg : Registration) (coins appendCoins : Array ByteArray) (notBeforeNs : Nat)
    (events events' : Array Session.Event) (rs rs' : Array Session.Result) (s s' : Session.State),
    Shared.run g (Session.State.init params reg coins appendCoins notBeforeNs) events = .ok (rs, s) →
    Shared.run g s events' = .ok (rs', s') →
    ∀ (t : Nat) (d : Array (Nat × Nat)), 1 ≤ params.minPasses → sampledOf s t = some d →
      sampledOf s' t = some d ∧ s'.record t = s.record t ∧
        ∀ u, Openable reg s'.appends (s'.record t) u ↔ Openable reg s.appends (s.record t) u

/-- **The release order**, per sample, the deployment's, which the code cannot give: sample `c`'s openings coin
reaches none of the events `D` gives a session before the one that samples it. -/
def ReleaseOrder (g : Nat) (params : Session.Params) (reg : Registration) (appendCoins : Array ByteArray)
    (notBeforeNs : Nat) (D : Array ByteArray → Array Session.Event) : Prop :=
  ∀ (coins coins' : Array ByteArray) (c k : Nat) (rs : Array Session.Result) (s : Session.State),
    (∀ j, j < params.windows + c → coins'[j]! = coins[j]!) →
    Shared.run g (Session.State.init params reg coins appendCoins notBeforeNs) ((D coins).extract 0 k) = .ok (rs, s) →
    sampledOf s (c * g) = none → (D coins')[k]? = (D coins)[k]?

/-- **`Live`, as the code states it**, per sample, on the events `D` a session sees on its coins: two sessions on coins
that agree below `W + c`, each run to a point where window `t` of sample `c` is sampled, see the same record of it and
the same openable receipts. -/
def LiveUnder (g : Nat) (params : Session.Params) (reg : Registration) (appendCoins : Array ByteArray)
    (notBeforeNs : Nat) (D : Array ByteArray → Array Session.Event) : Prop :=
  ∀ (coins coins' : Array ByteArray) (c k k' : Nat) (rs rs' : Array Session.Result) (s s' : Session.State),
    (∀ j, j < params.windows + c → coins'[j]! = coins[j]!) →
    Shared.run g (Session.State.init params reg coins appendCoins notBeforeNs) ((D coins).extract 0 k) = .ok (rs, s) →
    Shared.run g (Session.State.init params reg coins' appendCoins notBeforeNs) ((D coins').extract 0 k') =
      .ok (rs', s') →
    ∀ t, t / g = c → (sampledOf s t).isSome → (sampledOf s' t).isSome →
      s'.record t = s.record t ∧
        ∀ u, Openable reg s'.appends (s'.record t) u ↔ Openable reg s.appends (s.record t) u

end Properties.Poub.Shared
