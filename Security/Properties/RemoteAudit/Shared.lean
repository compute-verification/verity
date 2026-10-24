import Properties.RemoteAudit.Coins
import RemoteAudit.Shared

/-!
# One openings coin per draw (`RemoteAudit.Shared`)

`Properties.RemoteAudit.Coins` and the draw's statement in `Properties.RemoteAudit`, for a session that draws its
openings `g` windows at a time (`audit.Params.groups_per_coin`, `RemoteAudit.Shared.run g`): draw `c` is windows
`c·g … min(c·g + g, W) − 1` (`drawWindows`), drawn together at the close of the last of them by one coin,
`coins[W + c]`, on the openings key of `c`, over the draw's receipts in order, numbered across its windows
(`drawLengths`). Each window gets the draws in its passes, renumbered to its own (`shareOf`, `passesBefore`), and
`verify` checks them against its own record. Where `Coins` indexes by window `t`'s coin `W + t`, these index by
draw `c`'s, `W + c`.

* `DrawIsSubset` (`Shared.close`, the draw): the draw's values are the sampler's `min(s, U)`-subset of `[0, U)`, `U`
  the draw's receipts, each mapped to the receipt with that number, and each window of the draw holds its share.
* `AcceptDrawsOpenable` (`Refused`, per draw): once every window of draw `c` is accepted, each of the sampler's
  values is an openable receipt of the window it falls in (`receiptsBefore`).
* `OpeningsUnreadUntilDrawn` (`Live`, the coin order): a run that leaves draw `c`'s first window undrawn reads no
  coin at `W + c` or above.
* `RecordKeptAfterDraw`: once a window is drawn, the rest of the session keeps its draws, its record and its openable
  receipts, as at `g = 1`.
* `ReleaseOrder`, `LiveUnder`: the deployment's premise and `Live` as the code states it, per draw: on coins that
  agree below `W + c`, wherever two sessions have drawn draw `c`, each window of it has the same record and the same
  openable receipts.

At `g = 1`, `Shared.run 1` is `Session.run` (`RemoteAudit.Shared.run_one`), `c` is the window, and draw `c` is window
`c` alone (`drawWindows_one`, `drawLengths_one`): `OpeningsUnreadUntilDrawn` and `RecordKeptAfterDraw` give today's, and
`ReleaseOrder 1` and `LiveUnder 1` are today's (`Properties.Proofs.RemoteAudit.Shared`'s `*_one`). Today's
`DrawIsSubset` and `AcceptDrawsOpenable` are on `draws` and `verifyWindow` alone; these state them for a session's draw.
-/

open RemoteAudit

namespace Properties.RemoteAudit.Shared

/-- Draw `c`'s windows, at `g` windows a draw and `W` in all: `c·g … min(c·g + g, W) − 1`, the windows
`Shared.close` draws together. -/
def drawWindows (g W c : Nat) : List Nat := (List.range (min (c * g + g) W - c * g)).map (c * g + ·)

/-- The lengths of draw `c`'s passes, window by window in order: `Shared.close`'s `lengths`, whose positions number
the draw's receipts. -/
def drawLengths (g : Nat) (s : Session.State) (c : Nat) : Array Nat :=
  (drawWindows g s.params.windows c).foldl (fun acc u => acc ++ s.windows[u]!.plans.map (·.length)) #[]

/-- The passes of the windows from `first` up to window `u`, not counting `u`'s: where `u`'s passes start among its
draw's. -/
def passesBefore (s : Session.State) (first u : Nat) : Nat :=
  ((List.range (u - first)).map fun j => s.windows[first + j]!.plans.size).sum

/-- The receipts of the windows from `first` up to window `u`, not counting `u`'s: where `u`'s receipts start among
its draw's. -/
def receiptsBefore (s : Session.State) (first u : Nat) : Nat :=
  ((List.range (u - first)).map fun j => (s.windows[first + j]!.plans.map (·.length)).toList.sum).sum

/-- The draws of `drawn` in passes `base … base + n − 1`, renumbered from `base`: a window's share of its draw. -/
def shareOf (drawn : Array (Nat × Nat)) (base n : Nat) : Array (Nat × Nat) :=
  drawn.filterMap fun (i, p) => if base ≤ i && i < base + n then some (i - base, p) else none

/-- Window `u`'s verdict in a session, `none` until it has one. -/
def verdictOf (s : Session.State) (u : Nat) : Option Verdict := (s.windows[u]?).bind (·.verdict)

/-- Draw `c` (`t / g` of the window `t` it closes at): its values are `Key.subset`'s `min(s, U)`-subset of `[0, U)`
on the openings key of `c` and the coin `coins[W + c]`, `U` the draw's receipts (`drawLengths`); each value is mapped to
the receipt `(i, p)` of the draw's passes with that number, so they are distinct receipts of the draw, one for each
value; and each window `u` of the draw holds its share of them (`shareOf`), renumbered to its own passes. -/
def DrawIsSubset : Prop :=
  ∀ (g : Nat) (s : Session.State) (t : Int) (now : Nat) (drawn : Array (Nat × Nat)) (s' : Session.State),
    Shared.close g s t now = .ok (.drew drawn, s') →
      ∃ L us, FlockSoundness.Randomness.subset
          (openingsKey s.digest (t.toNat / g) s.coins[s.params.windows + t.toNat / g]!)
          (min s.params.samples (drawLengths g s (t.toNat / g)).toList.sum) (drawLengths g s (t.toNat / g)).toList.sum
          [.str "openings"] L = some us ∧
        drawn.size = us.size ∧ us.size = min s.params.samples (drawLengths g s (t.toNat / g)).toList.sum ∧
        (∀ k (hk : k < drawn.size), drawn[k].1 < (drawLengths g s (t.toNat / g)).size ∧
          drawn[k].2 < (drawLengths g s (t.toNat / g))[drawn[k].1]! ∧
          receiptIndex (drawLengths g s (t.toNat / g)) drawn[k].1 drawn[k].2 = us[k]!) ∧
        (∀ k₁ k₂ (h₁ : k₁ < drawn.size) (h₂ : k₂ < drawn.size), drawn[k₁] = drawn[k₂] → k₁ = k₂) ∧
        ∀ u ∈ drawWindows g s.params.windows (t.toNat / g),
          drawnOf s' u = some (shareOf drawn (passesBefore s (t.toNat / g * g) u) s.windows[u]!.plans.size)

/-- Once every window of draw `c` is accepted, the draw's values are the sampler's on its openings key and coin
`coins[W + c]`, and each value `v` is an openable receipt of the window `u` it falls in, its number there
`v − receiptsBefore`. -/
def AcceptDrawsOpenable : Prop :=
  ∀ (g : Nat) (params : Session.Params) (reg : Registration) (coins appendCoins : Array ByteArray) (notBeforeNs : Nat)
    (events : Array Session.Event) (rs : Array Session.Result) (s : Session.State),
    Shared.run g (Session.State.init params reg coins appendCoins notBeforeNs) events = .ok (rs, s) →
    ∀ (c : Nat), 1 ≤ g → 1 ≤ params.minPasses → c * g < params.windows →
      (∀ u ∈ drawWindows g params.windows c, verdictOf s u = some .accept) →
      ∃ L us, FlockSoundness.Randomness.subset (openingsKey reg.digest c coins[params.windows + c]!)
          (min params.samples (drawLengths g s c).toList.sum) (drawLengths g s c).toList.sum [.str "openings"] L =
          some us ∧
        ∀ v ∈ us, ∃ u ∈ drawWindows g params.windows c, receiptsBefore s (c * g) u ≤ v ∧
          Openable reg s.appends (s.record u) (v - receiptsBefore s (c * g) u)

/-- A run from `State.init` that leaves draw `c`'s first window undrawn reads no coin at `W + c` or above: on coins
`coins'` that agree with `coins` below `W + c`, the same events give the same results and the same state, with
`coins'`. -/
def OpeningsUnreadUntilDrawn : Prop :=
  ∀ (g : Nat) (params : Session.Params) (reg : Registration) (coins appendCoins : Array ByteArray) (notBeforeNs : Nat)
    (events : Array Session.Event) (rs : Array Session.Result) (s : Session.State),
    Shared.run g (Session.State.init params reg coins appendCoins notBeforeNs) events = .ok (rs, s) →
    ∀ (c : Nat) (coins' : Array ByteArray), 1 ≤ params.windows → drawnOf s (c * g) = none →
      (∀ j, j < params.windows + c → coins'[j]! = coins[j]!) →
      Shared.run g (Session.State.init params reg coins' appendCoins notBeforeNs) events =
        .ok (rs, { s with coins := coins' })

/-- Once a run from `State.init` has drawn window `t` (at `s`), any events after it keep the window's draws, its record
and its openable receipts: at `s'` they are those at `s`. -/
def RecordKeptAfterDraw : Prop :=
  ∀ (g : Nat) (params : Session.Params) (reg : Registration) (coins appendCoins : Array ByteArray) (notBeforeNs : Nat)
    (events events' : Array Session.Event) (rs rs' : Array Session.Result) (s s' : Session.State),
    Shared.run g (Session.State.init params reg coins appendCoins notBeforeNs) events = .ok (rs, s) →
    Shared.run g s events' = .ok (rs', s') →
    ∀ (t : Nat) (d : Array (Nat × Nat)), 1 ≤ params.minPasses → drawnOf s t = some d →
      drawnOf s' t = some d ∧ s'.record t = s.record t ∧
        ∀ u, Openable reg s'.appends (s'.record t) u ↔ Openable reg s.appends (s.record t) u

/-- **The release order**, per draw (the deployment's, which the code cannot give): `D` gives the events a session sees
on its coins, and draw `c`'s openings coin reaches none of them before the one that draws it. While the first `k`
events leave draw `c`'s first window undrawn, the next is the same on coins that agree below `W + c`. -/
def ReleaseOrder (g : Nat) (params : Session.Params) (reg : Registration) (appendCoins : Array ByteArray)
    (notBeforeNs : Nat) (D : Array ByteArray → Array Session.Event) : Prop :=
  ∀ (coins coins' : Array ByteArray) (c k : Nat) (rs : Array Session.Result) (s : Session.State),
    (∀ j, j < params.windows + c → coins'[j]! = coins[j]!) →
    Shared.run g (Session.State.init params reg coins appendCoins notBeforeNs) ((D coins).extract 0 k) = .ok (rs, s) →
    drawnOf s (c * g) = none → (D coins')[k]? = (D coins)[k]?

/-- **`Live`, as the code states it**, per draw, on the events `D` a session sees on its coins: two sessions on coins
that agree below `W + c`, each run to a point where window `t` of draw `c` is drawn, see the same record of it and the
same openable receipts. -/
def LiveUnder (g : Nat) (params : Session.Params) (reg : Registration) (appendCoins : Array ByteArray)
    (notBeforeNs : Nat) (D : Array ByteArray → Array Session.Event) : Prop :=
  ∀ (coins coins' : Array ByteArray) (c k k' : Nat) (rs rs' : Array Session.Result) (s s' : Session.State),
    (∀ j, j < params.windows + c → coins'[j]! = coins[j]!) →
    Shared.run g (Session.State.init params reg coins appendCoins notBeforeNs) ((D coins).extract 0 k) = .ok (rs, s) →
    Shared.run g (Session.State.init params reg coins' appendCoins notBeforeNs) ((D coins').extract 0 k') =
      .ok (rs', s') →
    ∀ t, t / g = c → (drawnOf s t).isSome → (drawnOf s' t).isSome →
      s'.record t = s.record t ∧
        ∀ u, Openable reg s'.appends (s'.record t) u ↔ Openable reg s.appends (s.record t) u

end Properties.RemoteAudit.Shared
