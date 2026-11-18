module

public import Properties.Poub.Remote

/-!
# PoUB's verifier with refresh on (`Poub`)

The code's half of refresh's conditions in the game (`ResourceAccounting.RateSketch.LifetimeCappedAll`, `Seasoned`,
`JoinsInOrder` and `OriginsFixed`, `security/Definitions/Poub/Carrier.lean`), stated against `Session.run` from
`State.init` with `Params.refresh` on and appends filling the slots of the public schedule `Params.slotAt`, and against
the window's check. They take no premise.

* `RegisteredCapped`: no plan of a window at or past `lifetime` holds a registered block;
* `ProbationKept`: a plan of window `t` holds a block of an append only once its payload is seasoned: committed at the
  registration, or born (`State.born`, its fresh slot's window) a window `b` with `b + lifetime ≤ t`;
* `OriginsOnRota`: in an accepted session every append fills a slot of the schedule, slots strictly increasing, with
  the origin and count the slot's kind names and the window the slot joins as its `joined` window: the origins are one
  map of the public parameters (`OriginsFixed`), and the joined windows are the schedule's, which never decrease
  (`JoinsInOrder`);
* `OpensOrigin`: in an accepted window, every sampled block of a re-key append decodes under the append's coin to a
  payload whose leaf opens at the origin's position under the origin's root (`originLeaf`);
* `CoinAfterAppend`: an append coin leaves only in reply to the append the session records as its `k`-th, and it is
  the coin of that append's slot; `CoinBySlot` is its form with refresh on, where every append names its slot.
-/

@[expose] public section

open Poub

namespace Properties.Poub

/-- **With refresh on, no plan holds a registered block at or past window `lifetime`.** A registered block's key is
the registration's, which joined window 0; to stay plannable its payload is re-keyed by an append (`receive` refuses
with `lifetime` a plan of window `t ≥ L` holding a block below `N`). -/
def RegisteredCapped : Prop :=
  ∀ (params : Session.Params) (reg : Registration) (coins appendCoins : Array ByteArray) (notBeforeNs : Nat)
    (events : Array Session.Event) (rs : Array Session.Result) (s : Session.State),
    Session.run (Session.State.init params reg coins appendCoins notBeforeNs) events = .ok (rs, s) →
      params.refresh = true →
      ∀ t (ht : t < s.windows.size) i (hi : i < s.windows[t].plans.size) p,
        p < s.windows[t].plans[i].length → s.windows[t].plans[i].block p < reg.blocks →
          t < params.lifetime

/-- **With refresh on, no plan holds an appended block whose payload is not yet seasoned.** A block of append `a` in a
plan of window `t`: unless `a` re-keys registered blocks, its payload's birth, the window of the fresh slot that first
committed it (`a`'s own slot, or the slot `k` it re-keys), is a window `b` with `b + L ≤ t` (`receive` refuses with
`probation`). -/
def ProbationKept : Prop :=
  ∀ (params : Session.Params) (reg : Registration) (coins appendCoins : Array ByteArray) (notBeforeNs : Nat)
    (events : Array Session.Event) (rs : Array Session.Result) (s : Session.State),
    Session.run (Session.State.init params reg coins appendCoins notBeforeNs) events = .ok (rs, s) →
      params.refresh = true →
      ∀ t (ht : t < s.windows.size) i (hi : i < s.windows[t].plans.size) p,
        p < s.windows[t].plans[i].length →
        ∀ a (ha : a < s.appends.size),
          s.appends[a].first ≤ s.windows[t].plans[i].block p →
          s.windows[t].plans[i].block p < s.appends[a].first + s.appends[a].count →
            match s.appends[a].origin with
            | .registered _ => True
            | .fresh => ∃ b, s.slotWindow s.appends[a].slot = some b ∧ b + params.lifetime ≤ t
            | .rekey k => ∃ b, s.slotWindow k = some b ∧ b + params.lifetime ≤ t

/-- Append `a` is what its slot's kind in the schedule names: a fresh append of at most `REKEY_MAX` blocks in a fresh
slot; the re-key of the whole class run `[lo, lo + n)` in a registered slot; or, in a re-key slot of fresh slot `k`,
the re-key of the whole of an earlier fresh append that fills slot `k`. -/
def OnRota (p : Session.Params) (N : Nat) (appends : Array Append) (a : Nat) : Prop :=
  match appends[a]? with
  | none => False
  | some x =>
    match p.slotAt N x.slot, x.origin with
    | some (_, .fresh), .fresh => x.count ≤ REKEY_MAX
    | some (_, .registered lo n), .registered b => b = lo ∧ x.count = n
    | some (_, .rekey k), .rekey k' =>
      k' = k ∧ ∃ b < a, ∃ y, appends[b]? = some y ∧ y.slot = k ∧ y.origin = .fresh ∧ y.count = x.count
    | _, _ => False

/-- **In an accepted session with refresh on, the appends are the public schedule's.** Every append fills a slot of
`Params.slotAt` (a function of the parameters and `N` alone, fixed before every coin), in strictly increasing slot
order, holding the origin and count its slot's kind names (`OnRota`), and joins the window its slot is laid out in.
So the map from an append to its origin is the schedule's, at every coin (the game's `OriginsFixed`), and the windows
the appends join are the schedule's, which increase with the slot (`JoinsInOrder`). -/
def OriginsOnRota : Prop :=
  ∀ (params : Session.Params) (reg : Registration) (coins appendCoins : Array ByteArray) (notBeforeNs : Nat)
    (events : Array Session.Event) (rs : Array Session.Result) (s : Session.State),
    Session.run (Session.State.init params reg coins appendCoins notBeforeNs) events = .ok (rs, s) →
      params.refresh = true → s.outcome = some "accepted" →
      ∀ a (ha : a < s.appends.size),
        OnRota params reg.blocks s.appends a ∧
        s.joined[a]? = s.slotWindow s.appends[a].slot ∧
        ∀ b (hb : b < s.appends.size), a < b → s.appends[a].slot < s.appends[b].slot

/-- **An accepted window opens every sampled block of a re-key at its origin.** For each sampled receipt whose block
`j` is in an append `a` with an origin leaf (`originLeaf`: for `registered x`, position `x + (j − first)` in the
registration's payload tree; for `rekey k`, the same offset in the tree of the append filling slot `k`, which
`OriginsOnRota` gives every re-key of an accepted session), its opening names that position `q`, its block decodes
under the bound key of its digest and `a`'s coin to a payload, and that payload's leaf, under the opening's salt and
path, is leaf `q` under the origin's root. -/
def OpensOrigin : Prop :=
  ∀ (reg : Registration) (appends : Array Append) (w : WindowRecord) (coin : ByteArray) (samples : Nat)
    (openings : Array Opening),
    verifyWindow reg appends w coin samples openings = some .accept →
      ∃ sampled, receiptSample reg.digest w.index coin samples (w.passes.map (·.plan.length)) = some sampled ∧
        openings.size = sampled.size ∧
        ∀ k (hk : k < sampled.size) (a : Append) q oroot odom,
          let o := openings[k]!
          let j := w.passes[sampled[k].1]!.plan.block sampled[k].2
          reg.blocks ≤ j → appendOf appends j = some a →
          originLeaf reg reg.digest appends a j = some (q, oroot, odom) →
            o.originPosition = some q ∧ ∃ payload,
              Codec.decode o.block j reg.key (some (o.digest, some a.coin)) = .ok (some payload) ∧
              odom.verifyOpening oroot odom.count q (hidden payload o.payloadSalt) SCHEMA o.payloadPath = true

/-- The appends a session's results answered, in order: the results `appended first coin`, as `(first, coin)`. -/
def appendedOf (rs : Array Session.Result) : Array (Nat × ByteArray) :=
  rs.filterMap fun
    | .appended first coin => some (first, coin)
    | _ => none

/-- **An append coin leaves only in reply to the append recorded in its slot.** In a session run on its events, the
result of event `e` is `appended first c` only when event `e` is an append, which the session records as its `k`-th
(`k` the `appended` results before `e`) with that root, count, origin and slot, starting at `first`, and `c` is the
append coin of its slot, `slot.getD k` (the slot it names, or `k` when it names none, as without refresh): the coin
leaves after the append's root and origin are received, an empty slot's coin never leaves, and no other result
carries an append coin. -/
def CoinAfterAppend : Prop :=
  ∀ (params : Session.Params) (reg : Registration) (coins appendCoins : Array ByteArray) (notBeforeNs : Nat)
    (events : Array Session.Event) (rs : Array Session.Result) (s : Session.State),
    Session.run (Session.State.init params reg coins appendCoins notBeforeNs) events = .ok (rs, s) →
      rs.size = events.size ∧
      ∀ e first c, rs[e]? = some (.appended first c) →
        let k := (appendedOf (rs.extract 0 e)).size
        ∃ root count origin slot, events[e]? = some (.append root count origin slot) ∧ ∃ hk : k < s.appends.size,
          s.appends[k].root = root ∧ (s.appends[k].count : Int) = count ∧ s.appends[k].origin = origin ∧
          s.appends[k].slot = slot.getD k ∧ s.appends[k].first = first ∧ s.appends[k].coin = c ∧
          appendCoins[slot.getD k]? = some c

/-- **Coins keyed by slot.** As `CoinAfterAppend` with refresh on, where every append names its slot `σ`: the coin
answering it is member `σ` of the append-coin family, `appendCoins[σ]`, whatever appends came before; empty slots'
members never leave. -/
def CoinBySlot : Prop :=
  ∀ (params : Session.Params) (reg : Registration) (coins appendCoins : Array ByteArray) (notBeforeNs : Nat)
    (events : Array Session.Event) (rs : Array Session.Result) (s : Session.State),
    Session.run (Session.State.init params reg coins appendCoins notBeforeNs) events = .ok (rs, s) →
      params.refresh = true →
      ∀ e first c, rs[e]? = some (.appended first c) →
        let k := (appendedOf (rs.extract 0 e)).size
        ∃ root count origin σ, events[e]? = some (.append root count origin (some σ)) ∧ ∃ hk : k < s.appends.size,
          s.appends[k].slot = σ ∧ s.appends[k].first = first ∧ s.appends[k].coin = c ∧ appendCoins[σ]? = some c

end Properties.Poub
