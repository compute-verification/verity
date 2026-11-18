module

public import Properties.Poub.Remote

/-!
# PoUB's verifier with refresh on (`Poub`)

The code's half of refresh's conditions in the game (`ResourceAccounting.RateSketch.LifetimeCappedAll`, `Seasoned`
and `ChoosesBefore`, `security/Definitions/Poub/Carrier.lean`), stated against `Session.run` from `State.init` with
`Params.refresh` on, for every input of their types. They take no premise.

* `RegisteredCapped`: no plan of a window at or past `lifetime` holds a registered block, so with `LifetimeCapped`
  every block a plan holds is read within `lifetime` windows of the one its key joined, the registration's at 0;
* `ProbationKept`: a plan of window `t` holds a block of an append only once its payload is seasoned: committed at the
  registration, or by a fresh append that joined a window `b` (`State.joined`) with `b + lifetime ≤ t`;
* `CoinAfterAppend`: an append coin leaves only in reply to its own append, after its root (a re-key's declaration) and
  its origin are received.
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
plan of window `t`: if `a` is fresh, it joined a window `b` with `b + L ≤ t`; if `a` re-keys append `k`'s blocks, `k` is
an earlier fresh append that joined a window `b` with `b + L ≤ t`; if `a` re-keys registered blocks, its payload was
committed at the registration (`receive` refuses with `probation`). The windows are the session's record
`State.joined`, the window being filled at the first pass received after each append. -/
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
            | .fresh => ∃ b, s.joined[a]? = some b ∧ b + params.lifetime ≤ t
            | .rekey k _ => k < a ∧ s.appends[k]?.map (·.origin) = some .fresh ∧
                ∃ b, s.joined[k]? = some b ∧ b + params.lifetime ≤ t

/-- The appends a session's results answered, in order: the results `appended first coin`, as `(first, coin)`. -/
def appendedOf (rs : Array Session.Result) : Array (Nat × ByteArray) :=
  rs.filterMap fun
    | .appended first coin => some (first, coin)
    | _ => none

/-- **An append coin leaves only in reply to its own append.** In a session run on its events, the result of event
`e` is `appended first c` only when event `e` is an append of a root, a count and an origin, which the session records
as its `k`-th append (`k` the `appended` results before `e`), starting at `first`, and `c` is the `k`-th append coin:
the coin is drawn after the append's root and origin are received, and no other result carries an append coin. -/
def CoinAfterAppend : Prop :=
  ∀ (params : Session.Params) (reg : Registration) (coins appendCoins : Array ByteArray) (notBeforeNs : Nat)
    (events : Array Session.Event) (rs : Array Session.Result) (s : Session.State),
    Session.run (Session.State.init params reg coins appendCoins notBeforeNs) events = .ok (rs, s) →
      rs.size = events.size ∧
      ∀ e first c, rs[e]? = some (.appended first c) →
        let k := (appendedOf (rs.extract 0 e)).size
        ∃ root count origin, events[e]? = some (.append root count origin) ∧ ∃ hk : k < s.appends.size,
          s.appends[k].root = root ∧ (s.appends[k].count : Int) = count ∧ s.appends[k].origin = origin ∧
          s.appends[k].first = first ∧ s.appends[k].coin = c ∧ appendCoins[k]? = some c

end Properties.Poub
