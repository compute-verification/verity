import Properties.RemoteAudit

/-!
# When the verifier reads a window's openings coin (`RemoteAudit`)

The code's half of the opening draw's premises (`Accounting.Remote.Draw.Assumptions`): what `verifyWindow` and the
session give of `Refused` and of `Live`, stated against the verifier's functions, for every input of their types.

A session's coins are `coins[0 … W − 1]`, window `t`'s challenge at `t` (`releaseCoin`), and `coins[W + t]`, window
`t`'s openings coin, which only `close t` reads, to draw the window's receipts (`draws`).

* `AcceptDrawsOpenable` (`Refused`, per window): an accepted window's draws are the sampler's values on its openings
  key, and each is a receipt of the window with an opening that passes `checkOpening` (`Openable`). So the receipts a
  window opens are among its openable ones, the model's correct receipts.
* `OpeningsUnreadUntilDrawn` (`Live`, the coin order): a run, on the same events, gives the same results and the same
  state, whatever the coins at `W + t` and above, as long as window `t` is not drawn by its end. Every challenge and
  the earlier windows' openings coins may be read; window `t`'s openings coin and the later ones are not.
* `RecordKeptAfterDraw` (`Live`, the receipts fixed at the draw): once window `t` is drawn, the rest of the session
  keeps its draws, its record (what `verify` checks the openings against: its challenge, the previous window's last
  root and its passes' roots, plans and heads, so its size `U_t`) and which of its receipts are openable, whatever
  events follow. A later append doesn't move the append a drawn receipt's block is in.

So the code gives `Live` once the events up to window `t`'s draw are fixed by the coins below `W + t`, and that is
what it cannot give: the order the deployment releases coins in, the challenger releasing `coins[W + t]` only to
`close t`'s draw and the prover seeing nothing of it before (the coin commitment's root comes only with the outcome,
`audit.Session.disclose`, so the guarantees take `Live` of the session the prover plays). `ReleaseOrder` names it, a
premise of the deployment's,
which no property here takes. `LiveUnder` is `Live` as the code states it, on the events a session sees on its
coins: two sessions on coins that agree below `W + t`, wherever each has drawn window `t`, see the same record and
the same openable receipts. `Proofs.Properties.RemoteAudit.Order` derives it from the three properties and
`ReleaseOrder` (`liveUnder`), a lemma rather than a property since its premise is the deployment's. And `Counted`
reads the trace's held blocks, which the code never sees: that a credited pass's wrong registered blocks are receipts
no opening passes is the hash binding of the receipts and the store's commitments, not a property of the verifier's
functions.

The statements need `1 ≤ W` (with no window, `start` reads `coins[0]`, window 0's openings coin) and
`1 ≤ min_passes` (with none, an empty window drawn on its deadline could take a root afterwards, once the window
before it is drawn), which `audit.py`'s `Params` and the difftest's `positive` check.
-/

open RemoteAudit

namespace Properties.RemoteAudit

/-- Receipt `u` of window `w` (its index among the window's receipts in pass order, `receiptIndex`) has an opening
that passes `checkOpening` under its pass's receipter: some pass `i` and position `p < L_i` with index `u`. -/
def Openable (reg : Registration) (appends : Array Append) (w : WindowRecord) (u : Nat) : Prop :=
  ∃ i p o, i < w.passes.size ∧ p < w.passes[i]!.plan.length ∧
    receiptIndex (w.passes.map (·.plan.length)) i p = u ∧
    checkOpening reg reg.digest appends w (receipterOf reg w i) i p o = none

/-- An accepted window's draws are the sampler's values `us` on its openings key, and each value is an openable
receipt of the window. -/
def AcceptDrawsOpenable : Prop :=
  ∀ (reg : Registration) (appends : Array Append) (w : WindowRecord) (coin : ByteArray) (samples : Nat)
    (openings : Array Opening),
    verifyWindow reg appends w coin samples openings = some .accept →
      ∃ L us, FlockSoundness.Randomness.subset (openingsKey reg.digest w.index coin)
          (min samples (w.passes.map (·.plan.length)).toList.sum) (w.passes.map (·.plan.length)).toList.sum
          [.str "openings"] L = some us ∧
        ∀ u ∈ us, Openable reg appends w u

/-- Window `t`'s draws in a session, `none` until `close t` draws them. -/
def drawnOf (s : Session.State) (t : Nat) : Option (Array (Nat × Nat)) := (s.windows[t]?).bind (·.drawn)

/-- A run from `State.init` that leaves window `t` undrawn reads no coin at `W + t` or above: on coins `coins'` that
agree with `coins` below `W + t`, the same events give the same results and the same state, with `coins'`. -/
def OpeningsUnreadUntilDrawn : Prop :=
  ∀ (params : Session.Params) (reg : Registration) (coins appendCoins : Array ByteArray) (notBeforeNs : Nat)
    (events : Array Session.Event) (rs : Array Session.Result) (s : Session.State),
    Session.run (Session.State.init params reg coins appendCoins notBeforeNs) events = .ok (rs, s) →
    ∀ (t : Nat) (coins' : Array ByteArray), 1 ≤ params.windows → drawnOf s t = none →
      (∀ j, j < params.windows + t → coins'[j]! = coins[j]!) →
      Session.run (Session.State.init params reg coins' appendCoins notBeforeNs) events =
        .ok (rs, { s with coins := coins' })

/-- Once a run from `State.init` has drawn window `t` (at `s`), any events after it keep the window's draws, its record
and its openable receipts: at `s'` they are those at `s`. -/
def RecordKeptAfterDraw : Prop :=
  ∀ (params : Session.Params) (reg : Registration) (coins appendCoins : Array ByteArray) (notBeforeNs : Nat)
    (events events' : Array Session.Event) (rs rs' : Array Session.Result) (s s' : Session.State),
    Session.run (Session.State.init params reg coins appendCoins notBeforeNs) events = .ok (rs, s) →
    Session.run s events' = .ok (rs', s') →
    ∀ (t : Nat) (d : Array (Nat × Nat)), 1 ≤ params.minPasses → drawnOf s t = some d →
      drawnOf s' t = some d ∧ s'.record t = s.record t ∧
        ∀ u, Openable reg s'.appends (s'.record t) u ↔ Openable reg s.appends (s.record t) u

/-- **The release order** (the deployment's, which the code cannot give): `D` gives the events a session sees on its
coins (the prover's and the world's choices fixed, the coin commitment among them), and window `t`'s openings coin
reaches none of them before the one that draws the window. While the first `k` events leave window `t` undrawn, the
next is the same on coins that agree below `W + t`. It is what `Live` needs beyond `OpeningsUnreadUntilDrawn`, up to the draw, and
`RecordKeptAfterDraw`, after it. -/
def ReleaseOrder (params : Session.Params) (reg : Registration) (appendCoins : Array ByteArray) (notBeforeNs : Nat)
    (D : Array ByteArray → Array Session.Event) : Prop :=
  ∀ (coins coins' : Array ByteArray) (t k : Nat) (rs : Array Session.Result) (s : Session.State),
    (∀ j, j < params.windows + t → coins'[j]! = coins[j]!) →
    Session.run (Session.State.init params reg coins appendCoins notBeforeNs) ((D coins).extract 0 k) = .ok (rs, s) →
    drawnOf s t = none → (D coins')[k]? = (D coins)[k]?

/-- **`Live`, as the code states it**, on the events `D` a session sees on its coins: two sessions on coins that agree
below `W + t`, each run to a point where window `t` is drawn, see the same record of it and the same openable
receipts. -/
def LiveUnder (params : Session.Params) (reg : Registration) (appendCoins : Array ByteArray) (notBeforeNs : Nat)
    (D : Array ByteArray → Array Session.Event) : Prop :=
  ∀ (coins coins' : Array ByteArray) (t k k' : Nat) (rs rs' : Array Session.Result) (s s' : Session.State),
    (∀ j, j < params.windows + t → coins'[j]! = coins[j]!) →
    Session.run (Session.State.init params reg coins appendCoins notBeforeNs) ((D coins).extract 0 k) = .ok (rs, s) →
    Session.run (Session.State.init params reg coins' appendCoins notBeforeNs) ((D coins').extract 0 k') =
      .ok (rs', s') →
    (drawnOf s t).isSome → (drawnOf s' t).isSome →
      s'.record t = s.record t ∧
        ∀ u, Openable reg s'.appends (s'.record t) u ↔ Openable reg s.appends (s.record t) u

end Properties.RemoteAudit
