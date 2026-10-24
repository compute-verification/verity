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

What the code cannot give is the order the deployment releases coins in: that the events up to window `t`'s draw are
themselves fixed by the coins below `W + t`, the challenger releasing `coins[W + t]` only to `close t`'s draw and the
prover seeing nothing else of it. And `Counted` reads the trace's held blocks, which the code never sees: that a
credited pass's wrong registered blocks are receipts no opening passes is the hash binding of the receipts and the
store's commitments, not a property of the verifier's functions.

The statements need `1 ≤ W` (with no window, `start` reads `coins[0]`, window 0's openings coin), which `audit.py`'s
`Params` and the difftest's `positive` check (`windows`).
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

end Properties.RemoteAudit
