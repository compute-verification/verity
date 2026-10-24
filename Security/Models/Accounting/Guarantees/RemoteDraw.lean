import Models.Accounting.Assumptions.RemoteDraw
import Models.Accounting.Assumptions.RemoteHeld

/-!
# The openings' pooled bound (trusted)

The memory line's `LiveCoins` premise (`Accounting.Remote.Held.Assumptions.LiveCoins`), derived from the verifier's
opening draw (`Accounting.Remote.Draw`): a session of windows, each fixed before its own openings coin (`Live`), whose
opened receipts an accepted session found correct (`Refused`), and whose wrong receipts count the trace's (`Counted`).

* **PooledDraw**: the probability that the session is accepted with more wrong receipts than its own record's pooled
  bound (`Session.pooled δ`, `audit.pooled`'s `wrong`) is at most `δ`. The bound is read at the outcome, so it holds
  however the prover sizes each window after the earlier windows' openings, at any number of passes a window: window
  `t`'s draw lands on correct receipts with probability at most `(g_t/U_t)^(s'_t)` given the earlier windows, those
  multiply over the session, and by weighted AM–GM their product is at most `δ` once the wrong receipts pass the bound.
* **LiveCoinsDrawn**: `LiveCoins W δ` for every `W` at least the pooled bound on every accepted outcome.
* **HeldDrawn**, **HeldOverTimeDrawn**, **HeldOverTimeLinksDrawn**: `Held` and its time-held forms with `LiveCoins`
  replaced by the draw's premises and `W` by the record's own pooled bound at error `δc`, so the line the ledger reads
  at the record's `audit.pooled` is the one stated.
-/

namespace Accounting.Guarantees.Remote

open Accounting.Remote.Held Accounting.Remote.Held.Assumptions Accounting.Remote.Draw
  Accounting.Remote.Draw.Assumptions

/-- **The openings' pooled bound.** On a session whose windows' sizes and correct receipts are fixed before their
openings (`Live`), whose accepted outcomes found every opened receipt correct (`Refused`) and count the trace's wrong
receipts among the windows' (`Counted`), the probability that the session is accepted with more wrong receipts than the
record's pooled bound at `δ` is at most `δ`. -/
def PooledDraw : Prop :=
  ∀ (n m : ℕ) (k : Fin n → ℕ) (X : Fin n → Type) [∀ t, Fintype (X t)] (D : Session n m k X)
    (T : Trace (Coins n m k X)) (δ : ℝ), 0 ≤ δ →
    Live D → Refused D T → Counted D T →
    T.Holds δ fun ω => (T.wrong ω : ℝ) ≤ D.pooled δ ω

/-- **Live coins, derived.** On a session meeting `PooledDraw`'s premises, `LiveCoins W δ` holds for every `W` at least
the record's pooled bound at `δ` on every accepted outcome. -/
def LiveCoinsDrawn : Prop :=
  ∀ (n m : ℕ) (k : Fin n → ℕ) (X : Fin n → Type) [∀ t, Fintype (X t)] (D : Session n m k X)
    (T : Trace (Coins n m k X)) (W δ : ℝ), 0 ≤ δ →
    Live D → Refused D T → Counted D T → (∀ ω, T.accepts ω → D.pooled δ ω ≤ W) →
    LiveCoins T W δ

/-- **The snapshot held at the record's pooled bound, under Identity.** `Held` on a trace over a session's coins, with
`LiveCoins` replaced by the draw's premises (`Live`, `Refused`, `Counted`) and `W` by the record's pooled bound at
`δc`. -/
def HeldDrawn : Prop :=
  ∀ (n m : ℕ) (k : Fin n → ℕ) (X : Fin n → Type) [∀ t, Fintype (X t)] (D : Session n m k X)
    (T : Trace (Coins n m k X)) (S s₀ e : ℕ) (s E β Br Bd δs δc : ℝ),
    0 < E → 0 ≤ s → 0 ≤ β → 0 ≤ δc →
    Identity T → SquaringCeiling T s E Br → Disks T β Bd → ChipCeiling T S → Routed T → Causal T s₀ e →
    SketchBound T δs → Live D → Refused D T → Counted D T →
    T.Holds (δs + δc) fun ω =>
      (T.planned ω : ℝ) - D.pooled δc ω - ((s / E + β) * ((e : ℝ) - s₀) + (Br + Bd)) - T.passes * S ≤ T.held ω

/-- **Held over time at the record's pooled bound, under Identity.** `HeldOverTime` with `LiveCoins` replaced by the
draw's premises and `W` by the record's pooled bound at `δc`. -/
def HeldOverTimeDrawn : Prop :=
  ∀ (n m : ℕ) (k : Fin n → ℕ) (X : Fin n → Type) [∀ t, Fintype (X t)] (D : Session n m k X)
    (T : Trace (Coins n m k X)) (K : Pace (Coins n m k X)) (S : ℕ) (s E β Br Bd δs δc : ℝ),
    0 < E → 0 ≤ s → 0 ≤ β → 0 ≤ δc →
    Identity T → SquaringCeiling T s E Br → Disks T β Bd → ChipCeiling T S → Routed T → Paced T K → Covers T →
    SketchBound T δs → Live D → Refused D T → Counted D T →
    T.Holds (δs + δc) fun ω =>
      ((K.stop ω - K.start ω : ℕ) : ℝ) * ((T.blocks : ℝ) - S)
          - 2 * K.longest ω * ((s / E + β) * (((K.stop ω - K.start ω : ℕ) : ℝ) + 2 * K.longest ω) + (Br + Bd))
          - K.fill ω T.blocks (D.pooled δc ω)
        ≤ T.heldOver ω (K.start ω) (K.stop ω)

/-- **Held over time with the links charged, at the record's pooled bound** (no Identity). `HeldOverTimeLinks` with
`LiveCoins` replaced by the draw's premises and `W` by the record's pooled bound at `δc`. -/
def HeldOverTimeLinksDrawn : Prop :=
  ∀ (n m : ℕ) (k : Fin n → ℕ) (X : Fin n → Type) [∀ t, Fintype (X t)] (D : Session n m k X)
    (T : Trace (Coins n m k X)) (K : Pace (Coins n m k X)) (S : ℕ) (s E β βl Br Bd Bl δs δc : ℝ),
    0 < E → 0 ≤ s → 0 ≤ β → 0 ≤ βl → 0 ≤ δc →
    Links T βl Bl → SquaringCeiling T s E Br → Disks T β Bd → ChipCeiling T S → Routed T → Paced T K → Covers T →
    SketchBound T δs → Live D → Refused D T → Counted D T →
    T.Holds (δs + δc) fun ω =>
      ((K.stop ω - K.start ω : ℕ) : ℝ) * ((T.blocks : ℝ) - S)
          - 2 * K.longest ω *
              ((s / E + β + βl) * (((K.stop ω - K.start ω : ℕ) : ℝ) + 2 * K.longest ω) + (Br + Bd + Bl))
          - K.fill ω T.blocks (D.pooled δc ω)
        ≤ T.heldOver ω (K.start ω) (K.stop ω)

end Accounting.Guarantees.Remote
