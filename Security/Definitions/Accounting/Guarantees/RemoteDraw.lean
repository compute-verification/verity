import Definitions.Accounting.Assumptions.RemoteDraw
import Definitions.Accounting.Assumptions.RemoteHeld

/-!
# The openings' pooled bound (trusted)

The memory line's `LiveCoins` premise (`Accounting.Remote.Held.Assumptions.LiveCoins`), derived from the verifier's
opening draw as it runs it (`Accounting.Remote.Draw`): a session of windows `D`, each window's size and correct receipts
fixed before its own openings coin (`Live`), whose opened receipts an accepted session found correct or moved off the
rows their roots hold (`Refused`, at the receipts `R`), and whose wrong receipts count the trace's (`Counted`). The
prover is sent each coin as it is released, and the challenger's commitment to the session's coins only once the
session has its outcome (`coins.Released`, `audit.Session.disclose`), so the session it plays is `D` itself: nothing it
holds before window `t`'s openings coin is released reads that coin. Window `t`'s openings are `Key.subset` on the
first blocks `prf t` gives its uniform coin, the sampler exact on each draw's first word and the fallbacks `rest` free;
`prf` is any function, the deployed one being SHA-512's keyed stream under
`derive(coin_t, OPENINGS_DOMAIN, {registration, window: t})`.

* **PooledDraw**: the probability that the session is accepted with more wrong receipts than its own record's pooled
  bound (`Session.pooled δ`, `audit.pooled`'s `wrong`) is at most `δ + Σ_t s_t / 2^64 + Σ_t adv_t + bind`, `adv_t` the
  advantage on `prf t` of window `t`'s distinguisher (`Session.distinguisher prf δ t`). The bound is read at the
  outcome, so it holds however the prover sizes each window after the earlier windows' coins, at any number of passes
  a window. On uniform first blocks window `t`'s draw lands on correct receipts with probability at most
  `(g_t/U_t)^(s'_t)` given the earlier windows, those multiply over the session, and by weighted AM–GM their product is
  at most `δ` once the wrong receipts pass the bound. The hybrid over windows trades window `t`'s uniform blocks for
  `prf t` at the cost of `s_t / 2^64` (some first word rejected) and `adv_t`. `bind` (`Receipts.binding`) is the
  probability that some opened receipt moved, presented and accepted at its root and rank with a row other than the one
  its root holds (`Receipts.held`, the plurality of the rows it shows over the window's continuations): off it, every
  receipt `Refused` lets through is correct.
* **LiveCoinsDrawn**: `LiveCoins W` at that error for every `W` at least the pooled bound on every accepted outcome.
* **HeldDrawn**, **HeldOverTimeDrawn**, **HeldOverTimeLinksDrawn**: `Held` and its time-held forms with `LiveCoins`
  replaced by the draw's premises and `W` by the record's own pooled bound at error `δc`, so the line the ledger reads
  at the record's `audit.pooled` is the one stated.

The advantages are C3's terms (claim `prf/sha-512`, `verity.ml.randomness`'s keyed streams). Each is against one
distinguisher a window, which the session fixes: it draws the other coins and the earlier windows' coins, runs the
session's prover and verifier through window `t`, computing the earlier windows' streams itself, and reads
`s'_t ≤ s_t` first blocks of the stream it is given, so a PRF distinguisher with `s_t` queries whose time is the
session's. A session whose prover is efficient has an efficient distinguisher, and one whose prover isn't gets no bound
from C3, rightly: a prover that held exactly the first openings under every 256-bit key of window `t`'s stream would
pass the window with almost all its receipts wrong. No term is the coin commitment's: the prover holds none of it while
the session runs. The binding term is bounded by `ReceiptsBindingOfCR` (`RemoteBinding`), under the collision
resistance of SHA-512 (claim `cr/sha-512`) for window `t`'s fork finder (`Receipts.finder`): it runs the prover twice
from window `t`'s cut on independent openings coins, checks both runs' openings, and outputs a collision when some
receipt opened in both carries different accepted rows at one root. The held row being the plurality, receipt `p`'s
chance of moving is at most the square root of the chance that the two runs show it with different rows, so window
`t`'s share of the term is at most `√(|L_t|·s_t·(q_t²/2^513 + split_t))`, `q_t` the finder's SHA-512 calls and
`split_t` the chance that the two runs' roots differ (`Receipts.split`). With roots fixed by the cut, `split_t = 0` and
the bound has `ledger.binding`'s form, `Σ_t √(|L_t|·s_t)·q_t/2^256.5`, which the code reads at the session's bounds on
a window's receipts and draws. The prover sends window `t`'s roots before it holds anything of window `t`'s openings
coin, but that the receipts' roots read only the cut is not a premise here (the gap `binding-split`), and
`ledger.binding` takes `split_t` as 0.
-/

namespace Accounting.Guarantees.Remote

open Accounting.Remote.Held Accounting.Remote.Held.Assumptions Accounting.Remote.Draw
  Accounting.Remote.Draw.Assumptions

/-- **The openings' pooled bound.** On a session whose windows' sizes and correct receipts are fixed before their
openings coins (`Live`), whose accepted outcomes found every opened receipt correct or moved off the row its root holds
(`Refused`, at the receipts `R`) and count the trace's wrong receipts among the windows' (`Counted`), with every
window's draw on the first blocks `prf t` its coin gives and `m < 2^448` receipts a window at most, the probability that
the session is accepted with more wrong receipts than the record's pooled bound at `δ` is at most `δ + Σ_t s_t / 2^64`,
plus the windows' distinguishers' advantages on `prf`, plus the receipts' binding term (`R.binding`: some opened receipt
moved off the row its root holds). -/
def PooledDraw : Prop :=
  ∀ (n m : ℕ) (k : Fin n → ℕ) (κ : ℕ) (X : Type) [Fintype X] (prf : Fin n → Pous.Bits κ → Blocks m)
    (rest : Coins n κ X → Fin n → ℕ → ℕ) (D : Session n m k κ X) (T : Trace (Coins n κ X)) (R : Receipts n κ X)
    (δ : ℝ), 0 ≤ δ → m < 2 ^ 448 → Live D → Refused D prf rest T R → Counted D T →
    T.Holds (δ + ((∑ t, k t : ℕ) : ℝ) / 2 ^ 64 +
        (∑ t, Pous.StreamedWipe.prfAdvantage (prf t) (D.distinguisher prf δ t)).toReal +
        (R.binding D prf rest).toReal) fun ω =>
      (T.wrong ω : ℝ) ≤ D.pooled δ ω

/-- **Live coins, derived.** On a session meeting `PooledDraw`'s premises, `LiveCoins W` at `PooledDraw`'s error holds
for every `W` at least the record's pooled bound at `δ` on every accepted outcome. -/
def LiveCoinsDrawn : Prop :=
  ∀ (n m : ℕ) (k : Fin n → ℕ) (κ : ℕ) (X : Type) [Fintype X] (prf : Fin n → Pous.Bits κ → Blocks m)
    (rest : Coins n κ X → Fin n → ℕ → ℕ) (D : Session n m k κ X) (T : Trace (Coins n κ X)) (R : Receipts n κ X)
    (W δ : ℝ), 0 ≤ δ → m < 2 ^ 448 → Live D → Refused D prf rest T R → Counted D T →
    (∀ ω, T.accepts ω → D.pooled δ ω ≤ W) →
    LiveCoins T W (δ + ((∑ t, k t : ℕ) : ℝ) / 2 ^ 64 +
      (∑ t, Pous.StreamedWipe.prfAdvantage (prf t) (D.distinguisher prf δ t)).toReal +
      (R.binding D prf rest).toReal)

/-- **The snapshot held at the record's pooled bound, under Identity.** `Held` on a trace over a session's coins, with
`LiveCoins` replaced by the draw's premises (`Live`, `Refused`, `Counted`) and `W` by the record's pooled bound at
`δc`. -/
def HeldDrawn : Prop :=
  ∀ (n m : ℕ) (k : Fin n → ℕ) (κ : ℕ) (X : Type) [Fintype X] (prf : Fin n → Pous.Bits κ → Blocks m)
    (rest : Coins n κ X → Fin n → ℕ → ℕ) (D : Session n m k κ X) (T : Trace (Coins n κ X)) (R : Receipts n κ X)
    (S s₀ e : ℕ) (s E β Br Bd δs δc : ℝ),
    0 < E → 0 ≤ s → 0 ≤ β → 0 ≤ δc → m < 2 ^ 448 →
    Identity T → SquaringCeiling T s E Br → Disks T β Bd → ChipCeiling T S → Routed T → Causal T s₀ e →
    SketchBound T δs → Live D → Refused D prf rest T R → Counted D T →
    T.Holds (δs + (δc + ((∑ t, k t : ℕ) : ℝ) / 2 ^ 64 +
        (∑ t, Pous.StreamedWipe.prfAdvantage (prf t) (D.distinguisher prf δc t)).toReal +
        (R.binding D prf rest).toReal)) fun ω =>
      (T.planned ω : ℝ) - D.pooled δc ω - ((s / E + β) * ((e : ℝ) - s₀) + (Br + Bd)) - T.passes * S ≤ T.held ω

/-- **Held over time at the record's pooled bound, under Identity.** `HeldOverTime` with `LiveCoins` replaced by the
draw's premises and `W` by the record's pooled bound at `δc`. -/
def HeldOverTimeDrawn : Prop :=
  ∀ (n m : ℕ) (k : Fin n → ℕ) (κ : ℕ) (X : Type) [Fintype X] (prf : Fin n → Pous.Bits κ → Blocks m)
    (rest : Coins n κ X → Fin n → ℕ → ℕ) (D : Session n m k κ X) (T : Trace (Coins n κ X)) (R : Receipts n κ X)
    (K : Pace (Coins n κ X)) (S : ℕ) (s E β Br Bd δs δc : ℝ),
    0 < E → 0 ≤ s → 0 ≤ β → 0 ≤ δc → m < 2 ^ 448 →
    Identity T → SquaringCeiling T s E Br → Disks T β Bd → ChipCeiling T S → Routed T → Paced T K → Covers T →
    SketchBound T δs → Live D → Refused D prf rest T R → Counted D T →
    T.Holds (δs + (δc + ((∑ t, k t : ℕ) : ℝ) / 2 ^ 64 +
        (∑ t, Pous.StreamedWipe.prfAdvantage (prf t) (D.distinguisher prf δc t)).toReal +
        (R.binding D prf rest).toReal)) fun ω =>
      ((K.stop ω - K.start ω : ℕ) : ℝ) * ((T.blocks : ℝ) - S)
          - 2 * K.longest ω * ((s / E + β) * (((K.stop ω - K.start ω : ℕ) : ℝ) + 2 * K.longest ω) + (Br + Bd))
          - K.fill ω T.blocks (D.pooled δc ω)
        ≤ T.heldOver ω (K.start ω) (K.stop ω)

/-- **Held over time with the links charged, at the record's pooled bound** (no Identity). `HeldOverTimeLinks` with
`LiveCoins` replaced by the draw's premises and `W` by the record's pooled bound at `δc`. -/
def HeldOverTimeLinksDrawn : Prop :=
  ∀ (n m : ℕ) (k : Fin n → ℕ) (κ : ℕ) (X : Type) [Fintype X] (prf : Fin n → Pous.Bits κ → Blocks m)
    (rest : Coins n κ X → Fin n → ℕ → ℕ) (D : Session n m k κ X) (T : Trace (Coins n κ X)) (R : Receipts n κ X)
    (K : Pace (Coins n κ X)) (S : ℕ) (s E β βl Br Bd Bl δs δc : ℝ),
    0 < E → 0 ≤ s → 0 ≤ β → 0 ≤ βl → 0 ≤ δc → m < 2 ^ 448 →
    Links T βl Bl → SquaringCeiling T s E Br → Disks T β Bd → ChipCeiling T S → Routed T → Paced T K → Covers T →
    SketchBound T δs → Live D → Refused D prf rest T R → Counted D T →
    T.Holds (δs + (δc + ((∑ t, k t : ℕ) : ℝ) / 2 ^ 64 +
        (∑ t, Pous.StreamedWipe.prfAdvantage (prf t) (D.distinguisher prf δc t)).toReal +
        (R.binding D prf rest).toReal)) fun ω =>
      ((K.stop ω - K.start ω : ℕ) : ℝ) * ((T.blocks : ℝ) - S)
          - 2 * K.longest ω *
              ((s / E + β + βl) * (((K.stop ω - K.start ω : ℕ) : ℝ) + 2 * K.longest ω) + (Br + Bd + Bl))
          - K.fill ω T.blocks (D.pooled δc ω)
        ≤ T.heldOver ω (K.start ω) (K.stop ω)

end Accounting.Guarantees.Remote
