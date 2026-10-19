import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Definitions.Accounting.Physical
import Models.Accounting.Assumptions.WipeRead

/-!
# The lifetime pool on the actual circuit: the bridge and the lag (trusted, frozen)

`Accounting.Guarantees.Actual.WipePool` bounds the un-held fractions a run tree of wipes labels (`Accounting.WipePlay`)
under `Accounting.Assumptions.TimedRoundSound`, which the labels alone make true. These state the pool on the server's
actual state instead, per read, with the premise restated on the circuit (`Models.Accounting.Assumptions.WipeRead`),
and what the state can gain between reads. Frozen statements of the MEM-55 red team: no theorem proves them yet.

The setting is `Accounting.Actual.heldAt`'s: a strategy `S` on coins `r = (ω₀, c)`, `n` timed reads over its life with
uniform indices `c : Fin n → B` on a store `B`, labels `lab ω₀ j`, read `j`'s index arriving at the server `X` at
`arrive r j`, the challenger's verdict `acc r j`, and read `j`'s un-held fraction `x_j` (`Accounting.Actual.unheldAt`)
at guessing level `2^-lam`.

* `WipeReadPool` (the bridge): `Pr[every read accepted ∧ Σ_j x_j ≥ H] ≤ exp(−H) + δ`. It is `WipePool` at `k = 1`,
  one read a wipe, with each read's `U` the blocks un-held at its arrival, so a server that drops, refetches or
  rebuilds between reads is charged at each read. With `k` reads a wipe at a steady held set, `Σ_j x_j` is `k` times
  the per-wipe sum, which is at most `ln(1/ε)/k` except with probability `ε + δ`.
* `TimedReadOfQuiet`: timed-read soundness follows at `δ = n · 2^-lam` from causality, live reads, quiet reads and a
  binding verdict, each a premise the network certifier and the challenger's code can carry.
* `WipeLag` (the lag): over any coins, the contents of `X`'s state at `t` number at most those at `s ≤ t` times
  `2^{β_X (t − s)}`, `β_X` the capacity of `X`'s boundary links. At any tick `s` between reads, the bits the state
  varies in are at least those at the next read `t` less `β_X (t − s)`: dropping between reads costs ingress, read by
  read. Nothing bounds the state between a wipe's last read and the next wipe's fill but the cap's own ingress term
  (`r ω` in `Accounting.Guarantees.Actual.ThreadedRate`).

What none of these says: that a held label occupies its `b` bits of the state. `heldAt` is information at a guessing
level, so the count of held blocks bounds how many bits the state varies in only through a compression bound on the
labels, the step that StreamedWipe's counting makes (PR #2726, not on `main`) and that an ideal-permutation premise
(C8) makes for PoUS. That bridge is open.
-/

namespace Accounting.Guarantees.Actual

open Finset

open Classical in
/-- **The lifetime pool per read, on the server's held set** (frozen, not proved). For a strategy `S` on coins
`r = (ω₀, c)` whose `n` reads are live coins (`Assumptions.LiveReads`: read `j`'s arrival tick and `X`'s state
content there are fixed by `ω₀` and the earlier indices) and meet timed-read soundness at `δ`
(`Assumptions.TimedReadSound`: on all but a `δ` share of the coins, every read the challenger accepts challenged a
block held at its arrival), and any `H`, the share of the coins on which the challenger accepts every read (`acc r j`
for each `j`) and the reads' un-held fractions (`Actual.unheldAt`, at guessing level `2^-lam`) sum to at least `H`
is at most `exp(−H) + δ`. -/
def WipeReadPool : Prop :=
  ∀ {Ω₀ B L : Type*} [Fintype Ω₀] [Fintype B] [LinearOrder L] {n : ℕ}
    (S : Ω₀ × (Fin n → B) → Actual.Run L) (X : Finset L) (lab : Ω₀ → Fin n → B → ℕ) (lam : ℕ)
    (arrive : Ω₀ × (Fin n → B) → Fin n → ℕ) (acc : Ω₀ × (Fin n → B) → Fin n → Prop) (δ H : ℝ),
    Assumptions.LiveReads S X arrive → Assumptions.TimedReadSound S X lab lam arrive acc δ →
    ((univ.filter fun r => (∀ j, acc r j) ∧ H ≤ ∑ j, Actual.unheldAt S X lab lam arrive j r).card : ℝ) ≤
      (Real.exp (-H) + δ) * Fintype.card (Ω₀ × (Fin n → B))

/-- **Timed-read soundness from quiet reads** (frozen, not proved). A strategy `S` causal from every tick
(`CausalFrom`), whose reads are live (`Assumptions.LiveReads`) and quiet for the deadline `D`
(`Assumptions.QuietReads`: what crosses into `X` during `[arrive r j, arrive r j + D)` is fixed by the arrival tick
and the index), with a verdict that accepts each answer for one label (`Assumptions.VerdictBinds`), meets timed-read
soundness at `δ = n · 2^-lam` (`Assumptions.TimedReadSound`). Among the coins whose read `j` arrives at one tick with
one content of `X`'s state, the index is uniform and apart from the labels; at each index what `X` sends during the
deadline is fixed, so the challenger accepts one label value, which a block not held takes on less than a `2^-lam`
share of them. -/
def TimedReadOfQuiet : Prop :=
  ∀ {Ω₀ B L : Type*} [Fintype Ω₀] [Fintype B] [LinearOrder L] {n : ℕ}
    (S : Ω₀ × (Fin n → B) → Actual.Run L) (X : Finset L) (lab : Ω₀ → Fin n → B → ℕ) (lam D : ℕ)
    (arrive : Ω₀ × (Fin n → B) → Fin n → ℕ) (acc : Ω₀ × (Fin n → B) → Fin n → Prop),
    CausalFrom S → Assumptions.LiveReads S X arrive → Assumptions.QuietReads S X arrive D →
    Assumptions.VerdictBinds S X lab arrive D acc →
    Assumptions.TimedReadSound S X lab lam arrive acc ((n : ℝ) / 2 ^ lam)

/-- **The lag between reads** (frozen, not proved). For a strategy `S` causal from every tick (`CausalFrom`) on links
as declared (`DeclaredLinks`: each link `e` delivers one of at most `2^{β e}` signals a tick each way, and every wire
between leaves runs on a declared link), a location `X`, ticks `s ≤ t` and any set `K` of coins, the contents of `X`'s
state at `t` over `K` (`Actual.contents`) number at most those at `s` times `2^{β_X (t − s)}`, where `β_X` is the
capacity of the links across `X`'s boundary (`linkCapacity`): what `X` holds at `t` is a function of what it held at
`s` and of what crossed into it meanwhile. -/
def WipeLag : Prop :=
  ∀ {Ω L E : Type*} [LinearOrder L] [Fintype E] (S : Ω → Actual.Run L) (ends : E → L × L) (β : E → ℕ)
    (link : ∀ r, Fin (S r).A.N → Fin (S r).A.N → E) (X : Finset L) (s t : ℕ) (K : Set Ω),
    CausalFrom S → DeclaredLinks S ends β link → s ≤ t →
    (Actual.contents S X t K).encard ≤
      (Actual.contents S X s K).encard * ((2 ^ (linkCapacity ends β X * (t - s)) : ℕ) : ℕ∞)

end Accounting.Guarantees.Actual
