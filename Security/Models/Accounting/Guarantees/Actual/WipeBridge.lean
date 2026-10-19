import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Definitions.Accounting.Physical
import Models.Accounting.Assumptions.WipeRead

/-!
# The lifetime pool on the actual circuit: the pool, the count and the lag (trusted, frozen)

`Accounting.Guarantees.Actual.WipePool` bounds the un-held fractions a run tree of wipes labels (`Accounting.WipePlay`)
under `Accounting.Assumptions.TimedRoundSound`, which the labels alone make true. These state the pool on the
challenger's actual verdicts instead, per read, what an answered block costs the server's state at the read, and what
the state can gain between reads. Frozen statements of the MEM-55 red team: no theorem proves them yet.

The setting is `Models.Accounting.Assumptions.WipeRead`'s: a strategy `S` on coins `r = (ω₀, c)`, `n` timed reads over
its life with uniform indices `c : Fin n → B` on a store `B`, labels `lab ω₀ j` of `b` bits, read `j`'s index reaching
the server `X` at `arrive r j`, the challenger's verdict `acc r j`, and read `j`'s un-answered fraction `x_j`
(`Accounting.Actual.unanswered`).

* `WipeReadPool` (the pool): `Pr[every read accepted ∧ Σ_j x_j ≥ H] ≤ exp(−H)`, under live verdicts alone. It is
  `WipePool` at `k = 1`, one read a wipe, with each read's `U` its un-answered blocks: the premise `TimedRoundSound`
  holds of that `U` by definition, so this is arithmetic, not physics.
* `AnsweredOfQuiet` (the count, the physics): at any read, the coins on which `X`'s content at the arrival is one of a
  set `C` of contents, at a tick of a set `T`, and the challenger would accept at least `a` of the indices, are at most
  a `|T| |C| C(|B|, a) 2^{-a b}` share. Each accepted index's label is fixed by the content, the tick and the index, so
  `a` answered blocks in `log₂ |C|` bits of content are `a` labels guessed, at `b − log₂ (e|B|/a)` bits a block.
* `WipeLag` (the lag): over any coins, the contents of `X`'s state at `t` number at most those at `s ≤ t` times
  `2^{β_X (t − s)}`, `β_X` the capacity of `X`'s boundary links. At any tick `s` between reads, the bits the state
  varies in are at least those at the next read `t` less `β_X (t − s)`: dropping between reads costs ingress, read by
  read. Nothing bounds the state between a wipe's last read and the next wipe's fill but the cap's own ingress term
  (`r ω` in `Accounting.Guarantees.Actual.ThreadedRate`).

Together, for a server whose content at every read is among `C`: every read passes with probability at most
`exp(−n (1 − (a − 1)/|B|)) + n |T| |C| C(|B|, a) 2^{-a b}`, with `a` the least count whose second term is small. This
is the shape of PoUS's continuous audit (`Pous.Guarantees.ContinuousAudit`: a bad history per round by counting the
memory at the send time, then the misses against `Σ q`), and of StreamedWipe's bound (PR #2726, not on `main`).
-/

namespace Accounting.Guarantees.Actual

open Finset

open Classical in
/-- **The lifetime pool per read, on the challenger's verdicts** (frozen, not proved). For verdicts `acc` on coins
`r = (ω₀, c)` with `n` uniform indices on `B`, each fixed by the coins up to its own read (`Assumptions.LiveVerdicts`),
and any `H`, the share of the coins on which the challenger accepts every read and the reads' un-answered fractions
(`Actual.unanswered`) sum to at least `H` is at most `exp(−H)`. -/
def WipeReadPool : Prop :=
  ∀ {Ω₀ B : Type*} [Fintype Ω₀] [Fintype B] {n : ℕ} (acc : Ω₀ × (Fin n → B) → Fin n → Prop) (H : ℝ),
    Assumptions.LiveVerdicts acc →
    ((univ.filter fun r => (∀ j, acc r j) ∧ H ≤ ∑ j, Actual.unanswered acc j r).card : ℝ) ≤
      Real.exp (-H) * Fintype.card (Ω₀ × (Fin n → B))

/-- **What an answered block costs** (frozen, not proved). A strategy `S` causal from every tick (`CausalFrom`), whose
reads are live (`Assumptions.LiveReads`) and quiet for the deadline `D` when accepted (`Assumptions.QuietReads`), with
a verdict that accepts each answer from `X` for one label (`Assumptions.VerdictBinds`) and uniform `b`-bit labels
(`Assumptions.UniformLabels`): for any read `j`, set `T` of ticks, set `C` of contents and count `a`, the share of the
coins on which read `j` arrives at a tick of `T`, `X`'s content there is in `C`, and at least `a` blocks are answered
(`Actual.answered`) is at most `|T| |C| C(|B|, a) 2^{-a b}`. Given the tick and the content, an index the challenger
would accept had its window's crossing fixed, so what `X` sent then, so its label: `a` answered blocks are `a`
labels equal to values the tick and content fix. -/
def AnsweredOfQuiet : Prop :=
  ∀ {Ω₀ B L : Type*} [Fintype Ω₀] [Fintype B] [LinearOrder L] {n : ℕ}
    (S : Ω₀ × (Fin n → B) → Actual.Run L) (X : Finset L) (lab : Ω₀ → Fin n → B → ℕ) (b D a : ℕ)
    (arrive : Ω₀ × (Fin n → B) → Fin n → ℕ) (acc : Ω₀ × (Fin n → B) → Fin n → Prop)
    (quiet : Fin n → ℕ → B → Set (ℕ × L × ℕ × L × ℕ × ℕ × ℕ)) (T : Finset ℕ) (C : Finset (List Bool))
    (j : Fin n),
    CausalFrom S → Assumptions.LiveReads S X arrive → Assumptions.QuietReads S X arrive D acc quiet →
    Assumptions.VerdictBinds S X lab arrive D acc → Assumptions.UniformLabels lab b →
    ((univ.filter fun r => arrive r j ∈ T ∧ (S r).content X (arrive r j) ∈ C ∧
        a ≤ (Actual.answered acc j r).card).card : ℝ) ≤
      T.card * C.card * ((Fintype.card B).choose a : ℝ) / 2 ^ (a * b) * Fintype.card (Ω₀ × (Fin n → B))

/-- **The lag between reads** (frozen, not proved). For a strategy `S` causal from every tick (`CausalFrom`) on links
as declared (`DeclaredLinks`: each link `e` delivers one of at most `2^{β e}` signals a tick each way, and every wire
between leaves runs on a declared link), a location `X`, ticks `s ≤ t` and any set `K` of coins, the contents of `X`'s
state at `t` over `K` (`Actual.contents`) number at most those at `s` times `2^{β_X (t − s)}`, where `β_X` is the
capacity of the links across `X`'s boundary (`linkCapacity`): what `X` holds at `t` is a function of what it held at
`s` and of what crossed into it meanwhile. It is `Accounting.SecurityProofs.Actual.acts_encard_le_links`'s count
(`Proofs/Accounting/Actual/Transducer.lean`) on `X`'s state. -/
def WipeLag : Prop :=
  ∀ {Ω L E : Type*} [LinearOrder L] [Fintype E] (S : Ω → Actual.Run L) (ends : E → L × L) (β : E → ℕ)
    (link : ∀ r, Fin (S r).A.N → Fin (S r).A.N → E) (X : Finset L) (s t : ℕ) (K : Set Ω),
    CausalFrom S → DeclaredLinks S ends β link → s ≤ t →
    (Actual.contents S X t K).encard ≤
      (Actual.contents S X s K).encard * ((2 ^ (linkCapacity ends β X * (t - s)) : ℕ) : ℕ∞)

end Accounting.Guarantees.Actual
