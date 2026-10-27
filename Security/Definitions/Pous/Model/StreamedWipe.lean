import Mathlib.Data.Nat.Choose.Basic
import Definitions.Pous.Game.Prob

/-!
# The streamed-label wipe (experimental)

The game of PoUS's streamed-label wipe, whose reference is
`verity/core/protocols/pous/streamed_wipe.py`. The memory challenger fills every memory tier of one server with `N`
labels of `b` bits, `L : Fin N → Bits b`. Once the fill is complete it reveals `k` indices `c : Fin k → Fin N`, uniform
and independent, one at a time, and accepts when every answer is its label (`Accepts`).

Before the wipe the server holds `q` bits of prior content `z`, uniform: a statement over a fixed `z` is false, since a
prover can hard-code it. Its fill is any function of `z` and the labels. When the fill ends the server holds
`σ : Bits M` inside the boundary, every tier counted, and the outside holds `ξ : Bits E`, every bit that left the
boundary during the wipe as the network certifier counts it. No label reaches the outside except through that egress.
Round `j`'s answer reads `σ`, `ξ`, the indices revealed before it, in order, and its own index. The outside can answer
too, from `ξ` and the indices it is sent, so whatever the two sides do during the challenges is a function of these.
The prover *keeps* `z` when a function of `σ`, `ξ` and the challenges returns it (`Keeps`): then `z` survives the
wipe, inside the boundary or out.

The labels are the auditor's PRF, `L = prf K` under a uniform key `K : Bits κ` (`winReal`). The ideal game draws them
uniform (`winIdeal`), and the difference is the PRF's advantage (`prfAdvantage`) against the distinguisher the game
makes of the prover (`wipeDistinguisher`): with coins `(z, c)` and labels `L`, it runs the prover and outputs whether
the prover won.

Provers are deterministic: a randomized prover is an average of deterministic ones (`Pous.pr`).
-/

namespace Pous.StreamedWipe

open scoped ENNReal

/-- A prover of the streamed-label wipe for `N` labels of `b` bits and `k` challenges, with `q` bits of prior content,
`M` bits inside the boundary when the fill ends and `E` bits of egress over the wipe. `fill z L` is what it holds
inside and what it has sent out when the fill ends; `answer σ ξ h i` is its answer to index `i` after the indices `h`;
`keep σ ξ c` is what it recovers of the prior content after the challenges `c`. -/
structure Prover (N k b q M E : ℕ) where
  fill : Bits q → (Fin N → Bits b) → Bits M × Bits E
  answer : Bits M → Bits E → List (Fin N) → Fin N → Bits b
  keep : Bits M → Bits E → (Fin k → Fin N) → Bits q

/-- The indices revealed before round `j`, in order. -/
def revealed {N k : ℕ} (c : Fin k → Fin N) (j : Fin k) : List (Fin N) :=
  (List.ofFn c).take j

/-- The challenger accepts: every round's answer is its label. -/
def Accepts {N k b q M E : ℕ} (P : Prover N k b q M E) (z : Bits q) (L : Fin N → Bits b) (c : Fin k → Fin N) : Prop :=
  ∀ j : Fin k, P.answer (P.fill z L).1 (P.fill z L).2 (revealed c j) (c j) = L (c j)

/-- The prior content survives the wipe. -/
def Keeps {N k b q M E : ℕ} (P : Prover N k b q M E) (z : Bits q) (L : Fin N → Bits b) (c : Fin k → Fin N) : Prop :=
  P.keep (P.fill z L).1 (P.fill z L).2 c = z

/-- The distinguisher the game makes of `P`: coins `(z, c)`, labels `L`, and it outputs whether `P` is accepted and
keeps `z`. -/
def wipeDistinguisher {N k b q M E : ℕ} (P : Prover N k b q M E) :
    (Bits q × (Fin k → Fin N)) × (Fin N → Bits b) → Prop :=
  fun x => Accepts P x.1.1 x.2 x.1.2 ∧ Keeps P x.1.1 x.2 x.1.2

/-- The ideal game: the probability that `P` is accepted and keeps `z`, with uniform labels. -/
noncomputable def winIdeal {N k b q M E : ℕ} (P : Prover N k b q M E) : ℝ≥0∞ :=
  pr (wipeDistinguisher P)

/-- The real game: the same, with the labels `prf K` under a uniform key `K`. -/
noncomputable def winReal {N k b q M E κ : ℕ} (prf : Bits κ → Fin N → Bits b) (P : Prover N k b q M E) : ℝ≥0∞ :=
  pr (fun x : (Bits q × (Fin k → Fin N)) × Bits κ => wipeDistinguisher P (x.1, prf x.2))

/-- The advantage of a distinguisher `D` with coins `ω` against `prf`: how much more often it outputs true on `prf K`,
`K` uniform, than on uniform labels (truncated at 0). -/
noncomputable def prfAdvantage {N b κ : ℕ} {ω : Type} [Fintype ω] (prf : Bits κ → Fin N → Bits b)
    (D : ω × (Fin N → Bits b) → Prop) : ℝ≥0∞ :=
  pr (fun x : ω × Bits κ => D (x.1, prf x.2)) - pr D

/-- The guarantee's bound at `g`: the sampling term `(g/N)^k`, the probability that `k` uniform challenges all land in
a set of at most `g` labels the prover can answer, and the compression term `k · C(N, g+1) · 2^(M+E) / 2^(q+(g+1)·b)`,
the probability that some round finds more than `g` labels answerable while `z` is kept. -/
noncomputable def bound (N k b q M E g : ℕ) : ℝ≥0∞ :=
  ((g : ℝ≥0∞) / N) ^ k + (k : ℝ≥0∞) * (N.choose (g + 1) : ℝ≥0∞) * 2 ^ (M + E) / 2 ^ (q + (g + 1) * b)

end Pous.StreamedWipe
