import Definitions.Accounting.Actual.WipeHeld

/-!
# Timed reads on the actual circuit: the lifetime pool's premises, restated (trusted)

The MEM-55 red team's restatement of `Accounting.Assumptions.TimedRoundSound` (owner memory-accounting, frozen as
statements, not yet proved or carded). That premise reads only `Accounting.WipePlay`'s labels: at every wipe, a passing
draw challenges no block of `U`. Taking `U` as the blocks whose every challenge fails makes it true by definition, and
taking `U = ∅` makes it true of any tree, so it says nothing about a server. These say it of the server's state, per
read (`Accounting.Actual.heldAt`), so a server that refetches or rebuilds between the rounds of one wipe is covered:
each read has its own held set, fixed before its index.

The setting is `Accounting.Actual.heldAt`'s: a strategy `S` on coins `r = (ω₀, c)`, `n` reads over the strategy's life
with indices `c : Fin n → B`, labels `lab ω₀ j`, read `j`'s index arriving at the server `X` at `arrive r j`, and the
memory challenger's verdict `acc r j` on read `j`. The challenger is the auditor's trusted device: it reads one answer
from `X` for read `j` and accepts when that answer left `X` within the deadline `D` of the arrival and is the label.

* `LiveReads`: read `j`'s index is a live coin, released at its read: its arrival tick and `X`'s state content there are
  fixed by `ω₀` and the earlier reads' indices.
* `TimedReadSound` (the restated premise): except with probability `δ`, every read the challenger accepts challenged a
  block held at its arrival.
* `QuietReads` and `VerdictBinds`, the premise `TimedReadSound` should follow from at `δ = n · 2^-lam`
  (`Accounting.Guarantees.Actual.TimedReadOfQuiet`): during each read's deadline nothing crosses into `X` but what the
  arrival tick and the index fix, and the challenger accepts each answer for at most one label.
-/

namespace Accounting.Assumptions

open Finset

/-- **Live reads**: read `j`'s index is the challenger's live coin, released at the read. In runs whose coins agree on
`ω₀` and on the indices of the reads before `j`, read `j` arrives at `X` at the same tick and finds the same content
of `X`'s state there. -/
def LiveReads {Ω₀ B L : Type*} [LinearOrder L] {n : ℕ} (S : Ω₀ × (Fin n → B) → Actual.Run L) (X : Finset L)
    (arrive : Ω₀ × (Fin n → B) → Fin n → ℕ) : Prop :=
  ∀ (j : Fin n) (r r' : Ω₀ × (Fin n → B)), r.1 = r'.1 → (∀ i : Fin n, i < j → r.2 i = r'.2 i) →
    arrive r j = arrive r' j ∧ (S r).content X (arrive r j) = (S r').content X (arrive r' j)

open Classical in
/-- **Timed-read soundness** (the restated premise of the lifetime pool, owner memory-accounting): except on at most a
`δ` share of the coins, every read the challenger accepts (`acc r j`) challenged a block the server held at the read's
arrival, at guessing level `2^-lam` (`Accounting.Actual.heldAt`). A server that names a label it doesn't hold, by a
guess, a fetch from outside `X` or a rebuild from what came in during the deadline, is what `δ` counts. -/
def TimedReadSound {Ω₀ B L : Type*} [Fintype Ω₀] [Fintype B] [LinearOrder L] {n : ℕ}
    (S : Ω₀ × (Fin n → B) → Actual.Run L) (X : Finset L) (lab : Ω₀ → Fin n → B → ℕ) (lam : ℕ)
    (arrive : Ω₀ × (Fin n → B) → Fin n → ℕ) (acc : Ω₀ × (Fin n → B) → Fin n → Prop) (δ : ℝ) : Prop :=
  ((univ.filter fun r => ∃ j, acc r j ∧ r.2 j ∉ Actual.heldAt S X lab lam arrive j r).card : ℝ) ≤
    δ * Fintype.card (Ω₀ × (Fin n → B))

/-- **Quiet reads** (the network side of a timed read): during the deadline `[arrive r j, arrive r j + D)` of each
read, what crosses into `X` (`Actual.Run.crossing`, with arrival ticks and sources as seen) is the same in every run
whose read `j` arrives at the same tick with the same index: the challenger's index and nothing that varies otherwise,
so no label comes in from outside `X` while the read is open. The network certifier on `X`'s boundary links sees every
such crossing. -/
def QuietReads {Ω₀ B L : Type*} {n : ℕ} (S : Ω₀ × (Fin n → B) → Actual.Run L) (X : Finset L)
    (arrive : Ω₀ × (Fin n → B) → Fin n → ℕ) (D : ℕ) : Prop :=
  ∀ (j : Fin n) (r r' : Ω₀ × (Fin n → B)), arrive r j = arrive r' j → r.2 j = r'.2 j →
    (S r).crossing (↑X)ᶜ ↑X (Set.Ico (arrive r j) (arrive r j + D)) =
      (S r').crossing (↑X)ᶜ ↑X (Set.Ico (arrive r' j) (arrive r' j + D))

/-- **The verdict binds one label** (the challenger's code): two runs in which the challenger accepts read `j`, with the
same index and arrival tick and the same sends of `X` from the gates it placed during the deadline
(`Actual.Run.sent`), have the same label for the challenged block. The verdict reads `X`'s answer from within the
deadline and accepts it for one label. -/
def VerdictBinds {Ω₀ B L : Type*} {n : ℕ} (S : Ω₀ × (Fin n → B) → Actual.Run L) (X : Finset L)
    (lab : Ω₀ → Fin n → B → ℕ) (arrive : Ω₀ × (Fin n → B) → Fin n → ℕ) (D : ℕ)
    (acc : Ω₀ × (Fin n → B) → Fin n → Prop) : Prop :=
  ∀ (j : Fin n) (r r' : Ω₀ × (Fin n → B)), acc r j → acc r' j → r.2 j = r'.2 j → arrive r j = arrive r' j →
    (S r).sent ↑X (Set.Ico (arrive r j) (arrive r j + D)) = (S r').sent ↑X (Set.Ico (arrive r' j) (arrive r' j + D)) →
    lab r.1 j (r.2 j) = lab r'.1 j (r'.2 j)

end Accounting.Assumptions
