import Definitions.Accounting.Actual.WipeHeld

/-!
# Timed reads on the actual circuit: the lifetime pool's premises, restated (trusted)

The MEM-55 restatement of `Accounting.Assumptions.TimedRoundSound` (owner memory-accounting, frozen as statements, not
yet proved or carded). That premise reads only `Accounting.WipePlay`'s labels: at every wipe, a passing draw
challenges no block of `U`. Taking `U` as the blocks whose every challenge fails makes it true by definition, and
taking `U = ∅` makes it true of any tree, so it says nothing about a server. Here `U` is read `j`'s un-answered
blocks (`Accounting.Actual.answered`), which makes the pool a theorem with no premise but `LiveVerdicts`, and these say
what an answered block costs the server's state, per read, so a server that refetches or rebuilds between the rounds
of one wipe is covered (`Accounting.Guarantees.Actual.AnsweredOfQuiet`).

The setting: a strategy `S` on coins `r = (ω₀, c)`, `n` reads over the strategy's life with indices `c : Fin n → B`,
labels `lab ω₀ j`, read `j`'s index reaching the server `X` at `arrive r j`, and the memory challenger's verdict
`acc r j` on read `j`. The challenger is the auditor's trusted device: it reads one answer from `X` for read `j` and
accepts when that answer left `X` within the deadline `D` of the arrival and is the label, and when the network
certifier on `X`'s boundary saw nothing cross into `X` during the deadline but the index.

Each premise is to hold of every strategy, given where the auditor's devices sit and what their code does: a premise a
cheating server can break by cheating keeps that cheat out of the theorem instead of catching it.

* `LiveReads`: read `j`'s index is a live coin, released at its read. With `arrive r j` the earliest tick the index
  can reach `X` (the challenger's send tick plus the least latency of any path to `X`), this follows from causality.
* `LiveVerdicts`: read `j`'s verdict is fixed by the coins up to read `j` (the challenger draws read `j + 1`'s index
  after read `j`'s verdict).
* `QuietReads`: on every read the challenger accepts, what crossed into `X` during the deadline is what the
  challenger's code allows for that read, arrival tick and index (`quiet`).
* `VerdictBinds`: the challenger accepts each answer from `X` for at most one label.
* `UniformLabels`: the labels of a read's wipe are uniform `b`-bit values, independent across blocks. A model: the
  deployed labels are a PRF of a 32-byte key (PR #2726), for which it is false, and the step to them is the PRF's
  advantage, outside this.
-/

namespace Accounting.Assumptions

open Finset

/-- **Live reads**: read `j`'s index is the challenger's live coin, released at the read. In runs whose coins agree on
`ω₀` and on the indices of the reads before `j`, read `j` arrives at `X` at the same tick and finds the same content
of `X`'s state there. With `arrive r j` the challenger's send tick plus the least latency of any path into `X`, a run
can't break it: nothing in `X` before then depends on read `j`'s index. With `arrive r j` the tick `X` reads the index,
a prover's box on the path that reads it first and sends `X` a hint breaks it. -/
def LiveReads {Ω₀ B L : Type*} [LinearOrder L] {n : ℕ} (S : Ω₀ × (Fin n → B) → Actual.Run L) (X : Finset L)
    (arrive : Ω₀ × (Fin n → B) → Fin n → ℕ) : Prop :=
  ∀ (j : Fin n) (r r' : Ω₀ × (Fin n → B)), r.1 = r'.1 → (∀ i : Fin n, i < j → r.2 i = r'.2 i) →
    arrive r j = arrive r' j ∧ (S r).content X (arrive r j) = (S r').content X (arrive r' j)

/-- **Live verdicts** (the challenger's code): read `j`'s verdict is the same in runs whose coins agree on `ω₀` and on
the indices of the reads up to `j`. The challenger draws read `j + 1`'s index after it has decided read `j`. -/
def LiveVerdicts {Ω₀ B : Type*} {n : ℕ} (acc : Ω₀ × (Fin n → B) → Fin n → Prop) : Prop :=
  ∀ (j : Fin n) (r r' : Ω₀ × (Fin n → B)), r.1 = r'.1 → (∀ i : Fin n, i ≤ j → r.2 i = r'.2 i) →
    (acc r j ↔ acc r' j)

/-- **Quiet reads** (the network side of a timed read): on every read the challenger accepts, what crossed into `X`
during the deadline `[arrive r j, arrive r j + D)` (`Actual.Run.crossing`, with arrival ticks and sources as seen) is
`quiet j t i`, fixed by the read, its arrival tick `t` and its index `i`: the challenger's index and nothing that varies
otherwise, so no label came in from outside `X` while the read was open. It is the challenger's check of the network
certifier's report together with the certifier's coverage: the certifier on `X`'s boundary links sees every such
crossing (`Accounting.Covered`) and reports it as it is. A server that fetches during a deadline fails the read; it
doesn't break this. -/
def QuietReads {Ω₀ B L : Type*} {n : ℕ} (S : Ω₀ × (Fin n → B) → Actual.Run L) (X : Finset L)
    (arrive : Ω₀ × (Fin n → B) → Fin n → ℕ) (D : ℕ) (acc : Ω₀ × (Fin n → B) → Fin n → Prop)
    (quiet : Fin n → ℕ → B → Set (ℕ × L × ℕ × L × ℕ × ℕ × ℕ)) : Prop :=
  ∀ (j : Fin n) (r : Ω₀ × (Fin n → B)), acc r j →
    (S r).crossing (↑X)ᶜ ↑X (Set.Ico (arrive r j) (arrive r j + D)) = quiet j (arrive r j) (r.2 j)

/-- **The verdict binds one label** (the challenger's code and placement): two runs in which the challenger accepts
read `j`, with the same index and arrival tick and the same sends of `X` from the gates it placed during the deadline
(`Actual.Run.sent`), have the same label for the challenged block. It holds of every strategy when the challenger
compares the exact label and reads its answer only from what `X` sent: its input links all end in `X`, or it reads the
network certifier's record of `X`'s egress. A challenger that reads whatever reaches it is broken by a relay, a
machine outside `X` that holds the labels and answers for it. -/
def VerdictBinds {Ω₀ B L : Type*} {n : ℕ} (S : Ω₀ × (Fin n → B) → Actual.Run L) (X : Finset L)
    (lab : Ω₀ → Fin n → B → ℕ) (arrive : Ω₀ × (Fin n → B) → Fin n → ℕ) (D : ℕ)
    (acc : Ω₀ × (Fin n → B) → Fin n → Prop) : Prop :=
  ∀ (j : Fin n) (r r' : Ω₀ × (Fin n → B)), acc r j → acc r' j → r.2 j = r'.2 j → arrive r j = arrive r' j →
    (S r).sent ↑X (Set.Ico (arrive r j) (arrive r j + D)) = (S r').sent ↑X (Set.Ico (arrive r' j) (arrive r' j + D)) →
    lab r.1 j (r.2 j) = lab r'.1 j (r'.2 j)

/-- **Uniform labels** (a model of the labels): for each read `j`, any set `A` of blocks and any values `v`, the labels
of `A` at read `j` are `v` on at most a `2^{-|A| b}` share of `ω₀`. The deployed labels are SHA-512 of a 32-byte key and
the index (PR #2726), on which one label fixes the rest; this is their ideal, and the step to them is the PRF's
advantage. -/
def UniformLabels {Ω₀ B : Type*} [Fintype Ω₀] {n : ℕ} (lab : Ω₀ → Fin n → B → ℕ) (b : ℕ) : Prop :=
  ∀ (j : Fin n) (A : Finset B) (v : B → ℕ),
    ((univ.filter fun ω => ∀ i ∈ A, lab ω j i = v i).card : ℝ) * 2 ^ (A.card * b) ≤ Fintype.card Ω₀

end Accounting.Assumptions
