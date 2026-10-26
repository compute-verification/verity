import Definitions.Accounting.Actual.WipeHeld

/-!
# Timed reads on the actual circuit: the lifetime pool's premises, restated (trusted)

The MEM-55 restatement of the lifetime pool's premise (owner memory-accounting; the guarantees that take these premises
are proved in `Proofs/Accounting/Actual/WipeReadPool.lean` and `WipeBridge.lean`, not yet carded). The lemma
`Accounting.SecurityProofs.Actual.WipePool` takes `WipePlay.UnheldFails`, which reads only `Accounting.WipePlay`'s
labels: at every wipe, a passing draw challenges no block of `U`. Taking `U` as the blocks whose every challenge fails
makes it true by definition, and taking `U = ∅` makes it true of any tree, so it says nothing about a server. Here `U`
is read `j`'s un-answered blocks (`Accounting.Actual.answered`), which makes the pool a theorem with no premise but
`LiveVerdicts`, and these say what an answered block costs the server's state, per read, so a server that refetches or
rebuilds between the rounds of one wipe is covered (`Accounting.Guarantees.Actual.AnsweredOfQuiet`).

The setting: a strategy `S` on coins `r = (ω₀, c)`, `n` reads over the strategy's life with indices `c : Fin n → B`,
labels `lab ω₀ j`, read `j`'s index reaching the server `X` at `arrive r j`, and the memory challenger's verdict
`acc r j` on read `j`. The challenger is the auditor's trusted device: it reads `X`'s answers for read `j` as the
network certifier on `X`'s egress (the leaves `Y`) saw them leave `X`, and accepts when those it saw at least `ℓ⁺`
after the arrival and within the deadline `D` are the label, and when the certifier on `X`'s boundary saw nothing
cross into `X` during the deadline but the index.

The re-clocking requirement: the certifier re-clocks read `j`'s index into `X` at a fixed offset from the challenger's
send, so `arrive r j` is a tick both devices know, and holds every other ingress of `X` until the window closes. Both
windows below are keyed by that tick: `QuietReads` needs the re-clocking to hold of jittery hardware, and
`LateAnswers` needs the challenger to know `arrive r j + ℓ⁺`.

Each premise is to hold of every strategy, given where the auditor's devices sit and what their code does: a premise a
cheating server can break by cheating keeps that cheat out of the theorem instead of catching it.

* `LiveReads`: read `j`'s index is a live coin, released at its read. With `arrive r j` at or before the earliest tick
  the index can reach `X` (the certifier's re-clocked delivery), this follows from causality.
* `LiveVerdicts`: read `j`'s verdict is fixed by the coins up to read `j` (the challenger draws read `j + 1`'s index
  after read `j`'s verdict).
* `QuietReads`: on every read the challenger accepts, what crossed into `X` during the deadline is what the
  challenger's code allows for that read, arrival tick and index, and the auditor's devices' own coins `ζ` (`quiet`).
* `LateAnswers` (the acceptance rule): the challenger accepts read `j` only when the answers it saw leave `X` during
  `[arrive r j + ℓ⁺, arrive r j + D)` are at least one and each the label. Under a latency ceiling of `ℓ⁺` from `X`
  to `Y` (`Accounting.Placed.LatencyAtMost`, the listed claim `latency-ceiling`) each such answer left `X` during the
  deadline, so one label sent before the arrival, or relayed from outside `X`, binds no verdict.
* `UniformLabels`: the labels of a read's wipe are uniform `b`-bit values, independent across blocks, given the
  devices' coins `ζ`. A model: the deployed labels are a PRF of the challenger's secret (PR #2726), for which it is
  false. `Accounting.Guarantees.Actual.WipeReadHeldPrf` takes the step to them, as the PRF's advantage, and doesn't
  take this.

`QuietReads`, `LateAnswers` and `LiveVerdicts` are what the challenger's code checks or does, stated on a verdict
`acc` read off the coins; once the challenger has a Lean function they are its refusals, proved of it, and the world
facts left are the certifier's coverage and records (`Accounting.Covered`, `Accounting.TrustedUnits`) and the latency
ceiling of `X`'s certified egress (note:20261019T2210Z-report-timed-round-redteam-3).
-/

namespace Accounting.Assumptions

open Finset

/-- **Live reads**: read `j`'s index is the challenger's live coin, released at the read. In runs whose coins agree on
`ω₀` and on the indices of the reads before `j`, read `j` arrives at `X` at the same tick and finds the same content
of `X`'s state there. With `arrive r j` the challenger's send tick plus the least latency of any path into `X`, or any
tick before it, a run can't break it: nothing in `X` before then depends on read `j`'s index. With `arrive r j` the
tick `X` reads the index, a prover's box on the path that reads it first and sends `X` a hint breaks it. -/
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
`quiet j t i z`, fixed by the read, its arrival tick `t`, its index `i` and the auditor's devices' own coins
`z = ζ ω₀` (the certifier's and the challenger's jitter): the challenger's index and nothing that varies otherwise, so
no label came in from outside `X` while the read was open. It is the challenger's check of the network certifier's
report together with the certifier's coverage: the certifier on `X`'s boundary links sees every such crossing
(`Accounting.Covered`) and reports it as it is. A server that fetches during a deadline fails the read; it doesn't
break this.

The coins `ζ` are what lets an honest deployment meet it: the index's arrival tick, the certifier's cells and anything
it forwards vary run to run, and an equality with no `ζ` holds of no jittery hardware. They must be the devices' own,
independent of the labels (`UniformLabels` at the same `ζ`): the certifier re-clocks the index into `X` at a fixed
offset from the challenger's send and holds every other ingress of `X` until the window closes, so nothing upstream of
it sets what `X` sees. Allowing `Q` crossings a read instead costs `log₂ Q` bits a block, and a switch upstream that
copied the fill gets them, by delaying each index by bits of its label. -/
def QuietReads {Ω₀ B L Z : Type*} {n : ℕ} (S : Ω₀ × (Fin n → B) → Actual.Run L) (X : Finset L)
    (arrive : Ω₀ × (Fin n → B) → Fin n → ℕ) (D : ℕ) (acc : Ω₀ × (Fin n → B) → Fin n → Prop) (ζ : Ω₀ → Z)
    (quiet : Fin n → ℕ → B → Z → Set (ℕ × L × ℕ × L × ℕ × ℕ × ℕ)) : Prop :=
  ∀ (j : Fin n) (r : Ω₀ × (Fin n → B)), acc r j →
    (S r).crossing (↑X)ᶜ ↑X (Set.Ico (arrive r j) (arrive r j + D)) = quiet j (arrive r j) (r.2 j) (ζ r.1)

/-- **Late answers** (the challenger's acceptance rule and placement): the challenger reads `X`'s answers at leaves `Y`
apart from `X`, the network certifier on `X`'s egress, and accepts read `j` only when what crossed from `X` into `Y`
during `[arrive r j + ℓ⁺, arrive r j + D)` (`Actual.Run.crossing`, at the readers' ticks) is not empty and each of it
carries the label of the challenged block.

With a latency ceiling of `ℓ⁺` from `X` to `Y` (`Placed.LatencyAtMost`), every such answer left `X` during
`[arrive r j, arrive r j + D)`, so it is among what `X` did in the window (`Actual.Run.sent`) and the verdict binds one
label to what `X` sent then. Answers that land before `arrive r j + ℓ⁺` are ignored, and the deadline loses `ℓ⁺`: a
label `X` sent before the arrival lands there, and red team 3's pre-send (note:20261019T2210Z-report-timed-round-redteam-3,
H2) is refused. A relay, a machine outside `X` that holds the labels, answers on wires that don't come from `X`. A
challenger that accepts when some answer is the label is broken by `X` sending several guesses, and the deployed one
(`streamed_wipe.Challenger.receive`) compares whatever 64 bytes reach it. -/
def LateAnswers {Ω₀ B L : Type*} {n : ℕ} (S : Ω₀ × (Fin n → B) → Actual.Run L) (X Y : Finset L)
    (lab : Ω₀ → Fin n → B → ℕ) (arrive : Ω₀ × (Fin n → B) → Fin n → ℕ) (ℓmax D : ℕ)
    (acc : Ω₀ × (Fin n → B) → Fin n → Prop) : Prop :=
  Disjoint X Y ∧ ∀ (j : Fin n) (r : Ω₀ × (Fin n → B)), acc r j →
    ((S r).crossing ↑X ↑Y (Set.Ico (arrive r j + ℓmax) (arrive r j + D))).Nonempty ∧
    ∀ p ∈ (S r).crossing ↑X ↑Y (Set.Ico (arrive r j + ℓmax) (arrive r j + D)), p.2.2.2.2.2.2 = lab r.1 j (r.2 j)

open Classical in
/-- **Uniform labels** (a model of the labels): for each read `j`, any set `A` of blocks, any values `v` and any value
`z` of the auditor's devices' own coins `ζ`, the labels of `A` at read `j` are `v` on at most a `2^{-|A| b}` share of
the `ω₀` with `ζ ω₀ = z`: uniform, and independent of the devices' jitter. With `ζ` constant it is the labels' marginal.
The deployed labels are SHA-512 of a 32-byte key and the index (PR #2726), on which one label fixes the rest; this is
their ideal, and the step to them is the PRF's advantage against the distinguisher that runs the system once and
outputs `Accounting.Guarantees.Actual.WipeReadHeld`'s event. `Accounting.Guarantees.Actual.WipeReadHeldPrf` states
that step, and its ideal world, a uniform label table, meets this. -/
def UniformLabels {Ω₀ B Z : Type*} [Fintype Ω₀] {n : ℕ} (lab : Ω₀ → Fin n → B → ℕ) (b : ℕ) (ζ : Ω₀ → Z) : Prop :=
  ∀ (j : Fin n) (A : Finset B) (v : B → ℕ) (z : Z),
    ((univ.filter fun ω => ζ ω = z ∧ ∀ i ∈ A, lab ω j i = v i).card : ℝ) * 2 ^ (A.card * b) ≤
      (univ.filter fun ω => ζ ω = z).card

end Accounting.Assumptions
