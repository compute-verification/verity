import Definitions.Accounting.Confidence
import Definitions.Pous.Model.Audit

/-!
# A location as PoUS's responder: the vocabulary (trusted)

The actual-circuit model's obligation B2: during a PoUS audit on the actual circuit, the location `X` the challenges
reach the prover through is PoUS's sequential responder. A strategy `S` maps PoUS's outcome (the primitive `ω`, the
encoder's coins and the `k` challenges) to a run. PoUS sends each challenge when the last answer arrives, so the ticks
are read off the run: challenge `j` reaches `X` at tick `t x j` at the outcome `x`, and *window* `j` is
`[t x j, t x (j + 1))` (`window`), where `t x k` is the tick the verifier's rule turns the last answer's arrival into.
The answer to challenge `j` is read off what `X` does during window `j` (`ans j`, applied to `acts`, of type
`Actual.Acts`). The run *passes* when every window lasts at most `span` ticks (`Timely`: the memory challenger ends the
audit at the first late answer) and every answer is the challenged block (`Passes`).

The responder's state before challenge `j` is `X`'s state's content at `t x j`, a string of at most `mem_X(t_j)` bits,
coded in `M + 1` bits by `lengthCode` (its bits, a `1`, then `0`s: there are `2^(M+1) − 1` strings of length at most
`M`), a `τ`-bit code of the tick `t x j`, and a code of the help. What `X` does during a window includes its
state at the window's end, in flight too, so causality alone carries the state from one challenge tick to the next.
-/

namespace Accounting

/-- **The length code**: a string `l` of at most `M` bits as `M + 1` bits, its bits, then a `1`, then `0`s. -/
def lengthCode (M : ℕ) (l : List Bool) : Pous.Bits (M + 1) :=
  fun i => if h : i.val < l.length then l[i.val] else decide (i.val = l.length)

/-- What a location does during a set of ticks, as `Actual.Run.acts` returns it. -/
abbrev Actual.Acts (L : Type*) : Type _ :=
  Set (ℕ × L × ℕ × ℕ × ℕ) × Set (ℕ × L × ℕ × L × ℕ × ℕ × ℕ) × Set (ℕ × ℕ × L × ℕ × ℕ × ℕ)

/-- **Window `j`** of a run's challenge ticks `t`: from challenge `j`'s tick to the next one's. -/
def window (t : ℕ → ℕ) (j : ℕ) : Set ℕ := Set.Ico (t j) (t (j + 1))

/-- **Every answer on time**: each of the `k` windows of the run's challenge ticks `t` lasts at most `span` ticks. -/
def Timely (t : ℕ → ℕ) (span k : ℕ) : Prop := ∀ j < k, t (j + 1) ≤ t j + span

variable {L : Type*}

/-- **The run passes the `k`-challenge audit** at the outcome `x`: every window of the run's challenge ticks `t x`
lasts at most `span` ticks, and for every challenge `j` the answer read off what `X` does during window `j` (`ans j`)
is block `I j` of the codeword the encoder made at `x`. -/
def Passes {I : Pous.IdealModel} (sch : Pous.Scheme I) {k : ℕ} (W : Pous.Bits sch.n)
    (S : Pous.Outcome sch k → Actual.Run L) (X : Finset L) (t : Pous.Outcome sch k → ℕ → ℕ) (span : ℕ)
    (ans : ℕ → Actual.Acts L → Pous.Bits sch.ℓ) (x : Pous.Outcome sch k) : Prop :=
  Timely (t x) span k ∧ ∀ j : Fin k, ans j ((S x).acts X (window (t x) j)) = (sch.enc W x.1 x.2.1).2 (x.2.2 j)

/-- **The ticks are read off the run** (PoUS's protocol): challenge `j + 1`'s tick is a function of the help,
challenge `j`, its tick and what `X` does from it on, so the answer's arrival is `X`'s doing, with no delay from
anything outside `X`. Two outcomes that agree on those over `[t j, v)` both have `t (j + 1) = v`, or neither has. -/
def ArrivalRule {I : Pous.IdealModel} {sch : Pous.Scheme I} {k : ℕ} (S : Pous.Outcome sch k → Actual.Run L)
    (X : Finset L) (t : Pous.Outcome sch k → ℕ → ℕ) {α : Type*} (help : Pous.Outcome sch k → α) : Prop :=
  ∀ x x' (j : Fin k) v, help x = help x' → x.2.2 j = x'.2.2 j → t x j = t x' j → t x j < v →
    (S x).acts X (Set.Ico (t x j) v) = (S x').acts X (Set.Ico (t x j) v) → t x (j + 1) = v → t x' (j + 1) = v

/-- **What reaches `X` in a window is fixed by the help and the challenge**: at two outcomes with the same help,
challenge `j` and tick `t j`, what crosses into `X` from `t j` until the earlier of their next challenge ticks is the
same. -/
def IngressFixed {I : Pous.IdealModel} {sch : Pous.Scheme I} {k : ℕ} [LinearOrder L]
    (S : Pous.Outcome sch k → Actual.Run L) (X : Finset L) (t : Pous.Outcome sch k → ℕ → ℕ) {α : Type*}
    (help : Pous.Outcome sch k → α) : Prop :=
  ∀ x x' (j : Fin k), help x = help x' → x.2.2 j = x'.2.2 j → t x j = t x' j →
    (S x).crossing (↑X)ᶜ ↑X (Set.Ico (t x j) (min (t x (j + 1)) (t x' (j + 1)))) =
      (S x').crossing (↑X)ᶜ ↑X (Set.Ico (t x j) (min (t x (j + 1)) (t x' (j + 1))))

/-- **The setup fixes the start**: at two outcomes with the same primitive and encoding, `X`'s state's content at the
first challenge tick, that tick and the help are the same. -/
def SetupFixed {I : Pous.IdealModel} {sch : Pous.Scheme I} {k : ℕ} [LinearOrder L] (W : Pous.Bits sch.n)
    (S : Pous.Outcome sch k → Actual.Run L) (X : Finset L) (t : Pous.Outcome sch k → ℕ → ℕ) {α : Type*}
    (help : Pous.Outcome sch k → α) : Prop :=
  ∀ x x', x.1 = x'.1 → sch.enc W x.1 x.2.1 = sch.enc W x'.1 x'.2.1 →
    (S x).content X (t x 0) = (S x').content X (t x' 0) ∧ t x 0 = t x' 0 ∧ help x = help x'

/-- **Challenge `j`'s tick takes at most `2^τ` values** on the outcomes whose windows all last at most `span`. -/
def TicksIn {I : Pous.IdealModel} {sch : Pous.Scheme I} {k : ℕ} (t : Pous.Outcome sch k → ℕ → ℕ) (span τ : ℕ) :
    Prop :=
  ∀ j : Fin k, {v | ∃ x, Timely (t x) span k ∧ t x j = v}.encard ≤ ((2 ^ τ : ℕ) : ℕ∞)

end Accounting
