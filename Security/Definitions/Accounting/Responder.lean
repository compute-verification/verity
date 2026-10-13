import Definitions.Accounting.Confidence
import Definitions.Pous.Model.Audit

/-!
# A location as PoUS's responder: the vocabulary (trusted)

Obligation B2 of `note:console/20261012T0645Z-draft-actual-circuit-model` (v2.6): during a PoUS audit on the actual
circuit, the location `X` the challenges reach the prover through is PoUS's sequential responder. A strategy `S` maps
PoUS's outcome (the primitive `ω`, the encoder's coins and the `k` challenges) to a run. Challenge `j` reaches `X` at
tick `t j`, and *window* `j` is `[t j, t (j + 1))` (`window`). The answer to challenge `j` is read off what `X` does
during window `j` (`ans j`, applied to `acts`, of type `Actual.Acts`), and the run *passes* when every answer is the
challenged block (`Passes`).

The responder's state before challenge `j` is `X`'s state's content at `t j`, a string of at most `mem_X(t_j)` bits,
coded in `M + 1` bits by `lengthCode` (its bits, a `1`, then `0`s: there are `2^(M+1) − 1` strings of length at most
`M`), together with a code of the help. Since v2.6 what `X` does during a window includes its state at the window's
end, in flight too, so causality alone carries the state from one challenge tick to the next.
-/

namespace Accounting

/-- **The length code**: a string `l` of at most `M` bits as `M + 1` bits, its bits, then a `1`, then `0`s. -/
def lengthCode (M : ℕ) (l : List Bool) : Pous.Bits (M + 1) :=
  fun i => if h : i.val < l.length then l[i.val] else decide (i.val = l.length)

/-- What a location does during a set of ticks, as `Actual.Run.acts` returns it. -/
abbrev Actual.Acts (L : Type*) : Type _ :=
  Set (ℕ × L × ℕ × ℕ × ℕ) × Set (ℕ × L × ℕ × L × ℕ × ℕ × ℕ) × Set (ℕ × ℕ × L × ℕ × ℕ × ℕ)

/-- **Window `j`** of a challenge schedule `t`: from challenge `j`'s tick to the next one's. -/
def window (t : ℕ → ℕ) (j : ℕ) : Set ℕ := Set.Ico (t j) (t (j + 1))

variable {L : Type*}

/-- **The run passes the `k`-challenge audit** at the outcome `x`: for every challenge `j`, the answer read off what `X`
does during window `j` (`ans j`) is block `I j` of the codeword the encoder made at `x`. -/
def Passes {I : Pous.IdealModel} (sch : Pous.Scheme I) {k : ℕ} (W : Pous.Bits sch.n)
    (S : Pous.Outcome sch k → Actual.Run L) (X : Finset L) (t : ℕ → ℕ) (ans : ℕ → Actual.Acts L → Pous.Bits sch.ℓ)
    (x : Pous.Outcome sch k) : Prop :=
  ∀ j : Fin k, ans j ((S x).acts X (window t j)) = (sch.enc W x.1 x.2.1).2 (x.2.2 j)

end Accounting
