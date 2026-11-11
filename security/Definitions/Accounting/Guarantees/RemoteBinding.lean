module

public import Definitions.Accounting.Remote.Fork
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Analysis.Complex.UpperHalfPlane.Basic
import Mathlib.Analysis.SpecialFunctions.Bernstein
import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Combinatorics.Enumerative.DyckWord
import Mathlib.Combinatorics.SimpleGraph.Triangle.Removal
import Mathlib.Data.NNRat.Floor
import Mathlib.Data.Nat.Choose.Multinomial
import Mathlib.Geometry.Euclidean.Altitude
import Mathlib.NumberTheory.Chebyshev
import Mathlib.NumberTheory.Height.NumberField
import Mathlib.NumberTheory.Height.Projectivization
import Mathlib.NumberTheory.LucasLehmer
import Mathlib.NumberTheory.SelbergSieve
import Mathlib.RingTheory.Radical.NatInt
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Mathlib.Topology.Sheaves.Init

/-!
# The receipts' binding bound (trusted)

The bound on the receipts' binding term (`Accounting.Remote.Draw.Receipts.binding`), the probability that some window
opens a receipt presented with an accepted row other than the one its root holds, which `PooledSample` and the memory
line's sampled guarantees charge. It is the forking argument: the held row is by definition the plurality of the rows the
receipt shows over the window's continuations (`Receipts.held`), so a receipt moves with probability at most the
square root of the chance that two runs from the cut on independent openings coins show it with two different accepted
rows (`Accounting.SecurityProofs.TS.Plurality`'s argument, at `cont t`). Two such runs at one root, rank and verifier
hash a SHA-512 collision (`Hash.Guarantees.RowBinding`): the fork's break (`Receipts.Broken`). Otherwise their roots
split (`Receipts.Split`). Over window `t`'s receipts, at most `s_t` opened a run, Cauchy–Schwarz over the receipts it
can open (`L_t`) gives its share. The statement has no premise beyond `L_t` holding what the window opens and the claim.

* `ReceiptsBindingOfCR`: under claim `cr/sha-512` at each window's fork finder (`Receipts.finder`), which costs `q_t`
  SHA-512 calls, the binding term is at most `Σ_t √(|L_t|·s_t·(q_t²/2^513 + split_t))`. Its step that assumes no hash,
  the fork form `ReceiptsBindingOfFork` (`Σ_t √(|L_t|·s_t·(fork_t + split_t))`), is a lemma:
  `Security/Proofs/Accounting/Lemmas/RemoteBinding.lean`.

`L_t` and `s_t` are the session's: a set holding every receipt window `t` opens at every outcome, and its draw cap, so
the code charges at the session's bounds on a window's receipts and draws, not at one record's `U_t` and `s'_t`. Where
every window's roots, ranks and verifiers are fixed by its cut, `split_t = 0` and the bound is
`Σ_t √(|L_t|·s_t)·q_t/2^256.5` (`Accounting.SecurityProofs.Remote.Binding.binding_le_of_fixed`), `ledger.binding`'s
charge. The deployed challenger sends the prover no commitment to window `t`'s coin before window `t`'s roots are in
(the coin commitment comes only with the outcome). That the roots then read only the cut is not stated here, so
`split_t` is not bounded: the gap `binding-split`.
-/

@[expose] public section

namespace Accounting.Guarantees.Remote

open Accounting.Remote.Draw

/-- **The receipts' binding term under `cr/sha-512`.** `ReceiptsBindingOfFork` with each window's fork finder
(`Receipts.finder`, at `q_t` SHA-512 calls) meeting `Claims.StrictBirthday`: the binding term is at most
`Σ_t √(|L_t|·s_t·(q_t²/2^513 + split_t))`. -/
def ReceiptsBindingOfCR : Prop :=
  ∀ {n m : ℕ} {s : Fin n → ℕ} {κ : ℕ} {X : Type} [Fintype X] [Nonempty X] (D : Session n m s κ X)
    (R : Receipts n κ X) (prf : Fin n → Pous.Bits κ → Blocks m) (rest : DrawCoins n κ X → Fin n → ℕ → ℕ)
    (L : Fin n → Finset ℕ) (q : Fin n → ℕ),
    (∀ t ω, D.opened prf rest t ω ⊆ L t) →
    (∀ t, Claims.StrictBirthday (fun u => (R.hash u).bytes) (R.finder D prf rest t (q t))) →
    (R.binding D prf rest).toReal ≤
      ∑ t, √((L t).card * s t * ((q t : ℝ) ^ 2 / 2 ^ (513 : ℝ) + (R.split D prf rest t).toReal))

end Accounting.Guarantees.Remote
