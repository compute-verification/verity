import Definitions.Accounting.Remote.Fork

/-!
# The receipts' binding bound (trusted)

The bound on the receipts' binding term (`Accounting.Remote.Draw.Receipts.binding`), the probability that some window
opens a receipt presented with an accepted row other than the one its root holds, which `PooledDraw` and the memory
line's drawn guarantees charge. It is the forking argument: the held row is by definition the plurality of the rows the
receipt shows over the window's continuations (`Receipts.held`), so a receipt moves with probability at most the
square root of the chance that two runs from the cut on independent openings coins show it with two different accepted
rows (`Accounting.SecurityProofs.TS.Plurality`'s argument, at `cont t`). Two such runs at one root, rank and verifier
hash a SHA-512 collision (`Hash.Guarantees.RowBinding`): the fork's break (`Receipts.Broken`). Otherwise their roots
split (`Receipts.Split`). Over window `t`'s receipts, at most `s_t` opened a run, Cauchy–Schwarz over the receipts it
can open (`L_t`) gives its share. Neither statement has a premise beyond `L_t` holding what the window opens.

* `ReceiptsBindingOfFork`: the fork form, assuming no hash. The binding term is at most
  `Σ_t √(|L_t|·s_t·(fork_t + split_t))`.
* `ReceiptsBindingOfCR`: the same under claim `cr/sha-512` at each window's fork finder (`Receipts.finder`), which
  costs `q_t` SHA-512 calls: `Σ_t √(|L_t|·s_t·(q_t²/2^513 + split_t))`.

`L_t` and `s_t` are the session's: a set holding every receipt window `t` opens at every outcome, and its draw cap, so
the code charges at the session's bounds on a window's receipts and draws, not at one record's `U_t` and `s'_t`. Where
every window's roots, ranks and verifiers are fixed by its cut, `split_t = 0` and the bound is
`Σ_t √(|L_t|·s_t)·q_t/2^256.5` (`Accounting.SecurityProofs.Remote.Binding.binding_le_of_fixed`), `ledger.binding`'s
charge. The deployed challenger sends the prover no commitment to window `t`'s coin before window `t`'s roots are in
(the coin commitment comes only with the outcome), the design note:20261025T0115Z-report-bind-close asks for; that
the roots then read only the cut is not stated here, so `split_t` is not bounded: the gap `binding-split`.
-/

namespace Accounting.Guarantees.Remote

open Accounting.Remote.Draw

/-- **The receipts' binding term, in its fork form.** For a session `D` on the first blocks `prf` and fallbacks `rest`,
its receipts `R`, and sets `L_t` holding every receipt window `t` opens, the binding term is at most
`Σ_t √(|L_t|·s_t·(fork_t + split_t))`: `fork_t` the chance that two runs from window `t`'s cut on independent openings
coins hash a SHA-512 collision between the checks of one receipt's two openings (`Receipts.fork`), `split_t` the chance
that their roots split (`Receipts.split`). -/
def ReceiptsBindingOfFork : Prop :=
  ∀ {n m : ℕ} {s : Fin n → ℕ} {κ : ℕ} {X : Type} [Fintype X] (D : Session n m s κ X) (R : Receipts n κ X)
    (prf : Fin n → Pous.Bits κ → Blocks m) (rest : Coins n κ X → Fin n → ℕ → ℕ) (L : Fin n → Finset ℕ),
    (∀ t ω, D.opened prf rest t ω ⊆ L t) →
    (R.binding D prf rest).toReal ≤
      ∑ t, √((L t).card * s t * ((R.fork D prf rest t).toReal + (R.split D prf rest t).toReal))

/-- **The receipts' binding term under `cr/sha-512`.** `ReceiptsBindingOfFork` with each window's fork finder
(`Receipts.finder`, at `q_t` SHA-512 calls) meeting `Claims.StrictBirthday`: the binding term is at most
`Σ_t √(|L_t|·s_t·(q_t²/2^513 + split_t))`. -/
def ReceiptsBindingOfCR : Prop :=
  ∀ {n m : ℕ} {s : Fin n → ℕ} {κ : ℕ} {X : Type} [Fintype X] [Nonempty X] (D : Session n m s κ X)
    (R : Receipts n κ X) (prf : Fin n → Pous.Bits κ → Blocks m) (rest : Coins n κ X → Fin n → ℕ → ℕ)
    (L : Fin n → Finset ℕ) (q : Fin n → ℕ),
    (∀ t ω, D.opened prf rest t ω ⊆ L t) →
    (∀ t, Claims.StrictBirthday (fun u => (R.hash u).bytes) (R.finder D prf rest t (q t))) →
    (R.binding D prf rest).toReal ≤
      ∑ t, √((L t).card * s t * ((q t : ℝ) ^ 2 / 2 ^ (513 : ℝ) + (R.split D prf rest t).toReal))

end Accounting.Guarantees.Remote
