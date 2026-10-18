import Definitions.Accounting.Attribution
import Definitions.Accounting.PouwWork
import Models.Accounting.Assumptions.Attribution

/-!
# PoUW's work floor on one node, an open conjecture (trusted)

A3 of CMP-12's converged model (v3.1), over `Accounting.PouwWork`'s vocabulary on the actual circuit: a `Prop` about one
strategy `S` and the verifier `P`, taken as a hypothesis by `Accounting.Guarantees.PouwLocalOff`, never an `axiom`. It
is an open conjecture, so it is not under `shape.premises`: it becomes a premise under `shape.conjectures` once it has a
falsification test that runs. The other three premises are in `Models.Accounting.Assumptions.PouwWork`.

`PouwFloorAdvice` is compute attribution's floor for a block that isn't isolated: the help that doesn't depend on the
salt is free (TT, `Pouw.Assumptions.TT`), and the rest costs the chance of finding the salt, taken as a hypothesis by
`Accounting.Guarantees.PouwBlockCredit`. It is an open conjecture too, scoped to the schemes whose noise is full-rank
and per-entry, and `verity/core/protocols/pouw/tests/test_pouw_floor_advice.py` is its falsification test.
-/

namespace Accounting.Assumptions

open Accounting.PouwWork

variable {Ω : Type} [Fintype Ω] {L : Type*} (S : Ω → Actual.Run L) (P : Verdict Ω L)

/-- **A3 the work floor, off a skip event** (v3.1's `FloorOff`; on paper at `L = 32` given a merge price, conjectured at
served η): for the marking `m`, some event `Z` of probability at most `δ` is such that at every coin outside `Z`, `Nmin`
blocks right at every coin force the node's credits outside `m`, plus `ρ` a bit of help, to at least `Wb · Nmin`.

`Z` covers a guarded circuit that is right at every coin yet skips credited gates at some. The floor is meant at a
removable marking: one whose erasure leaves the verdict and the right blocks unchanged at every coin (the converged
model's `Removable`), so that covert work earns no credit. The actual-circuit model has no erasure of gates, so the
marking is taken as given, and the premise is claimed only at markings that are removable. A factor-2 projection form
(`#unmixed + 2·#mixed ≥ floor`) is being proved as a bound that implies this one. -/
def PouwFloorOff (X : Finset L) (π : ∀ r, Fin (S r).A.N → ℕ) (m : ∀ r, Finset (Fin (S r).A.N))
    (Wb Nmin ρ : ℕ) (δ : ℝ) : Prop :=
  ∃ Z : Ω → Prop, Pouw.pr Z ≤ δ ∧
    ∀ r, ¬ Z r → Nmin ≤ goodAll S P → Wb * Nmin ≤ credits S P X π m r + ρ * help S P X r

/-- **The floor at a block whose outside holds advice about the salt** (conjectured; compute attribution's floor,
`note:20261018T0730Z-draft-attrib`): on coins split into the window's salt and the later draw, if at every draw what
the outside of the block `X` and the certifiers `cert` does during the window (`Verdict.window`, `[ts, td)`) takes at
most `A` values across salts (`Attribution.OutsideAdvice`), and the outside evaluates the noise hash at most `q` times
in the window (`OutsideQueries`, at the marking `hash`), then for the marking `m` some event `Z` of probability at most
`δ + q · A / #Ωs` is such that at every coin outside `Z`, `Nmin` blocks right at every coin force the block's own
credits outside `m` to at least `Wb · Nmin`.

It rests on TT (`Pouw.Assumptions.TT`: the floor counts post-salt work, so help that is the same at every salt is
precomputation delivered late, free at any width) and the random-oracle model of the noise hash, not on A3: A3's
`ρ · help` term survives help of nonzero width, so no union bound over A3 drops it. `δ` is the floor's failure for an
outside whose help is the same at every salt. The outside's view is a function of one of `A` advice values and its
oracle answers, and until one of its `q` calls hits the salt those answers don't depend on it, so it finds the salt
with probability at most `q · A / #Ωs`: `q · 2^(k − 512)` for a 512-bit salt and `A = 2^k`. The operands are bound
before the salt (decision 114: each operand root is hashed before the salt's release at `ts` and transmitted at `td`,
and `E₁`'s rows read `A`'s), so the prover can't steer operands against the salt.

Scoped to the schemes whose noise is full-rank and per-entry: each entry of `E₁` and `F₁` its own hash output, so the
window's noise lies in no salt-independent subspace of lower dimension (`ncp-v1`, `ncp-v1-shift24`, H1). It is false
for low-rank or structured noise: with `F = Σ_{t < r} c_t(salt) · G_t` for salt-independent `G_t`, a helper ships the
`r` products `A · G_t`, and the block finishes `A · F` in `r` multiply-adds an output instead of `k`. The Pearl schemes
(Komargodski–Weinstein noise of rank 32 or 64) are outside it. The marking `hash` is meant to mark every evaluation of
the scheme's noise hash, as `PouwFloorOff`'s `m` is meant removable. Its falsification test,
`verity/core/protocols/pouw/tests/test_pouw_floor_advice.py`, checks that each in-scope scheme's noise fills its noise
space, that every registered scheme is either in scope or excluded with its reason, and that the low-rank
counterexample is refused. An open conjecture, not under `shape.premises`. -/
def PouwFloorAdvice {Ωs Ωd : Type} [Fintype Ωs] [Fintype Ωd] {L : Type*} [LinearOrder L] [Fintype L]
    (S : Ωs × Ωd → Actual.Run L) (P : Verdict (Ωs × Ωd) L) (X cert : Finset L) (hash : ∀ r, Finset (Fin (S r).A.N))
    (π : ∀ r, Fin (S r).A.N → ℕ) (m : ∀ r, Finset (Fin (S r).A.N)) (Wb Nmin q A : ℕ) (δ : ℝ) : Prop :=
  (∀ d, Attribution.OutsideAdvice (fun s => S (s, d)) cert X P.window A) → OutsideQueries S hash cert X P.window q →
  ∃ Z : Ωs × Ωd → Prop, Pouw.pr Z ≤ δ + ((q * A : ℕ) : ℝ) / Fintype.card Ωs ∧
    ∀ r, ¬ Z r → Nmin ≤ goodAll S P → Wb * Nmin ≤ credits S P X π m r

end Accounting.Assumptions
