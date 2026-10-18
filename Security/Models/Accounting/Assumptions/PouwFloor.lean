import Definitions.Accounting.Attribution
import Definitions.Accounting.PouwWork

/-!
# PoUW's work floor on one node, an open conjecture (trusted)

A3 of CMP-12's converged model (v3.1), over `Accounting.PouwWork`'s vocabulary on the actual circuit: a `Prop` about one
strategy `S` and the verifier `P`, taken as a hypothesis by `Accounting.Guarantees.PouwLocalOff`, never an `axiom`. It
is an open conjecture, so it is not under `shape.premises`: it becomes a premise under `shape.conjectures` once it has a
falsification test that runs. The other three premises are in `Models.Accounting.Assumptions.PouwWork`.

`PouwFloorAdvice` is A3 at `k` bits of advice, compute attribution's floor for a block that isn't isolated: the help
that doesn't depend on the salt is free and the rest is counted in bits, taken as a hypothesis by
`Accounting.Guarantees.PouwBlockCredit`. It is an open conjecture too.
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

/-- **A3 at `k` bits of advice** (conjectured; compute attribution's floor, `note:20261018T0730Z-draft-attrib`): on
coins split into the window's salt and the later draw, if at every draw what the outside of the block `X` and the
certifiers `cert` does during the window (`Verdict.window`, `[ts, td)`) takes at most `2^k` values across salts
(`Attribution.OutsideAdvice`), then
for the marking `m` some event `Z` of probability at most `2^k · δ` is such that at every coin outside `Z`, `Nmin`
blocks right at every coin force the block's own credits outside `m` to at least `Wb · Nmin`.

It is A3 with the help that is the same at every salt free (pre-salt precomputation, the weights, prompts and
activations that don't depend on the salt) and the rest counted as `k` bits of advice: one fixed outside behaviour per
value of the advice, each a strategy whose help doesn't depend on the salt, so the floor at `δ` for each and a union
bound give `2^k · δ`. At `k = 0` it frees every crossing that is the same at every salt, where `PouwFloorOff` without
its `ρ` term frees none. An open conjecture, not under `shape.premises`. -/
def PouwFloorAdvice {Ωs Ωd : Type} [Fintype Ωs] [Fintype Ωd] {L : Type*} [LinearOrder L] [Fintype L]
    (S : Ωs × Ωd → Actual.Run L) (P : Verdict (Ωs × Ωd) L) (X cert : Finset L) (π : ∀ r, Fin (S r).A.N → ℕ)
    (m : ∀ r, Finset (Fin (S r).A.N)) (Wb Nmin k : ℕ) (δ : ℝ) : Prop :=
  (∀ d, Attribution.OutsideAdvice (fun s => S (s, d)) cert X P.window k) →
  ∃ Z : Ωs × Ωd → Prop, Pouw.pr Z ≤ 2 ^ k * δ ∧
    ∀ r, ¬ Z r → Nmin ≤ goodAll S P → Wb * Nmin ≤ credits S P X π m r

end Accounting.Assumptions
