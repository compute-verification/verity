module

public import Definitions.Hash.Keyed
public import Definitions.Hash.Hm96
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
# The lead's commitment premises the code can't be checked against yet (trusted; `Prop`s taken as hypotheses)

The lead's per-call commitments are keyed leaves (`Hash.Keyed`): instance `pos`'s public digest is
`H(ks ‖ salt ‖ M_top)` over its frame-v3 subtree's top message, under the window's secret key `ks` and live salt
`salt`, and the window's key leaf `keyLeaf`, the hm96 leaf over `ks` under 192 OS salt bytes, is recorded with the
challenger before `salt` is sampled. Binding them rests on two premises about what a sampled proof block's proof checks, which no
Lean function computes yet. Each is a named `Prop` whose docstring opens with its claim id.

* `SubtreeLayout`: every value a sampled proof block opens climbs, under its instance's own frame-v3 domain and the window's key,
  to the instance's registered digest.
* `SaltDerivation`: the key a sampled proof block's proof keys that climb with is the one the window's key leaf commits.

A sampled proof block's proof is the protocol's (`Opens pos rank v ks y₀ sibs`: it opens the value `v` at `rank` of instance
`pos` under the key `ks`, the key leaf's salt `y₀` and the path `sibs`): what it establishes is a parameter, so a
statement reads these at its own acceptance relation. `Pouw.Guarantees.KeyedLeafBinds` reads both.
-/

@[expose] public section

namespace Pouw.Assumptions

open Hash

/-- **`pouw-subtree-layout`: each instance's digest is the keyed leaf of its own frame-v3 subtree.** Whenever a sampled
proof block's accepted proof opens the value `v` at `rank` of instance `pos` under the key `ks` and the path `sibs`
(`Opens pos rank v ks y₀ sibs`), `v`'s frame leaf at `rank` under the instance's domain `dom pos` climbs along `sibs`,
keyed by `ks` and the window's salt `salt` at the top, to the instance's registered digest `root pos`
(`Keyed.accepts`). An instance is a call's `A` rows or a forward's words.

Rationale: a call's `A` rows are the leaves of their own subtree, in the call's domain, whose top message the window's
key hashes, and a sampled proof block's proof climbs its rows to that digest. The proof's keyed read, a gate Definition, is not
yet in the Lean, so this is a premise of the protocol until it is. With it and `SaltDerivation`, an opened value is
bound to its instance and rank, or the checks hashed a collision (`Pouw.Guarantees.KeyedLeafBinds`). -/
def SubtreeLayout (H : Msg → Digest) (salt : Digest) (dom : ℕ → Tree.Domain) (schema : String) (root : ℕ → Digest)
    (Opens : ℕ → ℕ → Msg → Digest → Msg → List Digest → Prop) : Prop :=
  ∀ pos rank v ks y₀ sibs, Opens pos rank v ks y₀ sibs →
    Keyed.accepts H ks salt (dom pos) (root pos) rank (Tree.leafMsg (dom pos) rank schema v) sibs

/-- **`pouw-salt-derivation`: every digest is keyed by the key committed before the window's salt.** Whenever a sampled
proof block's accepted proof opens instance `pos` under the key `ks` and the salt `y₀` (`Opens pos rank v ks y₀ sibs`), the
window's key leaf `keyLeaf` is the hm96 leaf over `ks` under `y₀` and the verifier's key. `keyLeaf` is recorded with the
challenger before the window's salt is sampled, so no key is chosen after it.

Rationale: the window's key `ks` is 64 OS bytes, and its hm96 leaf under 192 OS salt bytes is recorded in the window's
registration (`verity.core.protocols.pouw.salt.lead_record`) before the salt. A sampled proof block's proof opens `ks` against
`keyLeaf` and keys its climb with it, so that compression is forced work. The proof's opening of `ks` is not yet in the
Lean, so this is a premise of the protocol until it is. -/
def SaltDerivation (H : Msg → Digest) (key : Msg) (keyLeaf : Digest)
    (Opens : ℕ → ℕ → Msg → Digest → Msg → List Digest → Prop) : Prop :=
  ∀ pos rank v ks y₀ sibs, Opens pos rank v ks y₀ sibs → keyLeaf = Hm96.leaf H key ks y₀

end Pouw.Assumptions
