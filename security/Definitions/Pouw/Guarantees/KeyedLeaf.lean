module

public import Definitions.Pouw.KeyedLeaf
public import Definitions.Pouw.Assumptions.LeadCommit
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
# The keyed leaf hides and binds (trusted)

A window's lead commits each instance as a keyed leaf, `d = H(ks ‖ salt ‖ M_top)` under the window's key `ks` and live
salt `salt` (`Hash.Keyed`), and records the key leaf, the hm96 leaf over `ks`, before the salt is sampled.

* `Hides`: two windows' views (the key leaf and the digests, `Pouw.KeyedLeaf.view`) are indistinguishable. For any
  predicate `A` of the view, its probabilities at the tops `t` and `t'` differ by at most `A`'s advantage
  (`Pouw.KeyedLeaf.advantage`) at each, or two instances share a domain id. The advantage is two explicit
  distinguishers' advantages, each bounded by a claim: hm96's hiding game's by `statistical-hiding` (`Claims.Hm96Hiding`,
  `δ₁ ≤ 2^-193` at the pinned key, the `hash-derived-key` step, averaged over the independent key and applied to an
  event and its negation), and the keyed digest's game's by `prf/sha-512`: `ks ‖ salt` is SHA-512's first block, so the
  keyed digests are a cascade of the compression function keyed by its chaining input, over the top messages, and
  distinct domain ids at bytes 49 to 112 of each top message make them prefix-free. Both read `ks` and `y₀` as uniform:
  they are OS bytes (`uniform/io-getrandombytes`). Distinct instances are at distinct positions, and their domain ids,
  hashes of the positions, are distinct unless SHA-512 collides (`cr/sha-512`).
* `Binds`: under the premises `SubtreeLayout` and `SaltDerivation`, two values a sampled proof block's proofs open at one rank of
  one instance are equal, or the messages those openings hashed hold a collision of `H`.
-/

@[expose] public section

namespace Pouw.Guarantees.KeyedLeaf

open Hash Pouw.KeyedLeaf

/-- **A keyed window hides its tops.** For every hash `H`, verifier's key `key`, salt `salt`, `n` instances under the
domains `dom`, two assignments of top messages `t` and `t'` and predicate `A` of the view: two instances share a domain
id, or `A`'s probability at `t` is within `A`'s advantage at `t` plus its advantage at `t'` of its probability at `t'`. -/
def Hides : Prop :=
  ∀ (H : Msg → Digest) (key : Msg) (salt : Digest) (n : ℕ) (dom : Fin n → Tree.Domain) (t t' : Fin n → Keyed.Top)
    (A : Digest × (Fin n → Digest) → Prop),
    (∃ i j, i ≠ j ∧ (dom i).id = (dom j).id) ∨
      |real H key salt (tops dom t) A - real H key salt (tops dom t') A| ≤
        advantage H key salt (tops dom t) A + advantage H key salt (tops dom t') A

/-- **A keyed leaf binds its values.** Under `SubtreeLayout` and `SaltDerivation`, two values `v₁` and `v₂` that sampled
proof blocks' proofs open at `rank` of instance `pos` are equal, or the messages the openings hashed (the key leaf's under each
key, and each keyed climb's) hold a collision of `H`. -/
def Binds : Prop :=
  ∀ (H : Msg → Digest) (key : Msg) (keyLeaf salt : Digest) (dom : ℕ → Tree.Domain) (schema : String)
    (root : ℕ → Digest) (Opens : ℕ → ℕ → Msg → Digest → Msg → List Digest → Prop),
    Assumptions.SubtreeLayout H salt dom schema root Opens → Assumptions.SaltDerivation H key keyLeaf Opens →
    ∀ pos rank v₁ v₂ ks₁ ks₂ y₁ y₂ sibs₁ sibs₂, Opens pos rank v₁ ks₁ y₁ sibs₁ → Opens pos rank v₂ ks₂ y₂ sibs₂ →
      v₁ = v₂ ∨ CollisionIn H (Hm96.hashed H key ks₁ y₁ ++ Hm96.hashed H key ks₂ y₂ ++
        Keyed.hashed H ks₁ salt (dom pos) rank (Tree.leafMsg (dom pos) rank schema v₁) sibs₁ ++
        Keyed.hashed H ks₂ salt (dom pos) rank (Tree.leafMsg (dom pos) rank schema v₂) sibs₂)

end Pouw.Guarantees.KeyedLeaf
