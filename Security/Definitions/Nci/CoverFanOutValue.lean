import Definitions.Nci.CoverFanOut
import Mathlib.Data.Finset.Union

/-!
# The cover theorem's fan-out tightening: values read whole, from wiring

`FanOutReadIntensity` prices each value a block spends at `f` bits through the named premise `ValueReadHardness`. These
are the wiring properties that let the circuit set the price instead (`Models.Nci.Guarantees.CoverFanOutValue`):

* `Keyed.pathGates`: the gates of a value's key path, its first stop included.
* `Keyed.KeyDisjoint`: distinct activations of one token have disjoint key paths. `Keyed`'s (I) and (F) imply it
  (`Keyed.keyDisjoint`), so it is not a premise.
* `Keyed.PerValueCut`: the set-cut form of a per-value width, a max-flow property of the named circuit.
-/

namespace Nci.Cover.Keyed

variable {C : Circuit} {J L Tok Val : Type*} [DecidableEq Val] {M : MulBlocks C J L} {d aOut c V f : ℕ}

open Classical in
/-- **The gates of `v`'s key path**: the gates of the values that following keys from `v` reaches, its first stop
included. -/
noncomputable def pathGates (K : Keyed M Tok Val d aOut c V f) (v : Val) : Finset (Fin C.N) :=
  (K.vals.filter fun w => Follows K.key K.stops v w).biUnion K.gates

/-- **Key-disjoint activations**: two distinct activations of one token's counted products have disjoint key paths, so
no value lies on both. Kimi K2 meets it with multiplicity 1: every key is per coordinate, and MLA's `v` is per head. -/
def KeyDisjoint (K : Keyed M Tok Val d aOut c V f) : Prop :=
  ∀ p ∈ M.P, ∀ p' ∈ M.P, K.act p ≠ K.act p' → K.tok (K.act p) = K.tok (K.act p') → ∀ u,
    Follows K.key K.stops (K.act p) u → ¬ Follows K.key K.stops (K.act p') u

/-- **The per-value cut** at `g` legs: the first stop on each counted product's activation's key path is a root, and
any set of the key path's gates that cuts every path from `g` of the root's legs to the product's output gate, inside
the key path's gates and the product's, has at least `f` gates. A block handed fewer than `f` gates of the key path
therefore holds an exit of all but `g − 1` of the root's legs, and pays for them as a block that holds the root does.

It is the set-cut form of the red team's per-value width, a max-flow problem per root on the named circuit. The form
with `f` disjoint paths from each single leg fails for `f ≥ 3` in `silicon/fp4.py`: a product's output bit
`p[j]^s` feeds one Wallace full adder, whose two outputs cut it. A compressor's two outputs cut at most three products,
so fewer than `f` gates cut off only a few legs; the red team puts `g` near `f`. It costs the bound `g − 1` of a root's
`d` legs. Kimi K2 at NVFP4 (`f = 4`): every word on a key path is at least 4 gates wide (FP32 partial sums, the BF16
residual, the 4-bit code). -/
def PerValueCut (K : Keyed M Tok Val d aOut c V f) (g : ℕ) : Prop :=
  ∀ p ∈ M.P, ∃ r ∈ K.roots, Follows K.key K.stops (K.act p) r ∧
    ∀ G ⊆ K.legs r, g ≤ G.card → ∀ Z ⊆ K.pathGates (K.act p),
      (∀ q ∈ G, ∀ h ∈ C.exits (K.gates q), ∀ π, C.IsPath π h (M.o p) →
        (∀ x ∈ π.tail, x ∈ K.pathGates (K.act p) ∪ M.m p) → ∃ z ∈ Z, z ∈ π) →
      f ≤ Z.card

end Nci.Cover.Keyed
