import Definitions.Hash.Tree
import Definitions.Hash.Hm96

/-!
# The lead's hiding premises the code can't be checked against yet (trusted; `Prop`s taken as hypotheses)

Hiding the lead's commitments with hm96 instances salted from one committed key rests on two premises about what a drawn
unit's proof checks, which no Lean function computes yet. Each is a named `Prop` whose docstring opens with its claim id;
a `(proposed)` id is not yet in `verity.core.claims`. `Pouw.Guarantees.Lead.G` reads neither: its model has no
openings and no salts, and its commitment premises are `CommitL1w` and `HonestHashRuled`.

* `SubtreeLayout`: every row a drawn unit opens climbs, under its instance's own frame-v3 domain, to an inner digest
  that the instance's registered root hides as an hm96 leaf.
* `SaltDerivation`: every salt a drawn unit's proof uses derives from a key whose hm96 leaf is committed before the
  window's salt, and from the window's salt.

A drawn unit's proof is the protocol's (`Opens`): what it establishes is a parameter, so a statement reads these at its
own acceptance relation. The commitment premises that are properties of the code are in `Properties.Pouw.Lead.Commit`:
row-bound (`LeafFirstBlock`, `RowBound`) and the round coins' order (`RoundsOpenAfterRoots`).
-/

namespace Pouw.Assumptions

open Hash

/-- `pouw-subtree-layout` (proposed). **Each hm96 instance hides one frame-v3 subtree under its own domain**: whenever
a drawn unit's accepted proof opens the value `v` at `rank` of instance `pos` through the inner digest `x` and the salt
`y` (`Opens pos rank v x y`), the instance's registered root `root pos` is the hm96 leaf over `x` and `y` under the
verifier's key, and `v`'s frame-v3 leaf at `rank` climbs to `x` under the instance's domain `dom pos`. An instance is a
call's `A` rows (its root the one the call's coins are drawn after) or a forward's words.

Rationale: a call's `A` rows are the leaves of their own subtree, in the call's domain, whose root one hm96 instance
hides, and a drawn unit's proof climbs its rows to that root and checks the instance's commit string against the
registered leaf. The commitment code has no per-instance domain, no hm96 instance over a subtree root and no seed or
record reading one, so this is a premise of the protocol until it does. With row-bound, an opened row is bound to its
instance and rank, or the checks hashed a collision (`Hash.Guarantees.TreeBinding`). -/
def SubtreeLayout (H : Msg → Digest) (key : Msg) (dom : ℕ → Tree.Domain) (schema : String) (root : ℕ → Digest)
    (Opens : ℕ → ℕ → Msg → Digest → Msg → Prop) : Prop :=
  ∀ pos rank v x y, Opens pos rank v x y →
    root pos = Hm96.leaf H key x y ∧
      ∃ sibs, Tree.accepts H (dom pos) x rank (H (Tree.leafMsg (dom pos) rank schema v)) sibs

/-- `pouw-salt-derivation` (proposed). **Every salt derives from a key committed before the window's salt**: whenever a
drawn unit's accepted proof opens instance `pos` through a salt `y` (`Opens pos rank v x y`), it opens a key `ks`
against the key's committed hm96 leaf `keyLeaf` (under some salt `y₀`), and `y` is `F₂ (F₁ ks salt) pos`: the window's
key `k_w = F₁ ks salt` and the instance's salt from `k_w` and `pos`. `keyLeaf` is fixed before the window's salt `salt`
is drawn, so no salt is chosen after a coin.

Rationale: the deployed maps are `k_w = SHA-512(ks ‖ salt)` and `y = SHA-512(k_w ‖ pos ‖ r)` for `r = 0, 1, 2`, or
`verity.ml.randomness`'s `derive` and `stream` under the same key. A drawn unit's proof opens `ks` against `keyLeaf` and
recomputes `k_w` and `y`, so those compressions are forced work. The hiding hybrid
(`Pouw.SecurityProofs.Lead.Commit.hides_derived_os`) reads the derivation as the honest committer's; soundness and the
work credit read it as the proof's check. The commitment code's hm96 salts are its own stream under the committer's
key, which no proof checks, so this is a premise of the protocol until it does. -/
def SaltDerivation (H : Msg → Digest) (key : Msg) (keyLeaf : Digest) (salt : Msg) (F₁ : Digest → Msg → Msg)
    (F₂ : Msg → ℕ → Msg) (Opens : ℕ → ℕ → Msg → Digest → Msg → Prop) : Prop :=
  ∀ pos rank v x y, Opens pos rank v x y → ∃ ks y₀, keyLeaf = Hm96.leaf H key ks y₀ ∧ y = F₂ (F₁ ks salt) pos

end Pouw.Assumptions
