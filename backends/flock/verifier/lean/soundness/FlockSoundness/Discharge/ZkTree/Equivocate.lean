import FlockSoundness.Discharge.ZkTree.Hybrid

/-!
# The mixture's salts are collisions

`zk_session_sound_tree_at` asks `cr/sha-512` (`TableCRZ`) of `τ`'s finders at every view, and no single finder that
samples its own view from the honest distribution can stand in for them. `TableCRZ`'s finders
(`StrictCR.TreeRewind.finder`, `StrictCR.twoRuns`) run the session strategy twice after the table's first message, each
run on the game's own fresh coins. The strategy is `τ.play v`: `τ` shown the leaves `ℓ (q j)` of independent blocks and
salts `q`, which is the hybrid's ideal end (`mixOf`), and, at each opening of round `j` with the game's block `b`, a
salt drawn from the posterior `postK ℓ q j b` of `q j`'s leaf given `b`. A finder may draw `q` itself, uniform, which is
the honest distribution of the blocks and salts the leaves are made of. But to answer an opening it must draw from
`postK` at the game's `b`, which is `(q j).1` with probability `1/|Blk|` only, and at any other block a salt the
posterior can draw is a collision:

* `postK_opens`: a salt `postK ℓ q j b` can draw opens `q j`'s leaf at `b`, when some salt does.
* `leaf_collision`: two openings of one hm96 leaf `H (dig j b + M·y, cs y)` are a collision of the leaf hash `H`, of
  the salt hash `cs`, or of the round digest `dig`.
* `posterior_collision`: so at a block `b ≠ (q j).1`, the posterior's salt either opens nothing (no salt opens the leaf
  at `b`) or is a collision of `H`, `cs` or `dig`.

At hm96-sha512 all three are SHA-512, so the salt the finder needs is itself a SHA-512 collision, of the kind
`cr/sha-512` says no efficient finder finds. The obstruction is the rewinding against a committed coin tree: the
tree is statistically hiding (2^-256 a leaf, `treeCoins_le_model`), which is what makes the mixture exist, and
computationally binding (`cs`), which is what makes its posterior unsamplable. A uniform reduction needs an efficient
sampler of `postK`, an equivocation of the leaf, for instance a trapdoor in the key that a reduction drawing the key
could keep.
-/

namespace FlockSoundness.Discharge.ZkTree

open Discharge.ZkCoins

variable {I Blk Salt B Cs D : Type} [Fintype Salt]

/-- **A salt the posterior can draw opens the leaf**, when some salt opens it at the block. -/
theorem postK_opens (ℓ : I → Blk × Salt → D) (q : I → Blk × Salt) (j : I) (b : Blk) (y : Salt)
    (hy : postK ℓ q j b y ≠ 0) (hf : ∃ y', ℓ j (b, y') = ℓ j (q j)) : ℓ j (b, y) = ℓ j (q j) := by
  sorry

/-- **Two openings of one leaf are a collision**: of the leaf hash `H`, of the salt hash `cs`, or of the round's
digest `dig`. -/
theorem leaf_collision [AddCommGroup B] (dig : I → Blk → B) (My : Salt → B) (cs : Salt → Cs) (H : B × Cs → D) (j : I)
    {z z' : Blk × Salt} (h : leaf dig My cs H j z = leaf dig My cs H j z') (hne : z ≠ z') :
    (∃ u v, u ≠ v ∧ H u = H v) ∨ (z.2 ≠ z'.2 ∧ cs z.2 = cs z'.2) ∨
      (z.2 = z'.2 ∧ z.1 ≠ z'.1 ∧ dig j z.1 = dig j z'.1) := by
  sorry

/-- **The mixture's salt at another block is a collision**: at a block `b` other than `q j`'s, a salt the posterior
of `q j`'s leaf can draw opens nothing, or is a collision of `H`, of `cs`, or of `dig`. -/
theorem posterior_collision [AddCommGroup B] (dig : I → Blk → B) (My : Salt → B) (cs : Salt → Cs) (H : B × Cs → D)
    (q : I → Blk × Salt) (j : I) (b : Blk) (y : Salt) (hy : postK (leaf dig My cs H) q j b y ≠ 0)
    (hb : b ≠ (q j).1) :
    (∀ y', leaf dig My cs H j (b, y') ≠ leaf dig My cs H j (q j)) ∨ (∃ u v, u ≠ v ∧ H u = H v) ∨
      (y ≠ (q j).2 ∧ cs y = cs (q j).2) ∨ (y = (q j).2 ∧ dig j b = dig j (q j).1) := by
  sorry

end FlockSoundness.Discharge.ZkTree
