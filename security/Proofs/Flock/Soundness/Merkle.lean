module

public import Proofs.Flock.Soundness.MerklePair
public import Mathlib.Tactic.Basic

/-!
# Merkle binding

**Binding** (`opening_binding`): two openings of one position that both verify against the same
cap (`Definitions.Flock.Merkle`) but carry different columns yield an explicit collision of `H`. This is the step both
reductions to collision resistance use: the straight-line one (a prover that knows its table)
and the rewinding one (every prover, `Rewinding`). The development states it for our tree shape
directly; VCVio proves the same fact for its inductive trees
(`InductiveMerkleTree.getPutativeRootWithHash_binding_collision`).

`opening_bindingL` is `opening_binding` for any leaf scheme (`Leaf`), salted ones included.

Each of the three is the corollary of the lemma that names the two colliding strings
(`Proofs.Flock.Soundness.MerklePair`): `walkPair_spec`, `openPair_spec` and `openPairL_spec` at the leaf scheme's own
pair (`Leaf.pair`).
-/

@[expose] public section

namespace FlockSoundness.Merkle

variable {Col D : Type}

variable (H : List UInt8 → D) (E : Enc Col D)

/-- Two walks from different nodes that meet give a collision: `walkPair`'s (`walkPair_spec`). -/
theorem walk_binding (sibs sibs' : List D) (idx : ℕ) (a a' : D) (hlen : sibs.length = sibs'.length) (hne : a ≠ a')
    (heq : walk H E idx a sibs = walk H E idx a' sibs') : Collision H :=
  ⟨_, _, walkPair_spec H E sibs sibs' idx a a' hlen hne heq⟩

/-- **Merkle binding.** Two openings of one position that verify against one cap with different
columns yield a collision of `H`: `openPair`'s (`openPair_spec`). -/
theorem opening_binding (cap : ℕ → D) (depth pos : ℕ) (c c' : Col) (sibs sibs' : List D)
    (hc : c ≠ c') (h : Verifies H E cap depth pos c sibs) (h' : Verifies H E cap depth pos c' sibs') :
    Collision H :=
  ⟨_, _, openPair_spec H E cap depth pos c c' sibs sibs' hc h h'⟩

variable {S : Type}

theorem verifiesL_plain (cap : ℕ → D) (depth pos : ℕ) (column : Col) (sibs : List D) :
    VerifiesL H E (Leaf.plain H E) cap depth pos column () sibs ↔ Verifies H E cap depth pos column sibs :=
  Iff.rfl

/-- **Merkle binding, for any leaf scheme.** Two openings of one position that verify against one cap with different
columns, under any salts, yield a collision of `H`: `openPairL`'s at the scheme's own pair (`openPairL_spec`). -/
theorem opening_bindingL (Lf : Leaf Col S D H) (cap : ℕ → D) (depth pos : ℕ) (c c' : Col) (s s' : S)
    (sibs sibs' : List D) (hc : c ≠ c') (h : VerifiesL H E Lf cap depth pos c s sibs)
    (h' : VerifiesL H E Lf cap depth pos c' s' sibs') : Collision H :=
  ⟨_, _, openPairL_spec H E Lf Lf.pair Lf.pair_spec cap depth pos c c' s s' sibs sibs' hc h h'⟩

end FlockSoundness.Merkle
