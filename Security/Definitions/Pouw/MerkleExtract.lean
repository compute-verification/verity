import Mathlib

/-!
# Merkle commitments over an oracle, and their extractor (trusted)

The vocabulary of `Extract`, the commit-and-open premise of the no-fold design (every committed word is a SHA-512 Merkle
leaf), and of its discharge from SHA-512 as a random oracle (R-ACCEL-3; ABSORB-THM, CMP-132).

* `Enc`: the tree's leaf and node encodings into the oracle's domain, injective with disjoint images (frame-v3-sha512's
  domain separation, a protocol-code fact).
* `Opens h E d p w`: an opening of the word `w` along the path `p` verifies against the digest `d` under the hash `h`.
  The path is root first; each step is its direction, the sibling's digest and the path child's digest, the values the
  verifier's bottom-up recomputation passes through.
* `pts E p w`: the points that verification hashes.
* `ext h E L d dirs`: the extractor. It walks down from `d` along `dirs` through `L`, the points queried before the
  commitment, and returns the leaf word it reaches, or `none`.
* `kids E L`: the child digests named by `L`'s node points. With the root, they are the targets a later query must hit
  to open something `L` doesn't determine.
-/

namespace Pouw.MerkleExtract

/-- The tree's two encodings into the oracle's domain: injective, with disjoint images. -/
structure Enc (X D W : Type) where
  leaf : W → X
  node : D → D → X
  leaf_inj : Function.Injective leaf
  node_inj : ∀ {l r l' r' : D}, node l r = node l' r' → l = l' ∧ r = r'
  disj : ∀ w l r, leaf w ≠ node l r

variable {X D W : Type}

/-- The point a path step hashes: `c` the child on the path, `s` its sibling, `b` true when the path goes right. -/
def Enc.pt (E : Enc X D W) (b : Bool) (s c : D) : X := if b then E.node s c else E.node c s

/-- An opening verifies against `d` under `h`, root first; each step is (direction, sibling, child digest). -/
def Opens (h : X → D) (E : Enc X D W) : D → List (Bool × D × D) → W → Prop
  | d, [], w => h (E.leaf w) = d
  | d, (b, s, c) :: p, w => h (E.pt b s c) = d ∧ Opens h E c p w

/-- The points an opening's verification hashes. -/
def pts (E : Enc X D W) : List (Bool × D × D) → W → List X
  | [], w => [E.leaf w]
  | (b, s, c) :: p, w => E.pt b s c :: pts E p w

/-- The digests that appear as children in the log's node points. -/
def kids (E : Enc X D W) (L : Set X) : Set D := {c | ∃ l r, E.node l r ∈ L ∧ (c = l ∨ c = r)}

open Classical in
/-- The extractor: walk down from `d` along `dirs` through the points `L` queried before the commitment. -/
noncomputable def ext (h : X → D) (E : Enc X D W) (L : Set X) : D → List Bool → Option W
  | d, [] => if hw : ∃ w, E.leaf w ∈ L ∧ h (E.leaf w) = d then some (Classical.choose hw) else none
  | d, b :: bs =>
      if hn : ∃ lr : D × D, E.node lr.1 lr.2 ∈ L ∧ h (E.node lr.1 lr.2) = d then
        ext h E L (if b then (Classical.choose hn).2 else (Classical.choose hn).1) bs
      else none

end Pouw.MerkleExtract
