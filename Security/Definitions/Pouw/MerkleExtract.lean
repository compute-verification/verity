import Mathlib

/-!
# Merkle commitments over an oracle, and their extractor (trusted)

The vocabulary of `Extract`, the commit-and-open premise of the no-fold design (every committed word is a SHA-512 Merkle
leaf), and of its discharge from SHA-512 as a random oracle.

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
* `Prover X D q τ`: an adaptive prover against the random oracle in lazy-sampling form, `q` fresh queries answered by a
  uniform stream, the first `τ` before its root. `Prover.oracle` is the oracle it sees and `Prover.pre` its
  pre-commitment log.
* `ExtractFails E A u`: on the stream `u`, some opening the prover can verify from its queries disagrees with the
  extractor run on its pre-commitment log.
* `Consistent qry h u`: `u` is the answer stream the oracle `h` gives the querier `qry`, the bridge from a uniform
  oracle `X → D` to the uniform stream.
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

/-! ### The random-oracle model (lazy sampling) -/

/-- **An adaptive oracle prover** in the lazy-sampling form of the random oracle: its `q` queries, answered by the
stream `u`, the first `τ` of them before its commitment. Over a uniform oracle the answers at `q` distinct points are
`q` independent uniform values, so counting streams `u : Fin q → D` is the probability over the oracle.
* **Adaptive.** Query `i` is a function of the answers before it (`qry_dep`), and the root of the answers before the
  commitment (`root_dep`). Queries after the commitment may read every answer before them.
* **Distinct queries** (`fresh`; a normalization). A prover that queries a point twice gets the same answer twice, so
  dropping the repeat changes nothing it computes, and `q` counts distinct points. A real adversary's count is its
  number of distinct SHA-512 inputs on the commitment path.
* **`q` includes the verifier's evaluations** (a normalization, read by `ExtractFails`). Checking an opening hashes
  the points `pts E p w`: the leaf and one node per level. The prover is taken to query all of them itself, after its
  other queries. That is what the verifier does: an evaluation at a point nobody queried is a fresh uniform answer
  either way. So `q` is the prover's distinct queries plus, for each opening, its depth plus one, and it is `q`,
  not the prover's own count, that the bounds read.
* Off the queried points the oracle answers `default` (`oracle`), a value no bound reads: by the second normalization
  every point an opening's check hashes is a query. -/
structure Prover (X D : Type) (q τ : ℕ) where
  qry : Fin q → (Fin q → D) → X
  qry_dep : ∀ i u u', (∀ j : Fin q, j < i → u j = u' j) → qry i u = qry i u'
  fresh : ∀ u (i j : Fin q), j < i → qry i u ≠ qry j u
  root : (Fin q → D) → D
  root_dep : ∀ u u', (∀ j : Fin q, (j : ℕ) < τ → u j = u' j) → root u = root u'

namespace Prover

variable {q τ : ℕ} (A : Prover X D q τ)

open Classical in
/-- The oracle on the stream `u`: `u i` at the prover's `i`-th query, `default` off them. -/
noncomputable def oracle [Inhabited D] (u : Fin q → D) (x : X) : D :=
  if hx : ∃ i, A.qry i u = x then u (Classical.choose hx) else default

/-- The points queried before the commitment. -/
def pre (u : Fin q → D) : Set X := {x | ∃ i : Fin q, (i : ℕ) < τ ∧ A.qry i u = x}

end Prover

/-- **`u` is the answer stream the oracle `h` gives the querier `qry`**: each query's answer is `h` at that query. This
is how a prover that reads a uniform oracle `h : X → D` only through its queries meets the lazy-sampling form: every
`h` has such a stream, and each stream is the one of a `1/|D|^q` share of the oracles when the queries are adaptive
and distinct (`Proofs.Pouw.MerkleExtract.card_consistent_le`). -/
def Consistent {q : ℕ} (qry : Fin q → (Fin q → D) → X) (h : X → D) (u : Fin q → D) : Prop :=
  ∀ i, u i = h (qry i u)

open Classical in
/-- **The extraction fails on the stream `u`**: some opening of some word verifies against the prover's root under the
oracle, hashes only queried points, and isn't the word the extractor reads off the pre-commitment queries. "Hashes
only queried points" is the normalization that `q` includes the verifier's evaluations (`Prover`): it quantifies over
every opening whose check the run performs, once those evaluations are counted as queries. -/
def ExtractFails [Inhabited D] {q τ : ℕ} (E : Enc X D W) (A : Prover X D q τ) (u : Fin q → D) : Prop :=
  ∃ (p : List (Bool × D × D)) (w : W), Opens (A.oracle u) E (A.root u) p w ∧
    (∀ y ∈ pts E p w, ∃ i, A.qry i u = y) ∧ ext (A.oracle u) E (A.pre u) (A.root u) (p.map Prod.fst) ≠ some w

open Classical in
/-- The child digests of a point, if it is a node. -/
noncomputable def children (E : Enc X D W) [DecidableEq D] (x : X) : Finset D :=
  if hx : ∃ lr : D × D, E.node lr.1 lr.2 = x then {(Classical.choose hx).1, (Classical.choose hx).2} else ∅

end Pouw.MerkleExtract
