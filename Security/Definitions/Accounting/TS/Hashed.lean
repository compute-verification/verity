module

public import Definitions.Accounting.TS.Drawn
public import Definitions.Hash.RowLeaf
public import Definitions.Core.Interaction

/-!
# A sampled window's openings as frame-v3-sha512 paths (trusted)

`Drawn`'s openings with the hash underneath. A `SampledLayer` compares the values accepted proofs open (`opened`) with no
commitment under them. Here each position is an hm96 row (`Hash.RowLeaf`: `hm96-sha512/row-seg/v1` or
`hm96-sha512/row/v2`) of a frame-v3-sha512 tree, and an accepted proof's opening of a slot is an hm96 opening of its
position's leaf (the row's bits, its salt and the path's siblings), as Flock's knowledge extractor takes it from the
proof's witness: the rows are hidden, so the verifier sees none of them. The hash `H` is a parameter, SHA-512 the
instance of record, one hash for the frame tree, the hm96 leaf and the row digest.

* `HashedOpenings`: on each path, each position's leaf verifier (its tree's domain, the hm96 key and the format), rank
  and root; the value a row decodes to; and the opening of each slot that each proof's witness carries.
* `checked`: what the check of one opening of a position hashed: the hm96 salt's and leaf's messages, the frame leaf's
  and the path's nodes' (`RowLeaf.outerHashed`), then the row digest's (`RowLeaf.innerHashed`).
* `Verified`, `Unverified`: round `w`'s proof of its `i`-th sampled proof block opens slot `k` with an opening its position's
  verifier accepts at the position's root and rank, whose row decodes to the value the proof opens; some accepted proof
  opens position `ℓ` otherwise.
* `Split`: position `ℓ`'s verifier, root or rank differs between a path and its continuation from `ℓ`'s cut.
* `Finds`, `finder`: **the fork finder at positions `L`**, a SHA-512 collision finder (`Claims.Finder`) on the coins
  `(x, z)`. It runs the period on `x` and on its continuation from `ℓ`'s cut on `z`, and for a position `ℓ ∈ L` that an
  accepted proof opens on each run it outputs two different messages of one digest among what the two openings'
  checks hashed, when there are. Its cost is `q`, the SHA-512 calls of the two runs (the prover's, the extractor's and
  the verifier's) and of the checks of their openings.

Two accepted openings of one position on the two runs to different values (`SampledLayer.Forked`), each `Verified` at one
verifier, root and rank, have different rows, so their checks hashed a collision (`Hash.Guarantees.RowBinding`), which
`finder` outputs. A fork escapes it only through an opening that isn't `Verified` or a `Split`. No premise of the
credit guarantees rules either out yet: none ties a layer's `opened` to an opening of a committed leaf (the service's
soundness is about its relation alone), and none fixes a position's root by its cut (a window's registration is fixed
before its sampling coin, not before the coins its sample follows).
-/

@[expose] public section

namespace Accounting.TS

open Pouw.RelativeSecurity

/-- **A sampled layer's openings as hm96 rows of frame-v3-sha512 trees**, over `S` rounds of coins on `Paths C m × B`, `s`
samples a round, slots `Q`, positions `Pos` and values `V`:
* `leaf x ℓ`, `rank x ℓ`, `root x ℓ`: on `x`, what the verifier holds for position `ℓ`'s leaf (its tree's domain, the
  hm96 key and the format), the leaf's rank, and the root committed for it;
* `decode`: the value a row's bits give;
* `opening x w i k`: on `x`, the hm96 opening (row, salt, siblings) of slot `k` that round `w`'s proof of its `i`-th
  sampled proof block carries in its witness. -/
structure HashedOpenings (C B : Type) (S m s : ℕ) (Q Pos V : Type) where
  leaf : Paths (Paths C m × B) S → Pos → Hash.RowLeaf.Verifier
  rank : Paths (Paths C m × B) S → Pos → ℕ
  root : Paths (Paths C m × B) S → Pos → Hash.Digest
  decode : Hash.Row → V
  opening : Paths (Paths C m × B) S → Fin S → Fin s → Q → Hash.RowLeaf.Opening

namespace HashedOpenings

variable {C B : Type} {S m s : ℕ} {J Q Pos V : Type} (hl : HashedOpenings C B S m s Q Pos V)
  (pf : SampledLayer C B S m s J Q Pos V)

/-- What the check of the opening `o` of position `ℓ` on `x` hashed under `H`: the hm96 salt's and leaf's messages, the
frame leaf's and the path's nodes', then the row digest's. -/
def checked (H : Hash.Msg → Hash.Digest) (x : Paths (Paths C m × B) S) (ℓ : Pos) (o : Hash.RowLeaf.Opening) :
    List Hash.Msg :=
  Hash.RowLeaf.outerHashed H H (hl.leaf x ℓ) (hl.rank x ℓ) o ++ Hash.RowLeaf.innerHashed H (hl.leaf x ℓ) o

/-- **Round `w`'s proof of its `i`-th sampled proof block opens slot `k` by a verified path** on `x`: its opening is accepted by
the verifier of the slot's position at the position's root and rank, and its row decodes to the value the proof opens. -/
def Verified (H : Hash.Msg → Hash.Digest) (x : Paths (Paths C m × B) S) (w : Fin S) (i : Fin s) (k : Q) : Prop :=
  Hash.RowLeaf.accepts H H (hl.leaf x (pf.pos w (pf.drawnAt x w i) k)) (hl.root x (pf.pos w (pf.drawnAt x w i) k))
      (hl.rank x (pf.pos w (pf.drawnAt x w i) k)) (hl.opening x w i k) ∧
    hl.decode (hl.opening x w i k).row = pf.opened x w i k

/-- **Some accepted opening of position `ℓ` on `x` isn't verified**: an accepted proof of a sampled proof block opens one of its
slots, at `ℓ`, otherwise than by a verified path. -/
def Unverified (H : Hash.Msg → Hash.Digest) (x : Paths (Paths C m × B) S) (ℓ : Pos) : Prop :=
  ∃ w i k, pf.pos w (pf.drawnAt x w i) k = ℓ ∧ pf.accepted x w i ∧ ¬ hl.Verified pf H x w i k

/-- **Position `ℓ`'s leaf splits** between `x` and its continuation from `ℓ`'s cut on `z`: its verifier, root or rank
differs between them. Two verified openings bind each other only at one verifier, root and rank
(`Hash.Guarantees.RowBinding`). -/
def Split (x z : Paths (Paths C m × B) S) (ℓ : Pos) : Prop :=
  ¬ (hl.leaf (pf.cont ℓ x z) ℓ = hl.leaf x ℓ ∧ hl.root (pf.cont ℓ x z) ℓ = hl.root x ℓ ∧
    hl.rank (pf.cont ℓ x z) ℓ = hl.rank x ℓ)

/-- **What the fork finder at positions `L` outputs on `(x, z)`**: two different messages `uv` of one digest under `H`,
both among what the checks of two accepted openings of one position `ℓ ∈ L` hashed, one on `x` and one on its
continuation from `ℓ`'s cut on `z`. -/
def Finds (H : Hash.Msg → Hash.Digest) (L : Finset Pos) (x z : Paths (Paths C m × B) S)
    (uv : Hash.Msg × Hash.Msg) : Prop :=
  Claims.Collision (fun u => (H u).bytes) uv.1 uv.2 ∧
    ∃ ℓ ∈ L, ∃ w i k w' i' k', pf.pos w (pf.drawnAt x w i) k = ℓ ∧ pf.accepted x w i ∧
      pf.pos w' (pf.drawnAt (pf.cont ℓ x z) w' i') k' = ℓ ∧ pf.accepted (pf.cont ℓ x z) w' i' ∧
      uv.1 ∈ hl.checked H x ℓ (hl.opening x w i k) ++
        hl.checked H (pf.cont ℓ x z) ℓ (hl.opening (pf.cont ℓ x z) w' i' k') ∧
      uv.2 ∈ hl.checked H x ℓ (hl.opening x w i k) ++
        hl.checked H (pf.cont ℓ x z) ℓ (hl.opening (pf.cont ℓ x z) w' i' k')

open Classical in
/-- **The fork finder at positions `L`**, at `q` SHA-512 calls: on coins `(x, z)` it outputs what it finds (`Finds`), if
anything. -/
noncomputable def finder [Fintype C] [Fintype B] [Nonempty (Paths (Paths C m × B) S)] (H : Hash.Msg → Hash.Digest)
    (L : Finset Pos) (q : ℕ) : Claims.Finder where
  S := Paths (Paths C m × B) S × Paths (Paths C m × B) S
  out y := if h : ∃ uv, hl.Finds pf H L y.1 y.2 uv then some h.choose else none
  cost _ := q

end HashedOpenings

end Accounting.TS
