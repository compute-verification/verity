/-
What the rows of `FrameV3Read_v1{D, SCHEMA}`'s net (`frameRead`) force, against the frame-v3 tree's specification
(`Hash.Tree`) under SHA-512 (`sha512D`): part for part, the leaf part holds the leaf's digest (`LeafSound`), a node part
the node's (`NodeSound`), the comparison 1 iff its digests agree (`EqSound`), and the read 1 iff the climb from the leaf
along the path reaches the root (`ReadSound`), which at a table's count and depth is `Hash.Tree.accepts` (`readAccepts`).
Each part's SHA-512 is the chain `ShaSound` states.
-/
module

public import Definitions.Hash.Tree
public import Proofs.Accounting.ReadGates.Spec
public import Proofs.Accounting.ReadGates.FrameV3Net

@[expose] public section

namespace Accounting.ReadGates

open Flock.HmNets hiding and
open FlockSoundness.Discharge.Hm

/-- `frame_v3._check_schema`: every length `1 … n` of `uint(x)` pads a leaf message under `schema` to one block count. -/
def LeafBlocks (schema : String) (n : ℕ) : Prop :=
  ∀ L ∈ List.range' 1 n,
    (209 + (Hash.utf8 schema).length + 2 * L + 17 + 127) / 128 = (209 + (Hash.utf8 schema).length + 2 + 17 + 127) / 128

/-- **The leaf part** (`FrameV3Leaf_v1{B = n, SCHEMA}`): whenever its rows hold, it returns a digest's forms holding the
frame-v3 leaf of `value` at rank `x` under the domain id `domain` and `schema`, in any domain placing rank `x` at
position `x`. -/
def LeafSound (schema : String) : Prop :=
  ∀ (domain value : List Word) (x : List Lin) (n : ℕ), Sound (leafHash schema domain value x n) fun z out =>
    DigestForms domain → DigestForms value → x.length = 8 * n → 1 ≤ n → n ≤ 8 → LeafBlocks schema n →
      DigestForms out ∧ ∀ Dom : Hash.Tree.Domain, Dom.id = digestW z domain → Dom.position (numV z x) = numV z x →
        digestW z out = sha512D (Hash.Tree.leafMsg Dom (numV z x) schema (digestW z value).bytes)

/-- **A node part** (`FrameV3Node_v1{B = n}`): whenever its rows hold, it returns a digest's forms holding the frame-v3
node at `level` and `index` under the domain id `domain` over `cur` and `sib`, `cur` the right child iff `d`. -/
def NodeSound : Prop :=
  ∀ (domain : List Word) (level : Byte) (index : List Lin) (d : Lin) (cur sib : List Word) (n : ℕ),
    Sound (nodeHash domain level index d cur sib n) fun z out =>
      DigestForms domain → level.length = 8 → index.length = 8 * n → 1 ≤ n → n ≤ 8 → DigestForms cur →
        DigestForms sib →
        DigestForms out ∧ ∀ Dom : Hash.Tree.Domain, Dom.id = digestW z domain →
          digestW z out = sha512D (if ev z d
            then Hash.Tree.nodeMsg Dom (numV z level) (numV z index) (digestW z sib) (digestW z cur)
            else Hash.Tree.nodeMsg Dom (numV z level) (numV z index) (digestW z cur) (digestW z sib))

/-- **The comparison** (`LmsEqual_v1{N = 64}`): whenever its rows hold, its bit is 1 iff the two digests agree. -/
def EqSound : Prop :=
  ∀ x y : List Word, Sound (eqDigest x y) fun z out =>
    DigestForms x → DigestForms y → ev z out = decide (digestW z x = digestW z y)

/-- **The read** (`FrameV3Read_v1{D, SCHEMA}`): whenever its rows hold, its bit is 1 iff the frame-v3 climb from the leaf
of `value` at rank `pos` along `path` reaches `root`, in any domain of id `domain` placing rank `pos` at position `pos`. -/
def ReadSound (D : ℕ) (schema : String) : Prop :=
  ∀ (root domain value : List Word) (pos : List Lin) (path : List (List Word)),
    Sound (frameRead D schema root domain value pos path) fun z out =>
      DigestForms root → DigestForms domain → DigestForms value → pos.length = D → path.length = D →
        (∀ s ∈ path, DigestForms s) → 1 ≤ D → D ≤ 64 → LeafBlocks schema (indexBytes D) →
        ∀ Dom : Hash.Tree.Domain, Dom.id = digestW z domain → Dom.position (numV z pos) = numV z pos →
          ev z out = decide ((Hash.Tree.climb sha512D Dom 0 (numV z pos)
            (sha512D (Hash.Tree.leafMsg Dom (numV z pos) schema (digestW z value).bytes))
            (path.map (digestW z))).2 = digestW z root)

/-- **At a table's count and depth, the read's bit is `Hash.Tree.accepts`**: in a domain of `count` rows whose depth is
`D`, at a rank below the count, the read is 1 iff the root accepts the leaf of `value` at `pos` with the path. -/
theorem readAccepts {D : ℕ} {schema : String} (hR : ReadSound D schema) (root domain value : List Word) (pos : List Lin)
    (path : List (List Word)) :
    Sound (frameRead D schema root domain value pos path) fun z out =>
      DigestForms root → DigestForms domain → DigestForms value → pos.length = D → path.length = D →
        (∀ s ∈ path, DigestForms s) → 1 ≤ D → D ≤ 64 → LeafBlocks schema (indexBytes D) →
        ∀ Dom : Hash.Tree.Domain, Dom.id = digestW z domain → Dom.position (numV z pos) = numV z pos →
          D = Hash.Tree.depth Dom.count → numV z pos < Dom.count →
          (ev z out = true ↔ Hash.Tree.accepts sha512D Dom (digestW z root) (numV z pos)
            (sha512D (Hash.Tree.leafMsg Dom (numV z pos) schema (digestW z value).bytes)) (path.map (digestW z))) :=
  Sound.mono (hR root domain value pos path) fun z out h hr hd hv hp hpl hs h1 h64 hb Dom hid hpos hD hlt => by
    rw [h hr hd hv hp hpl hs h1 h64 hb Dom hid hpos, decide_eq_true_iff, Hash.Tree.accepts, List.length_map, hpl, ← hD]
    exact ⟨fun hc => ⟨hlt, rfl, hc⟩, fun hc => hc.2.2⟩

end Accounting.ReadGates
