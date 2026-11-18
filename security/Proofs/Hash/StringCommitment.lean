module

public import Definitions.Hash.Guarantees
public import Proofs.Hash.RowBinding
public import Proofs.Hash.Bytes
public import Mathlib.Data.Fintype.Pi

/-!
# Frame-v3 trees are string commitments, and they bind

`frameTree D schema` commits a string of 64-byte values under the domain `D`: rank `i` below the count holds the frame
leaf of value `i` under `schema`, the ranks above it pads, and the commitment is the root (`Tree.rootOf`); an opening
is the path's siblings, accepted when `Tree.accepts` takes the leaf's digest. Binding is tree binding
(`Hash.SecurityProofs.TreeBinding`), the leaf message binding its value (`Hash.BindingProofs.leafMsg_inj`).

`rowTree V` commits a string of rows under the verifier `V`'s domain, key and format: rank `i` below the domain's
count holds the hm96 leaf of row `i` under the coin's salt `i` (192 bytes), the ranks above it pads, and the commitment is the
tree's root (`Tree.rootOf`); an opening is `RowLeaf.Opening`, accepted at a rank when its row is the value and
`RowLeaf.accepts` holds with `G = H`. Binding is row binding (`Hash.SecurityProofs.RowBinding`): its two collisions,
under the outer and the inner hash, are both collisions of `H` among what the two checks hashed.
-/

@[expose] public section

namespace Hash.StringCommitment

/-- The leaf digest at rank `i` of the frame-v3 tree over the values `s` under `schema`, or the pad's above the count. -/
def frameTreeLeaf (H : Msg → Digest) (D : Tree.Domain) (schema : String) (s : List Digest) (i : ℕ) : Digest :=
  if i < D.count then H (Tree.leafMsg D i schema (s.getD i fun _ => 0).bytes) else H (Tree.padMsg D i)

/-- **The frame-v3 tree over `D` under `schema`, as a string commitment** of 64-byte values, with no coins. -/
def frameTree (D : Tree.Domain) (schema : String) : StringCommitment Digest where
  Com := Digest
  Dec := List Digest
  Op := List Digest
  Coins := Unit
  commit H s _ := (Tree.rootOf H D (frameTreeLeaf H D schema s), s)
  «open» H s p := (s.getD p fun _ => 0, Tree.sibsAt H D (frameTreeLeaf H D schema s) p)
  Opens H root p v sibs := Tree.accepts H D root p (H (Tree.leafMsg D p schema v.bytes)) sibs
  hashed H _ p v sibs := Tree.leafMsg D p schema v.bytes :: Tree.hashed H D p (H (Tree.leafMsg D p schema v.bytes)) sibs

/-- The leaf digest at rank `i` of the tree over the rows `s` under the salts `salt`, or the pad's above the count. -/
def rowTreeLeaf (H : Msg → Digest) (V : RowLeaf.Verifier) (s : List Row) (salt : ℕ → Msg) (i : ℕ) : Digest :=
  if i < V.domain.count then H (RowLeaf.leafMsg H H V i ⟨s.getD i [], salt i, []⟩) else H (Tree.padMsg V.domain i)

/-- **The frame-v3 tree of hm96 row leaves under `V`, as a string commitment.** -/
def rowTree (V : RowLeaf.Verifier) : StringCommitment Row where
  Com := Digest
  Dec := List Row × (ℕ → Msg)
  Op := RowLeaf.Opening
  Coins := Fin V.domain.count → Fin Hm96.SALT_BYTES → Fin 256
  commit H s r :=
    let salt : ℕ → Msg := fun i => if h : i < V.domain.count then List.ofFn fun j => (r ⟨i, h⟩ j).val.toUInt8 else []
    (Tree.rootOf H V.domain (rowTreeLeaf H V s salt), (s, salt))
  «open» H d p := (d.1.getD p [], ⟨d.1.getD p [], d.2 p, Tree.sibsAt H V.domain (rowTreeLeaf H V d.1 d.2) p⟩)
  Opens H root p v o := o.row = v ∧ RowLeaf.accepts H H V root p o
  hashed H _ p _ o := RowLeaf.outerHashed H H V p o ++ RowLeaf.innerHashed H V o

end Hash.StringCommitment

namespace Hash.SecurityProofs

/-- **The frame-v3 tree binds**, for every hash `H`, domain `D` and schema: two accepted openings of one rank to
different values are a collision of `H` among what the two checks hashed. -/
theorem frameTree_binds (D : Tree.Domain) (schema : String) (H : Msg → Digest) :
    (StringCommitment.frameTree D schema).Binds H := by
  intro c p v v' o o' h₁ h₂
  have ha : Tree.accepts H D c p (H (Tree.leafMsg D p schema v.bytes)) o := h₁
  have ha' : Tree.accepts H D c p (H (Tree.leafMsg D p schema v'.bytes)) o' := h₂
  rcases Hash.SecurityProofs.TreeBinding H D c p _ _ o o' ha ha' with h | h
  · rcases Hash.BindingProofs.eq_or_collisionIn (xs := StringCommitment.hashed (StringCommitment.frameTree D schema) H c p v o ++
        StringCommitment.hashed (StringCommitment.frameTree D schema) H c p v' o')
        (by simp [StringCommitment.frameTree]) (by simp [StringCommitment.frameTree]) h with hm | hc
    · exact .inl (Hash.BindingProofs.leafMsg_inj hm)
    · exact .inr hc
  · refine .inr (Hash.BindingProofs.collisionIn_mono ?_ h)
    intro x hx
    simp only [StringCommitment.frameTree, List.mem_append, List.mem_cons] at hx ⊢
    tauto

/-- **The frame-v3 tree of hm96 row leaves binds**, for every hash `H` and verifier `V`: two accepted openings of one
rank to different rows are a collision of `H` among what the two checks hashed. -/
theorem rowTree_binds (V : RowLeaf.Verifier) (H : Msg → Digest) : (StringCommitment.rowTree V).Binds H := by
  intro c p v v' o o' h₁ h₂
  obtain ⟨hv, ha⟩ : o.row = v ∧ RowLeaf.accepts H H V c p o := h₁
  obtain ⟨hv', ha'⟩ : o'.row = v' ∧ RowLeaf.accepts H H V c p o' := h₂
  rcases Hash.SecurityProofs.RowBinding H H V c p o o' ha ha' with h | h | h
  · exact .inl (hv.symm.trans (h.trans hv'))
  · refine .inr (Hash.BindingProofs.collisionIn_mono ?_ h)
    intro x hx
    simp only [StringCommitment.rowTree, List.mem_append] at hx ⊢
    tauto
  · refine .inr (Hash.BindingProofs.collisionIn_mono ?_ h)
    intro x hx
    simp only [StringCommitment.rowTree, List.mem_append] at hx ⊢
    tauto

end Hash.SecurityProofs
