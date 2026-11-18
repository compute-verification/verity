module

public import Definitions.Hash.StringCommitment
public import Definitions.Hash.Guarantees
public import Proofs.Hash.RowBinding

/-!
# The frame-v3 tree of hm96 row leaves is a string commitment, and it binds

`rowTree V` commits a string of rows under the verifier `V`'s domain, key and format: rank `i` below the domain's
count holds the hm96 leaf of row `i` under the coin's salt `i`, the ranks above it pads, and the commitment is the
tree's root (`Tree.rootOf`); an opening is `RowLeaf.Opening`, accepted at a rank when its row is the value and
`RowLeaf.accepts` holds with `G = H`. Binding is row binding (`Hash.SecurityProofs.RowBinding`): its two collisions,
under the outer and the inner hash, are both collisions of `H` among what the two checks hashed.
-/

@[expose] public section

namespace Hash.StringCommitment

/-- The leaf digest at rank `i` of the tree over the rows `s` under the salts `salt`, or the pad's above the count. -/
def rowTreeLeaf (H : Msg → Digest) (V : RowLeaf.Verifier) (s : List Row) (salt : ℕ → Msg) (i : ℕ) : Digest :=
  if i < V.domain.count then H (RowLeaf.leafMsg H H V i ⟨s.getD i [], salt i, []⟩) else H (Tree.padMsg V.domain i)

/-- **The frame-v3 tree of hm96 row leaves under `V`, as a string commitment.** -/
def rowTree (V : RowLeaf.Verifier) : StringCommitment Row where
  Com := Digest
  Dec := List Row × (ℕ → Msg)
  Op := RowLeaf.Opening
  Coins := Fin V.domain.count → Fin Hm96.SALT_BYTES → UInt8
  commit H s r :=
    let salt : ℕ → Msg := fun i => if h : i < V.domain.count then List.ofFn (r ⟨i, h⟩) else []
    (Tree.rootOf H V.domain (rowTreeLeaf H V s salt), (s, salt))
  «open» H d p := (d.1.getD p [], ⟨d.1.getD p [], d.2 p, Tree.sibsAt H V.domain (rowTreeLeaf H V d.1 d.2) p⟩)
  Opens H root p v o := o.row = v ∧ RowLeaf.accepts H H V root p o
  hashed H _ p _ o := RowLeaf.outerHashed H H V p o ++ RowLeaf.innerHashed H V o

end Hash.StringCommitment

namespace Hash.SecurityProofs

theorem collisionIn_mono {H : Msg → Digest} {xs ys : List Msg} (h : CollisionIn H xs) (hs : xs ⊆ ys) :
    CollisionIn H ys := by
  obtain ⟨x, hx, y, hy, hc⟩ := h
  exact ⟨x, hs hx, y, hs hy, hc⟩

/-- **The frame-v3 tree of hm96 row leaves binds**, for every hash `H` and verifier `V`: two accepted openings of one
rank to different rows are a collision of `H` among what the two checks hashed. -/
theorem rowTree_binds (V : RowLeaf.Verifier) (H : Msg → Digest) : (StringCommitment.rowTree V).Binds H := by
  rintro c p v v' o o' ⟨hv, ha⟩ ⟨hv', ha'⟩
  rcases Hash.SecurityProofs.RowBinding H H V c p o o' ha ha' with h | h | h
  · exact .inl (hv.symm.trans (h.trans hv'))
  · refine .inr (collisionIn_mono h ?_)
    intro x hx
    simp only [StringCommitment.rowTree, List.mem_append] at hx ⊢
    tauto
  · refine .inr (collisionIn_mono h ?_)
    intro x hx
    simp only [StringCommitment.rowTree, List.mem_append] at hx ⊢
    tauto

end Hash.SecurityProofs
