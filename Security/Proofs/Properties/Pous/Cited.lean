import Properties.Pous.Cited
import Proofs.Properties.Pous.Run

/-!
# Proofs of the cited run's properties (`Properties.Pous.Cited`): `rootDecodes` and `citedRunAccepted`

`rootDecodes`: an accepted `C*` decodes to the committed `W` (`acceptsDecodes`), so it has `q`'s shape, and has the
published root (`publishesRoot`), so its leaves and those of a `C*'` of that shape at that root have one root. Over
leaves of 64 bytes, one root means the same leaves unless two inner nodes' inputs collide (`mth_binds`); one leaf means
one leaf input, so one row digest, unless the two inputs collide; and one row digest means one block, the prefix
carrying the block's length, unless the two rows' inputs collide. So `C*'` is `C*` block for block, and decodes as it
does. `citedRunAccepted`: the cited form's acceptance is the citation and `Run.check` (`acceptsCited_deployed`), and
`Accepted` is the setup check's acceptance and `Run.check`.
-/

namespace Pous.Cited.SecurityProofs

open Flock PousCheck PousCheck.Properties
open PousCheck.Properties.SecurityProofs (append_cancel mth_binds leaves_entry root_mth acceptsDecodes publishesRoot)
open Pous.EndToEnd (leafInput blockInput)
open Pous.EndToEnd.SecurityProofs (flatten_length flatten_getElem? getElem!_of_some)

/-! ## `C*`'s leaves at any parameters -/

theorem shapedAt_of {q : P2.Params} {c : SetupProtocol.Commitment} {C : Array (Array ByteArray)}
    (h : DecodesToCommitment q c C) : ShapedAt q C := by
  obtain ⟨W, ⟨hCs, -, hseg⟩, -⟩ := h
  refine ⟨hCs, fun s hs => ⟨(hseg s hs).1, fun v hv => ?_⟩⟩
  obtain ⟨-, -, -, -, -, -, hb, -⟩ := (hseg s hs).2 v hv
  exact hb

theorem leaves_lists_at {q : P2.Params} {C : Array (Array ByteArray)} (hS : ShapedAt q C) (salt : ByteArray) :
    ∀ l ∈ C.toList.mapIdx (fun s seg => seg.toList.mapIdx fun v block => Merkle.packageLeaf salt s v block),
      l.length = q.blocksPerSegment := by
  intro l hl
  obtain ⟨s, hs, rfl⟩ := List.mem_iff_getElem.1 hl
  simp only [List.getElem_mapIdx, List.length_mapIdx, Array.length_toList]
  simp only [List.length_mapIdx, Array.length_toList] at hs
  exact (hS.2 s hs).1

theorem leaves_size_at {q : P2.Params} {C : Array (Array ByteArray)} (hS : ShapedAt q C) (salt : ByteArray) :
    (leaves salt C).size = q.blocks := by
  rw [leaves, List.size_toArray, flatten_length q.blocksPerSegment _ (leaves_lists_at hS salt)]
  simp [hS.1, P2.Params.blocks]

theorem leaves_getElem?_at {q : P2.Params} {C : Array (Array ByteArray)} (hS : ShapedAt q C) (salt : ByteArray)
    (s v : Nat) (hv : v < q.blocksPerSegment) :
    (leaves salt C)[s * q.blocksPerSegment + v]? = C[s]?.bind fun seg => seg[v]?.map (Merkle.packageLeaf salt s v) := by
  rw [leaves, List.getElem?_toArray, flatten_getElem? q.blocksPerSegment _ (leaves_lists_at hS salt) s v hv]
  simp only [List.getElem?_mapIdx, Array.getElem?_toList]
  cases C[s]? <;> simp [List.getElem?_mapIdx]

theorem leaves_getElem!_at {q : P2.Params} {C : Array (Array ByteArray)} (hS : ShapedAt q C) (salt : ByteArray)
    {s v : Nat} (hs : s < C.size) (hv : v < C[s].size) :
    (leaves salt C)[s * q.blocksPerSegment + v]! = Merkle.packageLeaf salt s v C[s][v] := by
  have hv' : v < q.blocksPerSegment := (hS.2 s hs).1 ▸ hv
  apply getElem!_of_some
  rw [leaves_getElem?_at hS salt s v hv', Array.getElem?_eq_getElem hs]
  simp [Array.getElem?_eq_getElem hv]

/-! ## The stores' strings -/

theorem mem_store_node {q : P2.Params} {salt : ByteArray} {C : Array (Array ByteArray)} {x : ByteArray}
    (hS : ShapedAt q C) (hx : x ∈ nodeInputs (leaves salt C) 0 q.blocks) : x ∈ storeStrings salt C := by
  unfold storeStrings
  dsimp only
  rw [leaves_size_at hS salt]
  exact List.mem_append_left _ hx

theorem mem_store_block (salt : ByteArray) (C : Array (Array ByteArray)) {s v : Nat} (hs : s < C.size)
    (hv : v < C[s].size) {z : ByteArray}
    (hz : z = leafInput salt s v (Merkle.rowDigest C[s][v]) ∨ z = blockInput C[s][v]) : z ∈ storeStrings salt C := by
  unfold storeStrings
  dsimp only
  refine List.mem_append_right _ (List.mem_flatten.2 ⟨(C[s].toList.mapIdx fun v block =>
      [leafInput salt s v (Merkle.rowDigest block), blockInput block]).flatten, ?_, ?_⟩)
  · exact List.mem_mapIdx.2 ⟨s, by simpa using hs, by simp⟩
  · refine List.mem_flatten.2 ⟨[leafInput salt s v (Merkle.rowDigest C[s][v]), blockInput C[s][v]], ?_, ?_⟩
    · exact List.mem_mapIdx.2 ⟨v, by simpa using hv, by simp⟩
    · rcases hz with rfl | rfl <;> simp

/-! ## The theorems -/

theorem rootDecodes : RootDecodes := by
  intro q c C pub prof hacc C' hS' hroot'
  have hD := acceptsDecodes q c C ⟨pub, prof, hacc⟩
  have hP := publishesRoot q c C pub prof hacc
  have hS := shapedAt_of hD
  by_cases hB : StoreBreak c.salt C C'
  · exact .inr hB
  left
  have e1 := root_mth hP.1
  rw [leaves_size_at hS] at e1
  have e2 := root_mth hroot'
  rw [leaves_size_at hS'] at e2
  rcases mth_binds _ _ (leaves_entry c.salt C) (leaves_entry c.salt C') q.blocks 0
      (by rw [leaves_size_at hS]; omega) (by rw [leaves_size_at hS']; omega) (e1.trans e2.symm) with
    hl | ⟨x, hx, y, hy, hxy⟩
  swap
  · exact absurd ⟨x, mem_store_node hS hx, y, mem_store_node hS' hy, hxy⟩ hB
  have hblock : ∀ s (hs : s < C.size) (hs' : s < C'.size) v (hv : v < C[s].size) (hv' : v < C'[s].size),
      C[s][v] = C'[s][v] := by
    intro s hs hs' v hv hv'
    have hsq : s < q.segments := hS.1 ▸ hs
    have hvq : v < q.blocksPerSegment := (hS.2 s hs).1 ▸ hv
    have hidx : s * q.blocksPerSegment + v < q.blocks := by
      have h1 : s * q.blocksPerSegment + v < (s + 1) * q.blocksPerSegment := by rw [Nat.succ_mul]; omega
      exact Nat.lt_of_lt_of_le h1 (Nat.mul_le_mul_right _ hsq)
    have hlv := hl (s * q.blocksPerSegment + v) (Nat.zero_le _) (by omega)
    rw [leaves_getElem!_at hS c.salt hs hv, leaves_getElem!_at hS' c.salt hs' hv'] at hlv
    have hrow : Merkle.rowDigest C[s][v] = Merkle.rowDigest C'[s][v] := by
      by_cases hli : leafInput c.salt s v (Merkle.rowDigest C[s][v]) = leafInput c.salt s v (Merkle.rowDigest C'[s][v])
      · exact (append_cancel hli rfl).2
      · exact absurd ⟨_, mem_store_block c.salt C hs hv (.inl rfl), _, mem_store_block c.salt C' hs' hv' (.inl rfl),
          hli, hlv⟩ hB
    by_cases hbi : blockInput C[s][v] = blockInput C'[s][v]
    · have hsz : (HmRow.rowPrefixV2 (8 * C[s][v].size)).size = (HmRow.rowPrefixV2 (8 * C'[s][v].size)).size := by
        rw [(hS.2 s hs).2 v hv, (hS'.2 s hs').2 v hv']
      exact (append_cancel hbi hsz).2
    · exact absurd ⟨_, mem_store_block c.salt C hs hv (.inr rfl), _, mem_store_block c.salt C' hs' hv' (.inr rfl),
        hbi, hrow⟩ hB
  have hCC : C' = C := by
    refine Array.ext (hS'.1.trans hS.1.symm) fun s hs' hs => ?_
    refine Array.ext ((hS'.2 s hs').1.trans (hS.2 s hs).1.symm) fun v hv' hv => ?_
    exact (hblock s hs hs' v hv hv').symm
  rw [hCC]
  exact hD

theorem citedRunAccepted : CitedRunAccepted := by
  intro trusted c C pub prof s digests r sid hset hcited
  cases prof
  exact ⟨hset, ((RunProtocol.acceptsCited_deployed trusted s pub digests r sid).1 hcited).2⟩

end Pous.Cited.SecurityProofs
