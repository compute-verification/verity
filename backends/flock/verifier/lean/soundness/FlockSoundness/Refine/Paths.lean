import FlockSoundness.Refine.Rep
import FlockSoundness.Protocol.Refine.Paths

/-!
# The Merkle checks are the model's path checks (PROTOCOL.md §14)

For the unsalted SHA-512 scheme (`MerkleScheme.sha512`), a passed `merkleCheck` is the model's `Merkle.Verifies` at
every query, with the model's hash `H512` (SHA-512 on byte lists, into 64-byte digests) and encodings `enc512` (a row is
its words' 16-byte encodings, a digest its bytes).
-/

namespace FlockSoundness.Refine

open Flock FlockLevel3 Model

local notation "𝔽" => FlockLevel3.GF128

/-! ## SHA-512 digests -/

theorem sha512_size (msg : ByteArray) : (Sha512.hash msg).size = 64 := by
  unfold Sha512.hash
  simp only [Id.run]
  exact Array.size_ofFn

/-! ## The word encoding is injective -/

theorem rowBytes_data (row : Array F128) : (MerkleScheme.rowBytes row).data.toList = colBytes row.toList := by
  unfold MerkleScheme.rowBytes
  rw [← Array.foldl_toList]
  have : ∀ (L : List F128) (out : ByteArray), (L.foldl F128.pushBytes out).data.toList =
      out.data.toList ++ colBytes L := by
    intro L
    induction L with
    | nil => intro out; simp [colBytes]
    | cons a L ih =>
        intro out
        rw [List.foldl_cons, ih, F128.pushBytes, ByteArray.data_append, Array.toList_append]
        simp [colBytes]
  rw [this]
  rfl

/-! ## The model's hash and encodings -/

theorem toDig_val {b : ByteArray} (h : b.size = 64) : (toDig b).1 = b := by simp [toDig, h]

theorem bytes_of_list (b : ByteArray) : (⟨b.data.toList.toArray⟩ : ByteArray) = b := by
  cases b; simp

theorem H512_append (a b : Dig) : (H512 (enc512.dig a ++ enc512.dig b)).1 = Sha512.hash (a.1 ++ b.1) := by
  simp only [H512, enc512]
  congr 1

/-- **One step up the tree** is the scheme's node hash. -/
theorem walk_val (msId : String) : ∀ (sibs : List Dig) (idx : ℕ) (acc : Dig),
    (Merkle.walk H512 enc512 idx acc sibs).1 = climb (MerkleScheme.sha512 msId) acc.1 idx (sibs.map (·.1))
  | [], _, _ => rfl
  | s :: rest, idx, acc => by
      simp only [Merkle.walk, List.map_cons, climb]
      rw [walk_val msId rest (idx / 2)]
      congr 1
      unfold Merkle.up
      split_ifs <;> rw [H512_append] <;> rfl

theorem climb_size (msId : String) : ∀ (sibs : List ByteArray) (idx : ℕ) (acc : ByteArray), acc.size = 64 →
    (climb (MerkleScheme.sha512 msId) acc idx sibs).size = 64
  | [], _, _, h => h
  | s :: rest, idx, acc, _ => by
      simp only [climb]
      exact climb_size msId rest (idx / 2) _ (by split_ifs <;> exact sha512_size _)

/-- **A passed Merkle check is the model's path check** (§14), for the unsalted SHA-512 scheme: every query's row has
the lane count, and its path verifies against the cap. -/
theorem merkleCheck_verifies (msId : String) {cap : Array ByteArray} {c d : ℕ} {pos : Array ℕ}
    {rows : Array (Array F128)} {sibs salts : Array ByteArray} {lanes : ℕ}
    (h : merkleCheck (MerkleScheme.sha512 msId) cap c d pos rows sibs salts lanes = .ok ()) (j : ℕ) (hj : j < pos.size) :
    rows[j]!.size = lanes ∧
      Merkle.Verifies H512 enc512 (fun i => toDig (cap.getD i ByteArray.empty)) (d - c) pos[j]! rows[j]!.toList
        ((sibsOf sibs (d - c) j).map toDig) := by
  unfold merkleCheck at h
  split at h
  · cases h
  split at h
  · cases h
  rename_i hw
  simp only [Bool.or_eq_true, not_or, Array.any_eq_true, not_exists, bne_iff_ne, ne_eq, not_not] at hw
  split at h
  · cases h
  rename_i hf
  have hop := List.find?_eq_none.1 hf j (List.mem_range.2 hj)
  have h2 : opens (MerkleScheme.sha512 msId) cap c d lanes pos[j]! rows[j]! (salts.getD j ByteArray.empty)
      (sibsOf sibs (d - c) j) = true := by
    cases h3 : opens (MerkleScheme.sha512 msId) cap c d lanes pos[j]! rows[j]! (salts.getD j ByteArray.empty)
      (sibsOf sibs (d - c) j)
    · rw [h3] at hop; exact absurd rfl hop
    · rfl
  simp only [opens, Bool.and_eq_true, decide_eq_true_eq] at h2
  obtain ⟨⟨⟨hsz, hlen⟩, -⟩, hclimb⟩ := h2
  have hsib : ∀ b ∈ sibsOf sibs (d - c) j, b.size = 64 := by
    intro b hb
    unfold sibsOf at hb
    rw [Array.mem_toList_iff] at hb
    obtain ⟨i, hi, rfl⟩ := Array.mem_iff_getElem.1 hb
    rw [Array.getElem_extract]
    exact hw.2 _ (by simp only [Array.size_extract] at hi; omega)
  have hmap : ((sibsOf sibs (d - c) j).map toDig).map (·.1) = sibsOf sibs (d - c) j := by
    rw [List.map_map]
    exact (List.map_congr_left fun b hb => toDig_val (hsib b hb)).trans (List.map_id _)
  have hleaf : (H512 (enc512.col rows[j]!.toList)).1 = (MerkleScheme.sha512 msId).leaf rows[j]! (salts.getD j ByteArray.empty) := by
    show Sha512.hash ⟨(colBytes rows[j]!.toList).toArray⟩ = Sha512.hash (MerkleScheme.rowBytes rows[j]!)
    rw [← rowBytes_data, bytes_of_list]
  refine ⟨hsz, by rw [List.length_map, hlen], ?_⟩
  apply Subtype.ext
  unfold Merkle.reach
  rw [walk_val msId, hmap, hleaf, hclimb, ← Nat.shiftRight_eq_div_pow]
  have h64 : (cap.getD (pos[j]! >>> (d - c)) ByteArray.empty).size = 64 := by
    rw [← hclimb]
    exact climb_size msId _ _ _ (sha512_size _)
  exact (toDig_val h64).symm

/-! ## The proof's openings pass the model's path checks -/

/-- **The proof's openings pass the model's path checks** (§14): for the unsalted SHA-512 scheme, a rep whose requests
passed the executable's Merkle checks (`ReqOK`) and whose caps are the proof's is `OpensOK` at openings that are the
proof's rows and siblings. -/
theorem opensOK_of (msId : String) (sch : Flock.Schedule) (hlv : ∀ i < sch.levels.size, LevelSeam (sch.levels.getD i default))
    (cap : Array ByteArray) (p : LigeritoProof) (r : ℕ) (pend : PendL 𝔽 Dig) (o : Opens 𝔽 Dig)
    (hcaps : pend.caps = (List.range (sch.levels.size - 1)).map (fun j => capDig p.recursiveCaps[j]!))
    (hreqs : ∀ req ∈ pend.reqs, ReqOK (MerkleScheme.sha512 msId) sch cap p req)
    (ho : ∀ ℓ j, o r ℓ j = openRow p sch ℓ j) :
    OpensOK H512 enc512 (capDig cap) r pend o := by
  intro req hreq
  obtain ⟨hlev, hlanes, hdepth, hmc⟩ := hreqs req hreq
  refine ⟨capDig (capAt cap p req.level), ?_, fun j hj => ?_⟩
  · unfold capOf capAt
    split_ifs with h0
    · rfl
    · rw [hcaps, List.getElem?_map, List.getElem?_range (by omega)]
      rfl
  · have hj' : j < req.positions.toArray.size := by simpa using hj
    obtain ⟨hsz, hver⟩ := merkleCheck_verifies msId hmc j hj'
    have hd := (hlv req.level hlev).2.2.1
    rw [ho]
    have hpos : req.positions.getD j 0 = req.positions.toArray[j]! := by
      rw [List.getD_eq_getElem _ _ hj, getElem!_pos _ j hj', List.getElem_toArray]
    have hrow : (((openAt p (sch.levels.size - 1) req.level).1)[j]!.toList.map fOf) =
        ((openAt p (sch.levels.size - 1) req.level).1)[j]!.toList :=
      (List.map_congr_left fun _ _ => rfl).trans (List.map_id _)
    refine ⟨?_, ?_⟩
    · show (((openAt p (sch.levels.size - 1) req.level).1)[j]!.toList.map fOf).length = _
      rw [List.length_map, Array.length_toList, hsz]
    · show Merkle.Verifies H512 enc512 _ _ _ (((openAt p (sch.levels.size - 1) req.level).1)[j]!.toList.map fOf) _
      rw [hrow, hdepth, hpos, ← hd]
      exact hver

end FlockSoundness.Refine
