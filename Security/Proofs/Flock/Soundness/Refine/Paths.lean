import Proofs.Flock.Soundness.Refine.Rep

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

theorem u64_of_bytes (x y : UInt64) (h : ∀ i < 8, (x >>> (8 * i).toUInt64).toUInt8 = (y >>> (8 * i).toUInt64).toUInt8) :
    x = y := by
  apply UInt64.toNat_inj.1
  apply Nat.eq_of_testBit_eq
  intro j
  by_cases hj : j < 64
  · have hi := congrArg UInt8.toNat (h (j / 8) (by omega))
    simp only [UInt64.toNat_toUInt8, UInt64.toNat_shiftRight, toNat_toUInt64 (8 * (j / 8)) (by omega),
      Nat.mod_eq_of_lt (by omega : 8 * (j / 8) < 64)] at hi
    have hb := congrArg (fun n => n.testBit (j % 8)) hi
    simp only [Nat.testBit_mod_two_pow, Nat.testBit_shiftRight] at hb
    rw [show 8 * (j / 8) + j % 8 = j by omega] at hb
    simpa [show j % 8 < 8 by omega] using hb
  · have h64 : 2 ^ 64 ≤ 2 ^ j := Nat.pow_le_pow_right (by norm_num) (by omega)
    rw [Nat.testBit_lt_two_pow (lt_of_lt_of_le x.toNat_lt h64), Nat.testBit_lt_two_pow (lt_of_lt_of_le y.toNat_lt h64)]

theorem toBytes_get (a : F128) (i : ℕ) (hi : i < 16) :
    a.toBytes.data[i]! = if i < 8 then (a.lo >>> (8 * i).toUInt64).toUInt8 else (a.hi >>> (8 * (i - 8)).toUInt64).toUInt8 := by
  unfold F128.toBytes
  rw [getElem!_pos _ i (by simpa using hi), Array.getElem_ofFn]

theorem toBytes_inj {a b : F128} (h : a.toBytes = b.toBytes) : a = b := by
  have hd := congrArg ByteArray.data h
  have hlo : a.lo = b.lo := u64_of_bytes a.lo b.lo fun i hi => by
    have := congrArg (·[i]!) hd
    simp only [toBytes_get a i (by omega), toBytes_get b i (by omega), hi, ite_true] at this
    exact this
  have hhi : a.hi = b.hi := u64_of_bytes a.hi b.hi fun i hi => by
    have := congrArg (·[i + 8]!) hd
    simp only [toBytes_get a (i + 8) (by omega), toBytes_get b (i + 8) (by omega),
      show ¬ i + 8 < 8 by omega, ite_false, Nat.add_sub_cancel] at this
    exact this
  cases a; cases b
  simp_all

theorem toBytes_length (a : F128) : a.toBytes.data.toList.length = 16 := by
  simp [F128.toBytes]

/-- A row's bytes: its words' 16-byte encodings, in order. -/
def colBytes (col : List F128) : List UInt8 := col.flatMap fun a => a.toBytes.data.toList

theorem colBytes_inj : Function.Injective colBytes := by
  intro l₁
  induction l₁ with
  | nil =>
      intro l₂ h
      cases l₂ with
      | nil => rfl
      | cons b l₂ =>
          have := congrArg List.length h
          simp only [colBytes, List.flatMap_nil, List.flatMap_cons, List.length_append, toBytes_length,
            List.length_nil] at this
          omega
  | cons a l₁ ih =>
      intro l₂ h
      cases l₂ with
      | nil =>
          have := congrArg List.length h
          simp only [colBytes, List.flatMap_nil, List.flatMap_cons, List.length_append, toBytes_length,
            List.length_nil] at this
          omega
      | cons b l₂ =>
          simp only [colBytes, List.flatMap_cons] at h
          obtain ⟨h1, h2⟩ := List.append_inj h (by rw [toBytes_length, toBytes_length])
          have hab : a = b := toBytes_inj (ByteArray.ext (Array.ext' h1))
          rw [hab, ih h2]

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

/-- A 64-byte digest. -/
abbrev Dig := {b : ByteArray // b.size = 64}

/-- SHA-512 on byte lists, into 64-byte digests. -/
def H512 (l : List UInt8) : Dig := ⟨Sha512.hash ⟨l.toArray⟩, sha512_size _⟩

/-- The tree's encodings: a row is its words' 16-byte encodings, a digest its 64 bytes. -/
def enc512 : Merkle.Enc (List 𝔽) Dig where
  col := colBytes
  dig := fun d => d.1.data.toList
  col_inj := colBytes_inj
  dig_inj := fun a b h => Subtype.ext (ByteArray.ext (Array.ext' h))
  dig_len := fun a b => by
    rw [Array.length_toList, Array.length_toList, ByteArray.size_data, ByteArray.size_data, a.2, b.2]

/-- A digest from bytes: the bytes when there are 64 of them, else a fixed digest. -/
def toDig (b : ByteArray) : Dig :=
  if h : b.size = 64 then ⟨b, h⟩ else ⟨⟨Array.replicate 64 0⟩, show (Array.replicate 64 (0 : UInt8)).size = 64 by simp⟩

theorem toDig_val {b : ByteArray} (h : b.size = 64) : (toDig b).1 = b := by simp [toDig, h]

theorem bytes_of_list (b : ByteArray) : (⟨b.data.toList.toArray⟩ : ByteArray) = b := by
  cases b; simp

/-- **The plain leaf**: SHA-512 of the row's encoding, its salt ignored: `MerkleScheme.sha512`'s leaf (`leafPlain_leaf`),
the leaf of a recursive session's inner statement (`flock-leaf/sha512-unsalted`, `Tags.withPlainLeaves`). -/
def leafPlain : Merkle.Leaf (List 𝔽) (List UInt8) Dig H512 where
  leaf c _ := H512 (enc512.col c)
  bind _ _ _ _ hc h := ⟨_, _, fun e => hc (enc512.col_inj e), h⟩

/-- **The executable's unsalted leaf is the model's plain leaf**, under any salt. -/
theorem leafPlain_leaf (id : String) (row : Array F128) (salt : ByteArray) (y : List UInt8) :
    (leafPlain.leaf row.toList y).1 = (MerkleScheme.sha512 id).leaf row salt := by
  show Sha512.hash ⟨(colBytes row.toList).toArray⟩ = Sha512.hash (MerkleScheme.rowBytes row)
  rw [← rowBytes_data, bytes_of_list]

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

/-- A cap of digests, entry by index. -/
def capDig (cap : Array ByteArray) : Cap Dig := fun i => toDig (cap.getD i ByteArray.empty)

/-- The proof's opening at level `ℓ`, query `j`: the opened row, and its siblings as digests. -/
def openRow (p : LigeritoProof) (sch : Flock.Schedule) (ℓ j : ℕ) : List 𝔽 × List Dig :=
  (((openAt p (sch.levels.size - 1) ℓ).1)[j]!.toList.map fOf,
    (sibsOf (openAt p (sch.levels.size - 1) ℓ).2.1 ((sch.levels.getD ℓ default).d - (sch.levels.getD ℓ default).cap)
      j).map toDig)

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
