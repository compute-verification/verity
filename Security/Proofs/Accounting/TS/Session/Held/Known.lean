import Proofs.Flock.C3.RegLevels.ValueTie
import Flock.RegKnown

/-!
# The known-content tie: a registrant's row at a port the verifier registers at known content is that content, or SHA-512 collides

`flock-verify --registered F` refuses a file whose port named at known content (`Registered.KNOWN_KEY`: each row's
u16 words and its salt) has a registered root other than the frame-v3-sha512 root of those rows' `hm96-sha512/row/v2`
leaves (`Registered.checkKnownAll`, run in `FlockVerify.registeredInput`). What that refusal establishes, from the
verifier's code:

* `checkKnownAll_ok`, `checkKnown_ok`: each named port is `own`'s first port of that name, of `hm96-sha512/row/v2`
  rows, as many as its known rows, and its root is `FV3.root` of their leaves (`Registered.knownLeaves`);
* `knownPayload_eq`: a known row's hm96 leaf is `hm96Input`'s of its `sha512/row/v2` bytes (`rowBytesV2`);
* `knownColl_spec`: a registrant's row that opens such a root at position `i` (`opensLeaf`, `ReadsX.committed`) has
  the bytes of known row `i`, or `knownColl` outputs two different SHA-512 inputs with one digest. As `valueColl`
  (`C3/RegLevels/ValueTie.lean`), the pair is the hm96 leaf's (`hm96Pair`) or the opening's against the verifier's
  tree (`openPairAs`, the schema-generic `openPairV`);
* `bits_of_v2`: a value whose `sha512/row/v2` encoding is a known row's bytes has that row's bits.

`knownColl` evaluates SHA-512 at most `5n + 2^(D+1) + 2D + 7` times, `n` the rows and `D = log2ceil n`, as
`valueColl` does, with the known rows' digests in place of the verifier's.
-/

namespace FlockC3.RegLevels

open Flock
open FlockSoundness FlockSoundness.Audit FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open FlockSoundness.Refine (H512)
open FlockSoundness.Binding (hm96Input hm96Pair hm96Pair_spec hm96_treeLeaf)

/-! ## What `checkKnownAll` checked -/

/-- **A port `checkKnownAll` accepts known content for** is `own`'s first port of that name, and `checkKnown` accepts it
at that content. -/
theorem checkKnownAll_ok {own : Array Registered.Port} {known : Array (String × Array (Array ℕ × ByteArray))}
    (h : Registered.checkKnownAll own known = .ok ()) {name : String} {ks : Array (Array ℕ × ByteArray)}
    (hm : (name, ks) ∈ known) :
    ∃ reg, own.toList.find? (·.name == name) = some reg ∧ Registered.checkKnown reg ks = .ok () := by
  have h1 := Registered.forM_ok h (name, ks) (by simpa using hm)
  dsimp only at h1
  split at h1
  · rename_i reg hf
    exact ⟨reg, hf, h1⟩
  · cases h1

/-- **What `checkKnown` checked of the registered root**: `hm96-sha512/row/v2` rows, as many as the known rows, and the
root is `FV3.root` of their leaves. -/
theorem checkKnown_ok {reg : Registered.Port} {ks : Array (Array ℕ × ByteArray)}
    (h : Registered.checkKnown reg ks = .ok ()) :
    reg.schema = HmRow.ROW_SCHEMA_V2 ∧ ks.size = reg.leaves ∧
      FV3.root reg.domain (Registered.knownLeaves reg ks) Sha512.hash = reg.root := by
  unfold Registered.checkKnown at h
  split at h
  · cases h
  · rename_i hs
    split at h
    · cases h
    · rename_i hl
      split at h
      · cases h
      · split at h
        · cases h
        · rename_i hr
          simp only [bne_iff_ne, ne_eq, Bool.or_eq_true, not_or, not_not] at hs hl
          exact ⟨hs.1, hl, eq_of_not_bne hr⟩

/-! ## A known row's leaf -/

theorem rowBytesV2_toList (row : Array ℕ) :
    (Registered.rowBytesV2 row).data.toList =
      (HmRow.rowPrefixV2 (16 * row.size)).data.toList ++ wordBytes row.toList := by
  unfold Registered.rowBytesV2
  rw [← Array.foldl_toList, foldl_bytes]

/-- **A known row's hm96 leaf is hm96's leaf over its `sha512/row/v2` bytes** (`hm96_treeLeaf`). -/
theorem knownPayload_eq (row : Array ℕ) (salt : ByteArray) :
    Registered.knownPayload row salt =
      (H512 (hm96Input (Registered.rowBytesV2 row).data.toList salt.data.toList)).1 := by
  have e : Sha512.hash (Registered.rowBytesV2 row) =
      Sha512.hash ⟨((Registered.rowBytesV2 row).data.toList).toArray⟩ :=
    congrArg Sha512.hash (ByteArray.ext (by simp))
  rw [Registered.knownPayload, e]
  exact (hm96_treeLeaf (Val := List UInt8) id Function.injective_id _ salt).symm

theorem knownLeaves_size (reg : Registered.Port) (ks : Array (Array ℕ × ByteArray)) :
    (Registered.knownLeaves reg ks).size = ks.size := by
  simp [Registered.knownLeaves]

theorem knownLeaves_width (reg : Registered.Port) (ks : Array (Array ℕ × ByteArray)) :
    ∀ x ∈ Registered.knownLeaves reg ks, x.size = 64 := by
  intro x hx
  simp only [Registered.knownLeaves, Array.mem_map] at hx
  obtain ⟨_, -, rfl⟩ := hx
  exact Sha512.hash_size _

/-! ## The extractor -/

/-- **The opening's extractor against the verifier's tree, at any leaf schema**: `openPairV` with the frame-v3 leaves
of `schema` (`Registered.leafInAs`) in place of `hm96-sha512/row/v1`'s. -/
def openPairAs (schema : String) (dom : ByteArray) (pos : ℕ) (x : ByteArray) (s : List ByteArray) (y : ByteArray)
    (t : List ByteArray) : ByteArray × ByteArray :=
  if Sha512.hash (Registered.leafInAs schema dom pos x) = Sha512.hash (Registered.leafInAs schema dom pos y) then
    (Registered.leafInAs schema dom pos x, Registered.leafInAs schema dom pos y)
  else Registered.climbPair dom (Sha512.hash (Registered.leafInAs schema dom pos x))
    (Sha512.hash (Registered.leafInAs schema dom pos y)) pos 0 s t

theorem openPairAs_collides {schema : String} {dom : ByteArray} {pos : ℕ} {x y : ByteArray} {s t : List ByteArray}
    (hx : x.size = 64) (hy : y.size = 64) (hne : x ≠ y) (hl : s.length = t.length) (hs : ∀ z ∈ s, z.size = 64)
    (ht : ∀ z ∈ t, z.size = 64)
    (h : Registered.climbL dom (Registered.leafAs schema dom pos x) pos 0 s =
      Registered.climbL dom (Registered.leafAs schema dom pos y) pos 0 t) :
    Registered.Collides (openPairAs schema dom pos x s y t) := by
  rw [Registered.leafAs_eq, Registered.leafAs_eq] at h
  unfold openPairAs
  split
  · rename_i hl'
    exact ⟨fun hh => hne (Registered.leafInAs_inj (schema := schema) (dom := dom) (pos := pos) (by rw [hx, hy]) hh), hl'⟩
  · rename_i hl'
    exact Registered.climbPair_collides dom _ _ _ _ pos 0 hl hs ht (Sha512.hash_size _) (Sha512.hash_size _) hl' h

/-- **The known-content tie's extractor** at known row `i` of `ks`, registered under `reg`, against a registrant's
opening of the row bytes `r` under the salt `y` through `path`: nothing where `r` is the known row's
`sha512/row/v2` bytes; else hm96's pair where the two hm96 leaves are one; else the opening's pair (`openPairAs`) up the
registrant's path and the verifier's tree over the known rows. Each pair holds both preimages. -/
def knownColl (reg : Registered.Port) (ks : Array (Array ℕ × ByteArray)) (i : ℕ) (r y : List UInt8) (path : ByteArray) :
    Option (List UInt8 × List UInt8) :=
  if r = (Registered.rowBytesV2 ks[i]!.1).data.toList then none
  else if H512 (hm96Input r y) =
      H512 (hm96Input (Registered.rowBytesV2 ks[i]!.1).data.toList ks[i]!.2.data.toList) then
    some (hm96Pair r y (Registered.rowBytesV2 ks[i]!.1).data.toList ks[i]!.2.data.toList)
  else
    some ((openPairAs reg.schema reg.domain i (H512 (hm96Input r y)).1 (Registered.sibs path (log2ceil reg.leaves))
        (H512 (hm96Input (Registered.rowBytesV2 ks[i]!.1).data.toList ks[i]!.2.data.toList)).1
        (sibsV reg.domain (Registered.knownLeaves reg ks) i)).1.data.toList,
      (openPairAs reg.schema reg.domain i (H512 (hm96Input r y)).1 (Registered.sibs path (log2ceil reg.leaves))
        (H512 (hm96Input (Registered.rowBytesV2 ks[i]!.1).data.toList ks[i]!.2.data.toList)).1
        (sibsV reg.domain (Registered.knownLeaves reg ks) i)).2.data.toList)

/-- **`knownColl`'s spec**: where `checkKnown` accepted `reg` at the known rows `ks`, and the registrant's hm96 leaf of
`r` under `y` opens `reg`'s root at `i`, `i` is a known row's position, and `r` is that row's `sha512/row/v2` bytes, or
`knownColl` collides. -/
theorem knownColl_spec {reg : Registered.Port} {ks : Array (Array ℕ × ByteArray)} {i : ℕ} {r y : List UInt8}
    {path : ByteArray} (hk : Registered.checkKnown reg ks = .ok ())
    (ho : Registered.opensLeaf reg (H512 (hm96Input r y)).1 i path = true) :
    i < ks.size ∧ (r = (Registered.rowBytesV2 ks[i]!.1).data.toList ∨
      ∃ a b, knownColl reg ks i r y path = some (a, b) ∧ a ≠ b ∧ H512 a = H512 b) := by
  obtain ⟨-, hn, hroot⟩ := checkKnown_ok hk
  simp only [Registered.opensLeaf, Bool.and_eq_true, decide_eq_true_eq] at ho
  obtain ⟨⟨hi, hp⟩, hc⟩ := ho
  rw [← hn] at hi
  refine ⟨hi, ?_⟩
  by_cases hr : r = (Registered.rowBytesV2 ks[i]!.1).data.toList
  · exact Or.inl hr
  refine Or.inr ?_
  unfold knownColl
  simp only [hr, ↓reduceIte]
  split
  · rename_i hh
    exact some_collides rfl (hm96Pair_spec _ _ _ _ hr hh)
  · rename_i hh
    have hiK : i < (Registered.knownLeaves reg ks).size := by rw [knownLeaves_size]; exact hi
    have hleaf : (Registered.knownLeaves reg ks)[i]! = Registered.leafAs reg.schema reg.domain i
        (H512 (hm96Input (Registered.rowBytesV2 ks[i]!.1).data.toList ks[i]!.2.data.toList)).1 := by
      rw [Registered.knownLeaves, getElem!_map_range _ hi, knownPayload_eq]
    have hV := climb_sibsV reg.domain (Registered.knownLeaves reg ks) hiK
    rw [hroot, hleaf] at hV
    have hcl : Registered.climbL reg.domain (Registered.leafAs reg.schema reg.domain i (H512 (hm96Input r y)).1) i 0
        (Registered.sibs path (log2ceil reg.leaves)) =
        Registered.climbL reg.domain (Registered.leafAs reg.schema reg.domain i
          (H512 (hm96Input (Registered.rowBytesV2 ks[i]!.1).data.toList ks[i]!.2.data.toList)).1) i 0
          (sibsV reg.domain (Registered.knownLeaves reg ks) i) := hc.trans hV.symm
    have hlen : (Registered.sibs path (log2ceil reg.leaves)).length =
        (sibsV reg.domain (Registered.knownLeaves reg ks) i).length := by
      rw [Registered.sibs_length, sibsV_length, knownLeaves_size, hn]
    have hcol := openPairAs_collides (H512 (hm96Input r y)).2
      (H512 (hm96Input (Registered.rowBytesV2 ks[i]!.1).data.toList ks[i]!.2.data.toList)).2
      (fun e => hh (Subtype.ext e)) hlen (Registered.sibs_size hp)
      (sibsV_size _ _ (knownLeaves_width _ _) hiK) hcl
    exact some_collides rfl (FlockSoundness.Registered.collides_bytes _ hcol)

/-! ## A known row's bits -/

/-- **The bits of a value whose `sha512/row/v2` bytes are a known row's**: where a value's row bytes under
`sha512/row/v2`'s first block (`rowEncP` at `Hidden.rowPre true`, a v2 row port's `encPortH`) are `rowBytesV2 row`, its
bit `b` is bit `b % 16` of the row's word `b / 16`, zero past the row. `bits_of_rowInV` without the padding. -/
theorem bits_of_v2 {W : ℕ} {v : Layout.BitVal W} {row : Array ℕ}
    (h : Layout.rowEncP (Hidden.rowPre true) v = (Registered.rowBytesV2 row).data.toList) (b : ℕ) (hb : b < W) :
    v.2 ⟨b, hb⟩ = (row[b / 16]!).testBit (b % 16) := by
  rw [rowBytesV2_toList] at h
  have hv : Layout.RowShaped v := by
    by_contra hv
    have e := congrArg List.head? h
    rw [FlockVBridge.Family.rowEncP_head_other hv, List.head?_append, FlockVBridge.Family.rowPrefixV2_head] at e
    cases e
  simp only [Layout.rowEncP, hv, ↓reduceIte] at h
  obtain ⟨-, hbytes⟩ := List.append_inj h (by
    rw [FlockVBridge.Family.rowPre_length, Array.length_toList]
    exact (Hidden.size_rowPrefixV2 _).symm)
  have hlen := congrArg List.length hbytes
  simp only [Layout.rowBytes, List.length_ofFn, wordBytes_length, Array.length_toList] at hlen
  by_cases hiv : b < v.1.val
  · have hj : b / 8 < (Layout.rowBytes v).length := by
      simp only [Layout.rowBytes, List.length_ofFn]; have := hv.1; omega
    have e := List.getElem_of_eq hbytes hj
    simp only [Layout.rowBytes, List.getElem_ofFn] at e
    rw [wordBytes_get] at e
    have := Layout.byteOf_bit (fun t => v.get (8 * (b / 8) + t.val)) ⟨b % 8, Nat.mod_lt _ (by norm_num)⟩
    rw [e] at this
    simp only [Layout.BitVal.get, show 8 * (b / 8) + b % 8 = b by omega, hb, ↓reduceDIte] at this
    rw [← this, show b / 8 / 2 = b / 16 by omega, Array.getElem!_toList]
    split
    · rw [testBit_lo _ _ (Nat.mod_lt _ (by norm_num)), show b % 8 = b % 16 by omega]
    · rw [testBit_hi _ _ (Nat.mod_lt _ (by norm_num)), show b % 8 + 8 = b % 16 by omega]
  · rw [hv.2 ⟨b, hb⟩ (by simp only; omega), getElem!_neg row (b / 16) (by have := hv.1; omega)]
    simp

end FlockC3.RegLevels
