import Properties.Pouw.PearlPlus
import Properties.Pouw.PearlPlus.Witnesses
import Proofs.Properties.Pouw.Commit

/-!
# Pearl+'s properties say something

Each Pearl+ property's witnesses (`Security/Properties/lean-audit.json`'s `witnesses`): an `honest` one, its premises at an
instance the code accepts, and a `nontrivial` one, its conclusion false at a verifier that accepts everything.

`OpeningBinds`: an hm96 tile tree of two positions accepts an opening of position 0 against the node over its two leaves
(`openingBinds_honest`), and two openings of one position whose values differ, each hashing three different messages with
three different SHA-512 digests, neither open one value nor collide (`openingBinds_nontrivial`). The digests are computed by
the kernel, one declaration each, once the hm96 mask of their zero salt is known to be zero (`Witness.mask_zero`), since
the mask's tables are too large for it to build.

`YIsTheUnscale`, `RefusedBeforeTheCoin`, `TileKeepsTheCap`, `LoadedGivesUp`, `PlanMeans`, `AcceptMeans` and
`TileAcceptMeans` have both witnesses. `LoadedGivesUp`'s, `PlanMeans`' and `AcceptMeans`' `honest` ones are at one honest
pass of the deployed scheme (`honestPass`, loaded from `honestManifest`): `load` reads it, `plan` draws its one tile, and
`check` accepts its run at no draws, since a drawn tile is replayed on the noise `check` derives itself (`fixedNoise`),
past what one kernel computation reaches. `TileAcceptMeans`' `honest` one is the tile of a test pass (`testPass`, k = 128,
1 × 1 tiles) at a test noise (`testNoise`): the property ranges over every noise, and the deployed one is another instance.
-/

namespace Properties.Pouw.PearlPlus

open PouwCommit PouwCheck.PearlPlus

namespace Witness

/-! ## The hm96 mask of a zero salt -/

theorem foldl_inv {α β : Type} (P : β → Prop) (f : β → α → β) :
    ∀ (l : List α) (init : β), P init → (∀ a ∈ l, ∀ b, P b → P (f b a)) → P (l.foldl f init)
  | [], _, h0, _ => h0
  | a :: l, init, h0, hs => foldl_inv P f l (f init a) (hs a (.head _) init h0) fun x hx => hs x (.tail _ hx)

theorem getElem!_set!_ne {α : Type} [Inhabited α] (t : Array α) {i j : Nat} (h : i ≠ j) (x : α) :
    (t.set! i x)[j]! = t[j]! := by
  rw [Array.set!_eq_setIfInBounds, getElem!_def, getElem!_def, Array.getElem?_setIfInBounds_ne h]

/-- `Hm96.Default512.tables` at any number of salt bytes `P`, size `N` and columns `C`: the kernel never sees the
numbers, so it never builds the table. -/
def tablesAt (P N : Nat) (C : Array UInt64) : Array UInt64 := Id.run do
  let mut t := Array.replicate N (0 : UInt64)
  for p in [0:P] do
    for v in [1:256] do
      let prev := 8 * (256 * p + (v &&& (v - 1)))
      let col := 8 * (8 * p + Nat.log2 (v &&& (256 - v)))
      let at_ := 8 * (256 * p + v)
      for k in [0:8] do t := t.set! (at_ + k) (t[prev + k]! ^^^ C[col + k]!)
  return t

theorem tables_at : Flock.Hm96.Default512.tables = tablesAt 192 (8 * 256 * 192) Flock.Hm96.Default512.cols := rfl

/-- The table's entries at a zero salt byte (`i / 8 % 256 = 0`) are never written. -/
theorem tablesAt_zero (P N : Nat) (C : Array UInt64) (i : Nat) (hi : i / 8 % 256 = 0) : (tablesAt P N C)[i]! = 0 := by
  unfold tablesAt
  simp only [Id.run, Std.Legacy.Range.forIn_eq_forIn_range', List.forIn_pure_yield_eq_foldl, pure_bind]
  let Z : Array UInt64 → Prop := fun t => ∀ i, i / 8 % 256 = 0 → t[i]! = 0
  refine foldl_inv Z _ _ _ ?_ ?_ i hi
  · intro i _
    rw [getElem!_def, Array.getElem?_replicate]
    split <;> simp_all <;> rfl
  intro p _ t ht
  refine foldl_inv Z _ _ _ ht ?_
  intro v hv t ht
  have hv : 1 ≤ v ∧ v < 256 := by
    rw [List.mem_range'] at hv
    obtain ⟨j, hj, rfl⟩ := hv
    simp only [Std.Legacy.Range.size] at hj
    omega
  refine foldl_inv Z _ _ _ ht ?_
  intro k hk t ht i hi
  have hk : k < 8 := by
    rw [List.mem_range'] at hk
    obtain ⟨j, hj, rfl⟩ := hk
    simp only [Std.Legacy.Range.size] at hj
    omega
  rw [getElem!_set!_ne t (by omega)]
  exact ht i hi

def accAt (P : Nat) (T : Array UInt64) (salt : ByteArray) (acc : Array UInt64) : Array UInt64 := Id.run do
  let mut acc := acc
  for p in [0:P] do
    let at_ := 8 * (256 * p + (salt.get! p).toNat)
    for k in [0:8] do acc := acc.set! k (acc[k]! ^^^ T[at_ + k]!)
  return acc

def outAt (acc : Array UInt64) : ByteArray := Id.run do
  let mut out := ByteArray.emptyWithCapacity 64
  for k in [0:8] do
    for j in [0:8] do out := out.push ((acc[k]! >>> (8 * j).toUInt64) &&& 0xff).toUInt8
  return out

/-- `Hm96.Default512.mask` at any number of salt bytes `P` and table `T`. -/
def maskAt (P : Nat) (T : Array UInt64) (salt : ByteArray) : ByteArray := Id.run do
  let mut acc := Array.replicate 8 (0 : UInt64)
  for p in [0:P] do
    let at_ := 8 * (256 * p + (salt.get! p).toNat)
    for k in [0:8] do acc := acc.set! k (acc[k]! ^^^ T[at_ + k]!)
  let mut out := ByteArray.emptyWithCapacity 64
  for k in [0:8] do
    for j in [0:8] do out := out.push ((acc[k]! >>> (8 * j).toUInt64) &&& 0xff).toUInt8
  return out

theorem mask_at (salt : ByteArray) :
    Flock.Hm96.Default512.mask salt = maskAt 192 Flock.Hm96.Default512.tables salt := rfl

theorem maskAt_split (P : Nat) (T : Array UInt64) (salt : ByteArray) :
    maskAt P T salt = outAt (accAt P T salt (Array.replicate 8 0)) := rfl

theorem accAt_zero (P : Nat) (T : Array UInt64) (salt : ByteArray) (hs : ∀ p, salt.get! p = 0)
    (hT : ∀ i, i / 8 % 256 = 0 → T[i]! = 0) : accAt P T salt (Array.replicate 8 0) = Array.replicate 8 0 := by
  unfold accAt
  simp only [Id.run, Std.Legacy.Range.forIn_eq_forIn_range', List.forIn_pure_yield_eq_foldl, pure_bind, hs]
  refine foldl_inv (· = Array.replicate 8 0) _ _ _ rfl ?_
  intro p _ t ht
  refine foldl_inv (· = Array.replicate 8 0) _ _ _ ht ?_
  intro k hk t ht
  have hk : k < 8 := by
    rw [List.mem_range'] at hk
    obtain ⟨j, hj, rfl⟩ := hk
    simp only [Std.Legacy.Range.size] at hj
    omega
  subst ht
  rw [hT _ (by simp only [UInt8.toNat_zero]; omega)]
  revert k
  decide

theorem outAt_zero : outAt (Array.replicate 8 0) = ⟨Array.replicate 64 0⟩ := by
  unfold outAt
  simp only [Id.run, Std.Legacy.Range.forIn_eq_forIn_range', List.forIn_pure_yield_eq_foldl, pure_bind]
  decide

theorem tables_zero (i : Nat) (hi : i / 8 % 256 = 0) : Flock.Hm96.Default512.tables[i]! = 0 := by
  rw [tables_at]; exact tablesAt_zero _ _ _ i hi

/-- **The hm96 mask of a zero salt is zero.** -/
theorem mask_zero (salt : ByteArray) (hs : ∀ p, salt.get! p = 0) :
    Flock.Hm96.Default512.mask salt = ⟨Array.replicate 64 0⟩ := by
  rw [mask_at, maskAt_split, accAt_zero _ _ _ hs tables_zero, outAt_zero]

/-! ## Trees -/

theorem rowsDomain_eq (b : ByteArray) (o : Int) (c : Nat) (dg : Bool) (hb : b.size = 64) (ho : -2 ≤ o) :
    rowsDomain b o c HM96 dg = .ok ⟨⟨.sha512, c, frameHash .sha512 "domain"
      [b, Flock.FV3.uint (o + 2).toNat, rangeIdentity .sha512 c, Flock.FV3.uint c]⟩, true, dg⟩ := by
  have h : ¬ o < -2 := by omega
  simp [rowsDomain, HM96, Domain.mk?, hb, h, bind, Except.bind, pure, Except.pure]

/-! ## The digests `openingBinds_nontrivial` compares

Each is one kernel computation of its own, so no declaration hashes more than the hm96 key and its digest. -/

def leafPrefixLit : ByteArray := ⟨#[
  0x76, 0x65, 0x72, 0x69, 0x74, 0x79, 0x2F, 0x68, 0x6D, 0x39, 0x36, 0x2D, 0x73, 0x68, 0x61, 0x35,
  0x31, 0x32, 0x2F, 0x6C, 0x65, 0x61, 0x66, 0x2F, 0x76, 0x31, 0x00, 0xAF, 0xC5, 0xA6, 0x0C, 0x85,
  0xD8, 0xAA, 0x75, 0xCB, 0x4C, 0x7F, 0x72, 0x65, 0x88, 0xD2, 0x99, 0xA1, 0x71, 0x2E, 0x1C, 0x6A,
  0x62, 0x1F, 0x09, 0xCA, 0x64, 0x80, 0x9A, 0x5F, 0xED, 0xD7, 0x44, 0xA0, 0xF6, 0xB0, 0x20, 0x1D,
  0x05, 0xE8, 0x2B, 0x29, 0xCD, 0x5B, 0x27, 0x4E, 0x6C, 0x29, 0x67, 0x0C, 0xD4, 0xE4, 0xAB, 0x10,
  0x83, 0x19, 0x11, 0xFB, 0x1F, 0x9E, 0x0C, 0x70, 0x19, 0xDF, 0x22, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]⟩

def domainIdLit : ByteArray := ⟨#[
  0xAB, 0xE3, 0xED, 0x51, 0x3D, 0x8D, 0x5D, 0x7C, 0x1A, 0xE4, 0x17, 0xC9, 0x51, 0xBE, 0x2B, 0x3E,
  0x40, 0x1C, 0xD5, 0x9F, 0xB1, 0xA8, 0x1E, 0xAF, 0x9E, 0x22, 0x12, 0x74, 0x3A, 0x5F, 0x4E, 0x5E,
  0xAF, 0xD5, 0x57, 0xEA, 0xE5, 0xCB, 0xDB, 0xEE, 0xEF, 0x77, 0xC3, 0x64, 0xE0, 0x5A, 0xC5, 0xC6,
  0x1D, 0xF6, 0x8B, 0x3C, 0xB5, 0x0C, 0x1C, 0x72, 0x71, 0x87, 0x65, 0x8D, 0x39, 0x59, 0x68, 0x53]⟩

def saltDigestLit : ByteArray := ⟨#[
  0x87, 0xA4, 0xFA, 0xB0, 0x60, 0x47, 0x4D, 0xEC, 0x4E, 0x83, 0x06, 0x22, 0xE6, 0x87, 0x4D, 0x34,
  0x72, 0x36, 0x27, 0x40, 0xDB, 0xD9, 0x7C, 0x15, 0x54, 0xB9, 0x4A, 0x9F, 0x57, 0x26, 0xDF, 0x88,
  0x75, 0x84, 0x34, 0x4B, 0x17, 0x5A, 0x38, 0xB4, 0xAE, 0xA5, 0x7B, 0x87, 0x36, 0xBB, 0x11, 0xFE,
  0xC4, 0xD8, 0x75, 0x5C, 0x42, 0x42, 0xC0, 0xC7, 0xB9, 0x9B, 0x23, 0x9C, 0x7C, 0x16, 0xD3, 0x3C]⟩

def leafDigestLit0 : ByteArray := ⟨#[
  0x7B, 0x62, 0x3D, 0xBD, 0x18, 0x2D, 0xF4, 0x76, 0x58, 0x2F, 0x13, 0x9B, 0x9D, 0x6F, 0xF0, 0x03,
  0x20, 0xEC, 0x26, 0xBD, 0xB3, 0xB0, 0x15, 0xF1, 0xA8, 0x8D, 0xE0, 0xFB, 0x47, 0x7B, 0x00, 0xF1,
  0x56, 0x69, 0x9D, 0xB4, 0x89, 0xF6, 0xE1, 0x0C, 0x22, 0x77, 0x1D, 0xEA, 0xD1, 0x15, 0xBD, 0x78,
  0x1D, 0x4F, 0x3F, 0x5A, 0xD3, 0xBA, 0x74, 0xD4, 0xC0, 0xAF, 0xBB, 0x51, 0x23, 0xF8, 0xB2, 0x03]⟩

def leafDigestLit1 : ByteArray := ⟨#[
  0xBF, 0xE2, 0xB4, 0xFA, 0xC8, 0xAC, 0x44, 0x35, 0xCF, 0x10, 0xCB, 0xD3, 0x99, 0xE3, 0xCD, 0x98,
  0x9D, 0x83, 0x44, 0x7E, 0x04, 0x6E, 0x51, 0x6E, 0x3D, 0x4C, 0xA7, 0xC0, 0x7A, 0x2B, 0x61, 0xD3,
  0xBE, 0x34, 0xF7, 0xC3, 0x0D, 0x55, 0x46, 0x8E, 0x9C, 0x3F, 0x65, 0xD9, 0xF0, 0x78, 0xD8, 0x35,
  0x5D, 0xF2, 0x8D, 0x37, 0xAC, 0xB5, 0xA0, 0xB4, 0x59, 0x1E, 0x25, 0x51, 0xE8, 0xCE, 0x6F, 0x9B]⟩

def frameDigestLit0 : ByteArray := ⟨#[
  0x9B, 0x77, 0x2B, 0x80, 0x0B, 0xC2, 0x37, 0x31, 0x6D, 0x2B, 0xB7, 0x32, 0x01, 0xBB, 0xFA, 0xEE,
  0x14, 0x74, 0x4E, 0x42, 0x53, 0x2C, 0xC2, 0x04, 0xDB, 0x67, 0x1C, 0xFF, 0xA9, 0x65, 0x95, 0xCC,
  0x96, 0xC2, 0xA2, 0xB4, 0x70, 0xDE, 0x9C, 0x76, 0x0D, 0xCA, 0x10, 0xDF, 0x75, 0xA5, 0x7A, 0xC2,
  0x21, 0xA0, 0x3D, 0x36, 0x36, 0xA8, 0x2A, 0xA5, 0xBA, 0x9F, 0xF7, 0xD3, 0xBC, 0xFB, 0xF4, 0xF4]⟩

def frameDigestLit1 : ByteArray := ⟨#[
  0x3F, 0xB9, 0xBF, 0x3A, 0x78, 0x02, 0x5D, 0x0E, 0x6D, 0x86, 0x5D, 0xBB, 0x59, 0xE1, 0x5C, 0xA5,
  0xFC, 0xB9, 0x12, 0x90, 0xA5, 0xEF, 0x90, 0x9D, 0x07, 0xD0, 0xC1, 0xF5, 0x9D, 0x40, 0x68, 0x0C,
  0x1F, 0xD1, 0x30, 0xE3, 0xA0, 0x96, 0x77, 0x1F, 0x67, 0x20, 0xD7, 0x60, 0x4B, 0xA0, 0xE3, 0xF3,
  0xAE, 0x11, 0x18, 0x95, 0x9B, 0x6E, 0xC4, 0x76, 0x9D, 0xD8, 0x93, 0x07, 0xB2, 0x10, 0x7F, 0xA7]⟩

theorem leafPrefix_eq : Flock.Hm96.Default512.leafPrefix = leafPrefixLit := by decide +kernel

theorem domainId_eq : frameHash .sha512 "domain" [⟨Array.replicate 64 0⟩, Flock.FV3.uint ((-1 : Int) + 2).toNat,
    rangeIdentity .sha512 1, Flock.FV3.uint 1] = domainIdLit := by decide +kernel

theorem saltDigest_eq :
    Flock.Sha512.hash (Flock.Hm96.Default512.saltPrefix ++ ⟨Array.replicate 192 0⟩) = saltDigestLit := by decide +kernel

theorem leafDigest_eq0 : Flock.Sha512.hash (leafPrefixLit ++ (⟨Array.map (fun i =>
    (⟨#[0] ++ Array.replicate 63 0⟩ : ByteArray).get! i ^^^ (⟨Array.replicate 64 0⟩ : ByteArray).get! i) (Array.range 64)⟩ ++
      saltDigestLit)) = leafDigestLit0 := by decide +kernel

theorem leafDigest_eq1 : Flock.Sha512.hash (leafPrefixLit ++ (⟨Array.map (fun i =>
    (⟨#[1] ++ Array.replicate 63 0⟩ : ByteArray).get! i ^^^ (⟨Array.replicate 64 0⟩ : ByteArray).get! i) (Array.range 64)⟩ ++
      saltDigestLit)) = leafDigestLit1 := by decide +kernel

theorem frameDigest_eq0 : Flock.Sha512.hash (frameBytes "leaf" [domainIdLit, Flock.FV3.uint 0, Flock.FV3.uint 0,
    Flock.HmRow.ROW_SCHEMA_SEG.toUTF8, leafDigestLit0]) = frameDigestLit0 := by decide +kernel

theorem frameDigest_eq1 : Flock.Sha512.hash (frameBytes "leaf" [domainIdLit, Flock.FV3.uint 0, Flock.FV3.uint 0,
    Flock.HmRow.ROW_SCHEMA_SEG.toUTF8, leafDigestLit1]) = frameDigestLit1 := by decide +kernel

end Witness

open Witness

/-! ## `OpeningBinds` -/

/-- An hm96 tile tree of two positions (owner −1, the transcript's) accepts the opening of position 0 against the node over
its two leaves. -/
theorem openingBinds_honest :
    ∃ d root sib, rowsDomain ⟨Array.replicate 64 0⟩ (-1) 2 HM96 true = .ok d ∧
      opens (.ok d) root TILE_SCHEMA ⟨0, ⟨Array.replicate 256 0⟩, [sib]⟩ = true := by
  let v : ByteArray := ⟨Array.replicate 256 0⟩
  let f : Domain := ⟨.sha512, 2, frameHash .sha512 "domain"
    [⟨Array.replicate 64 0⟩, Flock.FV3.uint ((-1 : Int) + 2).toNat, rangeIdentity .sha512 2, Flock.FV3.uint 2]⟩
  let lf := Flock.Hm96.Default512.leaf (v.extract 0 (v.size - HM96_SALT)) (v.extract (v.size - HM96_SALT) v.size)
  let leaf0 := f.leaf 0 Flock.HmRow.ROW_SCHEMA_SEG lf
  let sib := f.leaf 1 Flock.HmRow.ROW_SCHEMA_SEG lf
  have hv : v.size = 256 := by simp [v, ByteArray.size]
  have hsize : ∀ r s w, (f.leaf r s w).size = 64 := fun _ _ _ => Commit.frameHash_size ..
  refine ⟨⟨f, true, true⟩, f.node 0 0 leaf0 sib, sib,
    rowsDomain_eq _ _ _ _ (by simp [ByteArray.size]) (by decide), ?_⟩
  have hleaf : RowsDomain.leaf ⟨f, true, true⟩ 0 TILE_SCHEMA v = .ok leaf0 := by
    simp [RowsDomain.leaf, hv, HM96_SALT, pure, Except.pure, leaf0, lf]
  have hroot : (f.node 0 0 leaf0 sib).size = 64 := by
    simp only [Domain.node, Domain.fv3, Flock.FV3.Domain.node, Flock.FV3.Scheme.node, Flock.FV3.Scheme.hash,
      Flock.FV3.hash, f, Hash.scheme, Flock.FV3.Sha512.scheme]
    exact Commit.sha512_size _
  have hb : f.hash.bytes = 64 := rfl
  have hc : f.count = 2 := rfl
  simp only [opens, RowsDomain.verify]
  rw [show ({ data := Array.replicate 256 0 } : ByteArray) = v from rfl]
  unfold Domain.verify
  simp [hb, hc, hroot, hsize, sib, hv, Commit.depth2, hleaf, bind, Except.bind, pure, Except.pure]

theorem zeros_get (n p : Nat) : (ByteArray.mk (Array.replicate n 0)).get! p = 0 := by
  simp only [ByteArray.get!, getElem!_def, Array.getElem?_replicate]
  split <;> simp_all <;> rfl

theorem witnessOpening_split (b : UInt8) :
    let v := (witnessOpening b).value
    v.extract 0 (v.size - HM96_SALT) = ⟨#[b] ++ Array.replicate 63 0⟩ ∧
      v.extract (v.size - HM96_SALT) v.size = ⟨Array.replicate 192 0⟩ := by
  have hx : (ByteArray.mk (#[b] ++ Array.replicate 63 0)).size = 64 := by simp [ByteArray.size]
  have hz : (ByteArray.mk (Array.replicate 192 0)).size = 192 := by simp [ByteArray.size]
  have hv : (witnessOpening b).value.size - HM96_SALT = 64 := by
    simp only [witnessOpening, ByteArray.size_append, hx, hz, HM96_SALT]
  refine ⟨?_, ?_⟩
  · simp only [hv]; exact ByteArray.extract_append_eq_left hx.symm
  · simp only [hv]
    exact ByteArray.extract_append_eq_right hx.symm (by simp only [witnessOpening, ByteArray.size_append, hx, hz])

/-- Two openings of one position of an hm96 tile tree, whose digests differ, open different values, and the five
messages their checks hash (one salt digest's, and each opening's hm96 leaf input and leaf frame) have five SHA-512
digests: at a verifier that accepts both, the conclusion fails. -/
theorem openingBinds_nontrivial :
    ¬ ((witnessOpening 0).value = (witnessOpening 1).value ∨
      ∃ x ∈ rowsHashed witnessDomain (witnessOpening 0), ∃ y ∈ rowsHashed witnessDomain (witnessOpening 1),
        x ≠ y ∧ Flock.Sha512.hash x = Flock.Sha512.hash y) := by
  have hd : witnessDomain = ⟨⟨.sha512, 1, frameHash .sha512 "domain"
      [⟨Array.replicate 64 0⟩, Flock.FV3.uint ((-1 : Int) + 2).toNat, rangeIdentity .sha512 1, Flock.FV3.uint 1]⟩,
      true, true⟩ := by
    simp only [witnessDomain, rowsDomain_eq ⟨Array.replicate 64 0⟩ (-1) 1 true (by simp [ByteArray.size]) (by decide)]
  have hm := mask_zero _ (zeros_get 192)
  obtain ⟨hp0, hs0⟩ := witnessOpening_split 0
  obtain ⟨hp1, hs1⟩ := witnessOpening_split 1
  have hpath : ∀ b, (witnessOpening b).path = [] := fun _ => rfl
  have hpos : ∀ b, (witnessOpening b).position = 0 := fun _ => rfl
  have distinct : ∀ x ∈ rowsHashed witnessDomain (witnessOpening 0), ∀ y ∈ rowsHashed witnessDomain (witnessOpening 1),
      x ≠ y → Flock.Sha512.hash x ≠ Flock.Sha512.hash y := by
    simp only [rowsHashed, hp0, hs0, hp1, hs1, hd, ite_true, List.nil_append, Flock.Hm96.Default512.leaf,
      Flock.Hm96.Default512.treeLeaf, Flock.Hm96.Default512.commitString, hm, Properties.Pouw.Commit.pathFrames,
      hpath, hpos, List.zipIdx_nil, List.foldl_nil, List.append_nil]
    simp only [leafPrefix_eq, domainId_eq, saltDigest_eq, leafDigest_eq0, leafDigest_eq1]
    intro x hx y hy hne
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hx hy
    rcases hx with rfl | rfl | rfl <;> rcases hy with rfl | rfl | rfl
    · exact absurd rfl hne
    all_goals simp only [saltDigest_eq, leafDigest_eq0, leafDigest_eq1, frameDigest_eq0, frameDigest_eq1] <;> decide +kernel
  rintro (h | ⟨x, hx, y, hy, hne, heq⟩)
  · rw [h, hp1] at hp0
    exact absurd hp0 (by decide +kernel)
  · exact distinct x hx y hy hne heq

/-! ## `YIsTheUnscale` -/

/-- The H200's first stored cell of y (`Rule.H200_Y`'s): its U and α give the served word 0x3F7E, which the check's `yOf`
gives, and which is the Definition's unscale. -/
theorem yIsTheUnscale_honest :
    Rule.yOf 0x478391C0 0x426D4BA7 0x448F000E = some 0x3F7E ∧
      Pouw.PearlC.unscale 0x478391C0 0x426D4BA7 0x448F000E = 0x3F7E := by
  decide

/-- A y check that accepts every served word (`acceptAllYOf`) passes the H200's first cell with its y one unit in the last
place off (0x3F7F), which is not the Definition's unscale of the cell's U and α. -/
theorem yIsTheUnscale_nontrivial :
    ¬ (acceptAllYOf 0x3F7F 0x478391C0 0x426D4BA7 0x448F000E =
      some (Pouw.PearlC.unscale 0x478391C0 0x426D4BA7 0x448F000E)) := by
  decide

/-! ## `RefusedBeforeTheCoin` -/

/-- `plan` refuses a run that retains no pass, before any coin. -/
theorem refusedBeforeTheCoin_honest : plan emptyRun = .error emptyRefusal := rfl

/-- A verdict that accepts every run (`acceptAllCheck`) answers the run `plan` refuses with an acceptance, not the refusal. -/
theorem refusedBeforeTheCoin_nontrivial :
    ¬ (acceptAllCheck emptyRun noOpening =
      { accept := false, reason := some emptyRefusal.reason, refused := emptyRefusal.refused }) := by
  intro h
  cases h

/-! ## `LoadedGivesUp`, `PlanMeans` and `AcceptMeans`

The honest pass loads in the kernel once `load` is unfolded down to the salt coin's hm96 leaf, whose mask at its zero salt
is given (`maskOr`) and whose leaf prefix is a literal (`Witness.leafPrefix_eq`): the kernel computes the rest, the
coin's opening, the salt, the weights' and records' roots, and `plan`'s commitment and draw on that pass. -/

namespace Witness

/-- `Hm96.Default512.mask`, with its value at a zero salt given (`mask_zero`): the kernel reads this in its place, since it
can't build the mask's tables. -/
def maskOr (salt : ByteArray) : ByteArray :=
  if salt = ⟨Array.replicate 192 0⟩ then ⟨Array.replicate 64 0⟩ else Flock.Hm96.Default512.mask salt

theorem mask_eq_maskOr (salt : ByteArray) : Flock.Hm96.Default512.mask salt = maskOr salt := by
  unfold maskOr
  split
  · next h => subst h; exact mask_zero _ (zeros_get 192)
  · rfl

theorem eq_ok_of_toOption {ε α : Type} {e : Except ε α} {a : α} (h : e.toOption = some a) : e = .ok a := by
  cases e <;> simp_all [Except.toOption]

deriving instance DecidableEq for Matmul, UnitRecord, Scheme, Pass, Plan

end Witness

open Witness

/-- `load` reads the honest pass from its manifest under the epoch's coins: its salt coin opens against them after the
weights' root, the salt is the coin's, and its records keep the caps and are the ones whose root it published. -/
theorem loadedGivesUp_honest : load honestCoins honestManifest = .ok honestPass := by
  apply eq_ok_of_toOption
  simp only [load, epochSalt, coinReason, leafValue, Flock.Hm96.Default512.leaf, Flock.Hm96.Default512.treeLeaf,
    Flock.Hm96.Default512.commitString, mask_eq_maskOr, leafPrefix_eq]
  decide +kernel

/-- A `load` that accepts every manifest takes a record of one 1 × 1 × 1 call with two voluntary A rows, over V-EX's cap
⌈1/64⌉ = 1. -/
theorem loadedGivesUp_nontrivial :
    ¬ (acceptAllRecord.voluntaryA.length ≤ voluntaryCap acceptAllRecord.matmul.m) := by
  decide

/-- `plan` takes the honest pass's run at one draw: the draw names its one tile. -/
theorem planMeans_honest :
    plan (honestRun 1) = .ok ⟨[honestPass], runCommitment [honestPass], honestSource, [(0, 0)], 1⟩ := by
  apply eq_ok_of_toOption
  simp only [plan, honestRun, List.mapM_cons, List.mapM_nil, loadedGivesUp_honest]
  decide +kernel

/-- A `plan` that accepts every run with `acceptAllPlan` gives a root that is not the run commitment of its passes, a
64-byte digest. -/
theorem planMeans_nontrivial : ¬ (acceptAllPlan.root = runCommitment acceptAllPlan.passes) := by
  intro h
  have := congrArg ByteArray.size h
  rw [runCommitment, Commit.taggedSize] at this
  exact absurd this (by decide)

/-- `check` accepts the honest pass's run at no draws: its pass loads, and it excludes no row, so no tile is opened. A
drawn tile is replayed on the fixed operand's noise that `check` derives itself (`fixedNoise`: 2k noise lines and R²
FP32 chains of k = 1,024), past what one kernel computation reaches. -/
theorem acceptMeans_honest : (check (honestRun 0) noOpening).accept = true := by
  have h : plan (honestRun 0) = .ok ⟨[honestPass], runCommitment [honestPass], honestSource, [], 0⟩ := by
    apply eq_ok_of_toOption
    simp only [plan, honestRun, List.mapM_cons, List.mapM_nil, loadedGivesUp_honest]
    decide +kernel
  simp only [check, h]
  decide +kernel

/-- A verdict that accepts every run (`acceptAllCheck`) holds no plan, so none whose tiles it opened. -/
theorem acceptMeans_nontrivial : ∀ pl, ¬ ((acceptAllCheck emptyRun noOpening).plan = some pl) := by
  intro pl h
  cases h

/-! ## A tile: `TileAcceptMeans` and `TileKeepsTheCap` -/

/-- The test tile passes `checkTile` at the test noise: its A row, its B row and its leaf open against the test pass's
roots, the words formed from its rows are its leaf's, and the rule keeps its cap. `TileAcceptMeans` ranges over every
noise `wn`: the deployed check's is `fixedNoise` of the pass's salt and B's root (`noiseOf`), one instance, and this is
another, the test noise `testNoise`, given rather than derived (deriving it is 2k = 256 noise lines and 32 × 32 FP32
chains of k = 128). `opens` hands `RowsDomain.leaf` to the tree's check unapplied, so the unfolding reaches its hm96 leaf
with `unfoldPartialApp`. -/
theorem tileAcceptMeans_honest : checkTile testPass (.ok testNoise) 0 testOpening = none := by
  simp +unfoldPartialApp only [checkTile, opens, RowsDomain.verify, RowsDomain.leaf, Flock.Hm96.Default512.leaf,
    Flock.Hm96.Default512.treeLeaf, Flock.Hm96.Default512.commitString, mask_eq_maskOr, leafPrefix_eq]
  decide +kernel

/-- A tile check that accepts every tile passes tile 0 of a pass with no calls, which is no tile of the verifier's own. -/
theorem tileAcceptMeans_nontrivial :
    ∀ pos rows cols (_ops : Words.Ops) (_ch : Words.Checked), ¬ (noCallsPass.tile? 0 = some (pos, rows, cols)) := by
  intro pos rows cols _ _ h
  cases h

/-- The rule passes the tile of one 1 × 128 × 1 call whose two rows are credited: both rows keep the per-row rules, and each
of the word's four atoms changes its total, so none is flagged and the replayed debit is 0. -/
theorem tileKeepsTheCap_honest :
    Rule.checkOpened capScheme capRecord [0] [0] capOps = none ∧ (capRecord.credited [0] [0]).1 ≠ [] ∧
      (capRecord.credited [0] [0]).2 ≠ [] := by
  refine ⟨by decide +kernel, by decide, by decide⟩

/-- A rule that accepts every tile passes the tile of `capRecord` whose word has 32 codes, less than one promotion group,
on which the debit replay refuses, so no debit is replayed for its credited cell. -/
theorem tileKeepsTheCap_nontrivial :
    ∀ words, ¬ ((capRecord.credited [0] [0]).1.mapM (fun i => (capRecord.credited [0] [0]).2.mapM fun j =>
      Rule.wordDebit shortOps.a.codes[[0].idxOf i]! shortOps.b.codes[[0].idxOf j]!) = .ok words) := by
  intro words h
  have : ((capRecord.credited [0] [0]).1.mapM (fun i => (capRecord.credited [0] [0]).2.mapM fun j =>
      Rule.wordDebit shortOps.a.codes[[0].idxOf i]! shortOps.b.codes[[0].idxOf j]!)).toOption = none := by
    decide +kernel
  rw [h] at this
  cases this

end Properties.Pouw.PearlPlus
