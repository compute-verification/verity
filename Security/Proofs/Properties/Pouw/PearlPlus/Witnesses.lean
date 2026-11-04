import Properties.Pouw.PearlPlus
import Properties.Pouw.PearlPlus.Witnesses
import Proofs.Properties.Pouw.Commit
import Std.Data.String.ToNat

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
is given (`maskOr`) and whose leaf prefix is a literal (`Witness.leafPrefix_eq`), to the salt's and the draw's `derive`,
whose sorted contexts are given (`qsort_salt`, `qsort_run`), and to the weight ids' `String.toNat?`, whose value on digits
is given (`toNatOr`): the kernel computes the rest, the coin's opening, the salt, the weights' and records' roots, and
`plan`'s commitment and draw on that pass. -/

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

/-- `String.toNat?`, with its value on a string of digits given: the kernel reads this in its place, since it can't run
`String.toNat?`'s slice iteration. -/
def toNatOr (s : String) : Option Nat :=
  if s.toList ≠ [] ∧ ∀ c ∈ s.toList, c.isDigit = true then some (Nat.ofDigitChars 10 s.toList 0) else s.toNat?

theorem toNat?_eq_toNatOr (s : String) : s.toNat? = toNatOr s := by
  unfold toNatOr
  split
  · next h =>
    rw [String.toNat?_eq_some_ofDigitChars (String.isNat_of_isDigit (fun h' => h.1 (by simp [h'])) h.2),
      List.filter_eq_self.2 fun c hc => by
        have : c ≠ '_' := by rintro rfl; exact absurd (h.2 _ hc) (by decide)
        simpa using this]
  · rfl

/-! `derive` sorts its context by key with `Array.qsort`, whose recursion the kernel can't unfold: `cbv` sorts the two
contexts the witnesses derive under, the epoch salt's and the run draw's. -/

theorem qsort_salt (x y z : Part) :
    ([("epoch", x), ("coins", y), ("weights", z)].toArray.qsort fun a b => decide (a.1 < b.1)) =
      #[("coins", y), ("epoch", x), ("weights", z)] := by
  cbv

theorem qsort_run (x : Part) : ([("run", x)].toArray.qsort fun a b => decide (a.1 < b.1)) = #[("run", x)] := by
  cbv

deriving instance DecidableEq for Matmul, UnitRecord, Scheme, Pass, Plan

end Witness

open Witness

/-- `load` reads the honest pass from its manifest under the epoch's coins: its salt coin opens against them after the
weights' root, the salt is the coin's, and its records keep the caps and are the ones whose root it published. -/
theorem loadedGivesUp_honest : load honestCoins honestManifest = .ok honestPass := by
  apply eq_ok_of_toOption
  simp only [load, epochSalt, coinReason, leafValue, Flock.Hm96.Default512.leaf, Flock.Hm96.Default512.treeLeaf,
    Flock.Hm96.Default512.commitString, mask_eq_maskOr, leafPrefix_eq, saltOf, key?, FlockSoundness.Randomness.derive,
    qsort_salt, toNat?_eq_toNatOr]
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
  simp only [plan, honestRun, List.mapM_cons, List.mapM_nil, loadedGivesUp_honest, PouwCheck.PearlPlus.drawKey, key?,
    FlockSoundness.Randomness.derive, qsort_run]
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

namespace Witness

/-- `checkTile` passes a tile on which each of its checks passes, at a call that commits no y: the tile is the
verifier's own, the openings cover it and open against their roots, the words formed from the opened rows at the noise
`wn'` are `ch`, their leaf is the opened one, and the rule passes. Each hypothesis is one computation the kernel does on
its own. -/
theorem checkTile_eq_none {p : Pass} {wn : Except String Words.FixedNoise} {tid : Nat} {o : TileOpening} {pos : Nat}
    {rows cols : List Nat} {wn' : Words.FixedNoise} {ops : Words.Ops} {ch : Words.Checked}
    (ht : p.tile? tid = some (pos, rows, cols))
    (hc : (o.a.map (·.position) != rows || o.b.map (·.position) != cols) = false)
    (ha : o.a.all (opens (rowsDomain (activationBinding p.salt p.records[pos]!.matmul.index)
      p.records[pos]!.matmul.index p.records[pos]!.matmul.m HM96 false) p.rootsA[pos]! ROW_SCHEMA) = true)
    (hb : o.b.all (opens (rowsDomain (weightBinding p.records[pos]!.matmul.weight) (-2) p.records[pos]!.matmul.n HM96
      false) ((p.weightRoots.lookup p.records[pos]!.matmul.weight).getD ByteArray.empty) ROW_SCHEMA) = true)
    (hl : (o.leaf.position != tid || !opens (transcriptDomain p.salt p.weightsRoot p.count HM96) p.transcriptRoot
      TILE_SCHEMA o.leaf) = false)
    (hwn : wn = .ok wn')
    (he : Words.encode p.salt ((p.weightRoots.lookup p.records[pos]!.matmul.weight).getD ByteArray.empty) wn'
      p.records[pos]!.matmul.index p.records[pos]!.matmul.k rows (o.a.map (openedValue ·.value)) cols
      (o.b.map (openedValue ·.value)) = .ok ops)
    (hch : Words.checked ops = .ok ch)
    (hleaf : (Words.leaf ch != openedValue o.leaf.value) = false)
    (hr : Rule.checkOpened p.scheme p.records[pos]! rows cols ops = none)
    (hy : p.yRoots[pos]! = none) :
    checkTile p wn tid o = none := by
  subst hwn
  simp only [checkTile, ht, hc, ha, hb, hl, bind, Except.bind, he, hch, pure, Except.pure, hleaf, hr, hy,
    Bool.not_true, Bool.false_eq_true, ↓reduceIte]
  rfl

deriving instance DecidableEq for Words.Formed, Words.Ops, Words.Checked

/-- The test tile's formed operands and peel factors as the Python forms them (`PearlC.encode`): A's and B's codes,
α, ρ, E′ and FP32 words, and P_A's and P_B's rows of BF16 words. -/
def testOps : Words.Ops where
  a :=
    { codes := #[
          #[0x6a, 0xe3, 0xe5, 0xf0, 0xe4, 0xe1, 0x69, 0x5d, 0x70, 0x72, 0x6a, 0xf0, 0xd9, 0xe2, 0xec, 0x54,
            0xe0, 0x60, 0xb3, 0x56, 0xee, 0x6a, 0x60, 0x6d, 0x41, 0xe0, 0xe0, 0x76, 0x6a, 0x76, 0x7a, 0xea,
            0x6c, 0x67, 0x73, 0x5a, 0xd8, 0x6d, 0xf2, 0x66, 0xe6, 0xe0, 0xea, 0x55, 0x2f, 0xe6, 0x60, 0x5b,
            0xf0, 0x59, 0xed, 0x56, 0x62, 0x64, 0xe1, 0x62, 0xe5, 0xe1, 0x5e, 0xe5, 0x70, 0xdf, 0xe3, 0xee,
            0x71, 0x73, 0xf4, 0x69, 0xf1, 0x74, 0xea, 0x72, 0x64, 0xec, 0x6e, 0x72, 0x68, 0x69, 0xe1, 0xe7,
            0xd2, 0x30, 0x64, 0x5f, 0x53, 0xef, 0xed, 0x6b, 0x6e, 0x75, 0xeb, 0x73, 0xdb, 0xf0, 0x5c, 0xf6,
            0x67, 0xe5, 0x3e, 0x41, 0xf1, 0xe3, 0xe2, 0x6f, 0x5d, 0x60, 0x59, 0xed, 0x69, 0x69, 0xeb, 0x61,
            0x65, 0xee, 0xf6, 0x66, 0x4e, 0x5a, 0x6a, 0x75, 0xe2, 0x70, 0x67, 0x72, 0x6f, 0xe4, 0x6b, 0x72]],
      alpha := #[0x42a5f708], rho := #[0x3f4584ca],
      ep := #[
          #[0xc9, 0xba, 0xb8, 0x45, 0x43, 0xcd, 0xc6, 0xcb, 0xca, 0x40, 0x4b, 0x48, 0x23, 0x3a, 0x4a, 0x4d,
            0xc6, 0xb5, 0x48, 0x43, 0x3b, 0xcb, 0x4b, 0xbe, 0xc6, 0xb9, 0x4d, 0xc6, 0xc2, 0x4a, 0x45, 0x4b]],
      x := #[
          #[0xbed60000, 0x3ea70000, 0xc0080000, 0xbfb90000, 0x3f4b0000, 0xbf170000, 0x3f140000, 0x3f0a0000,
            0x3fa90000, 0x3f4f0000, 0x3f820000, 0xbde40000, 0xbf320000, 0xbf3b0000, 0xbef90000, 0xbf900000,
            0xbf0c0000, 0xbdbd0000, 0x3e800000, 0xbead0000, 0xbff60000, 0xbd940000, 0x3e660000, 0x3f8a0000,
            0x3f130000, 0xbf240000, 0xbf390000, 0x40000000, 0x3f410000, 0x3fea0000, 0x40080000, 0xbf510000,
            0x3ec50000, 0x3eea0000, 0x3f0f0000, 0x3f0a0000, 0x3e4e0000, 0x3e320000, 0xbfc00000, 0xbe290000,
            0xbf3f0000, 0x3e010000, 0xbeef0000, 0x3f1e0000, 0x3f510000, 0x3e9e0000, 0x3ea10000, 0x3dbe0000,
            0xbee50000, 0xbe280000, 0xbefd0000, 0x3ec60000, 0x3c670000, 0x3f140000, 0xbfaa0000, 0x3f630000,
            0xbf430000, 0xbf3b0000, 0xbe4a0000, 0xbf100000, 0x3e950000, 0xbf120000, 0xbf880000, 0xbf580000,
            0x3fa70000, 0x3d350000, 0xbf950000, 0x3c090000, 0xbfc70000, 0x3fe50000, 0xbfc30000, 0x3ef50000,
            0x3f0b0000, 0xbfb90000, 0x3e990000, 0x3f7f0000, 0x3eef0000, 0x3e850000, 0x3f720000, 0x3e240000,
            0x3eac0000, 0xbe010000, 0x3f210000, 0xbf700000, 0x3f4a0000, 0xbf010000, 0xbf8b0000, 0x3ebb0000,
            0x3fd80000, 0x3f760000, 0xbf040000, 0x3f320000, 0xbee80000, 0xbdfd0000, 0x3db70000, 0xc0140000,
            0x3e440000, 0xbf830000, 0xbf320000, 0xbfbc0000, 0xbfc20000, 0xbf710000, 0x3f530000, 0x3fd50000,
            0xbcce0000, 0x3f8b0000, 0xbe870000, 0xbff40000, 0x3e190000, 0x3ee40000, 0xbedb0000, 0x3e9a0000,
            0x3f120000, 0xbf5d0000, 0xbfbd0000, 0xbe620000, 0xbe580000, 0xbeb40000, 0x3f7c0000, 0x3fdc0000,
            0xbed50000, 0x3f320000, 0x3f700000, 0x3f360000, 0x3f860000, 0xbec70000, 0x3f3d0000, 0x40010000]] }
  b :=
    { codes := #[
          #[0x49, 0xd1, 0x6f, 0x56, 0xd4, 0xce, 0x5d, 0xe9, 0x69, 0xf5, 0x6b, 0xea, 0x5e, 0xdf, 0xeb, 0x2e,
            0x72, 0x5a, 0xf3, 0x64, 0xe9, 0xf2, 0xe1, 0xe0, 0xee, 0xe3, 0xdf, 0xf1, 0xe7, 0xd9, 0x60, 0xe7,
            0xea, 0x6d, 0x5a, 0xdc, 0x65, 0x67, 0xe6, 0xd3, 0x68, 0xe9, 0x6a, 0xf2, 0x66, 0xc5, 0xe8, 0xec,
            0x52, 0xe8, 0xd3, 0x73, 0xed, 0x67, 0x70, 0xef, 0xe2, 0xdb, 0xf4, 0x4c, 0x65, 0x5b, 0x6b, 0xde,
            0x5c, 0xee, 0xd0, 0x39, 0x67, 0xed, 0xd8, 0x56, 0xdf, 0x59, 0x4e, 0x6a, 0xf2, 0x68, 0x6e, 0x67,
            0xeb, 0x67, 0xf4, 0xe6, 0xc8, 0xed, 0xe8, 0xe1, 0x44, 0x6b, 0x70, 0xf2, 0xe2, 0x72, 0x74, 0x6f,
            0x5b, 0x67, 0xe6, 0xe4, 0x53, 0x3d, 0x6d, 0x57, 0x3e, 0xf1, 0x6c, 0x64, 0x55, 0xee, 0x75, 0x6e,
            0x72, 0xe6, 0xdc, 0xef, 0x6b, 0x74, 0xea, 0x62, 0x6b, 0xeb, 0x4d, 0xcf, 0xe1, 0x6f, 0xce, 0xad]],
      alpha := #[0x429201a5], rho := #[0x3f64bff6],
      ep := #[
          #[0x32, 0xca, 0xca, 0xbc, 0x44, 0xbb, 0x4b, 0x41, 0xcc, 0xc4, 0xcc, 0x44, 0x98, 0xbe, 0x41, 0xc9,
            0x4c, 0xc6, 0xc6, 0xca, 0x40, 0xc6, 0x44, 0xce, 0x4d, 0xbd, 0xc1, 0xa4, 0xbb, 0xc9, 0xcb, 0xcd]],
      x := #[
          #[0x3d0c0000, 0x3fae0000, 0x3f9c0000, 0xbf020000, 0xbe980000, 0xbf070000, 0x3f110000, 0xbd650000,
            0x3f3f0000, 0xbfec0000, 0x3fc80000, 0xbdc50000, 0x3f2e0000, 0xbe0b0000, 0xbec20000, 0x3eed0000,
            0x3f530000, 0xbe4f0000, 0xbe1c0000, 0x3f2f0000, 0xbf5e0000, 0xbfc10000, 0x3eca0000, 0xbf2b0000,
            0xbff50000, 0xbf500000, 0xbeef0000, 0xbf980000, 0xbfbf0000, 0x3d160000, 0x3f650000, 0xbe6e0000,
            0xbf3e0000, 0x3ec50000, 0x3f370000, 0xbe990000, 0x3f0b0000, 0x3f850000, 0xbe530000, 0xbf500000,
            0x3eb10000, 0x3e7d0000, 0x3f8c0000, 0xbfa40000, 0xbf290000, 0xbf560000, 0xbfdd0000, 0x3e010000,
            0x3f070000, 0xbf3d0000, 0x3fb10000, 0x3f520000, 0x3f200000, 0x3ecd0000, 0x3f740000, 0xbfaa0000,
            0x3f1d0000, 0x3f1a0000, 0xbfe20000, 0x3eb10000, 0xbe800000, 0x3f480000, 0xbee00000, 0xbc950000,
            0x3eaf0000, 0xbf600000, 0x3f190000, 0xbdd60000, 0x3efc0000, 0xbf050000, 0x3f8b0000, 0x3f1a0000,
            0xbe360000, 0x3f210000, 0x3fa10000, 0x3fe50000, 0xbfc90000, 0x3f620000, 0x3eee0000, 0xbdc00000,
            0xbf800000, 0x3fa00000, 0xbfa10000, 0x3f110000, 0x3fa60000, 0xbfcc0000, 0xbe9a0000, 0xbfa70000,
            0x3e790000, 0x3fc10000, 0x40010000, 0xbfe30000, 0xbf130000, 0x3f340000, 0x3fca0000, 0x3ed70000,
            0xbf3f0000, 0x3e980000, 0xbc880000, 0xbe500000, 0xbf3c0000, 0x3ec60000, 0x3e9d0000, 0xbdbe0000,
            0xbe630000, 0xbfa40000, 0xbef80000, 0x3f9a0000, 0xbe430000, 0xbfb80000, 0x3faa0000, 0x3f070000,
            0x40060000, 0x3d800000, 0xbeec0000, 0xbfb90000, 0x3fa90000, 0x40240000, 0xbf520000, 0xbf250000,
            0x3f180000, 0xbf540000, 0xbe8a0000, 0xbeb30000, 0x3e440000, 0x3f8c0000, 0x3cb40000, 0x3f6b0000]] }
  pa := #[
    #[0xc090, 0xbfa0, 0xbf80, 0x4050, 0x4030, 0xc0d0, 0xc060, 0xc0b0, 0xc0a0, 0x4000, 0x40b0, 0x4080,
      0x3e30, 0x3fa0, 0x40a0, 0x40d0, 0xc060, 0xbf50, 0x4080, 0x4030, 0x3fb0, 0xc0b0, 0x40b0, 0xbfe0,
      0xc060, 0xbf90, 0x40d0, 0xc060, 0xc020, 0x40a0, 0x4050, 0x40b0, 0xc4d8, 0x452d, 0xc336, 0xc599,
      0x4419, 0x45d4, 0x42bf, 0xc496, 0x452d, 0x455d, 0x447e, 0xc55d, 0x4494, 0xc585, 0x44b2, 0x4452,
      0x4553, 0x4505, 0x44bb, 0xc50b, 0xc48f, 0x4539, 0xc3da, 0x4396, 0xc472, 0xc4be, 0x456c, 0x44d0,
      0xc541, 0xc52d, 0x4512, 0x45a0]]
  pb := #[
    #[0xc47a, 0x4347, 0x4460, 0xc443, 0xc35f, 0xc395, 0x44ac, 0x4565, 0xc584, 0xc37d, 0x4530, 0x43f1,
      0x451e, 0x450b, 0xc472, 0xc48b, 0xc40c, 0x43c0, 0xc40f, 0xc417, 0x44e4, 0xc568, 0xc51c, 0xc504,
      0xc507, 0x42b0, 0xc552, 0xc21f, 0xc51c, 0x4580, 0xc52f, 0xc544, 0xbf20, 0x40a0, 0x40a0, 0x3fc0,
      0xc040, 0x3fb0, 0xc0b0, 0xc010, 0x40c0, 0x4040, 0x40c0, 0xc040, 0x3d80, 0x3fe0, 0xc010, 0x4090,
      0xc0c0, 0x4060, 0x4060, 0x40a0, 0xc000, 0x4060, 0xc040, 0x40e0, 0xc0d0, 0x3fd0, 0x4010, 0x3e40,
      0x3fb0, 0x4090, 0x40b0, 0x40d0]]

/-- The test tile's checked words, C̃ and U (`PearlC.checked`). -/
def testChecked : Words.Checked where
  c := #[
    #[0xc8374000]]
  u := #[
    #[0xc7b318d0]]

/-! The test tile's checks, one kernel computation each. `opens` hands `RowsDomain.leaf` to the tree's check unapplied, so
the unfolding reaches its hm96 leaf with `unfoldPartialApp`. -/

theorem test_tile : testPass.tile? 0 = some (0, [0], [0]) := by decide +kernel

theorem test_covers : (testOpening.a.map (·.position) != [0] || testOpening.b.map (·.position) != [0]) = false := by
  decide +kernel

theorem test_opensA : testOpening.a.all (opens (rowsDomain (activationBinding testPass.salt
    testPass.records[0]!.matmul.index) testPass.records[0]!.matmul.index testPass.records[0]!.matmul.m HM96 false)
    testPass.rootsA[0]! ROW_SCHEMA) = true := by
  simp +unfoldPartialApp only [opens, RowsDomain.verify, RowsDomain.leaf, Flock.Hm96.Default512.leaf,
    Flock.Hm96.Default512.treeLeaf, Flock.Hm96.Default512.commitString, mask_eq_maskOr, leafPrefix_eq]
  decide +kernel

theorem test_opensB : testOpening.b.all (opens (rowsDomain (weightBinding testPass.records[0]!.matmul.weight) (-2)
    testPass.records[0]!.matmul.n HM96 false)
    ((testPass.weightRoots.lookup testPass.records[0]!.matmul.weight).getD ByteArray.empty) ROW_SCHEMA) = true := by
  simp +unfoldPartialApp only [opens, RowsDomain.verify, RowsDomain.leaf, Flock.Hm96.Default512.leaf,
    Flock.Hm96.Default512.treeLeaf, Flock.Hm96.Default512.commitString, mask_eq_maskOr, leafPrefix_eq]
  decide +kernel

theorem test_opensLeaf : (testOpening.leaf.position != 0 || !opens (transcriptDomain testPass.salt testPass.weightsRoot
    testPass.count HM96) testPass.transcriptRoot TILE_SCHEMA testOpening.leaf) = false := by
  simp +unfoldPartialApp only [opens, RowsDomain.verify, RowsDomain.leaf, Flock.Hm96.Default512.leaf,
    Flock.Hm96.Default512.treeLeaf, Flock.Hm96.Default512.commitString, mask_eq_maskOr, leafPrefix_eq]
  decide +kernel

/-- `Words.encode` with its steps as parameters: the E line (`S`), the line key (`L`), the row seed (`RS`), the row's
parse (`P`), the forming (`F`) and the peels (`PA`, `PB`). -/
def encodeWith (S : ByteArray → Nat → Nat → Nat → Nat → Words.Reject (Array Nat)) (L : ByteArray → ByteArray)
    (RS : ByteArray → ByteArray → Nat → Nat → ByteArray → ByteArray) (P : ByteArray → Nat → Words.Reject (Array Nat))
    (F : Array (Array Nat) → Array (Array Nat) → Array (Array Nat) → Words.Reject Words.Formed)
    (PA PB : Words.FixedNoise → Words.Formed → Words.Reject (Array (Array Nat)))
    (salt rootB : ByteArray) (wn : Words.FixedNoise) (index k : Nat) (rows : List Nat) (aRows : List ByteArray)
    (cols : List Nat) (bRows : List ByteArray) : Words.Reject Words.Ops := do
  let ea ← (rows.zip aRows).toArray.mapM fun (i, r) => S (L (RS salt rootB index i r)) 0 0 0 Words.NORM_E
  let a ← F (← aRows.toArray.mapM (P · k)) ea wn.fa
  let keyB := L wn.seedB
  let eb ← cols.toArray.mapM fun j => S keyB 1 0 j Words.NORM_E
  let b ← F (← bRows.toArray.mapM (P · k)) eb wn.fb
  return ⟨a, b, ← PA wn a, ← PB wn b⟩

theorem encode_eq_encodeWith (salt rootB : ByteArray) (wn : Words.FixedNoise) (index k : Nat) (rows : List Nat)
    (aRows : List ByteArray) (cols : List Nat) (bRows : List ByteArray) :
    Words.encode salt rootB wn index k rows aRows cols bRows = encodeWith Words.sampleLine Words.lineKey Words.rowSeed
      Words.parseRow Words.formV1 Words.peelA Words.peelB salt rootB wn index k rows aRows cols bRows := rfl

/-- `encodeWith` of one A row and one B row, at any steps. -/
theorem encodeWith_one {S : ByteArray → Nat → Nat → Nat → Nat → Words.Reject (Array Nat)} {L : ByteArray → ByteArray}
    {RS : ByteArray → ByteArray → Nat → Nat → ByteArray → ByteArray} {P : ByteArray → Nat → Words.Reject (Array Nat)}
    {F : Array (Array Nat) → Array (Array Nat) → Array (Array Nat) → Words.Reject Words.Formed}
    {PA PB : Words.FixedNoise → Words.Formed → Words.Reject (Array (Array Nat))}
    {salt rootB : ByteArray} {wn : Words.FixedNoise} {index k i j : Nat} {ra rb : ByteArray}
    {ea eb xa xb : Array Nat} {a b : Words.Formed} {pa pb : Array (Array Nat)}
    (hea : S (L (RS salt rootB index i ra)) 0 0 0 Words.NORM_E = .ok ea) (hxa : P ra k = .ok xa)
    (hfa : F #[xa] #[ea] wn.fa = .ok a) (heb : S (L wn.seedB) 1 0 j Words.NORM_E = .ok eb) (hxb : P rb k = .ok xb)
    (hfb : F #[xb] #[eb] wn.fb = .ok b) (hpa : PA wn a = .ok pa) (hpb : PB wn b = .ok pb) :
    encodeWith S L RS P F PA PB salt rootB wn index k [i] [ra] [j] [rb] = .ok ⟨a, b, pa, pb⟩ := by
  simp only [encodeWith, List.zip_cons_cons, List.zip_nil_left, Array.mapM_eq_mapM_toList, List.mapM_cons,
    List.mapM_nil, hea, hxa, heb, hxb, hfa, hfb, hpa, hpb, bind, Except.bind, pure, Except.pure, Functor.map, Except.map]

/-- `encode` of one A row and one B row: each side's E line, its parsed row and its formed side, then the peel factors.
Each hypothesis is one computation the kernel does on its own. It is `encodeWith_one` at `encode`'s own steps
(`encode_eq_encodeWith`): a proof that unfolds `encode` at its steps has the kernel reduce a match on an E line
(`Words.sampleLine`) of an unknown row, which evaluates SHA-512 as far as the row lets it. -/
theorem encode_one {salt rootB : ByteArray} {wn : Words.FixedNoise} {index k i j : Nat} {ra rb : ByteArray}
    {ea eb xa xb : Array Nat} {a b : Words.Formed} {pa pb : Array (Array Nat)}
    (hea : Words.sampleLine (Words.lineKey (Words.rowSeed salt rootB index i ra)) 0 0 0 Words.NORM_E = .ok ea)
    (hxa : Words.parseRow ra k = .ok xa) (hfa : Words.formV1 #[xa] #[ea] wn.fa = .ok a)
    (heb : Words.sampleLine (Words.lineKey wn.seedB) 1 0 j Words.NORM_E = .ok eb)
    (hxb : Words.parseRow rb k = .ok xb) (hfb : Words.formV1 #[xb] #[eb] wn.fb = .ok b)
    (hpa : Words.peelA wn a = .ok pa) (hpb : Words.peelB wn b = .ok pb) :
    Words.encode salt rootB wn index k [i] [ra] [j] [rb] = .ok ⟨a, b, pa, pb⟩ := by
  rw [encode_eq_encodeWith]
  exact encodeWith_one hea hxa hfa heb hxb hfb hpa hpb

theorem size_testRowA : testRowA.size = 512 := by decide +kernel

theorem size_testRowB : testRowB.size = 512 := by decide +kernel

theorem size_zeroSalt : zeroSalt.size = HM96_SALT := by decide +kernel

/-- A row opened from `row ++ zeroSalt` is `row`: the opening drops the salt's 192 bytes. -/
theorem openedValue_salted (row : ByteArray) : openedValue (row ++ zeroSalt) = row :=
  ByteArray.extract_append_eq_left (by rw [ByteArray.size_append, size_zeroSalt, Nat.add_sub_cancel])

/-- Row A's `sha512/row-seg/v1` digest, the inner digest of its hm96 leaf and the key of its seed. -/
def rowDigestA : ByteArray :=
  beBytes 64 [0x140f6a9bd2d2b0d00682751661f9065654e014ac6608062bbd2fe6acd5bf29619737d459f93ea07519da2b55ab13234cd5298b6acbad448a0322ba066a66ae7c]

/-- Row A's seed (`Words.rowSeed`). -/
def rowSeedA : ByteArray :=
  beBytes 32 [0x84d7543fc5fbc6240cb86e24c7120d81495306013292f0b083d66d9149c1b666]

/-- The key of row A's E line (`Words.lineKey` of its seed). -/
def lineKeyA : ByteArray :=
  beBytes 32 [0x9f31bc07d278c3502ee6ffab08f73ce847b175833166316570371968113900d7]

/-- The key of B's E lines (`Words.lineKey` of seed_B). -/
def lineKeyB : ByteArray :=
  beBytes 32 [0xdec3da263f57eff0219219320eca345ad8292b1475b9b5c621d864115e2ece4e]

/-- Row A's E line: 32 E4M3 codes at norm 256. -/
def eA : Array Nat := #[
  0xe5, 0xd6, 0xd3, 0x61, 0x60, 0xe9, 0xe2, 0xe8, 0xe6, 0x5b, 0x67, 0x64, 0x3f, 0x56, 0x66, 0x69,
  0xe2, 0xd1, 0x63, 0x60, 0x58, 0xe8, 0x68, 0xda, 0xe2, 0xd5, 0x69, 0xe2, 0xde, 0x66, 0x61, 0x67]

/-- Column 0's E line of B: 32 E4M3 codes at norm 256. -/
def eB : Array Nat := #[
  0x4e, 0xe6, 0xe6, 0xd8, 0x60, 0xd7, 0x67, 0x5c, 0xe8, 0xe0, 0xe8, 0x60, 0xb3, 0xda, 0x5c, 0xe5,
  0x68, 0xe2, 0xe2, 0xe6, 0x5b, 0xe2, 0x60, 0xea, 0x69, 0xd9, 0xdc, 0xc0, 0xd7, 0xe4, 0xe7, 0xe9]

theorem test_digestA : Flock.HmRow.segDigest (8 * testRowA.size) testRowA = rowDigestA := by decide +kernel

theorem test_seedA : Words.rowSeed testSalt testRootB 0 0 testRowA = rowSeedA := by
  rw [Words.rowSeed, test_digestA]
  decide +kernel

theorem test_keyA : Words.lineKey rowSeedA = lineKeyA := by decide +kernel

theorem test_lineA : Words.sampleLine (Words.lineKey (Words.rowSeed testSalt testRootB 0 0 testRowA)) 0 0 0
    Words.NORM_E = .ok eA := by
  rw [test_seedA, test_keyA]
  apply eq_ok_of_toOption
  decide +kernel

theorem test_parseA : Words.parseRow testRowA 128 = .ok testOps.a.x[0]! := by
  apply eq_ok_of_toOption
  decide +kernel

theorem test_formA : Words.formV1 #[testOps.a.x[0]!] #[eA] testNoise.fa = .ok testOps.a := by
  apply eq_ok_of_toOption
  decide +kernel

theorem test_keyB : Words.lineKey testNoise.seedB = lineKeyB := by decide +kernel

theorem test_lineB : Words.sampleLine (Words.lineKey testNoise.seedB) 1 0 0 Words.NORM_E = .ok eB := by
  rw [test_keyB]
  apply eq_ok_of_toOption
  decide +kernel

theorem test_parseB : Words.parseRow testRowB 128 = .ok testOps.b.x[0]! := by
  apply eq_ok_of_toOption
  decide +kernel

theorem test_formB : Words.formV1 #[testOps.b.x[0]!] #[eB] testNoise.fb = .ok testOps.b := by
  apply eq_ok_of_toOption
  decide +kernel

theorem test_peelA : Words.peelA testNoise testOps.a = .ok testOps.pa := by
  apply eq_ok_of_toOption
  decide +kernel

theorem test_peelB : Words.peelB testNoise testOps.b = .ok testOps.pb := by
  apply eq_ok_of_toOption
  decide +kernel

theorem test_encode : Words.encode testPass.salt
    ((testPass.weightRoots.lookup testPass.records[0]!.matmul.weight).getD ByteArray.empty) testNoise
    testPass.records[0]!.matmul.index testPass.records[0]!.matmul.k [0] (testOpening.a.map (openedValue ·.value)) [0]
    (testOpening.b.map (openedValue ·.value)) = .ok testOps := by
  have hs : testPass.salt = testSalt := rfl
  have hr : (testPass.weightRoots.lookup testPass.records[0]!.matmul.weight).getD ByteArray.empty = testRootB := rfl
  have hi : testPass.records[0]!.matmul.index = 0 := rfl
  have hk : testPass.records[0]!.matmul.k = 128 := rfl
  have ha : testOpening.a.map (openedValue ·.value) = [testRowA] := by
    simp only [testOpening, List.map_cons, List.map_nil, openedValue_salted]
  have hb : testOpening.b.map (openedValue ·.value) = [testRowB] := by
    simp only [testOpening, List.map_cons, List.map_nil, openedValue_salted]
  rw [hs, hr, hi, hk, ha, hb]
  exact encode_one test_lineA test_parseA test_formA test_lineB test_parseB test_formB test_peelA test_peelB

theorem test_checked : Words.checked testOps = .ok testChecked := by
  apply eq_ok_of_toOption
  decide +kernel

theorem test_leaf : (Words.leaf testChecked != openedValue testOpening.leaf.value) = false := by decide +kernel

theorem test_rule : Rule.checkOpened testPass.scheme testPass.records[0]! [0] [0] testOps = none := by decide +kernel

theorem test_y : testPass.yRoots[0]! = none := rfl

end Witness

/-- The test tile passes `checkTile` at the test noise: its A row, its B row and its leaf open against the test pass's
roots, the words formed from its rows are its leaf's, and the rule keeps its cap. `TileAcceptMeans` ranges over every
noise `wn`: the deployed check's is `fixedNoise` of the pass's salt and B's root (`noiseOf`), one instance, and this is
another, the test noise `testNoise`, given rather than derived (deriving it is 2k = 256 noise lines and 32 × 32 FP32
chains of k = 128). Each check is its own kernel computation (`Witness.checkTile_eq_none`), the formed words and the
checked words given as the Python computes them (`Witness.testOps`, `Witness.testChecked`). -/
theorem tileAcceptMeans_honest : checkTile testPass (.ok testNoise) 0 testOpening = none :=
  checkTile_eq_none test_tile test_covers test_opensA test_opensB test_opensLeaf rfl test_encode test_checked test_leaf
    test_rule test_y

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
