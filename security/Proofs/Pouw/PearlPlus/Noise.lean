module

public import PouwPearlPlus.Words
public import Definitions.Pouw.PearlC.NoiseSha512
public import Proofs.Flock.Soundness.Discharge.Hm.Hash
public import Mathlib.Data.List.GetD
public import Mathlib.Data.List.Basic
public import Mathlib.Tactic.Ring
public import Mathlib.Tactic.NormNum
public import Mathlib.Tactic.Linarith
public import Mathlib.Tactic.FieldSimp
public import Mathlib.Tactic.Zify
public import Mathlib.Data.Nat.Sqrt
public import Mathlib.Data.List.Monad
public import Mathlib.Algebra.Order.BigOperators.Group.List
public import Proofs.Pouw.PearlC.CoreOps
public import Proofs.Pouw.PearlC.Peel
public import Proofs.Pouw.PearlC.PeelExactProofs
public import Proofs.Properties.Core.Evaluate
import all Init.Data.String.Defs

/-!
# Pearl+'s bridge, part (b), the noise: the check's -h3 noise is `NoiseSha512`'s

The check hashes with the compiled SHA-512 every tree reads (`Flock.Sha512.hash`) and keys a row's seed by its
`sha512/row-seg/v1` digest (`Flock.HmRow.segDigest`). The deployed semantics (`sha512Sem`) samples the same lines through the
spec's SHA-512 (`Pouw.PearlC.Sha512.digest`) and row digest (`NoiseSha512.rowDigest`). Each statement here says the
check's value is the spec's on the same bytes, and where the check returns a value, the spec's line or seed is it.
-/

@[expose] public section

namespace Pouw.PearlPlus.Bridge

open PouwCheck.PearlPlus

/-- A byte array's bytes, as the spec's lists of naturals. -/
def bytesNat (b : ByteArray) : List ℕ := b.data.toList.map UInt8.toNat

/-- **The compiled SHA-512 is the spec's.** -/
def HashIsTheSpec : Prop :=
  ∀ m : ByteArray, bytesNat (Flock.Sha512.hash m) = Pouw.PearlC.Sha512.digest (bytesNat m)

/-- **`sha512/row-seg/v1` on whole bytes is the spec's row digest.** -/
def SegDigestIsRowDigest : Prop :=
  ∀ r : ByteArray, bytesNat (Flock.HmRow.segDigest (8 * r.size) r) = Pouw.PearlC.NoiseSha512.rowDigest (bytesNat r)

/-- The roles -h3 hashes under. -/
def roles : List String := ["seed-A", "seed-B", "noise-line", "line"]

/-- **-h3's labelled hash and its seeds are the spec's.** -/
def SeedsAreTheSpec : Prop :=
  (∀ role ∈ roles, ∀ m : ByteArray,
    bytesNat (Words.s512 role m) = Pouw.PearlC.NoiseSha512.s512 role (bytesNat m)) ∧
  (∀ salt rootB : ByteArray,
    bytesNat (Words.seedB salt rootB) = Pouw.PearlC.NoiseSha512.seedB (bytesNat salt) (bytesNat rootB)) ∧
  ∀ (salt rootB row : ByteArray) (u i : ℕ), bytesNat (Words.rowSeed salt rootB u i row) =
    Pouw.PearlC.NoiseSha512.rowSeed (bytesNat salt) (bytesNat row) (bytesNat rootB) u i

/-- **A noise line is the spec's**, at E's norm 256 and F's line norm 16: where the check's line is a value, its 32 codes
are `NoiseSha512.sampleLine`'s. -/
def LineIsTheSpec : Prop :=
  ∀ (seed : ByteArray) (side factor line norm32 : ℕ) (norm : ℚ) (codes : Array ℕ), side < 256 → factor < 256 →
    (norm32 = Words.NORM_E ∧ norm = 256 ∨ norm32 = Words.NORM_F ∧ norm = 16) →
    Words.sampleLine (Words.lineKey seed) side factor line norm32 = .ok codes →
      codes.size = 32 ∧ ∀ t : Fin Pouw.PearlC.Params.pi.r,
        codes[t.val]! = (Pouw.PearlC.NoiseSha512.sampleLine (bytesNat seed) side factor line norm t).val

/-- **The fixed operand's noise is `unitNoise`'s**: where `fixedNoise` is a value, its seed is the spec's `seed_B` and its
k rows of `F_A` and `F_B` are the spec's basis lines. -/
def FixedNoiseIsTheSpec : Prop :=
  ∀ (salt rootB : ByteArray) (k : ℕ) (wn : Words.FixedNoise), Words.fixedNoise salt rootB k = .ok wn →
    bytesNat wn.seedB = Pouw.PearlC.NoiseSha512.seedB (bytesNat salt) (bytesNat rootB) ∧
      wn.fa.size = k ∧ wn.fb.size = k ∧ ∀ l < k, ∀ t : Fin Pouw.PearlC.Params.pi.r,
        (wn.fa[l]!)[t.val]! = (Pouw.PearlC.NoiseSha512.sampleLine (bytesNat wn.seedB) 0 1 l 16 t).val ∧
          (wn.fb[l]!)[t.val]! = (Pouw.PearlC.NoiseSha512.sampleLine (bytesNat wn.seedB) 1 1 l 16 t).val

namespace NoiseProofs

open FlockSoundness.Discharge

/-! ## The compression on 64-bit vectors is the spec's on naturals -/

theorem rotr_toNat (x : BitVec 64) (n : ℕ) (hn : n < 64) :
    (x.rotateRight n).toNat = Pouw.PearlC.Sha512.rotr x.toNat n := by
  rw [BitVec.toNat_rotateRight, Nat.mod_eq_of_lt hn]
  rfl

theorem bsig0_toNat (x : BitVec 64) : (Hm.bsig0 x).toNat = Pouw.PearlC.Sha512.bsig0 x.toNat := by
  rw [Hm.bsig0, BitVec.toNat_xor, BitVec.toNat_xor, rotr_toNat _ 28 (by norm_num), rotr_toNat _ 34 (by norm_num),
    rotr_toNat _ 39 (by norm_num)]
  rfl

theorem bsig1_toNat (x : BitVec 64) : (Hm.bsig1 x).toNat = Pouw.PearlC.Sha512.bsig1 x.toNat := by
  rw [Hm.bsig1, BitVec.toNat_xor, BitVec.toNat_xor, rotr_toNat _ 14 (by norm_num), rotr_toNat _ 18 (by norm_num),
    rotr_toNat _ 41 (by norm_num)]
  rfl

theorem ssig0_toNat (x : BitVec 64) : (Hm.ssig0 x).toNat = Pouw.PearlC.Sha512.ssig0 x.toNat := by
  rw [Hm.ssig0, BitVec.toNat_xor, BitVec.toNat_xor, rotr_toNat _ 1 (by norm_num), rotr_toNat _ 8 (by norm_num),
    BitVec.toNat_ushiftRight]
  rfl

theorem ssig1_toNat (x : BitVec 64) : (Hm.ssig1 x).toNat = Pouw.PearlC.Sha512.ssig1 x.toNat := by
  rw [Hm.ssig1, BitVec.toNat_xor, BitVec.toNat_xor, rotr_toNat _ 19 (by norm_num), rotr_toNat _ 61 (by norm_num),
    BitVec.toNat_ushiftRight]
  rfl

theorem chV_toNat (e f g : BitVec 64) : (Hm.chV e f g).toNat = Pouw.PearlC.Sha512.ch e.toNat f.toNat g.toNat := by
  rw [Hm.chV, ← BitVec.allOnes_xor, BitVec.toNat_xor, BitVec.toNat_and, BitVec.toNat_and, BitVec.toNat_xor,
    BitVec.toNat_allOnes, Pouw.PearlC.Sha512.ch, Pouw.PearlC.Sha512.compl, Nat.mod_eq_of_lt e.isLt]

theorem majV_toNat (a b c : BitVec 64) : (Hm.majV a b c).toNat = Pouw.PearlC.Sha512.maj a.toNat b.toNat c.toNat := by
  rw [Hm.majV, BitVec.toNat_xor, BitVec.toNat_xor, BitVec.toNat_and, BitVec.toNat_and, BitVec.toNat_and]
  rfl

theorem getD_map_toNat (w : List (BitVec 64)) (i : ℕ) : (w.map BitVec.toNat).getD i 0 = (w.getD i 0).toNat :=
  List.getD_map w 0 BitVec.toNat

theorem length_scheduleV : ∀ (n : ℕ) (w : List (BitVec 64)), (Hm.scheduleV n w).length = w.length + n
  | 0, _ => rfl
  | n + 1, w => by
      rw [Hm.scheduleV, length_scheduleV n, Hm.schedStepV, List.length_append, List.length_singleton]
      omega

theorem scheduleV_toNat : ∀ (n : ℕ) (w : List (BitVec 64)),
    (Hm.scheduleV n w).map BitVec.toNat = Pouw.PearlC.Sha512.schedule n (w.map BitVec.toNat)
  | 0, _ => rfl
  | n + 1, w => by
      rw [Hm.scheduleV, scheduleV_toNat n, Pouw.PearlC.Sha512.schedule]
      congr 1
      simp only [Hm.schedStepV, List.map_append, List.map_cons, List.map_nil, List.length_map, getD_map_toNat,
        BitVec.toNat_add, ssig0_toNat, ssig1_toNat, Pouw.PearlC.Sha512.add]
      congr 2
      omega

/-- The spec's registers of a vector state. -/
def regsNat (r : Flock.HmNets.Regs (BitVec 64)) : Pouw.PearlC.Sha512.Regs :=
  ⟨r.a.toNat, r.b.toNat, r.c.toNat, r.d.toNat, r.e.toNat, r.f.toNat, r.g.toNat, r.h.toNat⟩

theorem roundV_toNat (k wt : BitVec 64) (r : Flock.HmNets.Regs (BitVec 64)) :
    regsNat (Hm.roundV k wt r) = Pouw.PearlC.Sha512.round (regsNat r) k.toNat wt.toNat := by
  simp only [Hm.roundV, regsNat, Pouw.PearlC.Sha512.round, BitVec.toNat_add, bsig0_toNat, bsig1_toNat, chV_toNat,
    majV_toNat, Pouw.PearlC.Sha512.add]

theorem getD_toList_u64 (a : Array UInt64) (t : ℕ) : a.toList.getD t 0 = a[t]! := by
  rw [List.getD_eq_getElem?_getD, Array.getElem?_toList, getElem!_def]
  cases a[t]? <;> rfl

theorem k_toNat : Flock.Sha512.k.toList.map UInt64.toNat = Pouw.PearlC.Sha512.K := by decide

theorem kAt_toNat (t : ℕ) : (Flock.HmNets.kAt t).toNat = Pouw.PearlC.Sha512.K.getD t 0 := by
  rw [← k_toNat]
  refine Eq.trans ?_ (List.getD_map Flock.Sha512.k.toList 0 UInt64.toNat).symm
  rw [Flock.HmNets.kAt, UInt64.toNat_toBitVec, getD_toList_u64]

theorem roundsV_toNat (w : List (BitVec 64)) : ∀ (n t : ℕ) (r : Flock.HmNets.Regs (BitVec 64)),
    regsNat (Hm.roundsV w n t r) = (List.range' t n).foldl
      (fun s i => Pouw.PearlC.Sha512.round s (Pouw.PearlC.Sha512.K.getD i 0) (w.getD i 0).toNat) (regsNat r)
  | 0, _, _ => rfl
  | n + 1, t, r => by
      rw [Hm.roundsV, roundsV_toNat w n (t + 1), roundV_toNat, kAt_toNat, List.range'_succ, List.foldl_cons]

theorem zip_K (S : List ℕ) (hS : S.length = 80) :
    Pouw.PearlC.Sha512.K.zip S = (List.range 80).map fun i => (Pouw.PearlC.Sha512.K.getD i 0, S.getD i 0) := by
  have hK : Pouw.PearlC.Sha512.K.length = 80 := rfl
  apply List.ext_getElem
  · simp [hK, hS]
  · intro i h1 h2
    simp only [List.length_map, List.length_range] at h2
    rw [List.getElem_zip, List.getElem_map, List.getElem_range, List.getD_eq_getElem _ _ (by rw [hK]; exact h2),
      List.getD_eq_getElem _ _ (by rw [hS]; exact h2)]

theorem spec_compress8 (x0 x1 x2 x3 x4 x5 x6 x7 : ℕ) (w : List ℕ) (s : Pouw.PearlC.Sha512.Regs)
    (hx : [x0, x1, x2, x3, x4, x5, x6, x7].map (· % 2 ^ 64) = [x0, x1, x2, x3, x4, x5, x6, x7])
    (hs : (Pouw.PearlC.Sha512.K.zip (Pouw.PearlC.Sha512.schedule 64 ((w.map (· % 2 ^ 64)).take 16))).foldl
        (fun s kw => Pouw.PearlC.Sha512.round s kw.1 kw.2) ⟨x0, x1, x2, x3, x4, x5, x6, x7⟩ = s) :
    Pouw.PearlC.Sha512.compress [x0, x1, x2, x3, x4, x5, x6, x7] w =
      [Pouw.PearlC.Sha512.add x0 s.a, Pouw.PearlC.Sha512.add x1 s.b, Pouw.PearlC.Sha512.add x2 s.c,
        Pouw.PearlC.Sha512.add x3 s.d, Pouw.PearlC.Sha512.add x4 s.e, Pouw.PearlC.Sha512.add x5 s.f,
        Pouw.PearlC.Sha512.add x6 s.g, Pouw.PearlC.Sha512.add x7 s.h] := by
  rw [← hs]
  unfold Pouw.PearlC.Sha512.compress
  rw [hx]
  rfl

theorem toNat_mod64 (x : BitVec 64) : x.toNat % 2 ^ 64 = x.toNat := Nat.mod_eq_of_lt x.isLt

theorem ofList8 (c0 c1 c2 c3 c4 c5 c6 c7 : BitVec 64) :
    Flock.HmNets.Regs.ofList [c0, c1, c2, c3, c4, c5, c6, c7] = ⟨c0, c1, c2, c3, c4, c5, c6, c7⟩ := rfl

theorem feed_toNat (c0 c1 c2 c3 c4 c5 c6 c7 : BitVec 64) (R : Flock.HmNets.Regs (BitVec 64)) :
    (List.zipWith (· + ·) [c0, c1, c2, c3, c4, c5, c6, c7] R.toList).map BitVec.toNat =
      [Pouw.PearlC.Sha512.add c0.toNat (regsNat R).a, Pouw.PearlC.Sha512.add c1.toNat (regsNat R).b,
        Pouw.PearlC.Sha512.add c2.toNat (regsNat R).c, Pouw.PearlC.Sha512.add c3.toNat (regsNat R).d,
        Pouw.PearlC.Sha512.add c4.toNat (regsNat R).e, Pouw.PearlC.Sha512.add c5.toNat (regsNat R).f,
        Pouw.PearlC.Sha512.add c6.toNat (regsNat R).g, Pouw.PearlC.Sha512.add c7.toNat (regsNat R).h] := by
  simp only [Flock.HmNets.Regs.toList, List.zipWith_cons_cons, List.zipWith_nil_left, List.map_cons, List.map_nil,
    regsNat, BitVec.toNat_add, Pouw.PearlC.Sha512.add]

theorem rounds_spec (c0 c1 c2 c3 c4 c5 c6 c7 : BitVec 64) (msg : List (BitVec 64)) (hm : msg.length = 16) :
    (Pouw.PearlC.Sha512.K.zip (Pouw.PearlC.Sha512.schedule 64 (msg.map BitVec.toNat))).foldl
        (fun s kw => Pouw.PearlC.Sha512.round s kw.1 kw.2)
        ⟨c0.toNat, c1.toNat, c2.toNat, c3.toNat, c4.toNat, c5.toNat, c6.toNat, c7.toNat⟩ =
      regsNat (Hm.roundsV (Hm.scheduleV 64 msg) 80 0 ⟨c0, c1, c2, c3, c4, c5, c6, c7⟩) := by
  rw [← scheduleV_toNat, zip_K _ (by simp [length_scheduleV, hm]), List.foldl_map, roundsV_toNat,
    List.range_eq_range']
  simp only [getD_map_toNat]
  rfl

theorem compressV_toNat (c0 c1 c2 c3 c4 c5 c6 c7 : BitVec 64) (msg : List (BitVec 64)) (hm : msg.length = 16) :
    (Hm.compressV [c0, c1, c2, c3, c4, c5, c6, c7] msg).map BitVec.toNat =
      Pouw.PearlC.Sha512.compress [c0.toNat, c1.toNat, c2.toNat, c3.toNat, c4.toNat, c5.toNat, c6.toNat, c7.toNat]
        (msg.map BitVec.toNat) := by
  have hw : ((msg.map BitVec.toNat).map (· % 2 ^ 64)).take 16 = msg.map BitVec.toNat := by
    rw [List.take_of_length_le (by simp [hm])]
    rw [List.map_map]
    exact List.map_congr_left fun x _ => toNat_mod64 x
  have hh : [c0.toNat, c1.toNat, c2.toNat, c3.toNat, c4.toNat, c5.toNat, c6.toNat, c7.toNat].map (· % 2 ^ 64) =
      [c0.toNat, c1.toNat, c2.toNat, c3.toNat, c4.toNat, c5.toNat, c6.toNat, c7.toNat] := by
    simp only [List.map_cons, List.map_nil, toNat_mod64]
  rw [spec_compress8 _ _ _ _ _ _ _ _ _ (regsNat (Hm.roundsV (Hm.scheduleV 64 msg) 80 0 ⟨c0, c1, c2, c3, c4, c5, c6, c7⟩))
    hh (by rw [hw]; exact rounds_spec c0 c1 c2 c3 c4 c5 c6 c7 msg hm), Hm.compressV, ofList8, feed_toNat]

/-! ## The digest's bytes -/

theorem flatMap_range_map {α : Type} (k : ℕ) (hk : 0 < k) (F : ℕ → ℕ → α) :
    ∀ n, ((List.range n).flatMap fun a => (List.range k).map (F a)) =
      (List.range (n * k)).map fun i => F (i / k) (i % k)
  | 0 => by simp
  | n + 1 => by
      rw [List.range_succ, List.flatMap_append, List.flatMap_singleton, flatMap_range_map k hk F n,
        Nat.succ_mul, List.range_add, List.map_append, List.map_map]
      congr 1
      refine List.map_congr_left fun j hj => ?_
      rw [List.mem_range] at hj
      simp only [Function.comp]
      rw [Nat.add_comm, Nat.add_mul_div_right _ _ hk, Nat.div_eq_of_lt hj, Nat.zero_add, Nat.add_mul_mod_self_right,
        Nat.mod_eq_of_lt hj]

/-- The spec's words of a chaining value. -/
def phi (h : Array UInt64) : List ℕ := (List.range 8).map fun i => h[i]!.toNat

theorem bytesNat_digestOf (h : Array UInt64) :
    bytesNat (Hm.digestOf h) = (phi h).flatMap (Pouw.PearlC.Sha512.beBytes 8) := by
  rw [phi, List.flatMap_map]
  simp only [Pouw.PearlC.Sha512.beBytes]
  rw [flatMap_range_map 8 (by norm_num)]
  apply List.ext_getElem
  · simp [bytesNat, Hm.digestOf]
  · intro i h1 h2
    simp only [List.length_map, List.length_range] at h2
    simp only [bytesNat, Hm.digestOf, List.getElem_map, List.getElem_range, Array.toList_ofFn, List.getElem_ofFn]
    have e3 : (8 * (7 - i % 8)).toUInt64.toNat % 64 = 8 * (7 - i % 8) := by
      rw [Nat.toUInt64_eq, UInt64.toNat_ofNat', Nat.mod_eq_of_lt (by omega), Nat.mod_eq_of_lt (by omega)]
    rw [UInt64.toNat_toUInt8, UInt64.toNat_shiftRight, e3, Nat.shiftRight_eq_div_pow, pow_mul,
      show 8 - 1 - i % 8 = 7 - i % 8 by omega]
    norm_num

/-! ## The padded message and its blocks -/

theorem lenBytes_toNat (len : ℕ) : (Hm.lenBytes len).map UInt8.toNat = Pouw.PearlC.Sha512.beBytes 16 (8 * len) := by
  rw [Hm.lenBytes, Pouw.PearlC.Sha512.beBytes, List.map_map, List.range_eq_range']
  refine List.map_congr_left fun a ha => ?_
  simp only [List.mem_range'_1] at ha
  simp only [Function.comp, Nat.toUInt8_eq, UInt8.toNat_ofNat', Nat.shiftRight_eq_div_pow, pow_mul,
    show 16 - 1 - a = 15 - a by omega, Nat.mul_comm len 8]
  norm_num

theorem zeroPad_eq (len : ℕ) : Hm.zeroPad len = (128 - (len + 17) % 128) % 128 := by
  unfold Hm.zeroPad
  omega

theorem take_drop_eq (L : List ℕ) (o k : ℕ) (h : o + k ≤ L.length) :
    (L.drop o).take k = (List.range k).map fun j => L.getD (o + j) 0 := by
  apply List.ext_getElem
  · simp
    omega
  · intro j h1 h2
    simp only [List.length_map, List.length_range] at h2
    rw [List.getElem_take, List.getElem_drop, List.getElem_map, List.getElem_range,
      List.getD_eq_getElem _ _ (by omega)]

theorem chunks_drop (L : List ℕ) : ∀ (n a : ℕ), 128 * (a + n) ≤ L.length →
    Pouw.PearlC.Sha512.chunks n (L.drop (128 * a)) =
      (List.range' a n).map fun b => (List.range 128).map fun j => L.getD (128 * b + j) 0
  | 0, _, _ => rfl
  | n + 1, a, h => by
      rw [Pouw.PearlC.Sha512.chunks, List.drop_drop, show 128 * a + 128 = 128 * (a + 1) by ring,
        chunks_drop L n (a + 1) (by omega), List.range'_succ, List.map_cons, take_drop_eq L _ _ (by omega)]

theorem blockWords_eq (L : List ℕ) (b : ℕ) :
    Pouw.PearlC.Sha512.blockWords ((List.range 128).map fun j => L.getD (128 * b + j) 0) =
      (List.range 16).map fun i =>
        Pouw.PearlC.Sha512.beNat ((List.range 8).map fun j => L.getD (128 * b + 8 * i + j) 0) := by
  rw [Pouw.PearlC.Sha512.blockWords]
  refine List.map_congr_left fun i hi => ?_
  rw [List.mem_range] at hi
  rw [take_drop_eq _ _ _ (by simp; omega)]
  congr 1
  refine List.map_congr_left fun j hj => ?_
  rw [List.mem_range] at hj
  rw [List.getD_eq_getElem _ _ (by simp; omega), List.getElem_map, List.getElem_range, Nat.add_assoc]

/-! ## The compiled compression, chain and digest -/

theorem phi_compress (h : Array UInt64) (data : ByteArray) (off : ℕ) :
    phi (Flock.Sha512.compress h data off) =
      Pouw.PearlC.Sha512.compress (phi h) ((List.range 16).map fun i => (Hm.wordBE data off i).toNat) := by
  have e1 : phi (Flock.Sha512.compress h data off) = (Hm.toBV (Flock.Sha512.compress h data off)).map BitVec.toNat := by
    rw [← Hm.map_range_eq_toBV _ (Hm.size_compress h data off), List.map_map]
    rfl
  have e2 : (List.range 8).map (fun i => h[i]!.toBitVec) = [h[0]!.toBitVec, h[1]!.toBitVec, h[2]!.toBitVec,
      h[3]!.toBitVec, h[4]!.toBitVec, h[5]!.toBitVec, h[6]!.toBitVec, h[7]!.toBitVec] := rfl
  have e3 : [h[0]!.toBitVec.toNat, h[1]!.toBitVec.toNat, h[2]!.toBitVec.toNat, h[3]!.toBitVec.toNat,
      h[4]!.toBitVec.toNat, h[5]!.toBitVec.toNat, h[6]!.toBitVec.toNat, h[7]!.toBitVec.toNat] = phi h := rfl
  have e4 : ((List.range 16).map fun i => (Hm.wordBE data off i).toBitVec).map BitVec.toNat =
      (List.range 16).map fun i => (Hm.wordBE data off i).toNat := by
    rw [List.map_map]
    rfl
  rw [e1, Hm.compress_eq, e2, compressV_toNat _ _ _ _ _ _ _ _ _ (by simp), e3, e4]

theorem foldBE_toNat (g : ℕ → UInt8) : ∀ k ≤ 8,
    ((List.range' 0 k).foldl (fun (x : UInt64) j => x <<< 8 ||| (g j).toUInt64) 0).toNat =
      ((List.range' 0 k).map fun j => (g j).toNat).foldl (fun w b => w * 256 + b % 256) 0 ∧
    ((List.range' 0 k).map fun j => (g j).toNat).foldl (fun w b => w * 256 + b % 256) 0 < 2 ^ (8 * k)
  | 0, _ => by simp
  | k + 1, hk => by
      obtain ⟨ih, hb⟩ := foldBE_toNat g k (by omega)
      rw [List.range'_concat, List.foldl_append, List.foldl_cons, List.foldl_nil, List.map_append, List.foldl_append,
        List.map_cons, List.map_nil, List.foldl_cons, List.foldl_nil, Nat.zero_add, Nat.one_mul]
      generalize ((List.range' 0 k).map fun j => (g j).toNat).foldl (fun w b => w * 256 + b % 256) 0 = B at ih hb
      generalize (List.range' 0 k).foldl (fun (x : UInt64) j => x <<< 8 ||| (g j).toUInt64) 0 = X at ih
      have hg := UInt8.toNat_lt (g k)
      have hg8 : (g k).toNat < 256 := by simpa using hg
      have hB : B * 256 < 2 ^ 64 := by
        have : 2 ^ (8 * k) * 256 ≤ 2 ^ 64 := by
          rw [show (256 : ℕ) = 2 ^ 8 from rfl, ← pow_add]
          exact Nat.pow_le_pow_right (by norm_num) (by omega)
        have := Nat.mul_lt_mul_of_pos_right hb (show 0 < 256 by norm_num)
        omega
      refine ⟨?_, ?_⟩
      · rw [UInt64.toNat_or, UInt64.toNat_shiftLeft, UInt8.toNat_toUInt64, ih, show (8 : UInt64).toNat % 64 = 8 from rfl,
          Nat.mod_eq_of_lt (show B <<< 8 < 2 ^ 64 by rw [Nat.shiftLeft_eq]; exact hB),
          ← Nat.shiftLeft_add_eq_or_of_lt (i := 8) hg, Nat.shiftLeft_eq, Nat.mod_eq_of_lt hg8]
      · rw [Nat.mod_eq_of_lt hg8, show 8 * (k + 1) = 8 * k + 8 by ring, pow_add, show (2 : ℕ) ^ 8 = 256 by norm_num]
        generalize 2 ^ (8 * k) = P at hb ⊢
        clear hB ih hg
        omega

theorem getD_bytesNat (b : ByteArray) (k : ℕ) : (bytesNat b).getD k 0 = (b.get! k).toNat := by
  rw [Hm.get!_eq_getD]
  exact List.getD_map b.data.toList 0 UInt8.toNat

theorem wordBE_toNat (data : ByteArray) (off i : ℕ) :
    (Hm.wordBE data off i).toNat =
      Pouw.PearlC.Sha512.beNat ((List.range 8).map fun j => (bytesNat data).getD (off + 8 * i + j) 0) := by
  rw [Hm.wordBE, (foldBE_toNat _ 8 le_rfl).1, Pouw.PearlC.Sha512.beNat, List.range_eq_range']
  simp only [getD_bytesNat]

theorem phi_iv : phi Flock.Sha512.iv = Pouw.PearlC.Sha512.IV := by decide

theorem phi_chain (data : ByteArray) (n : ℕ) :
    phi (Hm.chain data n) = (List.range' 0 n).foldl (fun s b => Pouw.PearlC.Sha512.compress s
      ((List.range 16).map fun i =>
        Pouw.PearlC.Sha512.beNat ((List.range 8).map fun j => (bytesNat data).getD (128 * b + 8 * i + j) 0)))
      Pouw.PearlC.Sha512.IV := by
  rw [Hm.chain, ← phi_iv]
  refine (List.foldl_hom phi fun x y => ?_).symm
  rw [phi_compress]
  simp only [wordBE_toNat]

theorem length_bytesNat (b : ByteArray) : (bytesNat b).length = b.size := by
  simp [bytesNat]

theorem bytesNat_mod (b : ByteArray) : (bytesNat b).map (· % 256) = bytesNat b := by
  rw [bytesNat, List.map_map]
  exact List.map_congr_left fun x _ => Nat.mod_eq_of_lt x.toNat_lt

theorem bytesNat_padData (m : ByteArray) : bytesNat (Hm.padData m) = Pouw.PearlC.Sha512.pad (bytesNat m) := by
  rw [Pouw.PearlC.Sha512.pad, bytesNat_mod, length_bytesNat, ← zeroPad_eq, ← lenBytes_toNat,
    show bytesNat (Hm.padData m) = (Hm.padData m).data.toList.map UInt8.toNat from rfl, Hm.padData_data]
  simp only [List.map_append, List.map_cons, List.map_nil, List.map_replicate]
  rfl

theorem blocks_eq (m : ByteArray) : Pouw.PearlC.Sha512.blocks m.size = (Hm.padData m).size / 128 := by
  rw [Hm.size_padData, Pouw.PearlC.Sha512.blocks, Hm.zeroPad]
  omega

end NoiseProofs

open NoiseProofs FlockSoundness.Discharge in
theorem hashIsTheSpec : HashIsTheSpec := by
  intro m
  have hP := Hm.size_padData_mod m
  have hL := length_bytesNat (Hm.padData m)
  have hc := chunks_drop (bytesNat (Hm.padData m)) ((Hm.padData m).size / 128) 0 (by omega)
  rw [Nat.mul_zero, List.drop_zero] at hc
  rw [Hm.hash_eq, bytesNat_digestOf, phi_chain, Pouw.PearlC.Sha512.digest, ← bytesNat_padData, length_bytesNat,
    blocks_eq, hc, List.foldl_map]
  simp only [blockWords_eq]

namespace NoiseProofs

/-! ## Byte strings -/

theorem bytesNat_append (a b : ByteArray) : bytesNat (a ++ b) = bytesNat a ++ bytesNat b := by
  rw [bytesNat, bytesNat, bytesNat, ByteArray.toList_data_append, List.map_append]

theorem bytesNat_extract (b : ByteArray) (i e : ℕ) :
    bytesNat (b.extract i e) = ((bytesNat b).drop i).take (e - i) := by
  rw [bytesNat, bytesNat, ByteArray.data_extract, Array.toList_extract, List.extract_eq_take_drop, List.map_take,
    List.map_drop]

theorem bytesNat_zeros (k : ℕ) : bytesNat ⟨Array.replicate k 0⟩ = List.replicate k 0 := by
  rw [bytesNat, Array.toList_replicate, List.map_replicate]
  rfl

theorem bytesNat_pad (b : ByteArray) (n : ℕ) :
    bytesNat (Flock.Hm96.pad b n) = Pouw.PearlC.Sha512.padTo n (bytesNat b) := by
  rw [Flock.Hm96.pad, bytesNat_append, Pouw.PearlC.Sha512.padTo, length_bytesNat, show n - min n b.size = n - b.size by omega,
    bytesNat_zeros]

theorem bytesNat_be64 (x : ℕ) : bytesNat (Flock.HmRow.be64 x) = Pouw.PearlC.Sha512.beBytes 8 x := by
  simp only [bytesNat, Flock.HmRow.be64, Array.toList_map, Array.toList_range, List.map_map, Pouw.PearlC.Sha512.beBytes]
  refine List.map_congr_left fun i hi => ?_
  rw [List.mem_range] at hi
  simp only [Function.comp, Nat.toUInt8_eq, UInt8.toNat_ofNat', show 8 - 1 - i = 7 - i by omega]
  norm_num

theorem bytesNat_le (w x : ℕ) : bytesNat (Words.le w x) = Pouw.PearlC.Sha512.leBytes w x := by
  simp only [bytesNat, Words.le, Array.toList_map, Array.toList_range, List.map_map, Pouw.PearlC.Sha512.leBytes]
  refine List.map_congr_left fun i _ => ?_
  simp only [Function.comp, Nat.toUInt8_eq, UInt8.toNat_ofNat']
  norm_num

theorem bytesNat_be32_1024 : bytesNat (Flock.be32 1024) = Pouw.PearlC.Sha512.beBytes 4 1024 := by
  simp only [Flock.be32, Id.run, Std.Legacy.Range.forIn_eq_forIn_range', List.forIn_pure_yield_eq_foldl,
    Std.Legacy.Range.size, Nat.reduceSub, Nat.reduceAdd, Nat.sub_zero, Nat.div_one]
  decide

theorem bytesNat_rowLabel :
    bytesNat ("verity/sha512-row-seg/v1".toUTF8.push 0) = Pouw.PearlC.Sha512.ascii "verity/sha512-row-seg/v1" ++ [0] := by
  decide

theorem bytesNat_segmentLabel : bytesNat ("verity/sha512-row-seg/v1/segment".toUTF8.push 0) =
    Pouw.PearlC.Sha512.ascii "verity/sha512-row-seg/v1/segment" ++ [0] := by
  decide

theorem bytesNat_foldl_append (H : ℕ → ByteArray) : ∀ (L : List ℕ) (init : ByteArray),
    bytesNat (L.foldl (fun acc j => acc ++ H j) init) = bytesNat init ++ L.flatMap fun j => bytesNat (H j)
  | [], init => by simp
  | j :: L, init => by
      rw [List.foldl_cons, bytesNat_foldl_append H L, bytesNat_append, List.flatMap_cons, List.append_assoc]

theorem flatMap_congr {α β : Type} {l : List α} {f g : α → List β} (h : ∀ a ∈ l, f a = g a) :
    l.flatMap f = l.flatMap g := by
  rw [List.flatMap_def, List.flatMap_def, List.map_congr_left h]

/-! ## The row's segments -/

theorem rowSegs_length (n : ℕ) : (Pouw.PearlC.NoiseSha512.rowSegs n).length = (8 * n + 8192 - 1) / 8192 := by
  rw [Pouw.PearlC.NoiseSha512.rowSegs, List.length_append, List.length_replicate]
  by_cases h : n % 1024 = 0
  · rw [ite_eq_left h, List.length_nil]
    omega
  · rw [ite_eq_right h, List.length_singleton]
    omega

theorem rowSegs_eq (n : ℕ) : Pouw.PearlC.NoiseSha512.rowSegs n =
    (List.range ((8 * n + 8192 - 1) / 8192)).map fun j => min (n - 1024 * j) 1024 := by
  have hq : (List.range (n / 1024)).map (fun j => min (n - 1024 * j) 1024) = List.replicate (n / 1024) 1024 := by
    rw [List.map_congr_left (g := fun _ => 1024) fun j hj => by rw [List.mem_range] at hj; omega, List.map_const',
      List.length_range]
  rw [Pouw.PearlC.NoiseSha512.rowSegs]
  by_cases h : n % 1024 = 0
  · rw [ite_eq_left h, List.append_nil, show (8 * n + 8192 - 1) / 8192 = n / 1024 by omega, hq]
  · rw [ite_eq_right h, show (8 * n + 8192 - 1) / 8192 = n / 1024 + 1 by omega, List.range_succ, List.map_append, hq,
      List.map_singleton, show min (n - 1024 * (n / 1024)) 1024 = n % 1024 by omega]

theorem rowSegs_getD (n j : ℕ) (hj : j < (Pouw.PearlC.NoiseSha512.rowSegs n).length) :
    (Pouw.PearlC.NoiseSha512.rowSegs n).getD j 0 = min (n - 1024 * j) 1024 := by
  rw [rowSegs_length] at hj
  rw [rowSegs_eq, List.getD_eq_getElem _ _ (by simpa using hj), List.getElem_map, List.getElem_range]

theorem segShape_size (l : ℕ) : (Flock.HmRow.segShape l).1.size = (l + 8192 - 1) / 8192 := by
  simp [Flock.HmRow.segShape, Flock.HmRow.ROW_SEG_BITS]

theorem segShape_get (l j : ℕ) (hj : j < (l + 8192 - 1) / 8192) :
    (Flock.HmRow.segShape l).1[j]!.1 = min (l - 8192 * j) 8192 := by
  rw [getElem!_pos (Flock.HmRow.segShape l).1 j (by rw [segShape_size]; exact hj)]
  simp [Flock.HmRow.segShape, Flock.HmRow.ROW_SEG_BITS]

theorem bytesNat_rowPrefixSeg (n : ℕ) :
    bytesNat (Flock.HmRow.rowPrefixSeg (8 * n)) = Pouw.PearlC.Sha512.padTo 128
      (Pouw.PearlC.Sha512.ascii "verity/sha512-row-seg/v1" ++ [0] ++ Pouw.PearlC.Sha512.beBytes 8 (8 * n) ++
        Pouw.PearlC.Sha512.beBytes 8 (Pouw.PearlC.NoiseSha512.rowSegs n).length ++
        Pouw.PearlC.Sha512.beBytes 4 1024) := by
  rw [Flock.HmRow.rowPrefixSeg, bytesNat_pad, bytesNat_append, bytesNat_append, bytesNat_append, bytesNat_rowLabel,
    bytesNat_be64, bytesNat_be64, bytesNat_be32_1024, rowSegs_length, Flock.HmRow.ROW_SEG_BITS]

end NoiseProofs

open NoiseProofs in
theorem seg_eq (r : ByteArray) (j : ℕ) (hj : j ∈ List.range (Pouw.PearlC.NoiseSha512.rowSegs r.size).length) :
    bytesNat (Flock.Sha512.hash (Flock.HmRow.segmentPrefix j (Pouw.PearlC.NoiseSha512.rowSegs r.size).length
      (Flock.HmRow.segShape (8 * r.size)).1[j]!.1 ++
        r.extract (1024 * j) (1024 * j + ((Flock.HmRow.segShape (8 * r.size)).1[j]!.1 + 7) / 8))) =
    Pouw.PearlC.Sha512.digest (Pouw.PearlC.Sha512.padTo 128
      (Pouw.PearlC.Sha512.ascii "verity/sha512-row-seg/v1/segment" ++ [0] ++ Pouw.PearlC.Sha512.beBytes 8 j ++
        Pouw.PearlC.Sha512.beBytes 8 (Pouw.PearlC.NoiseSha512.rowSegs r.size).length ++
        Pouw.PearlC.Sha512.beBytes 8 (8 * (Pouw.PearlC.NoiseSha512.rowSegs r.size).getD j 0)) ++
      ((bytesNat r).drop (1024 * j)).take ((Pouw.PearlC.NoiseSha512.rowSegs r.size).getD j 0)) := by
  rw [List.mem_range] at hj
  have hm : (Flock.HmRow.segShape (8 * r.size)).1[j]!.1 = 8 * (Pouw.PearlC.NoiseSha512.rowSegs r.size).getD j 0 := by
    rw [segShape_get _ _ (by rw [← rowSegs_length]; exact hj), rowSegs_getD _ _ hj]
    omega
  rw [hashIsTheSpec, bytesNat_append, hm, Flock.HmRow.segmentPrefix, bytesNat_pad, bytesNat_append, bytesNat_append,
    bytesNat_append, bytesNat_segmentLabel, bytesNat_be64, bytesNat_be64, bytesNat_be64, bytesNat_extract,
    Nat.add_sub_cancel_left, show (8 * (Pouw.PearlC.NoiseSha512.rowSegs r.size).getD j 0 + 7) / 8 =
      (Pouw.PearlC.NoiseSha512.rowSegs r.size).getD j 0 by omega]

open NoiseProofs in
theorem segDigestIsRowDigest : SegDigestIsRowDigest := by
  intro r
  have hs := segShape_size (8 * r.size)
  simp only [Flock.HmRow.segDigest, Pouw.PearlC.NoiseSha512.rowDigest]
  rw [hashIsTheSpec, bytesNat_foldl_append, bytesNat_rowPrefixSeg, hs, length_bytesNat, ← rowSegs_length,
    flatMap_congr (seg_eq r)]

open NoiseProofs in
theorem s512_eq (role : String) (m : ByteArray)
    (hl : bytesNat (("pearl-c/h3s/" ++ role).toUTF8.push 0 ++
      ⟨Array.replicate (128 - (("pearl-c/h3s/" ++ role).toUTF8.push 0).size) 0⟩) = Pouw.PearlC.NoiseSha512.label role) :
    bytesNat (Words.s512 role m) = Pouw.PearlC.NoiseSha512.s512 role (bytesNat m) := by
  simp only [Words.s512]
  rw [bytesNat_extract, Nat.sub_zero, List.drop_zero, hashIsTheSpec, bytesNat_append, hl,
    Pouw.PearlC.NoiseSha512.s512]

open NoiseProofs in
theorem seedsAreTheSpec : SeedsAreTheSpec := by
  have hs : ∀ role ∈ roles, ∀ m : ByteArray,
      bytesNat (Words.s512 role m) = Pouw.PearlC.NoiseSha512.s512 role (bytesNat m) := by
    intro role h m
    simp only [roles, List.mem_cons, List.mem_nil_iff, or_false] at h
    rcases h with rfl | rfl | rfl | rfl <;> exact s512_eq _ m (by decide)
  refine ⟨hs, fun salt rootB => ?_, fun salt rootB row u i => ?_⟩
  · rw [Words.seedB, hs "seed-B" (by simp [roles]), bytesNat_append, Pouw.PearlC.NoiseSha512.seedB]
  · rw [Words.rowSeed, hs "seed-A" (by simp [roles]), bytesNat_append, bytesNat_append, bytesNat_append,
      segDigestIsRowDigest, bytesNat_le, Pouw.PearlC.NoiseSha512.rowSeed]

namespace NoiseProofs

open Pouw.Fp8Atom Pouw.PearlC Pouw.PearlC.PeelExact Pouw.PearlC.PeelExactProofs

/-! ## The check's FP32 and BF16 words on values -/

theorem floor_div_65536 (n : ℕ) : ⌊(n : ℚ) / 65536⌋ = ((n / 65536 : ℕ) : ℤ) := by
  rw [Int.floor_eq_iff]
  constructor
  · rw [le_div_iff₀ (by norm_num)]
    exact_mod_cast Nat.div_mul_le_self n 65536
  · rw [div_lt_iff₀ (by norm_num)]
    exact_mod_cast (show n < (n / 65536 + 1) * 65536 by omega)

/-- BF16's nearest-even rounding of `n` units of `2^-16`, as the check's carry `(n + 0x7FFF + lsb) >>> 16`. -/
theorem roundEven_div_65536 (n : ℕ) :
    roundEven ((n : ℚ) / 65536) = (((n + 32767 + n / 65536 % 2) / 65536 : ℕ) : ℤ) := by
  have hfrac : (n : ℚ) / 65536 - ((n / 65536 : ℕ) : ℤ) = ((n % 65536 : ℕ) : ℚ) / 65536 := by
    have : (n : ℚ) = ((n / 65536 : ℕ) : ℚ) * 65536 + ((n % 65536 : ℕ) : ℚ) := by
      exact_mod_cast (Nat.div_add_mod' n 65536).symm
    rw [this, Int.cast_natCast]; ring
  have h1 : ((n % 65536 : ℕ) : ℚ) / 65536 < 1 / 2 ↔ n % 65536 < 32768 := by
    rw [div_lt_iff₀ (by norm_num)]; norm_num
  have h2 : 1 / 2 < ((n % 65536 : ℕ) : ℚ) / 65536 ↔ 32768 < n % 65536 := by
    rw [lt_div_iff₀ (by norm_num)]; norm_num
  unfold roundEven
  simp only [floor_div_65536, hfrac, h1, h2]
  split_ifs <;> omega

theorem and_ff (u : ℕ) : u &&& 0xFF = u % 256 := Nat.and_two_pow_sub_one_eq_mod u 8
theorem and_ffff (u : ℕ) : u &&& 0xFFFF = u % 65536 := Nat.and_two_pow_sub_one_eq_mod u 16
theorem and_7fff (u : ℕ) : u &&& 0x7FFF = u % 32768 := Nat.and_two_pow_sub_one_eq_mod u 15
theorem and_one (u : ℕ) : u &&& 1 = u % 2 := Nat.and_two_pow_sub_one_eq_mod u 1

theorem finite_ok {w : ℕ} (hf : (w >>> 23) % 256 ≠ 255) : Words.finite w = .ok w := by
  simp [Words.finite, and_ff, hf]; rfl

theorem sgn_abs (b : Prop) [Decidable b] (y : ℚ) : |(if b then (-1 : ℚ) else 1) * y| = |y| := by
  split <;> simp

theorem sgn_neg (b : Prop) [Decidable b] {y : ℚ} (hy : 0 < y) : (if b then (-1 : ℚ) else 1) * y < 0 ↔ b := by
  split <;> simp_all [hy.le]

/-- A finite FP32 word's value, `±M·2^(e − 150)` with `M = 2^23 + mantissa`, is `bf16Rn`'s `±roundEven(M / 2^16)` at
`2^(e − 134)`. -/
theorem bf16Rn_fields (s : Prop) [Decidable s] (M e : ℕ) (h1 : 2 ^ 23 ≤ M) (h2 : M < 2 ^ 24) :
    bf16Rn ((if s then (-1 : ℚ) else 1) * ((M : ℚ) * 2 ^ ((e : ℤ) - 150))) =
      (if s then (-1 : ℚ) else 1) * (roundEven ((M : ℚ) / 65536) : ℚ) * 2 ^ ((e : ℤ) - 134) := by
  have hU : (2 : ℚ) ^ ((e : ℤ) - 150) = 2 ^ ((e : ℤ) - 134) / 65536 := by
    rw [show (e : ℤ) - 150 = ((e : ℤ) - 134) + (-16) by ring, zpow_add₀ two_ne_zero]; norm_num; ring
  have hA : 0 < (M : ℚ) * 2 ^ ((e : ℤ) - 150) := by
    have : (0 : ℚ) < M := by exact_mod_cast (show 0 < M by omega)
    exact mul_pos this (zpow_pos (by norm_num) _)
  have hb : binade |(if s then (-1 : ℚ) else 1) * ((M : ℚ) * 2 ^ ((e : ℤ) - 150))| = (e : ℤ) - 127 := by
    rw [sgn_abs, abs_of_pos hA]
    have hM1 : (2 : ℚ) ^ 23 ≤ M := by exact_mod_cast h1
    have hM2 : (M : ℚ) < 2 ^ 24 := by exact_mod_cast h2
    apply Proofs.binade_unique hA
    · rw [show (e : ℤ) - 127 = 23 + ((e : ℤ) - 150) by ring, zpow_add₀ two_ne_zero]
      exact mul_le_mul_of_nonneg_right (by exact_mod_cast hM1) (zpow_pos (by norm_num) _).le
    · rw [show (e : ℤ) - 127 + 1 = 24 + ((e : ℤ) - 150) by ring, zpow_add₀ two_ne_zero]
      exact mul_lt_mul_of_pos_right (by exact_mod_cast hM2) (zpow_pos (by norm_num) _)
  have hlt : ((if s then (-1 : ℚ) else 1) * ((M : ℚ) * 2 ^ ((e : ℤ) - 150)) < 0) ↔ s := sgn_neg s hA
  unfold bf16Rn ulpBf16
  rw [hb, sgn_abs, abs_of_pos hA, show (e : ℤ) - 127 - 7 = (e : ℤ) - 134 by ring]
  have hq : (M : ℚ) * 2 ^ ((e : ℤ) - 150) / 2 ^ ((e : ℤ) - 134) = (M : ℚ) / 65536 := by
    rw [hU]; field_simp
  rw [hq]
  by_cases hs : s
  · rw [ite_eq_left (hlt.2 hs), ite_eq_left hs]
  · rw [ite_eq_right (fun h => hs (hlt.1 h)), ite_eq_right hs]

theorem carry_split (W sb e q r : ℕ) (hsb : sb < 2) (he : e < 255) (hq : q < 128) (hr : r < 65536) :
    (65536 * (W * 65536 + sb * 32768 + e * 128 + q) + r + 32767 +
      (65536 * (W * 65536 + sb * 32768 + e * 128 + q) + r) / 65536 % 2) / 65536 % 65536 =
      sb * 32768 + e * 128 + (q + (r + 32767 + q % 2) / 65536) := by
  have h1 : (65536 * (W * 65536 + sb * 32768 + e * 128 + q) + r) / 65536 = W * 65536 + sb * 32768 + e * 128 + q := by
    rw [Nat.mul_add_div (by norm_num), Nat.div_eq_of_lt hr, Nat.add_zero]
  rw [h1, show (W * 65536 + sb * 32768 + e * 128 + q) % 2 = q % 2 by omega,
    show 65536 * (W * 65536 + sb * 32768 + e * 128 + q) + r + 32767 + q % 2 =
      65536 * (W * 65536 + sb * 32768 + e * 128 + q) + (r + 32767 + q % 2) by ring,
    Nat.mul_add_div (by norm_num)]
  have hc : (r + 32767 + q % 2) / 65536 ≤ 1 := by omega
  generalize (r + 32767 + q % 2) / 65536 = c at hc ⊢
  omega

theorem carry_word (w : ℕ) (he : w / 8388608 % 256 ≠ 255) :
    (w + 32767 + w / 65536 % 2) / 65536 % 65536 = w / 2147483648 % 2 * 32768 + w / 8388608 % 256 * 128 +
      (w % 8388608 / 65536 + (w % 8388608 % 65536 + 32767 + w % 8388608 / 65536 % 2) / 65536) := by
  have hw : w = 65536 * (w / 4294967296 * 65536 + w / 2147483648 % 2 * 32768 + w / 8388608 % 256 * 128 +
      w % 8388608 / 65536) + w % 8388608 % 65536 := by omega
  have := carry_split (w / 4294967296) (w / 2147483648 % 2) (w / 8388608 % 256) (w % 8388608 / 65536)
    (w % 8388608 % 65536) (by omega) (by omega) (by omega) (by omega)
  rw [← hw] at this
  exact this

theorem bf16_fields {sb e k : ℕ} (hsb : sb < 2) (he : e < 255) (hk : k ≤ 128)
    (hov : ¬ 32640 ≤ (sb * 32768 + e * 128 + k) % 32768) :
    (sb * 32768 + e * 128 + k) / 128 % 256 = e + k / 128 ∧ (sb * 32768 + e * 128 + k) % 128 = k % 128 ∧
      (sb * 32768 + e * 128 + k) / 32768 % 2 = sb ∧ e + k / 128 ≠ 255 := by
  have h : e * 128 + k < 32640 := by omega
  refine ⟨by omega, by omega, by omega, by omega⟩

/-- **The check's BF16 rounding is `bf16Rn`** on a finite word of magnitude at least `2^−126`: its carry
`(w + 0x7FFF + lsb) >>> 16` is `roundEven` of the low 16 bits, and a carry out of the mantissa is the next binade. -/
theorem toBf16_spec {w h : ℕ} {x : ℚ} (hx : wordQ w = some x) (he0 : (w >>> 23) % 256 ≠ 0)
    (hh : Words.toBf16 w = .ok h) : h < 65536 ∧ bf16Exp h ≠ 255 ∧ bf16Value h = bf16Rn x := by
  have hf : (w >>> 23) % 256 ≠ 255 := by
    intro hf
    simp [Words.toBf16, Words.finite, and_ff, hf] at hh
    cases hh
  rw [Words.toBf16, finite_ok hf] at hh
  dsimp only [bind, Except.bind] at hh
  split at hh
  · cases hh
  cases hh
  rename_i hov
  rw [wordQ_fields hf, Option.some.injEq] at hx
  subst hx
  rw [ite_eq_right he0, show max ((w >>> 23) % 256) 1 = (w >>> 23) % 256 by omega,
    bf16Rn_fields _ _ _ (by omega) (by omega), roundEven_div_65536]
  unfold bf16Value bf16Sig bf16ExpOr1 bf16Exp bf16Man bf16Sign
  simp only [Nat.shiftRight_eq_div_pow, Nat.reducePow, and_ffff, and_7fff, and_one] at hov hf he0 ⊢
  rw [carry_word w (by omega)] at hov ⊢
  obtain ⟨e, h_e⟩ : ∃ e, w / 8388608 % 256 = e := ⟨_, rfl⟩
  obtain ⟨m, h_m⟩ : ∃ m, w % 8388608 = m := ⟨_, rfl⟩
  obtain ⟨sb, h_s⟩ : ∃ sb, w / 2147483648 % 2 = sb := ⟨_, rfl⟩
  rw [h_e, h_m, h_s] at hov ⊢
  rw [h_e] at he0 hf
  have hm : m < 8388608 := by omega
  have hsb : sb < 2 := by omega
  have he256 : e < 256 := by omega
  clear h_e h_m h_s
  have h_re : (8388608 + m + 32767 + (8388608 + m) / 65536 % 2) / 65536 =
      128 + (m / 65536 + (m % 65536 + 32767 + m / 65536 % 2) / 65536) := by omega
  rw [h_re]
  have hk : m / 65536 + (m % 65536 + 32767 + m / 65536 % 2) / 65536 ≤ 128 := by omega
  generalize m / 65536 + (m % 65536 + 32767 + m / 65536 % 2) / 65536 = k at hk hov ⊢
  obtain ⟨f1, f2, f3, f4⟩ := bf16_fields hsb (by omega) hk hov
  rw [f1, f2, f3]
  refine ⟨by omega, f4, ?_⟩
  rw [ite_eq_right (show e + k / 128 ≠ 0 by omega), show max (e + k / 128) 1 = e + k / 128 by omega]
  have hs : ((sb == 1) = true) = (sb = 1) := by simp
  simp only [hs]
  rcases Nat.lt_or_ge k 128 with h | h
  · rw [Nat.div_eq_of_lt h, Nat.mod_eq_of_lt h, Nat.add_zero]
    push_cast; ring
  · obtain rfl : k = 128 := by omega
    rw [show (128 : ℕ) % 128 = 0 from rfl, show (128 : ℕ) / 128 = 1 from rfl]
    push_cast
    rw [show ((e : ℤ) + 1 - 134) = ((e : ℤ) - 134) + 1 by ring, zpow_add_one₀ two_ne_zero]
    ring

theorem word_fields {sb e mm : ℕ} (hsb : sb < 2) (he : e < 256) (hm : mm < 2 ^ 23) :
    ((sb * 2 ^ 31 + e * 2 ^ 23 + mm) >>> 23) % 256 = e ∧ (sb * 2 ^ 31 + e * 2 ^ 23 + mm) % 2 ^ 23 = mm ∧
      ((sb * 2 ^ 31 + e * 2 ^ 23 + mm) >>> 31) % 2 = sb := by
  simp only [Nat.shiftRight_eq_div_pow]
  omega

theorem wordQ_word {sb e mm : ℕ} (hsb : sb < 2) (he : e < 255) (he0 : e ≠ 0) (hm : mm < 2 ^ 23) :
    wordQ (sb * 2 ^ 31 + e * 2 ^ 23 + mm) =
      some ((if sb = 1 then (-1 : ℚ) else 1) * (((2 ^ 23 + mm : ℕ) : ℚ) * 2 ^ ((e : ℤ) - 150))) := by
  obtain ⟨f1, f2, f3⟩ := word_fields (e := e) hsb (by omega) hm
  rw [wordQ_fields (by rw [f1]; omega), f1, f2, f3, ite_eq_right he0, show max e 1 = e by omega]

/-- **`ofInt` is exact** below `2^24`: the word of `z` has exponent field `⌊log₂|z|⌋ + 127` and value `z`. -/
theorem ofInt_spec {z : ℤ} (h1 : z ≠ 0) (h2 : z.natAbs < 2 ^ 24) :
    wordQ (Words.ofInt z) = some (z : ℚ) ∧ (Words.ofInt z >>> 23) % 256 ≠ 0 := by
  have ha : z.natAbs ≠ 0 := Int.natAbs_ne_zero.2 h1
  have hl1 := Nat.log2_self_le ha
  have hl2 := Nat.lt_log2_self (n := z.natAbs)
  have hl3 : z.natAbs.log2 < 24 := (Nat.log2_lt ha).2 h2
  obtain ⟨e, he⟩ : ∃ e, z.natAbs.log2 = e := ⟨_, rfl⟩
  rw [he] at hl1 hl2 hl3
  have hd : z.natAbs - 2 ^ e < 2 ^ e := by rw [Nat.pow_succ] at hl2; omega
  have hm : (z.natAbs - 2 ^ e) <<< (23 - e) < 2 ^ 23 := by
    rw [Nat.shiftLeft_eq, show (2 : ℕ) ^ 23 = 2 ^ e * 2 ^ (23 - e) by rw [← Nat.pow_add]; congr 1; omega]
    exact Nat.mul_lt_mul_of_pos_right hd (Nat.two_pow_pos _)
  have hE : (e + 127) <<< 23 + (z.natAbs - 2 ^ e) <<< (23 - e) < 2 ^ 31 := by
    rw [Nat.shiftLeft_eq]; omega
  obtain ⟨sb, hsb, hs⟩ : ∃ sb, sb < 2 ∧ (if z < 0 then 0x80000000 else 0) = sb * 2 ^ 31 ∧ (sb = 1 ↔ z < 0) :=
    if hz : z < 0 then ⟨1, by omega, by rw [ite_eq_left hz]; rfl, by simp [hz]⟩
    else ⟨0, by omega, by rw [ite_eq_right hz]; rfl, by simp [hz]⟩
  have hw : Words.ofInt z = sb * 2 ^ 31 + (e + 127) * 2 ^ 23 + (z.natAbs - 2 ^ e) * 2 ^ (23 - e) := by
    unfold Words.ofInt
    simp only [ite_eq_right ha, he, hs.1, Nat.or_assoc, ← Nat.shiftLeft_add_eq_or_of_lt hm]
    rcases Nat.lt_or_ge sb 1 with h | h
    · obtain rfl : sb = 0 := by omega
      rw [Nat.zero_mul, Nat.zero_or, Nat.shiftLeft_eq, Nat.shiftLeft_eq]; omega
    · obtain rfl : sb = 1 := by omega
      rw [Nat.one_mul, Nat.or_comm, Nat.or_two_pow_eq_add_of_lt hE, Nat.shiftLeft_eq, Nat.shiftLeft_eq]; omega
  rw [hw]
  have hm' : (z.natAbs - 2 ^ e) * 2 ^ (23 - e) < 2 ^ 23 := by rw [← Nat.shiftLeft_eq]; exact hm
  obtain ⟨f1, -, -⟩ := word_fields (e := e + 127) hsb (by omega) hm'
  refine ⟨?_, by rw [f1]; omega⟩
  rw [wordQ_word (e := e + 127) hsb (by omega) (by omega) hm']
  congr 1
  have hv : (((2 ^ 23 + (z.natAbs - 2 ^ e) * 2 ^ (23 - e) : ℕ) : ℚ) * 2 ^ (((e + 127 : ℕ) : ℤ) - 150)) = z.natAbs := by
    have h23 : (2 : ℕ) ^ 23 = 2 ^ e * 2 ^ (23 - e) := by rw [← Nat.pow_add]; congr 1; omega
    rw [show 2 ^ 23 + (z.natAbs - 2 ^ e) * 2 ^ (23 - e) = z.natAbs * 2 ^ (23 - e) by
      rw [h23, ← Nat.add_mul]; congr 1; omega]
    push_cast
    rw [mul_assoc, ← zpow_natCast, ← zpow_add₀ two_ne_zero, show ((23 - e : ℕ) : ℤ) + ((e : ℤ) + 127 - 150) = 0 by omega,
      zpow_zero, mul_one]
  have hab : ((z.natAbs : ℕ) : ℚ) = |(z : ℚ)| := by simp
  rw [hv, hab]
  rcases lt_or_ge z 0 with hz | hz
  · rw [ite_eq_left (hs.2.2 hz), abs_of_neg (by exact_mod_cast hz)]; ring
  · rw [ite_eq_right (fun h => absurd (hs.2.1 h) (not_lt.2 hz)), abs_of_nonneg (by exact_mod_cast hz)]
    ring

/-- **A finite normal `f32Word` is `rn`**: from `2^−126` up, `f32Word` rounds to 24 bits in the value's binade as `rn`
does, and its exponent field is the binade's. -/
theorem f32Word_normal {y : ℚ} (hn : (2 : ℚ) ^ (-126 : ℤ) ≤ |y|) (hf : (f32Word y >>> 23) % 256 ≠ 255) :
    wordQ (f32Word y) = some (rn y) ∧ (f32Word y >>> 23) % 256 ≠ 0 := by
  have ha : 0 < |y| := lt_of_lt_of_le (zpow_pos (by norm_num) _) hn
  obtain ⟨hb1, hb2⟩ := Proofs.binade_spec ha
  obtain ⟨b, hbd⟩ : ∃ b, binade |y| = b := ⟨_, rfl⟩
  rw [hbd] at hb1 hb2
  have hb : -126 ≤ b := by
    by_contra h
    have : (2 : ℚ) ^ (b + 1) ≤ 2 ^ (-126 : ℤ) := zpow_le_zpow_right₀ (by norm_num) (by omega)
    linarith
  have hu : (0 : ℚ) < 2 ^ (b - 23) := zpow_pos (by norm_num) _
  have hq1 : (2 : ℚ) ^ 23 ≤ |y| / 2 ^ (b - 23) := by
    rw [le_div_iff₀ hu, ← zpow_natCast, ← zpow_add₀ two_ne_zero]; convert hb1 using 2; push_cast; ring
  have hq2 : |y| / 2 ^ (b - 23) < 2 ^ 24 := by
    rw [div_lt_iff₀ hu, ← zpow_natCast, ← zpow_add₀ two_ne_zero]; convert hb2 using 2; push_cast; ring
  obtain ⟨R, hRd⟩ : ∃ R, roundEven (|y| / 2 ^ (b - 23)) = R := ⟨_, rfl⟩
  have hR1 : 2 ^ 23 ≤ R := by
    rw [← hRd]; exact le_trans (Int.le_floor.2 (by exact_mod_cast hq1)) (Proofs.floor_le_roundEven _)
  have hR2 : R ≤ 2 ^ 24 := by
    have := Proofs.roundEven_le (|y| / 2 ^ (b - 23))
    have h2 : ⌊|y| / 2 ^ (b - 23)⌋ < 2 ^ 24 := Int.floor_lt.2 (by exact_mod_cast hq2)
    rw [hRd] at this
    omega
  obtain ⟨N, hN⟩ : ∃ N : ℕ, (N : ℤ) = R := ⟨R.toNat, Int.toNat_of_nonneg (by omega)⟩
  have hrn : rn y = (if y < 0 then -1 else 1) * ((N : ℚ) * 2 ^ (b - 23)) := by
    rw [rn, ulp, hbd, hRd, ← hN, Int.cast_natCast, mul_assoc]
  obtain ⟨sb, hsb, hs⟩ : ∃ sb, sb < 2 ∧ (if y < 0 then 2 ^ 31 else 0) = sb * 2 ^ 31 ∧ (sb = 1 ↔ y < 0) :=
    if hy : y < 0 then ⟨1, by omega, by rw [ite_eq_left hy]; rfl, by simp [hy]⟩
    else ⟨0, by omega, by rw [ite_eq_right hy]; rfl, by simp [hy]⟩
  have hsgn : (if sb = 1 then (-1 : ℚ) else 1) = if y < 0 then -1 else 1 := by
    by_cases hy : y < 0
    · rw [ite_eq_left (hs.2.2 hy), ite_eq_left hy]
    · rw [ite_eq_right (fun h => hy (hs.2.1 h)), ite_eq_right hy]
  have hN1 : 2 ^ 23 ≤ N := by omega
  have hN2 : N ≤ 2 ^ 24 := by omega
  have hw : f32Word y = if N = 2 ^ 24 then
        (if 127 < b + 1 then sb * 2 ^ 31 + 0x7F800000 else sb * 2 ^ 31 + (b + 1 + 127).toNat * 2 ^ 23 + 0)
      else (if 127 < b then sb * 2 ^ 31 + 0x7F800000 else sb * 2 ^ 31 + (b + 127).toNat * 2 ^ 23 + (N - 2 ^ 23)) := by
    unfold f32Word
    simp only [ite_eq_right ha.ne', hbd, max_eq_left hb, hRd, ← hN, Int.toNat_natCast, hs.1]
    by_cases hc : N = 2 ^ 24
    · simp only [hc, ite_true, show ¬ ((2 : ℕ) ^ 23 < 2 ^ 23) by omega, ite_false, Nat.sub_self]
    · simp only [hc, ite_false, show ¬ (N < 2 ^ 23) by omega]
  have hinf : ((sb * 2 ^ 31 + 0x7F800000) >>> 23) % 256 = 255 := by
    rw [Nat.shiftRight_eq_div_pow]; omega
  by_cases hc : N = 2 ^ 24
  · rw [ite_eq_left hc] at hw
    have h127 : ¬ 127 < b + 1 := fun h => hf (by rw [hw, ite_eq_left h]; exact hinf)
    rw [ite_eq_right h127] at hw
    obtain ⟨f1, -, -⟩ := word_fields (e := (b + 1 + 127).toNat) (mm := 0) hsb (by omega) (by norm_num)
    rw [hw, wordQ_word hsb (by omega) (by omega) (by norm_num), f1, hrn, hsgn, hc]
    refine ⟨?_, by omega⟩
    congr 1
    rw [show (((b + 1 + 127).toNat : ℕ) : ℤ) - 150 = (b - 23) + 1 by omega, zpow_add_one₀ two_ne_zero]
    push_cast; ring
  · rw [ite_eq_right hc] at hw
    have h127 : ¬ 127 < b := fun h => hf (by rw [hw, ite_eq_left h]; exact hinf)
    rw [ite_eq_right h127] at hw
    obtain ⟨f1, -, -⟩ := word_fields (e := (b + 127).toNat) (mm := N - 2 ^ 23) hsb (by omega) (by omega)
    rw [hw, wordQ_word hsb (by omega) (by omega) (by omega), f1, hrn, hsgn]
    refine ⟨?_, by omega⟩
    refine congrArg some ?_
    rw [show 2 ^ 23 + (N - 2 ^ 23) = N by omega, show (((b + 127).toNat : ℕ) : ℤ) - 150 = b - 23 by omega]

/-! ## The check's BF16 steps on values -/

theorem bind_eq_ok {ε α β : Type} {x : Except ε α} {f : α → Except ε β} {b : β} (h : (x >>= f) = .ok b) :
    ∃ a, x = .ok a ∧ f a = .ok b := by
  cases x with
  | error e => cases h
  | ok a => exact ⟨a, rfl, h⟩

theorem pure_eq_ok {ε α : Type} {a b : α} (h : (pure a : Except ε α) = .ok b) : a = b := by
  cases h; rfl

theorem ok_bind {ε α β : Type} (a : α) (f : α → Except ε β) : (Except.ok a >>= f) = f a := rfl

theorem finite_eq_ok {w a : ℕ} (h : Words.finite w = .ok a) : a = w ∧ (w >>> 23) % 256 ≠ 255 := by
  unfold Words.finite at h
  rw [and_ff] at h
  by_cases hf : (w >>> 23) % 256 = 255
  · simp [hf] at h
  · simp [hf] at h
    cases h
    exact ⟨rfl, hf⟩

theorem ofBf16_ok {h : ℕ} (he : bf16Exp h ≠ 255) : Words.ofBf16 h = .ok (h <<< 16) := by
  unfold Words.ofBf16 bf16Exp at *
  rw [and_ff]
  simp [he]
  rfl

theorem mul_word {a b : ℕ} {x y : ℚ} (hx : wordQ a = some x) (hy : wordQ b = some y) (h0 : x * y ≠ 0) :
    Verity.Fp32.f32MulV2 a b = f32Word (x * y) := by
  obtain ⟨ma, hda, rfl⟩ := CoreOps.decode_fin hx
  obtain ⟨mb, hdb, rfl⟩ := CoreOps.decode_fin hy
  have hv : (if (signBit a ^^ signBit b) = true then (-1 : ℚ) else 1) *
      (((ma * mb : ℕ) : ℚ) / ((1 <<< 149 : ℕ) : ℚ) / 2 ^ 149) =
      (if signBit a = true then (-1 : ℚ) else 1) * ((ma : ℚ) / 2 ^ 149) *
        ((if signBit b = true then (-1 : ℚ) else 1) * ((mb : ℚ) / 2 ^ 149)) := by
    rw [CoreOps.sgn_xor, Nat.one_shiftLeft]; push_cast; ring
  rw [CoreOps.mulV2_eq hda hdb, CoreOps.rnWord_round _ _ (by rw [Nat.one_shiftLeft]; exact Nat.two_pow_pos 149), hv,
    CoreOps.rnWord_of_ne h0]

theorem div_word {a b : ℕ} {x y : ℚ} (hx : wordQ a = some x) (hy : wordQ b = some y) (hx0 : x ≠ 0) (hy0 : y ≠ 0) :
    Verity.Fp32.f32DivV2 a b = f32Word (x / y) := by
  obtain ⟨ma, hda, rfl⟩ := CoreOps.decode_fin hx
  obtain ⟨mb, hdb, rfl⟩ := CoreOps.decode_fin hy
  have hmb : mb ≠ 0 := fun h => hy0 ((CoreOps.val_eq_zero _ _).2 h)
  rw [CoreOps.divV2_eq hda hdb hmb, CoreOps.rnWord_round _ _ (Nat.pos_of_ne_zero hmb), CoreOps.div_val,
    CoreOps.rnWord_of_ne (div_ne_zero hx0 hy0)]

/-- A BF16 word `h` (below `2^16`, finite) of value `v`. -/
def Bf16Of (h : ℕ) (v : ℚ) : Prop := h < 65536 ∧ bf16Exp h ≠ 255 ∧ bf16Value h = v

theorem ofBf16_rnd {h : ℕ} {y : ℚ} (hn : (2 : ℚ) ^ (-126 : ℤ) ≤ |y|)
    (hh : (Words.finite (f32Word y) >>= Words.toBf16) = .ok h) : Bf16Of h (bf16Rn (rn y)) := by
  obtain ⟨w', hf', hh'⟩ := bind_eq_ok hh
  obtain ⟨rfl, hf⟩ := finite_eq_ok hf'
  obtain ⟨hq, he0⟩ := f32Word_normal hn hf
  exact toBf16_spec hq he0 hh'

theorem bf16Mul_spec {a b h : ℕ} {x y : ℚ} (ha : Bf16Of a x) (hb : Bf16Of b y)
    (hn : (2 : ℚ) ^ (-126 : ℤ) ≤ |x * y|) (hh : Words.bf16Mul a b = .ok h) : Bf16Of h (bf16Rn (rn (x * y))) := by
  have h0 : x * y ≠ 0 := by
    intro h0; rw [h0, abs_zero] at hn; exact absurd hn (not_le.2 (zpow_pos (by norm_num) _))
  have hw := mul_word (wordQ_bf16 ha.2.1) (wordQ_bf16 hb.2.1) (by rw [ha.2.2, hb.2.2]; exact h0)
  rw [ha.2.2, hb.2.2, ← Properties.Evaluator.Fp32.f32MulV2] at hw
  unfold Words.bf16Mul at hh
  rw [ofBf16_ok ha.2.1, ofBf16_ok hb.2.1, ok_bind, ok_bind, hw] at hh
  exact ofBf16_rnd hn hh

theorem shl16_exp (b : ℕ) : ((b <<< 16) >>> 23) % 256 = bf16Exp b := by
  unfold bf16Exp; rw [Nat.shiftLeft_eq, Nat.shiftRight_eq_div_pow, Nat.shiftRight_eq_div_pow]; omega

theorem low31_ne {w : ℕ} {y : ℚ} (hf : (w >>> 23) % 256 ≠ 255) (hy : wordQ w = some y) (hy0 : y ≠ 0) :
    (w &&& 0x7FFFFFFF == 0) = false := by
  rw [show (0x7FFFFFFF : ℕ) = 2 ^ 31 - 1 from rfl, Nat.and_two_pow_sub_one_eq_mod]
  rw [wordQ_fields hf] at hy
  rw [beq_eq_false_iff_ne]
  intro h0
  have h1 : (w >>> 23) % 256 = 0 := by rw [Nat.shiftRight_eq_div_pow]; omega
  have h2 : w % 2 ^ 23 = 0 := by omega
  rw [h1, h2] at hy
  apply hy0
  rw [← Option.some.inj hy]
  simp

theorem bf16Div_spec {a b h : ℕ} {x y : ℚ} (ha : Bf16Of a x) (hb : Bf16Of b y) (hx0 : x ≠ 0) (hy0 : y ≠ 0)
    (hn : (2 : ℚ) ^ (-126 : ℤ) ≤ |x / y|) (hh : Words.bf16Div a b = .ok h) : Bf16Of h (bf16Rn (rn (x / y))) := by
  have hw := div_word (wordQ_bf16 ha.2.1) (wordQ_bf16 hb.2.1) (by rw [ha.2.2]; exact hx0) (by rw [hb.2.2]; exact hy0)
  rw [ha.2.2, hb.2.2, ← Properties.Evaluator.Fp32.f32DivV2] at hw
  have hc := low31_ne (by rw [shl16_exp]; exact hb.2.1) (wordQ_bf16 hb.2.1) (by rw [hb.2.2]; exact hy0)
  unfold Words.bf16Div at hh
  rw [ofBf16_ok ha.2.1, ofBf16_ok hb.2.1, ok_bind, ok_bind] at hh
  simp only [hc, Bool.false_eq_true, ite_false] at hh
  rw [hw] at hh
  exact ofBf16_rnd hn hh

theorem bf16Value_sgn (h : ℕ) : ∃ P : ℚ, 0 ≤ P ∧ (P = 0 ↔ bf16Value h = 0) ∧
    bf16Value h = (if (h >>> 15) % 2 = 1 then -1 else 1) * P := by
  refine ⟨(bf16Sig h : ℚ) * 2 ^ ((bf16ExpOr1 h : ℤ) - 134),
    mul_nonneg (Nat.cast_nonneg _) (zpow_pos (by norm_num) _).le, ?_, ?_⟩
  · unfold bf16Value
    rw [mul_assoc]
    constructor
    · intro h0; rw [h0, mul_zero]
    · intro h0
      rcases mul_eq_zero.1 h0 with h1 | h1
      · split at h1 <;> norm_num at h1
      · exact h1
  · unfold bf16Value bf16Sign
    rw [mul_assoc]
    by_cases hb : (h >>> 15) % 2 = 1
    · rw [ite_eq_left hb]; simp [hb]
    · rw [ite_eq_right hb]; simp [hb]

theorem sign_bf16 {h : ℕ} (h0 : bf16Value h ≠ 0) :
    (if ((h <<< 16) >>> 31) % 2 = 1 then 128 else 0) = (if bf16Value h < 0 then 128 else 0) := by
  have hs : ((h <<< 16) >>> 31) % 2 = (h >>> 15) % 2 := by
    rw [Nat.shiftLeft_eq, Nat.shiftRight_eq_div_pow, Nat.shiftRight_eq_div_pow]; omega
  obtain ⟨P, hP, hP0, hv⟩ := bf16Value_sgn h
  have hPpos : 0 < P := lt_of_le_of_ne hP (fun h' => h0 (hP0.1 h'.symm))
  rw [hs, hv]
  by_cases hb : (h >>> 15) % 2 = 1
  · rw [ite_eq_left hb, ite_eq_left hb, ite_eq_left (by linarith)]
  · rw [ite_eq_right hb, ite_eq_right hb, ite_eq_right (by linarith)]

theorem e4m3_bf16 {h : ℕ} (he : bf16Exp h ≠ 255) (h0 : bf16Value h ≠ 0) :
    e4m3Word (h <<< 16) = e4m3 (bf16Value h) := by
  apply Fin.ext
  have h1 : (e4m3Word (h <<< 16)).val =
      ((if ((h <<< 16) >>> 31) % 2 = 1 then 128 else 0) + e4m3Mag |bf16Value h|) % 256 := by
    unfold e4m3Word; rw [wordQ_bf16 he]
  have h2 : (e4m3 (bf16Value h)).val = ((if bf16Value h < 0 then 128 else 0) + e4m3Mag |bf16Value h|) % 256 := rfl
  rw [h1, h2, sign_bf16 h0]

theorem toFp8_spec {h c : ℕ} {v : ℚ} (hh : Bf16Of h v) (h0 : v ≠ 0) (hc : Words.toFp8 (h <<< 16) = .ok c) :
    c = (e4m3 v).val := by
  obtain ⟨hlt, he, rfl⟩ := hh
  unfold Words.toFp8 at hc
  obtain ⟨w, hf, hc⟩ := bind_eq_ok hc
  obtain ⟨rfl, -⟩ := finite_eq_ok hf
  cases hc
  rw [Properties.Evaluator.Casts.f32ToE4m3Sat,
    CoreOps.cast_e4m3Word (by rw [Nat.shiftLeft_eq]; omega), e4m3_bf16 he h0]

/-! ## The norms' BF16 words -/

theorem bf16Rn_pow (e : ℕ) : bf16Rn ((2 : ℚ) ^ e) = 2 ^ e := by
  have h := bf16Rn_fields False (2 ^ 23) (e + 127) (le_refl _) (by norm_num)
  rw [ite_eq_right id, one_mul, one_mul] at h
  have h2 : ((2 ^ 23 : ℕ) : ℚ) * 2 ^ (((e + 127 : ℕ) : ℤ) - 150) = 2 ^ e := by
    rw [show (((e + 127 : ℕ) : ℤ) - 150) = (e : ℤ) - 23 by push_cast; ring, zpow_sub₀ two_ne_zero, zpow_natCast]
    push_cast; field_simp
  rw [h2, roundEven_div_65536, show (2 ^ 23 + 32767 + 2 ^ 23 / 65536 % 2) / 65536 = 128 by norm_num,
    show (((e + 127 : ℕ) : ℤ) - 134) = (e : ℤ) - 7 by push_cast; ring, zpow_sub₀ two_ne_zero, zpow_natCast] at h
  rw [h]
  push_cast
  field_simp

theorem norm_E : Bf16Of Words.NORM_E 8192 ∧ bf16Rn ((256 : ℚ) * 32) = 8192 := by
  refine ⟨⟨by unfold Words.NORM_E; norm_num, by unfold Words.NORM_E bf16Exp; norm_num [Nat.shiftRight_eq_div_pow], ?_⟩, ?_⟩
  · unfold Words.NORM_E bf16Value bf16Sign bf16Sig bf16ExpOr1 bf16Exp bf16Man; norm_num [Nat.shiftRight_eq_div_pow]
  · have := bf16Rn_pow 13; norm_num at this ⊢; exact this

theorem norm_F : Bf16Of Words.NORM_F 512 ∧ bf16Rn ((16 : ℚ) * 32) = 512 := by
  refine ⟨⟨by unfold Words.NORM_F; norm_num, by unfold Words.NORM_F bf16Exp; norm_num [Nat.shiftRight_eq_div_pow], ?_⟩, ?_⟩
  · unfold Words.NORM_F bf16Value bf16Sign bf16Sig bf16ExpOr1 bf16Exp bf16Man; norm_num [Nat.shiftRight_eq_div_pow]
  · have := bf16Rn_pow 9; norm_num at this ⊢; exact this

/-! ## Magnitudes: each rounding keeps a value within a factor of two -/

theorem bf16Rn_half (z : ℚ) : |z| / 2 ≤ |bf16Rn z| ∧ |bf16Rn z| ≤ 2 * |z| := by
  have h1 := Peel.bf16Rn_sub_le z
  have h2 := abs_sub_abs_le_abs_sub z (bf16Rn z)
  have h3 := abs_sub_abs_le_abs_sub (bf16Rn z) z
  rw [abs_sub_comm] at h2
  have h4 : |z| / 2 ^ 8 ≤ |z| / 2 := div_le_div_of_nonneg_left (abs_nonneg _) (by norm_num) (by norm_num)
  have h5 : 0 ≤ |z| := abs_nonneg _
  constructor <;> linarith

theorem rn_half (z : ℚ) : |z| / 2 ≤ |rn z| ∧ |rn z| ≤ 2 * |z| := by
  have h1 := Peel.rn_sub_le z
  have h2 := abs_sub_abs_le_abs_sub z (rn z)
  have h3 := abs_sub_abs_le_abs_sub (rn z) z
  rw [abs_sub_comm] at h2
  have h4 : |z| / 2 ^ 24 ≤ |z| / 2 := div_le_div_of_nonneg_left (abs_nonneg _) (by norm_num) (by norm_num)
  have h5 : 0 ≤ |z| := abs_nonneg _
  constructor <;> linarith

theorem min_le : (2 : ℚ) ^ (-126 : ℤ) ≤ 1 / 2 ^ 20 := by
  rw [zpow_neg]; norm_num

/-- The scale's quotient and the scale: from `512 ≤ |a|` and `0 < |b| ≤ 2^25`, the quotient is at least `2^−126` and its
rounded scale at least `2^−18`. -/
theorem scale_bound {a b : ℚ} (ha : 512 ≤ |a|) (hb1 : 0 < |b|) (hb2 : |b| ≤ 2 ^ 25) :
    (2 : ℚ) ^ (-126 : ℤ) ≤ |a / b| ∧ 1 / 2 ^ 18 ≤ |bf16Rn (rn (a / b))| := by
  have hq : 1 / 2 ^ 16 ≤ |a / b| := by
    rw [abs_div, le_div_iff₀ hb1]
    calc 1 / 2 ^ 16 * |b| ≤ 1 / 2 ^ 16 * 2 ^ 25 := by gcongr
      _ = 512 := by norm_num
      _ ≤ |a| := ha
  refine ⟨le_trans min_le (le_trans (by norm_num) hq), ?_⟩
  have h1 := (rn_half (a / b)).1
  have h2 := (bf16Rn_half (rn (a / b))).1
  linarith

theorem elem_bound {p s : ℚ} (hp : 1 / 2 ≤ |p|) (hs : 1 / 2 ^ 18 ≤ |s|) :
    (2 : ℚ) ^ (-126 : ℤ) ≤ |p * s| ∧ bf16Rn (rn (p * s)) ≠ 0 := by
  have hps : 1 / 2 ^ 19 ≤ |p * s| := by
    rw [abs_mul]
    calc (1 : ℚ) / 2 ^ 19 = 1 / 2 * (1 / 2 ^ 18) := by norm_num
      _ ≤ |p| * |s| := mul_le_mul hp hs (by norm_num) (abs_nonneg _)
  refine ⟨le_trans min_le (le_trans (by norm_num) hps), fun h0 => ?_⟩
  have h1 := (rn_half (p * s)).1
  have h2 := (bf16Rn_half (rn (p * s))).1
  rw [h0, abs_zero] at h2
  linarith

/-! ## `isqrt` is `Nat.sqrt` -/

theorem newton_ge {n s x : ℕ} (hs : s * s ≤ n) (hx : 0 < x) : s ≤ (x + n / x) / 2 := by
  have h := Nat.div_add_mod n x
  have hr := Nat.mod_lt n hx
  by_contra hc
  have hq : x + n / x + 1 ≤ 2 * s := by omega
  obtain ⟨q, hq'⟩ : ∃ q, n / x = q := ⟨_, rfl⟩
  obtain ⟨r, hr'⟩ : ∃ r, n % x = r := ⟨_, rfl⟩
  rw [hq'] at h hq
  rw [hr'] at h hr
  have h1 : x * (x + q + 1) ≤ x * (2 * s) := Nat.mul_le_mul_left x hq
  zify at h hr hs h1
  nlinarith [sq_nonneg ((s : ℤ) - x)]

theorem newton_lt {n s x : ℕ} (hn : n < (s + 1) * (s + 1)) (hx : s < x) : n / x ≤ s := by
  have : n < (s + 1) * x := lt_of_lt_of_le hn (Nat.mul_le_mul_left _ hx)
  have := (Nat.div_lt_iff_lt_mul (by omega)).2 this
  omega

theorem go_eq {n s : ℕ} (hs : s * s ≤ n) (hn : n < (s + 1) * (s + 1)) (hs1 : 0 < s) :
    ∀ (f x : ℕ), s ≤ x → x - s < 2 ^ f → Evaluate.Fp32.isqrt.go n f x = s := by
  intro f
  induction f with
  | zero => intro x h1 h2; simp only [Evaluate.Fp32.isqrt.go]; simp at h2; omega
  | succ f ih =>
    intro x h1 h2
    simp only [Evaluate.Fp32.isqrt.go]
    have hge := newton_ge hs (show 0 < x by omega)
    rcases Nat.eq_or_lt_of_le h1 with rfl | hlt
    · rw [ite_eq_right (by omega)]
    · have hq := newton_lt hn hlt
      rw [ite_eq_left (by omega)]
      apply ih _ hge
      rw [Nat.pow_succ] at h2
      omega

theorem isqrt_eq (n : ℕ) : Evaluate.Fp32.isqrt n = Nat.sqrt n := by
  unfold Evaluate.Fp32.isqrt
  rcases Nat.eq_zero_or_pos n with rfl | hn
  · simp
  rw [ite_eq_right hn.ne']
  have hs := Nat.sqrt_le n
  have hlt := Nat.lt_succ_sqrt n
  have hs1 : 0 < Nat.sqrt n := Nat.sqrt_pos.2 hn
  have hL := Nat.lt_log2_self (n := n)
  obtain ⟨L, hLd⟩ : ∃ L, n.log2 = L := ⟨_, rfl⟩
  rw [hLd] at hL ⊢
  rw [Nat.one_shiftLeft]
  have hx0 : Nat.sqrt n < 2 ^ (L / 2 + 1) := by
    apply Nat.mul_self_lt_mul_self_iff.1
    calc Nat.sqrt n * Nat.sqrt n ≤ n := hs
      _ < 2 ^ (L + 1) := hL
      _ ≤ 2 ^ (L / 2 + 1) * 2 ^ (L / 2 + 1) := by
        rw [← Nat.pow_add]; exact Nat.pow_le_pow_right (by norm_num) (by omega)
  have hx1 : 2 ^ (L / 2 + 1) ≤ 2 ^ (L + 4) := Nat.pow_le_pow_right (by norm_num) (by omega)
  exact go_eq hs hlt hs1 _ _ hx0.le (by omega)

/-! ## Lists and byte arrays -/

theorem toList_loop (bs : ByteArray) : ∀ (k i : ℕ) (r : List UInt8), bs.size - i = k → i ≤ bs.size →
    ByteArray.toList.loop bs i r = r.reverse ++ bs.data.toList.drop i := by
  intro k
  induction k with
  | zero =>
    intro i r hk hi
    have hi' : i = bs.size := by omega
    rw [ByteArray.toList.loop, ite_eq_right (by omega), hi', ByteArray.size, ← Array.length_toList,
      List.drop_length, List.append_nil]
  | succ k ih =>
    intro i r hk hi
    have hlt : i < bs.size := by omega
    rw [ByteArray.toList.loop, ite_eq_left hlt, ih (i + 1) _ (by omega) (by omega)]
    have hl : i < bs.data.toList.length := by rw [Array.length_toList]; exact hlt
    rw [List.drop_eq_getElem_cons hl, List.reverse_cons, List.append_assoc, List.singleton_append]
    congr 2
    rw [Array.getElem_toList]
    obtain ⟨d⟩ := bs
    exact getElem!_pos d i hlt

theorem toList_eq (bs : ByteArray) : bs.toList = bs.data.toList := by
  unfold ByteArray.toList
  rw [toList_loop bs _ 0 [] rfl (Nat.zero_le _)]
  simp

theorem foldl_add {α : Type} (g : α → ℕ) : ∀ (l : List α) (a : ℕ),
    l.foldl (fun s x => s + g x) a = a + (l.map g).sum
  | [], a => by simp
  | x :: l, a => by rw [List.foldl_cons, foldl_add g l, List.map_cons, List.sum_cons]; ring

theorem mapM_ok {α β ε : Type} (f : α → Except ε β) : ∀ (l : List α) (ys : List β), l.mapM f = .ok ys →
    ys.length = l.length ∧ ∀ i (h1 : i < l.length) (h2 : i < ys.length), f l[i] = .ok ys[i]
  | [], ys, h => by
      rw [List.mapM_nil] at h
      cases h
      exact ⟨rfl, fun i h1 => absurd h1 (Nat.not_lt_zero _)⟩
  | a :: l, ys, h => by
      rw [List.mapM_cons] at h
      cases hfa : f a with
      | error e => rw [hfa] at h; cases h
      | ok b =>
        rw [hfa] at h
        cases hl : l.mapM f with
        | error e => simp only [hl] at h; cases h
        | ok bs =>
          simp only [hl] at h
          cases h
          obtain ⟨h1, h2⟩ := mapM_ok f l bs hl
          refine ⟨by simp [h1], fun i hi1 hi2 => ?_⟩
          cases i with
          | zero => exact hfa
          | succ i => exact h2 i (by simpa using hi1) (by simpa using hi2)

theorem array_mapM_ok {α β ε : Type} (f : α → Except ε β) (xs : Array α) (ys : Array β) (h : xs.mapM f = .ok ys) :
    ys.size = xs.size ∧ ∀ i (h1 : i < xs.size) (h2 : i < ys.size), f xs[i] = .ok ys[i] := by
  rw [Array.mapM_eq_mapM_toList] at h
  cases hl : xs.toList.mapM f with
  | error e => simp only [hl] at h; cases h
  | ok bs =>
    simp only [hl] at h
    cases h
    obtain ⟨h1, h2⟩ := mapM_ok f _ _ hl
    refine ⟨by simp [h1], fun i hi1 hi2 => ?_⟩
    have := h2 i (by simpa using hi1) (by simpa [h1] using hi1)
    simpa using this

/-! ## The digest's length -/

theorem compress_length (h w : List ℕ) : (Pouw.PearlC.Sha512.compress h w).length = 8 := by
  unfold Pouw.PearlC.Sha512.compress; rfl

theorem foldl_compress_length : ∀ (L : List (List ℕ)) (h : List ℕ), h.length = 8 →
    (L.foldl (fun h blk => Pouw.PearlC.Sha512.compress h (Pouw.PearlC.Sha512.blockWords blk)) h).length = 8
  | [], _, hh => hh
  | _ :: L, _, _ => foldl_compress_length L _ (compress_length _ _)

theorem digest_length (m : List ℕ) : (Pouw.PearlC.Sha512.digest m).length = 64 := by
  unfold Pouw.PearlC.Sha512.digest
  rw [List.length_flatMap]
  have hb : ∀ x, (Pouw.PearlC.Sha512.beBytes 8 x).length = 8 := fun x => by simp [Pouw.PearlC.Sha512.beBytes]
  simp only [hb, List.map_const', List.sum_replicate, Nat.nsmul_eq_mul]
  rw [foldl_compress_length _ _ rfl]

theorem s512_length (role : String) (m : List ℕ) : (Pouw.PearlC.NoiseSha512.s512 role m).length = 32 := by
  rw [Pouw.PearlC.NoiseSha512.s512, List.length_take, digest_length]
  rfl

/-! ## The line: its bytes, integers and scale -/

theorem ofInt_bf16 {z : ℤ} {h : ℕ} (h1 : z ≠ 0) (h2 : z.natAbs < 2 ^ 24) (hh : Words.toBf16 (Words.ofInt z) = .ok h) :
    Bf16Of h (bf16Rn z) := by
  obtain ⟨hq, he0⟩ := ofInt_spec h1 h2
  exact toBf16_spec hq he0 hh

theorem bytesNat_pair {a b : ℕ} (ha : a < 256) (hb : b < 256) : bytesNat (Words.bytesOf [a, b]) = [a, b] := by
  simp [bytesNat, Words.bytesOf, Nat.mod_eq_of_lt ha, Nat.mod_eq_of_lt hb]

theorem lineInt_byte (b : UInt8) :
    (if b.toNat >>> 7 = 1 then (-1 : ℤ) else 1) * ((b.toNat &&& 0x7F : ℕ) + 1 : ℤ) =
      (if b.toNat / 128 % 2 = 1 then -1 else 1) * (((b.toNat % 128 : ℕ) : ℤ) + 1) := by
  have hb := b.toNat_lt
  have h1 : b.toNat >>> 7 = b.toNat / 128 % 2 := by rw [Nat.shiftRight_eq_div_pow]; omega
  have h2 : b.toNat &&& 0x7F = b.toNat % 128 := Nat.and_two_pow_sub_one_eq_mod _ 7
  rw [h1, h2]

theorem xs_eq (raw : ByteArray) :
    (raw.toList.toArray.map fun b : UInt8 =>
      (if b.toNat >>> 7 = 1 then (-1 : ℤ) else 1) * ((b.toNat &&& 0x7F : ℕ) + 1 : ℤ)).toList =
      NoiseSha512.lineInts (bytesNat raw) := by
  rw [Array.toList_map, List.toList_toArray, toList_eq]
  unfold NoiseSha512.lineInts bytesNat
  rw [List.map_map]
  exact List.map_congr_left fun b _ => lineInt_byte b

theorem lineInts_natAbs {L : List ℕ} {x : ℤ} (hx : x ∈ NoiseSha512.lineInts L) : 1 ≤ x.natAbs ∧ x.natAbs ≤ 128 := by
  unfold NoiseSha512.lineInts at hx
  obtain ⟨b, -, rfl⟩ := List.mem_map.1 hx
  have : b % 128 < 128 := Nat.mod_lt _ (by norm_num)
  split <;> omega

theorem sq_eq (xs : Array ℤ) :
    xs.foldl (fun s x => s + x.natAbs * x.natAbs) 0 * 32 * 32 = (xs.toList.map fun x => x.natAbs ^ 2).sum * 1024 := by
  rw [← Array.foldl_toList, foldl_add (fun x : ℤ => x.natAbs * x.natAbs), Nat.zero_add]
  simp only [pow_two]
  ring

theorem sq_bounds {xs : List ℤ} (hl : xs.length = 32) (hx : ∀ x ∈ xs, 1 ≤ x.natAbs ∧ x.natAbs ≤ 128) :
    32 ≤ (xs.map fun x => x.natAbs ^ 2).sum ∧ (xs.map fun x => x.natAbs ^ 2).sum ≤ 524288 := by
  constructor
  · have := List.length_nsmul_le_sum (xs.map fun x => x.natAbs ^ 2) 1 (by
      intro y hy
      obtain ⟨x, hx', rfl⟩ := List.mem_map.1 hy
      have := (hx x hx').1
      exact Nat.one_le_pow _ _ this)
    rw [List.length_map, hl, Nat.nsmul_eq_mul] at this
    omega
  · have := List.sum_le_length_nsmul (xs.map fun x => x.natAbs ^ 2) 16384 (by
      intro y hy
      obtain ⟨x, hx', rfl⟩ := List.mem_map.1 hy
      have := (hx x hx').2
      calc x.natAbs ^ 2 ≤ 128 ^ 2 := Nat.pow_le_pow_left this 2
        _ = 16384 := by norm_num)
    rw [List.length_map, hl, Nat.nsmul_eq_mul] at this
    omega

theorem coe_list (I : List ℤ) : ((do let a ← I; pure (a : ℚ)) : List ℚ) = I.map (fun a : ℤ => (a : ℚ)) := by
  simp [List.map_eq_flatMap]

theorem norm_word {norm32 : ℕ} {norm : ℚ}
    (h : norm32 = Words.NORM_E ∧ norm = 256 ∨ norm32 = Words.NORM_F ∧ norm = 16) :
    Bf16Of norm32 (bf16Rn (norm * 32)) ∧ 512 ≤ |bf16Rn (norm * 32)| := by
  rcases h with ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩
  · rw [norm_E.2]; exact ⟨norm_E.1, by norm_num⟩
  · rw [norm_F.2]; exact ⟨norm_F.1, by norm_num⟩

end NoiseProofs

section Line

open Pouw.Fp8Atom Pouw.PearlC Pouw.PearlC.PeelExact Pouw.PearlC.PeelExactProofs

open NoiseProofs in
theorem lineIsTheSpec : LineIsTheSpec := by
  intro seed side factor line norm32 norm codes hside hfactor hnorm h
  obtain ⟨raw, hraw⟩ : ∃ raw, Words.s512 "line" (Words.lineKey seed ++ Words.bytesOf [side, factor] ++ Words.le 4 line) =
      raw := ⟨_, rfl⟩
  have hL : bytesNat raw = NoiseSha512.s512 "line"
      (NoiseSha512.s512 "noise-line" (bytesNat seed) ++ [side, factor] ++ Sha512.leBytes 4 line) := by
    rw [← hraw, (seedsAreTheSpec).1 "line" (by simp [roles]), bytesNat_append, bytesNat_append, Words.lineKey,
      (seedsAreTheSpec).1 "noise-line" (by simp [roles]), bytesNat_le, bytesNat_pair hside hfactor]
  unfold Words.sampleLine at h
  rw [hraw] at h
  obtain ⟨xs, hxs⟩ : ∃ xs : Array ℤ, (raw.toList.toArray.map fun b : UInt8 =>
      (if b.toNat >>> 7 = 1 then (-1 : ℤ) else 1) * ((b.toNat &&& 0x7F : ℕ) + 1 : ℤ)) = xs := ⟨_, rfl⟩
  have hxl : xs.toList = NoiseSha512.lineInts (bytesNat raw) := by rw [← hxs, xs_eq]
  simp only [hxs] at h
  rw [sq_eq, hxl, isqrt_eq] at h
  obtain ⟨d, hd, h⟩ := bind_eq_ok h
  obtain ⟨sc, hsc, h⟩ := bind_eq_ok h
  obtain ⟨hsize, hcodes⟩ := array_mapM_ok _ _ _ h
  -- the line's integers
  set I := NoiseSha512.lineInts (bytesNat raw) with hI
  have hIl : I.length = 32 := by
    rw [hI, NoiseSha512.lineInts, List.length_map, hL, s512_length]
  have hxsize : xs.size = 32 := by rw [← Array.length_toList, hxl, hIl]
  have hImem : ∀ x ∈ I, 1 ≤ x.natAbs ∧ x.natAbs ≤ 128 := fun x hx => lineInts_natAbs hx
  -- the scale
  obtain ⟨hS1, hS2⟩ := sq_bounds hIl hImem
  set S := (I.map fun x => x.natAbs ^ 2).sum with hS
  have hr1 : 0 < Nat.sqrt (S * 1024) := Nat.sqrt_pos.2 (by omega)
  have hr2 : Nat.sqrt (S * 1024) < 2 ^ 24 := Nat.sqrt_lt'.2 (by
    calc S * 1024 ≤ 524288 * 1024 := Nat.mul_le_mul_right _ hS2
      _ < (2 ^ 24) ^ 2 := by norm_num)
  have hD := ofInt_bf16 (z := (Nat.sqrt (S * 1024) : ℤ)) (by omega) (by rw [Int.natAbs_natCast]; exact hr2) hd
  rw [Int.cast_natCast] at hD
  obtain ⟨hN, hN512⟩ := norm_word hnorm
  have hR : (1 : ℚ) ≤ (Nat.sqrt (S * 1024) : ℚ) := by exact_mod_cast hr1
  have hR2 : (Nat.sqrt (S * 1024) : ℚ) < 2 ^ 24 := by exact_mod_cast hr2
  obtain ⟨hB1, hB2⟩ := bf16Rn_half (Nat.sqrt (S * 1024) : ℚ)
  rw [abs_of_pos (show (0 : ℚ) < (Nat.sqrt (S * 1024) : ℚ) by linarith)] at hB1 hB2
  have hb0 : 0 < |bf16Rn (Nat.sqrt (S * 1024) : ℚ)| := by linarith
  obtain ⟨hq, hsc0⟩ := scale_bound hN512 hb0 (by linarith)
  have hSc := bf16Div_spec hN hD (fun h0 => by rw [h0, abs_zero] at hN512; linarith)
    (fun h0 => by rw [h0, abs_zero] at hb0; exact lt_irrefl _ hb0) hq hsc
  have hscale : bf16Rn (rn (bf16Rn (norm * 32) / bf16Rn (Nat.sqrt (S * 1024) : ℚ))) = NoiseSha512.lineScale norm I := by
    rw [NoiseSha512.lineScale]
  rw [hscale] at hSc hsc0
  -- each code
  refine ⟨by rw [hsize, hxsize], fun t => ?_⟩
  have ht : t.val < 32 := t.isLt
  have hF := hcodes t.val (by rw [hxsize]; exact ht) (by rw [hsize, hxsize]; exact ht)
  have htI : t.val < I.length := by rw [hIl]; exact ht
  have hxt : xs[t.val]'(by rw [hxsize]; exact ht) = I[t.val]'htI := by
    rw [← Array.getElem_toList]; simp only [hxl]
  rw [hxt] at hF
  obtain ⟨x, hxdef⟩ : ∃ x, I[t.val]'htI = x := ⟨_, rfl⟩
  rw [hxdef] at hF
  obtain ⟨hx1, hx2⟩ := hImem x (hxdef ▸ List.getElem_mem htI)
  obtain ⟨a, ha, hF⟩ := bind_eq_ok hF
  obtain ⟨p, hp, hF⟩ := bind_eq_ok hF
  obtain ⟨w, hw, hF⟩ := bind_eq_ok hF
  have hx0 : x ≠ 0 := by omega
  have hA := ofInt_bf16 (z := x) hx0 (by omega) ha
  have hxq : (1 : ℚ) ≤ |(x : ℚ)| := by
    rw [← Int.cast_abs]; exact_mod_cast Int.one_le_abs hx0
  have hA2 : 1 / 2 ≤ |bf16Rn (x : ℚ)| := by
    have := (bf16Rn_half (x : ℚ)).1; linarith
  obtain ⟨hpn, hp0⟩ := elem_bound hA2 hsc0
  have hP := bf16Mul_spec hA hSc hpn hp
  rw [ofBf16_ok hP.2.1] at hw
  cases hw
  rw [getElem!_pos codes t.val (by rw [hsize, hxsize]; exact ht), toFp8_spec hP hp0 hF]
  unfold NoiseSha512.sampleLine NoiseSha512.lineList
  dsimp only
  rw [← hL, ← hI]
  rw [coe_list I, List.map_map, List.getD_eq_getElem _ _ (by simp only [List.length_map]; omega), List.getElem_map]
  simp only [Function.comp_apply, hxdef]

open NoiseProofs in
theorem NoiseProofs.basis_spec {seedB : ByteArray} {side k : ℕ} {fa : Array (Array ℕ)} (hside : side < 256)
    (h : Words.basis seedB side k = .ok fa) :
    fa.size = k ∧ ∀ l < k, ∀ t : Fin Params.pi.r,
      (fa[l]!)[t.val]! = (NoiseSha512.sampleLine (bytesNat seedB) side 1 l 16 t).val := by
  unfold Words.basis at h
  obtain ⟨hs, hf⟩ := array_mapM_ok _ _ _ h
  rw [Array.size_range] at hs
  refine ⟨hs, fun l hl t => ?_⟩
  have := hf l (by rw [Array.size_range]; exact hl) (by rw [hs]; exact hl)
  rw [Array.getElem_range] at this
  rw [getElem!_pos fa l (by rw [hs]; exact hl)]
  exact (lineIsTheSpec seedB side 1 l Words.NORM_F 16 _ hside (by norm_num) (.inr ⟨rfl, rfl⟩) this).2 t

open NoiseProofs in
theorem fixedNoiseIsTheSpec : FixedNoiseIsTheSpec := by
  intro salt rootB k wn h
  unfold Words.fixedNoise at h
  obtain ⟨fa, hfa, h⟩ := bind_eq_ok h
  obtain ⟨fb, hfb, h⟩ := bind_eq_ok h
  obtain ⟨gram, -, h⟩ := bind_eq_ok h
  rw [← pure_eq_ok h]
  obtain ⟨ha1, ha2⟩ := basis_spec (by norm_num) hfa
  obtain ⟨hb1, hb2⟩ := basis_spec (by norm_num) hfb
  exact ⟨(seedsAreTheSpec).2.1 salt rootB, ha1, hb1, fun l hl t => ⟨ha2 l hl t, hb2 l hl t⟩⟩

end Line

end Pouw.PearlPlus.Bridge
