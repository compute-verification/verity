import PouwPearlPlus.Words
import Definitions.Pouw.PearlC.NoiseSha512
import Proofs.Flock.Soundness.Discharge.Hm.Hash
import Mathlib.Data.List.GetD
import Mathlib.Data.List.Basic
import Mathlib.Tactic.Ring
import Mathlib.Tactic.NormNum

/-!
# Pearl+'s bridge, part (b), the noise: the check's -h3 noise is `NoiseSha512`'s

The check hashes with the compiled SHA-512 every tree reads (`Flock.Sha512.hash`) and keys a row's seed by its
`sha512/row-seg/v1` digest (`Flock.HmRow.segDigest`). The deployed semantics (`sha512Sem`) draws the same lines through the
spec's SHA-512 (`Pouw.PearlC.Sha512.digest`) and row digest (`NoiseSha512.rowDigest`). Each statement here says the
check's value is the spec's on the same bytes, and where the check returns a value, the spec's line or seed is it.
-/

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

end Pouw.PearlPlus.Bridge
