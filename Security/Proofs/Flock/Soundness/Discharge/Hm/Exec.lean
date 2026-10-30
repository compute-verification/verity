/-
The verifier's SHA-512 compression (`Flock.Sha512.compress`, straight-line over `UInt64` locals) is `Spec.compressV`:
FIPS 180-4's three loops as folds (`compress_unfold`), then each fold against the spec by induction.
-/
import Mathlib.Tactic
import Proofs.Flock.Soundness.Discharge.Hm.Spec
-- `hash_eq`'s `simp` over `Sha512.hash`'s loops (`Hm.Hash`) needs `Std.Legacy.Range.mk.congr_simp`, which
-- `FlockLevel3.LookupRows` already realizes; a second copy there is refused by the audit's kernel replay.
import Proofs.Flock.Level3.LookupRows

namespace FlockSoundness.Discharge.Hm

open Flock.HmNets hiding and

open Flock

/-- Message word `i` of the block at `off`, big-endian (`Sha512.compress`'s first loop). -/
def wordBE (data : ByteArray) (off i : ℕ) : UInt64 :=
  (List.range' 0 8).foldl (fun x j => x <<< 8 ||| (data.get! (off + 8 * i + j)).toUInt64) 0

/-- `Sha512.compress`'s schedule loop body at `i`. -/
def schedU (w : Array UInt64) (i : ℕ) : Array UInt64 :=
  w.push (w[i - 16]! + (Sha512.rotr w[i - 15]! 1 ^^^ Sha512.rotr w[i - 15]! 8 ^^^ w[i - 15]! >>> 7) + w[i - 7]! +
    (Sha512.rotr w[i - 2]! 19 ^^^ Sha512.rotr w[i - 2]! 61 ^^^ w[i - 2]! >>> 6))

abbrev U8 := UInt64 × UInt64 × UInt64 × UInt64 × UInt64 × UInt64 × UInt64 × UInt64

/-- `Sha512.compress`'s round loop body at `i`, on the state `(a, b, c, d, e, f, g, h)`. -/
def roundU (w : Array UInt64) (s : U8) (i : ℕ) : U8 :=
  (s.2.2.2.2.2.2.2 + (Sha512.rotr s.2.2.2.2.1 14 ^^^ Sha512.rotr s.2.2.2.2.1 18 ^^^ Sha512.rotr s.2.2.2.2.1 41) +
        (s.2.2.2.2.1 &&& s.2.2.2.2.2.1 ^^^ ~~~s.2.2.2.2.1 &&& s.2.2.2.2.2.2.1) + Sha512.k[i]! + w[i]! +
      ((Sha512.rotr s.1 28 ^^^ Sha512.rotr s.1 34 ^^^ Sha512.rotr s.1 39) +
        (s.1 &&& s.2.1 ^^^ s.1 &&& s.2.2.1 ^^^ s.2.1 &&& s.2.2.1)),
    s.1, s.2.1, s.2.2.1,
    s.2.2.2.1 + (s.2.2.2.2.2.2.2 + (Sha512.rotr s.2.2.2.2.1 14 ^^^ Sha512.rotr s.2.2.2.2.1 18 ^^^
        Sha512.rotr s.2.2.2.2.1 41) + (s.2.2.2.2.1 &&& s.2.2.2.2.2.1 ^^^ ~~~s.2.2.2.2.1 &&& s.2.2.2.2.2.2.1) +
        Sha512.k[i]! + w[i]!),
    s.2.2.2.2.1, s.2.2.2.2.2.1, s.2.2.2.2.2.2.1)

theorem wordBE_eq (data : ByteArray) (off i : ℕ) : wordBE data off i = Sha512.word data (off + 8 * i) := rfl

theorem schedU_eq (w : Array UInt64) (i : ℕ) :
    schedU w i = w.push (Sha512.sched w[i - 16]! w[i - 15]! w[i - 7]! w[i - 2]!) := rfl

theorem roundU_eq (w : Array UInt64) (s : U8) (i : ℕ) : roundU w s i = Sha512.round s i w[i]! := rfl

theorem getElem!_mapRange (f : ℕ → UInt64) {m i : ℕ} (hi : i < m) : ((List.range m).map f).toArray[i]! = f i := by
  simp [hi]

/-- The schedule loop over the words `f 0, …, f 15` gives `f 0, …, f (16 + n - 1)` when `f` follows the recurrence. -/
theorem sched_fold (f : ℕ → UInt64)
    (hf : ∀ i, 16 ≤ i → i < 80 → f i = Sha512.sched (f (i - 16)) (f (i - 15)) (f (i - 7)) (f (i - 2))) :
    ∀ n ≤ 64, (List.range' 16 n).foldl schedU ((List.range 16).map f).toArray = ((List.range (16 + n)).map f).toArray
  | 0, _ => rfl
  | n + 1, hn => by
      rw [List.range'_concat, List.foldl_append, sched_fold f hf n (by omega), List.foldl_cons, List.foldl_nil,
        schedU_eq, Nat.one_mul, getElem!_mapRange f (by omega), getElem!_mapRange f (by omega),
        getElem!_mapRange f (by omega), getElem!_mapRange f (by omega), ← hf _ (by omega) (by omega),
        ← Nat.add_assoc, List.range_succ, List.map_append, List.map_singleton, ← List.push_toArray]

/-- The round loop from `S 0` gives `S n` when each `S (i + 1)` is round `i` of `S i`. -/
theorem round_fold (w : Array UInt64) (S : ℕ → U8) (hS : ∀ i < 80, S (i + 1) = Sha512.round (S i) i w[i]!) :
    ∀ n ≤ 80, (List.range' 0 n).foldl (roundU w) (S 0) = S n
  | 0, _ => rfl
  | n + 1, hn => by
      rw [List.range'_concat, List.foldl_append, round_fold w S hS n (by omega), List.foldl_cons, List.foldl_nil,
        roundU_eq, Nat.zero_add, Nat.one_mul, hS n (by omega)]

theorem compress_unfold (h : Array UInt64) (data : ByteArray) (off : ℕ) :
    Sha512.compress h data off =
      let w := (List.range' 16 64).foldl schedU
        ((List.range' 0 16).foldl (fun w i => w.push (wordBE data off i)) (Array.mkEmpty 80))
      let s := (List.range' 0 80).foldl (roundU w) (h[0]!, h[1]!, h[2]!, h[3]!, h[4]!, h[5]!, h[6]!, h[7]!)
      #[h[0]! + s.1, h[1]! + s.2.1, h[2]! + s.2.2.1, h[3]! + s.2.2.2.1, h[4]! + s.2.2.2.2.1,
        h[5]! + s.2.2.2.2.2.1, h[6]! + s.2.2.2.2.2.2.1, h[7]! + s.2.2.2.2.2.2.2] := by
  unfold Sha512.compress
  extract_lets w0 w1 w2 w3 w4 w5 w6 w7 w8 w9 w10 w11 w12 w13 w14 w15 w16 w17 w18 w19 w20 w21 w22 w23 w24 w25
    w26 w27 w28 w29 w30 w31 w32 w33 w34 w35 w36 w37 w38 w39 w40 w41 w42 w43 w44 w45 w46 w47 w48 w49 w50 w51
    w52 w53 w54 w55 w56 w57 w58 w59 w60 w61 w62 w63 w64 w65 w66 w67 w68 w69 w70 w71 w72 w73 w74 w75 w76 w77
    w78 w79 s0 s1 s2 s3 s4 s5 s6 s7 s8 s9 s10 s11 s12 s13 s14 s15 s16 s17 s18 s19 s20 s21 s22 s23 s24 s25 s26
    s27 s28 s29 s30 s31 s32 s33 s34 s35 s36 s37 s38 s39 s40 s41 s42 s43 s44 s45 s46 s47 s48 s49 s50 s51 s52
    s53 s54 s55 s56 s57 s58 s59 s60 s61 s62 s63 s64 s65 s66 s67 s68 s69 s70 s71 s72 s73 s74 s75 s76 s77 s78
    s79 s80 W sR
  let f : ℕ → UInt64 := fun i => [
    w0, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, w14, w15, w16, w17, w18, w19, w20, w21, w22,
    w23, w24, w25, w26, w27, w28, w29, w30, w31, w32, w33, w34, w35, w36, w37, w38, w39, w40, w41, w42, w43,
    w44, w45, w46, w47, w48, w49, w50, w51, w52, w53, w54, w55, w56, w57, w58, w59, w60, w61, w62, w63, w64,
    w65, w66, w67, w68, w69, w70, w71, w72, w73, w74, w75, w76, w77, w78, w79].getD i 0
  have h1 : (List.range' 0 16).foldl (fun w i => w.push (wordBE data off i)) (Array.mkEmpty 80) =
      ((List.range 16).map f).toArray := rfl
  have hW : W = ((List.range 80).map f).toArray :=
    (congrArg (fun a => (List.range' 16 64).foldl schedU a) h1).trans
      (sched_fold f (fun i _ _ => by interval_cases i <;> rfl) 64 le_rfl)
  let S : ℕ → U8 := fun i => [
    s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15, s16, s17, s18, s19, s20, s21, s22,
    s23, s24, s25, s26, s27, s28, s29, s30, s31, s32, s33, s34, s35, s36, s37, s38, s39, s40, s41, s42, s43,
    s44, s45, s46, s47, s48, s49, s50, s51, s52, s53, s54, s55, s56, s57, s58, s59, s60, s61, s62, s63, s64,
    s65, s66, s67, s68, s69, s70, s71, s72, s73, s74, s75, s76, s77, s78, s79, s80].getD i default
  have hs : sR = s80 := by
    show (List.range' 0 80).foldl (roundU W) _ = _
    rw [hW]
    exact round_fold _ S (fun i _ => by interval_cases i <;> rfl) 80 le_rfl
  rw [hs]

/-! ## The loops are the spec -/

/-- The words of an array as bit vectors. -/
def toBV (w : Array UInt64) : List (BitVec 64) := w.toList.map UInt64.toBitVec

theorem getD_toBV (w : Array UInt64) (i : ℕ) : (toBV w).getD i 0 = (w[i]!).toBitVec := by
  rw [toBV, List.getD_eq_getElem?_getD, List.getElem?_map, Array.getElem?_toList, getElem!_def]
  cases w[i]? <;> rfl

theorem length_toBV (w : Array UInt64) : (toBV w).length = w.size := by simp [toBV]

theorem rotr_bv (x n : UInt64) (k : ℕ) (hk : 0 < k ∧ k < 64) (h1 : n.toBitVec % 64 = BitVec.ofNat 64 k)
    (h2 : (64 - n).toBitVec % 64 = BitVec.ofNat 64 (64 - k)) :
    (Sha512.rotr x n).toBitVec = x.toBitVec.rotateRight k := by
  unfold Sha512.rotr
  rw [UInt64.toBitVec_or, UInt64.toBitVec_shiftRight, UInt64.toBitVec_shiftLeft, h1, h2, BitVec.rotateRight_def,
    BitVec.ushiftRight_eq', BitVec.shiftLeft_eq', BitVec.toNat_ofNat, BitVec.toNat_ofNat, Nat.mod_eq_of_lt (a := k) (by omega),
    Nat.mod_eq_of_lt (a := k) (by omega), Nat.mod_eq_of_lt (a := 64 - k) (by omega)]

theorem shr_bv (x n : UInt64) (k : ℕ) (hk : k < 64) (h1 : n.toBitVec % 64 = BitVec.ofNat 64 k) :
    (x >>> n).toBitVec = x.toBitVec >>> k := by
  rw [UInt64.toBitVec_shiftRight, h1, BitVec.ushiftRight_eq', BitVec.toNat_ofNat, Nat.mod_eq_of_lt (a := k) (by omega)]

theorem loop1_toList (data : ByteArray) (off : ℕ) : ∀ (l : List ℕ) (w : Array UInt64),
    (l.foldl (fun w i => w.push (wordBE data off i)) w).toList = w.toList ++ l.map (wordBE data off)
  | [], w => by simp
  | i :: l, w => by
      rw [List.foldl_cons, loop1_toList data off l, Array.toList_push, List.map_cons, List.append_assoc]
      rfl

theorem schedU_bv (w : Array UInt64) : toBV (schedU w w.size) = schedStepV (toBV w) := by
  simp only [schedU, schedStepV, toBV, Array.toList_push, List.map_append, List.map_cons, List.map_nil,
    List.length_map, Array.length_toList]
  congr 2
  rw [← toBV, getD_toBV, getD_toBV, getD_toBV, getD_toBV]
  simp only [UInt64.toBitVec_add, UInt64.toBitVec_xor, ssig0, ssig1,
    rotr_bv _ 1 1 (by omega) (by decide) (by decide), rotr_bv _ 8 8 (by omega) (by decide) (by decide),
    rotr_bv _ 19 19 (by omega) (by decide) (by decide), rotr_bv _ 61 61 (by omega) (by decide) (by decide),
    shr_bv _ 7 7 (by omega) (by decide), shr_bv _ 6 6 (by omega) (by decide)]

theorem size_schedU (w : Array UInt64) (i : ℕ) : (schedU w i).size = w.size + 1 := by simp [schedU]

theorem loop2_bv : ∀ (n : ℕ) (w : Array UInt64), 16 ≤ w.size →
    toBV ((List.range' w.size n).foldl schedU w) = scheduleV n (toBV w)
  | 0, w, _ => rfl
  | n + 1, w, h => by
      rw [List.range'_succ, List.foldl_cons, scheduleV, ← schedU_bv w, ← size_schedU w w.size]
      exact loop2_bv n _ (by rw [size_schedU]; omega)

/-- The round state as registers. -/
def toR (s : U8) : Regs (BitVec 64) :=
  ⟨s.1.toBitVec, s.2.1.toBitVec, s.2.2.1.toBitVec, s.2.2.2.1.toBitVec, s.2.2.2.2.1.toBitVec, s.2.2.2.2.2.1.toBitVec,
    s.2.2.2.2.2.2.1.toBitVec, s.2.2.2.2.2.2.2.toBitVec⟩

theorem roundU_bv (w : Array UInt64) (s : U8) (i : ℕ) :
    toR (roundU w s i) = roundV (kAt i) (w[i]!).toBitVec (toR s) := by
  simp only [roundU, roundV, toR, kAt, chV, majV, bsig0, bsig1, UInt64.toBitVec_add, UInt64.toBitVec_xor,
    UInt64.toBitVec_and, UInt64.toBitVec_not,
    rotr_bv _ 14 14 (by omega) (by decide) (by decide), rotr_bv _ 18 18 (by omega) (by decide) (by decide),
    rotr_bv _ 41 41 (by omega) (by decide) (by decide), rotr_bv _ 28 28 (by omega) (by decide) (by decide),
    rotr_bv _ 34 34 (by omega) (by decide) (by decide), rotr_bv _ 39 39 (by omega) (by decide) (by decide)]

theorem loop3_bv (w : Array UInt64) : ∀ (n t : ℕ) (s : U8),
    toR ((List.range' t n).foldl (roundU w) s) = roundsV (toBV w) n t (toR s)
  | 0, t, s => rfl
  | n + 1, t, s => by
      rw [List.range'_succ, List.foldl_cons, roundsV, loop3_bv w n (t + 1), roundU_bv, getD_toBV]

/-- **The verifier's SHA-512 compression is `compressV`**: `Sha512.compress h data off`'s words, as bit vectors, are
`compressV` of `h`'s first eight words and the block's sixteen big-endian message words. -/
theorem compress_eq (h : Array UInt64) (data : ByteArray) (off : ℕ) :
    toBV (Sha512.compress h data off) =
      compressV ((List.range 8).map fun i => h[i]!.toBitVec) ((List.range 16).map fun i => (wordBE data off i).toBitVec) := by
  rw [compress_unfold]
  dsimp only
  have h16 : ((List.range' 0 16).foldl (fun w i => w.push (wordBE data off i)) (Array.mkEmpty 80)).size = 16 := by
    rw [← Array.length_toList, loop1_toList]; simp
  have hm : toBV ((List.range' 0 16).foldl (fun w i => w.push (wordBE data off i)) (Array.mkEmpty 80)) =
      (List.range 16).map fun i => (wordBE data off i).toBitVec := by
    rw [toBV, loop1_toList]; simp [List.range_eq_range']
  have hs := loop2_bv 64 _ (by rw [h16])
  rw [h16, hm] at hs
  rw [compressV, ← hs]
  have hr := loop3_bv (List.foldl schedU
    (List.foldl (fun w i => w.push (wordBE data off i)) (Array.mkEmpty 80) (List.range' 0 16)) (List.range' 16 64)) 80 0
    (h[0]!, h[1]!, h[2]!, h[3]!, h[4]!, h[5]!, h[6]!, h[7]!)
  have hcv : Regs.ofList ((List.range 8).map fun i => h[i]!.toBitVec) = toR (h[0]!, h[1]!, h[2]!, h[3]!, h[4]!, h[5]!,
      h[6]!, h[7]!) := rfl
  rw [hcv, ← hr]
  generalize List.foldl (roundU _) _ _ = s
  simp [toBV, toR, Regs.toList, List.range_succ]

end FlockSoundness.Discharge.Hm
