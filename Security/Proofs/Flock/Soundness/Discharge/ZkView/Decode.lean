import Proofs.Flock.Soundness.Discharge.ZkView.Statements
import Proofs.Flock.Soundness.Refine.Frames

/-!
# `Zk.decode` inverts `encZk`

`Reads m b x` says the reader `m`, started at the first byte of `b` inside any input that fits a `u64`, returns `x` and
stops after the last byte of `b`. `Reads.bind` composes it, so the proof walks `Zk.reader` field by field against
`encZk`'s pieces: integers (`Reads.u64`, `Reads.u32`, `Reads.u8`), field elements, length-prefixed vectors (exact lengths
from the `Shape`, or any length with a minimum element size), the per-level openings and loops, the level-0 messages and
the inner proof. `reads_reader` is the whole file; `decode_encode` starts it at offset 0 with nothing around it.
-/

namespace FlockSoundness.Discharge.ZkView

open Flock Reader

/-- `m` reads `b` at the cursor and returns `x`, whatever lies before and after, on an input whose length fits a `u64`. -/
def Reads {α : Type} (m : M α) (b : ByteArray) (x : α) : Prop :=
  ∀ pre post : ByteArray, (pre ++ b ++ post).size < 2 ^ 64 →
    m.run ⟨pre ++ b ++ post, pre.size⟩ = .ok (x, ⟨pre ++ b ++ post, pre.size + b.size⟩)

theorem Reads.bind {α β : Type} {m : M α} {k : α → M β} {b b' : ByteArray} {x : α} {y : β} (h : Reads m b x)
    (h' : Reads (k x) b' y) : Reads (m >>= k) (b ++ b') y := by
  intro pre post hfit
  have e1 : pre ++ (b ++ b') ++ post = pre ++ b ++ (b' ++ post) := by simp [ByteArray.append_assoc]
  have e2 : pre ++ b ++ (b' ++ post) = (pre ++ b) ++ b' ++ post := by simp [ByteArray.append_assoc]
  rw [e1] at hfit ⊢
  rw [StateT.run_bind, h pre (b' ++ post) hfit]
  change (k x).run ⟨pre ++ b ++ (b' ++ post), pre.size + b.size⟩ = _
  rw [e2] at hfit ⊢
  rw [show pre.size + b.size = (pre ++ b).size from ByteArray.size_append.symm, h' (pre ++ b) post hfit,
    ByteArray.size_append, ByteArray.size_append, Nat.add_assoc]

theorem Reads.pure {α : Type} (x : α) : Reads (Pure.pure x : M α) ByteArray.empty x := fun _ _ _ => rfl

/-- To show `Reads m b x`, `b` may be taken to fit a `u64`. -/
theorem Reads.of_fit {α : Type} {m : M α} {b : ByteArray} {x : α} (h : b.size < 2 ^ 64 → Reads m b x) : Reads m b x :=
  fun pre post hfit => h (by simp only [ByteArray.size_append] at hfit; omega) pre post hfit

theorem Reads.congr {α : Type} {m : M α} {b b' : ByteArray} {x x' : α} (h : Reads m b x) (hb : b = b') (hx : x = x') :
    Reads m b' x' := hb ▸ hx ▸ h

theorem get!_eq (b : ByteArray) (i : ℕ) : b.get! i = b.data[i]! := by cases b; rfl

theorem get!_mid (pre b post : ByteArray) (i : ℕ) (hi : i < b.size) :
    (pre ++ b ++ post).get! (pre.size + i) = b.get! i := by
  rw [get!_eq, get!_eq, getElem!_pos _ _ (by simp [ByteArray.size_data] at *; omega),
    getElem!_pos _ _ (by simpa [ByteArray.size_data] using hi)]
  simp only [ByteArray.data_append]
  rw [Array.getElem_append_left (by simp [ByteArray.size_data] at *; omega),
    Array.getElem_append_right (by simp [ByteArray.size_data])]
  congr 1
  simp [ByteArray.size_data]

theorem Reads.u8 (x : UInt8) : Reads Reader.u8 (ByteArray.mk #[x]) x := by
  intro pre post _
  have hs : (ByteArray.mk #[x]).size = 1 := rfl
  have hg := get!_mid pre (ByteArray.mk #[x]) post 0 (by rw [hs]; omega)
  simp only [Nat.add_zero] at hg
  simp +arith [Reader.u8, Reader.need, StateT.run_bind, hs, ByteArray.size_append]
  rw [hg]
  rfl


/-! ## Little-endian integers -/

theorem le64_size (x : UInt64) : (le64 x).size = 8 := by
  rw [← ByteArray.size_data, ← Array.length_toList, Refine.le64_data]
  simp

theorem le64_get! (x : UInt64) (i : ℕ) (hi : i < 8) : (le64 x).get! i = (x >>> (8 * i).toUInt64).toUInt8 := by
  have h := congrArg (·[i]?) (Refine.le64_data x)
  simp only [Array.getElem?_toList, List.getElem?_map, List.getElem?_range hi, Option.map_some] at h
  rw [get!_eq, getElem!_def, h]

theorem u64OfLE_eq (d : ByteArray) (o : ℕ) :
    u64OfLE d o = (List.range' 0 8).foldl (fun acc i => acc ||| (d.get! (o + i)).toUInt64 <<< (8 * i).toUInt64) 0 := by
  simp only [u64OfLE, Id.run, Std.Legacy.Range.forIn_eq_forIn_range', Std.Legacy.Range.size]
  rw [List.forIn_pure_yield_eq_foldl]
  rfl

theorem foldl_le64 (x : UInt64) (k : ℕ) (hk : k ≤ 8) :
    ((List.range' 0 k).foldl (fun acc i => acc ||| (x >>> (8 * i).toUInt64).toUInt8.toUInt64 <<< (8 * i).toUInt64)
      0).toNat = x.toNat % 2 ^ (8 * k) := by
  induction k with
  | zero => rw [Nat.mul_zero, Nat.pow_zero, Nat.mod_one]; rfl
  | succ k ih =>
    rw [List.range'_concat, List.foldl_append, List.foldl_cons, List.foldl_nil, UInt64.toNat_or, ih (by omega)]
    have h8 : (8 * (0 + 1 * k)).toUInt64.toNat = 8 * k := by
      rw [Nat.toUInt64_eq, UInt64.toNat_ofNat_of_lt' (by simp [UInt64.size]; omega)]
      omega
    rw [UInt64.toNat_shiftLeft, h8, UInt8.toNat_toUInt64, UInt64.toNat_toUInt8, UInt64.toNat_shiftRight, h8,
      Nat.mod_eq_of_lt (show 8 * k < 64 by omega)]
    apply Nat.eq_of_testBit_eq
    intro j
    simp only [Nat.testBit_or, Nat.testBit_mod_two_pow, Nat.testBit_shiftLeft, Nat.testBit_shiftRight]
    rcases Nat.lt_or_ge j (8 * k) with h | h
    · simp [h, show j < 8 * (k + 1) by omega, show ¬ 8 * k ≤ j by omega]
    rcases Nat.lt_or_ge j (8 * (k + 1)) with h' | h'
    · simp [h, h', show ¬ j < 8 * k by omega, show j < 64 by omega,
        show j - 8 * k < 8 by omega, show 8 * k + (j - 8 * k) = j by omega]
    · simp [show ¬ j < 8 * k by omega, show ¬ j < 8 * (k + 1) by omega, show ¬ j - 8 * k < 8 by omega]

theorem u64OfLE_mid (pre post : ByteArray) (x : UInt64) : u64OfLE (pre ++ le64 x ++ post) pre.size = x := by
  rw [u64OfLE_eq, List.foldl_ext _
    (fun acc i => acc ||| (x >>> (8 * i).toUInt64).toUInt8.toUInt64 <<< (8 * i).toUInt64) _ fun acc i hi => by
      have hi8 := (List.mem_range'_1.1 hi).2
      rw [get!_mid _ _ _ _ (by rw [le64_size]; omega), le64_get! x i (by omega)]]
  apply UInt64.toNat_inj.1
  rw [foldl_le64 x 8 le_rfl]
  exact Nat.mod_eq_of_lt x.toNat_lt

theorem Reads.u64 (x : UInt64) : Reads Reader.u64 (le64 x) x := by
  intro pre post _
  simp +arith [Reader.u64, Reader.need, StateT.run_bind, le64_size, ByteArray.size_append]
  rw [u64OfLE_mid]
  rfl

theorem le32_data (x : UInt32) :
    (le32 x).data.toList = (List.range 4).map fun i => (x >>> (8 * i).toUInt32).toUInt8 := by
  simp [le32, List.range', ByteArray.push, List.range_succ]
  rfl

theorem le32_size (x : UInt32) : (le32 x).size = 4 := by
  rw [← ByteArray.size_data, ← Array.length_toList, le32_data]
  simp

theorem le32_get! (x : UInt32) (i : ℕ) (hi : i < 4) : (le32 x).get! i = (x >>> (8 * i).toUInt32).toUInt8 := by
  have h := congrArg (·[i]?) (le32_data x)
  simp only [Array.getElem?_toList, List.getElem?_map, List.getElem?_range hi, Option.map_some] at h
  rw [get!_eq, getElem!_def, h]

theorem u32OfLE_eq (d : ByteArray) (o : ℕ) :
    u32OfLE d o = (List.range' 0 4).foldl (fun acc i => acc ||| (d.get! (o + i)).toUInt32 <<< (8 * i).toUInt32) 0 := by
  simp only [u32OfLE, Id.run, Std.Legacy.Range.forIn_eq_forIn_range', Std.Legacy.Range.size]
  rw [List.forIn_pure_yield_eq_foldl]
  rfl

theorem foldl_le32 (x : UInt32) (k : ℕ) (hk : k ≤ 4) :
    ((List.range' 0 k).foldl (fun acc i => acc ||| (x >>> (8 * i).toUInt32).toUInt8.toUInt32 <<< (8 * i).toUInt32)
      0).toNat = x.toNat % 2 ^ (8 * k) := by
  induction k with
  | zero => rw [Nat.mul_zero, Nat.pow_zero, Nat.mod_one]; rfl
  | succ k ih =>
    rw [List.range'_concat, List.foldl_append, List.foldl_cons, List.foldl_nil, UInt32.toNat_or, ih (by omega)]
    have h8 : (8 * (0 + 1 * k)).toUInt32.toNat = 8 * k := by
      rw [Nat.toUInt32_eq, UInt32.toNat_ofNat_of_lt' (by simp [UInt32.size]; omega)]
      omega
    rw [UInt32.toNat_shiftLeft, h8, UInt8.toNat_toUInt32, UInt32.toNat_toUInt8, UInt32.toNat_shiftRight, h8,
      Nat.mod_eq_of_lt (show 8 * k < 32 by omega)]
    apply Nat.eq_of_testBit_eq
    intro j
    simp only [Nat.testBit_or, Nat.testBit_mod_two_pow, Nat.testBit_shiftLeft, Nat.testBit_shiftRight]
    rcases Nat.lt_or_ge j (8 * k) with h | h
    · simp [h, show j < 8 * (k + 1) by omega, show ¬ 8 * k ≤ j by omega]
    rcases Nat.lt_or_ge j (8 * (k + 1)) with h' | h'
    · simp [h, h', show ¬ j < 8 * k by omega, show j < 32 by omega,
        show j - 8 * k < 8 by omega, show 8 * k + (j - 8 * k) = j by omega]
    · simp [show ¬ j < 8 * k by omega, show ¬ j < 8 * (k + 1) by omega, show ¬ j - 8 * k < 8 by omega]

theorem u32OfLE_mid (pre post : ByteArray) (x : UInt32) : u32OfLE (pre ++ le32 x ++ post) pre.size = x := by
  rw [u32OfLE_eq, List.foldl_ext _
    (fun acc i => acc ||| (x >>> (8 * i).toUInt32).toUInt8.toUInt32 <<< (8 * i).toUInt32) _ fun acc i hi => by
      have hi8 := (List.mem_range'_1.1 hi).2
      rw [get!_mid _ _ _ _ (by rw [le32_size]; omega), le32_get! x i (by omega)]]
  apply UInt32.toNat_inj.1
  rw [foldl_le32 x 4 le_rfl]
  exact Nat.mod_eq_of_lt x.toNat_lt

theorem Reads.u32 (x : UInt32) : Reads Reader.u32 (le32 x) x := by
  intro pre post _
  simp +arith [Reader.u32, Reader.need, StateT.run_bind, le32_size, ByteArray.size_append]
  rw [u32OfLE_mid]
  rfl

theorem extract_mid (pre b post : ByteArray) : (pre ++ (b ++ post)).extract pre.size (pre.size + b.size) = b := by
  have := ByteArray.extract_append_size_add (a := pre) (b := b ++ post) (i := 0) (j := b.size)
  rw [Nat.add_zero] at this
  rw [this, ByteArray.extract_append_eq_left rfl]

theorem Reads.bytes (b : ByteArray) {n : ℕ} (hn : b.size = n) : Reads (Reader.bytes n) b b := by
  subst hn
  intro pre post _
  simp +arith [Reader.bytes, Reader.need, StateT.run_bind, ByteArray.size_append]
  rw [ByteArray.append_assoc]
  rw [Nat.add_comm b.size pre.size, extract_mid]
  rfl

/-! ## Field elements -/

theorem toBytes_split (a : F128) : a.toBytes = le64 a.lo ++ le64 a.hi := by
  apply ByteArray.ext
  apply Array.toList_inj.1
  rw [ByteArray.toList_data_append, Refine.le64_data, Refine.le64_data]
  apply List.ext_getElem
  · simp [F128.toBytes]
  · intro i h1 h2
    simp only [F128.toBytes, Array.toList_ofFn, List.getElem_ofFn]
    rcases Nat.lt_or_ge i 8 with h | h
    · simp [h, List.getElem_append_left (by simpa using h : i < ((List.range 8).map _).length)]
    · simp at h2
      rw [List.getElem_append_right (by simpa using h)]
      simp [show ¬ i < 8 by omega]

theorem Reads.f128 (a : F128) : Reads Reader.f128 a.toBytes a := by
  unfold Reader.f128
  exact Reads.congr (Reads.bind (Reads.u64 a.lo) (Reads.bind (Reads.u64 a.hi) (Reads.pure _)))
    (by rw [toBytes_split, ByteArray.append_empty]) rfl

theorem Reads.f256 (x : F256) : Reads Reader.f256 (wF256 x) x := by
  unfold Reader.f256
  exact Reads.congr (Reads.bind (Reads.f128 x.c0) (Reads.bind (Reads.f128 x.c1) (Reads.pure _)))
    (by rw [wF256, ByteArray.append_empty]) rfl


/-! ## Vectors and loops -/

/-- The elements' encodings, one after another. -/
def catL {α : Type} (f : α → ByteArray) : List α → ByteArray
  | [] => ByteArray.empty
  | x :: l => f x ++ catL f l

theorem foldl_append_eq {α : Type} (f : α → ByteArray) :
    ∀ (l : List α) (init : ByteArray), l.foldl (fun b x => b ++ f x) init = init ++ catL f l
  | [], init => by simp [catL]
  | x :: l, init => by rw [List.foldl_cons, foldl_append_eq f l, catL, ByteArray.append_assoc]

theorem wVec_eq {α : Type} (f : α → ByteArray) (xs : Array α) : wVec f xs = wLen xs.size ++ catL f xs.toList := by
  rw [wVec, ← Array.foldl_toList, foldl_append_eq]

theorem catL_size_ge {α : Type} (f : α → ByteArray) (m : ℕ) :
    ∀ l : List α, (∀ x ∈ l, m ≤ (f x).size) → l.length * m ≤ (catL f l).size
  | [], _ => by simp
  | x :: l, h => by
    have := catL_size_ge f m l fun y hy => h y (List.mem_cons_of_mem _ hy)
    have hx := h x List.mem_cons_self
    rw [catL, ByteArray.size_append, List.length_cons, Nat.succ_mul]
    omega

theorem Reads.forIn_push {α : Type} (g : ℕ → Array α → M (ForInStep (Array α))) (f : α → ByteArray) :
    ∀ (l : List α) (s : ℕ) (acc : Array α),
      (∀ i (hi : i < l.length) acc, Reads (g (s + i) acc) (f l[i]) (.yield (acc.push l[i]))) →
      Reads (forIn (List.range' s l.length) acc g) (catL f l) (acc ++ l.toArray)
  | [], s, acc, _ => by simpa [catL] using Reads.pure acc
  | x :: l, s, acc, h => by
    rw [List.length_cons, List.range'_succ, List.forIn_cons]
    have h0 : Reads (g s acc) (f x) (.yield (acc.push x)) := h 0 (by simp) acc
    have ih := Reads.forIn_push g f l (s + 1) (acc.push x) fun i hi acc' => by
      have := h (i + 1) (by simp; omega) acc'
      simpa [Nat.add_assoc, Nat.add_comm 1 i] using this
    exact Reads.congr (Reads.bind h0 ih) rfl (by simp)

theorem Reads.forIn_range {α : Type} (g : ℕ → Array α → M (ForInStep (Array α))) (f : α → ByteArray) (xs : Array α)
    (n : ℕ) (hn : xs.size = n) (acc : Array α)
    (h : ∀ i (hi : i < xs.size) acc, Reads (g i acc) (f xs[i]) (.yield (acc.push xs[i]))) :
    Reads (forIn [:n] acc g) (catL f xs.toList) (acc ++ xs) := by
  subst hn
  rw [Std.Legacy.Range.forIn_eq_forIn_range']
  have := Reads.forIn_push g f xs.toList 0 acc fun i hi acc => by
    simpa using h i (by simpa using hi) acc
  simpa [Std.Legacy.Range.size] using this

theorem toNat_toUInt64_of_lt {n : ℕ} (h : n < 2 ^ 64) : n.toUInt64.toNat = n := by
  rw [Nat.toUInt64_eq, UInt64.toNat_ofNat_of_lt' (by simpa [UInt64.size] using h)]

theorem Reads.vecExact {α : Type} {elem : M α} {f : α → ByteArray} (xs : Array α) {want : ℕ} (what : String)
    (hw : xs.size = want) (hne : ∀ x ∈ xs, 1 ≤ (f x).size) (he : ∀ x ∈ xs, Reads elem (f x) x) :
    Reads (Reader.vecExact want elem what) (wVec f xs) xs := by
  subst hw
  unfold Reader.vecExact
  rw [wVec_eq]
  refine Reads.bind (Reads.u64 _) (Reads.of_fit fun hb => ?_)
  have h64 : xs.size < 2 ^ 64 := by
    have := catL_size_ge f 1 xs.toList (by simpa using hne)
    simp only [Array.length_toList, Nat.mul_one] at this
    omega
  simp only [toNat_toUInt64_of_lt h64, bne_self_eq_false, Bool.false_eq_true, ite_false]
  exact Reads.congr (Reads.bind (Reads.forIn_range _ f xs xs.size rfl _ fun i hi acc =>
    Reads.congr (Reads.bind (he _ (Array.getElem_mem hi)) (Reads.pure _)) ByteArray.append_empty rfl) (Reads.pure _))
    ByteArray.append_empty (by simp)


theorem Reads.vecAny {α : Type} {elem : M α} {f : α → ByteArray} (xs : Array α) {minSize : ℕ} (what : String)
    (h1 : 1 ≤ minSize) (hmin : ∀ x ∈ xs, minSize ≤ (f x).size) (he : ∀ x ∈ xs, Reads elem (f x) x) :
    Reads (Reader.vecAny minSize elem what) (wVec f xs) xs := by
  unfold Reader.vecAny
  rw [wVec_eq]
  refine Reads.bind (Reads.u64 _) (Reads.of_fit fun hb => ?_)
  have hm := catL_size_ge f minSize xs.toList (by simpa using hmin)
  simp only [Array.length_toList] at hm
  have h64 : xs.size < 2 ^ 64 := by
    have : xs.size ≤ xs.size * minSize := Nat.le_mul_of_pos_right _ h1
    omega
  simp only [toNat_toUInt64_of_lt h64]
  intro pre post hfit
  simp only [StateT.run_bind, StateT.run_get, pure_bind]
  rw [ite_eq_right_of_eq_false _ _ (eq_false (by simp only [ByteArray.size_append]; omega))]
  refine (?_ : Reads _ _ _) pre post hfit
  exact Reads.congr (Reads.bind (Reads.forIn_range _ f xs xs.size rfl _ fun i hi acc =>
    Reads.congr (Reads.bind (he _ (Array.getElem_mem hi)) (Reads.pure _)) ByteArray.append_empty rfl) (Reads.pure _))
    ByteArray.append_empty (by simp)

theorem Reads.map {α β : Type} {m : M α} {b : ByteArray} {x : α} (g : α → β) (h : Reads m b x) :
    Reads (g <$> m) b (g x) := by
  intro pre post hfit
  rw [StateT.run_map, h pre post hfit]
  rfl

theorem Reads.byteVec (b : ByteArray) : Reads Zk.byteVec (wBytes b) b := by
  unfold Zk.byteVec
  refine Reads.of_fit fun hb => ?_
  have h64 : b.size < 2 ^ 64 := by simp only [wBytes, ByteArray.size_append] at hb; omega
  exact Reads.congr (Reads.bind (Reads.u64 _) (Reads.bytes b (toNat_toUInt64_of_lt h64).symm)) rfl rfl

theorem toBytes_size (a : F128) : a.toBytes.size = 16 := by
  rw [toBytes_split, ByteArray.size_append, le64_size, le64_size]

theorem Reads.pair (q : F128 × F128) :
    Reads (do let a ← Reader.f128; let b ← Reader.f128; Pure.pure (a, b)) (wPair q) q :=
  Reads.congr (Reads.bind (Reads.f128 q.1) (Reads.bind (Reads.f128 q.2) (Reads.pure _)))
    (by rw [wPair, ByteArray.append_empty]) rfl

theorem Reads.pair256 (q : F256 × F256) :
    Reads (do let a ← Reader.f256; let b ← Reader.f256; Pure.pure (a, b)) (wF256 q.1 ++ wF256 q.2) q :=
  Reads.congr (Reads.bind (Reads.f256 q.1) (Reads.bind (Reads.f256 q.2) (Reads.pure _)))
    (by rw [ByteArray.append_empty]) rfl

theorem Reads.vec0 {α : Type} (elem : M α) (what : String) : Reads (Reader.vecExact 0 elem what) (wLen 0) #[] :=
  Reads.congr (Reads.vecExact (f := fun _ => ByteArray.empty) #[] what rfl (by simp) (by simp)) (by simp [wVec]) rfl

theorem Reads.f128s (xs : Array F128) {want : ℕ} (what : String) (hw : xs.size = want) :
    Reads (Reader.vecExact want Reader.f128 what) (wVec F128.toBytes xs) xs :=
  Reads.vecExact xs what hw (fun x _ => by rw [toBytes_size]; omega) fun x _ => Reads.f128 x

theorem Reads.u64s (xs : Array UInt64) {want : ℕ} (what : String) (hw : xs.size = want) :
    Reads (Reader.vecExact want Reader.u64 what) (wVec le64 xs) xs :=
  Reads.vecExact xs what hw (fun x _ => by rw [le64_size]; omega) fun x _ => Reads.u64 x

theorem Reads.hashes (sh : Shape) (hL : 0 < sh.hashLen) (xs : Array ByteArray) {want : ℕ} (what : String)
    (hw : xs.size = want) (hx : ∀ s ∈ xs, s.size = sh.hashLen) :
    Reads (Reader.vecExact want (Reader.bytes sh.hashLen) what) (wVec id xs) xs :=
  Reads.vecExact xs what hw (fun x h => by simp only [id, hx x h]; omega) fun x h => Reads.bytes x (hx x h)

theorem wVec_size_ge {α : Type} (f : α → ByteArray) (xs : Array α) : 8 ≤ (wVec f xs).size := by
  rw [wVec_eq, ByteArray.size_append, wLen, le64_size]
  omega

theorem Reads.rows (rows : Array (Array F128)) {q w : ℕ} (hq : rows.size = q) (hw : ∀ r ∈ rows, r.size = w) :
    Reads (Reader.vecExact q (Reader.vecExact w Reader.f128 "row") "opened_rows") (wVec (wVec F128.toBytes) rows) rows :=
  Reads.vecExact rows _ hq (fun x _ => le_trans (by omega) (wVec_size_ge _ x)) fun x h => Reads.f128s x _ (hw x h)


theorem Reads.salts (sh : Shape) (q : ℕ) (xs : Array ByteArray) (h : SaltsOk sh q xs) :
    Reads (match sh.saltLen with
      | none => (Pure.pure #[] : M (Array ByteArray))
      | some 0 => Reader.vecExact 0 (Reader.bytes 0) "opened_salts"
      | some n => Reader.vecExact q (Reader.bytes n) "opened_salts") (wSalts sh xs) xs := by
  unfold SaltsOk at h
  unfold wSalts
  rcases hs : sh.saltLen with _ | _ | n
  · simp only [hs] at h ⊢
    subst h
    exact Reads.pure _
  · simp only [hs] at h ⊢
    subst h
    exact Reads.congr (Reads.vec0 _ _) (by simp [wVec]) rfl
  · simp only [hs] at h ⊢
    exact Reads.vecExact xs _ h.1 (fun x hx => by simp [h.2 x hx]) fun x hx => Reads.bytes x (h.2 x hx)

theorem Reads.last {α β : Type} {m : M α} {g : α → β} {b : ByteArray} {x : α} {y : β} (h : Reads m b x)
    (hy : g x = y) : Reads (m >>= fun v => Pure.pure (g v)) b y :=
  Reads.congr (Reads.bind h (Reads.pure (g x))) ByteArray.append_empty hy

theorem Reads.inner (q : Zk.InnerProof) (hs : ∀ x ∈ q.salts, x.size = 192) (hp : ∀ l ∈ q.paths, ∀ x ∈ l, x.size = 64) :
    Reads Zk.innerReader (wInner q) q := by
  unfold Zk.innerReader
  simp only [wInner, ByteArray.append_assoc]
  refine Reads.bind (Reads.byteVec q.root) ?_
  refine Reads.bind (Reads.f128 q.rho) ?_
  refine Reads.bind (Reads.f128 q.sigma) ?_
  refine Reads.bind (Reads.byteVec q.tauCommit) ?_
  refine Reads.bind (Reads.byteVec q.tauSalt) ?_
  refine Reads.bind (Reads.f128 q.tau) ?_
  refine Reads.bind (Reads.vecAny q.y _ (by omega) (fun x _ => by rw [toBytes_size]) fun x _ => Reads.f128 x) ?_
  refine Reads.bind (Reads.vecAny q.opened _ (by omega)
    (fun x _ => by rw [wPair, ByteArray.size_append, toBytes_size, toBytes_size]) fun x _ => Reads.pair x) ?_
  refine Reads.bind (Reads.vecAny (f := id) q.salts _ (by omega) (fun x h => by simp [hs x h])
    fun x h => Reads.bytes x (hs x h)) ?_
  exact Reads.last (Reads.vecAny (f := wVec id) q.paths _ (by omega) (fun x _ => wVec_size_ge _ _)
    fun l h => Reads.vecAny (f := id) l _ (by omega) (fun x hx => by simp [hp l h x hx])
      fun x hx => Reads.bytes x (hp l h x hx)) rfl

theorem Reads.ringSwitch (v : Array F128) (hv : v.size = 128) :
    Reads (do
      let v ← Reader.vecExact 128 Reader.f128 "s_hat_v"
      let nonce ← Reader.u64
      if nonce != 0 then throw "ring switch grinding_nonce ≠ 0"
      Pure.pure v) (wVec F128.toBytes v ++ wLen 0) v := by
  refine Reads.bind (Reads.f128s v _ hv) ?_
  refine Reads.congr (Reads.bind (Reads.u64 _) ?_) ByteArray.append_empty rfl
  simp only [show ((0 : ℕ).toUInt64 != 0) = false from rfl, Bool.false_eq_true, ite_false]
  exact Reads.pure v

theorem fit_left {a b : ByteArray} (h : (a ++ b).size < 2 ^ 64) : a.size < 2 ^ 64 := by
  rw [ByteArray.size_append] at h
  omega

/-- `Zk.reader sh` reads the file `encZk sh p` back as `p`, wherever it sits in an input that fits a `u64`. -/
theorem reads_reader (sh : Shape) (p : Zk.ZkProof) (hp : Shaped sh p) : Reads (Zk.reader sh) (encZk sh p) p := by
  obtain ⟨⟨cap, ⟨m, lir, lbs, prof, nl, mh⟩, ⟨r1ab, r1c, rounds, fa, fb, fc⟩, ⟨lr, zp⟩, rs,
    ⟨icap, ⟨rows0, sib0, salt0⟩, rcaps, rps, ⟨yr, rowsR, sibR, saltR⟩, msgs, qn, ood, cbn, con⟩⟩,
    ⟨e0, e1, tP, ybar⟩, inner⟩ := p
  obtain ⟨hL, hR, -, hcap, h1ab, h1c, hrounds, hlr, hzp, hrs, hrs128, hicap, hl0, hrcs, hrc, hrps, hrp, hyr, hfin,
    hmsg, hqn, hood, hcbn, hcon, hsalts, hpaths⟩ := hp
  dsimp only at *
  unfold Zk.reader
  simp only [encZk, wProof, wParams, wLigerito, wLevel, wLevel0, ByteArray.append_assoc]
  rw [wVec_eq (wVec id) rcaps, wVec_eq (wLevel sh) rps]
  simp only [ByteArray.append_assoc]
  refine Reads.bind (Reads.hashes sh hL cap _ hcap.1 hcap.2) ?_
  refine Reads.bind (Reads.u64 m) ?_
  refine Reads.bind (Reads.u64 lir) ?_
  refine Reads.bind (Reads.u64 lbs) ?_
  refine Reads.bind (Reads.u32 prof) ?_
  rcases nl with _ | nl
  refine Reads.bind (Reads.u8 0) ?_
  simp only [pure_bind]
  swap
  simp only [ByteArray.append_assoc]
  refine Reads.bind (Reads.u8 1) ?_
  simp only []
  refine Reads.bind (Reads.map some (Reads.u64 nl)) ?_
  all_goals
    refine Reads.bind (Reads.u32 mh) ?_
    refine Reads.bind (Reads.f128s r1ab _ h1ab) ?_
    refine Reads.bind (Reads.f128s r1c _ h1c) ?_
    refine Reads.bind (Reads.vecExact rounds _ hrounds
      (fun x _ => by rw [wPair, ByteArray.size_append, toBytes_size, toBytes_size]; omega) fun x _ => Reads.pair x) ?_
    refine Reads.bind (Reads.f128 fa) ?_
    refine Reads.bind (Reads.f128 fb) ?_
    refine Reads.bind (Reads.f128 fc) ?_
    refine Reads.bind (Reads.vec0 _ _) ?_
    refine Reads.bind (Reads.vecExact lr _ hlr
      (fun x _ => by rw [wPair, ByteArray.size_append, toBytes_size, toBytes_size]; omega) fun x _ => Reads.pair x) ?_
    refine Reads.bind (Reads.f128s zp _ hzp) ?_
    refine Reads.bind (Reads.vec0 _ _) ?_
    refine Reads.bind (Reads.vec0 _ _) ?_
    refine Reads.bind (Reads.vecExact rs _ hrs
      (fun v _ => by have := wVec_size_ge F128.toBytes v; rw [ByteArray.size_append]; omega)
      fun v hv => Reads.ringSwitch v (hrs128 v hv)) ?_
    refine Reads.bind (Reads.vec0 _ _) ?_
    refine Reads.bind (Reads.hashes sh hL icap _ hicap.1 hicap.2) ?_
    refine Reads.bind (Reads.rows rows0 hl0.1 hl0.2.1) ?_
    refine Reads.bind (Reads.hashes sh hL sib0 _ hl0.2.2.1 hl0.2.2.2.1) ?_
    refine Reads.bind (Reads.salts sh _ salt0 hl0.2.2.2.2) ?_
    refine Reads.bind (Reads.u64 _) (Reads.of_fit fun hb => ?_)
    have hrc64 : rcaps.size < 2 ^ 64 := by
      have h1 := fit_left hb
      have := catL_size_ge (wVec id) 8 rcaps.toList fun x _ => wVec_size_ge id x
      rw [Array.length_toList] at this
      omega
    have hc : (rcaps.size != sh.sched.levels.size - 1) = false := by simp [hrcs]
    simp only [toNat_toUInt64_of_lt hrc64, hc, Bool.false_eq_true, ite_false]
    refine Reads.bind (x := rcaps) ?_ ?_
    · exact Reads.congr (Reads.forIn_range _ (wVec id) rcaps _ hrcs #[] fun i hi acc =>
        Reads.congr (Reads.bind (Reads.hashes sh hL _ _ (hrc i hi).1 (hrc i hi).2) (Reads.pure _))
          ByteArray.append_empty rfl) rfl Array.empty_append
    refine Reads.bind (Reads.u64 _) (Reads.of_fit fun hb => ?_)
    have hrp64 : rps.size < 2 ^ 64 := by
      have := catL_size_ge (wLevel sh) 8 rps.toList fun x _ => by
        have := wVec_size_ge (wVec F128.toBytes) x.rows
        simp only [wLevel, ByteArray.size_append]
        omega
      have h1 := fit_left hb
      rw [Array.length_toList] at this
      omega
    have hnp : (rps.size + 1 != sh.sched.levels.size - 1) = false := by simp; omega
    simp only [toNat_toUInt64_of_lt hrp64, hnp, Bool.false_eq_true, ite_false]
    refine Reads.bind (x := rps) ?_ ?_
    · exact Reads.congr (Reads.forIn_range _ (wLevel sh) rps _ hrps #[] fun i hi acc =>
        Reads.congr (Reads.bind (Reads.rows _ (hrp i hi).1 (hrp i hi).2.1)
          (Reads.bind (Reads.hashes sh hL _ _ (hrp i hi).2.2.1 (hrp i hi).2.2.2.1)
            (Reads.bind (Reads.salts sh _ _ (hrp i hi).2.2.2.2) (Reads.pure _))))
          (by simp only [wLevel, ByteArray.append_assoc, ByteArray.append_empty]) rfl)
        rfl Array.empty_append
    refine Reads.bind (Reads.f128s yr _ hyr) ?_
    refine Reads.bind (Reads.rows rowsR hfin.1 hfin.2.1) ?_
    refine Reads.bind (Reads.hashes sh hL sibR _ hfin.2.2.1 hfin.2.2.2.1) ?_
    refine Reads.bind (Reads.salts sh _ saltR hfin.2.2.2.2) ?_
    refine Reads.bind (Reads.vec0 _ _) ?_
    refine Reads.bind (Reads.vecExact msgs _ hmsg
      (fun x _ => by simp only [wF256, ByteArray.size_append, toBytes_size]; omega)
      fun x _ => Reads.pair256 x) ?_
    refine Reads.bind (Reads.u64s qn _ hqn) ?_
    refine Reads.bind (Reads.f128s ood _ hood) ?_
    refine Reads.bind (Reads.vec0 _ _) ?_
    refine Reads.bind (Reads.u64s cbn _ hcbn) ?_
    refine Reads.bind (Reads.u64s con _ hcon) ?_
    refine Reads.bind (Reads.u8 1) ?_
    simp only [pure_bind]
    refine Reads.bind (Reads.f256 e0) ?_
    refine Reads.bind (Reads.f256 e1) ?_
    refine Reads.bind (Reads.f256 tP) ?_
    refine Reads.bind (Reads.vecAny ybar _ (by omega)
      (fun x _ => by simp only [wF256, ByteArray.size_append, toBytes_size]; omega)
      fun x _ => Reads.f256 x) ?_
    exact Reads.last (Reads.inner inner hsalts hpaths) rfl

theorem decode_encode : DecodeEncode := by
  intro sh p hp
  have h := reads_reader sh p hp ByteArray.empty ByteArray.empty (by
    rw [ByteArray.empty_append, ByteArray.append_empty]
    exact hp.2.2.1)
  rw [ByteArray.empty_append, ByteArray.append_empty] at h
  unfold Zk.decode
  rw [show (⟨encZk sh p, 0⟩ : Reader) = ⟨encZk sh p, ByteArray.empty.size⟩ from rfl, h]
  rfl

end FlockSoundness.Discharge.ZkView
