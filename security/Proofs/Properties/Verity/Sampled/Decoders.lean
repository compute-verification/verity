module

public import Proofs.Properties.Verity.Sampled.Assembled
public import Proofs.Flock.Soundness.Discharge.ZkExec.RunsFrames
public import Proofs.Flock.Soundness.Discharge.ZkView.Decode
public import Proofs.Flock.Soundness.Discharge.HiddenExec.AcceptsP
public import Proofs.Flock.Soundness.Discharge.Hidden.SpecP
public import Proofs.Flock.Soundness.Refine.Table
public import Proofs.Flock.Soundness.Refine.Regions
public import Proofs.Flock.Soundness.Refine.Paths
import all Lean.Data.Json.Basic

/-! SPOT's coupling: the decoders and the field encoding. -/

@[expose] public section

namespace Verity.Sampled

open Lean (Json)
open Flock (SessionSpec RunRecord StreamRec Round)
open FlockSoundness.Discharge.Tables (SubmitJ)
open FlockComplete.JsonRT (be8 be8_fold fold_get shift_or)
open FlockSoundness.Refine (ebind_ok check_ok_iff mapM_ok_mem mapIdxM_ok_mem ptsOf)
open Claims.Bridge.Custody (decode_ok DecStream RoundFields PointField forall₂_find? forall₂_getElem forall₂_mem_left)
open FlockSoundness.Discharge.ZkHidden (TagsOkZHJ)

local notation "𝔽" => FlockLevel3.GF128

/-- The word whose bytes, most significant first, are bytes `o` to `o + 7` of `b`, as `f128OfHex` reads it. -/
def beWord (b : ByteArray) (o : ℕ) : UInt64 :=
  ((List.range' 0 8).map fun i => b.get! (o + i)).foldl (fun (s : UInt64) c => s <<< 8 ||| c.toUInt64) 0

/-- **The challenge 16 bytes are** as the record writes it: the high word's bytes, then the low word's, each most
significant first (`f128OfHex`'s reading). -/
def ofBytes16 (b : ByteArray) : Flock.F128 := ⟨beWord b 8, beWord b 0⟩

/-- **A challenge's 16 bytes** as the record writes them. -/
def f128Bytes (x : Flock.F128) : ByteArray := ⟨(be8 x.hi ++ be8 x.lo).toArray⟩

theorem ofBytes16_f128Bytes (x : Flock.F128) : ofBytes16 (f128Bytes x) = x := by
  have h0 : beWord (f128Bytes x) 0 = x.hi := by
    unfold beWord
    rw [show (List.range' 0 8).map (fun i => (f128Bytes x).get! (0 + i)) = be8 x.hi from rfl, be8_fold]
  have h8 : beWord (f128Bytes x) 8 = x.lo := by
    unfold beWord
    rw [show (List.range' 0 8).map (fun i => (f128Bytes x).get! (8 + i)) = be8 x.lo from rfl, be8_fold]
  unfold ofBytes16
  rw [h0, h8]

theorem beFold_toNat (a0 a1 a2 a3 a4 a5 a6 a7 : UInt8) :
    ([a0, a1, a2, a3, a4, a5, a6, a7].foldl (fun (s : UInt64) c => s <<< 8 ||| c.toUInt64) 0).toNat =
      a0.toNat * 2 ^ 56 + a1.toNat * 2 ^ 48 + a2.toNat * 2 ^ 40 + a3.toNat * 2 ^ 32 + a4.toNat * 2 ^ 24 +
        a5.toNat * 2 ^ 16 + a6.toNat * 2 ^ 8 + a7.toNat := by
  have h0 := a0.toNat_lt
  have h1 := a1.toNat_lt
  have h2 := a2.toNat_lt
  have h3 := a3.toNat_lt
  have h4 := a4.toNat_lt
  have h5 := a5.toNat_lt
  have h6 := a6.toNat_lt
  have h7 := a7.toNat_lt
  simp only [List.foldl_cons, List.foldl_nil, shift_or]
  simp
  omega

theorem be8_beFold (a0 a1 a2 a3 a4 a5 a6 a7 : UInt8) :
    be8 ([a0, a1, a2, a3, a4, a5, a6, a7].foldl (fun (s : UInt64) c => s <<< 8 ||| c.toUInt64) 0) =
      [a0, a1, a2, a3, a4, a5, a6, a7] := by
  generalize hx : [a0, a1, a2, a3, a4, a5, a6, a7].foldl (fun (s : UInt64) c => s <<< 8 ||| c.toUInt64) 0 = x
  have hn := beFold_toNat a0 a1 a2 a3 a4 a5 a6 a7
  rw [hx] at hn
  have h0 := a0.toNat_lt
  have h1 := a1.toNat_lt
  have h2 := a2.toNat_lt
  have h3 := a3.toNat_lt
  have h4 := a4.toNat_lt
  have h5 := a5.toNat_lt
  have h6 := a6.toNat_lt
  have h7 := a7.toNat_lt
  simp only [be8, List.cons.injEq, and_true]
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩ <;> apply UInt8.toNat_inj.1 <;>
    simp only [UInt64.toNat_toUInt8, UInt64.toNat_shiftRight, Nat.shiftRight_eq_div_pow, hn] <;> simp <;> omega

theorem map_range_get! (b : ByteArray) : (List.range b.size).map b.get! = b.data.toList := by
  apply List.ext_getElem (by rw [List.length_map, List.length_range, Array.length_toList, ByteArray.size_data])
  intro i h1 h2
  have hi : i < b.data.size := by
    rw [List.length_map, List.length_range] at h1
    rwa [ByteArray.size_data]
  simp only [List.getElem_map, List.getElem_range, Array.getElem_toList]
  simp only [ByteArray.get!, getElem!_pos b.data i hi]

theorem f128Bytes_ofBytes16 {b : ByteArray} (h : b.size = 16) : f128Bytes (ofBytes16 b) = b := by
  have e0 : beWord b 0 = [b.get! 0, b.get! 1, b.get! 2, b.get! 3, b.get! 4, b.get! 5, b.get! 6, b.get! 7].foldl
      (fun (s : UInt64) c => s <<< 8 ||| c.toUInt64) 0 := rfl
  have e8 : beWord b 8 = [b.get! 8, b.get! 9, b.get! 10, b.get! 11, b.get! 12, b.get! 13, b.get! 14,
      b.get! 15].foldl (fun (s : UInt64) c => s <<< 8 ||| c.toUInt64) 0 := rfl
  have hl := map_range_get! b
  rw [h] at hl
  unfold f128Bytes ofBytes16
  dsimp only
  rw [e0, e8, be8_beFold, be8_beFold]
  apply ByteArray.ext
  show List.toArray _ = b.data
  rw [← Array.toArray_toList (xs := b.data), ← hl]
  rfl

theorem f128OfHex_ok {s : String} {x : Flock.F128} (h : Flock.f128OfHex s = .ok x) :
    ∃ u, Flock.unhex s = .ok u ∧ u.size = 16 ∧ x = ofBytes16 u := by
  unfold Flock.f128OfHex at h
  cases hu : Flock.unhex s with
  | error e => rw [hu] at h; cases h
  | ok u =>
    rw [hu] at h
    simp only [bind, Except.bind] at h
    by_cases hs : u.size = 16
    · have h16 : (u.size != 16) = false := by simp [hs]
      rw [h16] at h
      simp only [Bool.false_eq_true, ite_false, Std.Legacy.Range.forIn_eq_forIn_range',
        List.forIn_pure_yield_eq_foldl] at h
      rw [fold_get u 0 _ rfl, fold_get u 8 _ rfl] at h
      exact ⟨u, rfl, hs, (Except.ok.inj h).symm⟩
    · have h16 : (u.size != 16) = true := by simp [hs]
      rw [h16] at h
      simp only [ite_true] at h
      cases h

theorem f128Bytes_size (x : Flock.F128) : (f128Bytes x).size = 16 := rfl

theorem mapM_toList_ok {α β ε : Type} {g : α → Except ε β} {xs : Array α} {ys : Array β}
    (h : xs.mapM g = .ok ys) : xs.toList.mapM g = .ok ys.toList := by
  rw [Array.mapM_eq_mapM_toList] at h
  cases hl : xs.toList.mapM g with
  | error e => rw [hl] at h; cases h
  | ok l =>
    rw [hl] at h
    simp only [Functor.map, Except.map, Except.ok.injEq] at h
    subst h
    simp

theorem foldlM_coins {f : ByteArray → Json → Option ByteArray} {g : Json → Flock.V Flock.F128}
    (hfg : ∀ acc x c b, f acc x = some b → g x = .ok c → b = acc ++ f128Bytes c) :
    ∀ (l : List Json) (acc b : ByteArray) (coins : List Flock.F128), l.foldlM f acc = some b →
      l.mapM g = .ok coins → b.data.toList = acc.data.toList ++ (coins.map fun x => (f128Bytes x).data.toList).flatten
  | [], acc, b, coins, hf, hg => by
    rw [List.foldlM_nil] at hf
    rw [List.mapM_nil] at hg
    cases hf
    cases hg
    simp
  | x :: l, acc, b, coins, hf, hg => by
    rw [List.foldlM_cons] at hf
    rw [List.mapM_cons] at hg
    cases ha : f acc x with
    | none => rw [ha] at hf; cases hf
    | some a =>
      rw [ha] at hf
      obtain ⟨c, hc, hg⟩ := ebind_ok hg
      obtain ⟨cs, hcs, hg⟩ := ebind_ok hg
      cases hg
      rw [foldlM_coins hfg l a b cs hf hcs, hfg acc x c a ha hc, ByteArray.data_append, Array.toList_append]
      simp

theorem recordCoins_bytes {cs : Array Json} {coins : Array Flock.F128} {b : ByteArray}
    (hf : Flock.Record.f128s cs = .ok coins) (hb : _root_.Spot.Challenger.recordCoins? (.arr cs) = some b) :
    b.data.toList = (coins.toList.map fun x => (f128Bytes x).data.toList).flatten := by
  simp only [_root_.Spot.Challenger.recordCoins?] at hb
  rw [← Array.foldlM_toList] at hb
  have h := foldlM_coins (fun acc x c b hf hg => ?_) cs.toList ByteArray.empty b coins.toList hb
    (mapM_toList_ok hf)
  · simpa using h
  · cases x with
    | str s =>
      have hg' : Flock.f128OfHex s = .ok c := hg
      obtain ⟨u, hu, hs, rfl⟩ := f128OfHex_ok hg'
      have hf' : (if s.length == 32 then (Flock.unhex s).toOption.map (acc ++ ·) else none) = some b := hf
      rw [hu, f128Bytes_ofBytes16 hs] at *
      split at hf'
      · simp only [Except.toOption, Option.map_some, Option.some.injEq] at hf'
        exact hf'.symm
      · cases hf'
    | _ => cases hf

/-- **A challenge's bits** under the record's encoding: bit `t % 8` of byte `t / 8` of its 16 bytes as the record writes
them, as the device's log reads a round's coins (`Verity.bitsOfBytes`). -/
def fieldBits (x : Flock.F128) : Fin 128 → Bool := fun t => Flock.Hm96.bit (f128Bytes x) t.val

theorem ofFn_fieldBits (x : Flock.F128) : List.ofFn (fieldBits x) = Verity.bitsOfBytes (f128Bytes x) := by
  unfold Verity.bitsOfBytes
  apply List.ext_getElem (by rw [List.length_ofFn, List.length_map, List.length_range, f128Bytes_size])
  intro i h1 h2
  rw [List.getElem_ofFn, List.getElem_map, List.getElem_range]
  rfl

/-- The challenge whose bits are `bs`. -/
def fieldOfBits (bs : Fin 128 → Bool) : Flock.F128 := ofBytes16 (Lines.coinBytes (List.ofFn bs))

theorem fieldOfBits_fieldBits (x : Flock.F128) : fieldOfBits (fieldBits x) = x := by
  rw [fieldOfBits, ofFn_fieldBits, Lines.coinBytes_bits, ofBytes16_f128Bytes]

theorem fieldBits_fieldOfBits (bs : Fin 128 → Bool) : fieldBits (fieldOfBits bs) = bs := by
  have hs : (Lines.coinBytes (List.ofFn bs)).size = 16 := by rw [Lines.coinBytes_size, List.length_ofFn]
  have h8 : (List.ofFn bs).length % 8 = 0 := by rw [List.length_ofFn]
  apply List.ofFn_injective
  rw [ofFn_fieldBits, fieldOfBits, f128Bytes_ofBytes16 hs, Lines.bits_coinBytes h8]

theorem fieldBits_injective : Function.Injective fieldBits := fun x y h => by
  rw [← fieldOfBits_fieldBits x, h, fieldOfBits_fieldBits]

/-- **The field encoding**: a challenge's bits under the record's encoding (`fieldBits`). -/
def fieldEnc : 𝔽 ≃ (Fin 128 → Bool) where
  toFun x := fieldBits x.val
  invFun bs := FlockLevel3.GF128.of (fieldOfBits bs)
  left_inv x := fieldOfBits_fieldBits x.val
  right_inv bs := fieldBits_fieldOfBits bs

theorem fieldEnc_apply (x : 𝔽) : fieldEnc x = fieldBits x.val := rfl

theorem bitsOfBytes_append (l₁ l₂ : List UInt8) :
    Verity.bitsOfBytes ⟨⟨l₁ ++ l₂⟩⟩ = Verity.bitsOfBytes ⟨⟨l₁⟩⟩ ++ Verity.bitsOfBytes ⟨⟨l₂⟩⟩ := by
  rw [bitsOfBytes_list, bitsOfBytes_list, bitsOfBytes_list, List.flatMap_append]

theorem bitsOfBytes_flatten :
    ∀ l : List Flock.F128, Verity.bitsOfBytes ⟨⟨(l.map fun x => (f128Bytes x).data.toList).flatten⟩⟩ =
      (l.map fun x => List.ofFn (fieldBits x)).flatten
  | [] => rfl
  | x :: l => by
    simp only [List.map_cons, List.flatten_cons]
    rw [bitsOfBytes_append, bitsOfBytes_flatten l, ofFn_fieldBits]

theorem bits_of_coins {cs : Array Json} {coins : Array Flock.F128} {b : ByteArray}
    (hf : Flock.Record.f128s cs = .ok coins) (hb : _root_.Spot.Challenger.recordCoins? (.arr cs) = some b) :
    Verity.bitsOfBytes b = (coins.toList.map fun x => List.ofFn (fieldBits x)).flatten :=
  (congrArg (fun l => Verity.bitsOfBytes ⟨⟨l⟩⟩) (recordCoins_bytes hf hb)).trans (bitsOfBytes_flatten coins.toList)

theorem length_chunks {β : Type} (enc : β → Fin 128 → Bool) :
    ∀ l : List β, ((l.map fun x => List.ofFn (enc x)).flatten).length = 128 * l.length
  | [] => by simp only [List.map_nil, List.flatten_nil, List.length_nil, Nat.mul_zero]
  | x :: l => by
    rw [List.map_cons, List.flatten_cons, List.length_append, List.length_ofFn, length_chunks enc l,
      List.length_cons]
    omega

theorem flatten_inj_chunks {α : Type} {k : ℕ} :
    ∀ {A B : List (List α)}, (∀ x ∈ A, x.length = k) → (∀ y ∈ B, y.length = k) → 0 < k →
      A.flatten = B.flatten → A = B
  | [], [], _, _, _, _ => rfl
  | [], y :: B, _, hB, hk, h => by
    have := congrArg List.length h
    rw [List.flatten_nil, List.flatten_cons, List.length_append, List.length_nil,
      hB y (List.mem_cons.2 (Or.inl rfl))] at this
    omega
  | x :: A, [], hA, _, hk, h => by
    have := congrArg List.length h
    rw [List.flatten_nil, List.flatten_cons, List.length_append, List.length_nil,
      hA x (List.mem_cons.2 (Or.inl rfl))] at this
    omega
  | x :: A, y :: B, hA, hB, hk, h => by
    rw [List.flatten_cons, List.flatten_cons] at h
    obtain ⟨hxy, hr⟩ := List.append_inj h
      (by rw [hA x (List.mem_cons.2 (Or.inl rfl)), hB y (List.mem_cons.2 (Or.inl rfl))])
    rw [hxy, flatten_inj_chunks (fun x hx => hA x (List.mem_cons.2 (Or.inr hx)))
      (fun y hy => hB y (List.mem_cons.2 (Or.inr hy))) hk hr]

theorem eq_ofFn_of_chunks {β : Type} {enc : β → Fin 128 → Bool} (henc : Function.Injective enc) {n : ℕ}
    {coins : List β} {c : Fin n → β}
    (h : (coins.map fun x => List.ofFn (enc x)).flatten = (List.ofFn fun i => List.ofFn (enc (c i))).flatten) :
    coins = List.ofFn c := by
  have hA := flatten_inj_chunks (k := 128)
    (fun x hx => by obtain ⟨a, -, rfl⟩ := List.mem_map.1 hx; exact List.length_ofFn)
    (fun y hy => by obtain ⟨i, rfl⟩ := List.mem_ofFn.1 hy; exact List.length_ofFn) (by decide) h
  rw [show (List.ofFn fun i => List.ofFn (enc (c i))) = (List.ofFn c).map fun x => List.ofFn (enc x) by
    rw [List.map_ofFn]; rfl] at hA
  exact (List.map_inj_right fun a b hab => henc (List.ofFn_injective hab)).1 hA

theorem roundBits_eq {n : ℕ} (c : Fin n → 𝔽) :
    roundBits fieldEnc c = ((List.ofFn fun i => (c i).val).map fun x => List.ofFn (fieldBits x)).flatten := by
  simp only [roundBits, List.map_ofFn, Function.comp_def, fieldEnc_apply]

theorem length_roundBits {n : ℕ} (c : Fin n → 𝔽) : (roundBits fieldEnc c).length = 128 * n := by
  rw [roundBits_eq, length_chunks, List.length_ofFn]

theorem coins_of_flatten {n : ℕ} {coins : Array Flock.F128} {c : Fin n → 𝔽}
    (h : (coins.toList.map fun x => List.ofFn (fieldBits x)).flatten = roundBits fieldEnc c) :
    coins.size = n ∧ ∀ i : Fin n, (c i).val = coins[i.val]! := by
  have hl : coins.toList = List.ofFn fun i => (c i).val :=
    eq_ofFn_of_chunks (enc := fieldBits) (c := fun i => (c i).val) fieldBits_injective h
  have hsz : coins.size = n := by
    have := congrArg List.length hl
    rwa [Array.length_toList, List.length_ofFn] at this
  refine ⟨hsz, fun i => ?_⟩
  have hi : i.val < coins.toList.length := by rw [Array.length_toList, hsz]; exact i.isLt
  rw [getElem!_pos coins i.val (hsz ▸ i.isLt), ← Array.getElem_toList hi, List.getElem_of_eq hl hi,
    List.getElem_ofFn]

theorem coins_of_bits {cs : Array Json} {coins : Array Flock.F128} {b : ByteArray} {n : ℕ} {c : Fin n → 𝔽}
    (hf : Flock.Record.f128s cs = .ok coins) (hb : _root_.Spot.Challenger.recordCoins? (.arr cs) = some b)
    (hbits : Verity.bitsOfBytes b = roundBits fieldEnc c) :
    coins.size = n ∧ ∀ i : Fin n, (c i).val = coins[i.val]! :=
  coins_of_flatten ((bits_of_coins hf hb).symm.trans hbits)

theorem pts_of_bits {mP : ℕ} {points : Array (Array Flock.F128)} {P : Fin 2 → Fin mP → 𝔽}
    (hsz : points[0]!.size = points[1]!.size)
    (h : ((points[0]! ++ points[1]!).toList.map fun x => List.ofFn (fieldBits x)).flatten = linkBits fieldEnc P) :
    ptsOf mP points = P := by
  rw [Array.toList_append, List.map_append, List.flatten_append, linkBits] at h
  have hl := congrArg List.length h
  rw [List.length_append, List.length_append, length_chunks, length_chunks, length_roundBits, length_roundBits,
    Array.length_toList, Array.length_toList] at hl
  obtain ⟨h0, h1⟩ := List.append_inj h (by rw [length_chunks, length_roundBits, Array.length_toList]; omega)
  have e0 := coins_of_flatten h0
  have e1 := coins_of_flatten h1
  funext k t
  obtain ⟨_ | _ | k, hk⟩ := k
  · exact (congrArg FlockLevel3.GF128.of (e0.2 t)).symm
  · exact (congrArg FlockLevel3.GF128.of (e1.2 t)).symm
  · omega

theorem fld_getD {x v : Json} {k : String} (h : Flock.Record.fld x k = .ok v) :
    _root_.Spot.getD x k = v := by
  unfold Flock.Record.fld at h
  show ((x.getObjVal? k).toOption).getD Json.null = v
  cases hx : x.getObjVal? k with
  | error e => rw [hx] at h; cases h
  | ok w => rw [hx] at h; exact Except.ok.inj h

theorem arr_getD {x : Json} {k : String} {a : Array Json} (h : Flock.Record.arr x k = .ok a) :
    _root_.Spot.getD x k = .arr a := by
  unfold Flock.Record.arr at h
  obtain ⟨v, hv, h⟩ := ebind_ok h
  rw [fld_getD hv]
  cases v with
  | arr a' => exact congrArg Json.arr (Except.ok.inj h)
  | _ => cases h

theorem str_getD {x : Json} {k s : String} (h : Flock.Record.str x k = .ok s) :
    _root_.Spot.getD x k = .str s := by
  unfold Flock.Record.str at h
  obtain ⟨v, hv, h⟩ := ebind_ok h
  rw [fld_getD hv]
  cases v with
  | str s' => exact congrArg Json.str (Except.ok.inj h)
  | _ => cases h

theorem eq_arr_of_getArr {p : Json} {a : Array Json} (h : p.getArr? = .ok a) : p = .arr a := by
  cases p with
  | arr a' => exact congrArg Json.arr (Except.ok.inj h)
  | _ => cases h

theorem list_find?_congr {α : Type} {q q' : α → Bool} :
    ∀ {l : List α}, (∀ a ∈ l, q a = q' a) → l.find? q = l.find? q'
  | [], _ => rfl
  | a :: l, h => by
    rw [List.find?_cons, List.find?_cons, h a (List.mem_cons.2 (Or.inl rfl)),
      list_find?_congr fun b hb => h b (List.mem_cons.2 (Or.inr hb))]

theorem find?_congr {α : Type} {xs : Array α} {q q' : α → Bool} (h : ∀ a ∈ xs, q a = q' a) :
    xs.find? q = xs.find? q' := by
  rw [← Array.find?_toList, ← Array.find?_toList]
  exact list_find?_congr fun a ha => h a (Array.mem_toList_iff.1 ha)

theorem repStreamName_ne (t : String) (r : ℕ) :
    (repStreamName t r == _root_.Spot.Challenger.TREE_SESSION) = false := by
  refine beq_false_of_ne fun h => ?_
  have hm : '/' ∈ (repStreamName t r).toList := by
    unfold repStreamName
    simp only [String.toList_append, List.append_assoc, List.mem_append]
    exact Or.inr (Or.inl (by decide))
  rw [h] at hm
  exact absurd hm (by decide)

theorem f128s_append {a b : Array Json} {A B : Array Flock.F128} (ha : Flock.Record.f128s a = .ok A)
    (hb : Flock.Record.f128s b = .ok B) : Flock.Record.f128s (a ++ b) = .ok (A ++ B) := by
  unfold Flock.Record.f128s at ha hb ⊢
  rw [Array.mapM_append, ha, hb]
  rfl

theorem recordRound_decode {spec : SessionSpec} {record : String} {sess : RunRecord} {J : Json} {name : String}
    {y : StreamRec} {k : ℕ} (hd : Flock.Record.decode spec record = .ok sess)
    (hJ : Flock.JsonParse.parse record = .ok J)
    (hn : (name == _root_.Spot.Challenger.TREE_SESSION) = false)
    (hy : sess.streams.find? (·.name == name) = some y) (hk : k < y.rounds.size) :
    ∃ cs, recordRound J name k = _root_.Spot.Challenger.recordCoins? (.arr cs) ∧
      Flock.Record.f128s cs = .ok y.rounds[k].coins ∧ ∀ b, y.rounds[k].bytes = some b → y.rounds[k].sha = spec.dg b := by
  obtain ⟨j, l, recs, hp, -, hrecs, -, -, hstr⟩ := decode_ok hd
  rw [hJ] at hp
  obtain rfl := Except.ok.inj hp
  have hy' : sess.streams.toList.find? (·.name == name) = some y := by rw [Array.find?_toList]; exact hy
  obtain ⟨x, ⟨-, rs, hrs, hrds⟩, hx⟩ := forall₂_find?
    (R := fun y x => DecStream spec x y) (p := fun y => y.name == name)
    (q := fun x => (Flock.Record.str x "stream").toOption == some name)
    (fun a b hab => by rw [hab.1]; rfl) (forall₂_swap hstr) hy'
  have hlen : rs.size = y.rounds.size := by simpa using hrds.length_eq
  have hrk : k < rs.size := hlen ▸ hk
  have hfield := forall₂_getElem hrds hrk
  rw [getElem!_pos y.rounds k hk] at hfield
  obtain ⟨⟨-, ⟨cs, hcs, hf⟩, -, -⟩, hsha⟩ := hfield
  have hx' : recs.find? (fun x => (Flock.Record.str x "stream").toOption == some name) = some x := by
    rw [← Array.find?_toList]; exact hx
  refine ⟨cs, ?_, hf, hsha⟩
  simp only [recordRound, hn, Bool.false_and, Bool.false_eq_true, ↓reduceIte, arr_getD hrecs]
  rw [find?_congr (q' := fun x => (Flock.Record.str x "stream").toOption == some name) ?_, hx']
  · simp only [Option.bind_some, arr_getD hrs, Array.getElem?_eq_getElem hrk, arr_getD hcs]
  · intro a ha
    obtain ⟨y', -, hy''⟩ := forall₂_mem_left hstr a (Array.mem_toList_iff.2 ha)
    rw [str_getD hy''.1, hy''.1]
    rfl

theorem link_coins {spec : SessionSpec} {record : String} {sess : RunRecord} {J : Json}
    (hd : Flock.Record.decode spec record = .ok sess) (hJ : Flock.JsonParse.parse record = .ok J)
    (hp2 : spec.points = 2) :
    ∃ cs, recordRound J _root_.Spot.Challenger.TREE_SESSION 0 =
        _root_.Spot.Challenger.recordCoins? (.arr cs) ∧
      Flock.Record.f128s cs = .ok (sess.points[0]! ++ sess.points[1]!) := by
  obtain ⟨j, l, recs, hp, hl, -, -, ⟨ps, hps, hpts⟩, -⟩ := decode_ok hd
  rw [hJ] at hp
  obtain rfl := Except.ok.inj hp
  have hsz : sess.points.toList.length = 2 := by rw [Array.length_toList, (FlockSoundness.Refine.decode_points hd).1, hp2]
  obtain ⟨P0, P1, hP⟩ := List.length_eq_two.1 hsz
  rw [hP] at hpts
  obtain ⟨p0, ps', ⟨a0, ha0, hf0⟩, hrest, hps0⟩ := List.forall₂_cons_right_iff.1 hpts
  obtain ⟨p1, ps'', ⟨a1, ha1, hf1⟩, hnil, rfl⟩ := List.forall₂_cons_right_iff.1 hrest
  obtain rfl := List.forall₂_nil_right_iff.1 hnil
  obtain rfl := eq_arr_of_getArr ha0
  obtain rfl := eq_arr_of_getArr ha1
  have hpa : sess.points = #[P0, P1] := Array.toList_inj.1 (show sess.points.toList = (#[P0, P1] : Array _).toList
    from hP.trans rfl)
  have e : sess.points[0]! ++ sess.points[1]! = P0 ++ P1 := by
    rw [hpa]
    rfl
  refine ⟨a0 ++ a1, ?_, by rw [e]; exact f128s_append hf0 hf1⟩
  simp only [recordRound, beq_self_eq_true, Bool.and_self, ↓reduceIte, fld_getD hl, arr_getD hps,
    ← Array.foldl_toList, hps0, List.foldl_cons, List.foldl_nil, Array.empty_append]

theorem pts_of_record {spec : SessionSpec} {record : String} {sess : RunRecord} {J : Json} {b : ByteArray}
    {mP : ℕ} {P : Fin 2 → Fin mP → 𝔽} (hd : Flock.Record.decode spec record = .ok sess)
    (hJ : Flock.JsonParse.parse record = .ok J) (hp2 : spec.points = 2)
    (hb : recordRound J _root_.Spot.Challenger.TREE_SESSION 0 = some b)
    (hbits : Verity.bitsOfBytes b = linkBits fieldEnc P) : ptsOf mP sess.points = P := by
  obtain ⟨cs, hrr, hf⟩ := link_coins hd hJ hp2
  rw [hrr] at hb
  have hpts := FlockSoundness.Refine.decode_points hd
  have h0 : 0 < sess.points.size := by have := hpts.1; omega
  have h1 : 1 < sess.points.size := by have := hpts.1; omega
  have hsz : sess.points[0]!.size = sess.points[1]!.size := by
    rw [hpts.2 sess.points[0]! (by rw [getElem!_pos sess.points 0 h0]; exact Array.getElem_mem_toList h0),
      hpts.2 sess.points[1]! (by rw [getElem!_pos sess.points 1 h1]; exact Array.getElem_mem_toList h1)]
  exact pts_of_bits hsz ((bits_of_coins hf hb).symm.trans hbits)

/-- **A round's place in the session**, at a layout of its rounds: the link's opening, or round `k` of table `t`'s
rep `r` stream. -/
inductive Slot where
  | link
  | rep (t : ℕ) (r : Fin 2) (k : ℕ)

/-- Whether a round is the link's opening. -/
def Slot.isLink : Slot → Bool
  | .link => true
  | .rep .. => false

/-- **The coin count a rep round's message closes with**: its last eight bytes, little-endian, the length field of the
header of the squeeze that answers it. -/
def tailCount (m : Claims.Msg) : ℕ := (Flock.u64OfLE ⟨m.toArray⟩ (m.length - 8)).toNat

theorem tailCount_sq (pre : List UInt8) (kind : UInt8) {n : ℕ} (hn : n < 2 ^ 64) :
    tailCount (pre ++ (FlockSoundness.Refine.Frame.sq kind n).enc) = n := by
  have he : (FlockSoundness.Refine.Frame.sq kind n).enc =
      [4, kind, 0, 0, 0, 0, 0, 0] ++ (Flock.le64 n.toUInt64).data.toList := by
    simp only [FlockSoundness.Refine.Frame.enc, FlockSoundness.Refine.Frame.op, FlockSoundness.Refine.Frame.kind,
      FlockSoundness.Refine.Frame.len, FlockSoundness.Refine.Frame.pay, FlockSoundness.Refine.header_data,
      List.append_nil]
  have hs : (Flock.le64 n.toUInt64).data.toList.length = 8 := by
    rw [Array.length_toList, ByteArray.size_data, FlockSoundness.Discharge.ZkView.le64_size]
  have hb : (⟨(pre ++ (FlockSoundness.Refine.Frame.sq kind n).enc).toArray⟩ : ByteArray) =
      ⟨(pre ++ [4, kind, 0, 0, 0, 0, 0, 0]).toArray⟩ ++ Flock.le64 n.toUInt64 ++ ByteArray.empty := by
    rw [ByteArray.append_empty]
    apply ByteArray.ext
    rw [ByteArray.data_append]
    show (pre ++ (FlockSoundness.Refine.Frame.sq kind n).enc).toArray =
      (pre ++ [4, kind, 0, 0, 0, 0, 0, 0]).toArray ++ (Flock.le64 n.toUInt64).data
    rw [he, ← List.append_assoc, ← List.append_toArray, Array.toArray_toList]
  have hl : (pre ++ (FlockSoundness.Refine.Frame.sq kind n).enc).length - 8 =
      (⟨(pre ++ [4, kind, 0, 0, 0, 0, 0, 0]).toArray⟩ : ByteArray).size := by
    rw [he, List.length_append, List.length_append, hs]
    show _ = (pre ++ [4, kind, 0, 0, 0, 0, 0, 0]).toArray.size
    rw [List.size_toArray, List.length_append]
    rfl
  unfold tailCount
  rw [hl, hb, FlockSoundness.Discharge.ZkView.u64OfLE_mid, Nat.toUInt64_eq, UInt64.toNat_ofNat',
    Nat.mod_eq_of_lt hn]

/-- **`Commit`'s roots in the link's message**: past `Σ`'s 64 bytes, one table's `root_B`. -/
def linkRoots (m : Claims.Msg) : Array ByteArray := #[⟨(m.drop 64).toArray⟩]

/-- Round `i`'s most likely message, where the layout `L` puts the link's opening there. -/
def linkAt {R : ℕ} (L : Fin R → Slot) (modes : Fin R → Option Claims.Msg) (i : Fin R) : Option Claims.Msg :=
  if (L i).isLink then modes i else none

/-- **A rep round's submission** at `J` tables: its table and rep, its message, and the coin count it closes with;
none at the link's opening or at a table past `J`. -/
def slotSub (J : ℕ) : Slot → Claims.Msg → Option (SubmitJ J)
  | .rep t r _, m => if h : t < J then some ⟨⟨t, h⟩, r, ⟨m.toArray⟩, tailCount m⟩ else none
  | .link, _ => none

/-- Round `i`'s submission, from its most likely message. -/
def subAt {R : ℕ} (J : ℕ) (L : Fin R → Slot) (modes : Fin R → Option Claims.Msg) (i : Fin R) :
    Option (SubmitJ J) :=
  (modes i).bind (slotSub J (L i))

/-- **The decoders at the layout `L`**: the registration by `reg`; `Commit`'s roots from the first round a stop fixes
that is the link's opening (`linkRoots`); a submission from the first round a stop fixes that is a rep round
(`slotSub`); and the proof files v2 reads in the play's output (`proofsIn`). -/
def decodersAt {R : ℕ} {Reg : Type} (reg : Option Claims.Msg → Reg) (L : Fin R → Slot) : Decoders R Reg where
  reg := reg
  root modes := (((List.finRange R).findSome? (linkAt L modes)).map linkRoots).getD #[]
  sub J modes := (List.finRange R).findSome? (subAt J L modes)
  proofs o := (proofsIn o).getD #[]

theorem findSome?_of_single {α β : Type} {f : α → Option β} {i : α} (hf : ∀ j, j ≠ i → f j = none) :
    ∀ {l : List α}, i ∈ l → l.findSome? f = f i
  | [], h => absurd h List.not_mem_nil
  | x :: l, h => by
    rw [List.findSome?_cons]
    by_cases hx : x = i
    · subst hx
      cases hfx : f x with
      | some b => rfl
      | none =>
        exact List.findSome?_eq_none_iff.2 fun j _ => by
          by_cases hj : j = x
          · rw [hj, hfx]
          · exact hf j hj
    · rw [hf x hx]
      exact findSome?_of_single hf ((List.mem_cons.1 h).resolve_left fun h' => hx h'.symm)

theorem decodersAt_reg {R : ℕ} {Reg : Type} (reg : Option Claims.Msg → Reg) (L : Fin R → Slot) :
    (decodersAt reg L).reg = reg := rfl

theorem root_single {R : ℕ} {Reg : Type} (reg : Option Claims.Msg → Reg) (L : Fin R → Slot)
    {modes : Fin R → Option Claims.Msg} {i : Fin R} {m : Claims.Msg} (hi : (L i).isLink = true)
    (hm : modes i = some m) (hs : ∀ j, j ≠ i → (L j).isLink = true → modes j = none) :
    (decodersAt reg L).root modes = linkRoots m := by
  have hf : ∀ j, j ≠ i → linkAt L modes j = none := fun j hj => by
    unfold linkAt
    by_cases hl : (L j).isLink = true
    · rw [ite_eq_left hl]
      exact hs j hj hl
    · rw [ite_eq_right hl]
  show (((List.finRange R).findSome? (linkAt L modes)).map linkRoots).getD #[] = linkRoots m
  rw [findSome?_of_single hf (List.mem_finRange i), linkAt, ite_eq_left hi, hm, Option.map_some, Option.getD_some]

theorem sub_single {R : ℕ} {Reg : Type} (reg : Option Claims.Msg → Reg) (L : Fin R → Slot) (J : ℕ)
    {modes : Fin R → Option Claims.Msg} {i : Fin R} {m : Claims.Msg} {t k : ℕ} {r : Fin 2}
    (hi : L i = .rep t r k) (ht : t < J) (hm : modes i = some m) (hs : ∀ j, j ≠ i → modes j = none) :
    (decodersAt reg L).sub J modes = some ⟨⟨t, ht⟩, r, ⟨m.toArray⟩, tailCount m⟩ := by
  have hf : ∀ j, j ≠ i → subAt J L modes j = none := fun j hj => by
    rw [subAt, hs j hj, Option.bind_none]
  show (List.finRange R).findSome? (subAt J L modes) = _
  rw [findSome?_of_single hf (List.mem_finRange i), subAt, hm, hi, Option.bind_some]
  exact dite_eq_left ht

theorem sub_eq_none {R : ℕ} {Reg : Type} (reg : Option Claims.Msg → Reg) (L : Fin R → Slot) (J : ℕ)
    {modes : Fin R → Option Claims.Msg} (h : ∀ i, (L i).isLink = false → modes i = none) :
    (decodersAt reg L).sub J modes = none := by
  show (List.finRange R).findSome? (subAt J L modes) = none
  refine List.findSome?_eq_none_iff.2 fun i _ => ?_
  unfold subAt
  cases hL : L i with
  | link => cases modes i <;> rfl
  | rep t r k =>
    rw [h i (by rw [hL]; rfl), Option.bind_none]

theorem proofs_of_sent {R : ℕ} {Reg : Type} (reg : Option Claims.Msg → Reg) (L : Fin R → Slot)
    {o : Verity.Exchange} {proofs : Array ByteArray} (h : sentProofs o.sent = some proofs) :
    (decodersAt reg L).proofs o = proofs := by
  show (sentProofs o.sent).getD #[] = proofs
  rw [h, Option.getD_some]

theorem setup_one {I : FlockSoundness.Discharge.Exec.Inputs} {own : Option ByteArray} {record : String}
    {spec : SessionSpec} {sts : Array Flock.Stmt} (hT : TagsOkZHJ I) (hc : I.count = 1)
    (hs : FlockSoundness.Discharge.Exec.setupOfP (FlockSoundness.Discharge.ZkExec.zkInputs I) own record =
      .ok (spec, sts)) :
    (∃ t, spec.tables = #[t]) ∧ spec.sigma.size = 64 ∧ spec.retainRounds = true ∧ spec.points = 2 := by
  unfold FlockSoundness.Discharge.Exec.setupOfP at hs
  rcases FlockSoundness.Discharge.HiddenExec.setupTables_okHP
      ((FlockSoundness.Discharge.ZkExec.zkTags_hiddenOutputs I).trans hT.1) hs with
    ⟨-, st, hst, rfl, -⟩ | ⟨h1, -⟩
  · obtain ⟨pi, hP⟩ := FlockSoundness.Discharge.Hidden.setupHidden_specP hst
    refine ⟨⟨st.tags.table, rfl⟩, ?_, ?_, rfl⟩
    · show st.sigma.size = 64
      rw [hP.sigma]
      exact FlockSoundness.Refine.sha512_size _
    · show st.tags.retainRounds = true
      rw [hP.tagsEq]
      exact (FlockSoundness.Discharge.ZkExec.zkTags_retainRounds I).trans hT.2.1
  · exact absurd h1 (by show ¬ 1 < I.count; omega)

theorem linkRoots_linkMsg {spec : SessionSpec} {record : String} {sess : RunRecord} {t : String} {m : Claims.Msg}
    (hd : Flock.Record.decode spec record = .ok sess) (ht : spec.tables = #[t]) (hσ : spec.sigma.size = 64)
    (hm : linkMsg record spec.tables = some m) : linkRoots m = sess.rootB := by
  have h := rootBOf_linkMsg hd ht
  rw [hm, Option.bind_some] at h
  unfold rootBOf at h
  rw [hσ] at h
  split at h
  · exact Option.some.inj h
  · cases h

theorem root_record {R : ℕ} {Reg : Type} (reg : Option Claims.Msg → Reg) (L : Fin R → Slot)
    {modes : Fin R → Option Claims.Msg} {i : Fin R} {m : Claims.Msg} {spec : SessionSpec} {record : String}
    {sess : RunRecord} {t : String} (hi : (L i).isLink = true) (hm : modes i = some m)
    (hs : ∀ j, j ≠ i → (L j).isLink = true → modes j = none) (hd : Flock.Record.decode spec record = .ok sess)
    (ht : spec.tables = #[t]) (hσ : spec.sigma.size = 64) (hl : linkMsg record spec.tables = some m) :
    (decodersAt reg L).root modes = sess.rootB :=
  (root_single reg L hi hm hs).trans (linkRoots_linkMsg hd ht hσ hl)

theorem check_n {n : ℕ} {a c d : Bool} {why : String} {u : Unit}
    (h : Flock.Record.check (a && decide (n ≤ 65536) && c && d) why = .ok u) : n ≤ 65536 := by
  have := check_ok_iff.1 h
  simp only [Bool.and_eq_true, decide_eq_true_eq] at this
  exact this.1.1.2

set_option maxHeartbeats 1000000 in
theorem decode_roundsN {spec : SessionSpec} {text : String} {sess : RunRecord}
    (h : Flock.Record.decode spec text = .ok sess) : ∀ s ∈ sess.streams.toList, ∀ r ∈ s.rounds.toList, r.n ≤ 65536 := by
  unfold Flock.Record.decode at h
  dsimp only at h
  repeat' (first | (obtain ⟨_, _, h⟩ := ebind_ok h) | (split at h))
  all_goals
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    intro s hs
    try dsimp only at hs
    obtain ⟨x, hx⟩ := mapM_ok_mem (by assumption) s hs
    try dsimp only at hx
    repeat' (first | (obtain ⟨_, _, hx⟩ := ebind_ok hx) | (split at hx))
    all_goals
      simp only [pure, Except.pure, Except.ok.injEq] at hx
      subst hx
      intro r hr
      obtain ⟨k, y, hg⟩ := mapIdxM_ok_mem (by assumption) r hr
      try dsimp only at hg
      repeat' (first | (obtain ⟨_, _, hg⟩ := ebind_ok hg) | (split at hg))
      all_goals
        simp only [pure, Except.pure, Except.ok.injEq] at hg
        subst hg
        exact check_n (by assumption)

theorem verifyRep_of_accepted {spec : SessionSpec} {tabs : Array Flock.Setup} {record : String}
    {proofs : Array ByteArray} {sess : RunRecord} (hv : Flock.Zk.verify spec tabs record proofs = .ok ())
    (hd : Flock.Record.decode spec record = .ok sess) (r : Fin 2) :
    ∃ (h0 : 0 < tabs.size) (sch : Flock.Schedule) (tPad : ℕ) (c : ByteArray),
      Flock.Zk.schedule tabs[0].wideLeaf tabs[0].m = .ok (sch, tPad) ∧
      Flock.Zk.verifyRep spec tabs[0] sch tPad sess 0 r.val proofs[r.val]! = .ok c := by
  obtain ⟨-, sess', schs, hd', -, h0, -, -, hall⟩ :=
    FlockSoundness.Discharge.ZkExec.zk_verify_ok spec tabs record proofs hv
  rw [hd] at hd'
  obtain rfl := Except.ok.inj hd'
  obtain ⟨hs, c, hsch, hr0, hr1⟩ := hall 0 h0
  refine ⟨h0, schs[0].1, schs[0].2, c, hsch, ?_⟩
  obtain ⟨_ | _ | n, hr⟩ := r
  · exact hr0
  · exact hr1
  · omega

theorem rep_tailCount {spec : SessionSpec} {record : String} {sess : RunRecord} {st : Flock.Setup}
    {sch : Flock.Schedule} {tPad i r : ℕ} {bytes cap : ByteArray} {y : StreamRec} {k : ℕ} {m : Claims.Msg}
    (hv : Flock.Zk.verifyRep spec st sch tPad sess i r bytes = .ok cap)
    (hsch : Flock.Zk.schedule st.wideLeaf st.m = .ok (sch, tPad))
    (hd : Flock.Record.decode spec record = .ok sess) (hret : spec.retainRounds = true)
    (hy : sess.streams.find? (·.name == repStreamName spec.tables[i]! r) = some y) (hk : k < y.rounds.size)
    (hm : roundMsg record (repStreamName spec.tables[i]! r) k = some m) : tailCount m = y.rounds[k].n := by
  obtain ⟨stream, extras, zp, tr, hf, -, -, -, -, hrun, htk, -, -⟩ :=
    FlockSoundness.Discharge.ZkExec.zk_verifyRep_ok spec st sch tPad sess i r bytes cap hv
  have hfy : stream = y := Option.some.inj (hf.symm.trans hy)
  rw [hfy] at hrun htk
  have hmem : y ∈ sess.streams.toList := Array.mem_toList_iff.2 (Array.mem_of_find?_eq_some hy)
  obtain ⟨pre, kind, hR⟩ := FlockSoundness.Discharge.ZkExec.zk_rep_counts st sch tPad r extras zp y.rounds _ tr hsch
    hrun htk (FlockSoundness.Discharge.Exec.decode_allRetained hret hd y hmem) k hk
  obtain ⟨b, hb, hm'⟩ := roundMsg_decode hd hret hy hk
  rw [hm] at hm'
  obtain rfl := Option.some.inj hm'
  simp only [getElem!_pos y.rounds k hk, FlockSoundness.Refine.Retains, hb, Option.map_some,
    Option.some.injEq] at hR
  rw [hR]
  exact tailCount_sq pre kind
    (Nat.lt_of_le_of_lt (decode_roundsN hd y hmem _ (Array.getElem_mem_toList hk)) (by decide))

theorem rep_tailCount_accepted {spec : SessionSpec} {tabs : Array Flock.Setup} {record : String}
    {proofs : Array ByteArray} {sess : RunRecord} (hv : Flock.Zk.verify spec tabs record proofs = .ok ())
    (hd : Flock.Record.decode spec record = .ok sess) (hret : spec.retainRounds = true) (r : Fin 2)
    {y : StreamRec} {k : ℕ} {m : Claims.Msg}
    (hy : sess.streams.find? (·.name == repStreamName spec.tables[0]! r.val) = some y) (hk : k < y.rounds.size)
    (hm : roundMsg record (repStreamName spec.tables[0]! r.val) k = some m) : tailCount m = y.rounds[k].n := by
  obtain ⟨h0, sch, tPad, c, hsch, hrep⟩ := verifyRep_of_accepted hv hd r
  exact rep_tailCount hrep hsch hd hret hy hk hm

theorem answered_of_record {spec : SessionSpec} {record : String} {sess : RunRecord} {J : Json} {name : String}
    {y : StreamRec} {k : ℕ} {m : Claims.Msg} {b : ByteArray} {n : ℕ} {c : Fin n → 𝔽}
    (hd : Flock.Record.decode spec record = .ok sess) (hJ : Flock.JsonParse.parse record = .ok J)
    (hret : spec.retainRounds = true) (hn : (name == _root_.Spot.Challenger.TREE_SESSION) = false)
    (hy : sess.streams.find? (·.name == name) = some y) (hk : k < y.rounds.size)
    (hm : roundMsg record name k = some m) (hb : recordRound J name k = some b)
    (hbits : Verity.bitsOfBytes b = roundBits fieldEnc c) :
    y.rounds[k].bytes = some ⟨m.toArray⟩ ∧ y.rounds[k].sha = spec.dg ⟨m.toArray⟩ ∧ y.rounds[k].n = n ∧
      y.rounds[k].coins.size = n ∧ ∀ i : Fin n, (c i).val = y.rounds[k].coins[i.val]! := by
  obtain ⟨cs, hrr, hf, hsha⟩ := recordRound_decode hd hJ hn hy hk
  rw [hrr] at hb
  obtain ⟨hsz, hc⟩ := coins_of_bits hf hb hbits
  obtain ⟨b', hb', hm'⟩ := roundMsg_decode hd hret hy hk
  rw [hm] at hm'
  obtain rfl := Option.some.inj hm'
  have hmem : y ∈ sess.streams.toList := Array.mem_toList_iff.2 (Array.mem_of_find?_eq_some hy)
  have hwf := FlockSoundness.Refine.decode_roundsWF hd y hmem k hk
  rw [getElem!_pos y.rounds k hk] at hwf
  exact ⟨hb', hsha _ hb', hwf.symm.trans hsz, hsz, hc⟩

end Verity.Sampled
