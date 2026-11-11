/-
Byte encodings and a total reader over `ByteArray` (`Flock.Verify`, `Flock.Proof`).
-/
module

public import Flock.Field

@[expose] public section
set_option backward.privateInPublic true
set_option backward.privateInPublic.warn false

namespace Flock

abbrev V := Except String

def le64 (x : UInt64) : ByteArray := Id.run do
  let mut b := ByteArray.emptyWithCapacity 8
  for i in [0:8] do b := b.push (x >>> (8 * i).toUInt64).toUInt8
  return b

def le32 (x : UInt32) : ByteArray := Id.run do
  let mut b := ByteArray.emptyWithCapacity 4
  for i in [0:4] do b := b.push (x >>> (8 * i).toUInt32).toUInt8
  return b

def be64 (x : UInt64) : ByteArray := Id.run do
  let mut b := ByteArray.emptyWithCapacity 8
  for i in [0:8] do b := b.push (x >>> (8 * (7 - i)).toUInt64).toUInt8
  return b

def be32 (x : UInt32) : ByteArray := Id.run do
  let mut b := ByteArray.emptyWithCapacity 4
  for i in [0:4] do b := b.push (x >>> (8 * (3 - i)).toUInt32).toUInt8
  return b

def ByteArray.ofString (s : String) : ByteArray := s.toUTF8

private def hexDigit (n : UInt8) : Char := if n < 10 then Char.ofNat (48 + n.toNat) else Char.ofNat (87 + n.toNat)

def hexOf (b : ByteArray) : String := b.foldl (fun s x => (s.push (hexDigit (x / 16))).push (hexDigit (x % 16))) ""

def unhex (s : String) : V ByteArray := do
  let cs := s.toList.toArray
  if cs.size % 2 != 0 then throw s!"odd-length hex"
  let v := fun (c : Char) =>
    if c.isDigit then some (c.toNat - 48) else if 'a' ≤ c && c ≤ 'f' then some (c.toNat - 87) else none
  let mut b := ByteArray.emptyWithCapacity (cs.size / 2)
  for i in [0:cs.size / 2] do
    match v (cs.getD (2 * i) ' '), v (cs.getD (2 * i + 1) ' ') with
    | some h, some l => b := b.push (h * 16 + l).toUInt8
    | _, _ => throw "not lowercase hex"
  return b

/-! ## The hex codec's round trip

`unhex` reads `hexOf`'s output back exactly (`Hex.unhex_hexOf`), so a key read back from a digest's hex name is the
digest it names (`Hex.unhex_eq_of_hexOf`). -/

namespace Hex

theorem foldl_loop_eq {β : Type} (f : β → UInt8 → β) (as : ByteArray) (stop : Nat) (h : stop ≤ as.size) :
    ∀ (i j : Nat) (b : β), i + j = stop →
      ByteArray.foldlM.loop (m := Id) (fun b a => pure (f b a)) as stop h i j b =
        ((as.data.toList.drop j).take i).foldl f b := by
  intro i
  induction i with
  | zero => intro j b _; unfold ByteArray.foldlM.loop; split <;> rfl
  | succ i ih =>
    intro j b hij
    unfold ByteArray.foldlM.loop
    have hj : j < stop := by omega
    rw [dite_eq_left_of_eq_true (eq_true hj)]
    have hjs : j < as.data.toList.length := by simp; omega
    rw [List.drop_eq_getElem_cons hjs]
    simp only [List.take_succ_cons, List.foldl_cons]
    rw [← ih (j + 1) _ (by omega)]
    simp [Array.getElem_toList]
    rfl

theorem byteArray_foldl_eq {β : Type} (f : β → UInt8 → β) (init : β) (as : ByteArray) :
    as.foldl f init = as.data.toList.foldl f init := by
  unfold ByteArray.foldl ByteArray.foldlM
  rw [dite_eq_left_of_eq_true (eq_true (Nat.le_refl _))]
  dsimp only
  rw [foldl_loop_eq f as as.size (Nat.le_refl _) (as.size - 0) 0 init (by omega)]
  simp only [List.drop_zero, Nat.sub_zero]
  rw [List.take_of_length_le (by simp)]
  rfl

/-- A nibble's lowercase hex digit. -/
def hd (d : Nat) : Char := if d < 10 then Char.ofNat (48 + d) else Char.ofNat (87 + d)

theorem foldl_push2 (d₁ d₂ : UInt8 → Char) : ∀ (l : List UInt8) (s : String),
    (l.foldl (fun s x => (s.push (d₁ x)).push (d₂ x)) s).toList =
      s.toList ++ l.flatMap fun x => [d₁ x, d₂ x]
  | [], s => by simp
  | x :: l, s => by
    rw [List.foldl_cons, foldl_push2 d₁ d₂ l]
    simp [String.toList_push]

theorem hexOf_toList (b : ByteArray) :
    (hexOf b).toList = b.data.toList.flatMap fun x => [hd (x.toNat / 16), hd (x.toNat % 16)] := by
  unfold hexOf
  rw [byteArray_foldl_eq, foldl_push2]
  simp only [String.toList_empty, List.nil_append]
  congr 1


/-- `unhex`'s digit value. -/
def hv (c : Char) : Option Nat :=
  if c.isDigit = true then some (c.toNat - 48)
  else if (decide ('a' ≤ c) && decide (c ≤ 'f')) = true then some (c.toNat - 87) else none

theorem hv_hd_nat : ∀ d < 16, hv (hd d) = some d := by decide +kernel

theorem hv_hd {d : Nat} (h : d < 16) : hv (hd d) = some d := hv_hd_nat d h

theorem length_flatMap2 (L : List UInt8) (f g : UInt8 → Char) :
    (L.flatMap fun x => [f x, g x]).length = 2 * L.length := by
  induction L with
  | nil => rfl
  | cons a L ih => simp only [List.flatMap_cons, List.length_append, ih, List.length_cons, List.length_nil]; omega

theorem flatMap_get? (L : List UInt8) (f g : UInt8 → Char) (i : Nat) (hi : i < L.length) :
    (L.flatMap fun x => [f x, g x])[2 * i]? = some (f L[i]) ∧
      (L.flatMap fun x => [f x, g x])[2 * i + 1]? = some (g L[i]) := by
  induction L generalizing i with
  | nil => simp at hi
  | cons a L ih =>
    cases i with
    | zero => simp
    | succ i =>
      have := ih i (by simpa using hi)
      simp only [List.flatMap_cons, List.getElem_cons_succ]
      rw [List.getElem?_append_right (by simp; omega), List.getElem?_append_right (by simp; omega)]
      simpa [show 2 * (i + 1) - 2 = 2 * i by omega, show 2 * (i + 1) + 1 - 2 = 2 * i + 1 by omega] using this

theorem flatMap_getD (L : List UInt8) (f g : UInt8 → Char) (i : Nat) (hi : i < L.length) :
    (L.flatMap fun x => [f x, g x]).toArray.getD (2 * i) ' ' = f L[i] ∧
      (L.flatMap fun x => [f x, g x]).toArray.getD (2 * i + 1) ' ' = g L[i] := by
  obtain ⟨h1, h2⟩ := flatMap_get? L f g i hi
  simp [Array.getD_eq_getD_getElem?, h1, h2]

theorem forIn_bytes (L : List UInt8) (body : Nat → ByteArray → V (ForInStep ByteArray))
    (hb : ∀ i (hi : i < L.length) acc, body i acc = .ok (.yield (acc.push L[i]))) :
    ∀ n s acc, s + n = L.length →
      forIn (List.range' s n) acc body = .ok ⟨acc.data ++ (L.drop s).toArray⟩ := by
  intro n
  induction n with
  | zero =>
    intro s acc h
    simp [show L.drop s = [] from List.drop_eq_nil_of_le (by omega)]
    rfl
  | succ n ih =>
    intro s acc h
    rw [List.range'_succ, List.forIn_cons, hb s (by omega) acc]
    show forIn (List.range' (s + 1) n) (acc.push L[s]) body = _
    rw [ih (s + 1) _ (by omega), List.drop_eq_getElem_cons (i := s) (l := L) (by omega)]
    simp [ByteArray.data_push]

/-- **Hex reads back**: `unhex` of `hexOf b` is `b`. -/
theorem unhex_hexOf (b : ByteArray) : unhex (hexOf b) = .ok b := by
  have hl := hexOf_toList b
  have hlen : (hexOf b).toList.toArray.size = 2 * b.data.toList.length := by
    rw [hl, List.size_toArray, length_flatMap2]
  unfold unhex
  simp only [hlen, Nat.mul_mod_right, bne_self_eq_false, Bool.false_eq_true, ite_false,
    Nat.mul_div_cancel_left _ (by decide : 0 < 2), Std.Legacy.Range.forIn_eq_forIn_range']
  have hd2 : ∀ i (hi : i < b.data.toList.length),
      (hexOf b).toList.toArray.getD (2 * i) ' ' = hd (b.data.toList[i].toNat / 16) ∧
        (hexOf b).toList.toArray.getD (2 * i + 1) ' ' = hd (b.data.toList[i].toNat % 16) := by
    intro i hi
    rw [hl]
    exact flatMap_getD _ _ _ i hi
  rw [forIn_bytes b.data.toList _ ?_ _ 0 _ (by simp)]
  · simp only [ByteArray.emptyWithCapacity, List.drop_zero, Array.toArray_toList]
    show Except.ok (⟨#[] ++ b.data⟩ : ByteArray) = _
    rw [Array.empty_append]
  · intro i hi acc
    have hx : b.data.toList[i].toNat < 256 := UInt8.toNat_lt _
    obtain ⟨e1, e2⟩ := hd2 i hi
    have v1 := hv_hd (d := b.data.toList[i].toNat / 16) (by omega)
    have v2 := hv_hd (d := b.data.toList[i].toNat % 16) (by omega)
    unfold hv at v1 v2
    simp only [e1, e2, v1, v2, Nat.div_add_mod', Nat.toUInt8_eq, UInt8.ofNat_toNat]
    rfl

/-- A hex name read back is the bytes it names: if `hexOf x` is `hex` and `unhex hex` is `key`, then `key` is `x`. So
an archived object's key, read from its name, is the object's SHA-512 once the object hashes to its name. -/
theorem unhex_eq_of_hexOf {x key : ByteArray} {hex : String} (hh : (hexOf x == hex) = true)
    (hk : unhex hex = .ok key) : key = x := by
  rw [← beq_iff_eq.1 hh, unhex_hexOf] at hk
  exact (Except.ok.inj hk).symm

end Hex

/-- Record hex of an F128: `{hi:016x}{lo:016x}` (`Flock.RecordText`). -/
def f128OfHex (s : String) : V F128 := do
  let b ← unhex s
  if b.size != 16 then throw "an F128 is 32 hex digits"
  let w := fun (o : Nat) => Id.run do
    let mut x : UInt64 := 0
    for i in [0:8] do x := (x <<< 8) ||| (b.get! (o + i)).toUInt64
    return x
  return ⟨w 8, w 0⟩

def u64OfLE (b : ByteArray) (o : Nat) : UInt64 := Id.run do
  let mut x : UInt64 := 0
  for i in [0:8] do x := x ||| ((b.get! (o + i)).toUInt64 <<< (8 * i).toUInt64)
  return x

def u32OfLE (b : ByteArray) (o : Nat) : UInt32 := Id.run do
  let mut x : UInt32 := 0
  for i in [0:4] do x := x ||| ((b.get! (o + i)).toUInt32 <<< (8 * i).toUInt32)
  return x

/-- A position in a byte string; every read checks bounds and fails with `Except`. -/
structure Reader where
  data : ByteArray
  pos : Nat
  deriving Inhabited

namespace Reader

abbrev M := StateT Reader V

def need (n : Nat) : M Unit := do
  let r ← get
  if r.pos + n > r.data.size then throw s!"truncated at byte {r.pos} (need {n})"

def u8 : M UInt8 := do
  need 1
  let r ← get
  set { r with pos := r.pos + 1 }
  return r.data.get! r.pos

def u32 : M UInt32 := do
  need 4
  let r ← get
  set { r with pos := r.pos + 4 }
  return u32OfLE r.data r.pos

def u64 : M UInt64 := do
  need 8
  let r ← get
  set { r with pos := r.pos + 8 }
  return u64OfLE r.data r.pos

/-- A grinding nonce: `0`, the only one a prover sends under live coins (R4); any other is refused. -/
def nonce0 (what : String) : M UInt64 := do
  let n ← u64
  if n != 0 then throw s!"{what} ≠ 0 (R4)"
  pure n

def f128 : M F128 := do
  let lo ← u64
  let hi ← u64
  return ⟨lo, hi⟩

def f256 : M F256 := do
  let c0 ← f128
  let c1 ← f128
  return ⟨c0, c1⟩

def bytes (n : Nat) : M ByteArray := do
  need n
  let r ← get
  set { r with pos := r.pos + n }
  return r.data.extract r.pos (r.pos + n)

/-- A `Vec<T>` whose length must be `want` (checked before any element is read, `Flock.Proof`). -/
def vecExact {α} (want : Nat) (elem : M α) (what : String) : M (Array α) := do
  let n ← u64
  if n.toNat != want then throw s!"{what}: length {n}, want {want}"
  let mut out := Array.mkEmpty want
  for _ in [0:want] do
    out := out.push (← elem)
  return out

/-- A `Vec<T>` of any length, bounded by the remaining input (each element takes at least `minSize` bytes). -/
def vecAny {α} (minSize : Nat) (elem : M α) (what : String) : M (Array α) := do
  let n ← u64
  let r ← get
  if n.toNat * minSize > r.data.size - r.pos then throw s!"{what}: length {n} exceeds the input"
  let mut out := Array.mkEmpty n.toNat
  for _ in [0:n.toNat] do
    out := out.push (← elem)
  return out

/-- `m` read from the first byte of `bytes`, refused unless it stops after the last (`Flock.Proof`): a file is the one encoding of
what it decodes to, so every byte the record's digest binds (S17) is one the verifier reads. -/
def runAll {α} (m : M α) (bytes : ByteArray) (what : String) : V α :=
  match m.run ⟨bytes, 0⟩ with
  | .error e => .error e
  | .ok (a, r) =>
    if r.pos = bytes.size then .ok a
    else .error s!"{what}: the file has {bytes.size} bytes and its encoding ends at byte {r.pos}"

theorem runAll_ok {α} {m : M α} {bytes : ByteArray} {what : String} {a : α} (h : runAll m bytes what = .ok a) :
    ∃ r : Reader, m.run ⟨bytes, 0⟩ = .ok (a, r) ∧ r.pos = bytes.size := by
  unfold runAll at h
  split at h
  · cases h
  · rename_i a' r hm
    split at h
    · cases h
      exact ⟨r, hm, by assumption⟩
    · cases h

theorem runAll_of_run {α} {m : M α} {bytes : ByteArray} {what : String} {a : α} {r : Reader}
    (h : m.run ⟨bytes, 0⟩ = .ok (a, r)) (hr : r.pos = bytes.size) : runAll m bytes what = .ok a := by
  unfold runAll
  rw [h]
  simp only [hr, ite_true]

/-- `runAll` for `Flock.Proof`'s plain pair, which takes one byte more: `zk_level0`'s `Option` tag, `None` (0). The zk patch puts the
field at the end of upstream's `LigeritoProof` and writes it as `None` on every plain proof, and upstream's builds don't write it,
so a file ends after `consistency_batch_grinding_nonces` or after one 0 after it (D17). -/
def runPlain {α} (m : M α) (bytes : ByteArray) (what : String) : V α :=
  match m.run ⟨bytes, 0⟩ with
  | .error e => .error e
  | .ok (a, r) =>
    if r.pos = bytes.size then .ok a
    else if bytes[r.pos]! ≠ 0 then .error s!"zk_level0: tag {bytes[r.pos]!}, want 0: a plain proof carries no hiding messages"
    else if r.pos + 1 = bytes.size then .ok a
    else .error s!"{what}: the file has {bytes.size} bytes and its encoding ends at byte {r.pos + 1}"

theorem runPlain_ok {α} {m : M α} {bytes : ByteArray} {what : String} {a : α} (h : runPlain m bytes what = .ok a) :
    ∃ r : Reader, m.run ⟨bytes, 0⟩ = .ok (a, r) ∧ (r.pos = bytes.size ∨ r.pos + 1 = bytes.size) := by
  unfold runPlain at h
  split at h
  · cases h
  · rename_i a' r hm
    split at h
    · cases h
      exact ⟨r, hm, .inl (by assumption)⟩
    · split at h
      · cases h
      · split at h
        · cases h
          exact ⟨r, hm, .inr (by assumption)⟩
        · cases h

theorem runPlain_of_run {α} {m : M α} {bytes : ByteArray} {what : String} {a : α} {r : Reader}
    (h : m.run ⟨bytes, 0⟩ = .ok (a, r)) (hr : r.pos = bytes.size) : runPlain m bytes what = .ok a := by
  unfold runPlain
  rw [h]
  simp only [hr, ite_true]

theorem runPlain_of_run_tag {α} {m : M α} {b : ByteArray} {what : String} {a : α} {r : Reader}
    (h : m.run ⟨b ++ ByteArray.mk #[0], 0⟩ = .ok (a, r)) (hr : r.pos = b.size) :
    runPlain m (b ++ ByteArray.mk #[0]) what = .ok a := by
  have hs : (b ++ ByteArray.mk #[0]).size = b.size + 1 := by rw [ByteArray.size_append]; rfl
  have h0 : (b ++ ByteArray.mk #[0])[b.size]! = 0 := by
    rw [getElem!_pos (b ++ ByteArray.mk #[0]) b.size (by omega), ByteArray.getElem_append_right (by omega)]
    simp only [Nat.sub_self]
    rfl
  unfold runPlain
  rw [h]
  simp only [hs, hr, ite_true]
  simp [h0]

end Reader

end Flock
