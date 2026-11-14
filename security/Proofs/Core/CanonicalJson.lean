module

public import CoreProtocol.Protocol
public import Proofs.Flock.Claims.Complete.Json.Canon
public import Proofs.Flock.Claims.Complete.Json.Parse
public import Proofs.Flock.Claims.Complete.Json.RoundTrip

/-!
# What `Core.canonicalJson` accepts

`Core.canonicalJson` reads a protocol's parameters, and a commitment or evidence that is JSON, with the total parser
(`Flock.JsonParse.parse`) and accepts where `Flock.canon` writes the bytes back. So what it accepts is a set a
theorem states, with no hypothesis on the bytes:

* `canonicalJson_accepts`: the bytes it accepts are exactly the canonical texts of printable values
  (`Claims.Json.Good`: every object a well-formed tree, every number an integer);
* `canonicalJson_good`: the value it returns is printable;
* `canonicalJson_same`: the value it returns on a printable value's canonical text is alike that value
  (`Claims.Json.Same`), so two printable values with one canonical text are read as one value;
* `canonicalJson_iff`: it returns `j` exactly where the bytes are a text the parser reads as `j` and `canon` writes
  back from `j`; on a text's own bytes (`canonicalJson_toUTF8_iff`), exactly where the parser reads the text as `j`
  and `canon` writes it back, so what it returns there is the parser's reading of the text.

The steps: bytes decode to a string exactly where they are its UTF-8 (`fromUTF8?_eq_some_iff`); a text of ASCII
without DEL has every byte below `0x7f` (`toUTF8_ascii`), and `canon` writes only such texts; the parser builds only
well-formed trees (`FlockComplete.JsonRT.parse_wf`), and one of those that `canon` prints has only integers
(`good_of_canon`); the parser reads `canon`'s text of a printable value back as a value alike it
(`FlockComplete.JsonRT.parse_canon`), which `canon` prints the same (`canon_same`).
-/

@[expose] public section

namespace Verity.CanonicalJson

open Lean (Json)
open Claims.Json (Ascii Same Good Wf)
open FlockComplete.JsonRT (parse_wf parse_canon canon_ascii canon_same)

/-! ## Bytes and texts -/

private theorem utf8Encode_ascii : ∀ (l : List Char), (∀ c ∈ l, c.toNat < 0x7f) →
    l.utf8Encode.data.all (· < 0x7f) = true
  | [], _ => by simp
  | c :: l, h => by
    have hc : c.val.toNat < 0x7f := h c (List.mem_cons_self ..)
    have h1 : c.utf8Size = 1 :=
      Char.utf8Size_eq_one_iff.2 (UInt32.le_iff_toNat_le.2 (by show c.val.toNat ≤ 127; omega))
    have hl := utf8Encode_ascii l fun d hd => h d (List.mem_cons_of_mem _ hd)
    rw [List.utf8Encode_cons, List.utf8Encode_singleton, String.utf8EncodeChar_eq_singleton h1]
    simp only [ByteArray.data_append, Array.all_append, hl, Bool.and_true]
    have hb : c.val.toUInt8 < 0x7f := by
      rw [UInt8.lt_iff_toNat_lt, UInt32.toNat_toUInt8]
      show c.val.toNat % 256 < 127
      omega
    simpa [List.data_toByteArray] using hb

/-- **A text of ASCII without DEL has every byte below `0x7f`**: `Core.canonicalJson`'s first check passes on it. -/
theorem toUTF8_ascii (s : String) (h : Ascii s) : s.toUTF8.data.all (· < 0x7f) = true := by
  rw [String.toUTF8_eq_toByteArray, ← String.utf8Encode_toList]
  exact utf8Encode_ascii _ h

/-- **Bytes decode to a string exactly where they are its UTF-8.** -/
theorem fromUTF8?_eq_some_iff {b : ByteArray} {s : String} : String.fromUTF8? b = some s ↔ b = s.toUTF8 := by
  constructor
  · intro h
    unfold String.fromUTF8? at h
    split at h
    · cases h
      rw [String.toUTF8_eq_toByteArray]
      rfl
    · cases h
  · rintro rfl
    unfold String.fromUTF8?
    have hv : s.toUTF8.IsValidUTF8 := by rw [String.toUTF8_eq_toByteArray]; exact s.isValidUTF8
    rw [dite_eq_left hv]
    congr 1
    apply String.toByteArray_inj.1
    simp [String.fromUTF8]

/-! ## The value it returns, and the text it returns it on -/

/-- **What `canonicalJson` returns it read with the total parser from the bytes' text, which `canon` writes back.** -/
theorem canonicalJson_ok {b : ByteArray} {j : Json} (h : Core.canonicalJson b = .ok j) :
    ∃ s, b = s.toUTF8 ∧ Flock.JsonParse.parse s = .ok j ∧ Flock.canon j = .ok s := by
  unfold Core.canonicalJson at h
  cases hb : b.data.all (· < 0x7f) with
  | false => rw [hb] at h; cases h
  | true =>
    rw [hb] at h
    cases hs : String.fromUTF8? b with
    | none => rw [hs] at h; cases h
    | some s =>
      rw [hs] at h
      dsimp only at h
      rw [ite_eq_left rfl] at h
      cases hp : Flock.JsonParse.parse s with
      | error e => rw [hp] at h; cases h
      | ok j₀ =>
        rw [hp] at h
        dsimp only at h
        cases hc : Flock.canon j₀ with
        | error e =>
          simp only [bind, Except.bind, pure, Except.pure, hc] at h
          cases h
        | ok s' =>
          simp only [bind, Except.bind, pure, Except.pure, hc] at h
          by_cases hne : (s' != s) = true
          · rw [ite_eq_left hne] at h; cases h
          · rw [ite_eq_right hne] at h
            have hj : j₀ = j := Except.ok.inj h
            have he : s' = s := by simpa using hne
            subst hj he
            exact ⟨s', fromUTF8?_eq_some_iff.1 hs, hp, hc⟩

/-- **A text the total parser reads as a value `canon` writes it back from is accepted, as that value.** -/
theorem canonicalJson_of {s : String} {j : Json} (hp : Flock.JsonParse.parse s = .ok j)
    (hc : Flock.canon j = .ok s) : Core.canonicalJson s.toUTF8 = .ok j := by
  unfold Core.canonicalJson
  rw [toUTF8_ascii s (canon_ascii j hc), fromUTF8?_eq_some_iff.2 rfl]
  dsimp only
  rw [ite_eq_left rfl, hp]
  dsimp only
  simp only [bind, Except.bind, pure, Except.pure, hc]
  rw [ite_eq_right (by simp)]

/-- **`canonicalJson` returns `j` exactly where the bytes are a text the total parser reads as `j` and `canon` writes
back from `j`.** -/
theorem canonicalJson_iff (b : ByteArray) (j : Json) :
    Core.canonicalJson b = .ok j ↔
      ∃ s, b = s.toUTF8 ∧ Flock.JsonParse.parse s = .ok j ∧ Flock.canon j = .ok s :=
  ⟨canonicalJson_ok, fun ⟨_, hb, hp, hc⟩ => hb ▸ canonicalJson_of hp hc⟩

/-- **On a text's bytes `canonicalJson` returns the total parser's reading of the text**, where `canon` writes the
text back from it, and refuses elsewhere. -/
theorem canonicalJson_toUTF8_iff (s : String) (j : Json) :
    Core.canonicalJson s.toUTF8 = .ok j ↔ Flock.JsonParse.parse s = .ok j ∧ Flock.canon j = .ok s := by
  constructor
  · intro h
    obtain ⟨s', hb, hp, hc⟩ := canonicalJson_ok h
    have hb' : s.toByteArray = s'.toByteArray := by simpa only [String.toUTF8_eq_toByteArray] using hb
    obtain rfl : s = s' := String.toByteArray_inj.1 hb'
    exact ⟨hp, hc⟩
  · exact fun ⟨hp, hc⟩ => canonicalJson_of hp hc

/-! ## A well-formed value that `canon` prints is printable -/

private theorem list_ok_of_mapM {α β ε : Type} {f : α → Except ε β} :
    ∀ {l : List α} {r : List β}, l.mapM f = .ok r → ∀ x ∈ l, ∃ y, f x = .ok y
  | [], _, _, x, hx => by simp at hx
  | a :: l, r, h, x, hx => by
    rw [List.mapM_cons] at h
    cases hb : f a with
    | error e => rw [hb] at h; cases h
    | ok b =>
      cases hl : l.mapM f with
      | error e => rw [hb] at h; simp only [bind, Except.bind] at h; rw [hl] at h; cases h
      | ok bs =>
        rcases List.mem_cons.1 hx with rfl | hx
        · exact ⟨b, hb⟩
        · exact list_ok_of_mapM hl x hx

private theorem array_ok_of_mapM {α β ε : Type} {f : α → Except ε β} {a : Array α} {r : Array β}
    (h : a.mapM f = .ok r) : ∀ x ∈ a, ∃ y, f x = .ok y := by
  rw [Array.mapM_eq_mapM_toList] at h
  cases hl : a.toList.mapM f with
  | error e => rw [hl] at h; cases h
  | ok l => exact fun x hx => list_ok_of_mapM hl x (Array.mem_toList_iff.2 hx)

/-- **A well-formed value that `canon` prints is printable**: `Flock.canon` returns only where every number, at
every depth, is an integer, so a value it prints whose objects are well-formed trees is `Claims.Json.Good`. -/
theorem good_of_canon : ∀ (j : Json), Wf j → ∀ s, Flock.canon j = .ok s → Good j
  | _, .null, _, _ => .null
  | _, .bool b, _, _ => .bool b
  | .num ⟨m, e⟩, .num _, _, h => by
    rw [Flock.canon] at h
    split at h
    · rename_i he
      obtain rfl : e = 0 := by simpa using he
      exact .num m
    · cases h
  | _, .str t, _, _ => .str t
  | .arr a, .arr hw, _, h => by
    rw [Flock.canon] at h
    cases hp : a.attach.mapM (fun x => Flock.canon x.1) with
    | error _ => rw [hp] at h; cases h
    | ok _ =>
      refine .arr fun x hx => ?_
      obtain ⟨y, hy⟩ := array_ok_of_mapM hp ⟨x, hx⟩ (Array.mem_attach _ _)
      exact good_of_canon x (hw x hx) y hy
  | .obj m, .obj hm hw, _, h => by
    rw [Flock.canon] at h
    cases hp : m.toArray.attach.mapM
        (fun p => do pure (p.1.1, ← Flock.canon p.1.2) : _ → Flock.V (String × String)) with
    | error _ => rw [hp] at h; cases h
    | ok _ =>
      refine .obj hm fun k v hkv => ?_
      have hmem : (k, v) ∈ m.toArray :=
        (Std.TreeMap.Raw.mem_toArray_iff_getElem?_eq_some hm).2
          (by rw [← Std.TreeMap.Raw.get?_eq_getElem?]; exact hkv)
      obtain ⟨y, hy⟩ := array_ok_of_mapM hp ⟨(k, v), hmem⟩ (Array.mem_attach _ _)
      cases hc : Flock.canon v with
      | error _ => rw [hc] at hy; cases hy
      | ok t => exact good_of_canon v (hw k v hkv) t hc

/-! ## What it accepts -/

/-- **What `canonicalJson` returns is printable**: every object a well-formed tree, every number an integer. -/
theorem canonicalJson_good {b : ByteArray} {j : Json} (h : Core.canonicalJson b = .ok j) : Good j := by
  obtain ⟨s, -, hp, hc⟩ := canonicalJson_ok h
  exact good_of_canon j (parse_wf s j hp) s hc

/-- **A printable value's canonical text is accepted, as a value alike it.** -/
theorem canonicalJson_of_canon {j : Json} {s : String} (hg : Good j) (hc : Flock.canon j = .ok s) :
    ∃ j', Core.canonicalJson s.toUTF8 = .ok j' ∧ Same j j' := by
  obtain ⟨j', hp, hs⟩ := parse_canon j s hg hc
  exact ⟨j', canonicalJson_of hp ((canon_same j j' hs).symm.trans hc), hs⟩

/-- **What `canonicalJson` accepts**: exactly the canonical texts of printable values. -/
theorem canonicalJson_accepts (b : ByteArray) :
    (∃ j, Core.canonicalJson b = .ok j) ↔ ∃ j s, Good j ∧ Flock.canon j = .ok s ∧ b = s.toUTF8 := by
  constructor
  · rintro ⟨j, h⟩
    obtain ⟨s, hb, -, hc⟩ := canonicalJson_ok h
    exact ⟨j, s, canonicalJson_good h, hc, hb⟩
  · rintro ⟨j, s, hg, hc, rfl⟩
    obtain ⟨j', h, -⟩ := canonicalJson_of_canon hg hc
    exact ⟨j', h⟩

/-- **What `canonicalJson` returns on a printable value's canonical text is alike that value.** -/
theorem canonicalJson_same {j₀ : Json} {s : String} (hg : Good j₀) (hc : Flock.canon j₀ = .ok s) {j : Json}
    (h : Core.canonicalJson s.toUTF8 = .ok j) : Same j₀ j := by
  obtain ⟨j', h', hs⟩ := canonicalJson_of_canon hg hc
  exact Except.ok.inj (h.symm.trans h') ▸ hs

end Verity.CanonicalJson
