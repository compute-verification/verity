import MemoryChallenger.Records
import Proofs.Properties.NetworkRecords.Bytes
import Proofs.Properties.Pous.MemoryChallenger.Sort

/-!
# The signed header determines the commitment

`header` writes the commitment's fields, how the session ended and its number of rounds as sorted JSON. The sort moves
no field by its value (`header_eq`, through `Sort.qsort_map`), each entry is free of commas when the ending is `Plain`, so
the entries come back by splitting at commas (`vals_inj`), and each value is read back from its entry (`header_inj`):
a hex string through `hexOf_inj`, a number through its decimal digits.
-/

namespace MemoryChallenger.Properties.SecurityProofs.Header

open MemoryChallenger Flock

/-- A character `pyJsonStr` writes as it is, and not a comma. -/
def plain (c : Char) : Bool := 0x20 ≤ c.toNat && c.toNat ≤ 0x7e && c != '"' && c != '\\' && c != ','

theorem foldl_plain {f : String → Char → String} (hf : ∀ b c, plain c = true → f b c = b.push c) :
    ∀ (l : List Char) (out : String), (∀ c ∈ l, plain c = true) → (l.foldl f out).toList = out.toList ++ l
  | [], out, _ => by simp
  | c :: l, out, h => by
    rw [List.foldl_cons, foldl_plain hf l _ (fun x hx => h x (List.mem_cons_of_mem _ hx)),
      hf _ _ (h c (List.mem_cons_self ..)), String.toList_push]
    simp

theorem pyJsonStr_toList (s : String) (h : ∀ c ∈ s.toList, plain c = true) :
    (pyJsonStr s).toList = '"' :: s.toList ++ ['"'] := by
  unfold pyJsonStr
  simp only [Id.run]
  rw [List.forIn_pure_yield_eq_foldl]
  show (String.toList (List.foldl _ "\"" s.toList ++ "\"")) = _
  rw [String.toList_append, foldl_plain _ _ _ h]
  · simp
  intro b c hc
  simp only [plain, Bool.and_eq_true, decide_eq_true_eq, bne_iff_ne, ne_eq] at hc
  split
  any_goals simp at hc
  have h8 : c.toNat ≠ 8 := by omega
  have h12 : c.toNat ≠ 12 := by omega
  simp [h8, h12, hc.1.1.1.1, hc.1.1.1.2]

/-- The header's keys, each with its place in `vals`. -/
def keys : List (String × Nat) :=
  [("binding", 0), ("block_bytes", 1), ("blocks", 2), ("clock", 3), ("coin_key_sha512", 4), ("coin_root", 5),
   ("coin_tree", 6), ("k", 7), ("limit_ns", 8), ("protocol", 9), ("records_tree", 10), ("session", 11), ("ended", 12),
   ("rounds", 13)]

/-- The header's values, in `keys`'s places. -/
def vals (c : SessionCommitment) (e : String) (n : Nat) : List String :=
  c.headerFields.map (·.2) ++ [pyJsonStr e, toString n]

/-- The order the header writes its keys in, which no field moves. -/
def order : Array (String × Nat) := keys.toArray.qsort fun x y => decide (x.1 < y.1)

def entry (c : SessionCommitment) (e : String) (n : Nat) (x : String × Nat) : String :=
  pyJsonStr x.1 ++ ":" ++ (vals c e n)[x.2]!

theorem header_eq (c : SessionCommitment) (e : String) (n : Nat) :
    header c e n = "{" ++ ",".intercalate (order.toList.map (entry c e n)) ++ "}" := by
  have hf : (c.headerFields ++ [("ended", pyJsonStr e), ("rounds", toString n)]).toArray =
      keys.toArray.map fun x => (x.1, (vals c e n)[x.2]!) := by
    simp [keys, vals, SessionCommitment.headerFields]
  unfold header
  simp only
  rw [hf, Sort.qsort_map, Array.toList_map, List.map_map]
  rfl

/-- Every character of the string is `plain`. -/
def Plain (s : String) : Prop := ∀ c ∈ s.toList, plain c = true

theorem plain_hex_nat : ∀ n < 256, ∀ c ∈ (Flock.hexOf ⟨#[n.toUInt8]⟩).toList, plain c = true := by decide +kernel

theorem plain_hexOf (b : ByteArray) : Plain (Flock.hexOf b) := by
  intro c hc
  rw [Properties.NetworkRecords.Bytes.hexOf_toList, List.mem_flatMap] at hc
  obtain ⟨x, -, hx⟩ := hc
  exact plain_hex_nat x.toNat x.toNat_lt c (by simpa using hx)

theorem plain_toString (n : Nat) : Plain (toString n) := by
  intro c hc
  rw [Nat.toString_eq_repr, Nat.toList_repr] at hc
  have := Nat.isDigit_of_mem_toDigits (by decide) (by decide) hc
  simp only [Char.isDigit, Bool.and_eq_true, decide_eq_true_eq, UInt32.le_iff_toNat_le] at this
  have h1 : 48 ≤ c.toNat := this.1
  have h2 : c.toNat ≤ 57 := this.2
  simp only [plain, Bool.and_eq_true, decide_eq_true_eq, bne_iff_ne, ne_eq]
  refine ⟨⟨⟨⟨by omega, by omega⟩, ?_⟩, ?_⟩, ?_⟩ <;> intro h <;> subst h <;> simp at h1 h2

theorem toString_inj {n n' : Nat} (h : toString n = toString n') : n = n' := by
  have e := congrArg String.toList h
  rw [Nat.toString_eq_repr, Nat.toString_eq_repr, Nat.toList_repr, Nat.toList_repr] at e
  rw [← Nat.ofDigitChars_ten_toDigits (n := n), e, Nat.ofDigitChars_ten_toDigits]

theorem pyJsonStr_inj {s s' : String} (hs : Plain s) (hs' : Plain s') (h : pyJsonStr s = pyJsonStr s') : s = s' := by
  have e := congrArg String.toList h
  rw [pyJsonStr_toList s hs, pyJsonStr_toList s' hs'] at e
  simp only [List.cons.injEq, List.append_cancel_right_eq, true_and] at e
  exact String.toList_inj.mp e

theorem comma_plain {s : String} (hs : Plain s) : ',' ∉ s.toList := fun h => by
  have := hs _ h
  simp [plain] at this

theorem comma_pyJsonStr {s : String} (hs : Plain s) : ',' ∉ (pyJsonStr s).toList := by
  rw [pyJsonStr_toList s hs]
  simpa using comma_plain hs

theorem plain_const : Plain CLOCK ∧ Plain COIN_TREE ∧ Plain PROTOCOL ∧ Plain RECORDS_TREE := by
  refine ⟨?_, ?_, ?_, ?_⟩ <;> intro c hc <;> revert c <;> decide

theorem plain_keys : ∀ x ∈ keys, Plain x.1 := by
  intro x hx c hc
  revert c
  simp only [keys, List.mem_cons, List.not_mem_nil, or_false] at hx
  rcases hx with h | h | h | h | h | h | h | h | h | h | h | h | h | h <;> subst h <;> decide

theorem comma_vals (c : SessionCommitment) {e : String} (he : Plain e) (n : Nat) :
    ∀ v ∈ vals c e n, ',' ∉ v.toList := by
  have hc := plain_const
  intro v hv
  simp only [vals, SessionCommitment.headerFields, List.map_cons, List.map_nil, List.cons_append, List.nil_append,
    List.mem_cons, List.not_mem_nil, or_false] at hv
  rcases hv with h | h | h | h | h | h | h | h | h | h | h | h | h | h <;> subst h
  · exact comma_pyJsonStr (plain_hexOf _)
  · exact comma_plain (plain_toString _)
  · exact comma_plain (plain_toString _)
  · exact comma_pyJsonStr hc.1
  · exact comma_pyJsonStr (plain_hexOf _)
  · exact comma_pyJsonStr (plain_hexOf _)
  · exact comma_pyJsonStr hc.2.1
  · exact comma_plain (plain_toString _)
  · exact comma_plain (plain_toString _)
  · exact comma_pyJsonStr hc.2.2.1
  · exact comma_pyJsonStr hc.2.2.2
  · exact comma_pyJsonStr (plain_hexOf _)
  · exact comma_pyJsonStr he
  · exact comma_plain (plain_toString _)

theorem comma_entry (c : SessionCommitment) {e : String} (he : Plain e) (n : Nat) {x : String × Nat} (hx : x ∈ keys) :
    ',' ∉ (entry c e n x).toList := by
  simp only [entry, String.toList_append, List.mem_append, not_or]
  refine ⟨⟨comma_pyJsonStr (plain_keys x hx), by decide⟩, ?_⟩
  rw [getElem!_def]
  split
  · exact comma_vals c he n _ (List.mem_of_getElem? ‹_›)
  · simp

theorem order_keys {x : String × Nat} (hx : x ∈ order.toList) : x ∈ keys :=
  List.mem_toArray.mp (Sort.qsort_mem _ (Array.mem_toList_iff.mp hx))

theorem keys_order {x : String × Nat} (hx : x ∈ keys) : x ∈ order.toList :=
  Array.mem_toList_iff.mpr (Sort.mem_qsort _ (List.mem_toArray.mpr hx))

/-- The header's values are read back from it, for plain endings. -/
theorem vals_inj {c c' : SessionCommitment} {e e' : String} {n n' : Nat} (he : Plain e) (he' : Plain e')
    (h : header c e n = header c' e' n') : ∀ x ∈ keys, (vals c e n)[x.2]! = (vals c' e' n')[x.2]! := by
  rw [header_eq, header_eq] at h
  have hq := congrArg String.toList h
  simp only [String.toList_append] at hq
  rw [List.append_assoc, List.append_assoc] at hq
  have hq := List.append_cancel_right (List.append_cancel_left hq)
  rw [String.toList_intercalate, String.toList_intercalate] at hq
  have hne : order.toList ≠ [] := by
    have : ("binding", 0) ∈ order := Sort.mem_qsort _ (by simp [keys])
    intro h0
    rw [← Array.mem_toList_iff, h0] at this
    simp at this
  have hq := congrArg (List.splitOn ',') hq
  have hcomma : ",".toList = [','] := by simp
  rw [hcomma, List.splitOn_intercalate _ ?_ (by simpa using hne),
    List.splitOn_intercalate _ ?_ (by simpa using hne)] at hq
  rotate_left
  · intro l hl
    rw [List.map_map, List.mem_map] at hl
    obtain ⟨x, hx, rfl⟩ := hl
    exact comma_entry c' he' n' (order_keys hx)
  · intro l hl
    rw [List.map_map, List.mem_map] at hl
    obtain ⟨x, hx, rfl⟩ := hl
    exact comma_entry c he n (order_keys hx)
  simp only [List.map_map] at hq
  intro x hx
  have := (List.map_inj_left.mp hq) x (keys_order hx)
  simp only [Function.comp, entry, String.toList_append, List.append_assoc] at this
  exact String.toList_inj.mp (List.append_cancel_left (List.append_cancel_left this))

/-- **A header determines the commitment it writes**, up to the coin key's hash, with how the session ended (if plain)
and its number of rounds. -/
theorem header_inj {c c' : SessionCommitment} {e e' : String} {n n' : Nat} (he : Plain e) (he' : Plain e')
    (h : header c e n = header c' e' n') :
    c.binding = c'.binding ∧ c.blockBytes = c'.blockBytes ∧ c.blocks = c'.blocks ∧
      Sha512.hash c.coinKey = Sha512.hash c'.coinKey ∧ c.coinRoot = c'.coinRoot ∧ c.k = c'.k ∧
      c.limitNs = c'.limitNs ∧ c.session = c'.session ∧ e = e' ∧ n = n' := by
  have v := vals_inj he he' h
  have v0 := v ("binding", 0) (by simp [keys])
  have v1 := v ("block_bytes", 1) (by simp [keys])
  have v2 := v ("blocks", 2) (by simp [keys])
  have v4 := v ("coin_key_sha512", 4) (by simp [keys])
  have v5 := v ("coin_root", 5) (by simp [keys])
  have v7 := v ("k", 7) (by simp [keys])
  have v8 := v ("limit_ns", 8) (by simp [keys])
  have v11 := v ("session", 11) (by simp [keys])
  have v12 := v ("ended", 12) (by simp [keys])
  have v13 := v ("rounds", 13) (by simp [keys])
  change pyJsonStr (hexOf c.binding) = pyJsonStr (hexOf c'.binding) at v0
  change toString c.blockBytes = toString c'.blockBytes at v1
  change toString c.blocks = toString c'.blocks at v2
  change pyJsonStr (hexOf (Sha512.hash c.coinKey)) = pyJsonStr (hexOf (Sha512.hash c'.coinKey)) at v4
  change pyJsonStr (hexOf c.coinRoot) = pyJsonStr (hexOf c'.coinRoot) at v5
  change toString c.k = toString c'.k at v7
  change toString c.limitNs = toString c'.limitNs at v8
  change pyJsonStr (hexOf c.session) = pyJsonStr (hexOf c'.session) at v11
  change pyJsonStr e = pyJsonStr e' at v12
  change toString n = toString n' at v13
  have hx {a b : ByteArray} (h : pyJsonStr (hexOf a) = pyJsonStr (hexOf b)) : a = b :=
    Properties.NetworkRecords.Bytes.hexOf_inj (pyJsonStr_inj (plain_hexOf a) (plain_hexOf b) h)
  exact ⟨hx v0, toString_inj v1, toString_inj v2, hx v4, hx v5, toString_inj v7, toString_inj v8, hx v11,
    pyJsonStr_inj he he' v12, toString_inj v13⟩

theorem plain_complete : Plain COMPLETE := by intro c hc; revert c; decide

theorem plain_late : Plain LATE := by intro c hc; revert c; decide

end MemoryChallenger.Properties.SecurityProofs.Header
