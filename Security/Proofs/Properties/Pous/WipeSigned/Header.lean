import Proofs.Properties.Pous.WipeSigned.Json
import Proofs.Properties.Pous.MemoryChallenger.Sort
import Proofs.Properties.NetworkRecords.Bytes

/-!
# The wipe's signed header determines its commitment

`StreamedWipe.header` writes the commitment's fields, how the session ended, the fill's readings, the number of batches
and the egress as sorted JSON. The sort moves no field by its value (`header_eq`, through `Sort.qsort_map`). Each entry
is its key, a colon and its value, and each value is read back from the value followed by a comma or the closing brace
(`Sep`): a string or the tiers because what they write is a prefix code (`WipeJson`), a number because its characters
are digits and a sign. So the entries come back one by one (`entries_inj`), and with them every field (`header_inj`),
whatever characters the ending and the tiers' names hold.
-/

namespace MemoryChallenger.Properties.SecurityProofs.WipeHeader

open MemoryChallenger Flock WipeJson

/-- The header's keys, each with its place in `vals`. -/
def keys : List (String × Nat) :=
  [("protocol", 0), ("records_tree", 1), ("server", 2), ("session", 3), ("labels", 4), ("label_bytes", 5), ("k", 6),
   ("batch", 7), ("limit_ns", 8), ("tiers", 9), ("clock", 10), ("ended", 11), ("fill_start_ns", 12),
   ("fill_end_ns", 13), ("batches", 14), ("egress_bits", 15)]

/-- The header's values, in `keys`'s places. -/
def vals (c : StreamedWipe.Commitment) (e : String) (fs fe : Int) (n : Nat) (eg : Int) : List String :=
  c.headerFields.map (·.2) ++ [pyJsonStr e, toString fs, toString fe, toString n, toString eg]

/-- The order the header writes its keys in, which no field moves. -/
def order : Array (String × Nat) := keys.toArray.qsort fun x y => decide (x.1 < y.1)

def entry (c : StreamedWipe.Commitment) (e : String) (fs fe : Int) (n : Nat) (eg : Int) (x : String × Nat) : String :=
  pyJsonStr x.1 ++ ":" ++ (vals c e fs fe n eg)[x.2]!

theorem header_eq (c : StreamedWipe.Commitment) (e : String) (fs fe : Int) (n : Nat) (eg : Int) :
    StreamedWipe.header c e fs fe n eg = "{" ++ ",".intercalate (order.toList.map (entry c e fs fe n eg)) ++ "}" := by
  have hf : (c.headerFields ++ [("ended", pyJsonStr e), ("fill_start_ns", toString fs), ("fill_end_ns", toString fe),
      ("batches", toString n), ("egress_bits", toString eg)]).toArray =
      keys.toArray.map fun x => (x.1, (vals c e fs fe n eg)[x.2]!) := by
    simp [keys, vals, StreamedWipe.Commitment.headerFields]
  unfold StreamedWipe.header
  simp only
  rw [hf, Sort.qsort_map, Array.toList_map, List.map_map]
  rfl

/-- A value `v` (or `v'`), followed by a comma or the closing brace, is read back with what follows it. -/
def Sep (v v' : List Char) : Prop :=
  ∀ {c c' : Char} {r r' : List Char}, (c = ',' ∨ c = '}') → (c' = ',' ∨ c' = '}') → v ++ c :: r = v' ++ c' :: r' →
    v = v' ∧ c :: r = c' :: r'

theorem sep_refl (v : List Char) : Sep v v := fun _ _ e => ⟨rfl, List.append_cancel_left e⟩

theorem sep_str (s s' : String) : Sep (pyJsonStr s).toList (pyJsonStr s').toList := fun _ _ e => by
  obtain ⟨h1, h2⟩ := pyJsonStr_split e
  exact ⟨by rw [h1], h2⟩

theorem sep_tiers (t t' : Array (String × Nat)) : Sep (tiersJson t).toList (tiersJson t').toList := fun _ _ e => by
  obtain ⟨h1, h2⟩ := tiers_split e
  exact ⟨by rw [h1], h2⟩

theorem sep_stop {v v' : List Char} (h : ∀ x ∈ v, x.isDigit = true ∨ x = '-')
    (h' : ∀ x ∈ v', x.isDigit = true ∨ x = '-') : Sep v v' := fun hc hc' e => by
  have stop : ∀ x : Char, x.isDigit = true ∨ x = '-' → ¬ (x = ',' ∨ x = '}') := by
    intro x hx hs
    rcases hs with rfl | rfl <;> rcases hx with hx | hx <;> revert hx <;> decide
  exact split_stop (P := fun x => x = ',' ∨ x = '}') (fun x m => stop x (h x m)) (fun x m => stop x (h' x m)) hc hc' e

theorem sep_nat (n n' : Nat) : Sep (toString n).toList (toString n').toList :=
  sep_stop (fun _ m => Or.inl (digit_of_mem m)) (fun _ m => Or.inl (digit_of_mem m))

theorem sep_int (x x' : Int) : Sep (toString x).toList (toString x').toList :=
  sep_stop (fun _ m => intStr_mem m) (fun _ m => intStr_mem m)

/-- **Entries read back**: from `[e₁, …, eₙ]` written between commas and ending at the closing brace, each entry being
a key part `kp x` then a value that `Sep` reads back, the values are read back. -/
theorem entries_inj {α : Type} (kp f f' : α → List Char) :
    ∀ (L : List α), L ≠ [] → (∀ x ∈ L, Sep (f x) (f' x)) →
      [','].intercalate (L.map fun x => kp x ++ f x) ++ ['}'] =
        [','].intercalate (L.map fun x => kp x ++ f' x) ++ ['}'] →
      ∀ x ∈ L, f x = f' x
  | [], h, _, _ => absurd rfl h
  | [x], _, hs, e => by
    simp only [List.map_cons, List.map_nil, List.intercalate_singleton, List.append_assoc] at e
    have e := List.append_cancel_left e
    obtain ⟨h1, -⟩ := hs x List.mem_cons_self (Or.inr rfl) (Or.inr rfl) e
    intro y hy
    rw [List.mem_singleton] at hy
    subst hy
    exact h1
  | x :: y :: L, _, hs, e => by
    rw [List.map_cons, List.map_cons, List.intercalate_cons_cons, List.map_cons, List.map_cons,
      List.intercalate_cons_cons] at e
    simp only [List.append_assoc, List.singleton_append] at e
    have e := List.append_cancel_left e
    obtain ⟨h1, h2⟩ := hs x List.mem_cons_self (Or.inl rfl) (Or.inl rfl) e
    have ih := entries_inj kp f f' (y :: L) (List.cons_ne_nil _ _) (fun z hz => hs z (List.mem_cons_of_mem _ hz))
      (List.cons.inj h2).2
    intro z hz
    rcases List.mem_cons.mp hz with rfl | hz
    · exact h1
    · exact ih z hz

theorem order_keys {x : String × Nat} (hx : x ∈ order.toList) : x ∈ keys :=
  List.mem_toArray.mp (Sort.qsort_mem _ (Array.mem_toList_iff.mp hx))

theorem keys_order {x : String × Nat} (hx : x ∈ keys) : x ∈ order.toList :=
  Array.mem_toList_iff.mpr (Sort.mem_qsort _ (List.mem_toArray.mpr hx))

theorem sep_vals (c c' : StreamedWipe.Commitment) (e e' : String) (fs fs' fe fe' : Int) (n n' : Nat) (eg eg' : Int) :
    ∀ x ∈ keys, Sep ((vals c e fs fe n eg)[x.2]!).toList ((vals c' e' fs' fe' n' eg')[x.2]!).toList := by
  intro x hx
  simp only [keys, List.mem_cons, List.not_mem_nil, or_false] at hx
  rcases hx with h | h | h | h | h | h | h | h | h | h | h | h | h | h | h | h <;> subst h
  · exact sep_refl (pyJsonStr StreamedWipe.PROTOCOL).toList
  · exact sep_refl (pyJsonStr StreamedWipe.RECORDS_TREE).toList
  · exact sep_str (hexOf c.server) (hexOf c'.server)
  · exact sep_str (hexOf c.session) (hexOf c'.session)
  · exact sep_nat c.params.labels c'.params.labels
  · exact sep_refl (toString StreamedWipe.LABEL_BYTES).toList
  · exact sep_nat c.params.k c'.params.k
  · exact sep_nat c.params.batch c'.params.batch
  · exact sep_nat c.params.limitNs c'.params.limitNs
  · exact sep_tiers c.params.tiers c'.params.tiers
  · exact sep_refl (pyJsonStr CLOCK).toList
  · exact sep_str e e'
  · exact sep_int fs fs'
  · exact sep_int fe fe'
  · exact sep_nat n n'
  · exact sep_int eg eg'

/-- The header's values are read back from it. -/
theorem vals_inj {c c' : StreamedWipe.Commitment} {e e' : String} {fs fs' fe fe' : Int} {n n' : Nat} {eg eg' : Int}
    (h : StreamedWipe.header c e fs fe n eg = StreamedWipe.header c' e' fs' fe' n' eg') :
    ∀ x ∈ keys, ((vals c e fs fe n eg)[x.2]!).toList = ((vals c' e' fs' fe' n' eg')[x.2]!).toList := by
  rw [header_eq, header_eq] at h
  have hq := congrArg String.toList h
  simp only [String.toList_append, String.toList_intercalate, List.map_map] at hq
  have hb : "{".toList = ['{'] := rfl
  have hc : ",".toList = [','] := rfl
  have he : "}".toList = ['}'] := rfl
  have hfun : ∀ (c : StreamedWipe.Commitment) (e : String) (fs fe : Int) (n : Nat) (eg : Int),
      (String.toList ∘ entry c e fs fe n eg) =
        fun x => ((pyJsonStr x.1).toList ++ [':']) ++ ((vals c e fs fe n eg)[x.2]!).toList := by
    intro c e fs fe n eg
    funext x
    simp only [Function.comp, entry, String.toList_append]
    rfl
  rw [hb, hc, he, List.append_assoc ['{'], List.append_assoc ['{'], hfun, hfun] at hq
  have hq := List.append_cancel_left hq
  have hne : order.toList ≠ [] := List.ne_nil_of_mem (keys_order (x := ("protocol", 0)) (by simp [keys]))
  have hs : ∀ y ∈ order.toList,
      Sep ((vals c e fs fe n eg)[y.2]!).toList ((vals c' e' fs' fe' n' eg')[y.2]!).toList :=
    fun y hy => sep_vals c c' e e' fs fs' fe fe' n n' eg eg' y (order_keys hy)
  have hv := entries_inj (fun x => (pyJsonStr x.1).toList ++ [':']) (fun x => ((vals c e fs fe n eg)[x.2]!).toList)
    (fun x => ((vals c' e' fs' fe' n' eg')[x.2]!).toList) order.toList hne
  have hv := hv hs
  have hv := hv hq
  exact fun x hx => hv x (keys_order hx)

theorem commitment_ext {c c' : StreamedWipe.Commitment} (h1 : c.server = c'.server) (h2 : c.session = c'.session)
    (h3 : c.params.labels = c'.params.labels) (h4 : c.params.k = c'.params.k) (h5 : c.params.batch = c'.params.batch)
    (h6 : c.params.limitNs = c'.params.limitNs) (h7 : c.params.tiers = c'.params.tiers) : c = c' := by
  obtain ⟨s, se, ⟨l, k, b, lim, t⟩⟩ := c
  obtain ⟨s', se', ⟨l', k', b', lim', t'⟩⟩ := c'
  simp only at h1 h2 h3 h4 h5 h6 h7
  subst h1 h2 h3 h4 h5 h6 h7
  rfl

/-- **The wipe's header determines its commitment**, how it ended, the fill's readings, its number of batches and its
egress. -/
theorem header_inj {c c' : StreamedWipe.Commitment} {e e' : String} {fs fs' fe fe' : Int} {n n' : Nat} {eg eg' : Int}
    (h : StreamedWipe.header c e fs fe n eg = StreamedWipe.header c' e' fs' fe' n' eg') :
    c = c' ∧ e = e' ∧ fs = fs' ∧ fe = fe' ∧ n = n' ∧ eg = eg' := by
  have v := vals_inj h
  have v2 := v ("server", 2) (by simp [keys])
  have v3 := v ("session", 3) (by simp [keys])
  have v4 := v ("labels", 4) (by simp [keys])
  have v6 := v ("k", 6) (by simp [keys])
  have v7 := v ("batch", 7) (by simp [keys])
  have v8 := v ("limit_ns", 8) (by simp [keys])
  have v9 := v ("tiers", 9) (by simp [keys])
  have v11 := v ("ended", 11) (by simp [keys])
  have v12 := v ("fill_start_ns", 12) (by simp [keys])
  have v13 := v ("fill_end_ns", 13) (by simp [keys])
  have v14 := v ("batches", 14) (by simp [keys])
  have v15 := v ("egress_bits", 15) (by simp [keys])
  change (pyJsonStr (hexOf c.server)).toList = (pyJsonStr (hexOf c'.server)).toList at v2
  change (pyJsonStr (hexOf c.session)).toList = (pyJsonStr (hexOf c'.session)).toList at v3
  change (toString c.params.labels).toList = (toString c'.params.labels).toList at v4
  change (toString c.params.k).toList = (toString c'.params.k).toList at v6
  change (toString c.params.batch).toList = (toString c'.params.batch).toList at v7
  change (toString c.params.limitNs).toList = (toString c'.params.limitNs).toList at v8
  change (tiersJson c.params.tiers).toList = (tiersJson c'.params.tiers).toList at v9
  change (pyJsonStr e).toList = (pyJsonStr e').toList at v11
  change (toString fs).toList = (toString fs').toList at v12
  change (toString fe).toList = (toString fe').toList at v13
  change (toString n).toList = (toString n').toList at v14
  change (toString eg).toList = (toString eg').toList at v15
  have hx {a b : ByteArray} (h : (pyJsonStr (hexOf a)).toList = (pyJsonStr (hexOf b)).toList) : a = b :=
    Properties.NetworkRecords.Bytes.hexOf_inj (pyJsonStr_inj (String.toList_inj.mp h))
  have ht : c.params.tiers = c'.params.tiers :=
    (tiers_split (r := []) (r' := []) (by rw [List.append_nil, List.append_nil]; exact v9)).1
  exact ⟨commitment_ext (hx v2) (hx v3) (natStr_inj v4) (natStr_inj v6) (natStr_inj v7) (natStr_inj v8) ht,
    pyJsonStr_inj (String.toList_inj.mp v11), intStr_inj v12, intStr_inj v13, natStr_inj v14, intStr_inj v15⟩

end MemoryChallenger.Properties.SecurityProofs.WipeHeader
