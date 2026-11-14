module

public import Proofs.Properties.Challenger.Entry
public import Proofs.Flock.Bytes
import all Init.Data.String.Defs
import all Init.Data.Ord.String
import all Lean.Data.Json.Basic
import all Init.Data.Repr
import all Init.Data.Int.Repr

/-!
# An object built from a list: its keys, its reads, its text and a digest's hex

What an honest party writes is an object built from a list of increasing keys (`Json.mkObj`), its values strings,
numbers, digests in hex and objects built the same way. A proof that the reader accepts it needs what each of the
reader's checks sees:

* its pairs are the list (`toList_mkObj`), so its keys are the list's (`keys?_mkObj`) and a read of a key is the
  list's lookup (`getD_mkObj`, `field?_mkObj`);
* `Flock.canon` writes it as its items' text, each key between quotes and its value's text (`canon_mkObj`,
  `objText`), no longer than its braces, its keys, its values' texts and four characters a key
  (`canon_mkObj_le`, with `canon_str_le` and `canon_num` for its leaves);
* a hex string is plain (`plain_hexOf`), twice its bytes long (`hexOf_len`), and a digest of that length
  (`isDigest_hex`);
* a tagged hash of a printable value is a digest (`tagged_ok`), as a value's well-formedness gives its keys in
  order (`wf_of_good`).

`PlainStr` is decided through `List.all`, so `decide` settles it for a literal.
-/

@[expose] public section
set_option backward.privateInPublic true
set_option backward.privateInPublic.warn false

open private isDigest from Spot.ChallengerSession

namespace Properties.Challenger.Built

open Lean (Json)
open _root_.Spot _root_.Spot.Challenger
open Claims.Json (Good Wf)
open FlockComplete.JsonRT (get?_mkObj keys_of isSome_get?_mkObj)
open Plain (PlainChar PlainStr canonStr_plain)

/-! ## Plain strings, decided -/

instance instDecidablePlainChar (c : Char) : Decidable (PlainChar c) :=
  inferInstanceAs (Decidable (' ' ≤ c ∧ c ≤ '~' ∧ c ≠ '"' ∧ c ≠ '\\'))

theorem plainStr_iff (s : String) : PlainStr s ↔ s.toList.all (fun c => decide (PlainChar c)) = true := by
  simp only [PlainStr, List.all_eq_true, decide_eq_true_eq]

instance instDecidablePlainStr (s : String) : Decidable (PlainStr s) := decidable_of_iff _ (plainStr_iff s).symm

/-! ## Hex -/

theorem plain_hexByte : (List.range 256).all (fun n =>
    (Flock.hexOf ⟨#[n.toUInt8]⟩).toList.all fun c => decide (PlainChar c)) = true := by decide +kernel

/-- **A hex string is plain.** -/
theorem plain_hexOf (b : ByteArray) : PlainStr (Flock.hexOf b) := by
  intro c hc
  rw [Properties.NetworkRecords.Bytes.hexOf_toList, List.mem_flatMap] at hc
  obtain ⟨x, -, hx⟩ := hc
  have hb := List.all_eq_true.1 plain_hexByte x.toNat (List.mem_range.2 x.toNat_lt)
  exact of_decide_eq_true (List.all_eq_true.1 hb c (by simpa using hx))

theorem hexOf_len (b : ByteArray) : (Flock.hexOf b).length = 2 * b.size := by
  rw [← String.length_toList, Properties.NetworkRecords.Bytes.hexOf_toList, List.length_flatMap]
  simp only [Properties.NetworkRecords.Bytes.hexOf_length, List.map_const', List.sum_replicate_nat,
    Array.length_toList, ByteArray.size]
  omega

/-- **A digest's hex is a digest**, of twice its bytes' count in characters. -/
theorem isDigest_hex {b : ByteArray} {n : Nat} (h : 2 * b.size = n) : isDigest (.str (Flock.hexOf b)) n = true := by
  show ((Flock.hexOf b).length == n && (Flock.unhex (Flock.hexOf b)).toOption.isSome) = true
  rw [hexOf_len, Flock.Hex.unhex_hexOf, h]
  simp [Except.toOption]

/-! ## A built object's pairs, keys, reads and text -/

/-- **An object built from a list of increasing keys lists it back.** -/
theorem toList_mkObj {l : List (String × Json)} (hs : (l.map (·.1)).Pairwise (· < ·)) :
    (Std.TreeMap.Raw.ofList l compare).toList = l := by
  have hwf : (Std.TreeMap.Raw.ofList l compare).WF := Std.TreeMap.Raw.WF.ofList
  have hd := CanonRead.distinct_of_keys hs
  have hkeys : (Std.TreeMap.Raw.ofList l compare).keys = l.map (·.1) :=
    keys_of hwf (hs.imp fun h => compareOfLessAndEq_eq_lt.2 h) fun k => isSome_get?_mkObj hd k
  have hfst : (Std.TreeMap.Raw.ofList l compare).toList.map Prod.fst = l.map (·.1) := by
    rw [Std.TreeMap.Raw.map_fst_toList_eq_keys, hkeys]
  have hlen : (Std.TreeMap.Raw.ofList l compare).toList.length = l.length := by
    have := congrArg List.length hfst
    rwa [List.length_map, List.length_map] at this
  apply List.ext_getElem hlen
  intro i h1 h2
  have hk : ((Std.TreeMap.Raw.ofList l compare).toList[i]).1 = (l[i]).1 := by
    have := congrArg (fun x => x[i]?) hfst
    simp only [List.getElem?_map, List.getElem?_eq_getElem h1, List.getElem?_eq_getElem h2, Option.map_some,
      Option.some.injEq] at this
    exact this
  have hv := (Std.TreeMap.Raw.mem_toList_iff_getElem?_eq_some hwf
    (k := ((Std.TreeMap.Raw.ofList l compare).toList[i]).1) (v := ((Std.TreeMap.Raw.ofList l compare).toList[i]).2)).1
    (List.getElem_mem h1)
  have hv' := get?_mkObj hd (k := (l[i]).1) (v := (l[i]).2) (List.getElem_mem h2)
  rw [Std.TreeMap.Raw.get?_eq_getElem?, ← hk, hv] at hv'
  exact Prod.ext hk (Option.some.inj hv')

/-- The text of an object's items, each a key and its value's text. -/
def objText (items : List (String × String)) : String :=
  "{" ++ ",".intercalate (items.map fun kv => Flock.canonStr kv.1 ++ ":" ++ kv.2) ++ "}"

theorem mapM_pairs {l : List (String × Json)} {ts : List String}
    (hv : List.Forall₂ (fun p t => Flock.canon p.2 = .ok t) l ts) :
    l.mapM (fun p => do pure (p.1, ← Flock.canon p.2) : String × Json → Flock.V (String × String)) =
      .ok (List.zipWith (fun p t => (p.1, t)) l ts) := by
  induction hv with
  | nil => rfl
  | cons h _ ih =>
    rw [List.mapM_cons, h, ih]
    rfl

theorem zipWith_keys {l : List (String × Json)} {ts : List String}
    (hv : List.Forall₂ (fun p t => Flock.canon p.2 = .ok t) l ts) :
    (List.zipWith (fun p t => (p.1, t)) l ts).map (·.1) = l.map (·.1) := by
  induction hv with
  | nil => rfl
  | cons _ _ ih => simp only [List.zipWith_cons_cons, List.map_cons, ih]

/-- **`canon` writes an object of increasing keys as its items' text**, each value's text in the list's order. -/
theorem canon_mkObj {l : List (String × Json)} {ts : List String} (hs : (l.map (·.1)).Pairwise (· < ·))
    (hv : List.Forall₂ (fun p t => Flock.canon p.2 = .ok t) l ts) :
    Flock.canon (Json.mkObj l) = .ok (objText (List.zipWith (fun p t => (p.1, t)) l ts)) := by
  have hq : (List.zipWith (fun p t => (p.1, t)) l ts).toArray.qsort (fun a b => a.1 < b.1) =
      (List.zipWith (fun p t => (p.1, t)) l ts).toArray :=
    CanonRead.qsort_keys (by rw [zipWith_keys hv]; exact hs)
  rw [Json.mkObj, CanonLine.canon_obj, Array.mapM_eq_mapM_toList, Std.TreeMap.Raw.toList_toArray, toList_mkObj hs,
    mapM_pairs hv]
  show Except.ok _ = Except.ok _
  rw [hq]
  rfl


/-- **An object built from increasing keys has those keys.** -/
theorem keys?_mkObj {l : List (String × Json)} (hs : (l.map (·.1)).Pairwise (· < ·)) :
    keys? (Json.mkObj l) = some (l.map (·.1)).toArray := by
  have h : (Std.TreeMap.Raw.ofList l compare).toArray.map (·.1) = (l.map (·.1)).toArray := by
    apply Array.toList_inj.1
    rw [Array.toList_map, Std.TreeMap.Raw.toList_toArray, toList_mkObj hs]
  show some (sortStrings _) = _
  rw [h, sortStrings, Entry.qsort_strs hs]

theorem getD_mkObj {l : List (String × Json)} (hs : (l.map (·.1)).Pairwise (· < ·)) (k : String) :
    getD (Json.mkObj l) k = (l.lookup k).getD .null := by
  rw [Json.mkObj, CanonRead.getD_obj, CanonRead.get?_ofList hs]

theorem field?_mkObj {l : List (String × Json)} (hs : (l.map (·.1)).Pairwise (· < ·)) (k : String) :
    field? (Json.mkObj l) k = l.lookup k := by
  rw [Json.mkObj, CanonLine.field?_obj, CanonRead.get?_ofList hs]

theorem intercalate_length_le (xs : List String) :
    (",".intercalate xs).length ≤ (xs.map fun x => x.length + 1).sum := by
  induction xs with
  | nil => exact Nat.le_refl _
  | cons x xs ih =>
    cases xs with
    | nil =>
      simp only [String.intercalate_singleton, List.map_cons, List.map_nil, List.sum_cons, List.sum_nil]
      omega
    | cons y ys =>
      rw [String.intercalate_cons_cons]
      simp only [String.length_append, List.map_cons, List.sum_cons] at ih ⊢
      have h1 : ",".length = 1 := rfl
      omega

/-- An object's text is its braces and, for each item, its key's text, a colon, its value's text and at most one
comma. -/
theorem objText_length_le (items : List (String × String)) :
    (objText items).length ≤ 2 + (items.map fun kv => (Flock.canonStr kv.1).length + kv.2.length + 2).sum := by
  have h := intercalate_length_le (items.map fun kv => Flock.canonStr kv.1 ++ ":" ++ kv.2)
  have hm : ((items.map fun kv => Flock.canonStr kv.1 ++ ":" ++ kv.2).map fun x => x.length + 1) =
      items.map fun kv => (Flock.canonStr kv.1).length + kv.2.length + 2 := by
    rw [List.map_map]
    refine List.map_congr_left fun kv _ => ?_
    have h1 : ":".length = 1 := rfl
    simp only [Function.comp_apply, String.length_append, h1]
    omega
  rw [hm] at h
  have h1 : "{".length = 1 := rfl
  have h2 : "}".length = 1 := rfl
  unfold objText
  simp only [String.length_append, h1, h2]
  omega

theorem texts_of {l : List (String × Json)} {bs : List Nat}
    (hv : List.Forall₂ (fun p b => ∃ t, Flock.canon p.2 = .ok t ∧ t.length ≤ b) l bs)
    (hk : ∀ k ∈ l.map (·.1), PlainStr k) :
    ∃ ts, List.Forall₂ (fun p t => Flock.canon p.2 = .ok t) l ts ∧
      ((List.zipWith (fun p t => (p.1, t)) l ts).map fun kv => (Flock.canonStr kv.1).length + kv.2.length + 2).sum ≤
        ((l.map (·.1)).map (·.length + 4)).sum + bs.sum := by
  induction hv with
  | nil => exact ⟨[], .nil, Nat.le_refl _⟩
  | cons h _ ih =>
    obtain ⟨t, ht, hb⟩ := h
    obtain ⟨ts, hts, hsum⟩ := ih fun k hk' => hk k (List.mem_cons_of_mem _ hk')
    refine ⟨t :: ts, .cons ht hts, ?_⟩
    have h1 : "\"".length = 1 := rfl
    simp only [List.zipWith_cons_cons, List.map_cons, List.sum_cons, canonStr_plain (hk _ List.mem_cons_self),
      String.length_append, h1]
    omega

/-- **`canon` writes an object of increasing plain keys in a text no longer than its parts**: two braces and, for
each key, the key, its value's text and four more characters. -/
theorem canon_mkObj_le {l : List (String × Json)} {ks : List String} {bs : List Nat}
    (hks : l.map (·.1) = ks) (hs : ks.Pairwise (· < ·)) (hk : ∀ k ∈ ks, PlainStr k)
    (hv : List.Forall₂ (fun p b => ∃ t, Flock.canon p.2 = .ok t ∧ t.length ≤ b) l bs) :
    ∃ t, Flock.canon (Json.mkObj l) = .ok t ∧ t.length ≤ 2 + ((ks.map (·.length + 4)).sum + bs.sum) := by
  subst hks
  obtain ⟨ts, hts, hsum⟩ := texts_of hv hk
  exact ⟨_, canon_mkObj hs hts, Nat.le_trans (objText_length_le _) (by omega)⟩

theorem canon_str_le {s : String} (h : PlainStr s) : ∃ t, Flock.canon (.str s) = .ok t ∧ t.length ≤ s.length + 2 := by
  refine ⟨_, by rw [Flock.canon]; rfl, ?_⟩
  have h1 : "\"".length = 1 := rfl
  rw [canonStr_plain h]
  simp only [String.length_append, h1]
  omega

theorem canon_num (m : Int) : Flock.canon (.num ⟨m, 0⟩) = .ok (toString m) := by
  rw [Flock.canon]
  rfl

/-! ## A printable value's tagged hash -/

theorem wf_of_good {j : Json} (h : Good j) : Wf j := by
  induction h with
  | null => exact .null
  | bool b => exact .bool b
  | num m => exact .num _
  | str s => exact .str s
  | arr _ ih => exact .arr ih
  | obj hm _ ih => exact .obj hm ih

/-- **A printable value has a tagged hash.** -/
theorem tagged_ok (tag : String) {j : Json} (hg : Good j) : ∃ d, taggedSha512 tag j = .ok d := by
  obtain ⟨s, hs⟩ := FlockComplete.JsonRT.canon_ok_some j hg
  obtain ⟨t, ht, -⟩ := CanonLine.canon_utf8 (Verity.Sampled.sortedKeys_of_wf j (wf_of_good hg)) hs
  exact ⟨_, by unfold taggedSha512; rw [ht]; rfl⟩

end Properties.Challenger.Built
