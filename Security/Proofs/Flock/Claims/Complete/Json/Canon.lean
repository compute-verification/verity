import Proofs.Flock.Claims.CanonEq
import Proofs.Flock.Claims.Complete.Json.Spec

/-!
# What completeness reads of `canon`, proved

`Flock.canon` is total, so what completeness reads of it alone is proved, not assumed: `canon` is the same on alike
values (`canon_same`: alike values are equivalent, `same_jsonEquiv`, and `canon` prints equivalent values alike,
`FlockDeploy.canon_jsonEquiv`), and returns on printable ASCII ones (`canon_ok`: every number an integer and every
string and key ASCII without DEL, so neither refusal fires). The round trips through the verifier's parser
`Flock.JsonParse.parse` (the printer's and `canon`'s) are proved too (`RoundTrip.lean`), and the parser's trees are well
formed (`Parse.lean`).

A value `canon` prints is ASCII (`ascii_of_canon`), and so is every value alike it (`ascii_of_same`) and an ASCII
object with an ASCII entry put in (`ascii_insert`): the honest prover's `Hello`, which `canon` printed, stays ASCII
once the verifier has read it back and put its nonce in.
-/

namespace FlockComplete.JsonRT

open Lean (Json JsonNumber)
open Flock

/-- **Alike values are equivalent** (`FlockDeploy.JsonEquiv`): one form, whatever their objects' key order. -/
theorem same_jsonEquiv : ∀ {a b : Json}, Same a b → FlockDeploy.JsonEquiv a b
  | _, _, .null => rfl
  | _, _, .bool _ => rfl
  | _, _, .num _ => rfl
  | _, _, .str _ => rfl
  | _, _, .arr hs he => (FlockDeploy.jsonEquiv_arr_iff _ _).2 ⟨hs, fun i h₁ h₂ => same_jsonEquiv (he i h₁ h₂)⟩
  | _, _, .obj hm hn hk hv =>
    (FlockDeploy.jsonEquiv_obj_iff hm hn).2 ⟨hk, fun k a b ha hb => same_jsonEquiv (hv k a b ha hb)⟩

/-- **`canon` is the same on alike values**. -/
theorem canon_same (a b : Json) (h : Same a b) : canon a = canon b :=
  FlockDeploy.canon_jsonEquiv (same_jsonEquiv h)

/-- `Flock.ascii` is `Ascii`'s condition on a string. -/
theorem ascii_iff (s : String) : Flock.ascii s = true ↔ ∀ c ∈ s.toList, c.toNat < 0x7f := by
  simp [Flock.ascii, List.all_eq_true]

/-- **An object whose every entry has an ASCII key and an ASCII value** is ASCII. -/
theorem ascii_obj {m : Std.TreeMap.Raw String Json compare}
    (h : ∀ p ∈ m.toList, (∀ c ∈ p.1.toList, c.toNat < 0x7f) ∧ Ascii p.2) : Ascii (.obj m) :=
  .obj (fun p hp => (h p hp).1) (fun p hp => (h p hp).2)

/-- **Every entry of an ASCII object** has an ASCII key and an ASCII value. -/
theorem ascii_obj_mem {m : Std.TreeMap.Raw String Json compare} (ha : Ascii (.obj m)) :
    ∀ p ∈ m.toList, (∀ c ∈ p.1.toList, c.toNat < 0x7f) ∧ Ascii p.2 := by
  cases ha with
  | obj hk hv => exact fun p hp => ⟨hk p hp, hv p hp⟩

/-- **`canon` returns on printable ASCII values**. -/
theorem canon_ok : ∀ j : Json, Good j → Ascii j → ∃ s, canon j = .ok s
  | _, .null, _ => ⟨_, by rw [canon]; rfl⟩
  | _, .bool _, _ => ⟨_, by rw [canon]; rfl⟩
  | _, .num _, _ => ⟨_, by rw [canon]; rfl⟩
  | .str s, .str _, .str ha => ⟨_, by rw [canon, if_pos ((ascii_iff s).2 ha)]; rfl⟩
  | .arr a, .arr h, .arr ha => by
    obtain ⟨r, hr⟩ := FlockDeploy.array_mapM_ok (f := fun x : {x // x ∈ a} => canon x.1) (a := a.attach)
      fun x _ => canon_ok x.1 (h x.1 x.2) (ha x.1 x.2)
    exact ⟨_, by rw [canon, hr]; rfl⟩
  | .obj m, .obj hm h, .obj hak hav => by
    obtain ⟨r, hr⟩ := FlockDeploy.array_mapM_ok
      (f := fun p : {p // p ∈ m.toArray} =>
        if ascii p.1.1 then do pure (p.1.1, ← canon p.1.2) else throw notAscii) (a := m.toArray.attach)
      fun p _ => by
        have hp : (p.1.1, p.1.2) ∈ m.toList := by
          rw [← Std.TreeMap.Raw.toList_toArray]
          exact Array.mem_toList_iff.2 p.2
        have hg := (Std.TreeMap.Raw.mem_toList_iff_getElem?_eq_some hm).1 hp
        obtain ⟨s, hs⟩ := canon_ok p.1.2 (h _ _ (by rw [Std.TreeMap.Raw.get?_eq_getElem?]; exact hg)) (hav _ hp)
        exact ⟨_, by rw [if_pos ((ascii_iff p.1.1).2 (hak _ hp)), hs]; rfl⟩
    exact ⟨_, by rw [canon, hr]; rfl⟩

private theorem bind_ok' {ε α β : Type} {x : Except ε α} {f : α → Except ε β} {b : β} (h : (x >>= f) = .ok b) :
    ∃ a, x = .ok a ∧ f a = .ok b := by
  cases x with
  | error e => cases h
  | ok a => exact ⟨a, rfl, h⟩

private theorem mapM_ok_mem' {α β ε : Type} {f : α → Except ε β} :
    ∀ {l : List α} {r : List β}, l.mapM f = .ok r → ∀ x ∈ l, ∃ y, f x = .ok y
  | [], _, _, _, hx => by cases hx
  | a :: as, _, h, x, hx => by
    rw [List.mapM_cons] at h
    obtain ⟨b, hb, h⟩ := bind_ok' h
    obtain ⟨bs, hbs, -⟩ := bind_ok' h
    rcases List.mem_cons.1 hx with rfl | hx
    · exact ⟨b, hb⟩
    · exact mapM_ok_mem' hbs x hx

private theorem array_mapM_ok_mem' {α β ε : Type} {f : α → Except ε β} {a : Array α} {r : Array β}
    (h : a.mapM f = .ok r) : ∀ x ∈ a, ∃ y, f x = .ok y := by
  rw [Array.mapM_eq_mapM_toList] at h
  cases hl : a.toList.mapM f with
  | error e => rw [hl] at h; cases h
  | ok l => exact fun x hx => mapM_ok_mem' hl x (Array.mem_toList_iff.2 hx)

private theorem ascii_of_canon_lt : ∀ (k : Nat) (j : Json) {s : String}, sizeOf j < k → canon j = .ok s → Ascii j
  | 0, _, _, h, _ => absurd h (Nat.not_lt_zero _)
  | k + 1, j, _, hk, h => by
    match j with
    | .null => exact .null
    | .bool b => exact .bool b
    | .num n => exact .num n
    | .str t =>
      rw [canon] at h
      by_cases ht : ascii t = true
      · exact .str ((ascii_iff t).1 ht)
      · rw [if_neg ht] at h; cases h
    | .arr a =>
      rw [canon] at h
      obtain ⟨_, hp, -⟩ := bind_ok' h
      refine .arr fun x hx => ?_
      obtain ⟨_, hy⟩ := array_mapM_ok_mem' hp ⟨x, hx⟩ (Array.mem_attach _ _)
      have hlt := Array.sizeOf_lt_of_mem hx
      simp only [Json.arr.sizeOf_spec] at hk
      exact ascii_of_canon_lt k x (by omega) hy
    | .obj m =>
      rw [canon] at h
      obtain ⟨_, hv, -⟩ := bind_ok' h
      refine ascii_obj fun p hp => ?_
      have hp' : p ∈ m.toArray := by
        rw [← Array.mem_toList_iff, Std.TreeMap.Raw.toList_toArray]
        exact hp
      obtain ⟨_, hy⟩ := array_mapM_ok_mem' hv ⟨p, hp'⟩ (Array.mem_attach _ _)
      try dsimp only at hy
      have hlt := sizeOf_lt_of_mem_toArray hp'
      by_cases ht : ascii p.1 = true
      · rw [if_pos ht] at hy
        obtain ⟨_, hc, -⟩ := bind_ok' hy
        exact ⟨(ascii_iff p.1).1 ht, ascii_of_canon_lt k p.2 (by omega) hc⟩
      · rw [if_neg ht] at hy; cases hy

/-- **A value `canon` prints is ASCII.** -/
theorem ascii_of_canon {j : Json} {s : String} (h : canon j = .ok s) : Ascii j :=
  ascii_of_canon_lt (sizeOf j + 1) j (Nat.lt_succ_self _) h

/-- **A value alike an ASCII one is ASCII.** -/
theorem ascii_of_same : ∀ {a b : Json}, Same a b → Ascii a → Ascii b
  | _, _, .null, _ => .null
  | _, _, .bool b, _ => .bool b
  | _, _, .num n, _ => .num n
  | _, _, .str _, h => h
  | _, _, .arr hs he, h => by
    cases h with
    | arr ha =>
    refine .arr fun x hx => ?_
    obtain ⟨i, hi, rfl⟩ := Array.mem_iff_getElem.1 hx
    exact ascii_of_same (he i (hs ▸ hi) hi) (ha _ (Array.getElem_mem _))
  | _, _, .obj hm hn hk hv, h => by
    have ha := ascii_obj_mem h
    refine ascii_obj fun ⟨k, v⟩ hp => ?_
    have hb := (Std.TreeMap.Raw.mem_toList_iff_getElem?_eq_some hn).1 hp
    rw [← Std.TreeMap.Raw.get?_eq_getElem?] at hb
    have hs := hk k
    rw [hb] at hs
    obtain ⟨a, ha'⟩ := Option.isSome_iff_exists.1 hs
    have hma : (k, a) ∈ _ := (Std.TreeMap.Raw.mem_toList_iff_getElem?_eq_some hm).2
      (by rw [← Std.TreeMap.Raw.get?_eq_getElem?]; exact ha')
    obtain ⟨hk', hav⟩ := ha _ hma
    exact ⟨hk', ascii_of_same (hv k a v ha' hb) hav⟩

/-- **A value alike an ASCII one is ASCII**, the other way round. -/
theorem ascii_of_same_rev : ∀ {a b : Json}, Same a b → Ascii b → Ascii a
  | _, _, .null, _ => .null
  | _, _, .bool b, _ => .bool b
  | _, _, .num n, _ => .num n
  | _, _, .str _, h => h
  | _, _, .arr hs he, h => by
    cases h with
    | arr ha =>
    refine .arr fun x hx => ?_
    obtain ⟨i, hi, rfl⟩ := Array.mem_iff_getElem.1 hx
    exact ascii_of_same_rev (he i hi (hs ▸ hi)) (ha _ (Array.getElem_mem _))
  | _, _, .obj hm hn hk hv, h => by
    have ha := ascii_obj_mem h
    refine ascii_obj fun ⟨k, v⟩ hp => ?_
    have hb := (Std.TreeMap.Raw.mem_toList_iff_getElem?_eq_some hm).1 hp
    rw [← Std.TreeMap.Raw.get?_eq_getElem?] at hb
    have hs := hk k
    rw [hb] at hs
    obtain ⟨b, hb'⟩ := Option.isSome_iff_exists.1 hs.symm
    have hnb : (k, b) ∈ _ := (Std.TreeMap.Raw.mem_toList_iff_getElem?_eq_some hn).2
      (by rw [← Std.TreeMap.Raw.get?_eq_getElem?]; exact hb')
    obtain ⟨hk', hbv⟩ := ha _ hnb
    exact ⟨hk', ascii_of_same_rev (hv k v b hb hb') hbv⟩

/-- **An entry of an ASCII object** is ASCII. -/
theorem ascii_get {m : Std.TreeMap.Raw String Json compare} (hm : m.WF) (ha : Ascii (.obj m)) {k : String} {v : Json}
    (hv : m.get? k = some v) : Ascii v := by
  exact (ascii_obj_mem ha (k, v) ((Std.TreeMap.Raw.mem_toList_iff_getElem?_eq_some hm).2
    (by rw [← Std.TreeMap.Raw.get?_eq_getElem?]; exact hv))).2

/-- **A key of an ASCII object** is ASCII. -/
theorem ascii_key {m : Std.TreeMap.Raw String Json compare} (hm : m.WF) (ha : Ascii (.obj m)) {k : String} {v : Json}
    (hv : m.get? k = some v) : ∀ c ∈ k.toList, c.toNat < 0x7f := by
  exact (ascii_obj_mem ha (k, v) ((Std.TreeMap.Raw.mem_toList_iff_getElem?_eq_some hm).2
    (by rw [← Std.TreeMap.Raw.get?_eq_getElem?]; exact hv))).1

/-- **A value an ASCII object reads at a key** is ASCII. -/
theorem ascii_getObjVal {m : Std.TreeMap.Raw String Json compare} (hm : m.WF) (ha : Ascii (.obj m)) {k : String}
    {v : Json} (hv : (Json.obj m).getObjVal? k = .ok v) : Ascii v := by
  simp only [Json.getObjVal?] at hv
  split at hv
  · cases hv
    exact ascii_get hm ha (by assumption)
  · cases hv

/-- **An ASCII object with an ASCII entry put in** is ASCII. -/
theorem ascii_insert {m : Std.TreeMap.Raw String Json compare} (hm : m.WF) (ha : Ascii (.obj m)) {k : String}
    {v : Json} (hk : ∀ c ∈ k.toList, c.toNat < 0x7f) (hv : Ascii v) : Ascii (.obj (m.insert k v)) := by
  replace ha := ascii_obj_mem ha
  refine ascii_obj fun ⟨k', v'⟩ hp => ?_
  have h := (Std.TreeMap.Raw.mem_toList_iff_getElem?_eq_some hm.insert).1 hp
  rw [Std.TreeMap.Raw.getElem?_insert hm] at h
  split at h
  · rename_i he
    cases h
    rw [← Std.LawfulEqCmp.compare_eq_iff_eq.1 he]
    exact ⟨hk, hv⟩
  · exact ha _ ((Std.TreeMap.Raw.mem_toList_iff_getElem?_eq_some hm).2 h)

/-- **An object is ASCII where it is with a new entry put in.** -/
theorem ascii_of_insert {m : Std.TreeMap.Raw String Json compare} (hm : m.WF) {k : String} {v : Json}
    (hn : m.get? k = none) (ha : Ascii (.obj (m.insert k v))) : Ascii (.obj m) := by
  replace ha := ascii_obj_mem ha
  refine ascii_obj fun ⟨k', v'⟩ hp => ha _ ?_
  have h := (Std.TreeMap.Raw.mem_toList_iff_getElem?_eq_some hm).1 hp
  refine (Std.TreeMap.Raw.mem_toList_iff_getElem?_eq_some hm.insert).2 ?_
  rw [Std.TreeMap.Raw.getElem?_insert hm]
  split
  · rename_i he
    rw [← Std.LawfulEqCmp.compare_eq_iff_eq.1 he, ← Std.TreeMap.Raw.get?_eq_getElem?, hn] at h
    cases h
  · exact h

end FlockComplete.JsonRT
