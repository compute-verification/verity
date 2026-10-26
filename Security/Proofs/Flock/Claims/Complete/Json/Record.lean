import Proofs.Flock.Claims.Complete.Json.Spec
import Proofs.Flock.Claims.Complete.Json.Hex
import Proofs.Flock.Claims.Complete.Verdict

/-!
# The honest session record as a JSON value

`recordJson` (the honest prover's record) builds one `Json` object and prints it with `printJ`. `recordObj` is that
object, written as a pure value of the record's parts with every digest `dgOf spec h`: `recordJson` through
`Hashes.ofSha512 h` returns `recordObj`, printed, when the statement names no keyed coin tree (`Bridge`).

* **The record's parts**: a round (`roundObj`), a rep stream (`streamObj`), a proof digest (`proofObj`), the link
  (`linkObj`, with `y = 0`, `yOf`) and the bytes `link_*` digests (`lbOf`).
* **Printable** (`Good`): every object of the record is a tree of distinct keys and every number a natural, so the record
  is printable when the statement's draw is (`good_recordObj`).
* **Key lookups**: an object built from a list of distinct keys has each key's value (`getObjVal?_mkObj`) and no other
  key (`get?_mkObj_none`); through `Same`, a parsed object's `Record.fld`, `str`, `nat`, `arr` and `isNull` read the
  printed object's values (`fld_same`, …), and a value with no object in it parses back to itself (`Plain.same_eq`).
-/

namespace FlockComplete.JsonRT

open Flock Transcript FlockComplete
open Lean (Json JsonNumber)

/-! ## The record's parts -/

/-- A recorded round `k` as the record writes it (`roundJ` through `Hashes.ofSha512 h`). -/
def roundList (spec : SessionSpec) (h : ByteArray → ByteArray) (k : Nat) (r : Round) : List (String × Json) :=
  [("k", Lean.toJson k), ("n", Lean.toJson r.n),
    (s!"msg_{spec.dk}", hexJ (dgOf spec h (r.bytes.getD ByteArray.empty))),
    ("coins", Json.arr (r.coins.map f128J)), ("msg", hexJ (r.bytes.getD ByteArray.empty)),
    ("msg_len", Lean.toJson (r.bytes.getD ByteArray.empty).size)]

def roundObj (spec : SessionSpec) (h : ByteArray → ByteArray) (k : Nat) (r : Round) : Json :=
  Json.mkObj (roundList spec h k r)

/-- A rep stream `(i, r, name)` with its recorded rounds. -/
def streamList (spec : SessionSpec) (h : ByteArray → ByteArray) (rootB : Array ByteArray)
    (q : (Nat × Nat × String) × Array Round) : List (String × Json) :=
  [("stream", Json.str q.1.2.2), ("parent", Json.str ""),
    ("label", Json.str (spec.domain spec.tables[q.1.1]! q.1.2.1)), ("at", Lean.toJson (0 : Nat)),
    (s!"root_{spec.dk}", hexJ (dgOf spec h rootB[q.1.1]!)),
    ("rounds", Json.arr ((Array.range q.2.size).map fun k => roundObj spec h k q.2[k]!))]

def streamObj (spec : SessionSpec) (h : ByteArray → ByteArray) (rootB : Array ByteArray)
    (q : (Nat × Nat × String) × Array Round) : Json :=
  Json.mkObj (streamList spec h rootB q)

/-- A rep stream's proof file digest. -/
def proofList (spec : SessionSpec) (h : ByteArray → ByteArray) (q : (Nat × Nat × String) × ByteArray) :
    List (String × Json) :=
  [("stream", Json.str q.1.2.2), (spec.dk, hexJ (dgOf spec h q.2))]

def proofObj (spec : SessionSpec) (h : ByteArray → ByteArray) (q : (Nat × Nat × String) × ByteArray) : Json :=
  Json.mkObj (proofList spec h q)

/-- The link's `y`: zero, one per point and table. -/
def yOf (spec : SessionSpec) : Array F128 := Array.replicate (spec.points * spec.tables.size) 0

/-- The bytes `link_*` digests (S14): `Σ`, each table's `root_B` in lexicographic table order, the points, `y`. -/
def lbOf (spec : SessionSpec) (rootB : Array ByteArray) (points : Array (Array F128)) : ByteArray :=
  (yOf spec).foldl F128.pushBytes
    (points.foldl (fun lb p => p.foldl F128.pushBytes lb)
      (((spec.tables.zip rootB).qsort (·.1 < ·.1)).foldl (fun lb q => lb ++ q.2) spec.sigma))

/-- Each table's `root_B` (or its digest), by table. -/
def rootBList (spec : SessionSpec) (f : ByteArray → ByteArray) (rootB : Array ByteArray) : List (String × Json) :=
  (spec.tables.zip rootB).toList.map fun q => (q.1, hexJ (f q.2))

/-- The exchange link: `Σ` as `root_F`, each `root_B`, no public words, the points, `y = 0`. -/
def linkList (spec : SessionSpec) (h : ByteArray → ByteArray) (rootB : Array ByteArray)
    (points : Array (Array F128)) : List (String × Json) :=
  [("mode", Json.str "exchange"), ("sigma", hexJ spec.sigma),
    ("root_b", Json.mkObj (rootBList spec id rootB)),
    (s!"root_b_{spec.dk}", Json.mkObj (rootBList spec (dgOf spec h) rootB)),
    ("root_f", hexJ spec.sigma), (s!"root_f_{spec.dk}", hexJ (dgOf spec h spec.sigma)),
    ("publics", Json.mkObj []), (s!"publics_{spec.dk}", Json.mkObj []),
    ("points", Json.arr (points.map fun p => Json.arr (p.map f128J))), ("y", Json.arr ((yOf spec).map f128J))]

def linkObj (spec : SessionSpec) (h : ByteArray → ByteArray) (rootB : Array ByteArray)
    (points : Array (Array F128)) : Json :=
  Json.mkObj (linkList spec h rootB points)

/-- **The honest session record**, with no keyed coin tree. -/
def recordList (spec : SessionSpec) (h : ByteArray → ByteArray) (rootB : Array ByteArray)
    (points : Array (Array F128)) (streams : Array (Array Round)) (proofs : Array ByteArray) : List (String × Json) :=
  [("kind", Json.str "flock-live-session/v1"), ("aborted", Json.null), ("hello", Json.str spec.hello),
    ("config", Json.mkObj [("require_link", Json.bool true)]), ("link_mode", Json.str "exchange"),
    ("link", linkObj spec h rootB points), ("unit_draw", spec.unitDraw.getD Json.null),
    ("streams", Json.arr ((spec.repStreams.zip streams).map (streamObj spec h rootB))),
    (s!"link_{spec.dk}", hexJ (dgOf spec h (lbOf spec rootB points))),
    (s!"proof_{spec.dk}", Json.arr ((spec.repStreams.zip proofs).map (proofObj spec h)))]

def recordObj (spec : SessionSpec) (h : ByteArray → ByteArray) (rootB : Array ByteArray)
    (points : Array (Array F128)) (streams : Array (Array Round)) (proofs : Array ByteArray) : Json :=
  Json.mkObj (recordList spec h rootB points streams proofs)

/-! ## Objects of distinct keys -/

/-- The keys of a list are distinct. -/
abbrev Distinct (l : List (String × Json)) : Prop := l.Pairwise fun a b => ¬ compare a.1 b.1 = .eq

theorem get?_mkObj {l : List (String × Json)} (hd : Distinct l) {k : String} {v : Json} (hm : (k, v) ∈ l) :
    (Std.TreeMap.Raw.ofList l compare).get? k = some v := by
  rw [Std.TreeMap.Raw.get?_eq_getElem?]
  exact Std.TreeMap.Raw.getElem?_ofList_of_mem (Std.ReflCmp.compare_self) hd hm

theorem getObjVal?_mkObj {l : List (String × Json)} (hd : Distinct l) {k : String} {v : Json} (hm : (k, v) ∈ l) :
    (Json.mkObj l).getObjVal? k = .ok v := by
  simp only [Json.mkObj, Json.getObjVal?, get?_mkObj hd hm]
  rfl

theorem get?_mkObj_none {l : List (String × Json)} {k : String} (hk : (l.map Prod.fst).contains k = false) :
    (Std.TreeMap.Raw.ofList l compare).get? k = none := by
  rw [Std.TreeMap.Raw.get?_eq_getElem?]
  exact Std.TreeMap.Raw.getElem?_ofList_of_contains_eq_false hk

/-- Every value an object of distinct keys holds is one of its list's. -/
theorem mem_of_get?_mkObj {l : List (String × Json)} (hd : Distinct l) {k : String} {v : Json}
    (hv : (Std.TreeMap.Raw.ofList l compare).get? k = some v) : (k, v) ∈ l := by
  cases hc : (l.map Prod.fst).contains k with
  | false => rw [get?_mkObj_none hc] at hv; cases hv
  | true =>
    obtain ⟨p, hp, hpk⟩ := List.mem_map.1 (List.contains_iff_mem.1 hc)
    have e := get?_mkObj hd (k := k) (v := p.2) (by rw [← hpk]; exact hp)
    rw [e] at hv
    cases hv
    rw [← hpk]
    exact hp

theorem good_mkObj {l : List (String × Json)} (hd : Distinct l) (hg : ∀ p ∈ l, Good p.2) : Good (Json.mkObj l) :=
  .obj Std.TreeMap.Raw.WF.ofList fun _ _ hv => hg _ (mem_of_get?_mkObj hd hv)

theorem good_toJson (n : Nat) : Good (Lean.toJson n) := .num n

theorem good_hexJ (b : ByteArray) : Good (hexJ b) := .str _

theorem good_arr_str (a : Array F128) : Good (Json.arr (a.map f128J)) :=
  .arr fun x hx => by
    obtain ⟨y, -, rfl⟩ := Array.mem_map.1 hx
    exact .str _

theorem good_arr_of {a : Array Json} (h : ∀ x ∈ a, Good x) : Good (Json.arr a) := .arr h

/-! ## The record's objects have distinct keys, and the record is printable -/

theorem dk_cases (spec : SessionSpec) : spec.dk = "sha512" ∨ spec.dk = "sha256" := by
  unfold SessionSpec.dk
  split <;> simp

theorem roundList_distinct (spec : SessionSpec) (h : ByteArray → ByteArray) (k : Nat) (r : Round) :
    Distinct (roundList spec h k r) := by
  rcases dk_cases spec with e | e <;> simp [roundList, e, toString]

theorem streamList_distinct (spec : SessionSpec) (h : ByteArray → ByteArray) (rootB : Array ByteArray)
    (q : (Nat × Nat × String) × Array Round) : Distinct (streamList spec h rootB q) := by
  rcases dk_cases spec with e | e <;> simp [streamList, e, toString]

theorem proofList_distinct (spec : SessionSpec) (h : ByteArray → ByteArray) (q : (Nat × Nat × String) × ByteArray) :
    Distinct (proofList spec h q) := by
  rcases dk_cases spec with e | e <;> simp [proofList, e]

theorem rootBList_one {spec : SessionSpec} {t : String} (ht : spec.tables = #[t]) (f : ByteArray → ByteArray)
    (rb : ByteArray) : rootBList spec f #[rb] = [(t, hexJ (f rb))] := by
  simp [rootBList, ht]

theorem linkList_distinct (spec : SessionSpec) (h : ByteArray → ByteArray) (rootB : Array ByteArray)
    (points : Array (Array F128)) : Distinct (linkList spec h rootB points) := by
  rcases dk_cases spec with e | e <;> simp [linkList, e, toString]

theorem recordList_distinct (spec : SessionSpec) (h : ByteArray → ByteArray) (rootB : Array ByteArray)
    (points : Array (Array F128)) (streams : Array (Array Round)) (proofs : Array ByteArray) :
    Distinct (recordList spec h rootB points streams proofs) := by
  rcases dk_cases spec with e | e <;> simp [recordList, e, toString]

theorem good_roundObj (spec : SessionSpec) (h : ByteArray → ByteArray) (k : Nat) (r : Round) :
    Good (roundObj spec h k r) :=
  good_mkObj (roundList_distinct spec h k r) (by simp [roundList, good_toJson, good_hexJ, good_arr_str])

theorem good_streamObj (spec : SessionSpec) (h : ByteArray → ByteArray) (rootB : Array ByteArray)
    (q : (Nat × Nat × String) × Array Round) : Good (streamObj spec h rootB q) := by
  refine good_mkObj (streamList_distinct spec h rootB q) ?_
  simp only [streamList, List.mem_cons, List.mem_nil_iff, or_false, forall_eq_or_imp, forall_eq]
  refine ⟨.str _, .str _, .str _, good_toJson _, good_hexJ _, good_arr_of fun x hx => ?_⟩
  obtain ⟨k, -, rfl⟩ := Array.mem_map.1 hx
  exact good_roundObj _ _ _ _

theorem good_proofObj (spec : SessionSpec) (h : ByteArray → ByteArray) (q : (Nat × Nat × String) × ByteArray) :
    Good (proofObj spec h q) :=
  good_mkObj (proofList_distinct spec h q) (by simp [proofList, good_hexJ]; exact .str _)

theorem good_linkObj {spec : SessionSpec} {t : String} (ht : spec.tables = #[t]) (h : ByteArray → ByteArray)
    (rb : ByteArray) (points : Array (Array F128)) : Good (linkObj spec h #[rb] points) := by
  refine good_mkObj (linkList_distinct spec h _ points) ?_
  simp only [linkList, rootBList_one ht, List.mem_cons, List.mem_nil_iff, or_false, forall_eq_or_imp, forall_eq]
  refine ⟨.str _, good_hexJ _, good_mkObj (by simp) (by simp [good_hexJ]),
    good_mkObj (by simp) (by simp [good_hexJ]), good_hexJ _, good_hexJ _, good_mkObj (by simp) (by simp),
    good_mkObj (by simp) (by simp), good_arr_of fun x hx => ?_, good_arr_str _⟩
  obtain ⟨p, -, rfl⟩ := Array.mem_map.1 hx
  exact good_arr_str _

/-- **The record is printable** when the statement's draw is. -/
theorem good_recordObj {spec : SessionSpec} {t : String} (ht : spec.tables = #[t]) (h : ByteArray → ByteArray)
    (rb : ByteArray) (points : Array (Array F128)) (streams : Array (Array Round)) (proofs : Array ByteArray)
    (hd : Good (spec.unitDraw.getD Json.null)) : Good (recordObj spec h #[rb] points streams proofs) := by
  refine good_mkObj (recordList_distinct spec h _ points streams proofs) ?_
  simp only [recordList, List.mem_cons, List.mem_nil_iff, or_false, forall_eq_or_imp, forall_eq]
  refine ⟨.str _, .null, .str _, good_mkObj (by simp) (by simp; exact .bool true), .str _, good_linkObj ht h rb points,
    hd, good_arr_of fun x hx => ?_, good_hexJ _, good_arr_of fun x hx => ?_⟩
  · obtain ⟨q, -, rfl⟩ := Array.mem_map.1 hx
    exact good_streamObj _ _ _ _
  · obtain ⟨q, -, rfl⟩ := Array.mem_map.1 hx
    exact good_proofObj _ _ _

/-! ## A value with no object parses back to itself -/

/-- A JSON value with no object in it. -/
inductive Plain : Json → Prop
  | null : Plain .null
  | bool (b : Bool) : Plain (.bool b)
  | num (n : JsonNumber) : Plain (.num n)
  | str (s : String) : Plain (.str s)
  | arr {a : Array Json} (h : ∀ x ∈ a, Plain x) : Plain (.arr a)

theorem Plain.same_eq : ∀ {a b : Json}, Plain a → Same a b → a = b
  | _, _, .null, h => (same_null h).symm
  | _, _, .bool _, h => (same_bool h).symm
  | _, _, .num _, h => (same_num h).symm
  | _, _, .str _, h => (same_str h).symm
  | .arr a, _, .arr hp, h => by
    obtain ⟨b, rfl, hs, he⟩ := same_arr h
    congr 1
    apply Array.ext hs
    intro i h₁ h₂
    exact Plain.same_eq (hp _ (Array.getElem_mem h₁)) (he i h₁ h₂)

theorem plain_arr_str (a : Array F128) : Plain (Json.arr (a.map f128J)) :=
  .arr fun x hx => by
    obtain ⟨y, -, rfl⟩ := Array.mem_map.1 hx
    exact .str _

/-! ## The decoder's reads through `Same` -/

theorem fld_same {j j' : Json} (h : Same j j') {k : String} {v : Json} (hv : j.getObjVal? k = .ok v) :
    ∃ v', Record.fld j' k = .ok v' ∧ Same v v' := by
  obtain ⟨v', e, hs⟩ := h.getObjVal hv
  exact ⟨v', by simp only [Record.fld, e], hs⟩

theorem fld_plain {j j' : Json} (h : Same j j') {k : String} {v : Json} (hv : j.getObjVal? k = .ok v) (hp : Plain v) :
    Record.fld j' k = .ok v := by
  obtain ⟨v', e, hs⟩ := fld_same h hv
  rw [e, hp.same_eq hs]

theorem str_same {j j' : Json} (h : Same j j') {k s : String} (hv : j.getObjVal? k = .ok (.str s)) :
    Record.str j' k = .ok s := by
  simp only [Record.str, fld_plain h hv (.str s)]
  rfl

theorem nat_same {j j' : Json} (h : Same j j') {k : String} {n : Nat} (hv : j.getObjVal? k = .ok (Lean.toJson n)) :
    Record.nat j' k = .ok n := by
  simp only [Record.nat, fld_plain h hv (.num _)]
  rfl

theorem arr_plain {j j' : Json} (h : Same j j') {k : String} {a : Array Json} (hv : j.getObjVal? k = .ok (.arr a))
    (hp : Plain (.arr a)) : Record.arr j' k = .ok a := by
  simp only [Record.arr, fld_plain h hv hp]
  rfl

theorem arr_same {j j' : Json} (h : Same j j') {k : String} {a : Array Json} (hv : j.getObjVal? k = .ok (.arr a)) :
    ∃ b : Array Json, Record.arr j' k = .ok b ∧ a.size = b.size ∧
      ∀ (i : Nat) (h₁ : i < a.size) (h₂ : i < b.size), Same a[i] b[i] := by
  obtain ⟨v', e, hs⟩ := fld_same h hv
  obtain ⟨b, rfl, hsz, he⟩ := same_arr hs
  exact ⟨b, by simp only [Record.arr, e]; rfl, hsz, he⟩

theorem isNull_null {j j' : Json} (h : Same j j') {k : String} (hv : j.getObjVal? k = .ok .null) :
    Record.isNull j' k = true := by
  obtain ⟨v', e, hs⟩ := h.getObjVal hv
  simp only [Record.isNull, e, same_null hs]

theorem isNull_str {j j' : Json} (h : Same j j') {k s : String} (hv : j.getObjVal? k = .ok (.str s)) :
    Record.isNull j' k = false := by
  obtain ⟨v', e, hs⟩ := h.getObjVal hv
  simp only [Record.isNull, e, same_str hs]

theorem isNull_none {l : List (String × Json)} {j' : Json} (h : Same (Json.mkObj l) j') {k : String}
    (hk : (l.map Prod.fst).contains k = false) : Record.isNull j' k = true := by
  obtain ⟨e, he⟩ := h.getObjVal_none rfl (get?_mkObj_none hk)
  simp only [Record.isNull, he]

/-! ## Byte equality -/

theorem bytes_beq_self (b : ByteArray) : (b == b) = true := by
  show (b.data == b.data) = true
  simp

theorem bytes_bne_self (b : ByteArray) : (b != b) = false := by
  simp [bne, bytes_beq_self]

end FlockComplete.JsonRT
