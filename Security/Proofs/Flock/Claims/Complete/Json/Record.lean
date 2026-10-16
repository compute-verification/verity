import Proofs.Flock.Claims.Complete.Json.Spec
import Proofs.Flock.Claims.Complete.Json.Hex
import Proofs.Flock.Claims.Complete.Verdict

/-!
# The honest session record as a JSON value

`recordJson` (the honest prover's record) builds one `Json` object and prints it with `compress`. `recordObj` is that
object, written as a pure value of the record's parts with every digest `dgOf spec h`: `recordJson` through
`Hashes.ofSha512 h` returns `recordObj`'s `compress` when the statement names no keyed coin tree (`Bridge`).

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
def roundObj (spec : SessionSpec) (h : ByteArray → ByteArray) (k : Nat) (r : Round) : Json :=
  Json.mkObj [("k", Lean.toJson k), ("n", Lean.toJson r.n),
    (s!"msg_{spec.dk}", hexJ (dgOf spec h (r.bytes.getD ByteArray.empty))),
    ("coins", Json.arr (r.coins.map f128J)), ("msg", hexJ (r.bytes.getD ByteArray.empty)),
    ("msg_len", Lean.toJson (r.bytes.getD ByteArray.empty).size)]

/-- A rep stream `(i, r, name)` with its recorded rounds. -/
def streamObj (spec : SessionSpec) (h : ByteArray → ByteArray) (rootB : Array ByteArray)
    (q : (Nat × Nat × String) × Array Round) : Json :=
  Json.mkObj [("stream", Json.str q.1.2.2), ("parent", Json.str ""),
    ("label", Json.str (spec.domain spec.tables[q.1.1]! q.1.2.1)), ("at", Lean.toJson (0 : Nat)),
    (s!"root_{spec.dk}", hexJ (dgOf spec h rootB[q.1.1]!)),
    ("rounds", Json.arr ((Array.range q.2.size).map fun k => roundObj spec h k q.2[k]!))]

/-- A rep stream's proof file digest. -/
def proofObj (spec : SessionSpec) (h : ByteArray → ByteArray) (q : (Nat × Nat × String) × ByteArray) : Json :=
  Json.mkObj [("stream", Json.str q.1.2.2), (spec.dk, hexJ (dgOf spec h q.2))]

/-- The link's `y`: zero, one per point and table. -/
def yOf (spec : SessionSpec) : Array F128 := Array.replicate (spec.points * spec.tables.size) 0

/-- The bytes `link_*` digests (S14): `Σ`, each table's `root_B` in lexicographic table order, the points, `y`. -/
def lbOf (spec : SessionSpec) (rootB : Array ByteArray) (points : Array (Array F128)) : ByteArray :=
  (yOf spec).foldl F128.pushBytes
    (points.foldl (fun lb p => p.foldl F128.pushBytes lb)
      (((spec.tables.zip rootB).qsort (·.1 < ·.1)).foldl (fun lb q => lb ++ q.2) spec.sigma))

/-- The exchange link: `Σ` as `root_F`, each `root_B`, no public words, the points, `y = 0`. -/
def linkObj (spec : SessionSpec) (h : ByteArray → ByteArray) (rootB : Array ByteArray)
    (points : Array (Array F128)) : Json :=
  Json.mkObj [("mode", Json.str "exchange"), ("sigma", hexJ spec.sigma),
    ("root_b", Json.mkObj ((spec.tables.zip rootB).toList.map fun q => (q.1, hexJ q.2))),
    (s!"root_b_{spec.dk}", Json.mkObj ((spec.tables.zip rootB).toList.map fun q => (q.1, hexJ (dgOf spec h q.2)))),
    ("root_f", hexJ spec.sigma), (s!"root_f_{spec.dk}", hexJ (dgOf spec h spec.sigma)),
    ("publics", Json.mkObj []), (s!"publics_{spec.dk}", Json.mkObj []),
    ("points", Json.arr (points.map fun p => Json.arr (p.map f128J))), ("y", Json.arr ((yOf spec).map f128J))]

/-- **The honest session record**, with no keyed coin tree. -/
def recordObj (spec : SessionSpec) (h : ByteArray → ByteArray) (rootB : Array ByteArray)
    (points : Array (Array F128)) (streams : Array (Array Round)) (proofs : Array ByteArray) : Json :=
  Json.mkObj [("kind", Json.str "flock-live-session/v1"), ("aborted", Json.null), ("hello", Json.str spec.hello),
    ("config", Json.mkObj [("require_link", Json.bool true)]), ("link_mode", Json.str "exchange"),
    ("link", linkObj spec h rootB points), ("unit_draw", spec.unitDraw.getD Json.null),
    ("streams", Json.arr ((spec.repStreams.zip streams).map (streamObj spec h rootB))),
    (s!"link_{spec.dk}", hexJ (dgOf spec h (lbOf spec rootB points))),
    (s!"proof_{spec.dk}", Json.arr ((spec.repStreams.zip proofs).map (proofObj spec h)))]

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
