module

public import Flock.Canon
public import Proofs.Core.QSort
public import Proofs.Flock.Recursive.ZkServerJson
public import Proofs.Flock.Recursive.ZkServerObj
public import Proofs.Flock.Claims.Complete.Json.RoundTrip
public import Proofs.Flock.Claims.CanonEq
public import CoreProtocol.Protocol
public import Proofs.Properties.CanonRoundTrip
public import Proofs.Properties.Verity.Sampled.HonestSession.JsonCanon
import all Init.Data.Ord.String
import all Init.Data.Repr

/-!
# `JsonCanon.JsonFacts`, proved

The honest-session witness's canonical-JSON facts (`Verity.Sampled.Instance.HonestSession.JsonCanon.JsonFacts`), proved
with no hypothesis: `Flock.canon` writes a plain value of ASCII strings as its `printJson` text (`canon_plain`), writes a
plain value and its parsed form the same (`canon_norm`), and a canonical text of ASCII parses back to `norm` of its value
(`canonicalJson_canon`). The first two are inductions over `Json` reusing landed printer/parser lemmas; the third parses
`canon`'s escaped text back through a parallel of ZkServerJson's `reads` whose string case is served by `esc_of`.
-/

@[expose] public section

namespace Verity.Sampled.Instance.HonestSession.JsonCanon.Facts

open Lean (Json JsonNumber)
open Flock (V canon canonStr)
open Flock.ZkServer
open FlockSoundness.Discharge.Recursion.ZkServerJson
open FlockSoundness.Discharge.Recursion.ZkServerJson renaming norm → fnorm
open FlockDeploy (jsonForm canon_jsonEquiv)
open Std.DTreeMap.Internal (Impl)

/-! ## `canon_norm`: `canon` writes a plain value and its parsed form the same -/

theorem fieldForms_toListModel (t : Impl String (fun _ => Json)) :
    FlockDeploy.fieldForms t = t.toListModel.map (fun p => (p.1, FlockDeploy.jsonForm p.2)) := by
  rw [FlockDeploy.fieldForms_eq, Impl.Const.toList_eq_toListModel_map, List.map_map]
  rfl

mutual
theorem norm_jsonEquiv : ∀ {j : Json}, plain j = true → FlockDeploy.jsonForm (fnorm j) = FlockDeploy.jsonForm j
  | .null, _ => rfl
  | .bool _, _ => rfl
  | .num _, _ => rfl
  | .str _, _ => rfl
  | .arr ⟨a⟩, h => by
    simp only [plain] at h
    rw [fnorm]
    simp only [jsonForm, FlockDeploy.jsonForms_eq]
    rw [norm_jsonEquivList a h]
  | .obj ⟨⟨t⟩⟩, h => by
    simp only [plain, Bool.and_eq_true] at h
    have hasc : t.toListModel.Pairwise (fun a b => compare a.1 b.1 = .lt) := of_decide_eq_true h.2
    have hf : t.toListModel.foldl (fun a e => a.insert e.1 (fnorm e.2)) (∅ : Std.TreeMap.Raw String Json) =
        (t.toListModel.map fun e => (⟨e.1, fnorm e.2⟩ : (_ : String) × Json)).foldl (fun a e => a.insert e.1 e.2) ∅ := by
      rw [List.foldl_map]
    have h0 : (∅ : Std.TreeMap.Raw String Json).inner.inner.toListModel = [] := rfl
    rw [fnorm]
    show FlockDeploy.jsonForm (.obj (normObj t ∅)) = _
    rw [jsonForm, jsonForm, fieldForms_toListModel, fieldForms_toListModel, normObj_eq, hf,
      toListModel_foldl _ _ Impl.WF.empty (by rw [h0, List.nil_append, List.pairwise_map]; exact hasc), h0,
      List.nil_append, List.map_map]
    congr 1
    apply FlockDeploy.FieldForms.ofList_inj.2
    apply List.map_congr_left
    intro e he
    simp only [Function.comp]
    rw [norm_jsonEquivObj t h.1 e he]
theorem norm_jsonEquivList : ∀ (a : List Json), plainList a = true →
    (normList a).map FlockDeploy.jsonForm = a.map FlockDeploy.jsonForm
  | [], _ => rfl
  | j :: js, h => by
    simp only [plainList, Bool.and_eq_true] at h
    simp only [normList, List.map_cons, norm_jsonEquiv h.1, norm_jsonEquivList js h.2]
theorem norm_jsonEquivObj : ∀ (t : Impl String (fun _ => Json)), plainObj t = true →
    ∀ e ∈ t.toListModel, FlockDeploy.jsonForm (fnorm e.2) = FlockDeploy.jsonForm e.2
  | .leaf, _ => by simp [Impl.toListModel]
  | .inner _ k v l r, h => by
    simp only [plainObj, Bool.and_eq_true] at h
    intro e he
    rw [Impl.toListModel_inner, List.mem_append, List.mem_cons] at he
    rcases he with he | rfl | he
    · exact norm_jsonEquivObj l h.1.1.1 e he
    · exact norm_jsonEquiv h.1.2
    · exact norm_jsonEquivObj r h.2 e he
end

/-- **`canon` writes a plain value and its parsed form the same.** -/
theorem canon_norm {j : Json} (h : plain j = true) : Flock.canon (fnorm j) = Flock.canon j :=
  canon_jsonEquiv (norm_jsonEquiv h)

mutual
/-- `Flock.canon`'s text of a plain value. -/
def cjChars : Json → String
  | .null => "null"
  | .bool b => if b then "true" else "false"
  | .num ⟨m, _⟩ => toString m
  | .str s => canonStr s
  | .arr ⟨a⟩ => "[" ++ ",".intercalate (cjList a) ++ "]"
  | .obj ⟨⟨t⟩⟩ => "{" ++ ",".intercalate (cjObj t) ++ "}"
def cjList : List Json → List String
  | [] => []
  | j :: js => cjChars j :: cjList js
def cjObj : Impl String (fun _ => Json) → List String
  | .leaf => []
  | .inner _ k v l r => cjObj l ++ [canonStr k ++ ":" ++ cjChars v] ++ cjObj r
end

theorem cjList_eq : ∀ a : List Json, cjList a = a.map cjChars
  | [] => rfl
  | j :: js => by rw [cjList, cjList_eq js]; rfl

theorem cjObj_eq : ∀ t : Impl String (fun _ => Json),
    cjObj t = t.toListModel.map (fun p => canonStr p.1 ++ ":" ++ cjChars p.2)
  | .leaf => rfl
  | .inner _ k v l r => by
    rw [cjObj, cjObj_eq l, cjObj_eq r, Impl.toListModel_inner]; simp

/-- `mapM` in `Except` returns the mapped list where `f` returns `.ok (g x)` on every element. -/
theorem list_mapM_ok {α β : Type} {f : α → Except String β} {g : α → β} :
    ∀ l : List α, (∀ x ∈ l, f x = .ok (g x)) → l.mapM f = .ok (l.map g)
  | [], _ => rfl
  | a :: as, h => by
    rw [List.mapM_cons, h a List.mem_cons_self, list_mapM_ok as (fun x hx => h x (List.mem_cons_of_mem _ hx))]
    rfl

theorem str_cmp_lt {a b : String} (h : compare a b = .lt) : a < b := by
  have h2 : compareOfLessAndEq a b = .lt := h
  exact compareOfLessAndEq_eq_lt.1 h2

theorem keys_sorted {β : Type} {m : List (String × β)} (hasc : m.Pairwise (fun a b => compare a.1 b.1 = .lt)) :
    ∀ i (hi : i < m.length) j (hj : j < m.length), decide (m[i].1 < m[j].1) = decide (i < j) := by
  intro i hi j hj
  rw [decide_eq_decide]
  rcases Nat.lt_trichotomy i j with h | h | h
  · have hc : compare m[i].1 m[j].1 = .lt := List.pairwise_iff_getElem.1 hasc i j hi hj h
    exact ⟨fun _ => h, fun _ => str_cmp_lt hc⟩
  · subst h; exact ⟨fun hx => absurd hx (String.lt_irrefl _), fun hx => absurd hx (Nat.lt_irrefl _)⟩
  · have hc : compare m[j].1 m[i].1 = .lt := List.pairwise_iff_getElem.1 hasc j i hj hi h
    have hlt : ¬ m[i].1 < m[j].1 := String.lt_asymm (str_cmp_lt hc)
    exact ⟨fun hx => absurd hx hlt, fun hx => absurd hx (Nat.not_lt.2 (Nat.le_of_lt h))⟩

theorem array_mapM_ok_map {α β : Type} {f : α → Except String β} {g : α → β}
    (a : Array α) (h : ∀ x ∈ a, f x = .ok (g x)) : a.mapM f = .ok (a.toList.map g).toArray := by
  rw [Array.mapM_eq_mapM_toList, list_mapM_ok a.toList (fun x hx => h x ((Array.mem_toList_iff).1 hx))]
  rfl

theorem canon_cj : ∀ (k : Nat) (j : Json), sizeOf j < k → plain j = true → canon j = .ok (cjChars j)
  | 0, _, h, _ => absurd h (Nat.not_lt_zero _)
  | k + 1, j, hk, hp => by
    match j with
    | .null => rw [canon]; rfl
    | .bool _ => rw [canon]; rfl
    | .num ⟨m, e⟩ =>
      simp only [plain, Bool.and_eq_true, beq_iff_eq] at hp
      rw [canon, cjChars]
      simp only [hp.1, beq_self_eq_true, if_true]
      rfl
    | .str s => rw [canon]; rfl
    | .arr ⟨a⟩ =>
      simp only [plain, plainList_iff] at hp
      rw [canon, cjChars]
      have hmap := array_mapM_ok_map (⟨a⟩ : Array Json).attach (f := fun x => canon x.1)
        (g := fun x => cjChars x.1) (by
          intro x _
          have hmem : x.1 ∈ (⟨a⟩ : Array Json) := x.2
          have hlt : sizeOf x.1 < k := by
            have := Array.sizeOf_lt_of_mem hmem
            simp only [Json.arr.sizeOf_spec] at hk
            omega
          exact canon_cj k x.1 hlt (hp x.1 (by simpa using hmem)))
      rw [hmap]
      have hparts : ((⟨a⟩ : Array Json).attach.toList.map (fun x => cjChars x.1)) = cjList a := by
        rw [cjList_eq]
        simp
      simp only [bind, Except.bind, pure, Except.pure]
      rw [List.toList_toArray, hparts]
    | .obj ⟨⟨t⟩⟩ =>
      simp only [plain, Bool.and_eq_true] at hp
      obtain ⟨hpo, hasc⟩ := hp
      rw [canon, cjChars]
      have hbridge : (⟨⟨t⟩⟩ : Std.TreeMap.Raw String Json).toArray.toList
          = t.toListModel.map (fun e => (e.1, e.2)) :=
        FlockSoundness.Discharge.Recursion.ZkServerObj.toArray_toList t
      have hmap := array_mapM_ok_map (⟨⟨t⟩⟩ : Std.TreeMap.Raw String Json).toArray.attach
        (f := fun p => (do pure (p.1.1, ← canon p.1.2) : Flock.V (String × String)))
        (g := fun p => ((p.1.1, cjChars p.1.2) : String × String)) (by
          intro x _
          have hmem : x.1 ∈ (⟨⟨t⟩⟩ : Std.TreeMap.Raw String Json).toArray := x.2
          have hmemL : x.1 ∈ (⟨⟨t⟩⟩ : Std.TreeMap.Raw String Json).toArray.toList :=
            Array.mem_toList_iff.2 hmem
          rw [hbridge, List.mem_map] at hmemL
          obtain ⟨e, he, hxe⟩ := hmemL
          have hv2 : x.1.2 = e.2 := by rw [← hxe]
          have hpv : plain x.1.2 = true := by rw [hv2]; exact ((plainObj_iff t).1 hpo e he).2
          have hlt : sizeOf x.1.2 < k := by
            have hs := Flock.sizeOf_lt_of_mem_toArray hmem
            omega
          have hc := canon_cj k x.1.2 hlt hpv
          simp only [bind, Except.bind, hc, pure, Except.pure])
      rw [hmap]
      simp only [bind, Except.bind, pure, Except.pure]
      have hvals_list : ((⟨⟨t⟩⟩ : Std.TreeMap.Raw String Json).toArray.attach.toList.map
          (fun p => ((p.1.1, cjChars p.1.2) : String × String)))
          = t.toListModel.map (fun e => (e.1, cjChars e.2)) := by
        rw [Array.toList_attach, List.map_attachWith,
          List.attach_map_val (f := fun q : String × Json => ((q.1, cjChars q.2) : String × String)),
          hbridge, List.map_map]
        rfl
      have hasc' : ((⟨⟨t⟩⟩ : Std.TreeMap.Raw String Json).toArray.attach.toList.map
          (fun p => ((p.1.1, cjChars p.1.2) : String × String))).Pairwise
          (fun a b => compare a.1 b.1 = .lt) := by
        rw [hvals_list, List.pairwise_map]
        exact (of_decide_eq_true hasc).imp (fun h => h)
      have hqsort : ((((⟨⟨t⟩⟩ : Std.TreeMap.Raw String Json).toArray.attach.toList.map
          (fun p => ((p.1.1, cjChars p.1.2) : String × String))).toArray).qsort
          (fun a b => a.1 < b.1)) = (((⟨⟨t⟩⟩ : Std.TreeMap.Raw String Json).toArray.attach.toList.map
          (fun p => ((p.1.1, cjChars p.1.2) : String × String))).toArray) := by
        apply Verity.QSort.qsort_sorted
        intro i hi j hj
        rw [List.size_toArray] at hi hj
        have hks := keys_sorted hasc' i hi j hj
        simpa [List.getElem_toArray] using hks
      rw [hqsort, List.toList_toArray, hvals_list]
      rw [cjObj_eq]
      simp only [List.map_map]
      rfl

/-! ## `canon_plain`: canon of a plain, printable value is its `printJson` text -/

open FlockComplete.JsonRT.ReadBack (cesc canonStr_toList)

/-- For a printable, plain character, `canonStr` writes it verbatim. -/
theorem cesc_id {c : Char} (h1 : c ≠ '"') (h2 : c ≠ '\\') (h20 : 0x20 ≤ c.toNat) (h7f : c.toNat < 0x7f) :
    (cesc c).toList = [c] := by
  have hn : c ≠ '\n' := by rintro rfl; exact absurd h20 (by decide)
  have hr : c ≠ '\r' := by rintro rfl; exact absurd h20 (by decide)
  have ht : c ≠ '\t' := by rintro rfl; exact absurd h20 (by decide)
  unfold cesc
  split
  · simp_all
  · simp_all
  · simp_all
  · simp_all
  · simp_all
  · rw [if_neg (show ¬ ((c.toNat == 8) = true) by simp only [beq_iff_eq]; omega),
      if_neg (show ¬ ((c.toNat == 12) = true) by simp only [beq_iff_eq]; omega),
      if_neg (show ¬ ((decide (c.toNat < 0x20) || decide (0x7f ≤ c.toNat)) = true) by
        simp only [Bool.or_eq_true, decide_eq_true_eq]; omega)]
    exact String.toList_singleton c

/-- A list of printable, plain characters maps through `cesc` to itself. -/
theorem flatMap_cesc_id : ∀ (l : List Char), (∀ c ∈ l, c ≠ '"' ∧ c ≠ '\\' ∧ 0x20 ≤ c.toNat ∧ c.toNat < 0x7f) →
    l.flatMap (fun c => (cesc c).toList) = l
  | [], _ => rfl
  | c :: l, h => by
    have hc := h c List.mem_cons_self
    rw [List.flatMap_cons, cesc_id hc.1 hc.2.1 hc.2.2.1 hc.2.2.2,
      flatMap_cesc_id l (fun c' hc' => h c' (List.mem_cons_of_mem _ hc'))]
    rfl

/-- `canon`'s text of a plain string of printable, plain characters is the string verbatim in quotes. -/
theorem canonStr_verbatim {s : String} (hp : plainStr s = true) (ha : ∀ c ∈ s.toList, c.toNat < 0x7f) :
    (Flock.canonStr s).toList = '"' :: s.toList ++ ['"'] := by
  rw [canonStr_toList, flatMap_cesc_id s.toList (fun c hc => by
    have h1 := (plainStr_chars hp) c hc
    simp only [Bool.and_eq_true, bne_iff_ne, ne_eq, decide_eq_true_eq] at h1
    refine ⟨h1.1.1, h1.1.2, ?_, ha c hc⟩
    have hb : (0x20 : UInt32).toNat = 32 := by decide
    have h2 := UInt32.le_iff_toNat_le.1 h1.2
    rw [hb] at h2
    show (0x20 : Nat) ≤ c.toNat
    exact h2)]

/-- `toString` of a nonnegative integer is `Nat.toDigits` of it. -/
theorem num_toList {m : Int} (hm : 0 ≤ m) : (toString m).toList = Nat.toDigits 10 m.toNat := by
  rw [Int.toString_eq_repr, Int.repr_eq_ite, if_pos hm]
  show (String.ofList (Nat.toDigits 10 m.toNat)).toList = _
  rw [String.toList_ofList]

theorem mem_intercalate {α : Type} {sep : List α} {x : List α} {c : α} : ∀ (L : List (List α)), x ∈ L → c ∈ x →
    c ∈ sep.intercalate L
  | [], hx, _ => by simp at hx
  | [x'], hx, hc => by
    rw [List.intercalate_singleton]; cases List.mem_singleton.1 hx; exact hc
  | x' :: y :: L, hx, hc => by
    rw [List.intercalate_cons_cons]
    rcases List.mem_cons.1 hx with rfl | hx'
    · exact List.mem_append_left _ (List.mem_append_left _ hc)
    · exact List.mem_append_right _ (mem_intercalate (y :: L) hx' hc)

/-- Printability of a value's text gives printability of each array element's text. -/
theorem ascii_arr {a : List Json} (ha : ∀ c ∈ Flock.ZkServer.jsonChars (.arr ⟨a⟩), c.toNat < 0x7f) :
    ∀ e ∈ a, ∀ c ∈ Flock.ZkServer.jsonChars e, c.toNat < 0x7f := by
  intro e he c hc
  apply ha
  rw [Flock.ZkServer.jsonChars]
  refine List.mem_append_left _ (List.mem_cons_of_mem _ ?_)
  exact mem_intercalate _ (by rw [jsonCharsList_eq]; exact List.mem_map_of_mem he) hc

/-- Printability of a value's text gives printability of each object member's key and value text. -/
theorem ascii_obj {t : Impl String (fun _ => Json)}
    (ha : ∀ c ∈ Flock.ZkServer.jsonChars (.obj ⟨⟨t⟩⟩), c.toNat < 0x7f) :
    ∀ e ∈ t.toListModel, (∀ c ∈ e.1.toList, c.toNat < 0x7f) ∧ (∀ c ∈ Flock.ZkServer.jsonChars e.2, c.toNat < 0x7f) := by
  intro e he
  have hmem : ∀ c ∈ (('"' :: e.1.toList ++ ['"', ':']) ++ Flock.ZkServer.jsonChars e.2), c.toNat < 0x7f := by
    intro c hc
    apply ha
    rw [Flock.ZkServer.jsonChars]
    refine List.mem_append_left _ (List.mem_cons_of_mem _ ?_)
    exact mem_intercalate _ (by rw [jsonCharsObj_eq]; exact List.mem_map_of_mem he) hc
  refine ⟨fun c hc => hmem c ?_, fun c hc => hmem c (List.mem_append_right _ hc)⟩
  exact List.mem_append_left _ (List.mem_cons_of_mem _ (List.mem_append_left _ hc))

theorem cj_toList : ∀ (k : Nat) (j : Json), sizeOf j < k → plain j = true →
    (∀ c ∈ Flock.ZkServer.jsonChars j, c.toNat < 0x7f) → (cjChars j).toList = Flock.ZkServer.jsonChars j
  | 0, _, h, _, _ => absurd h (Nat.not_lt_zero _)
  | k + 1, j, hk, hp, ha => by
    match j with
    | .null => rfl
    | .bool true => rfl
    | .bool false => rfl
    | .num ⟨m, e⟩ =>
      simp only [plain, Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at hp
      rw [cjChars, Flock.ZkServer.jsonChars, num_toList hp.2]
    | .str s =>
      simp only [plain] at hp
      rw [cjChars, Flock.ZkServer.jsonChars, canonStr_verbatim hp (by
        intro c hc
        apply ha
        rw [Flock.ZkServer.jsonChars]
        exact List.mem_cons_of_mem _ (List.mem_append_left _ hc))]
    | .arr ⟨a⟩ =>
      simp only [plain, plainList_iff] at hp
      rw [cjChars, Flock.ZkServer.jsonChars]
      simp only [String.toList_append, String.toList_intercalate]
      have hlist : (cjList a).map String.toList = Flock.ZkServer.jsonCharsList a := by
        rw [cjList_eq, jsonCharsList_eq, List.map_map]
        apply List.map_congr_left
        intro e he
        have hlt : sizeOf e < k := by
          have hmem : e ∈ (⟨a⟩ : Array Json) := by simpa using he
          have := Array.sizeOf_lt_of_mem hmem
          simp only [Json.arr.sizeOf_spec] at hk
          omega
        exact cj_toList k e hlt (hp e he) (ascii_arr ha e he)
      rw [hlist]
      rfl
    | .obj ⟨⟨t⟩⟩ =>
      simp only [plain, Bool.and_eq_true] at hp
      rw [cjChars, Flock.ZkServer.jsonChars]
      simp only [String.toList_append, String.toList_intercalate]
      have hlist : (cjObj t).map String.toList = Flock.ZkServer.jsonCharsObj t := by
        rw [cjObj_eq, jsonCharsObj_eq, List.map_map]
        apply List.map_congr_left
        intro e he
        simp only [Function.comp_apply]
        have hlt : sizeOf e.2 < k := by
          have hs : sizeOf e.2 < sizeOf (Json.obj (⟨⟨t⟩⟩ : Std.TreeMap.Raw String Json)) :=
            Flock.sizeOf_lt_of_mem_toArray (p := (e.1, e.2))
              (by rw [← Array.mem_toList_iff, FlockSoundness.Discharge.Recursion.ZkServerObj.toArray_toList]
                  exact List.mem_map_of_mem he)
          omega
        have hav := ascii_obj ha e he
        rw [String.toList_append, String.toList_append, canonStr_verbatim ((plainObj_iff t).1 hp.1 e he).1 hav.1,
          cj_toList k e.2 hlt ((plainObj_iff t).1 hp.1 e he).2 hav.2]
        show '"' :: e.1.toList ++ ['"'] ++ [':'] ++ Flock.ZkServer.jsonChars e.2
          = ('"' :: e.1.toList ++ ['"', ':']) ++ Flock.ZkServer.jsonChars e.2
        simp
      rw [hlist]
      rfl

/-- **`canon` of a plain, printable value is its `printJson` text.** -/
theorem canon_plain {j : Json} (hp : plain j = true)
    (ha : ∀ c ∈ (Flock.ZkServer.printJson j).toList, c.toNat < 0x7f) :
    Flock.canon j = .ok (Flock.ZkServer.printJson j) := by
  rw [canon_cj (sizeOf j + 1) j (Nat.lt_succ_self _) hp]
  congr 1
  rw [← String.ofList_toList (s := cjChars j),
    cj_toList (sizeOf j + 1) j (Nat.lt_succ_self _) hp (by
      rw [Flock.ZkServer.printJson, String.toList_ofList] at ha; exact ha)]
  rfl

/-! ## `canonicalJson_canon`: a canonical text of ASCII parses back to `norm` of its value

`cjc` is `jsonChars` with each string and key escaped as `canon` writes it (`cesc`). The parser `JsonParse.str` reads an
escaped run back to its characters unconditionally (`esc_of`/`cesc_esc`), so `cjReads` mirrors ZkServerJson's `reads`
with the string case served by `esc_of` instead of `str_plain`. -/

open Flock.JsonParse
open FlockComplete.JsonRT.ReadBack (esc_of cesc_esc)

/-- A string's text between its quotes, as `canon` writes it. -/
def escK (s : String) : List Char := s.toList.flatMap fun c => (cesc c).toList

mutual
/-- `jsonChars` with each string and key escaped as `canon` writes it. -/
def cjc : Json → List Char
  | .null => ['n', 'u', 'l', 'l']
  | .bool true => ['t', 'r', 'u', 'e']
  | .bool false => ['f', 'a', 'l', 's', 'e']
  | .num ⟨m, _⟩ => Nat.toDigits 10 m.toNat
  | .str s => '"' :: escK s ++ ['"']
  | .arr ⟨a⟩ => '[' :: [','].intercalate (cjcList a) ++ [']']
  | .obj ⟨⟨t⟩⟩ => '{' :: [','].intercalate (cjcObj t) ++ ['}']
def cjcList : List Json → List (List Char)
  | [] => []
  | j :: js => cjc j :: cjcList js
def cjcObj : Impl String (fun _ => Json) → List (List Char)
  | .leaf => []
  | .inner _ k v l r => cjcObj l ++ [('"' :: escK k ++ ['"', ':']) ++ cjc v] ++ cjcObj r
end

/-- An object member's text, key escaped. -/
def cjMember (e : (_ : String) × Json) : List Char := ('"' :: escK e.1 ++ ['"', ':']) ++ cjc e.2

theorem cjcList_eq : ∀ a : List Json, cjcList a = a.map cjc
  | [] => rfl
  | j :: js => by rw [cjcList, cjcList_eq js]; rfl

theorem cjcObj_eq : ∀ t : Impl String (fun _ => Json), cjcObj t = t.toListModel.map cjMember
  | .leaf => rfl
  | .inner _ k v l r => by
    rw [cjcObj, cjcObj_eq l, cjcObj_eq r, Impl.toListModel_inner]
    simp [cjMember]

/-- `canon`'s text of a plain value, as a character list, is `cjc`. -/
theorem cjc_eq : ∀ (k : Nat) (j : Json), sizeOf j < k → plain j = true → (cjChars j).toList = cjc j
  | 0, _, h, _ => absurd h (Nat.not_lt_zero _)
  | k + 1, j, hk, hp => by
    match j with
    | .null => rfl
    | .bool true => rfl
    | .bool false => rfl
    | .num ⟨m, e⟩ =>
      simp only [plain, Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at hp
      rw [cjChars, cjc, num_toList hp.2]
    | .str s =>
      rw [cjChars, cjc, canonStr_toList]; rfl
    | .arr ⟨a⟩ =>
      simp only [plain, plainList_iff] at hp
      rw [cjChars, cjc]
      simp only [String.toList_append, String.toList_intercalate]
      have hlist : (cjList a).map String.toList = cjcList a := by
        rw [cjList_eq, cjcList_eq, List.map_map]
        apply List.map_congr_left
        intro e he
        simp only [Function.comp_apply]
        have hlt : sizeOf e < k := by
          have hmem : e ∈ (⟨a⟩ : Array Json) := by simpa using he
          have := Array.sizeOf_lt_of_mem hmem
          simp only [Json.arr.sizeOf_spec] at hk
          omega
        exact cjc_eq k e hlt (hp e he)
      rw [hlist]
      rfl
    | .obj ⟨⟨t⟩⟩ =>
      simp only [plain, Bool.and_eq_true] at hp
      rw [cjChars, cjc]
      simp only [String.toList_append, String.toList_intercalate]
      have hlist : (cjObj t).map String.toList = cjcObj t := by
        rw [cjObj_eq, cjcObj_eq, List.map_map]
        apply List.map_congr_left
        intro e he
        simp only [Function.comp_apply]
        have hlt : sizeOf e.2 < k := by
          have hs : sizeOf e.2 < sizeOf (Json.obj (⟨⟨t⟩⟩ : Std.TreeMap.Raw String Json)) :=
            Flock.sizeOf_lt_of_mem_toArray (p := (e.1, e.2))
              (by rw [← Array.mem_toList_iff, FlockSoundness.Discharge.Recursion.ZkServerObj.toArray_toList]
                  exact List.mem_map_of_mem he)
          omega
        rw [String.toList_append, String.toList_append, canonStr_toList,
          cjc_eq k e.2 hlt ((plainObj_iff t).1 hp.1 e he).2]
        show ('"' :: escK e.1 ++ ['"']) ++ [':'] ++ cjc e.2 = cjMember e
        rw [cjMember]; simp
      rw [hlist]
      rfl

/-- The first character of `cjc j` leads a value, as the parser's whitespace skip expects. -/
theorem cjc_head (j : Json) : ∃ c cs, cjc j = c :: cs ∧ Lead c := by
  match j with
  | .null => exact ⟨_, _, rfl, .inl rfl⟩
  | .bool true => exact ⟨_, _, rfl, .inr (.inl rfl)⟩
  | .bool false => exact ⟨_, _, rfl, .inr (.inr (.inl rfl))⟩
  | .num ⟨m, _⟩ =>
    rcases Nat.eq_zero_or_pos m.toNat with h | h
    · exact ⟨'0', [], by rw [cjc, h, Nat.toDigits_zero],
        .inr (.inr (.inr (.inr (.inr (.inr ⟨0, by decide, rfl⟩)))))⟩
    · obtain ⟨d, cs, _, hd, he⟩ := toDigits_head _ h
      exact ⟨_, cs, by rw [cjc, he], .inr (.inr (.inr (.inr (.inr (.inr ⟨d, hd, rfl⟩)))))⟩
  | .str _ => exact ⟨_, _, by rw [cjc, List.cons_append], .inr (.inr (.inr (.inl rfl)))⟩
  | .arr ⟨_⟩ => exact ⟨_, _, by rw [cjc, List.cons_append], .inr (.inr (.inr (.inr (.inl rfl))))⟩
  | .obj ⟨⟨_⟩⟩ => exact ⟨_, _, by rw [cjc, List.cons_append], .inr (.inr (.inr (.inr (.inr (.inl rfl)))))⟩

theorem ws_cjc (j : Json) (r : List Char) (o : Nat) : ws (cjc j ++ r) o = (cjc j ++ r, o) := by
  obtain ⟨c, cs, he, hl⟩ := cjc_head j
  rw [he, List.cons_append, ws_cons _ _ (lead_ne hl).1]

theorem ws_cjc_joined (j : Json) (l : List (List Char)) (r : List Char) (o : Nat) :
    ws ([','].intercalate (cjc j :: l) ++ r) o = ([','].intercalate (cjc j :: l) ++ r, o) := by
  obtain ⟨c, cs, he, hl⟩ := cjc_head j
  rw [he, List.intercalate_cons_cons_left, List.cons_append, ws_cons _ _ (lead_ne hl).1]

/-- The escaped string parse: `str` reads a key or string's escaped run back to the string. -/
theorem str_canon (s : String) (k : Nat) (acc : String) (rest : List Char) (o : Nat) (hk : (escK s).length < k) :
    ∃ o', str k acc (escK s ++ '"' :: rest) o = .ok rest o' (acc ++ s) :=
  esc_of (fun c => (cesc c).toList) cesc_esc s k acc rest o hk

/-- `value` at fuel `k` reads each plain value whose `cjc` text is shorter than `k`, to `norm` of it. -/
def CjReads (k : Nat) : Prop :=
  ∀ j, plain j = true → (cjc j).length < k → ∀ rest, Follow rest → ∀ o,
    ∃ o', value k (cjc j ++ rest) o = .ok rest o' (norm j)

theorem cj_array_loop {K : Nat} (IH : ∀ m, m < K → CjReads m) : ∀ (l : List Json) (k : Nat) (acc : Array Json)
    (rest : List Char) (o : Nat), k ≤ K → l ≠ [] → (∀ e ∈ l, plain e = true) →
    ([','].intercalate (l.map cjc) ++ [']']).length < k → Follow rest →
    ∃ o', array k acc ([','].intercalate (l.map cjc) ++ ']' :: rest) o = .ok rest o' (acc ++ (l.map norm).toArray)
  | [], _, _, _, _, _, h, _, _, _ => absurd rfl h
  | [e], k + 1, acc, rest, o, hk, _, hp, hl, hr => by
    simp only [List.map_cons, List.map_nil, List.intercalate_singleton, List.length_append, List.length_singleton]
      at hl ⊢
    obtain ⟨o1, h1⟩ := IH k (by omega) e (hp e List.mem_cons_self) (by omega) (']' :: rest)
      (follow_cons _ (by simp)) o
    refine ⟨o1 + 1, ?_⟩
    simp only [array, h1, ↓reduceIte, ws_follow hr]
    simp
  | e :: e' :: es, k + 1, acc, rest, o, hk, _, hp, hl, hr => by
    simp only [List.map_cons, List.intercalate_cons_cons, List.length_append, List.length_singleton] at hl
    have he : [','].intercalate (cjc e :: cjc e' :: es.map cjc) ++ ']' :: rest =
        cjc e ++ (',' :: ([','].intercalate (cjc e' :: es.map cjc) ++ ']' :: rest)) := by simp
    obtain ⟨o1, h1⟩ := IH k (by omega) e (hp e List.mem_cons_self) (by omega)
      (',' :: ([','].intercalate (cjc e' :: es.map cjc) ++ ']' :: rest)) (follow_cons _ (by simp)) o
    have hl' : ([','].intercalate ((e' :: es).map cjc) ++ [']']).length < k := by
      simp only [List.map_cons, List.length_append, List.length_singleton]
      omega
    obtain ⟨o2, h2⟩ := cj_array_loop IH (e' :: es) k (acc.push (norm e)) rest (o1 + 1) (by omega) (by simp)
      (fun x hx => hp x (List.mem_cons_of_mem _ hx)) hl' hr
    refine ⟨o2, ?_⟩
    simp only [List.map_cons] at h2 ⊢
    rw [he]
    simp only [array, h1, show (',' = ']') = False by decide, ↓reduceIte, ws_cjc_joined, h2]
    simp

theorem ws_cj_members (e : (_ : String) × Json) (l : List (List Char)) (r : List Char) (o : Nat) :
    ws ([','].intercalate (cjMember e :: l) ++ r) o = ([','].intercalate (cjMember e :: l) ++ r, o) := by
  rw [show cjMember e = '"' :: (escK e.1 ++ ['"', ':'] ++ cjc e.2) by simp [cjMember],
    List.intercalate_cons_cons_left, List.cons_append, ws_cons _ _ (by decide)]

theorem cj_object_loop {K : Nat} (IH : ∀ m, m < K → CjReads m) : ∀ (l : List ((_ : String) × Json)) (k : Nat)
    (kvs : Std.TreeMap.Raw String Json) (rest : List Char) (o : Nat), k ≤ K → l ≠ [] →
    (∀ e ∈ l, plain e.2 = true) →
    ([','].intercalate (l.map cjMember) ++ ['}']).length < k → Follow rest →
    ∃ o', object k kvs ([','].intercalate (l.map cjMember) ++ '}' :: rest) o =
      .ok rest o' (l.foldl (fun a e => a.insert e.1 (norm e.2)) kvs)
  | [], _, _, _, _, _, h, _, _, _ => absurd rfl h
  | [e], k + 1, kvs, rest, o, hk, _, hp, hl, hr => by
    have hv := hp e List.mem_cons_self
    simp only [List.map_cons, List.map_nil, List.intercalate_singleton] at hl ⊢
    have hlen := hl
    simp only [cjMember, List.length_append, List.length_cons, List.length_nil] at hlen
    obtain ⟨o1, h1⟩ := str_canon e.1 k "" (':' :: (cjc e.2 ++ '}' :: rest)) (o + 1) (by omega)
    obtain ⟨o2, h2⟩ := IH k (by omega) e.2 hv (by omega) ('}' :: rest) (follow_cons _ (by simp)) (o1 + 1)
    refine ⟨o2 + 1, ?_⟩
    rw [show cjMember e ++ '}' :: rest = '"' :: (escK e.1 ++ '"' :: ':' :: (cjc e.2 ++ '}' :: rest)) by
      simp [cjMember]]
    simp only [object, h1, String.empty_append, ws_colon, ws_cjc, h2, ↓reduceIte, ws_follow hr, List.foldl_cons,
      List.foldl_nil]
  | e :: e' :: es, k + 1, kvs, rest, o, hk, _, hp, hl, hr => by
    have hv := hp e List.mem_cons_self
    simp only [List.map_cons, List.intercalate_cons_cons] at hl ⊢
    have hlen := hl
    simp only [cjMember, List.length_append, List.length_cons, List.length_nil] at hlen
    let more := [','].intercalate (cjMember e' :: es.map cjMember) ++ '}' :: rest
    obtain ⟨o1, h1⟩ := str_canon e.1 k "" (':' :: (cjc e.2 ++ ',' :: more)) (o + 1) (by omega)
    obtain ⟨o2, h2⟩ := IH k (by omega) e.2 hv (by omega) (',' :: more) (follow_cons _ (by simp)) (o1 + 1)
    have hl' : ([','].intercalate ((e' :: es).map cjMember) ++ ['}']).length < k := by
      simp only [List.map_cons, List.length_append, List.length_singleton] at hl ⊢
      omega
    obtain ⟨o3, h3⟩ := cj_object_loop IH (e' :: es) k (kvs.insert e.1 (norm e.2)) rest (o2 + 1) (by omega) (by simp)
      (fun x hx => hp x (List.mem_cons_of_mem _ hx)) hl' hr
    refine ⟨o3, ?_⟩
    rw [show cjMember e ++ [','] ++ [','].intercalate (cjMember e' :: es.map cjMember) ++ '}' :: rest =
      '"' :: (escK e.1 ++ '"' :: ':' :: (cjc e.2 ++ ',' :: more)) by simp [cjMember, more]]
    simp only [List.map_cons] at h3
    simp only [object, h1, String.empty_append, ws_colon, ws_cjc, h2, show (',' = '}') = False by decide, ↓reduceIte,
      more, ws_cj_members, h3, List.foldl_cons]

/-- `value` reads a plain value's `cjc` text back at every fuel that exceeds its length. -/
theorem cjReads : ∀ k, CjReads k := by
  intro k
  induction k using Nat.strongRecOn with
  | ind k IH =>
    intro j hp hl rest hr o
    cases k with
    | zero => exact absurd hl (Nat.not_lt_zero _)
    | succ k =>
    match j with
    | .null => exact ⟨_, by simp [cjc, Flock.JsonParse.value, Flock.JsonParse.word, Flock.JsonParse.lit, ws_follow hr, fnorm] <;> rfl⟩
    | .bool true => exact ⟨_, by simp [cjc, Flock.JsonParse.value, Flock.JsonParse.word, Flock.JsonParse.lit, ws_follow hr, fnorm] <;> rfl⟩
    | .bool false => exact ⟨_, by simp [cjc, Flock.JsonParse.value, Flock.JsonParse.word, Flock.JsonParse.lit, ws_follow hr, fnorm] <;> rfl⟩
    | .num ⟨m, e⟩ =>
      simp only [plain, Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at hp
      obtain ⟨rfl, hm⟩ := hp
      obtain ⟨o', ho⟩ := value_num m.toNat k hr o
      refine ⟨o', ?_⟩
      rw [cjc, ho, fnorm, JsonNumber.fromInt, Int.toNat_of_nonneg hm]
    | .str s =>
      simp only [cjc, List.length_cons, List.length_append, List.length_singleton] at hl
      obtain ⟨o1, h1⟩ := str_canon s k "" rest (o + 1) (by omega)
      refine ⟨o1, ?_⟩
      simp only [cjc, List.cons_append, List.append_assoc]
      simp [Flock.JsonParse.value, h1, ws_follow hr, fnorm]
    | .arr ⟨a⟩ =>
      simp only [plain, plainList_iff] at hp
      simp only [cjc, cjcList_eq] at hl ⊢
      cases a with
      | nil =>
        refine ⟨o + 1 + 1, ?_⟩
        simp [Flock.JsonParse.value, ws_cons _ _ (show ¬(']' = '\t' ∨ ']' = '\n' ∨ ']' = '\r' ∨ ']' = ' ') by decide),
          ws_follow hr, fnorm, normList]
      | cons e es =>
        have hl' : ([','].intercalate ((e :: es).map cjc) ++ [']']).length < k := by
          simp only [List.length_cons, List.length_append] at hl ⊢
          omega
        obtain ⟨o', ho⟩ := cj_array_loop IH (e :: es) k #[] rest (o + 1) (by omega) (by simp) hp hl' hr
        refine ⟨o', ?_⟩
        obtain ⟨c, cs, hc, hlead⟩ := cjc_head e
        have hne := (lead_ne hlead).2
        simp only [List.map_cons, hc, List.intercalate_cons_cons_left] at ho ⊢
        simp only [List.cons_append, List.append_assoc, List.nil_append] at ho ⊢
        simp only [Flock.JsonParse.value, ↓reduceIte, ws_cons _ _ (lead_ne hlead).1]
        split
        · rename_i h; simp at h
        · rename_i h; simp only [Prod.mk.injEq, List.cons.injEq] at h; exact absurd h.1.1 hne
        · rename_i heq
          simp only [Prod.mk.injEq] at heq
          obtain ⟨rfl, rfl⟩ := heq
          simp [ho, fnorm, normList_eq]
    | .obj ⟨⟨t⟩⟩ =>
      simp only [plain, Bool.and_eq_true, plainObj_iff] at hp
      simp only [cjc, cjcObj_eq] at hl ⊢
      rw [fnorm, normObj_eq]
      cases ht : t.toListModel with
      | nil =>
        refine ⟨o + 1 + 1, ?_⟩
        simp [Flock.JsonParse.value, ws_cons _ _ (show ¬('}' = '\t' ∨ '}' = '\n' ∨ '}' = '\r' ∨ '}' = ' ') by decide),
          ws_follow hr]
      | cons e es =>
        rw [ht] at hl hp
        have hl' : ([','].intercalate ((e :: es).map cjMember) ++ ['}']).length < k := by
          simp only [List.length_cons, List.length_append] at hl ⊢
          omega
        obtain ⟨o', ho⟩ := cj_object_loop IH (e :: es) k ∅ rest (o + 1) (by omega) (by simp)
          (fun x hx => (hp.1 x hx).2) hl' hr
        refine ⟨o', ?_⟩
        simp only [List.map_cons, show cjMember e = '"' :: (escK e.1 ++ ['"', ':'] ++ cjc e.2) by
          simp [cjMember], List.intercalate_cons_cons_left] at ho ⊢
        simp only [List.cons_append, List.append_assoc, List.nil_append, List.foldl_cons] at ho ⊢
        simp [Flock.JsonParse.value, ws_cons _ _ (show ¬('"' = '\t' ∨ '"' = '\n' ∨ '"' = '\r' ∨ '"' = ' ') by decide), ho]

/-- A plain value's `cjc` text parses back to `norm` of it. -/
theorem parse_cjc {j : Json} (h : plain j = true) : parse (String.ofList (cjc j)) = .ok (norm j) := by
  rw [parse_eq, String.toList_ofList]
  obtain ⟨o', ho⟩ := cjReads (2 * (cjc j).length + 2) j h (by omega) [] follow_nil 0
  rw [List.append_nil] at ho
  have hws : ws (cjc j) 0 = (cjc j, 0) := by simpa using ws_cjc j [] 0
  simp only [parseList, hws, ho]

/-- **A canonical text of ASCII parses back to `norm` of its value.** -/
theorem canonicalJson_canon {j : Json} {s : String} (hp : plain j = true)
    (hascii : (s.toUTF8.data.all (· < 0x7f)) = true) (hc : Flock.canon j = .ok s) :
    Core.canonicalJson s.toUTF8 = .ok (norm j) := by
  have hs : s = cjChars j := by
    have h1 := canon_cj (sizeOf j + 1) j (Nat.lt_succ_self _) hp
    rw [hc] at h1; exact Except.ok.inj h1
  have hsl : s.toList = cjc j := by rw [hs]; exact cjc_eq (sizeOf j + 1) j (Nat.lt_succ_self _) hp
  have hparse : parse s = .ok (norm j) := by
    rw [show s = String.ofList (cjc j) from by rw [← hsl, String.ofList_toList]]
    exact parse_cjc hp
  have hcn : Flock.canon (fnorm j) = .ok s := by rw [canon_norm hp, hc]
  unfold Core.canonicalJson
  simp only [hascii, Properties.CanonRoundTrip.fromUTF8?_toUTF8, hparse, hcn, bind, Except.bind, pure, Except.pure,
    bne_self_eq_false, Bool.false_eq_true, reduceIte]

end Verity.Sampled.Instance.HonestSession.JsonCanon.Facts

namespace Verity.Sampled.Instance.HonestSession.JsonCanon

/-- **The canonical-JSON reader's three facts, proved.** -/
theorem jsonFacts : JsonFacts where
  canon_plain := Facts.canon_plain
  canon_norm := Facts.canon_norm
  canonicalJson_canon := Facts.canonicalJson_canon

end Verity.Sampled.Instance.HonestSession.JsonCanon
