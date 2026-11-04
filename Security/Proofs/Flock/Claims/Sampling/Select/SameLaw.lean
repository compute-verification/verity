import Batteries.Tactic.OpenPrivate
import Proofs.Flock.Claims.Complete.Json.Canon
import Proofs.Flock.Claims.Complete.Json.Record
import Proofs.Flock.Claims.Sampling.Select.LetWalk
import Proofs.Flock.Claims.Sampling.Select.Steps
import Proofs.Pouw.Period.Sort
import SampledProofs.OneStage

/-!
# The record's law has one profile (`Properties.SampledProofs.SameLawProfile`)

`Draw.sameLaw` compares canonical bytes, and `Record.profileDraw` reads an object's keys by lookup. Two laws with one
canonical text parse back to one value (`FlockComplete.JsonRT.ReadBack.pr_parse`), and the profile reads alike values
alike (`SameLaw.profileDraw_same`), so the record's law, the rule's expanded law and the verifier's have one profile.
The verifier's law is one `OneStage.rule` accepts as it stands: its keys are exactly the ones the rule names, each found
by lookup, so its rebuilt tree (`SameLaw.raw_text`) prints its text and reads its profile.

* `SameLaw.profileDraw_same`: alike values have one profile.
* `SameLaw.raw_text`: a law `OneStage.rule` accepts as it stands prints a text of an alike well-formed object with its
  profile.
* `paramsOfJson_rule`: the verifier's law, as `OneStage.paramsOfJson` keeps it, is one `OneStage.rule` accepts as it
  stands, which discharges `SameLawProfile`'s premise.
* `sameLawProfile`; `sameLawProfile_honest` and `sameLawProfile_nontrivial`, its two witnesses.
* `nat?_of_jeq_toJson`: a value with the canonical bytes of `toJson n` reads as `n`.
-/

open private SampledProofs.OneStage.exactly from SampledProofs.OneStage

open SampledProofs

namespace Properties.SampledProofs.SecurityProofs

open Lean (Json JsonNumber)
open Claims.Json (Same Good)
open FlockComplete.JsonRT (Distinct good_mkObj good_toJson good_arr_of get?_mkObj get?_mkObj_none same_num)
open Properties.Pouw.Period.SecurityProofs.Sort (qsort_of_sorted)

namespace SameLaw

/-! ## Reading alike values alike -/

theorem getD_obj (m : Std.TreeMap.Raw String Json compare) (k : String) :
    getD (.obj m) k = (m.get? k).getD .null := by
  unfold getD field?
  simp only [Json.getObjVal?]
  cases m.get? k <;> rfl

theorem getD_same {a b : Json} (h : Same a b) (k : String) : Same (getD a k) (getD b k) := by
  cases h with
  | @obj m n hm hn hk hv =>
    rw [getD_obj, getD_obj]
    have e := hk k
    cases hm' : m.get? k with
    | none =>
      cases hn' : n.get? k with
      | none => exact .null
      | some y => rw [hm', hn'] at e; simp at e
    | some x =>
      cases hn' : n.get? k with
      | none => rw [hm', hn'] at e; simp at e
      | some y => exact hv k x y hm' hn'
  | _ => exact .null

def strOf : Json → Option String
  | .str s => some s
  | _ => none

theorem strOf_same {a b : Json} (h : Same a b) : strOf a = strOf b := by cases h <;> rfl

theorem int?_same {a b : Json} (h : Same a b) : int? a = int? b := by cases h <;> rfl

theorem nat?_same {a b : Json} (h : Same a b) : nat? a = nat? b := by cases h <;> rfl

theorem list_mapM_congr {α β : Type} (f : α → Option β) :
    ∀ (l l' : List α), l.length = l'.length → (∀ (i : ℕ) (h₁ : i < l.length) (h₂ : i < l'.length), f l[i] = f l'[i]) →
      l.mapM f = l'.mapM f
  | [], [], _, _ => rfl
  | x :: l, y :: l', hl, h => by
    have h0 := h 0 (by simp) (by simp)
    simp only [List.getElem_cons_zero] at h0
    have ih := list_mapM_congr f l l' (by simpa using hl) fun i h₁ h₂ =>
      h (i + 1) (Nat.succ_lt_succ h₁) (Nat.succ_lt_succ h₂)
    rw [List.mapM_cons, List.mapM_cons, h0, ih]
  | [], _ :: _, hl, _ => by simp at hl
  | _ :: _, [], hl, _ => by simp at hl

theorem array_mapM_congr {α β : Type} (f : α → Option β) {a b : Array α} (hs : a.size = b.size)
    (he : ∀ (i : ℕ) (h₁ : i < a.size) (h₂ : i < b.size), f a[i] = f b[i]) : a.mapM f = b.mapM f := by
  rw [Array.mapM_eq_mapM_toList, Array.mapM_eq_mapM_toList,
    list_mapM_congr f a.toList b.toList (by simpa using hs) fun i h₁ h₂ => by
      simpa using he i (by simpa using h₁) (by simpa using h₂)]

theorem getElem!_same {a b : Array Json} (hs : a.size = b.size)
    (he : ∀ (i : ℕ) (h₁ : i < a.size) (h₂ : i < b.size), Same a[i] b[i]) (i : ℕ) : Same a[i]! b[i]! := by
  by_cases hi : i < a.size
  · rw [getElem!_pos a i hi, getElem!_pos b i (hs ▸ hi)]
    exact he i hi (hs ▸ hi)
  · rw [getElem!_neg a i hi, getElem!_neg b i (by omega)]
    exact .null

/-- `Draw.ranges?`'s reading of one range. -/
def readRange (r : Json) : Option (Int × Int) :=
  match r with
  | .arr p => do
    if p.size != 2 then none
    let lo ← int? p[0]!
    let hi ← int? p[1]!
    if lo < hi then some (lo, hi) else none
  | _ => none

def readRangeOf (n : ℕ) (lo hi : Option Int) : Option (Int × Int) := do
  if n != 2 then none
  let lo ← lo
  let hi ← hi
  if lo < hi then some (lo, hi) else none

theorem readRange_arr (p : Array Json) : readRange (.arr p) = readRangeOf p.size (int? p[0]!) (int? p[1]!) := rfl

theorem readRange_same {a b : Json} (h : Same a b) : readRange a = readRange b := by
  cases h with
  | @arr x y hs he =>
    rw [readRange_arr, readRange_arr, hs, int?_same (getElem!_same hs he 0), int?_same (getElem!_same hs he 1)]
  | _ => rfl

theorem ranges?_arr (x : Array Json) : Draw.ranges? (.arr x) = if x.isEmpty then none else x.mapM readRange := rfl

theorem ranges?_same {a b : Json} (h : Same a b) : Draw.ranges? a = Draw.ranges? b := by
  cases h with
  | @arr x y hs he =>
    rw [ranges?_arr, ranges?_arr, array_mapM_congr readRange hs fun i h₁ h₂ => readRange_same (he i h₁ h₂),
      show x.isEmpty = y.isEmpty by simp only [Array.isEmpty, hs]]
  | _ => rfl

/-- `Record.profileDraw`'s reading of one stratum. -/
def readStratum (s : Json) : Option Core.Stratum := do
  let .str t := getD s "template" | none
  let rs ← Draw.ranges? (getD s "units")
  return ({ name := t, size := (rs.foldl (fun (a : Int) ((lo, hi) : Int × Int) => a + (hi - lo)) 0).toNat,
            k := ← nat? (getD s "k") } : Core.Stratum)

def readStratumOf (t : Option String) (rs : Option (Array (Int × Int))) (k : Option ℕ) : Option Core.Stratum := do
  let t ← t
  let rs ← rs
  return ({ name := t, size := (rs.foldl (fun (a : Int) ((lo, hi) : Int × Int) => a + (hi - lo)) 0).toNat,
            k := ← k } : Core.Stratum)

theorem readStratum_eq (s : Json) :
    readStratum s = readStratumOf (strOf (getD s "template")) (Draw.ranges? (getD s "units")) (nat? (getD s "k")) := by
  unfold readStratum
  generalize getD s "template" = t
  cases t <;> rfl

theorem readStratum_same {a b : Json} (h : Same a b) : readStratum a = readStratum b := by
  rw [readStratum_eq a, readStratum_eq b, strOf_same (getD_same h "template"), ranges?_same (getD_same h "units"),
    nat?_same (getD_same h "k")]

def readStrata : Json → Option (Array Core.Stratum)
  | .arr ss => ss.mapM readStratum
  | _ => none

theorem readStrata_same {a b : Json} (h : Same a b) : readStrata a = readStrata b := by
  cases h with
  | @arr x y hs he => exact array_mapM_congr readStratum hs fun i h₁ h₂ => readStratum_same (he i h₁ h₂)
  | _ => rfl

/-- `Record.profileDraw` over what it reads of a law: its kind, its `k`, its `p` and its strata. -/
def readProfile (law : Json) (k : Option ℕ) (p : Option String) (st : Option (Array Core.Stratum)) :
    Option Core.SamplingDesign :=
  match law with
  | .str "subset" => k.map .subset
  | .str "bernoulli" => do
    let p ← p
    let (a, b) ← Draw.probability? p
    return .bernoulli (← Core.Ratio.mk' a b)
  | .str "stratified" | .str "work" => do
    let strata ← st
    return .stratified strata
  | _ => none

theorem profileDraw_read (x : Json) :
    Record.profileDraw x =
      readProfile (getD x "law") (nat? (getD x "k")) (strOf (getD x "p")) (readStrata (getD x "strata")) := by
  unfold Record.profileDraw
  generalize getD x "p" = p
  generalize getD x "strata" = s
  cases p <;> cases s <;> rfl

theorem readProfile_law {a b : Json} (h : Same a b) (k : Option ℕ) (p : Option String)
    (st : Option (Array Core.Stratum)) : readProfile a k p st = readProfile b k p st := by
  cases h <;> rfl

/-- **Alike values have one profile.** -/
theorem profileDraw_same {a b : Json} (h : Same a b) : Record.profileDraw a = Record.profileDraw b := by
  rw [profileDraw_read a, profileDraw_read b, readProfile_law (getD_same h "law"), nat?_same (getD_same h "k"),
    strOf_same (getD_same h "p"), readStrata_same (getD_same h "strata")]

/-! ## An object's tree and its listing -/

/-- A key a tree's lookup finds is listed with its value, in any tree, well-formed or not. -/
theorem raw_get?_mem {m : Std.TreeMap.Raw String Json compare} {k : String} {v : Json} (h : m.get? k = some v) :
    (k, v) ∈ m.toList := by
  obtain ⟨⟨t⟩⟩ := m
  simp only [Std.TreeMap.Raw.get?, Std.DTreeMap.Raw.Const.get?, Std.TreeMap.Raw.toList,
    Std.DTreeMap.Raw.Const.toList, Std.DTreeMap.Internal.Impl.Const.toList_eq_toListModel_map, List.mem_map] at h ⊢
  induction t with
  | leaf => simp [Std.DTreeMap.Internal.Impl.Const.get?] at h
  | inner sz k' v' l r ihl ihr =>
    simp only [Std.DTreeMap.Internal.Impl.Const.get?] at h
    simp only [Std.DTreeMap.Internal.Impl.toListModel_inner, List.mem_append, List.mem_cons]
    split at h
    · obtain ⟨e, he, hk⟩ := ihl h
      exact ⟨e, .inl he, hk⟩
    · obtain ⟨e, he, hk⟩ := ihr h
      exact ⟨e, .inr (.inr he), hk⟩
    · rename_i hc
      obtain rfl := Option.some.inj h
      obtain rfl := Std.LawfulEqOrd.eq_of_compare hc
      exact ⟨_, .inr (.inl rfl), rfl⟩

theorem val_unique {l : List (String × Json)} {k : String} {v w : Json} (hn : (l.map (·.1)).Nodup)
    (hv : (k, v) ∈ l) (hw : (k, w) ∈ l) : v = w := by
  induction l with
  | nil => cases hv
  | cons p l ih =>
    obtain ⟨k', x⟩ := p
    rw [List.map_cons, List.nodup_cons] at hn
    rcases List.mem_cons.1 hv with e | hv' <;> rcases List.mem_cons.1 hw with e' | hw'
    · cases e; cases e'; rfl
    · cases e
      exact absurd ((List.mem_map (f := fun p : String × Json => p.1)).2 ⟨_, hw', rfl⟩) hn.1
    · cases e'
      exact absurd ((List.mem_map (f := fun p : String × Json => p.1)).2 ⟨_, hv', rfl⟩) hn.1
    · exact ih hn.2 hv' hw'

theorem distinct_of_nodup {l : List (String × Json)} (h : (l.map (·.1)).Nodup) : Distinct l := by
  unfold List.Nodup at h
  rw [List.pairwise_map] at h
  exact h.imp fun hne e => hne (Std.LawfulEqOrd.eq_of_compare e)

theorem keys_perm_ofList {l : List (String × Json)} (hd : Distinct l) (hnd : (l.map (·.1)).Nodup) :
    (l.map (·.1)).Perm ((Std.TreeMap.Raw.ofList l compare).toList.map (·.1)) := by
  have hm : (Std.TreeMap.Raw.ofList l compare).WF := Std.TreeMap.Raw.WF.ofList
  rw [List.perm_ext_iff_of_nodup hnd (FlockComplete.JsonRT.ReadBack.keys_nodup hm)]
  intro k
  rw [FlockComplete.JsonRT.ReadBack.mem_keys hm]
  constructor
  · intro hk
    obtain ⟨⟨k', v⟩, hp, rfl⟩ := List.mem_map.1 hk
    rw [get?_mkObj hd hp]
    rfl
  · intro hs
    cases hc : (l.map Prod.fst).contains k with
    | true => exact List.contains_iff_mem.1 hc
    | false => rw [get?_mkObj_none hc] at hs; simp at hs

/-! ## The law `OneStage.rule` accepts as it stands -/

theorem good_of_nat? {x : Json} {n : ℕ} (h : nat? x = some n) : Good x ∧ x ≠ .null := by
  cases x with
  | num m =>
    obtain ⟨mm, e⟩ := m
    by_cases he : e = 0
    · subst he
      exact ⟨.num mm, by simp⟩
    · simp [nat?, int?, he] at h
  | _ => simp [nat?, int?] at h

theorem get?_of_getD {m : Std.TreeMap.Raw String Json compare} {k : String} {v : Json} (h : getD (.obj m) k = v)
    (hv : v ≠ .null) : m.get? k = some v := by
  rw [getD_obj] at h
  cases hg : m.get? k with
  | none => rw [hg] at h; exact absurd h.symm hv
  | some w => rw [hg] at h; exact congrArg some h

theorem exactly_ok {j : Json} {ks : Array String} {w : String} {u : Unit}
    (h : SampledProofs.OneStage.exactly j ks w = .ok u) : keys? j = some (sortStrings ks) := by
  unfold SampledProofs.OneStage.exactly at h
  split at h
  · exact absurd h (throw_ne _ _)
  · rename_i hc
    simpa using hc

theorem exactly_of {j : Json} {ks : Array String} {w : String} (h : keys? j = some (sortStrings ks)) :
    SampledProofs.OneStage.exactly j ks w = .ok () := by
  unfold SampledProofs.OneStage.exactly
  rw [h]
  simp [pure, Except.pure]

theorem keys_perm_of {j : Json} {ks : Array String} (h : keys? j = some (sortStrings ks)) :
    ∃ m : Std.TreeMap.Raw String Json compare, j = .obj m ∧ (m.toList.map (·.1)).Perm ks.toList := by
  cases j with
  | obj m =>
    refine ⟨m, rfl, ?_⟩
    have e : sortStrings (m.toArray.map (·.1)) = sortStrings ks := Option.some.inj h
    have h1 : (sortStrings (m.toArray.map (·.1))).toList.Perm (m.toArray.map (·.1)).toList :=
      Array.perm_iff_toList_perm.1 (FlockComplete.JsonRT.ReadBack.qsort_perm _ _)
    have h2 : (sortStrings ks).toList.Perm ks.toList :=
      Array.perm_iff_toList_perm.1 (FlockComplete.JsonRT.ReadBack.qsort_perm _ _)
    rw [e] at h1
    rw [← Std.TreeMap.Raw.toList_toArray, ← Array.toList_map]
    exact h1.symm.trans h2
  | _ => cases h

/-- An object whose keys are a list without repeats, each found by lookup at a printable value. -/
def Shape (law : Json) : Prop :=
  ∃ m : Std.TreeMap.Raw String Json compare, law = .obj m ∧ ∃ ks : List String, ks.Nodup ∧
    (m.toList.map (·.1)).Perm ks ∧ ∀ k ∈ ks, ∃ v, m.get? k = some v ∧ Good v

theorem shape_bern {law : Json} {w s : String} {u : Unit} (hl : getD law "law" = .str "bernoulli")
    (hk : SampledProofs.OneStage.exactly law #["law", "p"] w = .ok u) (hs : getD law "p" = .str s) : Shape law := by
  obtain ⟨m, rfl, hp⟩ := keys_perm_of (exactly_ok hk)
  refine ⟨m, rfl, ["law", "p"], by decide, hp, ?_⟩
  simp only [List.mem_cons, List.mem_nil_iff, or_false, forall_eq_or_imp, forall_eq]
  exact ⟨⟨_, get?_of_getD hl (by simp), .str _⟩, ⟨_, get?_of_getD hs (by simp), .str _⟩⟩

theorem shape_k {law : Json} {kind w : String} {u : Unit} (hl : getD law "law" = .str kind)
    (hk : SampledProofs.OneStage.exactly law #["k", "law"] w = .ok u) (hn : ¬ (nat? (getD law "k")).isNone = true) :
    Shape law := by
  obtain ⟨m, rfl, hp⟩ := keys_perm_of (exactly_ok hk)
  refine ⟨m, rfl, ["k", "law"], by decide, hp, ?_⟩
  simp only [List.mem_cons, List.mem_nil_iff, or_false, forall_eq_or_imp, forall_eq]
  refine ⟨?_, ⟨_, get?_of_getD hl (by simp), .str _⟩⟩
  cases hv : nat? (getD (.obj m) "k") with
  | none => rw [hv] at hn; exact absurd rfl hn
  | some x =>
    obtain ⟨hg, hne⟩ := good_of_nat? hv
    exact ⟨_, get?_of_getD rfl hne, hg⟩

theorem throw_bind_eq {α β : Type} (e : String) (f : α → Flock.V β) : ((throw e : Flock.V α) >>= f) = throw e := rfl

theorem rule_shape {law : Json} (h : OneStage.rule law = .ok law) : Shape law := by
  unfold OneStage.rule at h
  try dsimp only at h
  try simp only [throw_bind_eq] at h
  repeat' walk_step
  all_goals first
    | exact shape_bern (by assumption) (by assumption) (by assumption)
    | exact shape_k (by assumption) (by assumption) (by assumption)

theorem rule_eq {x y : Json} (h : OneStage.rule x = .ok y) : y = x := by
  unfold OneStage.rule at h
  try dsimp only at h
  try simp only [throw_bind_eq] at h
  repeat' walk_step
  all_goals
    simp only [pure, Except.pure, Except.ok.injEq] at h
    exact h.symm

theorem rule_self {x y : Json} (h : OneStage.rule x = .ok y) : OneStage.rule y = .ok y := by
  obtain rfl := rule_eq h
  exact h

/-! ## The printed text of a law -/

/-- `canon_pr`'s object case for a tree `m`, possibly not well-formed, whose listing `m'` holds whole: `m`'s text is a
text of `m'`. -/
theorem canon_pr_obj {m m' : Std.TreeMap.Raw String Json compare} (hm' : m'.WF)
    (hget : ∀ p ∈ m.toList, m'.get? p.1 = some p.2) (hkeys : (m.toList.map (·.1)).Perm (m'.toList.map (·.1)))
    (hg : ∀ p ∈ m.toList, Good p.2) (s : String) (hs : Flock.canon (.obj m) = .ok s) :
    FlockComplete.JsonRT.ReadBack.Pr (.obj m') s.toList := by
  rw [Flock.canon] at hs
  cases hp : m.toArray.attach.mapM (fun p => do pure (p.1.1, ← Flock.canon p.1.2)) with
  | error e => rw [hp] at hs; cases hs
  | ok vals =>
    rw [hp] at hs
    cases hs
    obtain ⟨hlen, hz⟩ := FlockComplete.JsonRT.ReadBack.array_mapM_zip _ _ _ hp
    have hpair : ∀ q ∈ m.toArray.attach.toList.zip vals.toList,
        q.2.1 = q.1.1.1 ∧ Flock.canon q.1.1.2 = .ok q.2.2 := by
      intro q hq
      have := hz q hq
      cases hc : Flock.canon q.1.1.2 with
      | error e => rw [hc] at this; cases this
      | ok sv =>
        rw [hc] at this
        have e : Except.ok (q.1.1.1, sv) = (Except.ok q.2 : Except String (String × String)) := this
        injection e with e
        rw [← e]
        exact ⟨rfl, rfl⟩
    have hw : ∀ w ∈ vals.toList, ∃ v, m'.get? w.1 = some v ∧ Flock.canon v = .ok w.2 ∧ Good v := by
      intro w hw'
      obtain ⟨i, hi, rfl⟩ := List.getElem_of_mem hw'
      have hi' : i < m.toArray.attach.toList.length := hlen ▸ hi
      have hq : (m.toArray.attach.toList[i], vals.toList[i]) ∈ m.toArray.attach.toList.zip vals.toList := by
        have := List.getElem_mem (l := m.toArray.attach.toList.zip vals.toList) (n := i)
          (by rw [List.length_zip]; exact Nat.lt_min.2 ⟨hi', hi⟩)
        simpa [List.getElem_zip] using this
      obtain ⟨e1, e2⟩ := hpair _ hq
      have hmem : (m.toArray.attach.toList[i]).1 ∈ m.toList := by
        rw [← Std.TreeMap.Raw.toList_toArray]
        exact Array.mem_toList_iff.2 (m.toArray.attach.toList[i]).2
      refine ⟨_, ?_, e2, hg _ hmem⟩
      rw [e1]
      exact hget _ hmem
    have hkv : vals.toList.map (·.1) = m.toList.map (·.1) := by
      rw [← Std.TreeMap.Raw.toList_toArray]
      have ha : m.toArray.attach.toList.map Subtype.val = m.toArray.toList := by
        simpa using List.attach_map_subtype_val m.toArray.toList
      rw [← ha, List.map_map]
      exact (FlockComplete.JsonRT.ReadBack.map_eq_of_zip (fun p => p.1.1) (·.1) _ _ (by omega)
        fun q hq => (hpair q hq).1.symm).symm
    let items := vals.qsort (fun a b => a.1 < b.1)
    have hperm : items.toList.Perm vals.toList :=
      Array.perm_iff_toList_perm.1 (FlockComplete.JsonRT.ReadBack.qsort_perm _ _)
    let fs := items.toList.map fun w =>
      (w.1, (m'.get? w.1).getD .null, FlockComplete.JsonRT.ReadBack.escK w.1, w.2.toList)
    have hfs : ∀ f ∈ fs, ∃ w ∈ vals.toList,
        f = (w.1, (m'.get? w.1).getD .null, FlockComplete.JsonRT.ReadBack.escK w.1, w.2.toList) := by
      intro f hf
      obtain ⟨w, hw', rfl⟩ := List.mem_map.1 hf
      exact ⟨w, hperm.mem_iff.1 hw', rfl⟩
    have e : ("{" ++ ",".intercalate (items.toList.map fun kv => Flock.canonStr kv.1 ++ ":" ++ kv.2) ++
        "}").toList = '{' :: [','].intercalate (fs.map FlockComplete.JsonRT.ReadBack.fieldT) ++ ['}'] := by
      simp only [fs, String.toList_append, String.toList_intercalate, List.map_map]
      show '{' :: _ ++ ['}'] = _
      simp only [List.cons_append]
      congr 2
      rw [show [].append _ = _ from List.nil_append _, show ",".toList = [','] from rfl]
      congr 1
      apply List.map_congr_left
      intro p _
      simp [FlockComplete.JsonRT.ReadBack.fieldT, FlockComplete.JsonRT.ReadBack.canonStr_toList,
        FlockComplete.JsonRT.ReadBack.escK]
    rw [e]
    refine .obj m' hm' fs (fun f hf => ?_) ?_ (fun f hf => ?_) (fun f hf => ?_)
    · obtain ⟨w, hw', rfl⟩ := hfs f hf
      obtain ⟨v, hv, -⟩ := hw w hw'
      show m'.get? w.1 = some ((m'.get? w.1).getD .null)
      rw [hv]
      rfl
    · refine List.Perm.trans ?_ hkeys
      rw [← hkv]
      simp only [fs, List.map_map]
      exact hperm.map _
    · obtain ⟨w, hw', rfl⟩ := hfs f hf
      exact FlockComplete.JsonRT.ReadBack.esc_of _ FlockComplete.JsonRT.ReadBack.cesc_esc _
    · obtain ⟨w, hw', rfl⟩ := hfs f hf
      obtain ⟨v, hv, hc, hgv⟩ := hw w hw'
      simp only [hv, Option.getD_some]
      exact FlockComplete.JsonRT.ReadBack.canon_pr hgv w.2 hc

/-- **A law `OneStage.rule` accepts as it stands prints a text of an alike well-formed object with its profile**: its
tree rebuilt from its listing, whose lookups find what the law's find. -/
theorem raw_text {law : Json} {t : String} (hr : OneStage.rule law = .ok law) (ht : Flock.canon law = .ok t) :
    ∃ W, FlockComplete.JsonRT.ReadBack.Pr W t.toList ∧ Record.profileDraw W = Record.profileDraw law := by
  obtain ⟨m, rfl, ks, hks, hperm, hkv⟩ := rule_shape hr
  have hnd : (m.toList.map (·.1)).Nodup := hperm.nodup_iff.2 hks
  have hd : Distinct m.toList := distinct_of_nodup hnd
  have hm' : (Std.TreeMap.Raw.ofList m.toList compare).WF := Std.TreeMap.Raw.WF.ofList
  have hext : ∀ k, (Std.TreeMap.Raw.ofList m.toList compare).get? k = m.get? k := by
    intro k
    cases hv : m.get? k with
    | some v => exact get?_mkObj hd (raw_get?_mem hv)
    | none =>
      cases hc : (m.toList.map Prod.fst).contains k with
      | false => exact get?_mkObj_none hc
      | true =>
        obtain ⟨⟨k', v⟩, hp, rfl⟩ := List.mem_map.1 (List.contains_iff_mem.1 hc)
        obtain ⟨w, hw, -⟩ := hkv _ (hperm.mem_iff.1 (List.mem_map.2 ⟨_, hp, rfl⟩))
        have := hv.symm.trans hw
        simp at this
  have hgood : ∀ p ∈ m.toList, Good p.2 := by
    intro p hp
    obtain ⟨v, hv, hg⟩ := hkv p.1 (hperm.mem_iff.1 (List.mem_map.2 ⟨p, hp, rfl⟩))
    have e : p.2 = v := val_unique hnd hp (raw_get?_mem hv)
    rw [e]
    exact hg
  refine ⟨.obj (Std.TreeMap.Raw.ofList m.toList compare),
    canon_pr_obj hm' (fun p hp => get?_mkObj hd hp) (keys_perm_ofList hd hnd) hgood t ht, ?_⟩
  have e : ∀ k, getD (.obj (Std.TreeMap.Raw.ofList m.toList compare)) k = getD (.obj m) k := fun k => by
    rw [getD_obj, getD_obj, hext]
  simp only [profileDraw_read, e]

/-! ## The laws `Draw.derive` writes print -/

theorem good_units (us : Array (ℕ × ℕ)) :
    Good (Json.arr (us.map fun (lo, hi) => Json.arr #[Lean.toJson lo, Lean.toJson hi])) :=
  good_arr_of fun x hx => by
    obtain ⟨⟨lo, hi⟩, -, rfl⟩ := Array.mem_map.1 hx
    exact good_arr_of fun y hy => by
      simp at hy
      rcases hy with rfl | rfl <;> exact good_toJson _

theorem good_flockStratum (s : Flock.Draw.Stratum) : Good s.toJson := by
  refine good_mkObj (by simp) ?_
  simp only [List.mem_cons, List.mem_nil_iff, or_false, forall_eq_or_imp, forall_eq]
  exact ⟨.str _, good_units _, good_toJson _⟩

theorem good_flockWorkStratum (s : Flock.Draw.WorkStratum) : Good s.toJson := by
  refine good_mkObj (by simp) ?_
  simp only [List.mem_cons, List.mem_nil_iff, or_false, forall_eq_or_imp, forall_eq]
  exact ⟨.str _, good_units _, good_toJson _, good_toJson _, good_toJson _⟩

theorem good_lawToJson (L : Flock.Draw.Law) : Good L.toJson := by
  cases L <;> refine good_mkObj (by simp) ?_ <;>
    simp only [List.mem_cons, List.mem_nil_iff, or_false, forall_eq_or_imp, forall_eq]
  · exact ⟨.str _, good_toJson _⟩
  · exact ⟨.str _, .str _⟩
  · refine ⟨.str _, good_toJson _, good_arr_of fun x hx => ?_⟩
    obtain ⟨s, -, rfl⟩ := Array.mem_map.1 hx
    exact good_flockStratum s
  · refine ⟨.str _, good_toJson _, good_arr_of fun x hx => ?_⟩
    obtain ⟨s, -, rfl⟩ := Array.mem_map.1 hx
    exact good_flockWorkStratum s

theorem good_parseLaw {t : String} {j : Json} (h : Select.parseLaw t = .ok j) : Good j := by
  unfold Select.parseLaw at h
  try dsimp only at h
  try simp only [throw_bind_eq, pure_bind] at h
  repeat' walk_step
  all_goals
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    refine good_mkObj (by simp) ?_
    simp only [List.mem_cons, List.mem_nil_iff, or_false, forall_eq_or_imp, forall_eq]
    exact ⟨.str _, by first | exact good_toJson _ | exact .str _⟩

theorem good_lawOver {rule : Select.Rule} {n : ℕ} {named : Json} (h : rule.lawOver n = .ok named) : Good named := by
  cases rule with
  | all =>
    obtain rfl : Json.mkObj [("law", "subset"), ("k", Lean.toJson n)] = named := Except.ok.inj h
    refine good_mkObj (by simp) ?_
    simp only [List.mem_cons, List.mem_nil_iff, or_false, forall_eq_or_imp, forall_eq]
    exact ⟨.str _, good_toJson _⟩
  | law t => exact good_parseLaw h

theorem derive_cases {d : Program} {query law out : Json} {table : Option Json}
    (h : Draw.derive d query law table = .ok out) : out = law ∨ ∃ L : Flock.Draw.Law, out = L.toJson := by
  unfold Draw.derive at h
  cases hl : Draw.deriveLaw d query law table with
  | error e => rw [hl] at h; cases h
  | ok r =>
    rw [hl] at h
    have e : (r.map (·.toJson)).getD law = out := Except.ok.inj h
    cases r with
    | none => exact .inl e.symm
    | some L => exact .inr ⟨L, e.symm⟩

theorem good_derive {d : Program} {query law out : Json} {table : Option Json} (hg : Good law)
    (h : Draw.derive d query law table = .ok out) : Good out := by
  rcases derive_cases h with rfl | ⟨L, rfl⟩
  · exact hg
  · exact good_lawToJson L

theorem canon_of_jeq {a b : Json} (h : jeq a b = true) :
    ∃ s, Flock.canon a = .ok s ∧ Flock.canon b = .ok s := by
  unfold jeq at h
  split at h
  · exact ⟨_, ‹Flock.canon a = _›, by rw [‹Flock.canon b = _›, eq_of_beq h]⟩
  · simp at h

theorem jeq_self {a : Json} (h : Good a) : jeq a a = true := by
  obtain ⟨s, hs⟩ := FlockComplete.JsonRT.canon_ok_some a h
  unfold jeq
  rw [hs]
  simp

theorem verifier_text {d : Program} {query law out : Json} {table : Option Json} {t : String}
    (hr : OneStage.rule law = .ok law) (hd : Draw.derive d query law table = .ok out)
    (ht : Flock.canon out = .ok t) :
    ∃ W, FlockComplete.JsonRT.ReadBack.Pr W t.toList ∧ Record.profileDraw W = Record.profileDraw out := by
  rcases derive_cases hd with rfl | ⟨L, rfl⟩
  · exact raw_text hr ht
  · exact ⟨_, FlockComplete.JsonRT.ReadBack.canon_pr (good_lawToJson L) t ht, rfl⟩

/-! ## A number's canonical bytes -/

theorem toJson_canon (n : ℕ) : Flock.canon (Lean.toJson n) = .ok (Int.repr (Int.ofNat n)) := by
  show Flock.canon (Json.num ⟨Int.ofNat n, 0⟩) = _
  rw [Flock.canon]
  rfl

theorem digits_head (n : ℕ) : ∃ d ds, (Int.repr (Int.ofNat n)).toList = d :: ds ∧ '0' ≤ d ∧ d ≤ '9' := by
  obtain ⟨c, cs, hc, h0, h9⟩ := FlockComplete.JsonRT.ReadBack.toDigits_ne n
  exact ⟨c, cs, by simp [Int.repr, Nat.repr, hc], h0, h9⟩

theorem lit_head (s mid tail : String) (c : Char) (hs : s.toList = [c]) :
    (s ++ mid ++ tail).toList = c :: (mid.toList ++ tail.toList) := by
  rw [String.toList_append, String.toList_append, hs]
  rfl

end SameLaw

open SameLaw

/-- **The verifier's law, as `OneStage.paramsParsed` keeps it, is one `OneStage.rule` accepts as it stands.** -/
theorem paramsParsed_rule {j : Json} {p : OneStage.Params} (h : OneStage.paramsParsed j = .ok p) :
    OneStage.rule p.verifier.law = .ok p.verifier.law := by
  revert p
  show FlockSoundness.Refine.Ret (OneStage.paramsParsed j) fun p => OneStage.rule p.verifier.law = .ok p.verifier.law
  rw [OneStage.paramsParsed.eq_def]
  ret_walk_let [OneStage.rule]
  all_goals
    subst_vars
    exact rule_self (by assumption)

/-- **The verifier's law, as `OneStage.paramsOfJson` keeps it, is one `OneStage.rule` accepts as it stands**:
`SameLawProfile`'s premise holds of every verifier `OneStage.paramsOfJson` accepts. -/
theorem paramsOfJson_rule {j : Json} {p : OneStage.Params} (h : OneStage.paramsOfJson j = .ok p) :
    OneStage.rule p.verifier.law = .ok p.verifier.law := by
  unfold OneStage.paramsOfJson at h
  try dsimp only at h
  try simp only [throw_bind_eq] at h
  repeat' walk_step
  all_goals
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    exact paramsParsed_rule (by assumption)

/-- **The record's law has one profile** (`SameLawProfile`): the record's law has the canonical bytes of both expanded
laws, so the three parse back to one value, whose profile is each one's. -/
theorem sameLawProfile : SameLawProfile := by
  intro rule d query table n named law recLaw ruleLaw verifierLaw hn hr hd1 hd2 hs1 hs2
  obtain ⟨s, hrs, hrl⟩ := canon_of_jeq hs1
  obtain ⟨s', hrs', hvl⟩ := canon_of_jeq hs2
  rw [hrs] at hrs'
  cases hrs'
  obtain ⟨j1, hp1, hS1⟩ := FlockComplete.JsonRT.ReadBack.pr_parse
    (FlockComplete.JsonRT.ReadBack.canon_pr (good_derive (good_lawOver hn) hd1) s hrl)
  obtain ⟨W, hW, hWp⟩ := verifier_text hr hd2 hvl
  obtain ⟨j2, hp2, hS2⟩ := FlockComplete.JsonRT.ReadBack.pr_parse hW
  rw [hp1] at hp2
  cases hp2
  rw [← hWp, profileDraw_same hS2, ← profileDraw_same hS1]

theorem keys?_of {kvs : Std.TreeMap.Raw String Json compare} {l : List String} (hs : l.Pairwise (· < ·))
    (h : kvs.toArray.map (·.1) = l.toArray) : keys? (.obj kvs) = some l.toArray := by
  show some (sortStrings _) = _
  rw [h]
  exact congrArg some (Array.toList_inj.1 (qsort_of_sorted (as := l.toArray) hs))

theorem rule_subset2 :
    OneStage.rule (Json.mkObj [("law", "subset"), ("k", Lean.toJson (2 : ℕ))]) =
      .ok (Json.mkObj [("law", "subset"), ("k", Lean.toJson (2 : ℕ))]) := by
  have hs : sortStrings #["k", "law"] = #["k", "law"] :=
    Array.toList_inj.1 (qsort_of_sorted (as := #["k", "law"]) (by decide))
  have hk : keys? (Json.mkObj [("law", "subset"), ("k", Lean.toJson (2 : ℕ))]) = some #["k", "law"] :=
    keys?_of (by decide) (by decide +kernel)
  unfold OneStage.rule
  rw [exactly_of (hk.trans (congrArg some hs.symm))]
  rfl

/-- `SameLawProfile`'s premises hold together: the rule `all` over two units names `subset:2`, which `OneStage.rule`
accepts as it stands, `Draw.derive` leaves unchanged and has the canonical bytes of itself. -/
theorem sameLawProfile_honest :
    Select.Rule.all.lawOver 2 = .ok (Json.mkObj [("law", "subset"), ("k", Lean.toJson (2 : ℕ))]) ∧
      OneStage.rule (Json.mkObj [("law", "subset"), ("k", Lean.toJson (2 : ℕ))]) =
        .ok (Json.mkObj [("law", "subset"), ("k", Lean.toJson (2 : ℕ))]) ∧
      Draw.derive default Json.null (Json.mkObj [("law", "subset"), ("k", Lean.toJson (2 : ℕ))]) none =
        .ok (Json.mkObj [("law", "subset"), ("k", Lean.toJson (2 : ℕ))]) ∧
      Draw.sameLaw (Json.mkObj [("law", "subset"), ("k", Lean.toJson (2 : ℕ))])
        (Json.mkObj [("law", "subset"), ("k", Lean.toJson (2 : ℕ))]) = true ∧
      Record.profileDraw (Json.mkObj [("law", "subset"), ("k", Lean.toJson (2 : ℕ))]) = some (.subset 2) :=
  ⟨rfl, rule_subset2, rfl, jeq_self (good_lawToJson (.subset 2)), rfl⟩

/-- `SameLawProfile`'s conclusion is not trivial: two laws the profile tells apart. -/
theorem sameLawProfile_nontrivial :
    ¬ (Record.profileDraw (Json.mkObj [("law", "subset"), ("k", Lean.toJson (1 : ℕ))]) =
      Record.profileDraw (Json.mkObj [("law", "subset"), ("k", Lean.toJson (2 : ℕ))])) := by
  rw [show Record.profileDraw (Json.mkObj [("law", "subset"), ("k", Lean.toJson (1 : ℕ))]) = some (.subset 1) from rfl,
    show Record.profileDraw (Json.mkObj [("law", "subset"), ("k", Lean.toJson (2 : ℕ))]) = some (.subset 2) from rfl]
  simp

/-- **A value with the canonical bytes of `toJson n` reads as `n`** (`nat?`). -/
theorem nat?_of_jeq_toJson {a : Json} {n : ℕ} (h : jeq a (Lean.toJson n) = true) : nat? a = some n := by
  obtain ⟨s, ha, hn⟩ := canon_of_jeq h
  have key : Good a → nat? a = some n := fun hg => by
    obtain ⟨j1, hp1, hs1⟩ := FlockComplete.JsonRT.ReadBack.pr_parse (FlockComplete.JsonRT.ReadBack.canon_pr hg s ha)
    obtain ⟨j2, hp2, hs2⟩ :=
      FlockComplete.JsonRT.ReadBack.pr_parse (FlockComplete.JsonRT.ReadBack.canon_pr (good_toJson n) s hn)
    rw [hp1] at hp2
    cases hp2
    obtain rfl := same_num hs2
    obtain rfl := same_num hs1.symm
    first | rfl | simp [nat?, int?, JsonNumber.fromNat]
  rw [toJson_canon] at hn
  cases hn
  obtain ⟨d, ds, hd, -, h9⟩ := digits_head n
  cases a with
  | null => exact key .null
  | bool b => exact key (.bool b)
  | str t => exact key (.str t)
  | num m =>
    obtain ⟨mm, e⟩ := m
    by_cases he : e = 0
    · subst he
      exact key (.num mm)
    · exfalso
      rw [Flock.canon] at ha
      split at ha
      · rename_i hc
        exact he (by simpa using hc)
      · exact throw_ne _ _ ha
  | arr xs =>
    exfalso
    rw [Flock.canon] at ha
    cases hp : xs.attach.mapM (fun x => Flock.canon x.1) with
    | error e => rw [hp] at ha; cases ha
    | ok parts =>
      rw [hp] at ha
      obtain ⟨mid, hm⟩ : ∃ mid : String, "[" ++ mid ++ "]" = Int.repr (Int.ofNat n) := ⟨_, Except.ok.inj ha⟩
      rw [← hm, lit_head "[" _ _ '[' rfl] at hd
      injection hd with e
      subst e
      exact absurd h9 (by decide)
  | obj kvs =>
    exfalso
    rw [Flock.canon] at ha
    cases hp : kvs.toArray.attach.mapM (fun p => do pure (p.1.1, ← Flock.canon p.1.2)) with
    | error e => rw [hp] at ha; cases ha
    | ok vals =>
      rw [hp] at ha
      obtain ⟨mid, hm⟩ : ∃ mid : String, "{" ++ mid ++ "}" = Int.repr (Int.ofNat n) := ⟨_, Except.ok.inj ha⟩
      rw [← hm, lit_head "{" _ _ '{' rfl] at hd
      injection hd with e
      subst e
      exact absurd h9 (by decide)

end Properties.SampledProofs.SecurityProofs
