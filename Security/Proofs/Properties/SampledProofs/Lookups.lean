module

public import Proofs.Flock.Soundness.ExecCircuit
public import Proofs.Flock.Soundness.Discharge.Hidden.Facts
public import SampledProofs.OneStageZk
public import Mathlib.Data.List.Nodup
public import Mathlib.Tactic.DefEqTransformations

/-!
# The tables v2 hands `flock-verify` are the circuit's lookups'

v2 hands a session only those pinned MUFU tables the blobs hold that the circuit file's META `lookups` names
(`SampledProofs.OneStageZk.tablesFor`, `lookupNames`). Where the circuit file parses with them, they are its lookups'
tables:

* **The parse finds each lookup's table** (`lookups_given`, `lookups_given_hmOut`, `lookups_given_desc`): a circuit
  file `HmRow.parse` accepts with the tables `T` (with or without a typed template, with any nets first), and so one
  `HmOut.parse` or `Desc.parse` accepts, names in its lookups only pinned tables, and `T` holds one under each name. The
  walk is `parse_factsPre_nets`'s, keeping the file's text, its META line and its `lookups` (`lookupNames_eq`).
* **The session's tables are the lookups'** (`tablesFor_names`): where the circuit file parses with the tables v2
  hands, their names are the circuit's lookups.
* **Two sessions read the same tables** (`tablesFor_ofArray_eq`): where the circuit file parses with the tables v2 finds
  on the files `F` and with those it finds on the files `R`, and the first file of `F` and of `R` holding a pinned table
  are one file wherever both hold one, v2 hands the same tables at `F` as at `R`.
-/

@[expose] public section

open Flock
open Lean (Json)
open SampledProofs.OneStageZk (lookupNames lookupsOf metaLineOf tablesFor tablesOf)

namespace Verity.Sampled

/-! ## Loops and the lookups' names -/

/-- A `for` loop over a list in `Except` whose steps yield wherever they succeed: where it succeeds, each element's step
succeeded at some state, so it has what every successful step has. -/
theorem forIn_steps_all {α β : Type} (Q : α → Prop) {f : α → β → Except String (ForInStep β)}
    (hf : ∀ a b s, f a b = .ok s → Q a ∧ ∃ b', s = .yield b') :
    ∀ (l : List α) (b v : β), forIn l b f = .ok v → ∀ a ∈ l, Q a
  | [], _, _, _, a, ha => by cases ha
  | x :: l, b, v, h, a, ha => by
    rw [List.forIn_cons] at h
    obtain ⟨s, hs, h⟩ := FlockSoundness.except_bind_ok h
    obtain ⟨hq, b', rfl⟩ := hf x b s hs
    rcases List.mem_cons.1 ha with rfl | ha
    · exact hq
    · exact forIn_steps_all Q hf l b' v h a ha

/-- **The lookups' names off a file the parse reads**: where the file is the UTF-8 text `text`, whose META line `ml`
(`metaLineOf`) starts `META ` and holds the JSON `mj`, the names `lookupNames` reads are `mj`'s `lookups` keys. -/
theorem lookupNames_eq {bytes : ByteArray} {text ml : String} {mj : Json}
    (hs : String.fromUTF8? bytes = some text) (hm : ¬ ((!ml.startsWith "META ") = true))
    (hj : JsonParse.parse (ml.drop 5).toString = .ok mj) (hml : ml = metaLineOf text) :
    lookupNames bytes = lookupsOf mj := by
  subst hml
  unfold lookupNames
  rw [hs]
  dsimp only
  rw [ite_eq_right hm, hj]

set_option maxHeartbeats 4000000 in
/-- **The parse finds each lookup's table**: every name in the lookups of a circuit file `HmRow.parse` accepts with the
tables `tables` (`lookupNames`) is a pinned table's, and `tables` holds one under it. -/
theorem lookups_given {tags : Tags} {bytes : ByteArray} {tables : List (String × ByteArray)}
    {tmpl : Option Typed.Template} {pre : Array (String × Flock.Net)} {c : Flock.Circuit}
    (h : HmRow.parse tags bytes tables tmpl pre = .ok c) :
    ∀ l ∈ lookupNames bytes, (∃ t ∈ MUFU_TABLES, t.name = l) ∧ ∃ w, (l, w) ∈ tables := by
  unfold HmRow.parse at h
  cases tmpl
  case' none =>
    simp (config := { zeta := false }) only [Option.isSome_none, Bool.false_eq_true, ↓reduceIte, Option.mapM_none] at h
  case' some t =>
    simp (config := { zeta := false }) only [Option.isSome_some, ↓reduceIte, Option.mapM_some] at h
  all_goals
    (extract_lets (config := { underBinder := false, onlyGivenNames := true }) sha jp0 at h)
    split at h
    swap
    · obtain ⟨_, hte, -⟩ := FlockSoundness.except_bind_ok h; cases hte
    rename_i text0 hs
    (have hh := FlockSoundness.except_bind_ok h; clear h; obtain ⟨text, hte, h⟩ := hh)
    cases hte
    simp (config := { zeta := false }) only [jp0] at h
    (extract_lets (config := { underBinder := false, onlyGivenNames := true }) lines0 lines head jp1 at h)
    replace h := FlockSoundness.ite_throw_ok h
    simp (config := { zeta := false }) only [jp1] at h
    (extract_lets (config := { underBinder := false, onlyGivenNames := true }) name metaLine jp2 at h)
    obtain ⟨hmeta, h⟩ := FlockSoundness.ite_throw_ok' h
    simp (config := { zeta := false }) only [jp2] at h
    (extract_lets (config := { underBinder := false, onlyGivenNames := true }) jp3 at h)
    split at h
    swap
    · obtain ⟨_, hte, -⟩ := FlockSoundness.except_bind_ok h; cases hte
    rename_i jv hjv
    (have hh := FlockSoundness.except_bind_ok h; clear h; obtain ⟨mj, hte, h⟩ := hh)
    cases hte
    have hnames := lookupNames_eq hs hmeta hjv rfl
    simp (config := { zeta := false }) only [jp3] at h
    (have hh := FlockSoundness.except_bind_ok h; clear h; obtain ⟨f1, -, h⟩ := hh)
    (have hh := FlockSoundness.except_bind_ok h; clear h; obtain ⟨f2, -, h⟩ := hh)
    (extract_lets (config := { underBinder := false, onlyGivenNames := true }) jp4 at h)
    replace h := FlockSoundness.ite_throw_ok h
    simp (config := { zeta := false }) only [jp4] at h
    (extract_lets (config := { underBinder := false, onlyGivenNames := true }) jp5 jp6 at h)
    obtain ⟨classSha, h⟩ : ∃ classSha, jp5 classSha = .ok c := by
      split at h
      · replace h := FlockSoundness.ite_throw_ok h
        simp (config := { zeta := false }) only [jp6] at h
        (extract_lets (config := { underBinder := false, onlyGivenNames := true }) part rule jp7 at h)
        replace h := FlockSoundness.ite_throw_ok h
        simp (config := { zeta := false }) only [jp7] at h
        obtain ⟨cs, -, h⟩ := FlockSoundness.except_bind_ok h
        exact ⟨cs, h⟩
      · obtain ⟨cs, -, h⟩ := FlockSoundness.except_bind_ok h
        exact ⟨cs, h⟩
    simp (config := { zeta := false }) only [jp5] at h
    (extract_lets (config := { underBinder := false, onlyGivenNames := true }) nets0 at0 at h)
    (have hh := FlockSoundness.except_bind_ok h; clear h; obtain ⟨s1, -, h⟩ := hh)
    (extract_lets (config := { underBinder := false, onlyGivenNames := true }) nets1 at h)
    (have hh := FlockSoundness.except_bind_ok h; clear h; obtain ⟨u1, -, h⟩ := hh)
    (extract_lets (config := { underBinder := false, onlyGivenNames := true }) jpT at h)
    -- a template's derived keys are checked before the join point (`TEMPLATE_DERIVED`)
    obtain ⟨uT, h⟩ : ∃ uT, jpT uT = .ok c := by
      first
      | exact ⟨⟨⟩, h⟩
      | (obtain ⟨u, -, h⟩ := FlockSoundness.except_bind_ok h; exact ⟨u, h⟩)
    simp (config := { zeta := false }) only [jpT] at h
    (extract_lets (config := { underBinder := false, onlyGivenNames := true }) jpL at h)
    obtain ⟨lks, hlks, h⟩ : ∃ lks : Array (String × Json), lookupNames bytes = lks.map (·.1) ∧ jpL lks = .ok c := by
      split at h
      · rename_i kvs hk
        obtain ⟨l, hl, h⟩ := FlockSoundness.except_bind_ok h
        cases hl
        exact ⟨_, hnames.trans (by simp only [lookupsOf, hk]), h⟩
      · rename_i hk
        obtain ⟨l, hl, h⟩ := FlockSoundness.except_bind_ok h
        cases hl
        exact ⟨_, hnames.trans (by simp only [lookupsOf, hk, Array.map_empty]), h⟩
      · obtain ⟨l, hl, -⟩ := FlockSoundness.except_bind_ok h
        cases hl
      · rename_i e hk
        obtain ⟨l, hl, h⟩ := FlockSoundness.except_bind_ok h
        cases hl
        exact ⟨_, hnames.trans (by simp only [lookupsOf, hk, Array.map_empty]), h⟩
    simp (config := { zeta := false }) only [jpL] at h
    (have hh := FlockSoundness.except_bind_ok h; clear h; obtain ⟨s2, hlook, -⟩ := hh)
    rw [← Array.forIn_toList] at hlook
    have hall : ∀ a ∈ lks.toList, (∃ t ∈ MUFU_TABLES, t.name = a.1) ∧ ∃ w, (a.1, w) ∈ tables := by
      refine forIn_steps_all _ ?_ lks.toList _ _ hlook
      intro a b s hstep
      obtain ⟨lname, spec⟩ := a
      (extract_lets (config := { underBinder := false, onlyGivenNames := true }) nets at hstep)
      split at hstep
      · rename_i pinned hp
        (extract_lets (config := { underBinder := false, onlyGivenNames := true }) size at hstep)
        (have hh := FlockSoundness.except_bind_ok hstep; clear hstep; obtain ⟨_, -, hstep⟩ := hh)
        (have hh := FlockSoundness.except_bind_ok hstep; clear hstep; obtain ⟨_, -, hstep⟩ := hh)
        (have hh := FlockSoundness.except_bind_ok hstep; clear hstep; obtain ⟨_, -, hstep⟩ := hh)
        (extract_lets (config := { underBinder := false, onlyGivenNames := true }) jpA at hstep)
        replace hstep := FlockSoundness.ite_throw_ok hstep
        simp (config := { zeta := false }) only [jpA] at hstep
        (have hh := FlockSoundness.except_bind_ok hstep; clear hstep; obtain ⟨n, -, hstep⟩ := hh)
        (have hh := FlockSoundness.except_bind_ok hstep; clear hstep; obtain ⟨lo, -, hstep⟩ := hh)
        (extract_lets (config := { underBinder := false, onlyGivenNames := true }) jpB at hstep)
        replace hstep := FlockSoundness.ite_throw_ok hstep
        simp (config := { zeta := false }) only [jpB] at hstep
        split at hstep
        · rename_i nm words hw
          (extract_lets (config := { underBinder := false, onlyGivenNames := true }) jpC at hstep)
          replace hstep := FlockSoundness.ite_throw_ok hstep
          simp (config := { zeta := false }) only [jpC] at hstep
          (have hh := FlockSoundness.except_bind_ok hstep; clear hstep; obtain ⟨net, -, hstep⟩ := hh)
          (extract_lets (config := { underBinder := false, onlyGivenNames := true }) nets' at hstep)
          have hs' := Except.ok.inj hstep
          subst hs'
          refine ⟨⟨⟨pinned, List.mem_of_find?_eq_some hp, ?_⟩, words, ?_⟩, _, rfl⟩
          · simpa using List.find?_some hp
          · have hnm : nm = lname := by simpa using List.find?_some hw
            subst hnm
            exact List.mem_of_find?_eq_some hw
        · (have hh := FlockSoundness.except_bind_ok hstep; clear hstep; obtain ⟨w, hw, -⟩ := hh)
          cases hw
      · (have hh := FlockSoundness.except_bind_ok hstep; clear hstep; obtain ⟨w, hw, -⟩ := hh)
        cases hw
    intro l hl
    rw [hlks] at hl
    obtain ⟨a, ha, rfl⟩ := Array.mem_map.1 hl
    exact hall a (Array.mem_toList_iff.2 ha)

/-- **`HmOut.parse` finds each lookup's table**: it is `HmRow.parse` on the same file and tables (`parse_cases`). -/
theorem lookups_given_hmOut {tags : Tags} {bytes : ByteArray} {tables : List (String × ByteArray)} {c : Flock.Circuit}
    (h : HmOut.parse tags bytes tables = .ok c) :
    ∀ l ∈ lookupNames bytes, (∃ t ∈ MUFU_TABLES, t.name = l) ∧ ∃ w, (l, w) ∈ tables := by
  obtain ⟨-, ⟨-, hc⟩ | ⟨_, _, _, _, hc, -⟩ | ⟨_, _, _, hc, -⟩⟩ := FlockSoundness.Discharge.Hidden.parse_cases h
  all_goals exact lookups_given hc

/-- **`Desc.parse` finds each lookup's table**, as the session's descriptor reads the circuit file (`Desc.parse_flat`). -/
theorem lookups_given_desc {tags : Tags} {bytes : ByteArray} {tables : List (String × ByteArray)} {D : Flock.Desc}
    (h : Flock.Desc.parse tags bytes tables = .ok D) :
    ∀ l ∈ lookupNames bytes, (∃ t ∈ MUFU_TABLES, t.name = l) ∧ ∃ w, (l, w) ∈ tables :=
  lookups_given_hmOut (Flock.Desc.parse_flat h)

/-! ## The tables v2 hands -/

theorem mufu_names_nodup : (MUFU_TABLES.map (·.name)).Nodup := by decide

/-- A table v2 hands is under a name the circuit's lookups name. -/
theorem mem_tablesFor {c : ByteArray} {bs : _root_.Core.Blobs} {e : String × ByteArray} (h : e ∈ tablesFor c bs) :
    e.1 ∈ lookupNames c := by
  simp only [tablesFor, List.mem_filter] at h
  exact Array.contains_iff_mem.1 h.2

/-- **The session's tables are the lookups'**: where the circuit file parses with the tables v2 hands, their names are
the circuit's lookups. -/
theorem tablesFor_names {tags : Tags} {c : ByteArray} {bs : _root_.Core.Blobs} {D : Flock.Desc}
    (h : Flock.Desc.parse tags c (tablesFor c bs) = .ok D) (l : String) :
    l ∈ (tablesFor c bs).map (·.1) ↔ l ∈ lookupNames c := by
  constructor
  · intro hl
    obtain ⟨e, he, rfl⟩ := List.mem_map.1 hl
    exact mem_tablesFor he
  · intro hl
    obtain ⟨-, w, hw⟩ := lookups_given_desc h l hl
    exact List.mem_map.2 ⟨(l, w), hw, rfl⟩

/-- **The tables v2 hands on the files `F`**: each pinned table the circuit's lookups name, at the first of `F` whose
SHA-512's hex is its pinned digest. -/
theorem tablesFor_ofArray (c : ByteArray) (F : Array ByteArray) :
    tablesFor c (_root_.Core.Blobs.ofArray F) = MUFU_TABLES.filterMap fun t =>
      ((F.find? fun x => hexOf (Sha512.hash x) == t.sha512).map fun b => (t.name, b)).filter
        fun e => (lookupNames c).contains e.1 := by
  simp only [tablesFor, tablesOf, _root_.Core.Blobs.ofArray, Array.find?_map, Option.map_map, List.filter_filterMap]
  all_goals rfl

/-- Where the circuit file parses with the tables v2 hands on the files `X`, `X` holds each pinned table its lookups
name. -/
theorem held_of_parse {tags : Tags} {c : ByteArray} {X : Array ByteArray} {D : Flock.Desc}
    (h : Flock.Desc.parse tags c (tablesFor c (_root_.Core.Blobs.ofArray X)) = .ok D) {t : MufuTable}
    (ht : t ∈ MUFU_TABLES) (hc : (lookupNames c).contains t.name = true) :
    ∃ w, X.find? (fun x => hexOf (Sha512.hash x) == t.sha512) = some w := by
  obtain ⟨-, w, hw⟩ := lookups_given_desc h t.name (Array.contains_iff_mem.1 hc)
  rw [tablesFor_ofArray] at hw
  obtain ⟨t', ht', he⟩ := List.mem_filterMap.1 hw
  have he' := (Option.filter_eq_some_iff.1 he).1
  obtain ⟨a, ho, hpa⟩ := Option.map_eq_some_iff.1 he'
  have hn : t'.name = t.name := congrArg Prod.fst hpa
  obtain rfl : t' = t := List.inj_on_of_nodup_map mufu_names_nodup ht' ht hn
  exact ⟨_, ho⟩

/-- **Two sessions read the same tables**: where the circuit file `c` parses with the tables v2 hands on the files `F`
and with those it hands on the files `R`, and the first file of `F` and the first of `R` holding a pinned table are one
file wherever both hold one, v2 hands the same tables at `F` as at `R`. -/
theorem tablesFor_ofArray_eq {tags tags' : Tags} {c : ByteArray} {F R : Array ByteArray} {D D' : Flock.Desc}
    (hF : Flock.Desc.parse tags c (tablesFor c (_root_.Core.Blobs.ofArray F)) = .ok D)
    (hR : Flock.Desc.parse tags' c (tablesFor c (_root_.Core.Blobs.ofArray R)) = .ok D')
    (agree : ∀ t ∈ MUFU_TABLES, ∀ a b, F.find? (fun x => hexOf (Sha512.hash x) == t.sha512) = some a →
      R.find? (fun x => hexOf (Sha512.hash x) == t.sha512) = some b → a = b) :
    tablesFor c (_root_.Core.Blobs.ofArray F) = tablesFor c (_root_.Core.Blobs.ofArray R) := by
  rw [tablesFor_ofArray, tablesFor_ofArray]
  apply List.filterMap_congr
  intro t ht
  try beta_reduce
  cases hc : (lookupNames c).contains t.name with
  | true =>
    obtain ⟨a, ha⟩ := held_of_parse hF ht hc
    obtain ⟨b, hb⟩ := held_of_parse hR ht hc
    rw [ha, hb, agree t ht a b ha hb]
  | false =>
    have hn : ∀ X : Array ByteArray, ((X.find? fun x => hexOf (Sha512.hash x) == t.sha512).map fun b => (t.name, b)).filter
        (fun e => (lookupNames c).contains e.1) = none := by
      intro X
      cases X.find? fun x => hexOf (Sha512.hash x) == t.sha512 with
      | none => rfl
      | some b => exact Option.filter_some_neg hc
    rw [hn F, hn R]

end Verity.Sampled
