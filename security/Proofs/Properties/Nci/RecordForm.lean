module

public import Properties.Nci.Record
public import Proofs.Properties.Nci.RecordRead
public import Proofs.Properties.Nci.Lemmas

/-!
# The record check's own form, and what it carries and rates

`form` is the record check (`Record`) as a `Properties.Record.Form`: the functions the record check exports, each the
check's own. Its instance is the one `read` parsed, its wiring, slots and held rows the ones `read` gives, its
summary the two counts of `summarize` that the rate rule reads, its rate rule `failure` on that one pair, and its
layout `layoutOf`'s. `form_rate`: on a record's summary the form's rate rule is `rate`, the one `evaluate` decides by.

Two of the four statements `Properties.Record.Complies` follows from, of this form:

* `form_owns` (`Properties.Record.Owns`): a record `read` accepts carries its instance's own fields. `read` keeps the
  instance `parse` gave (`front_parse`, `finish_text`), and `parse` gives a field only as the JSON holds it
  (`parse_fields`): the two texts as strings, `gamma` a natural, `alpha` a pair of naturals with a positive second,
  and `pins` a list of strings, each the pin it keeps (`pins_array`).
* `form_rates` (`Properties.Record.Rates`): where the rate rule gives no failure on a summary, its gates are at most
  `Γ` and its fresh bits times `den` at most `num` times its gates. `failure` on one pair is its one test.
-/

@[expose] public section

namespace NciCheck.Record.SecurityProofs

open RecordText
open Lean (Json)

/-- **The record check's form**: what `Record` exports, as the statements about a record check read it. -/
def form : Properties.Record.Form where
  Case := Record.Case
  read := Record.read
  inst c := ⟨c.f.i.program, c.f.i.record, c.f.i.gamma, c.f.i.num, c.f.i.den, c.f.i.pins.toList⟩
  wiring c := c.W
  slot := Record.Case.slot
  held := Record.Case.heldAt
  summarize c := ⟨(Record.summarize c).gates, (Record.summarize c).fresh⟩
  rate Γ num den s := (failure Γ num den #[(s.gates, s.fresh)]).map (·.2)
  layout d := (Record.layoutOf d).map fun l => ⟨l.key, l.root, l.domain, l.ports, l.verdict⟩

/-- On a record's summary, the form's rate rule is the check's. -/
theorem form_rate (Γ num den : Nat) (c : Record.Case) :
    form.rate Γ num den (form.summarize c) = Record.rate Γ num den (Record.summarize c) := rfl

/-- A field `field` gives is the JSON's. -/
theorem field_ok {j : Json} {k : String} {v : Json} (h : field j k = .ok v) : j.getObjVal? k = .ok v := by
  unfold field at h
  simp only [pure, Except.pure, throw, throwThe, MonadExceptOf.throw] at h
  split at h
  · cases h; assumption
  · cases h

/-- A list read entry by entry, each entry a string kept as it is, is the list of the strings read. -/
private theorem pins_list (g : Json → Except String String) (hg : ∀ x s, g x = .ok s → x = .str s) :
    ∀ (l : List Json) (ps : List String), l.mapM g = .ok ps → l = ps.map Json.str
  | [], ps, h => by
    simp only [List.mapM_nil, pure, Except.pure, Except.ok.injEq] at h
    subst h
    rfl
  | a :: l, ps, h => by
    rw [List.mapM_cons] at h
    rcases hga : g a with e | c <;> simp only [hga, bind, Except.bind] at h
    · exact absurd h (by intro h; injection h)
    rcases hl : l.mapM g with e | cs <;> simp only [hl, pure, Except.pure] at h
    · exact absurd h (by intro h; injection h)
    injection h with h
    subst h
    simp only [hg a c hga, pins_list g hg l cs hl, List.map_cons]

/-- **The pins read are the pins listed**: an array read entry by entry, each entry a string kept as it is, is the
array of the strings read. -/
theorem pins_array (g : Json → Except String String) (hg : ∀ x s, g x = .ok s → x = .str s) (xs : Array Json)
    (ps : Array String) (h : xs.mapM g = .ok ps) : xs = (ps.toList.map Json.str).toArray := by
  rw [Array.mapM_eq_mapM_toList] at h
  rcases hl : xs.toList.mapM g with e | l <;> simp only [hl, Functor.map, Except.map] at h
  · exact absurd h (by intro h; injection h)
  injection h with h
  subst h
  have hxs := pins_list g hg _ l hl
  apply Array.ext'
  simpa using hxs

/-- **What `parse` accepts**: each field of the instance it gives is the JSON's own. -/
theorem parse_fields {j : Json} {i : Instance} (h : parse j = .ok i) :
    j.getObjVal? "program" = .ok (.str i.program) ∧ j.getObjVal? "record" = .ok (.str i.record) ∧
      (∃ v, j.getObjVal? "gamma" = .ok v ∧ natOf? v = some i.gamma) ∧
      (∃ a b, j.getObjVal? "alpha" = .ok (.arr #[a, b]) ∧ natOf? a = some i.num ∧ natOf? b = some i.den) ∧
      0 < i.den ∧ j.getObjVal? "pins" = .ok (.arr (i.pins.toList.map Json.str).toArray) := by
  have alpha : ∀ w a b, field j "alpha" = .ok w → pairOf? w = some (a, b) →
      j.getObjVal? "alpha" = .ok (.arr #[a, b]) := fun w a b h1 h2 => by
    rw [← (Properties.SecurityProofs.pairOf?_eq w a b).mp h2]
    exact field_ok h1
  have pins : ∀ (xs : Array Json) (ps : Array String) (g : Json → Except String String),
      (∀ x s, g x = .ok s → x = .str s) → field j "pins" = .ok (.arr xs) → xs.mapM g = .ok ps →
      j.getObjVal? "pins" = .ok (.arr (ps.toList.map Json.str).toArray) := fun xs ps g hg h1 h2 => by
    rw [← pins_array g hg xs ps h2]
    exact field_ok h1
  unfold parse at h
  simp only [bind, Except.bind, pure, Except.pure, throw, throwThe, MonadExceptOf.throw] at h
  repeat (split at h <;> try cases h)
  refine ⟨field_ok (by assumption), field_ok (by assumption), ⟨_, field_ok (by assumption), by assumption⟩,
    ⟨_, _, alpha _ _ _ (by assumption) (by assumption), by assumption, by assumption⟩,
    Nat.pos_of_ne_zero (by assumption), pins _ _ _ ?_ (by assumption) (by assumption)⟩
  intro x s hx
  split at hx
  · split at hx
    · cases hx; rfl
    · cases hx
  · cases hx

/-- `front` keeps the instance `parse` gave. -/
theorem front_parse {j tr : Json} {f : Front} (h : front j tr = .ok f) : parse j = .ok f.i := by
  unfold front at h
  simp only [bind, Except.bind, pure, Except.pure, throw, throwThe, MonadExceptOf.throw] at h
  repeat (split at h <;> try cases h)
  assumption

/-- **A record the check reads carries its instance's fields.** -/
theorem form_owns : Properties.Record.Owns form := by
  intro j tr c h
  have h' : read j tr = .ok c := h
  obtain ⟨f, pin, hf, hc⟩ := read_front h'
  obtain ⟨rfl, -⟩ := finish_text hc
  exact parse_fields (front_parse hf)

/-- **The rate rule on a summary**: where it gives no failure, the summary's gates are at most `Γ` and its fresh bits
times `den` at most `num` times its gates. -/
theorem form_rates : Properties.Record.Rates form := by
  intro Γ num den s h
  have h1 : failure Γ num den #[(s.gates, s.fresh)] = none := by
    cases hf : failure Γ num den #[(s.gates, s.fresh)] with
    | none => rfl
    | some x =>
      have h' : (failure Γ num den #[(s.gates, s.fresh)]).map (·.2) = none := h
      rw [hf] at h'
      simp at h'
  unfold failure at h1
  have h0 := List.findSome?_eq_none_iff.mp h1 0 (List.mem_range.mpr (by simp))
  have e : (#[(s.gates, s.fresh)] : Array (Nat × Nat))[0]! = (s.gates, s.fresh) := rfl
  have h2 : (if s.gates > Γ then some (0, "size") else if s.fresh * den > num * s.gates then some (0, "nci")
      else none) = (none : Option (Nat × String)) := by
    simpa only [e] using h0
  by_cases hg : s.gates > Γ
  · simp [hg] at h2
  · by_cases hn : s.fresh * den > num * s.gates
    · simp [hg, hn] at h2
    · exact ⟨Nat.not_lt.mp hg, Nat.not_lt.mp hn⟩

end NciCheck.Record.SecurityProofs
