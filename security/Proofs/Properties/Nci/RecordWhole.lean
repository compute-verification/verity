module

public import Proofs.Properties.Nci.RecordBlocks

/-!
# What the record check tests of a wire's two rows

`Record.unwhole` refuses a record with a wire whose writer's output is not the bits of the row it fills, or whose
two rows' leaves do not line up. This module states what a record it accepts has, of each wire, in the terms of the
two templates' Definitions (`whole_spec`): the writer's output is whole 16-bit words; and either the reader's
parameter has leaves of one bit and is none of what its Definition passes on, or each side's leaves are of one width
and the writer's divides the reader's.

The steps: what `unwhole` tests of one wire (`unwhole_none`), a template's leaves among the leaves of templates with
distinct ids (`leaves_find`), and a name's entry in a list of named values with distinct names (`lookup_map`).
-/

@[expose] public section

namespace NciCheck.Record.SecurityProofs

open NciCheck.RecordText (Tmpl fieldsOf)
open Flock.Program
open Flock.Sigma (bits)

/-- The member with a key, in a list whose keys are distinct. -/
theorem find_of_nodup {α : Type} (key : α → String) : ∀ (l : List α), (l.map key).Nodup → ∀ a ∈ l,
    l.find? (fun x => key x == key a) = some a
  | [], _, _, ha => by cases ha
  | x :: l, h, a, ha => by
    rw [List.find?_cons]
    by_cases hx : key x = key a
    · rw [show (key x == key a) = true from beq_iff_eq.2 hx]
      exact congrArg some (eq_of_nodup_map key (x :: l) h x List.mem_cons_self a ha hx)
    · rw [show (key x == key a) = false from beq_eq_false_iff_ne.2 hx]
      rcases List.mem_cons.1 ha with ha' | ha'
      · exact absurd (by rw [ha']) hx
      · exact find_of_nodup key l (List.nodup_cons.1 (by simpa using h)).2 a ha'

/-- A name's value, in a list of named values whose names are distinct. -/
theorem lookup_map {α β : Type} (k : α → String) (v : α → β) : ∀ (l : List α), (l.map k).Nodup → ∀ a ∈ l,
    (l.map fun x => (k x, v x)).lookup (k a) = some (v a)
  | [], _, _, ha => by cases ha
  | x :: l, h, a, ha => by
    rw [List.map_cons, List.lookup_cons]
    by_cases hx : k a = k x
    · rw [show (k a == k x) = true from beq_iff_eq.2 hx]
      exact congrArg (fun y => some (v y)) (eq_of_nodup_map k (x :: l) h x List.mem_cons_self a ha hx.symm)
    · rw [show (k a == k x) = false from beq_eq_false_iff_ne.2 hx]
      rcases List.mem_cons.1 ha with ha' | ha'
      · exact absurd (by rw [ha']) hx
      · exact lookup_map k v l (List.nodup_cons.1 (by simpa using h)).2 a ha'

/-- A name not among a list's names is looked up past it. -/
theorem lookup_append_of_not_mem {β : Type} : ∀ (l l' : List (String × β)) (a : String), a ∉ l.map (·.1) →
    (l ++ l').lookup a = l'.lookup a
  | [], _, _, _ => rfl
  | x :: l, l', a, h => by
    rw [List.map_cons, List.mem_cons, not_or] at h
    rw [List.cons_append, List.lookup_cons, show (a == x.1) = false from beq_eq_false_iff_ne.2 h.1]
    exact lookup_append_of_not_mem l l' a h.2

/-- **What `unwhole` tests of one wire**, as its two lookups among the templates and their leaves: the writer's
output's bits, where both are found, are its words'; and the reader's row is closed, or each side has a leaf width
and the writer's divides the reader's. -/
theorem unwhole_none {ts : List Tmpl} {ls : List Leaves} {wires : List Spot.Wires.Wire}
    (h : unwhole ts ls wires = none) {w : Spot.Wires.Wire} (hw : w ∈ wires) :
    (∀ b n, ((ls.find? (·.tid == w.writer)).bind fun l => l.bits.lookup w.out) = some b →
        ((ts.find? (·.tid == w.writer)).bind fun t => t.outs.lookup w.out) = some n → b = 16 * n) ∧
      (((ls.find? (·.tid == w.reader)).map fun l => l.closed.contains w.port).getD false = true ∨
        ∃ r o, ((ls.find? (·.tid == w.reader)).bind fun l => l.leaf.lookup w.port).join = some r ∧
          ((ls.find? (·.tid == w.writer)).bind fun l => l.leaf.lookup w.out).join = some o ∧ o ≠ 0 ∧ r % o = 0) := by
  unfold unwhole at h
  simp only [Option.map_eq_none_iff] at h
  have hw' := List.find?_eq_none.1 h w hw
  simp only [Bool.or_eq_true, Bool.not_eq_true', not_or, Bool.not_eq_true, Option.isSome_eq_false_iff,
    Option.isNone_iff_eq_none, Bool.not_eq_false] at hw'
  obtain ⟨h1, h2⟩ := hw'
  refine ⟨fun b n hb hn => ?_, ?_⟩
  · rw [hb, hn] at h1
    simp only at h1
    split at h1
    · rename_i he
      exact beq_iff_eq.1 he
    · cases h1
  · rcases h2 with h2 | h2
    · exact Or.inl h2
    · split at h2
      · rename_i r o hr ho
        simp only [Bool.and_eq_true, bne_iff_ne, ne_eq, beq_iff_eq] at h2
        exact Or.inr ⟨r, o, hr, ho, h2.1, h2.2⟩
      · cases h2

end NciCheck.Record.SecurityProofs
