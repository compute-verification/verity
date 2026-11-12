module

public import Mathlib.Tactic
public import Proofs.Properties.Nci.RecordBlocks

/-!
# The fresh rows of a slot, stated

`rowBits` is a reader row's bits as the summary counts a block's input bits: its port's 16-bit words over the port's
cut. `rowBits_eq` reads it off a template and a row of it, where ids and row names are distinct.

`FreshRows` states what the summary's fresh bits count: on a record `Record.read` accepts, the rows of one slot whose
kind is neither `inside` nor `theta` are, between them, no more bits than the summary's fresh
(`Record.Summary.fresh`), the greatest over the slots of a block's input bits less its declared `inside` and `theta`
bits.
-/

@[expose] public section

namespace NciCheck.Record.SecurityProofs

open RecordText
open Lean (Json)

/-- **A row's bits**, as the summary counts them: of a row of port `p` of the template `tid` of `ts`, the port's
16-bit words over its cut in the wiring `W`; none where `ts` has no such template or the template no such row. Every
row of one port has the same. -/
def rowBits (ts : List Tmpl) (W : Spot.Wires.Wiring) (tid p : String) : ℕ :=
  (((ts.find? (·.tid == tid)).bind fun t => t.rows.find? (·.1 == p)).map fun row =>
    16 * row.2.1 / W.cutsOf tid p).getD 0

/-- The first of a list that passes a test is the one member that passes it. -/
theorem find?_one {α : Type} (p : α → Bool) : ∀ (l : List α) (a : α), a ∈ l → p a = true →
    (∀ x ∈ l, p x = true → x = a) → l.find? p = some a
  | [], _, ha, _, _ => by cases ha
  | x :: l, a, ha, hp, hu => by
    rw [List.find?_cons]
    cases hx : p x with
    | true => rw [hu x List.mem_cons_self hx]
    | false =>
      rcases List.mem_cons.1 ha with rfl | ha'
      · rw [hx] at hp
        cases hp
      · exact find?_one p l a ha' hp fun y hy => hu y (List.mem_cons_of_mem _ hy)

/-- **A row's bits, of a template and a row of it**: where no two templates have one id and no two rows of the
template one name. -/
theorem rowBits_eq {ts : List Tmpl} (W : Spot.Wires.Wiring) (hd : (ts.map (·.tid)).Nodup) {t : Tmpl} (ht : t ∈ ts)
    (hn : (t.rows.map (·.1)).Nodup) {row : String × ℕ × ℕ} (hrow : row ∈ t.rows) :
    rowBits ts W t.tid row.1 = 16 * row.2.1 / W.cutsOf t.tid row.1 := by
  unfold rowBits
  rw [find?_one (fun x => x.tid == t.tid) ts t ht (beq_self_eq_true _) fun x hx hp =>
      eq_of_nodup_map (·.tid) ts hd x hx t ht (eq_of_beq hp),
    Option.bind_some,
    find?_one (fun x => x.1 == row.1) t.rows row hrow (beq_self_eq_true _) fun x hx hp =>
      eq_of_nodup_map (·.1) t.rows hn x hx row hrow (eq_of_beq hp)]
  rfl

/-- **The fresh rows of a slot are no more bits than the summary's fresh.** On a record `read` accepts, for a slot
`s` and any finite set of reader rows `(template, unit, port, row)`, each a row of a template of the record (its
port a row of the template, its row within the port's cut) whose unit is in slot `s` and whose kind is neither
`inside` nor `theta`: their bits (`rowBits`), summed, are at most the summary's fresh bits. -/
def FreshRows : Prop :=
  ∀ (j tr : Json) (c : Case), read j tr = .ok c → ∀ s : ℕ, s < c.S → ∀ F : Finset (String × ℕ × String × ℕ),
    (∀ x ∈ F, (∃ t ∈ c.f.ts, t.tid = x.1 ∧ ∃ row ∈ t.rows, row.1 = x.2.2.1) ∧
      x.2.2.2 < c.W.cutsOf x.1 x.2.2.1 ∧ c.slot x.1 x.2.1 = some s ∧
      c.rowKind ⟨x.1, x.2.1, x.2.2.1, x.2.2.2⟩ ≠ Kind.inside ∧
      c.rowKind ⟨x.1, x.2.1, x.2.2.1, x.2.2.2⟩ ≠ Kind.theta) →
    ∑ x ∈ F, rowBits c.f.ts c.W x.1 x.2.2.1 ≤ (summarize c).fresh

end NciCheck.Record.SecurityProofs
