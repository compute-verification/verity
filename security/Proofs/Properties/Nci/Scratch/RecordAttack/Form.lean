module

public import Proofs.Properties.Nci.Record
public import Proofs.Properties.Nci.RecordRead
public import NciCheck.Record

/-!
# The record check's own functions, as the interface takes them

A scratch file, never merged. `recordForm` is `NciCheck.Record`'s functions put in the fields of
`NciCheck.Properties.Record.Form`, so that a statement about a `Form` can be read at the check itself.

* `text_recordForm`: on every record the check reads, the form's Program, text, wiring, slots and held rows are the
  ones the first seven clauses of `Served` name.
* `layoutReads_recordForm`: the check's layout reads a Definition's parameters and return type alone.
* `rates_recordForm`: the check's rate rule gives the two bounds on a summary, at every `Γ`, `num` and `den`.
-/

@[expose] public section

namespace NciCheck.Properties.Record.Attack

open Lean (Json)
open Flock.Program (Prog Defn)

/-- The record check's functions as a `Form`. -/
def recordForm : Form where
  Case := NciCheck.Record.Case
  read := NciCheck.Record.read
  inst := fun c =>
    { program := c.f.i.program, record := c.f.i.record, gamma := c.f.i.gamma, num := c.f.i.num, den := c.f.i.den,
      pins := c.f.i.pins.toList }
  wiring := fun c => c.W
  slot := fun c => c.slot
  held := fun c => c.heldAt
  summarize := fun c =>
    { gates := (NciCheck.Record.summarize c).gates, fresh := (NciCheck.Record.summarize c).fresh }
  rate := fun Γ num den s => (NciCheck.failure Γ num den #[(s.gates, s.fresh)]).map (·.2)
  layout := fun d => (NciCheck.Record.layoutOf d).map fun l =>
    { key := l.key, root := l.root, domain := l.domain, opened := l.ports, verdict := l.verdict }

/-- The form's rate rule is the check's on the summary's one pair. -/
theorem rate_recordForm (Γ num den : ℕ) (c : NciCheck.Record.Case) :
    recordForm.rate Γ num den (recordForm.summarize c) =
      NciCheck.Record.rate Γ num den (NciCheck.Record.summarize c) := rfl

/-- **The check reads the record as the text states it**: on every record the check reads, the first seven clauses
of `Served` at `recordForm`, with the check's own Program and text. -/
theorem text_recordForm {j tr : Json} {c : NciCheck.Record.Case} (h : NciCheck.Record.read j tr = .ok c) :
    ∃ W : Spot.Wires.Wiring, Flock.Program.decode (recordForm.inst c).program.toUTF8 = .ok c.f.P ∧
      Flock.JsonParse.parse (recordForm.inst c).record = .ok c.f.doc ∧
      NciCheck.RecordText.wiring c.f.doc = .ok W ∧ (recordForm.wiring c).cuts = W.cuts ∧
      (recordForm.wiring c).wires = W.wires ∧ recordForm.slot c = NciCheck.RecordText.slot c.f.P c.f.doc ∧
      ∀ r hd, recordForm.held c r = some hd →
        NciCheck.RecordText.declared c.f.P c.f.doc r = some hd.declared := by
  obtain ⟨hP, hdoc⟩ := NciCheck.Record.SecurityProofs.inst_of_read h
  obtain ⟨W, hW, hcuts, hwires⟩ := NciCheck.Record.SecurityProofs.wiring_of_read h
  have hdec : Flock.Program.decode c.f.i.program.toUTF8 = .ok c.f.P := by
    unfold NciCheck.decodeProgram at hP
    split at hP
    · rename_i P hok
      have hPP := Except.ok.inj hP
      subst hPP
      exact hok
    · cases hP
  exact ⟨W, hdec, hdoc, hW, hcuts, hwires, NciCheck.Record.SecurityProofs.slot_of_read h,
    fun r hd hh => NciCheck.Record.SecurityProofs.declared_of_heldAt h hh⟩

theorem layoutOf_reads {d d' : Defn} (hp : d.params = d'.params) (hr : d.ret = d'.ret) :
    NciCheck.Record.layoutOf d = NciCheck.Record.layoutOf d' := by
  unfold NciCheck.Record.layoutOf
  rw [hp, hr]

/-- **The check's layout reads the signature alone.** -/
theorem layoutReads_recordForm : LayoutReads recordForm := by
  intro d d' hp hr
  exact congrArg (Option.map fun l : NciCheck.Record.Layout =>
    ({ key := l.key, root := l.root, domain := l.domain, opened := l.ports, verdict := l.verdict } : Roles))
    (layoutOf_reads hp hr)

/-- Where the rate rule refuses nothing on one pair, the pair meets both bounds. -/
theorem failure_one_none {Γ num den a b : ℕ} (h : NciCheck.failure Γ num den #[(a, b)] = none) :
    a ≤ Γ ∧ b * den ≤ num * a := by
  unfold NciCheck.failure at h
  have h0 : (0 : ℕ) ∈ List.range (#[(a, b)] : Array (ℕ × ℕ)).size := List.mem_range.mpr Nat.zero_lt_one
  have hf := List.findSome?_eq_none_iff.mp h 0 h0
  have hf' : (if a > Γ then some ((0 : ℕ), "size") else if b * den > num * a then some (0, "nci") else none) =
      none := hf
  by_cases h1 : a > Γ
  · rw [if_pos h1] at hf'
    cases hf'
  · rw [if_neg h1] at hf'
    by_cases h2 : b * den > num * a
    · rw [if_pos h2] at hf'
      cases hf'
    · exact ⟨Nat.le_of_not_gt h1, Nat.le_of_not_gt h2⟩

/-- **The check's rate rule bounds a summary**, whatever `Γ`, `num` and `den`: `den = 0` among them, where the second
bound says nothing. -/
theorem rates_recordForm : Rates recordForm := by
  intro Γ num den s h
  apply failure_one_none
  cases hf : NciCheck.failure Γ num den #[(s.gates, s.fresh)] with
  | none => rfl
  | some p =>
    have h' : (NciCheck.failure Γ num den #[(s.gates, s.fresh)]).map (·.2) = none := h
    rw [hf] at h'
    simp at h'

end NciCheck.Properties.Record.Attack
