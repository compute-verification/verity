module

public import Proofs.Properties.Nci.RecordTie
public import Proofs.Properties.Nci.DecodeText

/-!
# A served record, but for the decoder

A scratch file, never merged. `served_empty`: given that the decoder reads the text of the Program of no node, that
the template query finds no unit in what it reads and that it has no gate, the record of no wire lays out the
circuit of no gate. The text names no Program, so its readers give no slot and declare no row
(`Tie.slot_noProgram`), and the check has none: the text's tie costs the witness nothing more.
-/

@[expose] public section

namespace NciCheck.Properties.Record.Tie

open Lean (Json)
open Flock.Program (Prog Scope Ref Ty Defn)
open Flock.TemplateQuery (TRange)
open NciCheck.RecordText (Row Held Declared wiresOf)
open Flock.Program.Closed (programEmpty)

/-- The circuit of no gate. -/
abbrev circuitEmpty : Verity.Circuit where
  N := 0
  w := fun g => g.elim0
  isInput := fun g => g.elim0
  reads := fun g => g.elim0
  reads_lt := fun g => g.elim0
  input_reads := fun g => g.elim0
  op := fun g => g.elim0
  op_congr := fun g => g.elim0
  outputs := ∅

/-- `circuitEmpty`, nothing registered. -/
abbrev dEmpty : Verity.RegisteredCircuit := .ofCircuit circuitEmpty ∅ (fun g => g.elim0)

/-- **A served record, but for the decoder**: the check with no slot and no held row, on the record of no wire
beside the Program of no node, lays out the circuit of no gate. -/
theorem served_empty {P : Prog} (hdec : Flock.Program.decode programEmpty.toUTF8 = .ok P)
    (hev : Flock.TemplateQuery.evaluate (Flock.TemplateQuery.Desc.ofProg P) #[] = .ok #[]) (hg : gates P = 0) :
    Served (formTie programEmpty none none) () dEmpty := by
  obtain ⟨doc, hparse, hof⟩ := tie_of tie_empty
  have hprog := program_of hparse noProgram_empty
  have hunit : ∀ (tid : String) (u : ℕ) (sc : Scope), ¬ UnitAt P #[] tid u sc := by
    rintro tid u sc ⟨i, m, hi, -⟩
    exact absurd hi (by simp)
  have hleaf : ∀ (W : Spot.Wires.Wiring) (r : Row) (l g lo hi : ℕ), ¬ RowLeaf P #[] W r l g lo hi := by
    rintro W r l g lo hi ⟨sc, hu, -⟩
    exact hunit _ _ _ hu
  have hwriter : ∀ (tid : String) (u : ℕ) (port : String) (row : ℕ),
      (⟨[], []⟩ : Spot.Wires.Summary).writerOf tid u port row = none := fun _ _ _ _ => rfl
  refine ⟨P, doc, ⟨[], [], [], none⟩, ⟨[], []⟩, #[], fun x => x.elim0, hdec, hparse, hof, rfl, rfl,
    ?_, ?_, check_empty, hev, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · funext tid u
    exact (slot_noProgram hprog tid u).symm
  · intro r h hh
    exact absurd hh (by simp [formTie])
  · intro tid t ht
    simp [formTie] at ht
  · exact ⟨fun r l g lo hi h => absurd h (hleaf _ r l g lo hi),
      fun r l lo hi r' l' lo' hi' g h _ => absurd h (hleaf _ r l g lo hi)⟩
  · rintro r l' g' lo' hi' ⟨wt, wu, out, sc', -, -, hu, -⟩
    exact absurd hu (hunit _ _ _)
  · exact fun r l g lo hi _ h => absurd h (hleaf _ r l g lo hi)
  · exact fun r _ => hwriter _ _ _ _
  · refine ⟨fun x => x.elim0, fun g => ⟨?_, ?_⟩, fun x => x.elim0⟩
    · rintro ⟨x, -⟩
      exact x.elim0
    · rintro ⟨hlt, -⟩
      rw [hg] at hlt
      exact absurd hlt (Nat.not_lt_zero g)
  · exact fun x => x.elim0
  · exact fun x => x.elim0

/-- With a slot given to any unit, the same check on the same texts is served by no circuit. -/
theorem not_served_empty_slotted (s : ℕ) (d : Verity.RegisteredCircuit) :
    ¬ Served (formTie programEmpty (some s) none) () d :=
  not_served_slotted programEmpty s none d

end NciCheck.Properties.Record.Tie
