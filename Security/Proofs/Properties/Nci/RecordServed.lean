module

public import Proofs.Properties.Nci.Record
import all Lean.Data.Json.Basic
import all Init.Data.Ord.String
import all Init.Data.String.Defs
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
import all Init.Data.List.Control
import all Init.Data.Array.Basic
import all Init.Data.Array.DecidableEq
import all Init.WF

/-!
# Is any circuit served? A record, its text, and what stands in the way

A scratch file, never merged, beside `Proofs.Properties.Nci.RecordAttack`.

* **The text's tie, through the kernel.** `readTie` is what `Served` reads of a record's text: the parse, then
  `Spot.Wires.Wiring.ofJson` of `wiresOf`. The kernel runs it on texts given as their characters: a text of no wire
  (`tie_empty`), one of a cut and a wire (`tie_wire`), `wires` given twice (`tie_twice`: the last), `cuts` null and
  `wires` absent and a text that is a list (`tie_cutsNull`, `tie_noWires`, `tie_list`: refused), `templates` and
  `order` stated (`tie_dropped`: read as none), and `blocks` and `registered` stated (`tie_rest`: read as the text of
  no wire is).
* **A served record, but for the decoder.** `served_empty`: the record of no wire over the Program of no node lays
  out the circuit of no gate, given that `Flock.Program.decode` reads the Program's text, that the template query
  finds no unit in it and that it has no gate. Those three are about the decoder's output, which the kernel does not
  compute: the decoder keeps its definitions in a hash map. Every other clause of `Served` is proved.
-/

@[expose] public section

namespace NciCheck.Properties.Record.Attack

open Lean (Json)
open Flock.Program (Prog Scope Ref Ty Defn)
open Flock.TemplateQuery (TRange)

deriving instance DecidableEq for Spot.Wires.Port
deriving instance DecidableEq for Spot.Wires.Template
deriving instance DecidableEq for Spot.Wires.Wire
deriving instance DecidableEq for Spot.Wires.Wiring
deriving instance DecidableEq for Spot.Wires.Refusal
deriving instance DecidableEq for Spot.Wires.Summary

theorem ok_of_toOption {ε α : Type} {x : Except ε α} {a : α} (h : x.toOption = some a) : x = .ok a := by
  cases x <;> simp_all [Except.toOption]

/-! ### The text's tie -/

/-- What `Served` reads of a record's text, given as its characters: the wiring `Spot.Wires.Wiring.ofJson` reads of
the parsed text's `wiresOf`, none where the parse or the reading refuses. -/
def readTie (cs : List Char) : Option Spot.Wires.Wiring :=
  (Flock.JsonParse.parseList cs).toOption.bind fun doc => (Spot.Wires.Wiring.ofJson (wiresOf doc)).toOption

/-- `readTie` is `Served`'s two clauses on the text. -/
theorem tie_of {cs : List Char} {W : Spot.Wires.Wiring} (h : readTie cs = some W) :
    ∃ doc, Flock.JsonParse.parse (String.ofList cs) = .ok doc ∧ Spot.Wires.Wiring.ofJson (wiresOf doc) = .ok W := by
  unfold readTie at h
  obtain ⟨doc, hdoc, hW⟩ := Option.bind_eq_some_iff.1 h
  refine ⟨doc, ?_, ok_of_toOption hW⟩
  rw [Flock.JsonParse.parse_eq, String.toList_ofList]
  exact ok_of_toOption hdoc

/-- A text `readTie` refuses serves no circuit, whatever the check. -/
theorem not_served_of_tie {F : Form} {c : F.Case} {cs : List Char} (hrec : (F.inst c).record = String.ofList cs)
    (h : readTie cs = none) (d : Verity.RegisteredCircuit) : ¬ Served F c d := by
  rintro ⟨P, doc, W, S, rs, at_, -, hparse, hof, -⟩
  rw [hrec, Flock.JsonParse.parse_eq, String.toList_ofList] at hparse
  have h' : readTie cs = (Spot.Wires.Wiring.ofJson (wiresOf doc)).toOption := by
    unfold readTie
    rw [hparse]
    rfl
  rw [h', hof] at h
  simp [Except.toOption] at h

/-- `{"wires":[]}` -/
def recordEmpty : List Char :=
  ['{', '"', 'w', 'i', 'r', 'e', 's', '"', ':', '[', ']', '}']

/-- A cut of `B`'s port `x` in two, and one wire of two rows from `A`'s output `o0`: `cuts` is
`{"B":{"x":2}}`, and the wire has `at` 0, `base` 0, `cut` `[0,2]`, `radix` `[2]` and `step` `[1]`. -/
def recordWire : List Char :=
  ['{', '"', 'c', 'u', 't', 's', '"', ':', '{', '"', 'B', '"', ':', '{', '"', 'x', '"', ':', '2', '}', '}',
   ',', '"', 'w', 'i', 'r', 'e', 's', '"', ':', '[', '{', '"', 'a', 't', '"', ':', '0', ',', '"', 'b', 'a',
   's', 'e', '"', ':', '0', ',', '"', 'c', 'u', 't', '"', ':', '[', '0', ',', '2', ']', ',', '"', 'o', 'u',
   't', '"', ':', '"', 'o', '0', '"', ',', '"', 'p', 'o', 'r', 't', '"', ':', '"', 'x', '"', ',', '"', 'r',
   'a', 'd', 'i', 'x', '"', ':', '[', '2', ']', ',', '"', 'r', 'e', 'a', 'd', 'e', 'r', '"', ':', '"', 'B',
   '"', ',', '"', 's', 't', 'e', 'p', '"', ':', '[', '1', ']', ',', '"', 'w', 'r', 'i', 't', 'e', 'r', '"',
   ':', '"', 'A', '"', '}', ']', '}']

/-- `wires` given twice: one wire from `A` to `B`, then `[]`. -/
def recordTwice : List Char :=
  ['{', '"', 'w', 'i', 'r', 'e', 's', '"', ':', '[', '{', '"', 'a', 't', '"', ':', '0', ',', '"', 'b', 'a',
   's', 'e', '"', ':', '0', ',', '"', 'o', 'u', 't', '"', ':', '"', 'o', '0', '"', ',', '"', 'p', 'o', 'r',
   't', '"', ':', '"', 'x', '"', ',', '"', 'r', 'a', 'd', 'i', 'x', '"', ':', '[', '1', ']', ',', '"', 'r',
   'e', 'a', 'd', 'e', 'r', '"', ':', '"', 'B', '"', ',', '"', 's', 't', 'e', 'p', '"', ':', '[', '0', ']',
   ',', '"', 'w', 'r', 'i', 't', 'e', 'r', '"', ':', '"', 'A', '"', '}', ']', ',', '"', 'w', 'i', 'r', 'e',
   's', '"', ':', '[', ']', '}']

/-- `{"cuts":null,"wires":[]}` -/
def recordCutsNull : List Char :=
  ['{', '"', 'c', 'u', 't', 's', '"', ':', 'n', 'u', 'l', 'l', ',', '"', 'w', 'i', 'r', 'e', 's', '"', ':',
   '[', ']', '}']

/-- `{"cuts":{}}` -/
def recordNoWires : List Char :=
  ['{', '"', 'c', 'u', 't', 's', '"', ':', '{', '}', '}']

/-- `[]` -/
def recordList : List Char :=
  ['[', ']']

/-- No wire, with `order` `[["T",0,1]]` and `templates` a template `T` of one unit. -/
def recordDropped : List Char :=
  ['{', '"', 'o', 'r', 'd', 'e', 'r', '"', ':', '[', '[', '"', 'T', '"', ',', '0', ',', '1', ']', ']', ',',
   '"', 't', 'e', 'm', 'p', 'l', 'a', 't', 'e', 's', '"', ':', '{', '"', 'T', '"', ':', '{', '"', 'o', 'u',
   't', 's', '"', ':', '{', '}', ',', '"', 'r', 'o', 'w', 's', '"', ':', '{', '}', ',', '"', 'u', 'n', 'i',
   't', 's', '"', ':', '1', '}', '}', ',', '"', 'w', 'i', 'r', 'e', 's', '"', ':', '[', ']', '}']

/-- No wire, with `blocks` an inference grid of no slot and `registered` a parameter `w`. -/
def recordRest : List Char :=
  ['{', '"', 'b', 'l', 'o', 'c', 'k', 's', '"', ':', '{', '"', 'i', 'n', 'f', 'e', 'r', 'e', 'n', 'c', 'e',
   '"', ':', '{', '"', 'e', 'n', 't', 'r', 'i', 'e', 's', '"', ':', '[', ']', ',', '"', 's', 'l', 'o', 't',
   's', '"', ':', '[', ']', '}', '}', ',', '"', 'r', 'e', 'g', 'i', 's', 't', 'e', 'r', 'e', 'd', '"', ':',
   '{', '"', 'p', 'a', 'r', 'a', 'm', 's', '"', ':', '[', '"', 'w', '"', ']', '}', ',', '"', 'w', 'i', 'r',
   'e', 's', '"', ':', '[', ']', '}']

theorem tie_empty : readTie recordEmpty = some ⟨[], [], [], none⟩ := by decide +kernel

theorem tie_wire :
    readTie recordWire = some ⟨[], [("B", [("x", 2)])], [⟨"B", "x", "A", "o0", 0, [2], 0, [1], 0, 2⟩], none⟩ := by
  decide +kernel

/-- `wires` given twice: the last is read. -/
theorem tie_twice : readTie recordTwice = some ⟨[], [], [], none⟩ := by decide +kernel

theorem tie_cutsNull : readTie recordCutsNull = none := by decide +kernel

theorem tie_noWires : readTie recordNoWires = none := by decide +kernel

theorem tie_list : readTie recordList = none := by decide +kernel

/-- `templates` and `order` stated: neither is read. -/
theorem tie_dropped : readTie recordDropped = some ⟨[], [], [], none⟩ := by decide +kernel

/-- `blocks` and `registered` stated: the text reads as the text of no wire does. -/
theorem tie_rest : readTie recordRest = readTie recordEmpty := by decide +kernel

/-! ### The wires check on the wiring of no wire -/

theorem check_empty_stated : Spot.Wires.check ⟨[], [], [], some []⟩ = .ok ⟨[], []⟩ :=
  ok_of_toOption (by decide +kernel)

theorem check_empty_default : Spot.Wires.check ⟨[], [], [], none⟩ = .ok ⟨[], []⟩ :=
  ok_of_toOption (by decide +kernel)

/-! ### A served record, but for the decoder -/

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

/-- A descriptor whose root has no node. -/
def programEmpty : String :=
  "{\"annotations\":{},\"callees\":[],\"definitions\":{\"Root_v1\":{\"bindings\":{},\"body\":{\"nodes\"" ++
  ":[],\"params\":[],\"ret\":{\"concat\":[]},\"ret_type\":[\"Tuple\",[]]},\"form\":\"composite\",\"id\"" ++
  ":\"Root_v1\"}},\"program_fn\":\"Root_v1\",\"root\":\"Root_v1\",\"schema\":\"verity-ir/descriptor/v1" ++
  "\",\"types\":[]}"
/-- A check that reads every instance as the record of no wire over `programEmpty`, its wiring stating the order
`o`, with no slot and no held row. -/
def formEmpty (o : Option (List (String × ℕ × ℕ))) : Form where
  Case := Unit
  read := fun _ _ => .ok ()
  inst := fun _ =>
    { program := programEmpty, record := String.ofList recordEmpty, gamma := 0, num := 0, den := 1, pins := [] }
  wiring := fun _ => { templates := [], cuts := [], wires := [], order := o }
  slot := fun _ _ _ => none
  held := fun _ _ => none
  summarize := fun _ => ⟨0, 0⟩
  rate := fun _ _ _ _ => none
  layout := fun _ => none

/-- **A served record, but for the decoder.** Given that the decoder reads `programEmpty`, that the template query
finds no unit in what it reads and that it has no gate, the record of no wire lays out the circuit of no gate. -/
theorem served_empty (o : Option (List (String × ℕ × ℕ))) {P : Prog}
    (hcheck : Spot.Wires.check ⟨[], [], [], o⟩ = .ok ⟨[], []⟩)
    (hdec : Flock.Program.decode programEmpty.toUTF8 = .ok P)
    (hev : Flock.TemplateQuery.evaluate (Flock.TemplateQuery.Desc.ofProg P) #[] = .ok #[])
    (hg : gates P = 0) : Served (formEmpty o) () dEmpty := by
  obtain ⟨doc, hparse, hof⟩ := tie_of tie_empty
  have hunit : ∀ (tid : String) (u : ℕ) (sc : Scope), ¬ UnitAt P #[] tid u sc := by
    rintro tid u sc ⟨i, m, hi, -⟩
    exact absurd hi (by simp)
  have hleaf : ∀ (W : Spot.Wires.Wiring) (r : Row) (l g lo hi : ℕ), ¬ RowLeaf P #[] W r l g lo hi := by
    rintro W r l g lo hi ⟨sc, hu, -⟩
    exact hunit _ _ _ hu
  have hwriter : ∀ (tid : String) (u : ℕ) (port : String) (row : ℕ),
      (⟨[], []⟩ : Spot.Wires.Summary).writerOf tid u port row = none := fun _ _ _ _ => rfl
  refine ⟨P, doc, ⟨[], [], [], none⟩, ⟨[], []⟩, #[], fun x => x.elim0, hdec, hparse, hof, rfl, rfl, hcheck, hev,
    ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro tid t ht
    simp [formEmpty] at ht
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

/-- With the order stated, the wires check is the kernel's. -/
theorem served_empty_stated {P : Prog} (hdec : Flock.Program.decode programEmpty.toUTF8 = .ok P)
    (hev : Flock.TemplateQuery.evaluate (Flock.TemplateQuery.Desc.ofProg P) #[] = .ok #[])
    (hg : gates P = 0) : Served (formEmpty (some [])) () dEmpty :=
  served_empty (some []) check_empty_stated hdec hev hg

/-- So, given the three, the check lays out a circuit for every record it reads. -/
theorem laidOut_empty {P : Prog} (hdec : Flock.Program.decode programEmpty.toUTF8 = .ok P)
    (hev : Flock.TemplateQuery.evaluate (Flock.TemplateQuery.Desc.ofProg P) #[] = .ok #[])
    (hg : gates P = 0) : LaidOut (formEmpty (some [])) :=
  fun _ _ _ _ => ⟨dEmpty, served_empty_stated hdec hev hg⟩

end NciCheck.Properties.Record.Attack
