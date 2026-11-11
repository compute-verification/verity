module

public import Proofs.Properties.Nci.RecordAttack
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
  (`tie_empty`), one of a cut (`tie_cuts`), `wires` given twice (`tie_twice`: the last), `cuts` null and `wires`
  absent and a text that is a list (`tie_cutsNull`, `tie_noWires`, `tie_list`: refused), `templates` and `order`
  stated (`tie_dropped`: read as none), `blocks` and `registered` stated (`tie_rest`) and a stated record's `cap`,
  `format` and `templates` (`tie_stated`): each read as the text of no wire is; a cut of 1 is read as stated
  (`tie_cutOne`). Two texts of one tie serve the same circuits (`served_retext_tie`). A text with a wire is not run
  here: the reading of a wire sorts its keys by a recursion the kernel does not unfold.
* **A served record, but for the decoder.** `served_empty`: the record of no wire over the Program of no node lays
  out the circuit of no gate, given that `Flock.Program.decode` reads the Program's text, that the template query
  finds no unit in it and that it has no gate. Those three are about the decoder's output, which the kernel does not
  compute: the decoder keeps its definitions in a hash map. Every other clause of `Served` is proved.
* **One unit.** `served_one`: the record of no wire over a Program of one input gate and one unit that reads it lays
  out the circuit of those two gates, given that the wires check passes the unit's wiring (`check_one`, through the
  kernel, its order stated: the kernel does not unfold the order the check makes where none is stated) and given the
  decoder's output on that Program as `OneUnit` states it, gate by gate. With the unit's row held, θ is the input
  gate (`served_one_held`) and the block reads no fresh bit (`fresh_one_held`); with nothing held it reads sixteen
  (`served_one_fresh`, `fresh_one`). The record's text is the same in both, and registers nothing; a text with a
  stated record's `cap`, `format` and `templates` serves the same (`served_one_stated`).
* **What the summary counts.** `counts_one_checked`: on that one text, the check that holds the row counts every
  block at no fresh bit, in every circuit the record lays out (`served_one_shape`), and the check that holds nothing
  does not.
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

/-- A text that parses, and whose wiring `Spot.Wires.Wiring.ofJson` reads, is one `readTie` reads. -/
theorem readTie_of {cs : List Char} {doc : Json} {W : Spot.Wires.Wiring}
    (hp : Flock.JsonParse.parse (String.ofList cs) = .ok doc)
    (ho : Spot.Wires.Wiring.ofJson (wiresOf doc) = .ok W) : readTie cs = some W := by
  rw [Flock.JsonParse.parse_eq, String.toList_ofList] at hp
  have h' : readTie cs = (Spot.Wires.Wiring.ofJson (wiresOf doc)).toOption := by
    unfold readTie
    rw [hp]
    rfl
  rw [h', ho] <;> rfl

/-- **Two texts of one tie serve the same circuits**, whatever else either states. -/
theorem served_retext_tie {F : Form} {c : F.Case} {d : Verity.RegisteredCircuit} {s : F.Case → String}
    {cs cs' : List Char} (hrec : (F.inst c).record = String.ofList cs) (hs : s c = String.ofList cs')
    (h : readTie cs' = readTie cs) (hd : Served F c d) : Served (retext F s) c d := by
  have hd' := hd
  obtain ⟨P, doc, W, S, rs, at_, -, hparse, hof, hc, hw, -⟩ := hd'
  rw [hrec] at hparse
  obtain ⟨doc', hp', ho'⟩ := tie_of (h.trans (readTie_of hparse hof))
  exact served_retext_of F s c d ((congrArg Flock.JsonParse.parse hs).trans hp') ho' hc hw hd

/-- `{"wires":[]}` -/
def recordEmpty : List Char :=
  ['{', '"', 'w', 'i', 'r', 'e', 's', '"', ':', '[', ']', '}']

/-- `{"cuts":{"B":{"x":2}},"wires":[]}` -/
def recordCuts : List Char :=
  ['{', '"', 'c', 'u', 't', 's', '"', ':', '{', '"', 'B', '"', ':', '{', '"', 'x', '"', ':', '2', '}', '}',
   ',', '"', 'w', 'i', 'r', 'e', 's', '"', ':', '[', ']', '}']

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

theorem tie_cuts : readTie recordCuts = some ⟨[], [("B", [("x", 2)])], [], none⟩ := by decide +kernel

/-- `wires` given twice: the last is read. -/
theorem tie_twice : readTie recordTwice = some ⟨[], [], [], none⟩ := by decide +kernel

theorem tie_cutsNull : readTie recordCutsNull = none := by decide +kernel

theorem tie_noWires : readTie recordNoWires = none := by decide +kernel

theorem tie_list : readTie recordList = none := by decide +kernel

/-- `templates` and `order` stated: neither is read. -/
theorem tie_dropped : readTie recordDropped = some ⟨[], [], [], none⟩ := by decide +kernel

/-- `blocks` and `registered` stated: the text reads as the text of no wire does. -/
theorem tie_rest : readTie recordRest = readTie recordEmpty := by decide +kernel

/-- `{"cuts":{"B":{"x":1}},"wires":[]}` -/
def recordCutOne : List Char :=
  ['{', '"', 'c', 'u', 't', 's', '"', ':', '{', '"', 'B', '"', ':', '{', '"', 'x', '"', ':', '1', '}', '}',
   ',', '"', 'w', 'i', 'r', 'e', 's', '"', ':', '[', ']', '}']

/-- A cut of 1 the text states is read as stated, not as none. -/
theorem tie_cutOne : readTie recordCutOne = some ⟨[], [("B", [("x", 1)])], [], none⟩ := by decide +kernel

/-- `{"cap":1,"format":"verity/window-circuit/v1","templates":[["Copy_v1",1]],"wires":[]}` -/
def recordStated : List Char :=
  ['{', '"', 'c', 'a', 'p', '"', ':', '1', ',', '"', 'f', 'o', 'r', 'm', 'a', 't', '"', ':', '"', 'v', 'e',
   'r', 'i', 't', 'y', '/', 'w', 'i', 'n', 'd', 'o', 'w', '-', 'c', 'i', 'r', 'c', 'u', 'i', 't', '/', 'v',
   '1', '"', ',', '"', 't', 'e', 'm', 'p', 'l', 'a', 't', 'e', 's', '"', ':', '[', '[', '"', 'C', 'o', 'p',
   'y', '_', 'v', '1', '"', ',', '1', ']', ']', ',', '"', 'w', 'i', 'r', 'e', 's', '"', ':', '[', ']', '}']

/-- A text with a stated record's `cap`, `format` and `templates` is read as the text of no wire is. -/
theorem tie_stated : readTie recordStated = readTie recordEmpty := by decide +kernel

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

/-! ### One unit -/

/-- A descriptor of one input gate and one unit: root node 0 is the input gate, root node 1 a call of `Copy_v1` on
it, and `Copy_v1(x)` is one gate of two operands, both `x`. -/
def programOne : String :=
  "{\"annotations\":{},\"callees\":[\"And16_v1\",\"Input16_v1\",\"Copy_v1\"],\"definitions\":{\"And16_v" ++
  "1\":{\"form\":\"primitive\",\"id\":\"And16_v1\",\"params\":[[\"a\",[\"Value\",16]],[\"b\",[\"Value\"" ++
  ",16]]],\"ret\":[\"Value\",16]},\"Copy_v1\":{\"bindings\":{},\"body\":{\"nodes\":[{\"args\":[{\"refs" ++
  "\":{\"affine\":[\"p\",0,0,1]},\"type\":0},{\"refs\":{\"affine\":[\"p\",0,0,1]},\"type\":0}],\"fn\":0" ++
  ",\"form\":\"primitive\",\"out\":0}],\"params\":[[\"x\",[\"Value\",16]]],\"ret\":{\"affine\":[\"n\",0" ++
  ",0,1]},\"ret_type\":[\"Value\",16]},\"form\":\"composite\",\"id\":\"Copy_v1\"},\"Input16_v1\":{\"for" ++
  "m\":\"primitive\",\"id\":\"Input16_v1\",\"params\":[],\"ret\":[\"Value\",16]},\"Root_v1\":{\"binding" ++
  "s\":{},\"body\":{\"nodes\":[{\"args\":[],\"fn\":1,\"form\":\"primitive\",\"out\":0},{\"args\":[{\"re" ++
  "fs\":{\"affine\":[\"n\",0,0,1]},\"type\":0}],\"fn\":2,\"form\":\"call\",\"out\":0}],\"params\":[],\"" ++
  "ret\":{\"affine\":[\"n\",1,0,1]},\"ret_type\":[\"Value\",16]},\"form\":\"composite\",\"id\":\"Root_v" ++
  "1\"}},\"program_fn\":\"Root_v1\",\"root\":\"Root_v1\",\"schema\":\"verity-ir/descriptor/v1\",\"types" ++
  "\":[[\"Value\",16]]}"
/-- What the decoder's output on `programOne` is taken to be: the template query's one range, the root's two gates
with their primitives and operands, and the unit's activation with its one parameter, its return and where its
parameter resolves. Each line is evaluated on the decoder's output beside this file; none is proved. -/
structure OneUnit (P : Prog) : Prop where
  ranges : Flock.TemplateQuery.evaluate (Flock.TemplateQuery.Desc.ofProg P) #["Copy_v1"] =
    .ok #[⟨1, "Copy_v1", 0, 1⟩]
  rootGates : NciCheck.Properties.gates P = 2
  gate0 : ∃ p sc, P.gate 0 = .ok (p, #[], sc) ∧ (P.defs[p]!).ret = .value 16 ∧
    Flock.Program.isInputId (P.defs[p]!).key = true
  gate1 : ∃ p sc, P.gate 1 = .ok (p, #[0, 0], sc) ∧ (P.defs[p]!).ret = .value 16 ∧
    Flock.Program.isInputId (P.defs[p]!).key = false
  off : (P.child P.rootScope 1 0).off = 1
  unitGates : (P.defs[(P.child P.rootScope 1 0).d]!).gates = 1
  params : (P.defs[(P.child P.rootScope 1 0).d]!).params = #[("x", .value 16)]
  ret : (P.defs[(P.child P.rootScope 1 0).d]!).ret = .value 16
  port : Flock.Program.resolve P (P.child P.rootScope 1 0) ⟨true, 0, 0⟩ = .ok 0

/-- The template query's one range. -/
def rsOne : Array TRange := #[⟨1, "Copy_v1", 0, 1⟩]

/-- `Copy_v1`'s template: one unit, one row `x` of one word, one output `o0` of one word. -/
def templateOne : Spot.Wires.Template :=
  { units := 1, rows := [("x", { words := 1, outer := 1 })], outs := [("o" ++ toString 0, 1)] }

/-- The wiring of no wire over `Copy_v1`'s one unit, its order stated. -/
def wiringOne : Spot.Wires.Wiring :=
  { templates := [("Copy_v1", templateOne)], cuts := [], wires := [], order := some [("Copy_v1", 0, 1)] }

/-- The gates each gate of `circuitOne` reads. -/
def readsOne : Fin 2 → Finset (Fin 2) := ![∅, {0}]

/-- Two gates of 16 bits: gate 0 an input, gate 1 reading it. -/
abbrev circuitOne : Verity.Circuit where
  N := 2
  w := fun _ => 16
  isInput := fun g => decide (g.val = 0)
  reads := readsOne
  reads_lt := by decide
  input_reads := by decide
  op := fun _ _ => 0
  op_congr := fun _ _ _ _ => rfl
  outputs := {1}

/-- One block: gate 1. -/
def blocksOne : Verity.Partition circuitOne 1 where
  unit := fun g => if g.val = 0 then none else some 0
  input_iff := by decide

/-- `circuitOne`, nothing registered. -/
abbrev dOne : Verity.RegisteredCircuit := .ofPartition circuitOne ∅ (by decide) 1 blocksOne

/-- `circuitOne`, its input gate registered. -/
abbrev dOneHeld : Verity.RegisteredCircuit := .ofPartition circuitOne {0} (by decide) 1 blocksOne

/-- A check that reads every instance as the record of no wire over `programOne`, every unit in block 0, holding
every row for the reason `hd`, or none, its summary one gate and `fresh` fresh bits. -/
def formOne (hd : Option Held) (fresh : ℕ) : Form where
  Case := Unit
  read := fun _ _ => .ok ()
  inst := fun _ =>
    { program := programOne, record := String.ofList recordEmpty, gamma := 1, num := 16, den := 1, pins := [] }
  wiring := fun _ => wiringOne
  slot := fun _ _ _ => some 0
  held := fun _ _ => hd
  summarize := fun _ => ⟨1, fresh⟩
  rate := fun _ _ _ _ => none
  layout := fun _ => none

theorem unitAt_one {P : Prog} {tid : String} {u : ℕ} {sc : Scope} :
    UnitAt P rsOne tid u sc ↔ tid = "Copy_v1" ∧ u = 0 ∧ sc = P.child P.rootScope 1 0 := by
  constructor
  · rintro ⟨i, m, hi, ht, hm, hu, hsc⟩
    have hi0 : i = 0 := by
      have : i < 1 := hi
      omega
    subst hi0
    have hm0 : m = 0 := by
      have : m < 1 := hm
      omega
    subst hm0
    have ht' : "Copy_v1" = tid := ht
    subst ht'
    exact ⟨rfl, hu.trans rfl, hsc⟩
  · rintro ⟨rfl, rfl, rfl⟩
    exact ⟨0, 0, by decide, rfl, by decide, rfl, rfl⟩

theorem bits_value : bitsOf (Ty.value 16) = 16 := by
  unfold bitsOf
  rw [Flock.Program.Ty.widths] <;> first | rfl | decide

theorem dvd_value : 16 ∣ bitsOf (Ty.value 16) := ⟨1, bits_value⟩

theorem words_value : bitsOf (Ty.value 16) / 16 = 1 := by rw [bits_value] <;> first | rfl | decide

theorem outs_value : (bitsOf (Ty.value 16) + 15) / 16 = 1 := by rw [bits_value] <;> first | rfl | decide

theorem leaves_value : (Ty.value 16).leaves = 1 := by rw [Flock.Program.Ty.leaves] <;> first | rfl | decide

theorem widths_value : (Ty.value 16).widths = #[16] := by rw [Flock.Program.Ty.widths] <;> first | rfl | decide

theorem inRow_value {d : Defn} {t : Ty} (ht : d.params[0]!.2 = t) (hl : t.leaves = 1) (hw : t.widths = #[16]) :
    inRow d 0 1 0 = (0, #[16]) := by
  unfold inRow
  rw [ht, hl, hw] <;> first | rfl | decide

theorem inRow_one {d : Defn} (hp : d.params = #[("x", .value 16)]) : inRow d 0 1 0 = (0, #[16]) :=
  inRow_value (t := Ty.value 16) (by rw [hp] <;> rfl) leaves_value widths_value

theorem inLeaf_one {P : Prog} (h : OneUnit P) {port : String} {row l g lo hi : ℕ} :
    InLeaf P (P.child P.rootScope 1 0) 1 port row l g lo hi ↔
      port = "x" ∧ row = 0 ∧ l = 0 ∧ g = 0 ∧ lo = 0 ∧ hi = 16 := by
  constructor
  · rintro ⟨i, hi_, hport, hrow, hl, hres, hlo, hhi⟩
    rw [h.params] at hi_ hport
    have hi0 : i = 0 := by
      have : i < 1 := hi_
      omega
    subst hi0
    have hrow0 : row = 0 := by omega
    subst hrow0
    rw [inRow_one h.params] at hl hres hlo hhi
    have hl0 : l = 0 := by
      have : l < 1 := hl
      omega
    subst hl0
    have hlo0 : lo = 0 := hlo.trans (by first | rfl | decide)
    subst hlo0
    exact ⟨hport.symm, rfl, rfl, Except.ok.inj (hres.symm.trans h.port), rfl, hhi⟩
  · rintro ⟨rfl, rfl, rfl, rfl, rfl, rfl⟩
    refine ⟨0, ?_, ?_, by decide, ?_, ?_, ?_, ?_⟩
    · rw [h.params] <;> decide
    · rw [h.params] <;> rfl
    · rw [inRow_one h.params] <;> decide
    · rw [inRow_one h.params] <;> exact h.port
    · rw [inRow_one h.params] <;> first | rfl | decide
    · rw [inRow_one h.params] <;> first | rfl | decide

theorem rowLeaf_one {P : Prog} (h : OneUnit P) {r : Row} {l g lo hi : ℕ} :
    RowLeaf P rsOne wiringOne r l g lo hi ↔
      r.tid = "Copy_v1" ∧ r.unit = 0 ∧ r.port = "x" ∧ r.row = 0 ∧ l = 0 ∧ g = 0 ∧ lo = 0 ∧ hi = 16 := by
  constructor
  · rintro ⟨sc, hu, hin⟩
    obtain ⟨ht, hu0, rfl⟩ := unitAt_one.1 hu
    have hin' : InLeaf P (P.child P.rootScope 1 0) 1 r.port r.row l g lo hi := hin
    exact ⟨ht, hu0, (inLeaf_one h).1 hin'⟩
  · rintro ⟨ht, hu0, hrest⟩
    exact ⟨_, unitAt_one.2 ⟨ht, hu0, rfl⟩, (inLeaf_one h).2 hrest⟩

theorem writer_none (tid : String) (u : ℕ) (port : String) (row : ℕ) :
    (⟨[], []⟩ : Spot.Wires.Summary).writerOf tid u port row = none := rfl

theorem not_wired_none (P : Prog) (rs : Array TRange) (W : Spot.Wires.Wiring) (g : ℕ) :
    ¬ Wired P rs W ⟨[], []⟩ g := by
  rintro ⟨r, l, lo, hi, hw, -⟩
  rw [writer_none] at hw
  simp at hw

theorem templateOf_one {d : Defn} (hp : d.params = #[("x", .value 16)]) (hr : d.ret = .value 16) :
    TemplateOf d templateOne := by
  unfold TemplateOf
  rw [hp, hr]
  refine ⟨?_, ?_, ?_, ?_⟩
  · intro i i' hi hi' _
    have h1 : i < 1 := hi
    have h2 : i' < 1 := hi'
    omega
  · intro i hi
    have hi0 : i = 0 := by
      have : i < 1 := hi
      omega
    subst hi0
    exact dvd_value
  · intro p q
    constructor
    · intro hpq
      obtain ⟨rfl, rfl⟩ := Prod.mk.inj (List.mem_singleton.1 hpq)
      exact ⟨0, by decide, rfl, words_value.symm, rfl⟩
    · rintro ⟨i, hi, hp', hw, ho⟩
      have hi0 : i = 0 := by
        have : i < 1 := hi
        omega
      subst hi0
      have hp2 : p = "x" := hp'
      have hw2 : q.words = 1 := hw.trans words_value
      have ho2 : q.outer = 1 := ho
      obtain ⟨w, o⟩ := q
      have hw3 : w = 1 := hw2
      have ho3 : o = 1 := ho2
      subst hp2 hw3 ho3
      exact List.mem_singleton.2 rfl
  · intro o n
    have e4 : (bitsOf (fieldsOf (Ty.value 16))[0]! + 15) / 16 = 1 := outs_value
    constructor
    · intro hon
      obtain ⟨rfl, rfl⟩ := Prod.mk.inj (List.mem_singleton.1 hon)
      exact ⟨0, by decide, rfl, e4.symm⟩
    · rintro ⟨f, hf, ho, hn⟩
      have hf0 : f = 0 := by
        have : f < 1 := hf
        omega
      subst hf0
      have hn' : n = 1 := hn.trans e4
      subst ho hn'
      exact List.mem_singleton.2 rfl

theorem templates_one {P : Prog} (h : OneUnit P) : Templates P rsOne wiringOne := by
  intro tid t ht
  obtain ⟨rfl, rfl⟩ := Prod.mk.inj (List.mem_singleton.1 ht)
  refine ⟨fun u => ⟨fun hu => ⟨P.child P.rootScope 1 0, unitAt_one.2 ⟨rfl, ?_, rfl⟩⟩, ?_⟩, fun u sc hu => ?_⟩
  · have : u < 1 := hu
    omega
  · rintro ⟨sc, hsc⟩
    obtain ⟨-, hu0, -⟩ := unitAt_one.1 hsc
    subst hu0
    exact (Nat.zero_lt_one : 0 < 1)
  · obtain ⟨-, -, rfl⟩ := unitAt_one.1 hu
    exact templateOf_one h.params h.ret

theorem ownRows_one {P : Prog} (h : OneUnit P) : OwnRows P rsOne wiringOne := by
  constructor
  · intro r l g lo hi hleaf
    obtain ⟨-, -, -, -, -, hg, -, -⟩ := (rowLeaf_one h).1 hleaf
    obtain ⟨p, sc, hgate, -, hin⟩ := h.gate0
    subst hg
    exact ⟨p, #[], sc, hgate, hin⟩
  · intro r l lo hi r' l' lo' hi' g h1 h2
    obtain ⟨a1, a2, a3, a4, a5, -⟩ := (rowLeaf_one h).1 h1
    obtain ⟨b1, b2, b3, b4, b5, -⟩ := (rowLeaf_one h).1 h2
    obtain ⟨t, u, p, w⟩ := r
    obtain ⟨t', u', p', w'⟩ := r'
    have e1 : t = t' := a1.trans b1.symm
    have e2 : u = u' := a2.trans b2.symm
    have e3 : p = p' := a3.trans b3.symm
    have e4 : w = w' := a4.trans b4.symm
    subst e1 e2 e3 e4
    exact ⟨rfl, a5.trans b5.symm⟩

theorem aligned_none (P : Prog) (rs : Array TRange) (W : Spot.Wires.Wiring) : Aligned P rs W ⟨[], []⟩ := by
  rintro r l' g' lo' hi' ⟨wt, wu, out, sc', hw, -⟩
  rw [writer_none] at hw
  simp at hw

theorem filled_none (P : Prog) (rs : Array TRange) (W : Spot.Wires.Wiring) : Filled P rs W ⟨[], []⟩ := by
  intro r l g lo hi hw
  rw [writer_none] at hw
  simp at hw

theorem reads_zero : ∀ a : Fin circuitOne.N, a ∉ circuitOne.reads 0 := by decide

theorem reads_one : ∀ a : Fin circuitOne.N, a ∈ circuitOne.reads 1 ↔ a = 0 := by decide

theorem laysOut_one {P : Prog} (h : OneUnit P) :
    LaysOut P rsOne wiringOne ⟨[], []⟩ circuitOne (fun x => x.val) := by
  refine ⟨fun a b hab => Fin.ext hab, fun g => ⟨?_, ?_⟩, fun x => ?_⟩
  · rintro ⟨x, rfl⟩
    refine ⟨?_, not_wired_none _ _ _ _⟩
    rw [h.rootGates]
    exact x.isLt
  · rintro ⟨hlt, -⟩
    rw [h.rootGates] at hlt
    exact ⟨⟨g, hlt⟩, rfl⟩
  · rcases x with ⟨_ | _ | n, hx⟩
    · obtain ⟨p, sc, hgate, hret, hin⟩ := h.gate0
      refine ⟨p, #[], sc, hgate, hret, hin.symm, fun a => ⟨fun ha => absurd ha (reads_zero a), ?_⟩⟩
      rintro ⟨o, ho, -⟩
      simp at ho
    · obtain ⟨p, sc, hgate, hret, hin⟩ := h.gate1
      refine ⟨p, #[0, 0], sc, hgate, hret, hin.symm, fun a => ⟨fun ha => ?_, ?_⟩⟩
      · have ha0 : a = 0 := (reads_one a).1 ha
        subst ha0
        exact ⟨0, by first | simp | decide, Lands.here 0 (not_wired_none _ _ _ _)⟩
      · rintro ⟨o, ho, hl⟩
        have ho0 : o = 0 := by simpa using ho
        have hao : a.val = o := lands_eq_of_not_wired (not_wired_none _ _ _ _) hl
        exact (reads_one a).2 (Fin.ext (hao.trans ho0))
    · have hx' : n + 1 + 1 < 2 := hx
      omega

theorem divides_one {P : Prog} (h : OneUnit P) :
    Divides P rsOne (fun _ _ => some 0) (D := circuitOne) (fun x => x.val) blocksOne := by
  intro x hx
  have hx1 : x = 1 := by
    revert x
    decide
  subst hx1
  refine ⟨"Copy_v1", 0, P.child P.rootScope 1 0, 0, unitAt_one.2 ⟨rfl, rfl, rfl⟩, ?_, ?_, rfl, by decide⟩
  · rw [h.off] <;> decide
  · rw [h.off, h.unitGates] <;> decide

theorem names_fresh {P : Prog} :
    Names P rsOne wiringOne (fun _ => false) (D := circuitOne) (fun x => x.val) ∅ :=
  fun x => ⟨fun hx => absurd hx (Finset.notMem_empty x), fun ⟨_, _, _, _, hh, _⟩ => absurd hh (by simp)⟩

theorem names_held {P : Prog} (h : OneUnit P) :
    Names P rsOne wiringOne (fun _ => true) (D := circuitOne) (fun x => x.val) {0} := by
  intro x
  constructor
  · intro hx
    have hx0 : x = 0 := Finset.mem_singleton.1 hx
    subst hx0
    exact ⟨⟨"Copy_v1", 0, "x", 0⟩, 0, 0, 16, rfl, (rowLeaf_one h).2 ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩⟩
  · rintro ⟨r, l, lo, hi, -, hleaf⟩
    obtain ⟨-, -, -, -, -, hg, -, -⟩ := (rowLeaf_one h).1 hleaf
    exact Finset.mem_singleton.2 (Fin.ext hg)

/-- **One unit served, but for the decoder.** Given the decoder's output on `programOne` as `OneUnit` states it,
the record of no wire lays out `circuitOne` in one block, θ the gates of the rows the check holds. -/
theorem served_one (hd : Option Held) (n : ℕ) (θ : Finset (Fin circuitOne.N))
    (hθ : ∀ g ∈ θ, circuitOne.isInput g = true) {P : Prog} (hcheck : Spot.Wires.check wiringOne = .ok ⟨[], []⟩)
    (hdec : Flock.Program.decode programOne.toUTF8 = .ok P) (h : OneUnit P)
    (hnames : Names P rsOne wiringOne (fun _ => hd.isSome) (D := circuitOne) (fun x => x.val) θ) :
    Served (formOne hd n) () (.ofPartition circuitOne θ hθ 1 blocksOne) := by
  obtain ⟨doc, hparse, hof⟩ := tie_of tie_empty
  exact ⟨P, doc, ⟨[], [], [], none⟩, ⟨[], []⟩, rsOne, fun x => x.val, hdec, hparse, hof, rfl, rfl, hcheck,
    h.ranges, templates_one h, ownRows_one h, aligned_none _ _ _, filled_none _ _ _, fun _ _ => rfl, laysOut_one h,
    hnames, divides_one h⟩

/-- Nothing held: θ is empty. -/
theorem served_one_fresh (n : ℕ) {P : Prog} (hcheck : Spot.Wires.check wiringOne = .ok ⟨[], []⟩)
    (hdec : Flock.Program.decode programOne.toUTF8 = .ok P) (h : OneUnit P) : Served (formOne none n) () dOne :=
  served_one none n ∅ (by decide) hcheck hdec h names_fresh

/-- The unit's row held as a registered read: θ is the input gate. The record's text is the one `served_one_fresh`
reads, and registers nothing. -/
theorem served_one_held (n : ℕ) {P : Prog} (hcheck : Spot.Wires.check wiringOne = .ok ⟨[], []⟩)
    (hdec : Flock.Program.decode programOne.toUTF8 = .ok P) (h : OneUnit P) :
    Served (formOne (some (.read "w")) n) () dOneHeld :=
  served_one (some (.read "w")) n {0} (by decide) hcheck hdec h (names_held h)

/-- Nothing held: the block reads sixteen fresh bits. -/
theorem fresh_one : ∀ u, Nci.novelReads dOne (dOne.𝓘.gates u) = 16 := by decide

/-- The row held: the block reads no fresh bit. -/
theorem fresh_one_held : ∀ u, Nci.novelReads dOneHeld (dOneHeld.𝓘.gates u) = 0 := by decide

/-- So the block meets the rate rule at `α = 0` where the check holds the row, -/
theorem meets_one_held (u : Fin dOneHeld.n) : Meets dOneHeld dOneHeld.𝓘 1 0 u := by
  refine ⟨?_, ?_⟩
  · revert u
    decide
  · rw [fresh_one_held u]
    simp

/-- and does not where it holds nothing. -/
theorem not_meets_one (u : Fin dOne.n) : ¬ Meets dOne dOne.𝓘 1 0 u := by
  rintro ⟨-, h⟩
  rw [fresh_one u] at h
  norm_num at h

/-! ### What every circuit of the one unit's record is, and what its summary counts -/

/-- Every circuit the one unit's record lays out: a computed gate is the Program's gate 1 and reads only its gate
0, and θ is the gates of the rows the check holds. -/
theorem served_one_shape (hd : Option Held) (n : ℕ) {P : Prog}
    (hcheck : Spot.Wires.check wiringOne = .ok ⟨[], []⟩)
    (hdec : Flock.Program.decode programOne.toUTF8 = .ok P) (h : OneUnit P) {d : Verity.RegisteredCircuit}
    (hs : Served (formOne hd n) () d) :
    ∃ at_ : Fin d.D.N → ℕ, Function.Injective at_ ∧ (∀ x, d.D.isInput x = false → at_ x = 1) ∧
      (∀ x a, d.D.isInput x = false → a ∈ d.D.reads x → at_ a = 0) ∧
      Names P rsOne wiringOne (fun _ => hd.isSome) at_ d.θ := by
  obtain ⟨P', doc, W, S, rs, at_, hdec', -, -, -, -, hcheck', hev, -, -, -, -, -, hlay, hnames, -⟩ := hs
  have hP : P' = P := Except.ok.inj (hdec'.symm.trans hdec)
  subst hP
  have hS : S = ⟨[], []⟩ := Except.ok.inj (hcheck'.symm.trans hcheck)
  subst hS
  have hrs : rs = rsOne := Except.ok.inj (hev.symm.trans h.ranges)
  subst hrs
  obtain ⟨hinj, hran, hgates⟩ := hlay
  have key : ∀ x, d.D.isInput x = false → at_ x = 1 ∧ ∀ a, a ∈ d.D.reads x → at_ a = 0 := by
    intro x hx
    have hlt : at_ x < 2 := by
      have hlt' := ((hran (at_ x)).1 ⟨x, rfl⟩).1
      rw [h.rootGates] at hlt'
      exact hlt'
    obtain ⟨p, ops, sc, hgate, -, hin, hreads⟩ := hgates x
    obtain ⟨p0, sc0, hgate0, -, hin0⟩ := h.gate0
    obtain ⟨p1, sc1, hgate1, -, -⟩ := h.gate1
    have h01 : at_ x = 0 ∨ at_ x = 1 := by omega
    rcases h01 with h0 | h1
    · rw [h0, hgate0] at hgate
      have hp : p0 = p := (Prod.mk.inj (Except.ok.inj hgate)).1
      subst hp
      exact absurd ((hin.trans hin0).symm.trans hx) (by decide)
    · refine ⟨h1, fun a ha => ?_⟩
      rw [h1, hgate1] at hgate
      have hops : (#[0, 0] : Array ℕ) = ops := (Prod.mk.inj (Prod.mk.inj (Except.ok.inj hgate)).2).1
      subst hops
      obtain ⟨o, ho, hl⟩ := (hreads a).1 ha
      have ho0 : o = 0 := by simpa using ho
      exact (lands_eq_of_not_wired (not_wired_none _ _ _ _) hl).trans ho0
  exact ⟨at_, hinj, fun x hx => (key x hx).1, fun x a hx ha => (key x hx).2 a ha, hnames⟩

/-- A gate of a block is computed. -/
theorem computed_of_gates {d : Verity.RegisteredCircuit} {u : Fin d.n} {g : Fin d.D.N} (hg : g ∈ d.𝓘.gates u) :
    d.D.isInput g = false := by
  have hu : d.𝓘.unit g = some u := by simpa [Verity.Partition.gates] using hg
  cases hi : d.D.isInput g with
  | false => rfl
  | true =>
    have hnone := (d.𝓘.input_iff g).2 hi
    rw [hnone] at hu
    exact absurd hu (by simp)

/-- **The summary counts every block, at no fresh bit**, where the check holds the unit's row: in every circuit the
record lays out, the block reads no bit outside θ. The record's text registers nothing. -/
theorem counts_one_held {P : Prog} (hcheck : Spot.Wires.check wiringOne = .ok ⟨[], []⟩)
    (hdec : Flock.Program.decode programOne.toUTF8 = .ok P) (h : OneUnit P) :
    Counts (formOne (some (.read "w")) 0) := by
  intro j tr c _ d hs u
  obtain ⟨at_, hinj, hcomp, hreads, hnames⟩ := served_one_shape _ _ hcheck hdec h hs
  have hempty : d.D.into (d.𝓘.gates u) \ d.θ = ∅ := by
    refine Finset.eq_empty_of_forall_notMem fun a ha => ?_
    obtain ⟨hinto, hnot⟩ := Finset.mem_sdiff.1 ha
    have hinto' : a ∉ d.𝓘.gates u ∧ ∃ g ∈ d.𝓘.gates u, a ∈ d.D.reads g := by
      simpa [Verity.Circuit.into] using hinto
    obtain ⟨-, g, hg, hag⟩ := hinto'
    have ha0 : at_ a = 0 := hreads g a (computed_of_gates hg) hag
    refine hnot ((hnames a).2 ⟨⟨"Copy_v1", 0, "x", 0⟩, 0, 0, 16, rfl, ?_⟩)
    rw [ha0]
    exact (rowLeaf_one h).2 ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩
  refine ⟨?_, ?_⟩
  · have hzero : Nci.novelReads d (d.𝓘.gates u) = 0 := by
      unfold Nci.novelReads
      rw [hempty]
      simp [Verity.Circuit.width]
    exact (Nat.le_of_eq hzero).trans (Nat.zero_le _)
  · rcases Finset.eq_empty_or_nonempty (d.𝓘.gates u) with he | ⟨g, hg⟩
    · exact Or.inr he
    · have hone : d.𝓘.gates u = {g} :=
        Finset.eq_singleton_iff_unique_mem.2 ⟨hg, fun g' hg' =>
          hinj ((hcomp g' (computed_of_gates hg')).trans (hcomp g (computed_of_gates hg)).symm)⟩
      have hcard : (d.𝓘.gates u).card = 1 := by
        rw [hone]
        exact Finset.card_singleton g
      exact Or.inl hcard

/-- The check that holds nothing does not count at no fresh bit: the block reads sixteen. -/
theorem not_counts_one {P : Prog} (hcheck : Spot.Wires.check wiringOne = .ok ⟨[], []⟩)
    (hdec : Flock.Program.decode programOne.toUTF8 = .ok P) (h : OneUnit P) : ¬ Counts (formOne none 0) := by
  intro hC
  have h16 := (hC Json.null Json.null () rfl dOne (served_one_fresh 0 hcheck hdec h) ⟨0, Nat.zero_lt_one⟩).1
  rw [fresh_one] at h16
  exact absurd h16 (by decide)

/-! ### The wires check on the one unit's wiring, through the kernel -/

theorem check_one : Spot.Wires.check wiringOne = .ok ⟨[], []⟩ := ok_of_toOption (by decide +kernel)

/-- So one unit is served, given only the decoder's output. -/
theorem served_one_checked {P : Prog} (hdec : Flock.Program.decode programOne.toUTF8 = .ok P) (h : OneUnit P) :
    Served (formOne none 16) () dOne ∧ Served (formOne (some (.read "w")) 0) () dOneHeld :=
  ⟨served_one_fresh 16 check_one hdec h, served_one_held 0 check_one hdec h⟩

/-- One Program and one record's text, which registers nothing: the check that holds the unit's row counts every
block at no fresh bit, and the check that holds nothing does not. -/
theorem counts_one_checked {P : Prog} (hdec : Flock.Program.decode programOne.toUTF8 = .ok P) (h : OneUnit P) :
    Counts (formOne (some (.read "w")) 0) ∧ ¬ Counts (formOne none 0) :=
  ⟨counts_one_held check_one hdec h, not_counts_one check_one hdec h⟩

/-- The one unit on a text with a stated record's `cap`, `format` and `templates`: `Served` reads of it what it reads
of the text of no wire. -/
theorem served_one_stated {P : Prog} (hdec : Flock.Program.decode programOne.toUTF8 = .ok P) (h : OneUnit P) :
    Served (retext (formOne none 16) fun _ => String.ofList recordStated) () dOne :=
  served_retext_tie (F := formOne none 16) (c := ()) (s := fun _ => String.ofList recordStated)
    (cs := recordEmpty) (cs' := recordStated) rfl rfl tie_stated (served_one_checked hdec h).1

end NciCheck.Properties.Record.Attack
