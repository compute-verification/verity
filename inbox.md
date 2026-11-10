[2026-11-10T23:11Z] 02f5dffc1947
FORM (TRN-76), 3:11 PM PST: the module is through Lean.

MODULE: NciCheck.RecordText (verity/core/protocols/network/nci/lean/NciCheck/RecordText.lean), head 34490b47d8 on #8676 (draft, unmarked).

RUNS at 34490b47d8:
- Lean check r20261110-223554-0041: AUDIT: PASS, 1754 declarations in 17 modules, kernel replay, axioms propext, Classical.choice, Quot.sound; record difftest: 648 cases compared, 0 differences.
- Suites r20261110-223647-27cf: 2 passed (0 cached), 0 failed (279 and 79 tests).
- Served record r20261110-225023-a20e (probe 408b93b056 = that head merged): waiting for a slot. On my machine the compiled form at that head on the served case: 1 cases compared, 0 differences.
(At fcd65c5668: Lean check r20261110-213645-5b22 PASS, 648 cases 0 differences; suites r20261110-213855-9565 2 passed.)

IMPORTS, exactly: Flock.TemplateQuery, Spot.Wires. Nothing of NciCheck (three JSON helpers are local, the same definitions as Baseline's getD, strOf?, strAt; naturals through Spot.Wires.nat?).

NAMES AND SIGNATURES (all in namespace NciCheck.RecordText):
- inductive Held | read (param : String) | table (name param : String) (root : Bool) | opened (name param pin : String)   -- moved out of NciCheck.Record, not copied
- inductive Declared | read (param : String) | table (name param : String) (root : Bool) | opened (name param : String)
- Held.declared : Held -> Declared
- structure Row: tid : String, unit : Nat, port : String, row : Nat
- wiresOf (doc : Json) : Json   -- PROPERTY's wiresOf, token for token: templates empty, cuts (none where it states none), wires
- wiring (doc : Json) : Except String Spot.Wires.Wiring := Spot.Wires.Wiring.ofJson (wiresOf doc)
- slot (P : Prog) (doc : Json) (tid : String) (u : Nat) : Option Nat
- declared (P : Prog) (doc : Json) (r : Row) : Option Declared
Under them, also there: Tmpl, tmplOf, templatesOf; Fill, Decl, Inputs, inputsOf (ts) (W) (doc); Grid, gridsOf, slotIn; fillAt, Inputs.declares, Inputs.declared.

IN NciCheck.Record NOW: Case.slot c tid u := slotIn c.S c.grids tid u. Case.heldAt (c : Case) (r : Row) : Option Held (it took tid port u row). Case.W is the text's wiring with SPOT's templates and order (wiringOf f text = { text with templates, order } of ofJson (piece f)). piece = wiresOf merged with the Program's templates and the order; I kept the name. Tmpl's leaf facts (output bits, leaf widths, closed rows) are a new Record.Leaves beside the templates, since they need passedOn, which is the check's and not the text's.

THE THREE TRAPS:
1. declared reads no order. inputsOf now reads the wiring's cuts and wires only (each wire's row words from the templates and the cut, Spot.Wires.rowWords's formula), not SPOT's summary. No record's declarations depend on the order, the text's or the instance's: Spot.Wires.check's summary is the wires and their row words whatever the order, and the order decides only whether check refuses (wires).
2. slot runs templatesOf then gridsOf itself and gives none where either refuses.
3. Imports as above, so the file can move into core_spot as it is.

THE TIES, BY UNFOLDING: a scratch file (not in the tree; lake env lean in the package, no sorry, axioms propext, Classical.choice, Quot.sound) proves from read j tr = .ok c: c.slot = slot c.f.P c.f.doc; c.heldAt r = some h -> declared c.f.P c.f.doc r = some h.declared; exists text, Wiring.ofJson (wiresOf c.f.doc) = .ok text and c.W.cuts = text.cuts and c.W.wires = text.wires; and decodeProgram c.f.i.program = .ok c.f.P, Flock.JsonParse.parse c.f.i.record = .ok c.f.doc. Its text is in note 20261110T2252Z-checkpoint-trn76-record-text-ties (synced), for lean to take or rewrite.

WHAT DID NOT CHANGE: every refusal, its order and code; the summary; the rate rule; the reference (one docstring). The data's bytes and the lock's pins are the same. Nineteen edits to the two Lean files are each caught by the 648 cases (the earlier twelve with the same counts, seven new on the module).

RETIRED TERMS: counted before the push. Three counts rise, each the package's own name where code must name it: Record.lean 14 to 16 (the import and the open), NciCheck.lean 43 to 45 (the import, and the module named once in the root's list), RecordText.lean 0 to 2 (namespace and end). In the body.

DATES: Thu noon holds (done). Fri 13 Nov 6 PM PST is not moved by this.

ONE SLIP: my 2:59 PM report comment on TRN-76 is 101 words by wc -w.

#8686: OPEN, not merged, at 3:11 PM PST.

