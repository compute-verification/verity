[2026-11-10T22:03Z] 1a406a2d2cc2
MESSAGE FROM YOUR COORDINATOR (training, bc-b43da88b-5d25-4463-9b8c-da4df6f84207). Your 1:40 PM PST comments are read: the closed-row cases and items 4 and 5 are what was asked, and early. Lean ruled again on LEAN-238 (its comment of 21:57Z, section 2 and "Asks"): read it. It replaces item 4 of my 21:16Z message. MY RULING: #8676 CARRIES ONE MORE THING, a module of its own for what lean's property names.

WHY. Lean's property (`Security/Properties/Nci/Record.lean` on `cursor/lean-served-record-property-fc8d`, at 331e6dbc98c3; read its lines 93 to 160 and 296 to 350) must never import `NciCheck.Record`. Today it keeps its own copies of `Held` and of the text's wiring (`wiresOf`), and states the slots and held rows over readers that do not exist (`Readers`, `RowsOfText`). When your module exists, its copies go and it names yours. So lean no longer reads `NciCheck.Record.piece` by name: you may rename or rebuild `piece`.

WHAT THE MODULE HOLDS (one new file in `nci_check`, which `NciCheck/Record.lean` imports and builds on; you name it):
1. `Held`: the one inductive (`read`, `table`, `opened`), moved, not copied.
2. The text's wiring: what `Spot.Wires.Wiring.ofJson` is given of the parsed record alone, its `cuts` (none where it states none) and its `wires`, with no template. PROPERTY's `wiresOf` is that function. `piece` is then this plus the Program's templates and the order.
3. `slot` and `declared` as functions of the decoded Program and the parsed text, with PROPERTY's `Readers` signatures unless you have a reason to differ: `slot : Prog -> Json -> String -> Nat -> Option Nat` (template id, unit) and `declared : Prog -> Json -> Row -> Option Declared`, where `Declared` is `Held` without an opening's pin. Define `Row` and `Declared` there if the signatures need them. Neither reads the trusted input, the pins or any other field of the instance: whether a declared row is θ, and the pin, stay in `Record.lean`.
4. `Case.slot` and `Case.heldAt` are BUILT ON those two, so that for every record `read` accepts, `Case.slot c = slot P doc` and `heldAt c r = some h` gives `declared P doc r = some h.declared` (PROPERTY's `RowsOfText`). Aim for that to hold by unfolding, not by a second computation that happens to agree.

THREE TRAPS I SAW READING fcd65c566829 (check them, I ran nothing):
- `piece` takes `order` from the instance where the record states none, and `inputsOf` takes the wiring and SPOT's summary. `declared` has neither the instance nor a wiring built with its order. Say on TRN-76 what `declared` reads where the text states no order, and whether any accepted record's declarations depend on the order.
- `slot` today is `Case.slot` over `c.grids` and `c.S`; as a function of the Program and the text it runs `templatesOf` and `gridsOf` itself and gives none where either refuses.
- `nci_check` requires `core_spot`, and `core_spot` does not require `nci_check` (the two lakefiles). Proofs is asked whether SPOT's audit will read `slot` and `declared`; if it says yes, the file has to move into `core_spot`. So keep the module's imports to `Flock` and `Spot.Wires` if you can, with nothing of `NciCheck`, and say on TRN-76 exactly what it imports.

WHAT DOES NOT CHANGE: every refusal, its order and its code; the summary and the rate rule; the Python reference's behaviour. The difftest's 648 cases still compare with 0 differences, and the Lean check passes with its kernel replay.

REPORT: one comment on TRN-76 (and the same by inbox to me) when the module is through Lean: the module's name, the full names and signatures of `Held`, the text's wiring, `slot`, `declared` (and `Row`, `Declared` if yours), its imports, the head, and the run ids. I hand it to lean; I have told lean it is there by Thu 12 Nov noon PST. If you cannot hold that, or the module would move your mark past Fri 13 Nov 6 PM PST, tell me by inbox before you start, with what it costs. Finish the runs you have queued at fcd65c5668 first only if they are cheap to keep; they need not gate this.

Run the retired-terms count before you push.

