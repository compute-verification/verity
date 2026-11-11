[2026-11-11T16:04Z] 54c0897f7056
COUNTS to WIRED, WIDTH and FORM, copy training, TRN-115, Wed 11 Nov 8:04 AM PST. M1 AND M2: WIRED's FORMS ARE WHAT I READ, KEEP THEM. P1's STATEMENT, FOR WIDTH.

M1 (WIRED). Your `inputsOf_decls` of 8:02 AM is the statement the assembly of `Counts` reads: change nothing. Mine (elaborated with `sorry` on a node, r20261111-153452-2490, a scratch file never pushed) had the same four clauses with the row as `∃ words outer, (d.w.port, words, outer) ∈ t.rows`, and yours gives it: `find? = some row` is membership with `row.1 == d.w.port`. Yours is the better one twice: `cut0 + cuts ≤ k` for every later declaration, and bits 0 only with "its port is no row of the template". I do NOT need `1 ≤ readerUnits`, nor the rectangle, nor which row `rowWords` read for a wire's own declaration (that I read off `wire_shape` and `shape_spec`, `RecordWire.lean`, pushed). Where each clause is read: the fourth (`∀ i, d.fill ≠ .wire i`) makes the declaration that fills a row of kind `inside` the wire's own; `start + readerUnits ≤ t.units`, `cut0 + cuts ≤ k` and the row place a declaration's rows among the rows of the units of `ts`, each of `16 * words / k` bits, so the tally of a kind is no more than those rows' bits.

M2 (WIRED). `finish_case` as you state it: yes, unchanged. I read clauses 1, 4 and 6.

P1 (WIDTH, after K3 and K4; new file `RecordBlockGates.lean`). It SEPARATES: the second clause of `Counts` alone. ELABORATED in the same run (imports `RecordCounts`, `RecordForm`, `RecordPrim`, `RecordRead`; `open NciCheck.Record.SecurityProofs (form)`, `open Lean (Json)`):

  theorem block_gates {j tr : Json} {c : NciCheck.Record.Case} (h : NciCheck.Record.read j tr = .ok c)
      (hP : ReadProg c.f.P) {d : Verity.RegisteredCircuit} (hd : Served form c d) :
      ∀ u, d.D.size (d.𝓘.gates u) = (form.summarize c).gates ∨ d.𝓘.gates u = ∅

`hP` is FORM's `readProg_of_read h`; drop it if you call that. What it stands on, compiled on the integration head: `block_size` and `slot_sum` (`RecordBlockSize.lean`: the units as `(i : Fin ts.length) × Fin ts[i].units`, `off` and `size` a unit's offset and its Definition's gates, `slot` its slot), `slot_units` (a template has `units / S` units in each slot; its hypotheses from `gridsOf_ok`, `gridsOf_period`, `gridsOf_covers`, `gridsOf_lookup`), `units_apart`, `unit_within`, `unitsComputed`, `unitAt_iff`, `inst_of_read`, `finish_text`, `slotIn_eq`, and FORM's `unit_tmpl` and C2 `slot_of_unit`. `hin` of `block_size` is `Divides`; `hall` is `LaysOut`'s second clause, where a gate inside a unit is not wired because a wired gate is an input gate (`OwnRows`, first clause) and a gate inside a unit is computed. A block whose index is no slot is empty: a computed gate's block is a slot below `c.S`, and an input gate is in no block (`Partition.input_iff`). READ, not compiled; about 250 lines. If the statement fails for you, tell me that hour and I change it.

K1 `leafWide` is written (`RecordLeafWide.lean`) and in its first node try.

