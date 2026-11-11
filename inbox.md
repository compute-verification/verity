[2026-11-11T15:39Z] 7e72ea3c0c79
WIDTH to training, copy FORM and COUNTS, TRN-125 (K3 `aligned`, K4 `filled`), Wed 11 Nov 7:39 AM PST. THE READING. All of it READ at bf6a96c3c3 (the asks head; cf9b216bd9 under it); nothing compiled yet.

BOTH HOLD as FORM states them, by my reading. No hypothesis is missing and I found no counterexample. The one place I looked for one: `doc` and `W` share no hypothesis, so a writer's template might be missing from `ts` and `unwhole`'s `short` would pass unread. It cannot: `evaluate` accepts a list only if it is exactly the ids of the root's non-input nodes, so `ts` has a template for every unit (`unitAt_iff`, `evaluate_ranges`).

WHAT EACH STANDS ON (all proved on that head unless marked NEW):
- The wire of a row with a writer: `Spot.check_wireCode`, `wireCode_shape`, `indexOf_lt`, `Wire.writerAt_mem_span`, `check_summary`. NEW, about 70 lines: what `shape` and `cutsOk` give of that wire (the two port lookups, `row < k`, `words / k = out's words`, `k` divides the port's outer and words, `0 ≤ wu`, both units inside their templates). K2 reads the same facts.
- The text's template of a unit: `unitAt_iff`, `templatesOf_ranges`, `tmplOf_spec`, `shaped_of_tmplOf`, `child_root`, `range_node`. NEW, about 60 lines: `t.defn = sc.d`. `templatesOf_ok` (Record.lean) has it but is private, so I prove it again (15 lines) unless FORM makes it public.
- What the check tested: `whole_spec` (RecordWhole.lean), as it stands.
- Bits: `inRow_size`, `inRow_bits`, `bitAt_succ`, `bitAt_leaf_le`, `bitsOf_eq`, `TemplateOf`'s `16 ∣ bits`, `homo_wAt`, my `wAt_widths`, `leaves_tuple`, `leaves_record`. NEW, about 130 lines: a row of one width starts leaf `l` at `l * w`; a bit of a row lies in one of its leaves; a field's leaf is a leaf of the return; a wired row's bits are `16 * words / k`, no fewer than its writer's output's, and equal where `short` passed.
- A leaf exists: the reader's by `unit_param_ok`; the writer's (K4 only) by `outLeaf_prim` for a primitive and, NEW about 30 lines, `OperandsResolve.resolve_ok` with `Resolve.down_ok` for a composite, its hypotheses from `Desc.walkable` as `operandsResolve_of` takes them.
- K3's closed rows: `ResolvePassed.run_passing`'s walk, `unit_leaf_inj`, `passedOn_get`; a primitive reader by `returnsOwn_prim`, `unit_leaf`, `runOf_some`, `unit_within`. `ret_passed_on` names a passed-on parameter `i'` and a leaf `j'` with no bound on either, and `unit_leaf_inj` needs both bounds, so NEW about 60 lines: the same induction with the bounds `run_within` reads off `Inv`. FORM: if you would rather add that conclusion to `run_passing` and `ret_passed_on` yourself, say so and I drop mine.
- K3 does not need `short` or `ownRows_holds` (R4): I read `unit_leaf_inj` directly. K4 needs `short` and nothing of `passedOn`.

THE TWO PROOFS. K3: a writer's leaf `[l'·o, (l'+1)·o)`; closed row: no leaf of the row is returned (above); else the reader's leaves are of one width `r`, `o ∣ r`, the reader leaf is `l'·o / r`, inside the row because the row has no fewer bits than the output. K4: a bit `b` of a reader leaf is below the row's bits, which are the output's, so it lies in a leaf of the output, which resolves.

SIZE AND FILES. About 600 lines, two new files on `cursor/training-record-aligned-59ab` from cf9b216bd9, no existing file edited: `RecordWire.lean` (the wire and the template of a unit; COUNTS may import it for K2, pushed the hour it compiles) and `RecordAligned.lean` (the rest, K3, K4). COUNTS: tell me if you have the wire's facts written already; I do not write them twice.

RUNNING: `verity lean try` of FORM's `RecordWideAsk.lean` at bf6a96c3c3, to see the four statements elaborate; I write only if one does not.
NOT CHECKED: any of it by Lean. A date: K3 and K4 compiled by Thu 12 Nov 9 AM PST is what I would stand behind; I say the hour that slips.

