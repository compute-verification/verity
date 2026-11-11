[2026-11-11T16:24Z] 4f6367970acf
REPLOT to training, TRN-111, Wed 11 Nov 8:24 AM PST. THE DRAFT PULL REQUEST IS OPEN FOR YOU TO READ: https://github.com/compute-verification/verity/pull/9010 (cursor/training-tokens-correction-55a1 at 4eb020b29c onto cursor/training-replot-published-55a1 at 26d6c24708; draft, not marked, not to main, no table in the tree; linked to TRN-111). Nothing is published.

ITS BODY, in your order: the cells that fall first (the standard 20M row first, "as built today it is high by 0.11%", beside 99bf@2301), then one table a layer (page, search, overlay, frontier-fit panel) of rows and ranges, the page's 24 falling rows with both values, the rises, the counts (compared, moved, not moved, refused, open), the fronts and the coverage cell, the goals' figures, the 1.9e7 at 1B in one line, which records are restated and which keep their count with the rule, the three art ids with the sentence on the records of record, the four records apart, "what is not corrected here" (grade's five page rows named and the 63 search rows; the six rows that print 1; float32 attention; the three stored inputs), what was not checked and not changed, what was run and read. #8854 need not land first.

FOUR THINGS YOU SHOULD KNOW BEFORE YOU READ IT:
1. A count of mine is corrected by one row. My 4:04 AM said of the page's 116 rows "27 print no rho; 89 compared". From the tables: 28 print none (27 refused, 1 unmeasured) and 88 are compared: 54 move, 34 do not. The body has 28 and 88. No figure changes.
2. Two of grade's five page rows change what they print on Thursday: eb14@101742 and 78f9@59602 print 18,870 today and 23,700 and 23,710 in the corrected tables (they are among the rises). The body names both values. Whether grade's 0.26% holds on the corrected figure is grade's to say; I did not check it.
3. The six search rows that print 1 (8833@177, a207@168): their slowdowns print 1 in every build, but they are pilot rows, so their start's compute and gain cells move with the other 774 (-0.125%). My 8:06 AM "no table of mine moves them" was about the slowdown only.
4. The page's note as served is 497 characters against the builder's cap of 500; it is the cut's note shortened to carry the tables' art id. Neither paragraph below fits in it: they are page text, yours to place.

DRAFT FOR THE PAGE, the correction's note (falls before rises, the high rows first and as their own sentence):
"Corrected: a step's tokens. The search's records stated one forward's tokens for a training step that runs several forwards, and these tables read that count. On 24 of this page's 197 rows (19 runs) the published slowdown was too high, by 0.02% to 0.23%; in the search table on 73 of 1,497 rows (25 runs), by up to 0.31%. On 37 rows (27 runs) it was too low, by 0.03% to 72% (search table: 168 rows of 56 runs, by up to 104.5%). On 7 of these rows (15 in the search table) the slowdown at the served price fell while the default one rose. No row was added or dropped, and p and the term that binds changed on none. Known and not corrected by these tables, which the later rebuild from retraced records corrects: no row that prints a slowdown above 1 moves by the trace's float32 attention, since the term that binds on those 1,134 rows reads neither share; six search rows of two records (8833@177, wildcard/onestage; a207@168, wildcard/expo; three gain targets each) print 1 and read low, and would print about 1.2 or more, binding the token term, with no page row changing; and on the search table the printed sigma and the token, decoding and reduction cells move (sigma falls by up to 14%, the affected share moves by 0.87 to 2.12 times)."

DRAFT FOR THE PAGE, next to the stored inputs:
"The stored inputs published with this table mislead on three counts, and no printed slowdown, share or binding term reads them: a stream coordinate is priced at 16 bits where the traced values are wider (1,116 search rows, 147 of the page's 166); the update's size on the three quantization rows is 8 times loose; and 906 rows whose step holds state have no gradient term."

INFRA'S ANNOUNCEMENT (merge main into each unmarked branch): I have NOT merged main into my three branches and ask you to confirm that is right. The pull request's two branches sit at the published builder's commit on purpose (a merge renames te_core's tree and the one command no longer finds its traces), and cursor/training-replot-55a1 stays as the source of the published table by your TRN-98 ruling.

NEXT, without waiting: the hand-off (where each of the three stored inputs is produced, with file and function; what is not checked), then the closing comment on TRN-111. 106f waits on your word of grade's answer.

[2026-11-11T16:24Z] 2f85fff02365
WIRED to training, FORM and COUNTS (training: please pass to WIDTH, whose id I do not hold), TRN-116, Wed 11 Nov 8:25 AM PST. C2 `slot_of_unit` AND C1 `names_divides` COMPILE ON A NODE AND ARE PUSHED. THIS IS ALSO THE READING OF BOTH: both statements elaborate and are true as FORM wrote them.

RAN: `research run … lean_changed.py --try security/Proofs/Properties/Nci/RecordCircuit.lean --siblings`, run r20261111-161603-41d7 on vy-nebius-cpu-1, at commit 65745e031e of `cursor/training-record-rows-05f2` (pushed; origin head 65745e031e81), first try. Its lines: "lean exit 0 in 35 s"; "axiom probe: none of its 2 theorems uses an axiom beyond propext, Classical.choice and Quot.sound"; siblings clean (68 imported, 3820 unchecked). No warning or error names the file.

THE STATEMENTS ARE FORM's, word for word. RAN a comparison of each theorem's text, from `theorem` to `:= by`, against `RecordCircuitAsk.lean` at 5d797e2d28: SAME for both. No hypothesis added, none dropped.

One new file, `security/Proofs/Properties/Nci/RecordCircuit.lean`, no existing file edited. It imports `Proofs.Properties.Nci.Record` beside the three the ask imports, for `unitAt_iff` and `templatesOf_ranges`.

How each goes, so the callers know what they stand on:
- C2: `read_front`, `finish_text` (the grids are `gridsOf` of the front's templates), `front_text`, `unitAt_iff` (the unit is below the units of the text's template of that id, whatever templates `rs` was asked for), `templatesOf_ranges` (the text's template ids are distinct), `gridsOf_covers` (a grid holds the unit), `gridsOf_ok` (`1 ≤ S`). The slot is `(u - g.start) / g.q % c.S` of the first grid that holds `u`.
- C1: θ is the filter of `D`'s gates that are leaves of a held row; each is an input gate by `OwnRows`' first clause and `LaysOut`'s kinds. The division gives an input gate `none` and a computed gate the slot of a unit `GatesOwn` puts it in, picked by `Classical.axiomOfChoice`. It does not use that units are apart, nor `LaysOut`'s injectivity, nor `OwnRows`' second clause.

Retired-terms sites in the file, RAN `test_retired_terms.offenders`: 0.

A commit on top, dc0d6d943c, changes two lines of the header docstring only; its node try (r-id in my next message) is running and I push it when it passes.

COUNTS and WIDTH: P1 can call `slot_of_unit` from 65745e031e now.

Names compiled, all of mine: R1 to R5, I1 to I3, M1, M2, C1, C2. Names left: none.

