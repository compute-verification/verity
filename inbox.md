[2026-11-13T09:41Z] 4e8c2d7dd8a1
WALKER, Fri 13 Nov, 1:41 AM PST.

**Fix 3's design, before any code: the reader decides "the mask makes these rows differ" once per softmax node and once per select node, from the Refs' parts and the selects' condition structure, and never values a score word; anything off that form falls back to today's word-by-word valuing.**

First, the form lf's step actually has (RAN: r20261113-093600-657f and r20261113-093834-86a6, structure dumps of tiny(2,16,64)'s looped body; READ: work.py):
- `SoftmaxRef` batch S reads `F32Add` batch A with axes (0,0) of two operands. One is the scores (Refs parts into `LinearRef` product nodes). The other is the mask, a Concat whose every part points at an argument-less constant, `Const32[0]` or `Const32[0xff800000]` (-inf).
- `_safe_softmax`'s guard: E = `F32Eq`(A's words one to one, the -inf constant). Then a fold of K-1 `BitAnd` batch nodes: node k reads node k-1 (identity) and `Strided(E, base k, stride K, count rows)`. Then `SelectF32(fold[row], 0, S)`, and the next product (P·V) reads the select.
- Today, `row_of` values every score word and `unselected` values every E and A word, memoizing each one in `value_of`. That is the sink that grows as ctx² × heads × batch × layers. `masked_rows` also keeps K constants a row in `info`.

1. **The rule.** A softmax row counts as masked when its scores are one product row plus constant-node words, both read from A's operands' Refs parts. The row's constants are read off the mask's parts (bits and run lengths), and the product being a composite Definition makes every score word data. A select over S counts as unselected on row r when its condition is a `BitAnd` fold whose leaves are `F32Eq`(A's words, a constant c), covering exactly row r's K words once each. Its condition is then the constant "every mask word of row r, added to data, gives c", which is what `_generic` gives word by word. It is decided in `masked_rows` (once per softmax batch node) and in `unselected` (once per select batch node, cached by node). `softmax_rows` maps the product operand's runs through the select's Refs onto S's leaves instead of listing words. `info` keeps each row's constants as runs.
2. **Structure only.** The rule reads which node each part points at and that node's kind (argument-less constant, composite product, `F32Add`, `F32Eq`, `BitAnd`, `SelectF32`). It also reads the Refs fields that say which words are read (Affine identity; Strided base, stride and count), and checks the fold's bases are exactly {0..K-1}. It never accepts a count or shape that merely matches.
3. **Fallback.** Any part off that form (a mask word from a data node, a scaled score chain, a fold missing or repeating a word, a primitive product, an unknown op) makes that node fall back to today's `value` word by word: the same answer, slower. The rule never guesses. A `Reader` flag turns the rule off, and the off path stays as the test's reference.
4. **The test.** On a node, every `Work` equal with the rule on and off, for tiny(2,16,64), s(2,16,64) and s(2,64,512). In the suite, tiny with the rule on and off, equal (beside #9676's `tiny_work.json`, which must still match). Also two hand-built bodies in test_work's style, each off the form by one word: one mask word read from an input instead of a constant, and one fold node whose Strided base repeats a word. Each must read exactly as with the rule off. test_work and test_program_reading stay unchanged.
5. **Before and after:** time and peak for s(2,256,512) and tiny(2,1024,64), and s-t20 if it then reads. INFERRED: this removes the `value_of` growth and `info`'s K-a-row. It does not touch the Reader-level `_leaves` cache, which is fixes 1 and 2. So s-t20 may still need them.

The branch is cut from #9676's head 3a94714cfb, on its own `-eb6e` branch. I start the code on your go.

Other runs:
- RAN: d4d5 and 72e0 had already ended at their 5,400 s timeouts, unfinished, before your message, so there was nothing to stop.
- RAN: 54ec (r20261113-061108-54ec) is still running. Its last heartbeat is at 11,841 s: node 10,760 of 16,403 of the second summary, RSS 115.7 GiB, peak 137.6 GiB, against a stop at 144 GiB. I'll send its end, or its last heartbeat if it stops.
- No merge of `main` until the chains land.

