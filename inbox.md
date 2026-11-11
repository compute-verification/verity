[2026-11-11T01:50Z] 790cd0d40e10
TIE, job 2: run `614b`'s uncredited share, priced with `rho.min_cut` at #8703's head.

Marks. **[run]**: printed by `/agent/repos/verity/.venv/bin/python /tmp/trn80/price_614b.py ~/wt/tier` on my VM at
`9310a8a736eb` (CPU, under a second; script and output whole in `note:training/20261111T0152Z-checkpoint-trn80-tie-price-614b`).
**[read]**: TIE2's figure (CMP-380 and its note), not rerun. **[derived]**: my arithmetic on read figures.

**First: #8703's datacenter and world rows are wrong by the body's own definition, found while pricing.** They take
671,369 untied words a node, which is ln(100)/459 of the ticket weight of ONE execution phase of the run of record, 4
blocks [read, your appendix], beside block counts at the census's peak rate, 1.31e8 blocks a node's phase [derived].
Scaled alike, a node's bound is 2.2e13 words and the period holds 166.5 untied words a block over a week, 38.85 over a
month and 3.19 over a year [derived], not 5.1e-6. The function then gives column C of (iii): 1,614 for a datacenter's
week, where the body says 37,653. The head is untouched. I am adding a correction to #8703's body now (head unmoved);
the tree's `PERIOD` rows and the README want a commit of their own at the crossing, when the head moves anyway. Say
if you want either differently.

## (i) What the command prints at this record's share

`s` is the uncredited MACs over the work in the policy's unit (the linears plus twice attention and head):
335,544,320 / (4 × 36,241,670,144). The record gives the inferred share to every digit: 83,886,080 MACs a block.

| over | uncredited y words | uncredited work `s` | cut number, every word tied | limit `1/(s + 1/ρ)` | the uncredited words counted as untied words of their blocks: cut number | and their MACs charged |
|---|---|---|---|---|---|---|
| the run | 0.1825% [read] | 0.2315% (0.2463% of the linears' [read]) [derived] | 37,653.4 [run] | **427.1** [run] | 1,598.5 at `k` = 916,128 [run] | **340.1** [run] |
| round 2, the worst, at 3 rows | 0.4076% [read] | 0.8373% [derived] | 37,653.4 | **119.1** [run] | 475.4 [run] | 95.5 [run] |
| round 2 at 7 rows | 0.9511% [read] | 1.9536% [derived] | 37,653.4 | **51.1** [run] | 116.8 [run] | 35.6 [run] |
| every excluded row in one block | 0.7298% [derived] | 0.9259% [derived] | 37,653.4 | 107.7 [run] | 183.1 [run] | 67.9 [run] |

- The command prints columns four and five. The last two are what `bound_charged`'s reading asks of an uncredited real
  cell (an untied word: 32 fresh bits in its block, the MACs that feed it charged whole); the tied print leaves the
  words out of `k`.
- Round 2's range: 3 to 7 of the 11 excluded `down` rows are round 2's [derived from TIE2's tiles]; its work is
  4 × 1,502,871,552 MACs. The rows read the round's share as every block's.
- Last row: the excluded rows' slots are not in the store [read], so one block may hold them all.

## (ii) What 1,000 needs

**Route A, the forming excludes less.**

- Limit alone: `s` ≤ 0.0973%, 35,279,163 MACs a block, 141,116,652 a run of 4 blocks. Of the record's 335,544,320,
  194,427,668 must go: **57.9%** [run].
- With the words in `k` too (the record scaled evenly): **63.0%** must go; 124,023,168 MACs and 35,578 words a run
  may stay [run].
- In rows. The record: 4 `o` rows (4,194,304 MACs, 2,048 words each), 4 `gate_up` (33,554,432; 16,384), 11 `down`
  (16,777,216; 2,048) [read]. The largest sets that may stay, words in `k` [run]: 8 `down`; 4 `o` and 6 `down`;
  3 `o` and 7 `down`; 2 `gate_up` and 3 `down`; 4 `o`, 1 `gate_up`, 4 `down`; 4 `o`, 2 `gate_up`, 2 `down`;
  2 `o`, 1 `gate_up`, 5 `down`; 2 `o` and 3 `gate_up`. So at least one `gate_up` row goes whatever else does, and with
  every `gate_up` row gone 3 to 5 `down` rows go too.
- In bands. The 19 rows sit in 11 bands [read]. A band uncredited whole, as a voluntary row's is, is over a run's
  whole allowance by itself: `o`'s 64 rows are 268,435,456 MACs, 1.9 times it (limit 532.4); `qkv`'s 2.9 times
  (356.6); `down`'s 7.6 times (134.5); `gate_up`'s 15.2 times (67.4) [run]. So 1,000 leaves room for no voluntary row
  at this class's run.

**Route B, the second sample by y words, taken per block** (δ = 1/100; `t` tickets a block bound its wrong y words by
1 − δ^(1/t) of its 13,189,120; the wrong words go in `k`, and their MACs are at most 8,192 a word, `down`'s).

| `t` a block | wrong words a block, at most | cut number with them in `k` | their MACs, of a block's work | the number | tickets a run against today's 459 |
|---|---|---|---|---|---|
| 459 | 131,665 | 107.5 | 2.976% | 25.6 | 4 times |
| 1,840 | 32,969 | 1,011.4 | 0.745% | 118.5 | 16 times |
| 4,639 | 13,086 | 4,466.2 | 0.296% | 314.3 | 40 times |
| **14,789** | 4,106 | 13,928.5 | 0.0928% | **1,000.0** | 129 times |

All [run] but the last column [derived]. By the limit alone it is 14,102. TIE2's 1,840 a block gives 1,000 by the
words alone and 118.5 with their work. Were the wrong words' work at the run's mean (2,582 MACs a word), 5,339 a
block would do (46 times), but a sample weighed by y words does not give that.

Neither route reaches a dropped round or a version-1 W [read]: `614b` is outside the statements until rerun.

## (iii) Week, month, year; a datacenter and the world

Each cell: the cut number with the best threshold, one β_x, then in brackets the limit beside it at `s` = 0.2315%.
A datacenter and the world print the same figure in every cell (their ratio of untied words to blocks is the same)
[run].

| period | A. #8703's rows as they stand (671,369 a node) | B. A with the record's uncredited words in `k`, their MACs charged | C. the wrong-tile words scaled with the blocks | D. C with the record's uncredited words too |
|---|---|---|---|---|
| week | 37,653.0 (427.1) | 1,598.5 → 340.1 | 1,614.1 (340.8) | **253.0** |
| month | 37,653.3 (427.1) | 1,598.5 → 340.1 | 4,787.8 (396.3) | **313.5** |
| year | 37,653.4 (427.1) | 1,598.5 → 340.1 | 20,422.0 (423.1) | **337.7** |

- The other two readings under C: one β_x at the top budget, no block charged, 52.8, 202.2, 1,336.6 (A: 13,529.4);
  a β_x a block, 53.2, 209.6, 1,844.3 (A: 37,653.4). Under D: 32.8, 63.5, 87.3 and 42.2, 118.5, 284.4 [run].
- One node at the run's own rate, 4 blocks a phase, which is a premise row and not scaled: 4,032 blocks a week
  5,088.7 (398.2), with the record's words 254.0; 17,280 a month 9,482.5 (413.2), 313.9; 210,240 a year 22,391.1
  (423.9), 337.7 [run].
- Pooled, at `s` = 0.2315% of the compute covered [run, blocks × `c` × `s`]: a datacenter's uncredited work is 5.28e25
  AND gates a week of 2.281e28, 2.26e26 a month of 9.779e28, 2.75e27 a year of 1.19e30; the world's 1.98e29 of
  8.555e31, 8.49e29 of 3.666e32, 1.03e31 of 4.46e33. The share no figure holds is 1 over the number: 0.234% at
  427.1; 0.395%, 0.319%, 0.296% at D's; 0.1% at 1,000.
- **Proved:** nothing in this table. `HECharged` is proved only from `CountsOn`, which waits on #8700.
- **Premise:** (H) and (E) at the circuit with the untied words as inputs; the bound on a period's wrong-tile words
  (CMP-380's, no Lean statement); 32 bits an untied word; `mac_gates = size / N`.
- **Scaled:** every row. The block counts are every machine at its data-sheet peak on the class's block; a node's
  bound in C and D is one run's, scaled by the node's blocks a phase; the record's share is one run's of four
  rounds, taken as every block's; one block's most untied words under C and D is its whole 13,189,120.
- Not in any cell: dropped rounds, voluntary rows and a version-1 W, which nothing bounds [read]. With them the
  number is 1.

[2026-11-11T01:51Z] 2048f3441c7f
From grade (TRN-94), 5:56 PM PST. Your first question, answered by reading (no run): NO published plot row's slowdown or inefficiency reads the carried state or the gradient's cut. I go on to the fix.

What I read: the uncut overlay the page is cut from (art:ae1ceadb…, 458 rows, te_core 4.13.5) and te_core 4.13.5's formalism at the replot branch's 0fec9694ff.
- p = C/H reads no trace.
- 289 rows have a rho; 165 are refused and 4 unmeasured, with none. All 289 bind on fresh_products. None rests on Fit.
- rho's max is over fresh_products, token and decoding. None of the three reads the gradient's cut, the partial cut or the state. The token kind's cuts run from the carried values to the update values a token's ops read, and the optimizer is on no token's path.
- The cut and the state are read in two places only: Fit's two checks, and the reduction and update sensitivities.
- Fit cannot hold on any of the 289 at any value of them: its compute check needs 8|E| + s tau <= k, and the least 8|E| is 3,865,250 bits against k of 65,536 or 61,440.

What each of the 289 does publish that reads the gradient's cut: its two sensitivities, in the row's kinds cell, beside rho and never in it. 46 rows also carry a delta_grad_bits cell (cut plus state). Those are what can move.

Not yet settled, and not about the state or the cut: whether the rule changes a product row itself (an optimizer product's flags or widths), which would move fresh_products. By reading I expect not; the fix and a re-read decide it, and I tell you before anything else if one moves.

Also: the same line is in te_core 4.13.5 (structure.py line 509 at 0fec9694ff), on main, and in the move's tree at 178f49c90c89 (line 505). I cut the branch from the move's tree.

[2026-11-11T01:53Z] 50af28a76490
TIE, two lines after job 2's table (Tue 10 Nov 05:53 PM PST). (1) #8703's body now carries the correction; the head is unmoved at 9310a8a736eb and the body reads back identical. It is a first bullet under "Read this first" with the six datacenter and world rows at the consistent scale and their pooled figures (run at the head, pinned by no test), and one line on each of the two tables and on the pooled bullet saying their datacenter and world rows are wrong. The tree's PERIOD rows (the test only; the README states none) are not corrected at this head: a commit of its own at the crossing, if you agree. The body before this edit is whole in note training/20261111T0126Z-checkpoint-trn80-tie-h-circuit. (2) A wrong word in my table's pooled line: those totals are in the gates the policy's c counts, every two-input gate, not AND gates.

[2026-11-11T01:54Z] 13f02ea91e38
TRN-93 PAIR: done (Tue 10 Nov 5:54 PM PST). Leave #5173 closed. Nothing needed from you. Note: training-20261111T0150Z-report-trn93-pair-bound (lanes/training/20261111T0150Z-report-trn93-pair-bound.md). Evidence: art:c0b741ae47f270e5e9ae4e8e0aee62beb879460554a6e96471a76d91463d0d6a. No Lean, no PR, no branch, no tree change, CPU only.

PART 1. HeldRate does not stand as written; the bound's number is not contradicted.
- Statement (read at ca17e8df8fe; the Lean says what the PR body says): for every block, βw x (weights its counted products read) + βe x (entries they descend from) - ℓ <= the description length of what the block reads. The 103 uses βw = 2.5 bits a held weight, βe = 0.87 x 7,168 = 6,236 bits a token's entry.
- Break (counted, no compression assumed): a block holds one column of wq_b (65,536 weights a step) and reads one bf16 coordinate a token, 16 bits. Filled to k_S = 3.193e11 bits it works on 2.0e10 tokens; the premise allows 5.12e7. 390 times over; an ℓ that saves it is 1.2e14 bits and empties the bound. Same for the head's column and an expert gate's. Any raw entry narrower than 390 of 7,168 coordinates undercuts the price.
- Conditional break: enter at a layer's attention latents (2,048 values a token), hold wq_b, wo_a, wo_b: 3.49 times over at 0.87 bits a value, within the premise at raw FP4. Breaks if latents code under 3.04 bits a value; unmeasured.
- The conclusion holds against every block I built. Pair limit 1.635e18 forward products a serving block. Column tile 1.31e15; coded column tile 2.41e16 (E1's fan-out limit); attention latents 5.09e16; held layers 1, 2, 3: 3.15e16, 4.71e16, 4.68e16; last 2 layers and the head 7.45e16, the most, 21.9 times under. The limit is loose because a token uses 7 of 385 experts and the pair count credits it with the whole layer. That block is 3.1 times E1's fan-out limit, so main's 886 (fanout-priced) is not a bound at the serving block either.
- Strategies that fail: held layers (pays every weight and a full-width entry; premise over bits = 1.00). Replay from data (the state's update is 3.93e12 bits, 12.3 times k_S; only the first 48 steps after a registration, 1.4e-4 of the run's 3.5e5). The cheap prefix (already carved out, expensive_share 0.918). The red team's 20 families behind (H)'s measured cost (each pays raw width for every weight and entry value; they sink (H) through values recomputed behind the cut, which HeldRate never charges).
- Relation to (H): independent as stated. (H) implies at no cost the per-value form (β_w x weights + β_x x entry values <= k), which gives no bound at the serving block (limit 1.17e22, above the 2.04e20 the block's gates hold). The per-token form, the factor 7,168 that makes the 103, is not implied, and is the part that is false. (H) has no measured β past 8e8 bits.

PART 2. Today: slowdown at least 102.8, a week, a month and a year, one datacenter and the world. Ran at main 2b869ed35d (rho is byte-identical at origin/main a930252c49):
  cd /agent/repos/verity && uv run python pair_scratch.py --tornado --json pair-main.json
over | period | slowdown >= | hidden training's share of the period's gates | reference runs that fit
one datacenter | week | 102.8 | 0.389% | 0.04806
one datacenter | month | 102.8 | 0.389% | 0.206
one datacenter | year | 102.8 | 0.389% | 2.506
the world | week | 102.8 | 0.389% | 180.2
the world | month | 102.8 | 0.389% | 772.3
the world | year | 102.8 | 0.389% | 9,396
- Serving block alone 114.5; 102.8 = 1/(0.001 + 0.999/114.46). Range over each parameter: 3.04 to 710; corner 2.985.
- Inputs since 27 Oct: none the class reads. #5173's own tree prints 102.79951506627111, and so does main (ran both, registries diffed). eps_dark became eps_unreg (same value); the session k's range ends at 9.1e7, was 8.8e7 (low end of the range 5.29 to 4.96, figure unmoved); the rest is renames and rows the class never read.
- main as is: 886.2. With the fan-out classes withdrawn once a layer's update fits in a block: 3.04. Adding pair without that withdrawal leaves 886.2, so the class moves the figure only together with it.
- Proved: only in #5173's Lean at its head, none on main (the count of pairs a block, given HeldRate and EvaluatesBlocks). Premise: HeldRate (false as stated), EvaluatesBlocks, no product straddling blocks, the reference's efficiency, the cheap-prefix carve-out, the κ/κ_fwd conversion (rho's, in no Lean), ε_unreg. Scaled: one serving block at the data sheet's rate to 12,500 and 4.687e7 servers and to each period.

PART 3. Landing is a rewrite.
- 923bdcc7e4 (6 Nov) deleted all five Nci.Ldl files (655 lines at #5173's head; #5173 added 295). Products, ActivationRate, FanOutBound, HeldRate, PairBound have nothing on main. Decoder/LDL, Registration, CKPartition, EvaluatesBlocks, BlockDescription, Counting map to core's Verity.Decoder/LDL, Committed, Partition.IsCK, Evaluates, Guarantees.BlockDescription/Counting (Lemma 1 and Theorem 3 are proved in core).
- None of #8490, #8502, #8504, #8593, #8595 removes anything a restated bound reads.
- About 300 lines: statements 60, proof 70, witnesses 170 (a prefix-free decoder; #5173's was not).
- Lock: one new premise, a 12th named assumption (Daniel's to rule) or open with an issue and date, plus the open Verity.Evaluates, Committed.Honest, Partition.IsCK and a domain hypothesis (each pair counted once).
- With no new Lean: cite Nci.Formalism's Slowdown and TokenCapClean under training-cut-cost and training-token-work at the pair limit; then rho only (one class, premise rows, the withdrawal, the attack, tests; #5173's rho side was +305 -105).
- Period: composes as onsite's bound of record does, through rho's serving-block sum; no Lean statement composes any onsite training bound over a monitoring period.

RECOMMENDATION. Leave closed. The one fact: the 103 comes from a limit 22 times above the best block known, not from a true premise. About 103 could go to Daniel as a further onsite option only restated: main's token-work bound at the pair limit, an entry priced by its values plus a rate for narrow entries (about 1.6e5 forward products a bit by my list keeps 102.8). Premise, unmeasured at 3.19e11 bits, both prices 290 times past their credible 1.1e9.

NOT CHECKED. No GPU or attack run (paused): no measured code rate for entries or latents, no β at 3.19e11 bits. No traced step of the reference (shapes from models.toml). Whether κ/κ_fwd is sound for backward products. No Lean built (the TokenCapClean reading is arithmetic on the definitions). The narrow-entry rate is from my list, not a search. Nothing re-run after tonight's move; the scripts put benchmarks/rho on sys.path and need the new path after it.

