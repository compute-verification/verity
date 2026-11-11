[2026-11-11T02:24Z] 26b5ca8dd7ca
TIE, final message of this turn, Tue 10 Nov 06:24 PM PST. The three jobs are done. Neither PR branch was touched (#8591 at 3bd2ed9a166a, #8703 at 9310a8a736eb, both still marked); no Lean build, no GPU run. The move was not on main when I last fetched (6:14 PM PST, a930252c4993).

# JOB 1

Does anything of the design need (H) at `d` itself? No. `HECharged` takes (H), (E), the reads, T, S and the work at one registered circuit, any; for a period with untied words that is the circuit with those words as inputs, where a block reads k + 32 w. What is taken at `d` is only that each block reads at most k, and its gates c. (H) at `d` is read only where no word is untied (`HE`, `bound`, `DampenerRemotePeriod`); it would imply (H) at the untied circuit and is false of the work an untied word is for. [read, lean's #8700 at d5bc5d162559; not built by me]

Edits made (head unmoved): #8703's body says (H) and (E) are asked at the circuit with the untied words as inputs, not at `d`, in the "Read this first" bullet and in the premise list ("budget" for "cap" there); `bound_on_caps_he` is `bound_on_budgets_he`; the hand-off note 20261110T2048Z has a dated correction. Note: training/20261111T0126Z-checkpoint-trn80-tie-h-circuit.

# JOB 2

TIE, job 2: run `614b`'s uncredited share, priced with `rho.min_cut` at #8703's head.

Marks. **[run]**: printed by `/agent/repos/verity/.venv/bin/python /tmp/trn80/price_614b.py ~/wt/tier` on my VM at
`9310a8a736eb` (CPU, under a second; script and output whole in `note:training/20261111T0152Z-checkpoint-trn80-tie-price-614b`).
**[read]**: TIE2's figure (CMP-380 and its note), not rerun. **[derived]**: my arithmetic on read figures.

**First: #8703's datacenter and world rows are wrong by the body's own definition, found while pricing.** They take
671,369 untied words a node, which is ln(100)/459 of the ticket weight of ONE execution phase of the run of record, 4
blocks [read, your appendix], beside block counts at the census's peak rate, 1.31e8 blocks a node's phase [derived].
Scaled alike, a node's bound is 2.2e13 words and the period holds 166.5 untied words a block over a week, 38.85 over a
month and 3.19 over a year [derived], not 5.1e-6. The function then gives column C of (iii): 1,614 for a datacenter's
week, where the body says 37,653. The head is untouched. #8703's body carries the correction (head unmoved), and the
tree's rows have their commit ready for the crossing (job 3, below).

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
  gates (the gates the policy's `c` counts, every two-input gate) a week of 2.281e28, 2.26e26 a month of 9.779e28, 2.75e27 a year of 1.19e30; the world's 1.98e29 of
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

# JOB 3

Note: training/20261111T0203Z-checkpoint-trn80-tie-crossing-rehearsal (the crossing in nine steps, as commands), with training/20261111T0222Z-checkpoint-trn80-tie-crossing-fix-commit (the fleet rows' commit, the crossed body's generator, the check of your two figures).

Conflicts, all [run] on scratch branches cursor/training-cross-rehearsal-8591-20af (9e17e8a69a44) and cursor/training-cross-rehearsal-8703-20af (4d6dc7e34e3a, then 67bdbbbe5eec):
- #8591: none, by `restack.py` or by a plain merge of 178f49c90c89.
- #8703 by `restack.py`: none (status clean; the lock and the import list merge by their drivers).
- #8703 by a plain `git merge 178f49c90c89`: one file-location conflict, `Security/Proofs/Nci/CutCharged.lean` (fix: `git add security/Proofs/Nci/CutCharged.lean && git commit --no-edit`), and one silent break, the lock's `Nci.Cut.Guarantees.HECharged` entry keeps two `Security/` paths (fix: exact replace to `security/` in security/lean-audit.json, its own commit). Use restack; the plain merge is written down only as the fallback.
- Suites on the crossed trees, six each, all passed: r20261111-014053-109d (#8591's copy, rho 602 passed), r20261111-014119-f7b8 (#8703's copy, rho 610 passed).

Lean after the landing:
- #8591: nothing. All 32 packages' Lean inputs keys at its crossed head equal the move's head's [run, lean_changed.py --inputs-key].
- #8703: three packages' keys differ (security, security/Proofs, security/Properties); at most 53 of 4,630 source modules are downstream of its four Lean files [run, import closure]. It needs one records run (`lean_changed.py --records security --update`, --cwd clone) and the mark's preflight; the kernel replay is the queue's.
- Reuse: on the node that holds ci's audit of the moved tree (cpu-4, r20261111-003108-8598 [read, TOP-398]) the node's artifact cache should cover every module but those 53 (expected from how the cache is keyed, not tried: I started no Lean build).

Time estimate, from the landing: both crossed and pushed in about 15 minutes; #8591 re-marked 15 minutes after that with the six suites (about an hour with all 50 touched suites); #8703's records run 25 to 35 minutes on cpu-4 with the cache warm, 45 to 115 minutes on another node (the earlier records run took 5,186 s, 4,787 s of it the security/Proofs build [read, r20261110-180637-c614]). The node name for the records run is untried. The records run could go before the landing on the rehearsal branch, since the gate takes a pass of any commit on the same keys: your call, I started none.

Ready for the crossing, so it is a repeat:
- The fleet rows' commit: 67bdbbbe5eec on cursor/training-cross-rehearsal-8703-20af, `git cherry-pick 67bdbbbe5eec` onto the crossed #8703. It pins the six fleet rows at the node's own rate, that w = 0 gives 1 at each, and bound = 1 at a block untied whole; every row's untied words come from one function of its blocks and phases; the README and the docstring say a figure there states its threshold. No Lean file. rho 610 passed and repository 280 passed on it (r20261111-021419-e5c9) [run].
- The crossed body of #8703: `cross_body.py` in the 0222Z note regenerates both tables' 24 rows and the bullets from the function, and the moved paths; "What ran" is left for the crossing's runs.
- Every printed figure on the per-node scaling is listed in the 0203Z note: none in #8591; in #8703 the test's six PERIOD rows and four count-alone rows, and in its body the period table's six rows, the pooled table's eighteen and two bullets. The commit and the generator cover all of them.

Open, yours to rule: whether the census's peak rate is the scale for "a datacenter" (your note's first unchecked item); I pinned the peak-rate rows, as the body states them.

