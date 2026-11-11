[2026-11-11T02:44Z] 5b198aee5ff3
TIE, Tue 10 Nov 06:44 PM PST. JOB 4 DONE, and the records run is in the 0203Z note. Note: training-20261111T0243Z-checkpoint-trn80-tie-lean-form (script and output whole). Nothing published tonight is above lean's form; nothing for you to correct.

(1) WHICH READING. All three are lean's form [read: bound_on_rows, bound_on_budgets_he, outOn, outOn_le at d5bc5d162559]. One beta_x with w = bound_on_budgets_he at G = the blocks with at most w untied words. One beta_x at the top budget = bound_on_budgets_he at G = every block. A beta_x a block = bound_on_rows, beta_x(u) the table's row for kappa_u. Each has the outputs pooled once against the whole Q; none pairs a block's budget with its own outputs. A script that evaluates lean's right side without bound_charged reproduces all 60 cells to the decimal [run].
The one difference: with a threshold, bound_charged takes H at the most blocks that can be charged AND Q with none charged. To be lean's form exactly it would take the largest right side over the splits: n charged blocks, Q over the other blocks with untied - n(w+1) words. It prints less than lean proves, never more.

(2) THE ROWS [run, CPU, crossed #8703 at 67bdbbbe5eec]. calc -> lean's form at the developer's best split:
- one beta_x with w: week 1,614.1 -> 2,758.1; month 4,787.8 -> 6,931.5; year 20,422.0 -> 20,728.8 (same w).
- a beta_x a block with its w: 1,694.1 -> 3,016.8; 5,779.8 -> 9,446.7; 24,820.5 -> 29,705.3.
- no threshold, equal in both: 52.8 / 202.2 / 1,336.6 and 53.2 / 209.6 / 1,844.3.
- the world's rows are a datacenter's.
- one node, four blocks a phase: equal up to 17,280 blocks (3,677.1 / 5,088.7 / 7,883.9 / 9,482.5); then 12,564.7 -> 12,858.8 (54,300), 22,391.1 -> 22,611.5 (210,240), 28,795.0 -> 28,869.2 (1e6), 35,550.2 -> 35,550.8 (25,165,824), 37,647.8 both (1e10). Without a threshold and with a beta_x a block, equal.
"lean" figures are to a 1% grid in n, so they can be slightly high; they are never under calc.

(3) PUBLISHED [read from ASK-269 at 6:27 PM and the two goals at 6:26 PM; lean's side run]. None is above lean's form:
- 1,614 / 4,788 / 20,422: under (2,758 / 6,932 / 20,729).
- 53, 202 to 210, 1,337 to 1,844: equal. Both ends are lean's pooled form. The lower takes (H) once at the top budget, the upper at each block's own row. So "lean proves the lower end" is not right for these: bound_on_rows gives the upper end, on (H) at five rows instead of one. It fits this morning's 15,995 to 17,714, already replaced.
- w = 0 gives 1: equal. 37 and 35,500 at 2.5e7 blocks: 37.0 and 35,550.8.
- "the target from 2,400 blocks": lean's form gives 2,396.4 there, and passes 1,000 from 798 blocks.
- 253 / 314 / 338: 254.8 / 314.1 / 337.7.
- 427, 340, the other option's figures, 0.097% and the rest: every block at one budget, where the pooled form IS the one-block bound (37,653.4 and 1,598.5 at 1, 4, 1e6 and 1.651e15 blocks [run]).

(4) THE CROSSING COMMIT: no change. It pins figures lean's form implies. Tightening bound_charged to the best split raises the threshold figures by 1.7 / 1.4 / 1.02 times; that changes the function, its tests and published figures, so it is your call and not the crossing's. After #8700 lands, HE_ROWS (a draft id) can name bound_on_rows.

RECORDS RUN: r20261111-023046-96bd, cpu-1's research pool, Lean slot since 6:33 PM, 180 min timeout. Check and the copy onto the crossed #8703 are in the 0203Z note's 6:37 PM addition. All 32 Lean inputs keys at 67bdbbbe5eec equal the rehearsed crossing's [run]. Spent if the move's tree changes from c731a2385a3a. origin/main was a930252c4993 at 6:42 PM: the move is not on main. I am not waiting on either. Ending my turn; resume me for the crossing.

