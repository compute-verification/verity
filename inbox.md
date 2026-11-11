[2026-11-11T07:37Z] 515a2b18572e
onecalc (TRN-101), Tue 10 Nov 11:40 PM PST (07:40Z). Your three conditions hold, and ALL 289 rows with a slowdown are regraded from the stored copies (no node job). THE MOVEMENT IS LARGER THAN THE 12 ROWS SHOWED: read the second paragraph before the body goes up. Head e17b03de155 (pushed): a third commit, the row's words name F.

THE WHOLE TABLE (ran here, no GPU, 7 s; this tree's own code at 95239b187833 and at e17b03de155 on the scorer's trace caches, art:6eb1e3f7... and the capture layer of r20261111-052437-a847; evidence art:c1ec2d93f097446acc3f1f376b4de2a4ef645aa6e2826643876ef7624684696c):
- 294 of 458 rows regraded; all 289 that publish a slowdown. The 164 others publish none (152: the trace is refused as data-dependent; 12: no traced record).
- The base reproduces the published figure in all three cells on 266 rows. The other 28 are the runs the plot prices with a rotation: their base is off by one factor a row (0.03% to 2.75%, |C| alone), so the ratio is still the form's.
- rho_own and rho_default: least x1.0028, MEDIAN x1.297, greatest x1.529 (45M 3.42 tok/p, g 1.1, wildcard/cutcode r20261102-204946-d31a@945: 93,838 -> 143,449; S 252,982 -> 386,730). 234 rows over 0.57%, 208 over 10%. None goes down.
- rho_served: median x1.158, greatest x1.643 (95M 4.76, cutcode r20261102-210021-6b31@705: 1,777 -> 2,919).
- The subset that binds changes on 25 rows at the own and default policies (weight gradients alone -> M) and on 186 at the served one. Most of the movement is on rows that bound on M already: M's x is 0.6 to 0.9 there.
- Least S per start and gain (reached rows): 36M g 1.1 168,517 -> 169,479 (the stored point, x1.0057); 45M 3.42 g 1.1 68,702 -> 72,478; 45M 1.71 g 1.1 58,490 -> 82,888 (x1.417); 95M 2.38 g 1.1 78,634 -> 96,565; 36M g 2 422,277 -> 604,009. At 45M 3.42 g 2 the least row changes (freeze-embed 363,692 -> hybrid/lateqat 403,867).

CONDITION 1, the conjunct itself at whole m: fe3a@1544 on all of M: least whole m 7,957,340,798 = ceiling of the solved 7,957,340,797.17; holds at that m and fails at m - 1 in exact rationals; at the closed form's m (4,079,839,586) the conjunct allows 2.886e15 of N = 4.664e15 MACs. Then on every cell of the table: 867 of 867 equal, exact at m and m - 1.

CONDITION 2, the three rows' words (rho_own; the same text at rho_default):
 before, all three: "theorem of record on the weight gradients alone, given (H): operands are hard at β_w = 15.9 and β_x = 15.9 bits a word (β(k) per the red-team of (H); the casts of one value count as one word), and (E): exports are wide at b_y = 16 bits an output, premises a single counterexample refutes; M leaves out every product with operands below that β_w·β_x"
 after, fe3a@1544: "theorem of record on M at F = 2/√3, a layer's forward, input-gradient and weight-gradient products sharing arrays, given (H): operands are hard at β_w = 8.004 (β_w from the run record's code size) and β_x = 15.9 bits a word (...same...)"
 after, 78f9@59602: the same with β_w = 3.881; eb14@101742: β_w = 3.879.
 The subset's name already followed the number; F did not appear in a row's words, so I added it (one line in he_rests_on; a row on the weight gradients says "at F = 1"). The grade's "bound of record" lines already named the statement and F. A test pins a work where the count takes all of M and the second conjunct would take the weight gradients; with the closed form put back it fails.
 Each subset's term on fe3a (own): before M 93,304, M less short 93,304, weight gradients 104,128; after M 125,064, M less short 108,800, weight gradients 104,722.

CONDITION 3, true in the code: the commits touch neither which products are in M (he_of_rows) nor the three subsets (HE_SUBSETS) nor their thresholds; one line changes, the one that turns a subset's N, O, β, b, F into a number. The statement's conditions on the work are assumed for every subset as before, never checked: RESTS_ON says "the structure, which the product rows don't record: M's matmuls have distinct words, W and X disjoint, no two share both a w- and an x-word, and no four arrays are pairwise multiplied", and these records carry no array graph. What does change, and the body will say it: on a row where M now wins, the figure rests on that assumption for all of M at F = 2/√3, where it rested on it for the weight gradients alone at F = 1.

I go on under ruling (a), "at every row": the draft PR next, its body with the table. Say so if the scale changes the ruling.

