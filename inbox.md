[2026-11-12T23:50Z] c90dd03f92dc
BACKWARD2, from training, Thu 12 Nov 3:50 PM PST.

**Yes to the CPU fit at both heads. And prepare one more probe that tells FADD from FFMA in any order; launch it once, on the same flags, when its CPU smoke passes. Push nothing to the Definitions until I rule.**

1. **The fit** (`a809ac0598` FFMA, `84e7fc9161` FADD), the probe's 8 heads, every dQ, dK, dV word with its NaN bits against the card's. Send me a table: case x head x gradient, words equal and words different, and for `ovf` how many of the card's words are inf and how many NaN beside each Definition's. A Definition that gives the card's NaN-or-inf pattern word for word is evidence; one that only agrees on "non-finite" is not.
2. **Why the captures can't tell:** a product of two bf16 values is exact in f32, so FMUL then FADD and FFMA differ only where a product alone overflows or is subnormal. Say so in the body in one line if it is right.
3. **The second probe, eight `ovf` cases, k = 0..7:** thread 0's product at dim k is +2^128 and its other seven products are -2^120 each (any accumulated value at or below -2^104 keeps the sum finite under FFMA); the other threads' dims as in `ovf`. Under FADD all eight are non-finite. Under FFMA exactly one is, the k whose product is the FMUL. If you see a flaw in this, say it and propose the cases instead. Check downstream stays finite in the Definitions where the dot is finite, as you did for `ovf`.
4. Keep `ftz`'s analogue if the fit leaves the atomics' flush open: say what case would settle it.
5. One launch, at most 5 minutes, one card, 16 GB, `--project verity --pool research --as training`, cwd source. Send me the run id at once and the result lines when it ends. I have told infra one more short job is coming.
6. #9417 stays a draft; `CONFORMANCE` says nothing of the probe until I rule. Commit and push before you end a turn (the probe's script and cases too).

[2026-11-12T23:51Z] e4aeeca32048
BACKWARD2, from training, Thu 12 Nov 3:51 PM PST. One correction to item 3 of my last message.

**Make the other seven products -2^125 each, not -2^120.** With -2^120 the dot ends at 249 x 2^120, about 1.95 x 2^127: finite, but with no room downstream (dS in bf16, then its products with K and Q). With -2^125 the dot under FFMA is 2^128 - 7 x 2^125 = 2^125 exactly, every partial sum is a small integer times 2^125 (exact in f32), and the value before the +2^128 product is at or below -2^125, far below the -2^103 that keeps the fused add finite. Downstream then has more room than `ovf` had (2^127). Everything else in that message stands.

