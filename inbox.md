[2026-11-13T00:16Z] 6325648f7dc1
BACKWARD2, Thu 12 Nov 4:16 PM PST.

**The fits settle both instructions: the dQ atomics flush to zero (the FTZ head gives every `ftz` word, the head without FTZ misses about 18,500 dQ words a head), and on `ovf` only the FADD head gives the card's words, NaN for NaN. With the second probe, that points to the dP sum being FFMA with each thread's dim 1 as the FMUL.**

Both fits are RAN, on the first probe's 8 heads (4 of `ftz`, 4 of `ovf`), with all 65,536 words of each gradient compared:
- `a809ac0598` (FFMA, FMUL on dim 0, FTZ): r20261112-234814-8ecd, run record art:2bc4f744b2095d72504d05029677579884b9b2fd751aa8735a0d74113074625d.
- `84e7fc9161` (FMUL then FADD, no FTZ): r20261112-234851-c9e1, run record art:ee9e59603d0d9ad3a71c242c6c7106185e14f17612b52c5df06e06e1d56dd506.

| case | head | gradient | `a809ac0598` equal / different | `84e7fc9161` equal / different |
| --- | --- | --- | --- | --- |
| ftz | 0, 1, 2, 3 | dQ | 65,536 / 0 each | 46,910 / 18,626; 47,004 / 18,532; 46,549 / 18,987; 47,035 / 18,501 |
| ftz | 0, 1, 2, 3 | dK, dV | 65,536 / 0 each | 65,536 / 0 each |
| ovf | 0, 1, 2, 3 | dQ, dK | 0 / 65,536 each | 65,536 / 0 each |
| ovf | 0, 1, 2, 3 | dV | 65,536 / 0 each | 65,536 / 0 each |

NaN and inf on `ovf`, dQ and dK, in the four heads (262,144 words a gradient):
- The card: 262,144 NaN (every word 0x7FFF) and 0 inf (READ, the first probe's archive).
- `84e7fc9161`: equal to the card word for word, so 262,144 NaN (0x7FFF) and 0 inf.
- `a809ac0598`: equal on no word. Each head lists its first 200 differing words a gradient, and all 1,600 of those are finite where the card's are NaN. INFERRED, from the twin: every word is finite, since the dP sum is a finite 2^127.

So (INFERRED):
- FTZ (must 2) is confirmed on the card's words.
- On `ovf`, FADD's pattern is the card's word for word, but the second probe says FFMA. The two agree here because dim 1's 2^128 product is rounded alone under both FADD and FFMA with dim 1 as the FMUL, so both give a dP sum of inf.
- The reading that fits all three facts is FMUL on dim 1, FFMA of dim 0 into it, then dims 2 to 7 by FFMA. The facts are: second probe, fma1 alone non-finite; first probe, `ovf` NaN; 64 captured heads, every word.

On the next step:
- On your ruling I make that change and its test, then fit both probes' heads at that head on the CPU pool. The second probe's run record is art:7ba7ac7f3a42314d7a9473eac81e1fb214134d1d4467d1954aad0376ab5a8824 and the first's art:9b8fed01914051a93281aa6d273cec6075cd796fcf4a0c11411c2a055346be50. The 64 captured heads need no rerun for it: at normal magnitudes it rounds exactly as both fitted heads do. INFERRED: a bf16 product is exact in f32, so FMUL then FFMA of the first two products rounds p0 + p1 once whichever is fused.
- For the body, the line on why the captures can't tell: "A bf16 times bf16 product is exact in f32, so FMUL then FADD and FFMA, and which product is the FMUL, differ only where a product alone overflows or is subnormal; no captured word has one." It is right: the 64 heads' every-word fits at both heads (FADD, and FFMA on dim 0) are the evidence.

