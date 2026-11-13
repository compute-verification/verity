[2026-11-13T02:14Z] ce42f77c6702
BACKWARD2, Thu 12 Nov 6:14 PM PST

**Every fit at the fixed head gives every word, and none of the three changed orders does; the head is `3b384b35f2`, pushed, with suites and circuit-check running on it.**

Fits at `e4440c634a`, the Pre fix. `3b384b35f2` changes only CONFORMANCE's text and `evidence.json`. All RAN on CPU, every word, compared as raw uint16 bits.

| run | heads | dQ equal | dK equal | dV equal |
| --- | --- | --- | --- | --- |
| r20261113-002743-4e0e, captured, deterministic | 32 | 2,097,152 of 2,097,152 | all | all |
| same run, captured, default mode (no Definition) | 32 | 2,097,097 of 2,097,152 | all | all |
| r20261113-002814-ac8e, first probe `ovf` | 4 | all 262,144 | all | all |
| same run, first probe `ftz` | 4 | all | all | all |
| r20261113-002826-1139, second probe `fma0` to `fma7` | 32 (4 a case) | all 2,097,152 | all | all |

- The default-mode heads give the same 2,097,097 dQ words as the fit at `1178eb5439` (RAN r20261112-212456-3c41), as expected: that mode's dQ is added by atomics in the card's schedule.
- NaN (READ the probes' npz): the card's NaN words are all 0x7FFF. They are dQ and dK of `ovf` and `fma1`, 262,144 words a gradient, and no other case has a NaN or an inf. The Definition gives the same bits, since every such word is among the equal ones. It gives no other NaN.

Condition 2 (RAN r20261113-002701-acc6, at `e4440c634a`, on CPU). Each order is fitted on two whole captured heads; the counts are words that differ from the card's.

| order of a thread's sum | `s/det/wide` pair 3 | `l16/det/wide` pair 3 |
| --- | --- | --- |
| the kernel's (control): dim 1 FMUL, then 0, 2, 3, ..., 7 | 0 | 0 |
| dims 2 and 3 swapped | 46 dQ | 0 |
| dims 2 to 7 reversed | 79 dQ | 51 dQ |
| dim 0 fused after dim 2 | 0 | 61 dQ |

- dK and dV are equal in every row of that table, so no changed order fits every word. So the captures exclude each of the three changed orders, two of them on one head each.
- What the evidence does not tell: it does not exclude every other order of dims 2 to 7. Only these three were fitted, and the twin found most heads tell none of them.

Condition 1, in CONFORMANCE at `3b384b35f2` (READ the file): "The probes' words tell that the dQ atomics flush to zero and, of the dP sum, two things only: a thread's second dim's product (dim 1) is rounded alone, and each of its other seven is fused. That its first dim (dim 0) is fused next and dims 2 to 7 then in turn rests on the captured heads' every-word fit, which no probe tells."

Condition 4: `evidence.json`'s `cc12_fa2_backward_capture` gains `probe_ovf_ftz`, `fit_ovf_ftz`, `probe_fma` and `fit_fma`, each an `art:` run record. `fit` is now the 64-head fit at `e4440c634a`. Nothing else of the probes is in the tree. The overflow-row test (`test_the_dp_sum_multiplies_a_thread_s_second_dim_alone`) is built from the Definition's inputs, and the quick tier passes on the VM (13 passed, 1 deselected).

Forward: changed nothing. The probe case goes in the handoff note. INFERRED: the two forms differ only where every split's product in a column is exactly -0. For the split with the largest LSE, whose scale is at least 1/4, that needs its O to round to -0: a tiny negative P V sum, about -2^-143, times 1/total. So the case rests on the tensor-core chain keeping a subnormal sum.

Running at `3b384b35f2`: suites r20261113-021349-f32c (the base is the merge-base `6d95fcc68d`; the suites at `a809ac0598`, r20261112-233635-4858, gave "59 suites: 59 passed (0 cached), 0 failed") and circuit-check r20261113-021421-6f37. The body follows when both are in, well before Fri 10 AM PST.

