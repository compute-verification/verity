[2026-11-12T21:02Z] ffa30efb94a4
PREMISE, Thu 12 Nov, 1:02 PM PST. Review of #9272 at 795e1c770e08 (still the head).

**NOT READY: 4 must-fixes, each mechanical. All three tests do fail when their repeat is taken away, and both suites pass on the head merged with main.**

MUST FIX
1. test_batch_premise.py:18 still has EQUAL = 1e-4, so APART is 0.1: the pair you rejected. Fix: EQUAL = 2e-5, which makes APART 0.02, and change the comment to "under a hundredth of a bf16 word's step". RAN r20261112-202651-b6ca, seeds 0-999: at 2e-5/0.02 every equality and control holds at all 1000 seeds. The worst equal pair is 1.6e-6 (12 times under EQUAL). The worst controls are 0.0279 (shared prefix), 0.0451 and 0.0548 (run, rotation on and off) and 0.0201 (proportions, seed 376). At 1e-4/0.1 the controls fail at 25 to 67 seeds of the 1000.
2. The third test's parting assertion (line 119) goes. At 0.02 it fails at 4 of 1000 seeds (worst 0.0120, seed 189). It does hold something the earlier lines don't: the repeat stops at the second attention. Your ruling still drops it. If you want that claim held, a floor of 100*EQUAL holds at all 1000 seeds with 6 times margin. RAN, the same run.
3. The branch doesn't merge with main. Conflicts are in training_slowdown/library.py:33 and tests/test_library.py:7, import lines only. Fix: keep both sides. RAN: my merge baa68ddee8 (local scratch branch, not pushed).
4. Once merged, the premise is stated twice. main's 6242aba2e0 restates it, with its own wording of the three breaks, in circuit_properties/work.py:90-95 (module docstring), circuit_properties/README.md:56 and test_value_reading.py:8-10. So done-condition 1 holds on the PR's tree but not on the merge. Fix: those three name training_slowdown.work.P_BATCH, and test_program_reading.py gets one line beside its SHARED_PREFIX check: assert TW.P_BATCH in " ".join(WK.__doc__.split()). Those files belong to another worker. READ.

SHOULD FIX
5. "Earlier tokens in the same proportions" (work.py:29, README:371-372, test docstring :109) has to count the position's own token: causal attention reads places 0..p. In the test's own batch, place 1's earlier tokens are [1] and place 3's are [1,2,1], which are not in the same proportions. Fix: "tokens up to it, its own included". INFERRED from model.py.
6. The run case. M leaves out position 0 (work.py:180), so a run of 2 doesn't break P_BATCH as written, and a run of r over-charges r-2 positions, not r-1. Fix: "a run of three or more", "(r-2)" (work.py:26-28, README:371). INFERRED from work.py.
7. library.py:56: the "proved" part says "given its two premises", while the "premise" part beside it now names three. Fix: "given its premises".
8. The proportions control's worst seed is 0.0201, 0.6% over APART. Option: APART = 500*EQUAL (0.01). My patch keeps your 0.02. Your call.
9. The testing skill asks a swept bound's comment to name its seeds and worst value; that pulls against the no-measured-number rule. My patch names them. Strike them if you rule otherwise.

NOTE: main now also has a slow test of the three breaks (test_value_reading.py::test_a_batch_that_breaks_the_premise_repeats_pairs_the_reading_charges). It holds the circuit reading, while this PR holds the forward's words, so keep both. On the backward: P_BATCH's comment says exactly which words, the forward's, which forward products and weight gradients read. That is true of he_of_rows, which charges dw with the activation operand X repeating and dY not. So your condition is met. READ.

Stated once: yes on the PR's tree. RESTS_ON, RESTS_ON_CELL, he_rests_on, library.PARTS["premise"] and grade's "proved" all interpolate P_BATCH, and none copies its text. No on the merge (item 4).

Would each test fail without its repeat: yes, all three, each at its equality line. RAN r20261112-202651-b6ca, 8 mutations:
- shared prefix: rows sharing only their first token, or non-causal attention;
- run: the run cut at place 1, or an absolute position embedding, each with the rotation on and off;
- proportions: the rotation kept, or one 1 with three 2s.
At the held batches, all pass.

Tree rules: no number, run id, date or name added; no wall-clock; ruff clean (RAN locally). The test file takes 1.49 s for 4 tests (RAN b6ca). It can't flake: CPU float32, seeded, and at seed 0 the equalities are 15 times or more under EQUAL.

Suites at the head merged with main (RAN r20261112-192301-747a, baa68ddee8):
passed  benchmarks/training_slowdown 580 passed in 496 s; 580 outcomes recorded
passed  benchmarks/training_slowdown/grader 28 passed, 3 skipped in 35 s; 28 outcomes recorded

In flight: r20261112-205811-987f, the same two suites plus benchmarks/circuit_properties, on the merge with every fix except 8. The patch is in the note. Its result follows in a second message.

Full review: note:training-20261112T2100Z-report-premise-review-9272

