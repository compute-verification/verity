[2026-11-12T19:02Z] b621cb436b15
TERMGB to the network-accounting coordinator, Thu 12 Nov 11:04 AM PST. **NET-394, items 1 and 2 as you ruled them at 10:51 AM, are built and pushed: `cursor/na-certifier-term-simulated-b4e6` at `c3e29d66cf7b`, two commits on `main` `0582899c4392`. proofs' worker can build on it now. The harness suite fails at this head, on the mark before any label: 12 of its 14 tests. The count's term is next.**

**What the branch holds** [ran: both test files at the head, 21 passed and 15 passed]
- `OF_RECORD` stays 2. The document's `value`, the tables' column order and the result's metric are as on `main`. No number moves: the document at `--response-ids 1024` differs from `main`'s in the label, the new input, the new `mark` and two legend lines, and in nothing else [ran: a diff of the two documents].
- `SET_TIME` is "with a simulated set-time card"; its short form, the rows `timing` and `all three`, the lines above the tables and the README say the same.
- `MARKS` has a fourth mark, `simulated`. One input carries it: `set_time_card`, value `simulated`, source soundness's clause in your words, with no measured figure. The document's `with_set_time_card` carries `mark: simulated`. The schema stays `certifier-term/v2`: what is new is one input and one key.
- With `--out` the result holds one more measurement, `certifier_term/modeled/simulated/metric`: the scan's row over the world's year at D_j = 1, in GB against the same target. Its `inputs` open with the row it is of and then the card as the document states it, the clause whole. The kit has no mark of its own on a measurement [read: `kit.py`], so nothing was refused: "simulated" is in the name and the inputs.

**The harness suite at this head** [ran: `benchmarks/harness/tests/test_exfiltration.py`; at `main`'s tree all 14 pass]
- 12 fail, 2 pass. 11 fail at `verity_bench/exfiltration.py:394`, its own check of marks: "an input is marked one of measured, modeled, assumed, with its source: set_time_card". The twelfth, `test_verity_bench_exfiltration_is_the_commands_runner`, exits 2 at its line 732, the same refusal through the runner.
- So the first line proofs' worker needs is `exfiltration.py:102`, its `MARKS`, taking `simulated`.
- With that one line given in memory and nothing written [ran]: 13 pass, and one fails on the label's text, `test_what_rests_on_the_set_time_card_is_printed_only_under_the_calculators_label` at its line 523. Lines 523, 524, 529 and 530 of the test spell "with a set-time card (not yet tested)"; each needs "with a simulated set-time card".
- One more reader states the old way and is not mine to touch: `benchmarks/exfiltration/README.md` lines 36 and 106 ("no timing test has shown").
- I changed nothing under `benchmarks/harness/`. My pull request cannot pass that suite alone: it lands with proofs' change or after it. Say if you want it otherwise.

**Other readers** [read: a grep of the tree for the calculator's names]
- `chain_record.py leak` reads `per_link`, `at_link_rate`, `READINGS` and `WEIGHTS`; `deviations.py` reads `terms`. Neither reads the document, the label or the marks, and I changed none of those functions: each prints what it printed.
- The cards (`benchmarks/spot_figures/cards`) are read by the calculator and read nothing of it: unchanged.
- The only reader of the document outside the benchmark is the harness's `exfiltration.py`.

**Next:** the count's term. From #9378 at `1d6ebf70ff8c` the rule is R = lanes × ⌈(T + ingress buckets − 1)/(D + 1)⌉ [read: `WardenParams.phase_releases`]. Its worker's 336 is that rule at 16 lanes, T = 600, D = 30 and 30 ingress buckets [computed]. At the benchmark's 1,280 buckets and the same schedule it gives R = 688, and six classes are 6 × log2(689) = 56.6 bits a link a phase [computed]. The lanes, the delay and the ingress buckets are not inputs of the benchmark today: I make them inputs, assumed, at that schedule's values, unless you say otherwise.

**The suites' run** `r20261112-181342-dd6c` has a slot: `running` at 11:03 AM PST [read: its status].

A slip: my last message said 10:54 AM PST; the clock read 10:52.

TIMER: none

