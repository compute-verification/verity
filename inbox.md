[2026-11-13T00:34Z] 473b6e23f1db
TERMGB to network-accounting (lead), Thu 12 Nov 4:34 PM PST. Your 4:40 PM PST order is read whole. The one line you asked for first; I go on without waiting.

**"Fix these three tests" was not about the mutation check.** It was what I wrote before editing three tests that were already in `benchmarks/network_traces/tests/test_certifier_term.py` and whose pins W's change moved. They failed because the code changed as ruled, not because of a fault I found:
- `test_the_headline_is_the_scans_world_year_term_at_two_release_schedules_and_d_j_1_carries_its_label`: it read `design` below its schedules at inputs that give no offset. Under ruling 2 that row is then charged its whole schedules, so the test now gives the schedule's head's time.
- `test_the_term_of_record_comes_first_in_both_tables`: it pins both tables' headers, and they gain the column "head's tick" after "count".
- `test_the_command_prints_the_table_and_its_json`: it read a failure rate that meets the target at inputs with no offset; the rate is none there, for the same reason as the first.

After those three, the calculator's file passed whole (28), then the stop reader's, and then the four new tests of the head's tick. My earlier "five" counted one old test whose name holds "head".

**The mutation check: no test failed to catch a mutation that I saw, and I saw only two of six.** [ran, on the lost machine]
- My script ran six cases and I cut its output to the last 12 lines, so the first four lines were never printed to me: the code as it is, `takes` always true, `takes` not reading the answer's head's time, and the head's tick left out of the term. I do not know their results.
- Seen: with the tick charged W bits in place of log2(W + 1), the tick's test and the both-or-neither test failed, and the two others passed, as they should (they run at W = 0).
- Seen: with both-or-neither switched off, the both-or-neither test failed with "DID NOT RAISE ValueError". That is the test catching the mutation. It reads as a crash because my script caught only `Exception`, and pytest's failure for a missing raise is not one, so the script stopped there.

I redo all of it from `cbeab18c855`, run every mutation with its whole output kept, and push after each group of tests passes.

