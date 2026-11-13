[2026-11-13T17:42Z] bfacb83f6a92
TERMGB to the lead, Fri 13 Nov 9:41 AM PST. Your 9:35 AM rulings are read. Two heads.

**#9533** `cursor/na-certifier-term-delta-b4e6`: `2389a3d7ad68` -> `10414e54a035`, a fast-forward [ran; read from origin].
- The commit is `10414e54a035`, one file, `benchmarks/harness/tests/test_exfiltration.py`, 3 lines added and 2 removed:
  - in `test_under_a_research_run_the_runner_records_its_three_figures_and_prints_the_same`, the pinned first input of the network certifier's figure: `"modeled: the network certifier's term of record, its `scan` row ... at D_j = 2, on an assumed class of at most 1,024 ids a response"` is now `"modeled (run, not proved): ..."`, the rest of the string as it was, rewrapped over its two lines;
  - in `test_the_legend_line_shows_a_rows_mark_as_its_calculator_gives_it`, one line added before the unmarked print is made: `monkeypatch.setattr(lb, "document", marked(None))`.
- At this head the harness's file passes whole on the VM, 44 of 44 [ran].

**#9501** `cursor/na-stop-verdicts-b4e6`: `f4d2e20e290d` -> `0ab489582d96`, fast-forwards [ran; read from origin]. Four steps:
1. `45171bc4bdff`: the tool's result (run `r20261113-165436-acea`: `clean`, 28 crossed, none skipped).
2. `603ccdc11e17`: the hand merge of `main` `d119240998be` (run `r20261113-173153-6cc8`). One file conflicted, `benchmarks/harness/tests/test_exfiltration.py`, proofs' file, in the two `NO_VERDICTS` lines. The harness's own test printed the digests at the merged tree in that run and passed on them [ran]; the job checked that the tests read the merged tree's own packages [ran]. Against the automatic merge the commit differs in those two lines alone [ran]. Pins step rc 0, nothing changed [ran].
3. `3aa592070f9a`: a merge of #9533's head `10414e54a035`, so that #9501 holds the branch it is built on again and its two tests' mend. One file conflicted, the same, in the same two lines; #9501's digests stand. Against `603ccdc11e17` the merge changes the five lines of #9533's mend and nothing else [ran: the two diffs compared].
4. `0ab489582d96`: the reader's test takes its schedule from the cost runner's `costed`. `test_stop_verdicts.py`: 27 passed on the VM [ran].

**Two things to know about #9501's head.**
- **Its `NO_VERDICTS` digests are its own and differ from #9533's.** #9501's calculator moves the print with no verdicts of record. At the old head that pin was left as #9533's and its test was one of the nine red ones; by your order of this morning it is now what the harness's test prints at #9501's tree, and that test passes.
- **Eight of the harness's tests are red at this head, 36 pass** [ran on the VM]: the old body's nine by name, less the pin's test. The cause is the one the body gives: the harness hands the calculator no verdict records, so the rows `all three` and `design` charge their schedules and the harness's pins of their figures and of the rows that take the output count at 0 no longer hold. Nothing new came with the carry.

Next, on #9501's head, each its own commit: `FORMATS` empty, the ingress mend with its two tests, the schedule's check on the ids with its test. Then the comment on NET-431, the runs at the two heads, the bodies, the writer.

