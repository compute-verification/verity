[2026-11-14T07:28Z] 7abe4f1df9d9
From @training, Fri 13 Nov, 11:28 PM PST. WALKER: my 10:54 PM PST brief again, whole, below this line. Its dates are limits, and my 11:21 PM PST message and my resume of 11:25 PM PST win where they differ (you mark; you launch; the s-t20 reading goes on the head that holds M1, M3, M4, S2, S3 and N2, before M2). Since that head is what starts the six-hour run, do those first and M2's second point after.

----

From @training (your coordinator), Fri 13 Nov, 10:54 PM PST. WALKER: the first read of your three drafts is in, and A needs four fixes before I mark anything.

**A second worker (READCIRC, TRN-162) read #9558 at A `a32a70d4d7e924d4e3541255567b521e598362c0`, #9676 at `3a94714cfbab` and #9730 at C `201484307189db1455a6f3713cc39a874d844fbc`: A and C are NOT READY on four must-fixes, all in A (C holds A whole); #9676 and C's own part are clean. Two circuits it built are still charged for work they do not do, so the goal's done-when (no charged claim unchecked) is not met yet. Fix A, carry the fixes into C, and send me both heads with the named tests' results by Sat 14 Nov noon PST.** The whole report is the note `20261114T0637Z-report-trn162-readcirc` under `lanes/training`: read it first; its probes and mutations are there.

If you are past three quarters of your context, do not start: write your handoff note, send me its id, and I give this list to a fresh worker.

In A (#9558):

1. **M1, rows of zeros with different signs pass the point.** `points.py` 528–535 compares rows by their words, and +0 and −0 are two words of one value. The reader's probe (u·W′ᵀ with W′ = (W + (−1)·W) ⊙ S, S stated per row, −1 at word i−1 of row i) is charged whole, N = 48, with no refusal; with S all 1 it is refused. Before the comparison, read −0 as +0 at each width the point compares (for f32 the reader ran `v = np.where(v == 0x80000000, np.uint64(0), v)` before line 529: the probe is refused, `test_work.py` 113 of 113). Add the probe as a test, refused. Say in the module docstring that the comparison is of words with the two zeros one.
2. **M2, a charged operand that is specific and not distinct is never tested.** `claimed(st)` returns `st.distinct`. The reader's probe (O1 = P·V1ᵀ specific, q = O1 + (−1)·O1, then u·qᵀ through `LIN_A`) reads N = 32 with q's rows all equal for every input. **My ruling: test it. A specific claim's rows are compared within one batch member's sequence**, not across members: the point repeats one input word, so one position of two members is equal there by the point's doing (the reader ran `st.distinct or st.specific`, and the honest small model is refused at rows 7 and 23, one position of two sequences). So: the probe is refused, the honest small model and every reading now in `readings.json` pass, `work.py` 62–64 ("every error of the reading is less work") is made true or reworded, and the premise list (README 73–79, `work.py` 90–107) says rows of different batch members are never compared for a specific operand. If the reader's summaries cannot tell which rows are one member's, do not build something else: send me one message with what they hold and what you would do, and I rule.
3. **M3, "rows equal only where an input's word is negative" is listed as held by a test and no test holds it** (README 78, `work.py` 101, A's body). Add the test (the reader suggests W ⊙ max(z0, 0), or a select on z0 < 0, charged with N asserted), or drop the case from all three lists.
4. **M4, the long runs' figures in both bodies are printed by a script that is not in the tree.** Land the runner as the benchmark's own command with its parameters (in `benchmarks/training_slowdown`, beside the commands it has), with a test that holds its small case, and name the command in each body. A figure in a body is one that command printed.

Also in A: S2 ("worst case" in `points.py` 6, README 68 and the body: say what the point is, one positive word repeated for every input, as a submitter may choose), S3 (`monkeypatch.setattr(PT.Walk, "run", counted)` in `test_work.py` 892–903), N2 (`list(e.rows)` at `test_work.py` 1205 raises where `rows` is None), and N1's caveat in the docs: the seed is public, so a submitter can try circuit variants until the point misses an identity that holds on a thin set; state it as the premise it is. Leave S4 and N3.

In C (#9730): merge A's new head, and add the reader's S1 as two parametrized cases of the masked-attention test, "first-masked" and "computed-mask", each asserting the structured reading equals the word-by-word one; show me that `_nth(vals, b)` → `_nth(vals, 0)` (`work.py` 1057) and dropping the `uniform(...) is _MIXED` guard (1027) each now fail. Put the three trailer lines at the end of #9730's body: `Lane training`, `Refs TRN-145`, `Goal training/slowdown-from-submitted-circuit`.

Then, in this order:

- On your VM, named files only: the new tests, `test_work.py`, and `training_slowdown`'s `-k point_evaluates`. Each of the reader's probes refused; each of its mutations (M-A1, M-A2, M-C1, M-C2) failing a test, by name.
- The changed suites, fresh, at A's new head and at C's new head, one run each.
- The s-t20 reading again at C's new head with the runner now in the tree, since the claims change: one run (the last took six hours). The bodies' tables come from it.

The check slots are full tonight: launch each run once, no twin, and name any that waits.

By Sat noon PST, one message by inbox: A's and C's heads (40 hex), what each commit changes, the VM results, the run ids in flight, and one line on whether Sun 15 Nov 6 PM PST still holds for the goal (the s-t20 run's start decides it). When the suites and the s-t20 run are done, a second message with each run's id, `tree_sha` and own last line. READCIRC then reads only the delta, and I read and mark. Both bodies true of their new heads; no machine name in them. Do not merge `main` into either.

