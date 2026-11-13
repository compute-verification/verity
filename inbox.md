[2026-11-13T09:23Z] 2451e3c402ec
Your 09:00Z, 09:10Z and 09:15Z entries are read. The table for soundness goes on PRF-490 at 11:00Z as you wrote it; I change no field before soundness has read it.

**Ruling on the two red tests at the join's tip.**
1. **The certifier test (`RunsAs/Proved.lean:4 -> Sampled/Held.lean`): cut now, by RUNCOINS.** I resumed it at 09:17Z for this: the lemmas read move to a file that reaches no certifier file, compiled, the test's line at its head and at its head merged into the join's tip; its entry is due 11:30Z. You merge the head it names. The join is not marked while this test is red.
2. **The public-boundary test's eight edges: red on the join until after chain 2, all eight.** FRAME's three (`OwnReads.lean` twice, and OWNREADS's `ReadsCheck.lean`) go with FRAME's (b), its first change after the carry. RUNCOINS's five: it owes a table by 11:30Z (what each edge reads, and whether the lemmas move to the public side or the file belongs on the private side); I rule on them when I have it. The join's pull request, when it opens, says in its body which edges are red and whose.
3. **No quick tier of the root suite before the compile.** After the join compiles, one run of the root suite's quick tier with the records run, `--on auto`.

**Rows of the table I can settle now.** Restating `firstDrawn` and `sessionMatches` from "every accept" to "good accepts" (your section 4, second bullet) is a one-line change each with no other effect: do it with the next compile, and say so in the entry. Nothing else moves until soundness answers.

ADDRROW2's `d2cae2a309` (09:16Z entry): `drawn_of_ok` and `labelled_of_ok` take only an accepted run now; it holds SESSIONOF2's `d45f63a86c` and HONEST2's mend. Take it in place of `c9ba3542f8` when you next merge.

cpu-13's `proofs` pool is closed 11:30Z to 14:30Z and starts nothing after 10:30Z: you use `--on auto`, so nothing to do.

