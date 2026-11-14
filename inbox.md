[2026-11-14T01:58Z] 8ecc221a3a43
## Coordinator, Sat 14 Nov 01:58Z: #9854: option 1, one third-path compile, the body says what the run showed

From the proofs coordinator, Sat 14 Nov 01:58Z. Your 01:50Z note is read whole. Stopping was right. The ruling: OPTION 1, with one small run added, and option 2 folded into Tuesday's H200 run. Not option 3.

1. THE MOVE STANDS. #9854 keeps its head `786a282da3efa10fa1dc148e057a4fb9cb1f4edf`; no change to any kernel or test text.
2. ONE RUN, on a CPU node with nvcc, to turn your [guess] into a [run]: compile `sha512_forms.cu` (or `row_commit_cases.cu`) from a third path, at sm_120a and sm_90a, and read the first 8 digits of its anonymous namespace's name. A third value says nvcc makes them from the path it is given; the same value as base or head says it does not, and then stop and write me. No other artifact is rebuilt.
3. THE BODY, rewritten to what the run showed, in this order:
   - the count as the run printed it: 229 same, 12 refused alike, 4 different across the move;
   - the four by name, and where they differ: only in the first 8 digits of the anonymous namespace's name, with the artifact `art:2f16a990c771a125661a64dc9dc601330d045c25c17b4e46d549507fc7f9c08e`; with those digits zeroed, base and head are one file, byte for byte;
   - "no kernel's text changes" stays; "the same machine code" is said only of the nine cubins the six runners build and of every source outside the moved directory. The words "every digest the same" do not appear;
   - item 2's result, tagged [run];
   - the run that passed at the head is `r20261114-005018-e34f`; the cancelled `c446` is named nowhere;
   - a "Not established" list: the four device tests of the moved suite on a GPU at the head (skipped on every CPU node), and the six cubins compared in no build, each with its reason.
4. NO GPU JOB NOW. The four device tests run in Tuesday's H200 run, which the mark already waits for. Put the exact pytest selection for them in the body, so whoever launches that run has it.
5. Then the note with the slug `ready-9854`, saying in its first line that it is ready to mark only after Tuesday's device tests pass. I do not mark before then.

Your timer may stay as it is until the ready note; cancel it with that note.

[2026-11-14T02:31Z] 3648f506b6f0
## Coordinator, Sat 14 Nov 02:31Z: the public export is cancelled

**The public export is cancelled** (Daniel, through top, Fri 5:01 PM PST, on TOP-239): there is no export; `verity` is the public repository whole. Top deletes the export tool, the boundary files and their tests; its pull request opens by Sat 14 Nov 10 AM PST.

- Put no more time into `public_boundary.toml` or its test. Whatever #9854 changes there is moot.
- If top's deletion is on `main` before #9854 lands, drop that hunk when you merge `main` (take the deletion) and say so in the body. If #9854 lands first, do nothing.
- Nothing else in my 01:58Z word changes: option 1, the third-path compile, the body's exact words, the four device tests in Tuesday's H200 run, then `ready-9854`.

