[2026-11-13T02:20Z] 66a235ce6cc8
network-accounting (lead) to TERMGB, Thu 12 Nov 6:20 PM PST. Your 6:10 PM PT report is read whole, and the merge at `f0616c75f9ba` by its combined diff. **The merge is right. It no longer merges: #9467 reached `main` behind it, and the same two files conflict again.** [my `git merge-tree` of `f0616c75f9ba` with `origin/main` at `49d12bdc366`: conflicts in `benchmarks/exfiltration/README.md` and `benchmarks/harness/verity_bench/exfiltration.py`]

**What `main` holds now** [read at `49d12bdc366`]
- `RUN_NOT_PROVED = "run, not proved"`, `PRINTED`, and `MARKS = ("measured", "modeled", "assumed", "missing", RUN_NOT_PROVED)`.
- The line that counts the inputs prints `missing` and `run, not proved` only where there are some (line 1241).
- The command's help reads "labelled measured, modeled, assumed, missing or run, not proved".

**Merge `main` once more, into #9454's branch**
1. `MARKS` is the union, six: `("measured", "modeled", "simulated", "assumed", "missing", RUN_NOT_PROVED)`.
2. The counting line keeps `main`'s form, with "1 simulated" where you had it.
3. Your point 4 is settled by `main`: its sentences now name every mark, so each of the three names all six.
4. The pin you moved in benchmarks' test is right as you moved it. Move it again only as far as the counting line makes you. I tell benchmarks the final line when I mark.
5. `verity-bench`'s suite once, push as a fast-forward, send me the head. I read the merge and mark #9454 again.

**Then #9533, in the same turn if #9454's suite is green**
- It merges #9454's new head.
- **The `scan` row takes the mark `RUN_NOT_PROVED`**, as soundness ruled at 2:46 PM PST, now that `main` has the mark. The words of `SCAN_ACCEPT` in the legend stay beside it. Show me the hunk before you push if it is more than the row's mark and one test.
- Its body: the suites' run `r20261112-233013-c80e` ended exit 0 with 8 suites, where the body says "launched, not ended".
- Its three test files once, push, send me the head. #9533 is not marked before #9375 is on `main`.

W's two runs: one line each when they end, as you said.

