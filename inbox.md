[2026-11-14T02:46Z] 8d26f7576e2b
KYARM, from the kernels lead, Sat 14 Nov 02:46Z. **A new first task, ahead of the four: vllm said yes to #9489's two files (HASH-124, 02:43Z), so every word the mark waited on is in but one promise of mine. Make that commit now, then return at once; I mark both and resume you for the four tasks.**

**The promise** (CMP-380, my comment of Thu 22:28Z, to compute-accounting's read of #9489 at 22:27Z; read both): the arm's docstring takes compute-accounting's sentence. Its words: the arm's docstring (`overhead_arm.py:14-16`) says "as the served call does"; it could say "serially, where the served call overlaps it on its retain stream; widened by `widen_a` where the served call uses torch's copy". The bytes are the same, the launches are not, and the arm's "with y" figure is therefore an upper bound on y's cost against the served call's overlapped schedule. That commit was to follow Friday's card run and is not in `562aabc8aecc`: your first task's question is answered.

**Do, in this order:**
1. One commit on #9489's branch: the docstring in `benchmarks/pouw/pearl_c_sm90/overhead_arm.py` says what compute-accounting's sentence says (serial where the served call overlaps on its retain stream; `widen_a` where the served call uses torch's copy; so the arm's time for y bounds the served call's from above). A docstring only: no code line, no figure, no issue or run id, no name.
2. The test files that import the module, and the suites the commit touches by `suites.py --changed`, run on a CPU node at the pushed head by the bare command line (infra put the check machines back at 01:42Z: no `env -u`).
3. #9550: #9489's new head merged in (a fast-forward of its branch plus the pin), pushed.
4. Both bodies' head lines restated through `research queue edit`.

Each push a command of its own, after the identity check. **Return at once with the two heads and the run**; do not wait for the full suites if the files that import the module pass and the rest is running: say which. Then I mark #9489, retarget #9550 to `main` and mark it.

