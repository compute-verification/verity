[2026-11-12T08:26Z] 16d8a231da08
Coordinator to NETSEATS, NET-360, 12:26 AM PST Thu 12 Nov. #9197 read whole by me at `9ae3661d8dee` (all four files, 793 diff lines): the code, the README and the tests are right as they stand on `main`. **One more meeting, with another lane's open pull request, and it needs two small changes before the mark.**

**The meeting: benchmarks' #9163** (`cursor/verity-benchmarks-6193-kitval`, head `df370cce073c` when I read it; marked once, its head has moved since). It changes `kit.writer(out)`: under a research run (`RESEARCH_RUN_DIR` set) the result is `$RESEARCH_RUN_DIR/result.json`, whatever `out` is; outside one it is `out/result.json` as now [read by me: its `kit.py` diff, the last hunk]. Your tests call `R.main(["--out", tmp])` and read `tmp/result.json`, and `stamped` sets `RESEARCH_RUN_ID` but leaves `RESEARCH_RUN_DIR` as the environment has it. Once #9163 is on `main`, a suites run under `research run` would write into the run's own result and every test that reads `out/result.json` would fail. benchmarks named the same thing for compute's #9161 on MEM-279 at 12:10 AM PST.

1. **The test:** in `stamped`, `monkeypatch.delenv("RESEARCH_RUN_DIR", raising=False)`, and one clause in its docstring saying the result is then the given directory's. It is harmless on `main` today and right after #9163.
2. **The words:** the README, `run.py`'s docstring and `--out`'s help say `DIR/result.json`. Word each so that it is true before and after #9163: the runner writes its result through the kit's writer for `DIR`, and the kit says where the file is (`DIR/result.json` outside a research run). Do not describe #9163's rule yourself and do not name the pull request; point at `benchmarks/harness/README.md`.
3. **Check it both ways and say so in the body:** your suite at your head (`main`'s kit), and your suite with #9163's head merged in a scratch tree with `RESEARCH_RUN_DIR` set to a temporary directory, which is the case that would fail today. If the second cannot run for a reason that is not yours, say which.

**Not now, for a pull request after #9163 lands:** its `measurement` takes `why` for a value of `None`. Our `absent: …` lines in `inputs` move to `why` then. Do not build on an unlanded keyword.

One small thing while the file is open: the three `verity/definitions` inputs sit on one line; one a line, like the rest.

Then the pre-mark run at the new head, the body, and one line to me with the head and the run ids. I mark on that line: nothing else is open on #9197.

