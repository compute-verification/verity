[2026-11-12T08:43Z] bd5466883a45
NETSEATS to coordinator, NET-360, #9197: your three points and the small one are in. Head `368902c124843f858cf81bdba69c561c3fb29314`, pushed; body rewritten at it; still unmarked.

Commits on top of `9ae3661d8dee`, nothing else rides: `8fe8221491` (`stamped` does `monkeypatch.delenv("RESEARCH_RUN_DIR", raising=False)`, with the clause in its docstring), `d81f5391b1` (README, `run.py`'s docstring and `--out`'s help: the result goes through the kit's writer for `DIR`, and the kit says where the file is, `DIR/result.json` outside a research run, pointing at `benchmarks/harness/README.md`; the other change's rule is not described and the pull request not named), `368902c124` (the three `verity/definitions` inputs one a line).

Runs, all [ran]:
- Pre-mark run at this head: `r20261112-083105-4a24`, 4 suites, 4 passed, 0 failed (benchmarks/certifier_term 14, verity-graph 208, verity-network-traces 540, repository 310). The one at `9ae3661d8dee`, `r20261112-081506-d71d`, never got a slot and I cancelled it as superseded.
- `main`'s kit, my head, on the VM under the read guard: 14 passed; the runner's existing test under its guard: 20 passed; ruff and the ratchet gate pass; 28 of 28 wrong runners fail the suite.
- #9163's kit: scratch merge of my head into `df370cce073c` (own `uv sync`, not committed, merge-tree names no file), `RESEARCH_RUN_DIR` set to a temporary directory: suite under its guard, 14 passed, nothing written in that directory; unset, 14 passed.
- Scratch merge into lean's tip, now `a88eee998fa2`: 14 passed and 20 passed under the guards. Into `main`, now `19fb19a21e40`: 14 passed, 20 passed, the five root tests 53 passed. merge-tree against all three names no file.

One thing you should know, because it differs from what your message expected: **a suites run was already safe; the case that fails is pytest run directly.** `tools/verity/check/suites.py` gives a suite's tests no `RESEARCH_*` variable of the run it is in (`RUNNER_PREFIX`, in `run`) [read]. So in the scratch merge with #9163's kit and `RESEARCH_RUN_DIR` set, the test file of `9ae3661d8dee` passes through `suites.py` too, 14 passed [ran]. Run as `python -m pytest tests/test_result.py` there, it fails, 13 failed and 1 passed, with `result.json` written into that directory; this head's passes, 14 passed, nothing written there [ran]. The fixture's line is right and stays; the body says which case it guards. If benchmarks told compute the same about a suites run under `research run`, the same holds for #9161.

Also: `research queue edit --body-file` replaces the body verbatim, so the `Refs NET-360` line `open` appended was gone after my edit; I put it back as the body's last line and read it back. `Goal:` is still yours.

Not done, as you said: nothing built on #9163's `why`; the `absent: …` lines stay in `inputs` until it lands.

