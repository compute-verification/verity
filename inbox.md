[2026-11-12T22:12Z] 46408d65fab4
STEP (TRN-144) report, 22:12Z.

Draft: https://github.com/compute-verification/verity/pull/9514, unmarked, base `cursor/vllm-refactor-c1f1`, head `cursor/trn144-step-fused-attention-2556` at 3a13b1497f1e, linked to TRN-144 [RAN, `research queue open`; `gh pr view 9514`: isDraft true, that base, that head]. `open` took the non-main base. The body's first lines say BACKWARD's commits are #9417's (cb20f4171f41 by 0250976ab0). It lists only my own commits' six files (+579/−87), and GRADER's four files separately [READ, the body].

Head: 3a13b1497f is a `--no-ff` merge of GRADER's `cursor/trn149-grader-fused-attention-45e9` at bbc4af4a2d3c. Its tree is bbc4af4a2d's exactly [RAN, `git diff --quiet bbc4af4a2d HEAD`]. `adapters/`, `verity/` and `tests/` are unchanged since my c14944ed8a [RAN, `git diff --quiet`].

What passes:
- Grader and training_slowdown suites at the head, with `VERITY_GRADER_SLOWDOWN_CODE` set: 576 passed and 39 passed, none skipped; ruff clean [RAN, r20261112-211746-1937].
- Root suite `repository` at the head: 356 passed [RAN, r20261112-215348-3a57].
- The ten training test files with their slow cases, `test_stock` and `test_gemm_targets` under `VERITY_REGRESSION=1`, at c14944ed8a: 480 passed, 2 skipped (no vLLM); `tests/lint` rc 0 with bookkeeping warnings only, none in my files; `tests/test_vllm_public_api.py` 4 passed [RAN, r20261112-194903-bdcc].
- circuit-check on the four flash attention sidecars: 34 targets, 0 failed [RAN, r20261112-193755-1356].
- Gates, kernel against math path [RAN, r20261112-191237-18e5 and r20261112-210219-6d10]:
  - forward math / kernel: 8.40 at [2,2,16,32], 13.09 at [2,8,1024,64], 14.04 at H = 12, 16 and 24.
  - backward math / kernel: 5.65 at [2,2,16,32], 11.46 to 11.47 at H = 8 through 24.
  - kernel / count: 0.63 to 0.75 at T = 1024.
  - The math joint's peak reached 22.1 GB at [2,24,1024,64]. The full table is in the body.

What does not pass, and whose:
- `adapters/vllm/tests/program/test_frontend_adapter_imports.py::test_every_import_of_the_adapter_is_listed`, in the adapter quick tier at c14944ed8a [RAN, r20261112-202444-897c: 17,367 passed, 1 failed, 398 skipped, 7 xfailed; rerun by name on my VM, same failure].
  - The message: `{'verity_vllm.config.vllm_revision': ['export_compat.py']}` is not listed in `verity_vllm/program/frontend/__init__.py`'s docstring.
  - It is a merge interaction, not my code: the base adds the test, and `main`'s `export_compat.py` (9452a119f9, 9e0e82f5a3) adds the import; #9417's branch brought it here [READ, `git rev-parse` of the three files at the base, cb20f4171f41 and origin/main].
  - The fix is one docstring line in that `__init__.py`, which belongs to the refactor's owner and which the refactor needs when it merges `main`. I have not made it. Say if you want it on my line.
- The two grader failures I reported earlier are fixed by GRADER's merged commits [RAN, r20261112-211746-1937].
- The ratchet's 293 findings are `main`'s code that #9417's branch brought, none in my files [RAN locally, `ratchet_gate.py` at 4c9fd1e2c36b].

Queued: nothing. Every run above is done.

Harness answer: nothing under `benchmarks/training_slowdown` calls `torch.use_deterministic_algorithms` [READ, the tree; `run.py`'s `train()`]. A step traced from that job's settings is therefore refused at its flash backward (`side_input`, naming the fix).

Friction [RAN]: under `--cwd source`, run r20261112-194901-c38f sat waiting for a slot while holding the shared tree's Lean lock. That blocked my r20261112-194903-bdcc for about 35 minutes. I cancelled c38f and relaunched it with `--cwd clone`. A waiting run shouldn't hold the shared tree's lock.

