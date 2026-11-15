[2026-11-15T18:46Z] 248eba519a45
From STEP to the training lead, Sun 15 Nov 10:45 AM PST. Model: claude-opus-5-5-high. Subscriptions held: none. Your four facts are read and acted on.

PR1 is ready for you to open and mark. I have no tool that opens a PR (gh here is read-only), so it is a pushed branch and a body; the body follows at the end.
- Branch: `cursor/training-one-translator-4207`, head `3c14bde30a6` on `origin/main` (two commits). [ran: git ls-remote]
- verity-torch, `--fresh`: 703 passed at `0e991436aca` (r20261115-172632-f281), and 705 passed at PR2's head, which holds PR1 (r20261115-173414-cf5c). [ran]

PR2 is `cursor/training-translator-switch-28a9` at `5ba3566e695`, stacked on PR1. It switches the capture to `export_steps` and deletes `capture/trace.py`.
- At its head, `--fresh` passed: verity-torch 705, `benchmarks/training_slowdown` 803, the grader 36 with 3 skipped (r20261115-173414-cf5c). [ran]
- One entry, `s-t20-1.1x-adamw`, gives a byte-identical library row before and after. [ran, local]
- The full-library equality is still running: before (main) r20261115-173306-0110 and after (PR2) r20261115-173329-4bc4, both over an hour so far, 3 h timeout. I send PR2's body once they compare.

PR3 is `cursor/training-translator-one-copy-28a9` at `83425b43417`, reworked to your fact 4.
- A file outside `adapters/torch/` and outside tests counts as a second translator if it calls a graph tracer and also runs a backward or builds an optimizer, or if it builds a Definition named `Train*`.
- `EXCEPTION` holds only `registry/torch_step.py` (TrainStep_v1, with `numerics/torch_step.py` named in its reason), and the list only shrinks. The vLLM served-forward exports and `fa2_train_bwd`/`torch_train` don't match, so they aren't listed. [ran: local, 4 passed]
- On `main` and on PR1 alone, it flags `trace.py`. [ran]
- I left "owner circuits" out of the file: AGENTS.md lets only the skills name roles. Say if you want it in the test.
- Per fact 3, PR3 waits for tonight's move and needs recutting onto the moved tree. I won't rebase, so after the move I'll put the file on a new branch from the moved `main`. Root suite on `83425b43417`: r20261115-184508-b793, running. The earlier head `221255e0a65` passed 369 (r20261115-174032-c469). [ran]

For you to read on PR1:
- Between PR1 and PR2, the adapter's `train_trace` and the benchmark's `trace.py` are two copies. Land them close together, or together.
- The stubs' `gpu` and `cc` parameters are dropped; nothing passed them.
- The `CaptureRefused` text no longer names `routes/lora.py`.

PR1 body:

The PyTorch adapter traces and derives consecutive training steps: `export_steps` and `steps_program` in `verity_torch.frontend.train_step` stop raising `NotImplementedError`.

- `export_steps(step_fns, model, example_batch, *, measured, optimizers=(), mid_run=False, device=None)` traces the steps as one graph with `make_fx` under `FakeTensorMode`, on a fake copy of the model and its optimizers, so the real ones are untouched. Each node carries the step it belongs to (`te_step`), and an integer batch's placeholders are named `tokens_N`. It returns the graph, its fake inputs and a `StepLayout`. The layout gains each attribute's kind and fqn (`attrs`), the optimizer state (`state`), `measured` and `mid_run`.
- `steps_program(...)` lifts the parameters and buffers to placeholders and drops the profiler nodes. It then re-traces the graph functionalized and turns each in-place write to a parameter or buffer into an output. It derives with `training_rules()` and records the layout in the Program's `train_step` annotation. An op with no rule (ReLU's `threshold_backward`) gives `Unsupported`, as `export_step` does.
- `verity_torch.frontend.train_trace` holds the tracing the training benchmark's capture used. It is moved verbatim: the step, backward and attention markers (`STEP`, `BACKWARD`, `ATTENTION`), and the refusals as `CaptureRefused` with the reason, op and user frame (data-dependent control flow, `deepcopy`, an op the route replaced, an attention call that returns in a dtype other than its marker's). The benchmark switches to it in the next PR.
- The stubs' `gpu` and `cc` parameters are dropped. Nothing passed them, and the trace doesn't read them.

Tests in `adapters/torch/tests/frontend/test_train_steps.py`:
- the trace runs on a copy and marks each step;
- token placeholders;
- the control-flow and `deepcopy` refusals at the caller's line;
- a `measured` out of range;
- the Program equals torch's two SGD steps bit for bit;
- `mid_run` gives no theta;
- an unsupported op;
- the attention markers, and the dtype refusal.

Checked:
- `uv run tools/check/suites.py verity-torch --fresh`: 703 passed at 0e991436aca (r20261115-172632-f281), and 705 passed at the next PR's head, which holds this one (r20261115-173414-cf5c).
- The one test over 5 s without the `slow` mark is `test_train_step.py::test_tiny_bf16_step`, which predates this PR.

Lane training
Refs TRN-66
Goal training/one-pytorch-translator

