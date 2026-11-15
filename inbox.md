[2026-11-15T19:18Z] 45df786a3a61
From STEP to the training lead, Sun 15 Nov 11:18 AM PST. The figures are equal: PR2 is ready for you to open, stacked on PR1.

- Equality [ran]: `library.py --out` at the base `24fd3ecd634` (r20261115-173306-0110) and at PR2's head `5ba3566e695` (r20261115-173329-4bc4). Both runs succeeded.
  - Over all 64 entries, the two `library.json` files are byte-identical (same sha256), with the same rows in the same order.
  - Correction to my sizing: the library has 64 entries, not 65. [ran: git ls-files of `attacks/*.toml` on main and on the branch]
- PR3's root suite on `83425b43417` passed: `repository`, 369 tests (r20261115-184508-b793). [ran]

Branches [ran: pushed]:
- PR1: `cursor/training-one-translator-4207` at `3c14bde30a6`.
- PR2: `cursor/training-translator-switch-28a9` at `5ba3566e695`.
- PR3: `cursor/training-translator-one-copy-28a9` at `83425b43417`. It waits for tonight's move, and I'll recut it onto the moved tree.

Not done:
- No PR is opened. I have no tool for it, and gh is read-only.
- PR3 is not yet on the moved tree.
- "owner circuits" is not in the test file (my last message says why).

PR2 body:

The training benchmark's capture traces its step through the PyTorch adapter, and the benchmark's own translator, `capture/trace.py`, is deleted.

- `capture/steps.py` keeps what the benchmark measures around the trace: `Capture`, the token dims, the axis resizing, the fresh optimizers, `capture_step` and `capture_steps`, all verbatim. Its one change is that `_trace` calls `verity_torch.frontend.train_step.export_steps` and wraps the layout's `attrs` and `state` in a `Capture`.
- `structure.py` reads the markers (`ATTENTION`, `BACKWARD`) and `CaptureRefused` from `verity_torch.frontend.train_trace`'s `__all__`.
- `structure.py`, `products.py` and `circuit.py` price the traced graph, and they stay in the benchmark.
- `run.py` imports the capture before the job's import screen is built. The adapter's trace takes over torch's dispatch, which the screen refuses to a route, so it must be imported first. What the trace runs is still the tree's source.
- `test_products.py` drops the dtype-refusal test, which the adapter's `test_train_steps.py` now holds. It patches the adapter's `_marked_attention` in place of the deleted module's.
- `VERSION` and `products.CAPTURE` are unchanged, since what is measured is unchanged.

Checked:
- Every library entry's figures are equal before and after. `library.py --out` ran at the base 24fd3ecd634 (r20261115-173306-0110) and at this head 5ba3566e695 (r20261115-173329-4bc4). Over all 64 entries, the two `library.json` files are byte-identical (same sha256), with the same rows in the same order.
- `uv run tools/check/suites.py verity-torch benchmarks/training_slowdown benchmarks/training_slowdown/grader --fresh` at 5ba3566e695: 705, 803 and 36 passed, with 3 skipped (r20261115-173414-cf5c).

Stacked on `cursor/training-one-translator-4207`.

Lane training
Refs TRN-66
Goal training/one-pytorch-translator

PR3 body:

A root test, `tests/test_one_translator.py`, fails on a second translator from PyTorch to circuits.

- No tracked Python file outside `adapters/torch/` calls a PyTorch graph tracer (`TRACERS`: `make_fx`, `functionalize`, `torch.export`, `FakeTensorMode`, `symbolic_trace`, `jit.trace`, dynamo's export and the aot entry points). Each call is resolved through the names the file's imports bind, aliases included.
- The exceptions are `ALLOWED`, each with why:
  - `vllm_meta.py`, the vLLM adapter's export of a served model's forward, which the PyTorch adapter's derivation reads;
  - eight vLLM adapter tests that export a toy served forward;
  - `benchmarks/training_slowdown/tests/test_flops.py`, which traces one op to check the counter's dtype log.
- `ALLOWED` only shrinks: an entry whose file no longer traces fails the test.
- The training benchmark's sources outside its tests read only names that the adapter's modules list in `__all__`, whether imported or read off an imported module.
- A planted second copy (an aliased `make_fx`, `torch.export.export`, an aliased `FakeTensorMode` module) is found, `torch.compile` isn't, and so are private reads.

On `main` and on the first PR alone, the test fails on `benchmarks/training_slowdown/training_slowdown/capture/trace.py` (its `FakeTensorMode` and `make_fx`). On the second PR, which deletes that file, it passes.

The test is AST only. A tracer reached through `getattr` or an import by string is not seen; the docstring says so.

Checked:
- `uv run tools/check/suites.py repository benchmarks/training_slowdown benchmarks/training_slowdown/grader --fresh` at 221255e0a65: 369, 803 and 36 passed, with 3 skipped (r20261115-174032-c469).
- Locally, the new file passes its 4 tests and ruff reports nothing.

Stacked on the second PR (`cursor/training-translator-switch-28a9`). The root `tests/` move on Monday will carry this file with the rest.

Lane training
Refs TRN-66
Goal training/one-pytorch-translator

[2026-11-15T19:19Z] 2e7d6d351365
From STEP to the training lead, Sun 15 Nov 11:19 AM PST. Correction: the PR3 body in my last message was the old one. Use this one, which matches 83425b43417:

A root test, `tests/test_one_translator.py`, fails on a second translator from a PyTorch training step to a circuit.

- Outside `adapters/torch/` and outside tests, no tracked source may trace a training step. That means calling a PyTorch graph tracer (`TRACERS`: `make_fx`, `functionalize`, `torch.export`, `FakeTensorMode`, `symbolic_trace`, `jit.trace`, dynamo's export, the aot entry points) in a file that also runs a backward or builds an optimizer. Calls are resolved through the names the file's imports bind, aliases included.
- Nor may it describe a step by hand, that is, build a Definition whose name starts with `Train`.
- `EXCEPTION` names the one file that does either today: `adapters/vllm/verity_vllm/definitions/registry/torch_step.py` (`TrainStep_v1`, with its numerics model `numerics/torch_step.py`). It only shrinks: an entry that no longer matches fails.
- The vLLM adapter's export of a served forward runs no backward, and the kernels' `fa2_train_bwd.py` and `torch_train.py` define no `Train*` Definition. Neither matches.
- The training benchmark's sources outside its tests read only names that the adapter's modules list in `__all__`, whether imported or read off an imported module.
- Planted copies are found: an aliased `make_fx` over a step with a backward, `torch.export.export` beside an optimizer, an aliased `FakeTensorMode` module beside `autograd.grad`, and a `Train*` Definition. So are private reads. A forward-only export, `torch.compile` and a `CacheStep` Definition are not flagged.

On `main` and on the first PR alone, the test fails on `benchmarks/training_slowdown/training_slowdown/capture/trace.py` (its `FakeTensorMode` and `make_fx`). On the second PR, which deletes that file, it passes.

The test is AST only. It doesn't see a tracer reached through `getattr` or an import by string, a step traced in a file that only receives its backward, or a step's Definition under another name. The docstring says so.

Checked:
- `uv run tools/check/suites.py repository --fresh` at 83425b43417: 369 passed (r20261115-184508-b793).
- With `benchmarks/training_slowdown` and its grader at the earlier head 221255e0a65: 369, 803 and 36 passed, with 3 skipped (r20261115-174032-c469).

Stacked on the second PR (`cursor/training-translator-switch-28a9`). It is recut onto the tree after the root `tests/` move.

Lane training
Refs TRN-66
Goal training/one-pytorch-translator

