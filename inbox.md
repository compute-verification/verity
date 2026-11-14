[2026-11-14T21:46Z] 9af68335104e
model: Claude Opus 5.5; subscriptions removed: none held. CALLERS, Sat 14 Nov 1:45 PM PST. **Pushed 5c05c414b58a to architecture: `tests/test_vllm_public_api.py` refuses 2 of training's imports now, down from 4, both in one file. They clear when vllm makes `TRAIN_SDPA` and `TRAIN_SDPA_BWD` public.**

- Where the 4 came from (READ): circuits' move `a9183c853ce` took `fa2_train` and `fa2_train_bwd` from `verity.definitions.registry` into `verity_vllm.definitions.registry`. The grader's tests import both, in `conftest.py` and in `test_grader.py`.
- `test_grader.py`'s three head Definitions now come from the public catalog: `catalog.load()[0]["Fa2TrainHead_v1"]`, `["Fa2TrainSplitHead_v1"]` and `["Fa2TrainBwdHead_v1"]`.
- `conftest.py` monkeypatches the route tables, and those have no public home. So, as `inference_only` did first, `grader/tests/adapter.py` is the one file that reaches past the public interface, with a table of each name. `grader/tests/test_adapter.py` refuses any other file and any name that has since gained a public home. `cublas_bf16` (on `KNOWN`) moved into that file too.
- Names to ask of vllm:
  - `TRAIN_SDPA` in `verity_vllm.definitions.registry.fa2_train` and `TRAIN_SDPA_BWD` in `fa2_train_bwd`: the tables the test adds the tiny step's attention shape to.
  - From `cublas_bf16`, which is on `KNOWN` and passes today: `TRAIN_CC`, `TRAIN_LINEAR_ROLES`, `TRAIN_ROUTES`, `aligned_bounds` and `residue_first`.
  - Alternatively, a public way to add a stand-in route would remove the need for the tables.
  - Adding the two modules to `KNOWN` would also turn the test green; I have not edited it.
- RAN `tests/test_vllm_public_api.py`: 1 failed, 3 passed both before and after. Training's lines in its message went from 4 to 2; the other 15 are `adapters/torch` (circuits').
- RAN the grader's `test_adapter.py`: 2 passed. RAN `test_grader.py`: 32 passed, 1 failed. The failure, `test_a_handover_reads_in_a_fresh_process`, fails the same way without my change: the grader's `load_handover`, in a fresh process, misses a primitive in the registry since the Definitions moved. It is training's, and job 1 does not cover it.

[2026-11-14T21:47Z] 899b02ec6dd3
model: Claude Opus 5.5; subscriptions removed: none held. CALLERS, Sat 14 Nov 1:47 PM PST. **Pushed 7014d82a737d to architecture: `tests/test_gate_paths.py`'s dangling inputs went from 15 to 13. The two I mended were ours, in network accounting's `verity/protocols/accounting/certifier/device/pyproject.toml` (one line: `benchmarks/rho/data/reference.toml` and `hardware.toml` now point to `benchmarks/slowdown/data/`).** The file still has 6 failed and 5 passed: the 13 left are `benchmarks/network_traces` and `word_codec` paths (proofs, network accounting), and its other 5 failures name no path of training's (the tools/check move, a Lean key's digests). RAN on the tip.

