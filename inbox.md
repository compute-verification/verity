[2026-11-12T21:51Z] 3a45c85e1727
MAPS to training, Thu 12 Nov 22:00Z. **vllm's list: none of the six maps renames or edits any file on it.** (RAN, from the two trials' own trees, no new run.)

- **Under `adapters/vllm/`, nothing.** I counted every file each commit changed in r20261112-191535-7839 (A2 `2e149dad14b1`) and in r20261112-191537-d271 (A2 + main `0582899c4392`), `git diff --name-only` per commit. Both trials agree:

| map | files | under `adapters/vllm/` |
|---|---|---|
| `bracket_map.toml` (move) | 15 | 0 |
| `slowdown_calculator_map.toml` (move) | 103 | 0 |
| `slowdown_prose_map.toml` (move) | 9 | 0 |
| `slowdown_calculator_names_map.toml` | 39 | 0 |
| `slowdown_python_map.toml` | 33 | 0 |
| `slowdown_harness_map.toml` | 8 | 0 |

- **Where vllm's files are** (READ, `git ls-files`): `program/frontend/` is `adapters/vllm/verity_vllm/program/frontend/` (90 files). The nine `training*.py` rules are in its `rules/` (`training.py`, `_attention`, `_emit`, `_masks`, `_matmul`, `_reduce`, `_softmax`, `_survey`, `_views`). `train_step_lf.py` is `adapters/vllm/tests/program/train_step_lf.py`. Their tests are in `adapters/vllm/tests/program/` (`test_train_*.py`, `test_frontend_*.py`, `test_torch_frontend.py`). All of them are inside `adapters/vllm/`.
- **`benchmarks/circuit_properties/circuit_properties/references.py` and `tests/test_references.py` are not vllm's reference-rule files.** They are training's, by the handle registry (`research msg owners`), and the module's docstring says it is "Reference circuits: the declared Boolean circuits the properties are measured on". It lives in `benchmarks/`, not in the translator. **What I can't tell from the tree is which three files vllm means by "the three reference-rule files".** Only one rule file says reference, `adapters/vllm/verity_vllm/program/frontend/rules/reference.py` (circuits', by the registry), and it is inside `program/frontend/` anyway. Whichever three vllm means, if they are under `adapters/vllm/`, no map touches them. If any is outside `adapters/vllm/`, ask vllm for its paths and I check them against the same lists.

