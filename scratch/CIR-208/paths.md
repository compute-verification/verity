# CIR-208: every path string the superseded-versions check holds, and the move's row for each

Made at #8463's head `b1409cffb06f0e1b46ca8ee97fc85c28c412523b` (`main` `8c11a1d5ab` merged). The maps are read on
`origin/cursor/top-moves-c-b396` at `680f1541f8c3`: `tools/move/definitions_kept_map.toml`, `definitions_merge_map.toml`,
`definitions_served_map.toml` and `definitions_lake_map.toml`, which run in that order. "Row" is the `[paths]` row whose old
side is the longest prefix of the string. "None" means no row of the four maps holds it, so the move leaves it as it is.

## Run, checked, and only read

- **Run.** `r20261113-025528-cc6a` took this head merged with `680f1541f8c3` (scratch commit `2ef2f56688`) and ran the kept,
  merge and served maps with `tools/move/layout.py --map M --no-lock`, one commit each. On the moved tree every one of the
  list's 239 `file` strings names a file; the test file passes (`3 passed`); `--prune` keeps the same 239 entries and
  changes only the order of 25 lines; and the test file's graph task passes with `undeclared_reads: []`. The maps rewrote
  207 of the list's lines and no line of the test file or of `tests/README.md`. The job is `scratch/CIR-208/cross_move.sh`.
- **Checked against that run.** The row computed below for each of the list's 64 files gives the path that run wrote, for
  all 64 (`make_paths.py` stops otherwise).
- **Only read.** The Lake packages' map: none of its rows holds a string here. `cross_move.sh` runs it as a second phase
  from its next run on; `inflight.md` names that run.

## The test file, `tools/verity/circuit_check/tests/test_superseded_versions.py`

No path in it is put together from parts: each is one whole string from the repository's root, or is taken from `__file__`.

| line | string | what it is | row |
|---|---|---|---|
| 15, 43 | `tests/lint/allowlists/superseded_versions.json` | the list: named in the docstring, and `LIST = REPO / "..."`, one whole string joined to the root | none |
| 22, 50 | `adapters/vllm/manifests/records.json` | prose, in the docstring and in the failure's text (`FIX`); the test never opens it | none |
| 28 | `adapters/vllm/tools/ratchet_gate.py` | prose, in the docstring | none |
| 31, 44 | `tools/verity/circuit_check/tests/test_superseded_versions.py` | the docstring's command line, and `RULE`, the text of the list's `rule` | none |
| 42 | `Path(__file__).resolve().parents[4]` | `REPO`, the root: four directories above the test file, so it holds while the file keeps its depth | none: no map moves `tools/verity/circuit_check/` |
| 52 | `Path(__file__).resolve().relative_to(REPO)` | `PRUNE`, the command the failure prints: the file's own place | follows the file |
| 58 | `circuit_check` (a module) | the fresh interpreter imports `sidecars` and `targets` from it | none |
| 59 | `verity.circuits.defs` (a module) | the fresh interpreter's `REGISTRY` | none: the kept and merge maps move modules into `verity.circuits` and rename none that is there |

The registries' loader, `circuit_check.targets.load_registries`, is not this change's, and it names two modules the served
map renames (`[modules]`): `verity.definitions.registry` to `verity_vllm.definitions.registry` and
`verity.definitions.numerics` to `verity_vllm.definitions.numerics`. The test calls the loader and names neither.

## The suite's declaration, the README and the list's `rule`

| where | string | what it is | row |
|---|---|---|---|
| `tools/verity/circuit_check/pyproject.toml` 205 | `tests/lint/allowlists/superseded_versions.json` | the suite's `inputs`: the one path this change adds | none |
| the same, 227 and 256 | `tests/test_superseded_versions.py` | the test's name in two `[[tool.verity.tests.reads]]` groups, from the suite's directory | none |
| `tests/README.md` 35 | `lint/allowlists/superseded_versions.json` | prose, from `tests/` | none |
| `tests/README.md` 36 | `tools/verity/circuit_check/tests/test_superseded_versions.py` | prose | none |
| the list, line 2 | `tools/verity/circuit_check/tests/test_superseded_versions.py` | inside its `rule` text | none |

The two reads groups the test's name joins are the tree's own, shared with the other tests that load the registries;
this change adds no path to either. The maps rewrite their paths for all of those tests at once.

## The list's `file` strings: 239 entries in 64 files

| map | row | files | entries |
|---|---|---|---|
| `definitions_kept_map.toml` | `"verity/definitions/ops/" = "verity/circuits/ops/"` | 6 | 24 |
| `definitions_served_map.toml` | `"verity/definitions/registry/" = "adapters/vllm/verity_vllm/definitions/registry/"` | 41 | 183 |
| none | none: the file stays where it is | 17 | 32 |

After the move the entries of the moved files are out of the list's order (by file, then name, then version), which the
test doesn't read. `--prune` writes the order again and changes nothing else.

### `definitions_kept_map.toml`, row `verity/definitions/ops/`: 6 files

| file in the list | entries | after the move |
|---|---|---|
| `verity/definitions/ops/boolean/attention.py` | 5 | `verity/circuits/ops/boolean/attention.py` |
| `verity/definitions/ops/boolean/elementwise.py` | 4 | `verity/circuits/ops/boolean/elementwise.py` |
| `verity/definitions/ops/boolean/gemm.py` | 4 | `verity/circuits/ops/boolean/gemm.py` |
| `verity/definitions/ops/boolean/mufu.py` | 1 | `verity/circuits/ops/boolean/mufu.py` |
| `verity/definitions/ops/fp32.py` | 9 | `verity/circuits/ops/fp32.py` |
| `verity/definitions/ops/mufu.py` | 1 | `verity/circuits/ops/mufu.py` |

### `definitions_served_map.toml`, row `verity/definitions/registry/`: 41 files

| file in the list | entries | after the move |
|---|---|---|
| `verity/definitions/registry/b1.py` | 24 | `adapters/vllm/verity_vllm/definitions/registry/b1.py` |
| `verity/definitions/registry/boolean_attention.py` | 21 | `adapters/vllm/verity_vllm/definitions/registry/boolean_attention.py` |
| `verity/definitions/registry/boolean_dense.py` | 4 | `adapters/vllm/verity_vllm/definitions/registry/boolean_dense.py` |
| `verity/definitions/registry/boolean_dsa.py` | 2 | `adapters/vllm/verity_vllm/definitions/registry/boolean_dsa.py` |
| `verity/definitions/registry/boolean_dsa_norm.py` | 3 | `adapters/vllm/verity_vllm/definitions/registry/boolean_dsa_norm.py` |
| `verity/definitions/registry/boolean_dsa_topk_mask.py` | 3 | `adapters/vllm/verity_vllm/definitions/registry/boolean_dsa_topk_mask.py` |
| `verity/definitions/registry/boolean_fp8.py` | 6 | `adapters/vllm/verity_vllm/definitions/registry/boolean_fp8.py` |
| `verity/definitions/registry/boolean_fp8_moe.py` | 4 | `adapters/vllm/verity_vllm/definitions/registry/boolean_fp8_moe.py` |
| `verity/definitions/registry/boolean_int8.py` | 2 | `adapters/vllm/verity_vllm/definitions/registry/boolean_int8.py` |
| `verity/definitions/registry/boolean_moe.py` | 1 | `adapters/vllm/verity_vllm/definitions/registry/boolean_moe.py` |
| `verity/definitions/registry/boolean_norms.py` | 9 | `adapters/vllm/verity_vllm/definitions/registry/boolean_norms.py` |
| `verity/definitions/registry/boolean_scalar.py` | 2 | `adapters/vllm/verity_vllm/definitions/registry/boolean_scalar.py` |
| `verity/definitions/registry/boolean_silu.py` | 2 | `adapters/vllm/verity_vllm/definitions/registry/boolean_silu.py` |
| `verity/definitions/registry/boolean_softcap.py` | 3 | `adapters/vllm/verity_vllm/definitions/registry/boolean_softcap.py` |
| `verity/definitions/registry/boolean_sparse_mla.py` | 1 | `adapters/vllm/verity_vllm/definitions/registry/boolean_sparse_mla.py` |
| `verity/definitions/registry/cublas_bf16.py` | 4 | `adapters/vllm/verity_vllm/definitions/registry/cublas_bf16.py` |
| `verity/definitions/registry/cuda_rmsnorm.py` | 1 | `adapters/vllm/verity_vllm/definitions/registry/cuda_rmsnorm.py` |
| `verity/definitions/registry/dead_free.py` | 7 | `adapters/vllm/verity_vllm/definitions/registry/dead_free.py` |
| `verity/definitions/registry/dense.py` | 9 | `adapters/vllm/verity_vllm/definitions/registry/dense.py` |
| `verity/definitions/registry/dsa.py` | 2 | `adapters/vllm/verity_vllm/definitions/registry/dsa.py` |
| `verity/definitions/registry/dsa_topk_mask_cut.py` | 3 | `adapters/vllm/verity_vllm/definitions/registry/dsa_topk_mask_cut.py` |
| `verity/definitions/registry/dsa_topk_reads.py` | 2 | `adapters/vllm/verity_vllm/definitions/registry/dsa_topk_reads.py` |
| `verity/definitions/registry/fa2_check_inf.py` | 6 | `adapters/vllm/verity_vllm/definitions/registry/fa2_check_inf.py` |
| `verity/definitions/registry/fa2_softcap.py` | 3 | `adapters/vllm/verity_vllm/definitions/registry/fa2_softcap.py` |
| `verity/definitions/registry/fa3_check_inf.py` | 3 | `adapters/vllm/verity_vllm/definitions/registry/fa3_check_inf.py` |
| `verity/definitions/registry/fp8.py` | 6 | `adapters/vllm/verity_vllm/definitions/registry/fp8.py` |
| `verity/definitions/registry/fp8_moe.py` | 4 | `adapters/vllm/verity_vllm/definitions/registry/fp8_moe.py` |
| `verity/definitions/registry/gemm_bias.py` | 2 | `adapters/vllm/verity_vllm/definitions/registry/gemm_bias.py` |
| `verity/definitions/registry/gumbel_boolean.py` | 1 | `adapters/vllm/verity_vllm/definitions/registry/gumbel_boolean.py` |
| `verity/definitions/registry/int8_exact.py` | 5 | `adapters/vllm/verity_vllm/definitions/registry/int8_exact.py` |
| `verity/definitions/registry/io_shell.py` | 1 | `adapters/vllm/verity_vllm/definitions/registry/io_shell.py` |
| `verity/definitions/registry/layer_norm.py` | 1 | `adapters/vllm/verity_vllm/definitions/registry/layer_norm.py` |
| `verity/definitions/registry/moe.py` | 5 | `adapters/vllm/verity_vllm/definitions/registry/moe.py` |
| `verity/definitions/registry/prims.py` | 1 | `adapters/vllm/verity_vllm/definitions/registry/prims.py` |
| `verity/definitions/registry/ref_prims.py` | 5 | `adapters/vllm/verity_vllm/definitions/registry/ref_prims.py` |
| `verity/definitions/registry/sampling.py` | 6 | `adapters/vllm/verity_vllm/definitions/registry/sampling.py` |
| `verity/definitions/registry/serve3.py` | 1 | `adapters/vllm/verity_vllm/definitions/registry/serve3.py` |
| `verity/definitions/registry/topp_word_gates.py` | 2 | `adapters/vllm/verity_vllm/definitions/registry/topp_word_gates.py` |
| `verity/definitions/registry/topp_words.py` | 3 | `adapters/vllm/verity_vllm/definitions/registry/topp_words.py` |
| `verity/definitions/registry/torch_embedding.py` | 1 | `adapters/vllm/verity_vllm/definitions/registry/torch_embedding.py` |
| `verity/definitions/registry/value_inputs.py` | 12 | `adapters/vllm/verity_vllm/definitions/registry/value_inputs.py` |

### No row: 17 files

| file in the list | entries | after the move |
|---|---|---|
| `adapters/vllm/verity_vllm/program/frontend/seed_rule.py` | 1 | the same |
| `verity/protocols/certifier/prover/cache_step_v2.py` | 1 | the same |
| `verity/protocols/certifier/prover/compile_slice_v2.py` | 1 | the same |
| `verity/protocols/certifier/prover/compile_slice_v3.py` | 1 | the same |
| `verity/protocols/certifier/prover/compile_slice_v4.py` | 3 | the same |
| `verity/protocols/certifier/prover/compile_slice_v5.py` | 1 | the same |
| `verity/protocols/certifier/prover/compile_slice_v6.py` | 1 | the same |
| `verity/protocols/certifier/verifier/arith_encode.py` | 1 | the same |
| `verity/protocols/certifier/verifier/cache_step.py` | 1 | the same |
| `verity/protocols/certifier/verifier/charge_chain.py` | 1 | the same |
| `verity/protocols/certifier/verifier/compile_slice.py` | 2 | the same |
| `verity/protocols/poub/verifier/commitments/gates/block_check.py` | 2 | the same |
| `verity/protocols/pouw/prover/circuit/cap.py` | 4 | the same |
| `verity/protocols/pouw/prover/circuit/h1tile.py` | 4 | the same |
| `verity/protocols/pouw/prover/circuit/h1zy.py` | 4 | the same |
| `verity/protocols/pouw/prover/circuit/live.py` | 1 | the same |
| `verity/protocols/pouw/prover/circuit/segment.py` | 3 | the same |

