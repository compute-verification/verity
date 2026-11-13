For @proofs and @vllm: this sits on the long row's switch ([#9318](https://github.com/compute-verification/verity/pull/9318)) and shows its commits until that lands. Its own commits serve the DSA norm's `_v3` parts. It changes proofs' files: `boolean_dsa_norm.py` and the three GLM-5.3 partition files, which move in this PR together with the ids, as proofs asked. It changes vllm's files: `test_served_unit_rows.py`'s pin, the norm-split test, the description-size record and its allowlist. It changes circuits' own norm tests, the GPU capture tool and one table of circuit-check's. It moves served statements: the norm's templates in step 16's partition and both prefill partitions. `STATEMENTS` is re-pinned from the test's own output on the merged tree, and the count stays main's 38. One revert undoes it: no `_v2` Definition, binding or pin is removed, and the partition files' caps are unchanged.

The head, `a62164a32a`, holds `main` at `1e4d5c993c`, which has #9375, through #9318's head `05dc686227`. It is the earlier head `3da6c052b5` merged with that head of #9318's, a clean merge. Main changed none of this change's own files since `15662e7c40`: `boolean_dsa_norm.py`, `targets.py`, the norm's four test files, the GPU tool and the three one-stage partition files. The merge moves no pin, no generated file and no `STATEMENTS` (below).

Lane circuits
Refs CIR-251
Goal circuits/programs-not-unrolled

## What changes

- `verity/definitions/registry/boolean_dsa_norm.py`: the three `opt_in = True` lines go (`DsaRMSNormSumSq_v3`, `DsaRMSNormApply_v3`, `DsaRMSNorm_v3`), with the docstring lines that described them. `boolean_version` now gives the `_v3`s.
- `benchmarks/one_stage/glm53_prefill_tp8_step0.json`, `glm53_prefill_tp8_step0_core.json` and `glm53_all_families_tp8_step16.json`: `templates` and `chunk_cap` name `DsaRMSNormSumSq_v3{N=512,…}`, `{N=2048,…}` (in the full prefill file only) and `DsaRMSNormApply_v3{G=64}`. The GEMM entries PRF-588 renames are left as they are.
- The caps are unchanged: prefill 8,192 (N=512), 2,048 (N=2048) and 16,384 (G=64); step 16 1 and 8. Two runs at this head hold them:
  - `r20261112-181310-35e4` built each norm template's statement and its `chunk_rows.chunk_cap`: 16,384 for `DsaRMSNormApply_v3{G=64}`, 2,048 for `DsaRMSNormSumSq_v3{N=2048}` and 8,192 for `{N=512}`. Each prefill cap equals it and each step-16 cap is at most it.
  - `r20261112-185944-481a` staged one statement at each step-16 cap on random rows: 8 `DsaRMSNormApply_v3{G=64}` units (k_log 22, 2 units a block, 4 blocks) and 1 `DsaRMSNormSumSq_v3{N=512}` unit (k_log 22, 1 block). Each passed the stager's reference check and is in the proved form (`ks_form_why` none). Each is inside the spec shape: k_log at most 27, and m at most 35, where the script computes m as k_log plus the bits of the block count.
- `adapters/vllm/tests/pipeline/test_served_unit_rows.py`: `STATEMENTS` goes from main's `738cbcd5…4f4f7e` to `7abb1051…2dc490c4f`, with 38 statements: the value the test printed on the merged tree `ec8a5d943b` (run `r20261112-175309-4983`, whose served templates this head keeps).
- `adapters/vllm/tests/program/test_dsa_norm.py`: `test_registrations` expects `boolean_version` to give `DsaRMSNorm_v3{N=512,EPS=1e-05}`.
- `adapters/vllm/tests/program/test_dsa_norm_calls.py`: `boolean_version` gives the `_v3` (`calls.id`), and the docstrings say so.
- `adapters/vllm/tests/properties/glm53_fused_rmsnorm_capture_gpu.py`: the bind check expects `DsaRMSNorm_v3`. It is a GPU tool in no suite, and was not run.
- `adapters/vllm/tests/pipeline/test_serving_rows_norm_split.py`: the DSA norm's Boolean path binds `DsaRMSNormSumSqV3` and `DsaRMSNormApplyV3`.
- `tools/verity/circuit_check/src/circuit_check/targets.py`: `CALLEE_CALLS` lists the `_v3` row's pieces (`DsaRMSNormSquares_v1`, `DsaRMSNormSquareFma_v1`, `DsaRMSNormMeanRsqrt_v1`, `DsaRMSNormScale_v1`, `DsaRMSNormOut_v1`) as Calls passed by `DsaRMSNorm_v3`, and adds `DsaRMSNorm_v3` to `F32Add_v3`'s. `DsaRMSNorm_v3` is now the DSA norm word's Boolean version, so its served rows are Calls over the partition budget (7,818,990 gates at N=512 and 32,674,537 at N=2048), and their callees partition them. The same table already does this for `RMSNormFusedCudaVector8_v4`'s pieces. Only circuit-check reads the table, so it moves no served statement.
- `adapters/vllm/tools/description_sizes.json`: against main's record, 27 entries out (the 24 long-row `_v1`s and the norm's 3 `_v2`s) and 27 in; no other entry differs (below).
- `adapters/vllm/tests/tools/allowlists/description_sizes.json`: the norm's three `_v2` entries are out, beside #9318's six. Main's allowlist has 14 entries; this one has 5.

Left as it is: `adapters/vllm/manifests/capture-maps/glm-moe-dsa-eager-hopper-tp8-v0.json` line 262 still says, in prose, that `DsaRMSNorm_v1` is "(DsaRMSNorm_v2 on bits)". The map is vllm's and many tests read it, so I didn't change it here.

This branch doesn't touch `boolean_dsa_topk_mask.py` (`_count_sum`, `_count_tree`, `_sum_counts`).

## Size record

Regenerated from main's by `description_sizes.py update -j 8` in run `r20261112-175309-4983` at the merge `ec8a5d943b` ("95 measured, 5 kept, 0 failed; 5 of 100 past 4,194,304 bytes"). `update` rewrites `run`, `seconds` and `gb` in every entry it measures, so the job then put back main's entry for every template whose `bytes` and `largest` equal main's (68 entries; a step of the job, not committed, which CIR-268 makes the tool's own). Against main's record:

- 27 entries in, 27 out, 0 rewritten.
- Templates past 4,194,304 bytes: 31 of 100 in main's, 5 of 100 here.
- The norm's descriptions, in bytes: `DsaRMSNormSumSq{N=2048}` 293,095,727 at `_v2`, 1,090,688 at `_v3`; `{N=512}` 66,279,016 and 1,088,488; `DsaRMSNormApply{G=64}` 10,931,022 and 269,724. The two `_v3` sums of squares build in about 18 to 21 s, so their rebuild tests are slow-marked.

## circuit-check

These ran at the head before the merge with main, `e7840d8e9e`; main changed no circuit, sidecar or pin file of the norm's or the long row's since, and its one change to circuit-check is an ordering fix in `targets._registered`.

- **The norm's sidecar** (`boolean_dsa_norm.circuit-check.py`, `-j 2`, run `r20261112-161742-2046`): 20 targets, 1 failure, 0 unchecked.
  - Every served `_v3` root passed: `DsaRMSNorm_v3` at N=2048 and N=512 (their partition checked by their callees), `DsaRMSNormSumSq_v3` at N=2048 and N=512, and `DsaRMSNormApply_v3{G=64}`. So did the `_v3`s at the model statics, every piece, `DsaLatentRow_v2`, `TritonBlockRMSNorm_v2` and the `_v2` sum of squares and apply unit. 19 targets were compared with the reference on random vectors.
  - The one failure is `DsaRMSNorm_v2{N=256,EPS=1e-05}`: `circuit-check/error … Undecided: whether a served Program Calls it needs the whole catalog built`. Checked on its own with `--no-as-call`, it passed (`boolean_version` no longer gives it, and no partition file or `verity_vllm` module names it). `circuit-check --all` decides it. At this head, its shard 2 (`r20261112-185656-2b58`, below) checked `DsaRMSNorm_v2{N=256,EPS=1e-05}` with no failure.
- **Before the `CALLEE_CALLS` change** (run `r20261112-160633-bff4`, at `a28474c4b4`): both served `DsaRMSNorm_v3` rows failed `partition/unpartitioned` (`DsaRMSNormSquares_v1 is no Call`).
- No pin was added or moved by circuit-check, and `--update-pins` was not run.

## The merge with #9318's `05dc686227`

What was checked at this head in `r20261113-035011-8758` (cpu-9, 16 cores; a `--cwd clone` job, since the record's step writes into the tree):

- **`test_dsa_topk_reads.py`, every test:** 40 passed, 1 skipped (the one `VERITY_ML_DEFINITIONS_SLOW=1` runs). That includes the merged region of #9318's body: the slot's k_log is 22 and the `_v2` merge's is at most 26.
- **The size record:** `description_sizes.py update -j 8`, main's tool, rewrote the record byte for byte as committed ("95 measured, 0 of them written; 100 entries kept as they stood, 0 failed"). `test_description_sizes.py`: 104 passed.
- **`STATEMENTS`:** `test_served_unit_rows.py` passed with this branch's pin unchanged.
- **Pins:** none moved, and `--update-pins` was not run. `circuit-check --all` at this head (below) found no pin mismatch.

## `circuit-check --all`, at this head

`circuit-check --all` ran at `a62164a32a` (each shard's `head.txt`), in the four shards and with the same command as before: 64 GB asked, timeout 21,600 s. All four ended rc 0, and all were read: 17,340 targets, 0 failures. Each shard built 871 catalog bindings and reused none.

- `r20261113-035118-af3d` (1/4, 12 jobs): 4,328 targets, 0 failures, 977 unchecked.
- `r20261113-035153-b2c7` (2/4, 24 jobs): 4,320 targets, 0 failures, 964 unchecked.
- `r20261113-035206-a1cf` (3/4, 24 jobs): 4,402 targets, 0 failures, 991 unchecked.
- `r20261113-035219-058a` (4/4, 11 jobs): 4,290 targets, 0 failures, 1,039 unchecked.

The unchecked are warnings, which never fail.

## Runs at this head

- `r20261113-035051-a369` (cpu-14, 16 cores, 64 GB asked; took its slot at 04:00Z), source `a62164a32a` (its `head.txt`). It runs `scratch`'s `headsuites2.sh`, in three steps:
  - `suites.py --changed 1e4d5c993c` whole, not `--quick`: the 60 suites the change from that `main` touches, `verity-vllm` and `repository` among them, with every slow test.
  - `repository` by name.
  - The two long-row files with `-rA`.
- Its first step, read at 06:02Z: 48 of the 60 suites had passed, none had failed and none was cached. `repository` had 392 passed.
- 12 suites had not ended at 06:02Z and are not read: `verity-vllm`, `verity-numerical`, `research`, circuit-check's suite, `verity/circuits`, `verity/commitments`, `verity/commitments/windows`, `verity-core-evaluate`, `verity/definitions/ops`, `verity/definitions`, `verity-flock` and `verity/protocols/pouw/verifier`. The two long-row files are in `verity-vllm`. The run's last two steps had not started.
- `ratchet_gate.py origin/main HEAD --json` at `a62164a32a`, base `1e4d5c993c08`: `"failures": []`. The change from `1e4d5c993c08` passed, 14,952 to 14,943 ratchet exceptions.

## Runs at `3da6c052b5`, before the merge with main at `1e4d5c993c`

- `r20261112-234308-dae3` (cpu-3, 8 cores, 48 GB asked; waited 2 min 43 s), source `3da6c052b5` (its `head.txt`). It runs `scratch`'s `headsuites.sh` with `--changed cc269e5261`.
  - 23 suites: 23 passed.
  - `verity-vllm` had 14,364 passed, 401 skipped and none failed. `test_served_unit_rows.py` and the record's tests are in that suite, so the merge moves no pin.
  - `repository`: 352 passed.
  - Its last two steps (`verity-vllm` and `repository` by name, then the long-row files with `-rA`) were still running at 00:44Z and are not read.
- `ratchet_gate.py origin/main HEAD --json` at `3da6c052b5`, base `7e70df68bd18`: `"failures": []`. The change from `15662e7c40` passed, 15,002 to 14,993 ratchet exceptions.

## Runs at `cc269e5261`, before the merge of #9318's template commit

A wait is from asking for a slot to taking it, and a hold from taking it to the run's end (each run's `slot.json` and `status.json`).

- `r20261112-180808-372b` (cpu-3, 16 cores, 48 GB asked; waited 26 min 22 s, held 3,910 s): `suites.py --changed 15662e7c40 --quick` at source `cc269e5261` (its `head.txt`). 57 suites: 56 passed and 1 failed. `verity-vllm` had 14,367 passed, 3 failed and 401 skipped. The three failures are vllm's long-row tests that #9318's switch moves: `test_long_row.py::test_the_served_row_is_4213_units_of_56_templates` and both cases of `test_serving_rows_bits.py::test_every_unit_of_the_long_row_but_its_read_is_a_bits_template_on_its_definitions_rows`. Each expects `DsaTopKDigitPick_v1` and `DsaTopKMaskBlock_v1` where `dsa_topk_mask_cut.plan` now gives the `_v2`s; #9318's body has them and the pin they move. The norm's tests, `test_served_unit_rows.py` among them, are not among the failures.
- `r20261112-181241-66fd` (cpu-1, 18 cores, 96 GB asked; waited 16 min 8 s, held 3,945 s): `VERITY_CIRCUIT_CHECK_SLOW=1`, `test_every_call_the_suite_checks_is_partitioned` and `test_a_call_whose_partition_is_unchecked_passes_call_status_to_its_callees`. Both passed, in 3,940 s. They are the opt-in tests that hold the `CALLEE_CALLS` entries.
- `r20261112-181310-35e4` and `r20261112-185944-481a`: the caps, above.

- `ratchet_gate.py origin/main HEAD --json` at `cc269e5261`, base `15662e7c40`: `"failures": []`, 15,002 to 14,993 ratchet exceptions.

## Runs before the merge with main

At `a28474c4b4` and `e7840d8e9e`, whose norm files this head keeps:
- `r20261112-142231-5a5c`: 56 of 56 suites the change from `19fb19a21e` touches passed (`verity-vllm` 14,288 passed, 420 skipped), with `test_served_unit_rows.py`, `test_serving_rows_norm_split.py`, `test_dsa_norm.py` and `test_dsa_norm_calls.py` run whole.
- `r20261112-155804-9483`: `test_dsa_norm_calls.py`, every test: 12 passed.
- `r20261112-160236-70e3`: `test_mnnvl_and_dsa_norm_splits.py`, every test: 31 passed.
- `r20261112-163603-7690`: the 7 suites the `CALLEE_CALLS` change touches: 7 passed.

## With #9370

#9370 (`cursor/circuits-cir265-count-sums-6f04`) is not in the `main` this head holds. So this head was merged with #9370's head `66ddfa5136` on a scratch branch: `3b2b4c2566`, a clean merge, not pushed. No pin moves:

- `r20261112-181003-66d6` (cpu-4, 23 cores, 48 GB asked; waited 14 min 49 s, held 3,336 s): the four long-row sidecars (`dsa_topk_reads`, `dsa_topk_mask_cut`, `dsa_topk_mask`, `boolean_dsa_topk_mask`, `-j 2`). 385 targets, 0 failures, 103 unchecked (warnings).
- `r20261112-180921-d93f` (cpu-9, 22 cores, 48 GB asked; waited 75 min 18 s, held 4,873 s): `suites.py --changed cc269e5261 --quick`, then `test_description_sizes.py` and `test_served_unit_rows.py` whole.
  - Suites: 57 in all, 56 passed and 1 failed. `verity/definitions` had 864 passed. `verity-vllm` had 14,359 passed and 3 failed, the same three long-row tests as at this head without #9370.
  - The pin tests: 100 passed. That is the record's 99 rebuilt to their recorded bytes, and `STATEMENTS`.
- #9370's slow tests (`test_dsa_count_sums.py`, `test_dsa_long_row_calls.py`) were not run here.

## `circuit-check --all`, at `cc269e5261`, before both merges

`circuit-check --all`, in `ci.toml`'s four shards. Each runs `python -m circuit_check --all -q --jobs $(nproc) --out … --cache ~/.cache/verity-check/circuit-check --catalog-units --mem-gb 60 --shard K/4 --shard-dir …`, with 64 GB asked and a timeout of 21,600 s. The shards are `r20261112-185628-23b5` (1/4), `r20261112-185656-2b58` (2/4), `r20261112-185710-1932` (3/4) and `r20261112-185725-1d95` (4/4), all on cpu-14. A first launch passed `--records-manifest`, which this tree's circuit-check refuses; `check.py` adds it only where `circuit-check` takes it. Its shards 1 and 4 exited 2 on it, and shards 2 and 3 were cancelled before they started.

All four were read: 17,236 targets, 1 failure. Each shard built 829 catalog bindings and reused none.

- Shard 1 had 4,302 targets and 0 failures. Shard 2 had 4,297 and 0 failures. Shard 3 had 4,376 and 0 failures.
- Shard 4 had 4,261 targets and 1 failure: `long_row.circuit-check.json`'s `DsaTopKReadRowsCut_v1{N=12,TOPK=2,BLOCK=4,READ=EmbeddingRead_v1{V=16,H=1}}`, sha512 `d27c07cc…` to `877f04d2…` (`lowering/pin-mismatch`). It is vllm's long-row pin that #9318's switch moved; #9318's template commit, merged here, re-pins it, and the five long-row sidecars pass with that pin (#9318's body). `--all` was not run again at this head.
- Unchecked (warnings, which never fail): 960, 949, 978 and 1,028.

## Not run

- The whole uncached check, the Lean check and lean-agreement. No Lean or Flock file of this change's own moves.
- At the earlier heads, the slow tests of suites whose test modules the change doesn't reach (`--quick`); at this head they are in the run above, in its 12 suites not yet read.
- `glm53_fused_rmsnorm_capture_gpu.py`, a GPU tool in no suite.
- Staging or proving any unit at served size: not this PR's.
- #9254 and #9075 are still open and regenerate the record too; #9075 also changes `glm53_all_families_tp8_step16.json` and `glm53_prefill_tp8_step0.json`. Whichever reaches `main` second, this or they, takes one more merge.

