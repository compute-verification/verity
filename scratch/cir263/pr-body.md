Lane circuits
Refs CIR-263
Goal vllm/production-vllm-reproducible

Each statement is marked: [run] read from a run's output, [read] read from the source or the tree, [inferred] my reasoning or arithmetic, not checked by a run.

## What changed and why

GLM-5.3's recipe serves with five draft tokens from one MTP draft layer. At vLLM `d9105ea80`, that layer's own three kernels had no Definition [read]. This holds them to registered Definitions at GLM-5.3's sizes, binds them for circuit-check, and adds the captures that compared each kernel's words on an H200.

- **`_fused_eh_norm_kernel`** (enorm and hnorm in one launch, pin `models/deepseek_v32/common/kernels.py:987-1019`) is `MtpEhNorm_v1{H, BLOCK, WARPS, EPS}`, new in `verity/definitions/registry/mtp_draft.py`, bound at H 6144, BLOCK 8192, WARPS 4, EPS 1e-5 [read]. At position 0 it replaces the embedding row by bf16 +0, the whole int64 position compared to 0; each half is then the registered `TritonBlockRMSNorm_v1` at 6,144 values in a masked block of 8,192 [read]. It is not `DsaRMSNorm_v1`, which is that norm only at a power-of-two row [read].
- **Staging.** A whole 6,144 row is past Flock's proved block (2^29 bits for `MtpEhNorm_v1`, 2^28 for `TritonBlockRMSNorm_v1`) [run: `r20261112-141330-0ffd`], so the row is staged as its cut, declared in `CUTS` [read]: the position select `MtpEhZero_v1{H}`, each half's scale `TritonBlockRMSNormSumSq_v1{N, BLOCK, WARPS, EPS}`, and 96 of the registered `DsaRMSNormApply_v1{G=64}` a half [read].
- **Boolean versions**, as Calls in `verity/definitions/registry/boolean_mtp_draft.py`: `MtpEhZero_v2`, `TritonBlockRMSNormSumSq_v2` with its fused-eps epilogue `TritonBlockRMSNormMeanRsqrt_v1`, and `MtpEhNorm_v2`, the cut on bits with its own `CUTS` [read]. With them circuit-check compares each word Definition with a realization on random vectors and pins it.
- **`eh_proj`** (pin `models/deepseek_v32/nvidia/mtp.py:68`) is a replicated `nn.Linear(12288, 6144, bias=False)` in bf16; on the H200 it is `F.linear` on cuBLAS [read]. It needs no new Definition. `verity/definitions/registry/mtp_draft.circuit-check.py` now binds both of its routes: `Gemm_v2/glm53-mtp-eh-proj`, core's `Gemm_v2{K=12288, N=6144, DOT=HopperBF16WgmmaDot16_v1}`, where cuBLAS does not split K; and `CublasSplitKGemm_v1/glm53-mtp-eh-proj-2`, `cublas_bf16`'s `CublasSplitKGemm_v1{K=12288, N=6144, SPLIT_K=2, DOT=HopperBF16WgmmaDot16_v1}`, at the one split count the route scan found up to 8,192 tokens [read]. Each binding's id equals the one the capture of record bound and compared with [run: `r20261113-154513-8180`; read: the sidecar's targets built here and compared with the capture's `bind_check([2])`]. A whole row of either is past Flock's proved block range (2^38 bits against 2^27) [run: `r20261113-201855-8dbc`], and a served row of either is cut along N into slices, each the same Definition at N = 1 on x and one row of W (`SLICES` in `adapters/vllm/verity_vllm/pipeline/serving_templates.py`) [read]. So each route is bound at its slice too: `Gemm_v2/glm53-mtp-eh-proj-slice`, `Gemm_v2{K=12288, N=1, DOT=HopperBF16WgmmaDot16_v1}`, and `CublasSplitKGemm_v1/glm53-mtp-eh-proj-2-slice`, `CublasSplitKGemm_v1{K=12288, N=1, SPLIT_K=2, DOT=HopperBF16WgmmaDot16_v1}`. These two are what the capture's reference evaluated at each of its 177 sampled coordinates (`_reference` in `glm53_bf16_gemm_capture_gpu.py`) [read].
- **The indexer's logits at six query rows a request.** On sm_90 `next_n = 6` is not native (DeepGEMM's paged logits are native at 1, 2 and 4, pin `utils/deep_gemm.py:560-568`), so vLLM flattens each request's six queries into six rows at `next_n = 1` (`v1/attention/backends/mla/indexer.py:72-119`, `:672-692`) [read]. The kernel is the one `DsaMqaLogits_v3` was captured at, at shorter key counts [read], so no new Definition or binding.
- **Captures**, under `adapters/vllm/tests/properties/`: `glm53_mtp_eh_norm_capture_gpu.py`, `glm53_mtp_eh_proj_capture_gpu.py`, `glm53_mtp_indexer_logits_capture_gpu.py`, and `glm53_mtp_capture_card.sh`, which runs one under one lease on one card after checking the card is idle [read].
- **Each capture's record is a registered property's**: `seal` refuses a schema no property names, so the three are registered in `REGISTRY` in `adapters/vllm/verity_vllm/records/property_record.py`, each schema id written once, in its row, and each capture reads its `SCHEMA` from that row [read].
  - `records/schema_ids.py` is left as on `main`, and no new module holds the ids. The typed-records lint (`adapters/vllm/tests/lint/test_typed_records.py`) allows a schema literal anywhere under `verity_vllm/records/`, so either place passes it [read]. `schema_ids.py`'s docstring keeps the ids of records that have no type in the package. A property record's type is `PropertyRecord`, and `REGISTRY` names its schema: every other property's id is a literal in its row, and none is in `schema_ids.py` [read].
  - An import of `schema_ids` into `property_record.py` would also meet the two lines the refactor line (`cursor/vllm-refactor-c1f1`) adds among that file's imports [run: `git merge-tree`].
- **The split-K probe at eh_proj's shape.** `splitk_probe_words` in the sibling `glm53_bf16_gemm_capture_gpu.py` could not separate the strided partition from the contiguous one at K 12288, 2 splits, bf16 output, and raised [run: `r20261113-031522-215e`]. Where its random rows leave those two equal, and only there, it now plants a crafted third row, so every probe that existed before is unchanged [read].
- **`eh_proj`'s shape in the sibling's `MODELS`** is there only while a run or a scan process lasts (`registered()`); importing the module used to add it, and 18 of the sibling's tests then saw a third model [run: `r20261113-040826-9fec`].
- **A CPU test of each capture's `main`**, `adapters/vllm/tests/properties/test_glm53_mtp_captures.py`: the GPU is stood in for by the twin's words at the launch; everything after it is real, through the worker pool to the sealed record and the exit status [read].
- `verity/definitions/tests/catalog_ids.json` lists the seven new ids; no registered Definition's words changed [read].

## What the captures of record gave

One H200 card, each run read on its node, Fri 13 Nov 15:45Z to 15:52Z, one lease each with `glm53_mtp_capture_card.sh`, at branch head `ce09de03c318f40d29efbec7cca2536a8b7cf735`. The card held no compute process and 0 MiB at each start. Each record sealed with `ok` true, its digest recomputed from the fetched file, and each names vLLM `0.28.1rc1.dev472+gd9105ea80` [run]. A word is compared with the Definition's NumPy twin; the Definition's own evaluator runs on a sample and equals the twin on all of it [run].

- `r20261113-154513-8180`, `eh_proj`, digest `41442bea4181c34d43a40ac11596b2e9bb85456f2b8c8376e24ebaee2cef5e1b` [run].
  - Route scanned at every M from 1 to 8,192 tokens: split-K 2 at M 105-111, 129-159, 161-223, 272, 288 and 336, no split elsewhere [run].
  - 24 launches (M 1, 2, 3, 4, 5, 6, 8, 12, 16, 24, 32, 48, 64, 96, 105, 128, 192, 256, 384, 512, 768, 1024, 1025, 8192), each deterministic and on the scanned route; 4,458 rows of 6,144 words (every row up to M 1,024, 384 rows of M 1,025 and of M 8,192): 27,389,952 words compared, 0 differ; the evaluator on 177 coordinates, 0 differ [run].
  - The two split launches (M 105, 192): 297 rows, 1,824,768 words, 0 differ from `CublasSplitKGemm_v1` [run]. The job's probe, 32 crafted words a split launch: contiguous f32 halves fit all 32, strided halves miss 1, bf16 partials miss 2 [run].
  - Recomputed afterwards on a CPU from the run's kept words, not by the job: of the 1,824,768 split words, 0 differ under contiguous f32 halves, 5,653 under strided halves, 675,620 under bf16 partials [run, on my VM, not a recorded run].
- `r20261113-155013-668e`, `eh_norm`, digest `5584f173348e50978eb87d2309a56f096521c291f473950f27f76b8ff446fc11` [run].
  - Two launches of 1,024 positions, each half compared whole: 4,096 rows, 25,165,824 words, 0 differ; `MtpEhNorm_v1`'s evaluator on 102 rows, 0 differ [run].
  - One PTX specialisation, hash `88978f9f23bf0bf743216e79cbc564e294a845fda819eeccf30c8ca355408b7d`, as pinned from the pin's source; `.reqntid 128`; SASS: 2 `FFMA` of the mean and eps, 2 `MUFU.RSQ`, 0 `MUFU.RCP` [run].
  - Words that refute each alternative reading, fewest first (p half, e half): unfused eps 16 and 21; per-vector chains 31 and 33; 8 warps 41 and 32; a contiguous sum 52 and 36; `x * (r * w)` 56 and 62; `RMSNormTriton_v1` 84 and 111; the position's low word alone 12,288 (e); the position ignored 49,152 (e) [run].
- `r20261113-155125-8b02`, `indexer_logits`, digest `253714da690d3c2e1b3ede3fc9a2a99ee20cf06a23688472f6302d31060bd35d` [run].
  - `native_next_n_supported(6)` false and no varlen on this card; vLLM's own `_PREPARE_UNIFORM_DECODE_KERNEL` gave the lengths and block table expected [run].
  - 9 requests (54 rows): all 50,664 logits, and 50,000 random pairs over them again. 64 requests (384 rows, 702,153 logits): 59 rows whole and 50,000 random pairs, 179,605 distinct logits [run; the distinct counts are mine, from the kept words]. 290,933 comparisons in all, 0 differ, 0 unresolved; the evaluator on 144 logits, 0 differ [run].

**What differs between `ce09de03c318` and this head** [read: `git diff ce09de03c318 HEAD`]. The branch was carried across `main`'s layout moves (`tools/move/restack.py`), then merged with `main` `f479876a8758`, then two commits to the sidecar: eh_proj's whole rows, then their slices. It was then merged with `main` `d260e6bb9cb3` (below). In the captures' own files the only change is the move's import rewrite in `glm53_mtp_eh_proj_capture_gpu.py` (`verity.definitions.ops` to `verity.circuits.ops`, in `bind_check`). The sibling `glm53_bf16_gemm_capture_gpu.py`, whose scan and launch the eh_proj capture runs, changed by `main`'s own work: imports moved to the top, type annotations, an equivalent rewrite of the route join's adjacency test, and `bind_check`'s default model list read at call time; none of it is on the path the eh_proj capture runs [read]. The Definitions' modules changed only by the move's import rewrites. After the second merge, the captures import `seal` from `records.property_record` and read their `SCHEMA` from its `REGISTRY` row, with the same ids as before. The sibling changed only that import [read]. No capture ran at this head [inferred from the above: they stay of record].

## The merge with `main` `d260e6bb9cb3`

`main` had moved `verity_vllm/properties/record.py` to `verity_vllm/records/property_record.py`, so the branch no longer merged [run: `git merge-tree`]. The merge (`0d95e7a29033`), and one commit on it (`8ab720de29ac`), make these changes [read]:
- `records/__init__.py` is `main`'s text: its docstring names no module this branch adds.
- The three `REGISTRY` rows go into `records/property_record.py`, each with its schema id written in the row.
- The three captures, their test and their docstrings name `records.property_record`.
- `records/mtp_captures.py` is deleted.

`git merge-tree --write-tree` of this head:
- Clean against `origin/main`, against the seven `queue/*` candidates, and against circuits' other open heads. Against #8763 it conflicts only where #8763 conflicts with `main` itself, in two of circuit-check's test files [run].
- Against `origin/cursor/vllm-refactor-c1f1` it conflicts in `commit/row_tables_device.py` and `tools/verity/check/public_boundary.toml` only, the two files where that line conflicts with `main` itself [run].
- At `0d95e7a29033` it also conflicted in `records/property_record.py` [run], through the `schema_ids` import that `8ab720de29ac` removes.

## circuit-check

`r20261114-080211-687f` at this head, `8ab720de29ac6dc20fe574c5834a2b691b1dfefe`, in its own clone [run]. The merge with `main` changed circuit-check's own code and the Boolean modules it reads, so it ran again here [read: `git diff`].

- The update pass on both sidecars and the two slices: 14 targets, 0 failures, exit 0; the pins' diff after it is empty [run].
- The report on both sidecars, `DsaRMSNormApply_v1{G=64}` (the apply the norm's cut uses, `dsa_norm`'s binding), the two slices and the two whole rows: 17 targets, 4 failures, 6 `vectors/none` warnings; 11 targets compared with a realization on random vectors [run].
  - Every Definition added here passes: `MtpEhNorm_v2` at H 256 and 6144, `MtpEhZero_v2` at H 16, 256 and 6144, `TritonBlockRMSNormSumSq_v2` and `TritonBlockRMSNormMeanRsqrt_v1` at N 256 and 6144 [run].
  - The word Definitions with no realization at their bound size (`MtpEhNorm_v1`, `MtpEhZero_v1` and `TritonBlockRMSNormSumSq_v1` at 6144, and `DsaRMSNormApply_v1{G=64}`) carry `vectors/none` and no failure; their Boolean versions above are compared at the same size [run].
  - `Gemm_v2{K=12288, N=1, …}` and `CublasSplitKGemm_v1{K=12288, N=1, SPLIT_K=2, …}`: ok [run].
  - The four failures are on the two whole rows, when named by id: `block/past-range` on both (2^38 bits; Flock's proved range is 2^27), and on the split-K row also `partition/unpartitioned` and `correspondence/unprobed-table` [run]. `circuit-check --all` checks neither whole row, since a family's members over 150,000 gates are left out unless pinned (`select` in `tools/verity/circuit_check/src/circuit_check/targets.py`) [read]. The served indexer's whole-row bindings in `cublas_bf16.circuit-check.py` are in the same position [read; that they too are past range is my arithmetic, inferred].
- The same 14 and 17 targets gave the same verdicts at `08d6acdced22`, before the merge [run: `r20261113-210950-1125`].

## Checks at this head

All at `8ab720de29ac6dc20fe574c5834a2b691b1dfefe`, against the `main` it merged, `d260e6bb9cb3f9278634d55f09aa9b2ebbd14e1b`.

- **Suites.** SUITES_LINE
- **The adapter's ratchet gate**, `python3 adapters/vllm/tools/ratchet_gate.py d260e6bb9cb3 HEAD --json`: verdict passed, exit 0; it prints "no retired-term site is new in the 14 file(s) it changes", and the same for direct environment reads, silent except handlers and lane text [run: `r20261114-080211-ab72`].
- **Pins**, `python3 tools/verity/check/pins.py --stale`: exit 0 [run: `r20261114-080211-ab72`].
- **Retired terms**, `tests/test_retired_terms.py` and `tests/lint/test_repo_retired_terms.py`: 43 passed, with 5 bookkeeping warnings (stale `ALLOWED` entries, maintained docs, Lean names, a count's growth) [run: `r20261114-080211-ab72`]. This branch changes no `.lean` or `.md` file, and no line it adds holds a retired term [read: `git diff --stat`, and the gate above].

## Runs

Each read on its node; every head below is in this branch's history [run: `git merge-base --is-ancestor`].

- At this head, `8ab720de29ac`: circuit-check `r20261114-080211-687f`; suites RUNS_SUITES; gate, pins and retired terms `r20261114-080211-ab72`.
- At `08d6acdced22`, before the merge with `d260e6bb9cb3`: circuit-check `r20261113-210950-1125`; suites against `f479876a8758` `r20261113-213643-9f06` (cut at its limit, 53 passed, none failed), `r20261114-013209-983d` and `r20261114-013424-7b75` (the eleven it never reached, all passed but circuit-check's own suite's input guard, which `r20261114-061636-54bc` did not reproduce); gate, pins and retired terms `r20261114-013321-78ad`; `research.carry` `r20261114-015101-b9b2`.
- At `7c96ba47aed0`, which differs from `08d6acdced22` only in the sidecar's two slice bindings: circuit-check on both sidecars `r20261113-194646-9c64`, and on the whole rows by id `r20261113-201855-8dbc` (the 4 failures above) [run].
- At the merge `3e9855683f6c`: pins `r20261113-194328-88d1`.
- At `ce09de03c318`: the three captures of record above.
- Earlier, before the carry: `r20261112-141330-0ffd` at `0fbd2a35cb3c` (the whole norm row past range); `r20261113-031522-215e` at `920597d5ca7f` (the split-K probe that raised); `r20261113-040826-9fec` at `78884a1b8aa7` (the 18 sibling tests that saw a third model).

## Not stated

- That a served draft step makes these launches. Each capture calls the kernel directly with its own operands on a card with the pinned vLLM installed; no run here serves with speculative decoding. Which token counts a draft step gives `eh_proj` (and so which split span it lands in) is my reading of the source [inferred].
- `eh_proj` above 8,192 tokens, and any split count but 2 [run: none up to 8,192]. At 2 splits, forward, reverse and tree reductions are one order, so the capture separates only the partition and the partials' type [inferred].
- The indexer's 64-request launch whole: 522,548 of its 702,153 logits are not compared [run, my count].
- `fused_norm_rope` and `fused_q` with `has_indexer=False` (draft steps 2 to 5, and GLM-5.3's 58 IndexShare target layers), a separate Triton specialisation of each: CIR-286's [read].
- The rest of the layer (embedding, input norm, MLA, MoE, head): read and listed on CIR-263, not built or captured here.
- The capture map's `mtp` section, the fold and the step compiler (vllm's, VLLM-393), and the accept rule.

## What I trust least

That eh_proj's N = 1 slices are what a served eh_proj row's proof blocks hold [inferred]. I read it from `SLICES` and from the capture's `_reference`, which evaluates those slices. No run here stages eh_proj's row through the serving templates and checks the units it gets. The two whole-row bindings stay in the sidecar as the statics the capture compared. They fail circuit-check when named by id, and `--all` does not check them [run, read]. A reviewer may prefer them dropped.

`research.carry ce09de03c318 08d6acdced22` exits 1, kind `authored`: `08d6acdced22` changes the sidecar [run: `r20261114-015101-b9b2`]. At the merge it said `hand-resolved` (a conflict in `adapters/vllm/tests/properties/test_record.py`, resolved by hand in `fd8d266dc198`) [run: `r20261113-194915-eb71`]. So the carry does not vouch for this head.
