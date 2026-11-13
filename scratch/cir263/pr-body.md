Lane circuits
Refs CIR-263
Goal vllm/production-vllm-reproducible

## What changed and why

GLM-5.3's recipe serves with five draft tokens from one MTP draft layer. At vLLM `d9105ea80`, that layer's own kernels had no Definition. This holds them to registered Definitions at GLM-5.3's sizes, and adds capture scripts that compare each kernel's words on an H200.

- **`_fused_eh_norm_kernel`** (enorm and hnorm in one launch) is `MtpEhNorm_v1{H, BLOCK, WARPS, EPS}`, new in `verity/definitions/registry/mtp_draft.py`. Position 0 zeroes the embedding row, the whole int64 compared, and then each half is the registered `TritonBlockRMSNorm_v1` at 6,144 in a block of 8,192 on 4 warps. It is not `DsaRMSNorm_v1`, which is that norm only at a power-of-two row with no masked block.
- **Staging.** A whole 6,144 row is past C-Flock's proved block of 2^27 bits: 2^29 for `MtpEhNorm_v1`, 2^28 for `TritonBlockRMSNorm_v1`. So the row is staged as its cut, declared in `CUTS`. The units are:
  - the position select, `MtpEhZero_v1{H}`;
  - each half's scale, `TritonBlockRMSNormSumSq_v1{N, BLOCK, WARPS, EPS}`, the one f32 every element reads;
  - 96 of the registered `DsaRMSNormApply_v1{G=64}` a half.
  - The cut's ANDs equal the whole's.
- **Boolean versions**, written as Calls in `boolean_mtp_draft.py`: `MtpEhZero_v2`, `TritonBlockRMSNormSumSq_v2` with its fused-eps epilogue `TritonBlockRMSNormMeanRsqrt_v1`, and `MtpEhNorm_v2`, the cut on bits with its own `CUTS`. With them, circuit-check compares each word Definition with a realization on random vectors and pins it.
- **`eh_proj`** is a replicated `nn.Linear(12288, 6144, bias=False)` in bf16; on the H200 it is `F.linear` on cuBLAS. Scanned on the card at every M from 1 to 8,192 tokens, cuBLAS splits K in 2 at M 105-111, 129-159, 161-223, 272, 288 and 336 and nowhere else. Where it does not split the Definition is the registered `Gemm_v2{K=12288, N=6144, DOT=HopperBF16WgmmaDot16_v1}`; where it does, the registered `CublasSplitKGemm_v1{K=12288, N=6144, SPLIT_K=2, DOT=HopperBF16WgmmaDot16_v1}`. The capture of record compared 27,389,952 words at 24 values of M and found 0 that differ (Runs). <the two bindings in `mtp_draft.circuit-check.py`>.
- **The indexer's logits at six query rows a request**: on sm_90, next_n = 6 is not native, so vLLM flattens each request into six rows at next_n = 1. The kernel is the one `DsaMqaLogits_v3` was captured at, at shorter key counts. The capture of record ran vLLM's own flattening and compared 290,933 logit words with `DsaMqaLogit_v3{H=32, D=128}`: 0 differ (Runs, with what it did not compare).
- **Captures**, beside the others under `adapters/vllm/tests/properties/`: `glm53_mtp_eh_norm_capture_gpu.py`, `glm53_mtp_eh_proj_capture_gpu.py` and `glm53_mtp_indexer_logits_capture_gpu.py`, with `glm53_mtp_capture_card.sh`, which runs one under a lease on one card after checking the card is idle. They add no `# noqa`: a phase that raises ends the capture with its traceback and writes no record. The one catch, around reading a scan process's `scan.json`, names `OSError` and `JSONDecodeError` and leaves that chunk `ok` False. No capture of record ran before that change (`920597d5ca`).
- **Each capture's record is a registered property's.** `seal` refuses a schema that no property names, so the three captures are registered in `verity_vllm/properties/record.py` (`glm53_mtp_eh_norm_capture`, `glm53_mtp_eh_proj_capture`, `glm53_mtp_indexer_logits_capture`), with their schema ids in a new `verity_vllm/records/mtp_captures.py`: the typed-records lint keeps schema literals in `records/` and its list only shrinks.
- **The split-K probe at eh_proj's shape.** `splitk_probe_words` in `glm53_bf16_gemm_capture_gpu.py` could not tell the strided partition from the contiguous one at K 12288, 2 splits, bf16 output, and raised. Where its random rows leave those two equal, and only there, it now plants a third row (2^25 in block 0, -2^25 in block 1, +1 in block `splits`) that the two partitions sum differently. That is exactly where it raised before, so every probe that existed is unchanged.
- **A CPU test of each capture's `main`**, `tests/properties/test_glm53_mtp_captures.py`: the GPU is stood in for by the twin's words at the launch, and everything after it is real, through the worker pool to the sealed record, the GO / NO-GO line and the exit status, with a flipped word found where it was put. The first run on a card found the two slips above only after its comparisons; this test finds both without a card.
- **The eh_proj capture's shape in the sibling's `MODELS`.** The capture reuses `glm53_bf16_gemm_capture_gpu`'s scan and launch, which read a model's shapes from its `MODELS`. It now puts its shape there only while a run or a scan process lasts (`registered`), and takes it out when they end or raise. Importing the module used to add it, and the sibling's own tests, which read every model there, then failed on a third one.
- No registered Definition's words changed; `catalog_ids.json` lists the seven new ids.

## circuit-check

`uv run circuit-check --update-pins` on both sidecars, then `circuit-check` on them, in `r20261112-151627-ac2a`. The pins diff after the update pass is empty. The result: 14 targets, 0 failures, and 4 unchecked, which are warnings.

~~~text
ok        definition DsaRMSNormApply_v3{G=64}  failures=0 unchecked=0
unchecked definition MtpEhNorm_v1{H=6144,BLOCK=8192,WARPS=4,EPS=1e-05}  failures=0 unchecked=1
ok        definition TritonBlockRMSNormMeanRsqrt_v1{N=6144,EPS=1e-05}  failures=0 unchecked=0
unchecked definition TritonBlockRMSNormSumSq_v1{N=6144,BLOCK=8192,WARPS=4,EPS=1e-05}  failures=0 unchecked=1
ok        definition TritonBlockRMSNormMeanRsqrt_v1{N=256,EPS=1e-05}  failures=0 unchecked=0
ok        definition MtpEhZero_v2{H=256}  failures=0 unchecked=0
ok        definition MtpEhNorm_v2{H=256,BLOCK=512,WARPS=1,EPS=1e-05}  failures=0 unchecked=0
unchecked definition MtpEhZero_v1{H=6144}  failures=0 unchecked=1
unchecked definition DsaRMSNormApply_v1{G=64}  failures=0 unchecked=1
ok        definition MtpEhZero_v2{H=16}  failures=0 unchecked=0
ok        definition MtpEhZero_v2{H=6144}  failures=0 unchecked=0
ok        definition TritonBlockRMSNormSumSq_v2{N=6144,BLOCK=8192,WARPS=4,EPS=1e-05}  failures=0 unchecked=0
ok        definition MtpEhNorm_v2{H=6144,BLOCK=8192,WARPS=4,EPS=1e-05}  failures=0 unchecked=0
ok        definition TritonBlockRMSNormSumSq_v2{N=256,BLOCK=512,WARPS=1,EPS=1e-05}  failures=0 unchecked=0
circuit-check: 14 target(s), 0 failure(s), 4 unchecked (warnings, which never fail) on 4 Definition(s)
~~~

All four unchecked are `vectors/none`, on word Definitions bound at 6,144 and on the apply at G = 64: no realization compared them on random vectors at those statics. Their Boolean versions are compared at small statics and pinned at 6,144.

<eh_proj bindings: run id and lines.>

## Runs

- `r20261112-141330-0ffd` (circuit-check, the first form, one whole row): no wrong word, but both 6,144 targets failed `block/past-range`, at 2^29 and 2^28 bits. That is why the row is staged as its cut.
- `r20261112-144600-48a7` (circuit-check with the cut): 0 failures, but no pins, because no word Definition had a Boolean version. That is why `boolean_mtp_draft` exists.
- `r20261112-145958-9b2e` (update pass and report): wrote the `and`-count pins.
- `r20261112-151029-3dd7` (update pass and report): filled the digest pins at 6,144, and found `MtpEhNorm_v2{6144}` past the range (2^28), so it got its own `CUTS`.
- `r20261112-151627-ac2a` (update pass and report): above.
- `r20261112-152848-3d1d` (suites, at `bc8cb977a5`): of the 57 suites the change from its base touches, run with `--quick`, 56 passed. `repository`, run whole, had 351 passed and 1 failed, `tests/test_public_boundary.py::test_no_public_file_needs_one_the_export_does_not_copy`: the card script named a path under `tools/research/`, which the public export doesn't copy. `6a53fbc419` fixes that: the script takes the node's `gpu-lease`.
- `r20261112-165318-9eb5` (suites, at `6a53fbc419`): the 57 suites the change touches, `--quick`, all passed (4 from cache), to `out/suites.json`. `repository`, whole with its slow tests, passed its 352 tests, to `out/suites-repository.json`. <rerun at the final head after the eh_proj bindings.>
- `r20261112-210847-6529` (`repository`, whole, at `920597d5ca`): 352 passed, 0 failed.
- `r20261112-215601-1b52` (`verity-vllm`, whole, at `920597d5ca`): 14,983 passed, 0 failed, 485 skipped.
- `r20261113-031522-215e` (node A, card 0, cores 0-9, one lease of 124 s, at `920597d5ca`; **not a capture of record**: it sealed no record). `eh_norm`: 25,165,824 words compared (4 x 6,291,456), 0 differ; the PTX's hash and counts as pinned. `indexer_logits`: 290,933 words compared (100,664 + 190,269), 0 differ. Both then failed at `seal` (the schemas were not registered). `eh_proj`: the route scan finished (below) and 14 unsplit launches ran, then the split-K probe raised before any comparison.
- `r20261113-033120-282e` (a CPU pod, at `bcc3d97e7d`; computed after the fact from the words that run kept, not by the job that ran the kernel): the 14 unsplit `eh_proj` launches at M 1 to 96 against `Gemm_v2{K=12288,N=6144,DOT=HopperBF16WgmmaDot16_v1}`: 321 rows, 1,972,224 words compared, 0 differ.
- `r20261113-035227-475a` (`repository`, whole, at `78884a1b8a`): 352 passed, 0 failed.
- `r20261113-040826-9fec` (`verity-vllm`, whole, at `78884a1b8a`): 14,979 passed, 18 failed, 485 skipped. All 18 are the sibling capture's tests (17 in `test_glm53_bf16_gemm_capture.py`, 1 in `test_sampled_replay_glm53_rows.py`) seeing `glm53-mtp` in `MODELS` once the new test file imported the eh_proj capture. `ce09de03c3` fixes it.
- `r20261113-052134-678c` (`repository`, whole, at `ce09de03c3`): 352 passed, 0 failed.
- `r20261113-052846-14e2` (`verity-vllm`, whole, at `ce09de03c3`): 14,998 passed, 0 failed, 485 skipped.
- **The captures of record**, node A, card 0, cores 0-9, Fri 13 Nov, each one lease with `glm53_mtp_capture_card.sh` at `ce09de03c318f40d29efbec7cca2536a8b7cf735`; the card held no compute process and 0 MiB at each start; each record is sealed with `ok` true, its digest recomputed from the fetched file, and names vLLM `0.28.1rc1.dev472+gd9105ea80`. <if the head moved since: what differs since `ce09de03c3`>.
  - `r20261113-154513-8180`, `eh_proj` (lease 15:45:42Z to 15:48:31Z), digest `41442bea4181c34d43a40ac11596b2e9bb85456f2b8c8376e24ebaee2cef5e1b`. Routes over M 1 to 8,192: 13 spans, split 2 in six of them. 24 launches (M 1, 2, 3, 4, 5, 6, 8, 12, 16, 24, 32, 48, 64, 96, 105, 128, 192, 256, 384, 512, 768, 1024, 1025, 8192), each deterministic and on the route the scan gave. 4,458 rows of 6,144 words (every row up to M 1,024; 384 rows of M 1,025 and of M 8,192): 27,389,952 words compared with the twin, 0 differ; the Definition's evaluator on 177 coordinates, 0 differ. The two split launches (M 105, 192): 297 rows, 1,824,768 words, 0 differ. Split order: of the probe's 32 crafted words a split launch, contiguous f32 halves fit all, strided halves miss 1, bf16 partials miss 2.
  - The same run's kept words, on this VM's CPU afterwards (`scratch` job `ehproj_split_alternatives.py`, not in the job that ran the kernel): of the 1,824,768 split words, 0 differ under contiguous f32 halves, 5,653 under strided halves, 675,620 under bf16 partials.
  - `r20261113-155013-668e`, `eh_norm` (lease 15:50:23Z to 15:50:53Z), digest `5584f173348e50978eb87d2309a56f096521c291f473950f27f76b8ff446fc11`. Two launches of 1,024 positions, each half whole: 4,096 rows, 25,165,824 words, 0 differ; `MtpEhNorm_v1`'s evaluator on 102 rows, 0 differ. One PTX specialisation, hash `88978f9f23bf0bf743216e79cbc564e294a845fda819eeccf30c8ca355408b7d` as pinned, `.reqntid 128`; SASS: 2 `FFMA` of the mean and eps, 2 `MUFU.RSQ`, 0 `MUFU.RCP`. Words that refute each alternative, fewest first: unfused eps 16 (p) and 21 (e); per-vector chains 31 and 33; 8 warps 41 and 32; a contiguous sum 52 and 36; `x * (r * w)` 56 and 62; `RMSNormTriton_v1` 84 and 111; the low word of the position alone 12,288; the position ignored 49,152. The five bindings bind.
  - `r20261113-155125-8b02`, `indexer_logits` (lease 15:51:35Z to 15:51:55Z), digest `253714da690d3c2e1b3ede3fc9a2a99ee20cf06a23688472f6302d31060bd35d`. `native_next_n_supported(6)` false and no varlen on this card; `_PREPARE_UNIFORM_DECODE_KERNEL`'s lengths and block table as expected. 9 requests (54 rows): all 50,664 logits compared, and 50,000 random (row, key) pairs over them again. 64 requests (384 rows, 702,153 logits): 59 rows whole and 50,000 random pairs, 179,605 distinct logits. 290,933 comparisons in all, 0 differ, 0 unresolved; the Definition's evaluator on 144 logits, 0 differ.

## Not run

- <until bound: circuit-check on `eh_proj`'s two bindings.>
- `eh_proj` above 8,192 tokens, and any split count but 2: no M up to 8,192 splits otherwise. At 2 splits, summing the partials forward, in reverse or as a tree is one order, so the capture tells apart only the partition (contiguous or strided) and the partials' type (f32 or bf16).
- The indexer's 64-request launch whole: 522,548 of its 702,153 logits are not compared.
- `fused_norm_rope` and `fused_q` with `has_indexer=False` (draft steps 1 to 4): a separate Triton specialisation of each, whose PTX no capture has read.
- The draft layer served: no run here serves with speculative decoding, and the capture map's `mtp` section, the fold and the step compiler are vllm's.
- The rest of the layer (embedding, input norm, MLA, MoE, head): read and listed on CIR-263, not built or captured here.
- No Lean and no Flock run: the change adds Definitions and captures only.
