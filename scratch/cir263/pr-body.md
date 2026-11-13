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
- **`eh_proj`**: <route capture result; the Gemm_v2 and CublasSplitKGemm_v1 bindings at K 12288, N 6144>.
- **The indexer's logits at six query rows a request**: on sm_90, next_n = 6 is not native, so vLLM flattens each request into six rows at next_n = 1. The kernel is the one `DsaMqaLogits_v3` was captured at. <capture result>.
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
- `r20261113-052846-14e2` (`verity-vllm`, whole, at `ce09de03c3`): <counts>.
- <the captures of record: run id, head, rows and words compared, words that differ>.

## Not run

<list>
