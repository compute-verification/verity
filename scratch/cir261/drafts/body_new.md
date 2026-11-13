<!-- CURSOR_AGENT_PR_BODY_BEGIN -->
Lane circuits
Refs CIR-261
Goal vllm/production-vllm-reproducible

## What changed and why

Under `--kv-cache-dtype fp8`, vLLM d9105ea80 stores each GLM-5.3 key as one 656-byte `fp8_ds_mla` row:
- 512 E4M3 NoPE bytes;
- four f32 scales, one per 128 values;
- 64 bf16 RoPE words.

On a dense prefill step with cached context, the model reads those rows back to 576 bf16 values with `cp_gather_and_upconvert_fp8_kv_cache`. This PR states that read-back as a Definition and holds vLLM's own kernel to it.

- **`verity/definitions/registry/fp8_ds_mla_upconvert.py`.** This is a new module with a new Definition, `Fp8DsMlaUpconvertRow_v1{DN, DR}`, with the signature `(nope: DN e4m3, scale: DN/128 f32, rope: DR bf16) -> DN + DR bf16`.
  - NoPE value d is `F2fpBf16(F32Mul_v2(E4m3ToF32(nope[d]), scale[d / 128]))`. The RoPE words are copied as stored.
  - The docstring cites the pin's source lines and the kernel's sm_90 SASS:
    - per value, `F2FP.F16.E4M3.UNPACK_B`, then `HADD2.F32` (f16 to f32), then a plain `FMUL` (no `.FTZ`), then `F2FP.BF16.F32.PACK_AB`;
    - no `FFMA`, so the multiply and the conversion don't contract.
  - The module also has:
    - a numpy kernel registered against the Definition;
    - `split_rows` for the byte layout;
    - `reference_word`, an exact rational reference;
    - `HAND_VECTORS`, single values worked by hand: the largest and smallest codes, both zeros, the NaN codes, a scale that is not a power of two, ties to bf16 both ways, a double rounding, subnormal products (including on ties), a subnormal scale, overflow, and `0 × inf`;
    - `EDGE_SCALES`.
  - Its conformance text says `model-untested` and names the capture as its hardware check, as the registry's other read Definitions do. I left that label alone now that the capture has run.
  - No registered Definition's words change.
- **`fp8_ds_mla_upconvert.circuit-check.py`.** The sidecar, with bindings at GLM-5.3's row (`DN=512, DR=64`) and a small one (`DN=128, DR=8`). Its update pass wrote no pins.
- **`verity/definitions/tests/test_fp8_ds_mla_upconvert.py`.** It checks:
  - each hand vector through the exact reference, the Definition and its numpy kernel;
  - all 256 codes under 16 edge scales against the exact reference;
  - that the f16 hop is exact on every non-NaN code;
  - the 656-byte layout;
  - the numpy kernel against the Definition (`self_check`);
  - the refusals.
- **`catalog_ids.json`** lists the new id.
- **`adapters/vllm/tests/properties/fp8_read_back_capture_gpu.py`.** A GPU capture that holds every bf16 word vLLM's kernel writes to the Definition:
  - It runs planted rows through the op directly, behind a shuffled block table and at source offsets, into a poisoned workspace.
  - Then it serves the GLM-5.3 stand-in with an fp8 cache, with its latent norm gain set per 128-value group so the write quantises groups of very different sizes. Every call of the op is wrapped there, and the rows it read are found by the kernel's own task map.
  - It also sweeps prompt lengths around `index_topk` to record which lengths take the route.
  - It catches no exception wholesale. A phase that raises ends the script non-zero with its traceback in the run's log, and the `capture.json` it wrote before that phase says `ok: false`. The library record alone catches by name (`ImportError`, `OSError`), so the rows' verdict is still written when vLLM, its library or `cuobjdump` can't be read.
  - Its compare phase also records the `_C_stable_libtorch.abi3.so` the run loaded: its sha256 against a given one, whether it registers the page kernel under the name its SASS was read by, and the kernel's SASS opcode counts where `cuobjdump` can list them.
- **`adapters/vllm/tests/properties/test_fp8_read_back_capture.py`.** The capture's CPU test. It checks:
  - the task map against a page-by-page emulation of the kernel;
  - that the planted rows carry every hand vector and every code;
  - that the compare finds a flipped word and refuses an empty set;
  - the SASS opcode count on a literal listing;
  - that the hooked layer is the one whose impl reads the context back;
  - that the kernel's name is found in a library's bytes.

## The capture on node A

**At which head it ran.** No capture has run at this PR's head `0978f5af04`. The last ran at `79cb0b3e28`. `0978f5af04` changes only how the capture fails: the two blanket catches in its capture phase are gone, and the library record catches `ImportError` and `OSError` by name. It computes the same rows, comparisons and record fields. Where before a raising phase was recorded and the script went on, now the script stops there with its traceback, over a record that says `ok: false`. The capture of record runs at the head in the next card row.

Run `r20261112-192752-fca0`, at `79cb0b3e28`, on card 3 with cores 34-43. The card is an H200 (capability 9.0), vLLM is `0.28.1rc1.dev472+gd9105ea80`, and the stand-in's attention is `FlashMLASparseImpl` over an `fp8_ds_mla` cache with 64-row blocks. The run exited 0.

`r20261112-174811-7ee1`, at the head's parent `185d1e34ae`, gave the same rows, counts and steps.

| Rows | Compared | Differ |
|---|---|---|
| Planted, held to the Definition's evaluator | 1,413 | 0 |
| Served, all 7 layers, held to the Definition's numpy kernel inside the capture | 127,274 | 0 |
| Served, first layer, held to the Definition's evaluator | 18,182 | 0 |

- **Planted rows.** 1,060 rows go through the op as three requests at source offsets 0, 37 and 69, and the first 353 again without `seq_starts`. No output row was left poisoned and none was unmapped.
  - They hold every hand vector, every E4M3 code under the edge scales, and uniform bytes.
  - Counted in the rows: 8,591 NaN codes, 47 subnormal scales, 42 non-finite scales, all 256 scale exponents, 9,771 subnormal products, 10,622 infinite products and 12,428 NaN products.
- **Served rows.** The pin's own write produced them, from 19 requests over 53 steps. The op was called 154 times, 22 times per layer.
  - Counted in the first layer's rows: 366,168 subnormal codes, 34,741 zero codes, 28,046 subnormal products and 8 scale exponents. They hold no NaN code and no non-finite scale; the planted rows cover those.
  - Every served call had `seq_starts` of 0. The planted rows cover the other offsets.
- The first try, `r20261112-173411-207c` at `ef693941ef`, compared the same planted rows (1,413, 0 differing) and no served row. My hook looked for `_compute_prefill_context` on the attention layer, and the pin has it on the layer's impl. `185d1e34ae` fixes that.

## Which build node A runs

Node A's library is not the one whose SASS I first read. It is the library of the pin's cu129 wheel, and the kernel's SASS in it is the same.

| | Wheel sha256 | `_C_stable_libtorch.abi3.so` sha256 | Bytes |
|---|---|---|---|
| The pin's default wheel, read first | `0198513e…9120` | `6acb4075…ac42` | 81,651,368 |
| The pin's cu129 wheel, which `pod_bootstrap.sh` pins and installs | `7aa52ac7…bbb3` | `683d3249…7471` | 506,418,112 |

- The capture recorded the library node A loaded: sha256 `683d3249f3a3e244105fcb9d966bc3979a3c20e1ad045b745bbee05650b37471`. I downloaded the cu129 wheel on the VM and hashed its library: the same as node A's. `r20261112-174811-7ee1` was given the default wheel's library hash to check against and recorded `matches: false`; the run at the head was given the cu129 wheel's and recorded `matches: true`.
- I then listed the kernel's sm_90 SASS in both libraries on the VM, with pip's `cuobjdump` and `nvdisasm`. The two listings are the same 336 instructions, operands included: 16 `F2FP.F16.E4M3.UNPACK_B`, 16 `HADD2.F32`, 16 `FMUL`, 8 `F2FP.BF16.F32.PACK_AB`, no `FFMA`, no `FMUL.FTZ`, and no `.FTZ` outside the index's `F2I`.
- The listing was not made on node A. Its `cuobjdump` has no `nvdisasm` and exited 1. What the run shows on the node is the file's hash and that it registers the page kernel under the same name.
- The Definition's docstring says "the pin's `_C_stable_libtorch.abi3.so`" and names no wheel. It is true of both.

## Which requests take the route

Each condition is marked read (in the pin's source) or run (in `r20261112-192752-fca0`, the first attention layer, tensor-parallel 1, steps of at most 512 tokens).

A request's cached rows are read back in a step when all of these hold:

1. **The cache is `fp8_ds_mla`.** Read: `mla_attention.py:2833`. Run: the engine chose it under `kv_cache_dtype="fp8"`.
2. **The request is a prefill in the step's split.** Read: `sparse_mla_attention.py:273-278`, `v1/attention/backends/utils.py:803-839`, `v1/attention/backends/mla/flashmla_sparse.py:265` and `:280-292`.
   - FlashMLA's sparse backend wants uniform decodes, with a threshold off sm_100 of 128 tokens for 16 or 32 query heads per GPU and 256 otherwise. GLM-5.3 has 64 heads, so it is 256 at tensor-parallel 1 and 8.
   - If the step's first request has more tokens than the threshold, every request is a prefill.
   - Otherwise the prefills are the requests from the first one whose token count differs from the first request's. A step whose requests all have one count has no prefill.
   - With prefill context parallel on, a request that is still prefilling also counts as a prefill (`sparse_mla_attention.py:276`). Read, not run.
   - Run: alone in a step, chunks of 88, 64, 52, 28 and 1 tokens were not prefills, and chunks of 476 to 512 tokens were. Beside one-token decodes, chunks of 8, 14, 74, 100, 137 and 174 tokens were prefills: their contexts were read back.
3. **The step attends densely.** Read: `sparse_mla_attention.py:285-310`, `mla_attention.py:1113-1136`, `sparse_mla_attention.py:80-106`.
   - The largest computed-plus-scheduled length over the step's prefills is at most `index_topk` (2048), and `attention_config.sparse_mla_force_mqa` is off.
   - The masked route is refused off sm_100, so on an H200 a prefill is dense or top-k.
   - The layer has a dense prefill backend (`mla_attention.py:581-614`, `sparse_mla_attention.py:287`). Without one a step gets no prefill metadata and attends top-k only. Run: the stand-in's layer had one, since its steps went dense.
   - Run: dense at lengths up to 2047 and 2048. Not run: a prefill past 2048. The sweep's chunks that ended past 2048 had 1 or 52 tokens and were not prefills.
4. **Some prefill in the step has computed tokens**, from a prefix-cache hit or an earlier chunk. Read: `sparse_mla_attention.py:239-247`, `mla_attention.py:1967-1968`. Run: the rows read in a step were the sum of the computed tokens of its prefills (766 = 256 + 510, 1,019 = 763 + 256), and none where no prefill had any.
5. **Decode context parallel is off.** Read: `mla_attention.py:3104-3120`. With it on, the same op is called from `_context_parallel_compute_prefill_context` (`:2955`). Not run.

By request length, one request at a time (run):

| Prompt tokens | Steps that read rows back | Rows read |
|---|---|---|
| 600 | none: the second chunk has 88 tokens | 0 |
| 1500 | chunks 2 and 3 | 512, 1,024 |
| 2047, 2048 | chunks 2, 3 and 4 | 512, 1,024, 1,536 |
| 2049, 2100 | chunks 2, 3 and 4; the tail past 2048 is not a prefill | 512, 1,024, 1,536 |
| 1500, 2048, 2049 again, after a prefix-cache hit of 1,472, 1,984 and 2,048 tokens | none: 28, 64 and 1 tokens are left | 0 |

## circuit-check

Run `r20261112-134850-f540`, `circuit-check fp8_ds_mla_upconvert.circuit-check.py --update-pins`:

~~~text
ok        definition Fp8DsMlaUpconvertRow_v1{DN=128,DR=8}  failures=0 unchecked=0 q_call=384u/10240b
ok        definition Fp8DsMlaUpconvertRow_v1{DN=512,DR=64}  failures=0 unchecked=0 q_call=1536u/40960b
circuit-check: vectors a realization compared with the reference on: 2 random, 0 edge-only, 0 none
circuit-check: 2 target(s), 0 failure(s), 0 unchecked (warnings, which never fail) on 0 Definition(s)
~~~

At `ef693941ef`, run `r20261112-152637-c6d8` (`circuit-check fp8_ds_mla_upconvert.circuit-check.py`, without `--update-pins`) gave rc 0:

~~~text
ok        definition Fp8DsMlaUpconvertRow_v1{DN=128,DR=8}  failures=0 unchecked=0 q_call=384u/10240b (0.3s: correspondence 0.2)
ok        definition Fp8DsMlaUpconvertRow_v1{DN=512,DR=64}  failures=0 unchecked=0 q_call=1536u/40960b (0.8s: correspondence 0.7)
circuit-check: vectors a realization compared with the reference on: 2 random, 0 edge-only, 0 none
circuit-check: 2 target(s), 0 failure(s), 0 unchecked (warnings, which never fail) on 0 Definition(s), report /workspace/research/runs/r20261112-152637-c6d8/out/circuit-check.json
~~~

In `r20261112-134850-f540`, the numpy kernel was compared on 64 vectors per binding (38 random), with 0 mismatches. The partition holds 128 and 512 output units, with no redundant or free gates.

Both runs called `uv run circuit-check` with no `uv sync` before it.

`79cb0b3e28` changes one word of the Definition's conformance text (`sm_90` to `cc 9.0`) and none of its words. With the sync, at `79cb0b3e28`, `r20261112-192652-7776` gave rc 0:

~~~text
circuit-check: 2 target(s), 0 failure(s), 0 unchecked (warnings, which never fail) on 0 Definition(s), report /workspace/research/runs/r20261112-192652-7776/out/circuit-check.json
~~~

The head `0978f5af04` changes no file circuit-check reads.

## Runs

Card runs, on node A (`vy-spot-glm53`), card 3, cores 34-43, one lease each under `GPU_LEASE_WHO=circuits-fp8readback`. The card was idle at each start, so none waited.

| Run | At | Lease held | What I read |
|---|---|---|---|
| `r20261112-173411-207c` | `ef693941ef` | 40 s | Planted: 1,413 rows, 0 differ. Served: no row compared; the hook found no layer (fixed in `185d1e34ae`). |
| `r20261112-174811-7ee1` | `185d1e34ae` | 47 s | Exit 0. The figures in "The capture on node A". Library `matches: false` against the default wheel's hash. |
| `r20261112-192752-fca0` | `79cb0b3e28` | 48 s | Exit 0. The same figures. Library `matches: true` against the cu129 wheel's hash. The compare ended at 19:32:13Z, off the card. |

Check-slot runs:

- **`r20261112-175217-033e`, with the sync**, at `185d1e34ae` on vy-nebius-cpu-8: `uv sync -q --all-packages --extra torch-cpu && uv run tools/verity/check/suites.py verity-vllm repository`. It waited 2m18s for slot r2 (17:52:40Z to 17:54:58Z) and held it until the run ended at 19:15:49Z: 4,851 s by my subtraction.
  - repository: passed, 352 passed.
  - verity-vllm: **failed**, 15,015 passed, 1 failed, 485 skipped. The one failure is `tests/lint/test_p08_facts.py::test_no_new_violations`.
  - That failure is this PR's. The lint reads string literals in `verity/definitions/` too, and the Definition's conformance text named `sm_90`. `79cb0b3e28` writes `cc 9.0` there, as the registry's other conformance texts do.
- **At the head `0978f5af04`, with the sync** (`uv sync -q --all-packages --extra torch-cpu` before `suites.py`):
  - `r20261112-221618-0259`, `repository`: passed, 352 passed, 0 failed. It waited 11 s for slot r2 on vy-nebius-cpu-7 and held it from 22:16:53Z to the run's end at 22:23:59Z.
  - `r20261112-222441-fadb`, `verity-vllm`: running at 22:40Z: it took slot r4 on vy-nebius-cpu-5 at 22:34:30Z after 9m24s. **Nothing is read from it yet.**
- **`r20261112-192652-7776`, with the sync**, at `79cb0b3e28` on vy-nebius-cpu-7: circuit-check on the sidecar, then `suites.py verity/definitions circuit-check verity-vllm repository`.
  - circuit-check: rc 0, 2 targets, 0 failures. repository: 352 passed.
  - I cancelled it at 22:15Z, before it reported `verity/definitions`, the circuit-check suite or verity-vllm. It had waited 56 min for slot r2 and held it for 113 min. It averaged about one core, and its head was superseded.
- `r20261112-192536-de27`: the same job launched with the default 3,600 s timeout, which verity-vllm outlasts. I cancelled it while it waited for a slot. It ran nothing.
- `r20261112-134850-f540`: circuit-check with `--update-pins` on both bindings, no sync. Both are ok, and no pin was written. The run's rc 1 came from the job's own `cp` of the pins file, which didn't exist. It waited 1m49s for check slot r10 and held it for about 17 s.
- **Run without `uv sync`, so not in the suites' own environment**: `r20261112-140436-e34f` at `f4f4110287` and `r20261112-152637-c6d8` at `ef693941ef`. Each ran circuit-check on the sidecar and `suites.py verity/definitions circuit-check verity-vllm repository`.
  - Each reported verity-vllm with 570 failed and 82 errors. I wrote here that none was this PR's and took them to be on `main`. That was wrong: `test_p08_facts.py::test_no_new_violations` is in `r20261112-152637-c6d8`'s list, and it was this PR's. With the sync, it is the only failure.
  - What they still show: circuit-check's report on the sidecar, and passes of `verity/definitions` (942 passed, 17 skipped), the circuit-check suite (428 passed, 5 skipped) and repository (352 passed), in an environment that is not the suites' own.
  - `r20261112-140436-e34f` waited 58 s for slot r0 and held it from 14:05:53Z until about 15:24Z (the end is from status, not a record). `r20261112-152637-c6d8` took slot r0 1 s after its job started and held it for 4,463 s.

On the VM, at the head `0978f5af04`:

- `python3 adapters/vllm/tools/ratchet_gate.py origin/main HEAD --json` (base `5f0c008aa0`): `failures: []`, 0 `# noqa` lines added and 0 removed. At `79cb0b3e28` it failed on the three `# noqa: BLE001` lines of the capture's catches.
- `adapters/vllm/tests/properties/test_fp8_read_back_capture.py`: 47 passed.
- `ruff check` on the capture file: passes.

On the VM, at `79cb0b3e28`:

- `adapters/vllm/tests/lint/test_p08_facts.py`: 3 passed. At `185d1e34ae` it fails with the one violation above.
- `adapters/vllm/tests/lint/test_p11_names.py` and `test_p12_shared_keys.py`: 3 passed each.
- `verity/definitions/tests/test_fp8_ds_mla_upconvert.py`: 36 passed.
- `adapters/vllm/tests/properties/test_fp8_read_back_capture.py`: 47 passed.
- `tests/lint/test_repo_retired_terms.py`: 13 passed. It warns about bookkeeping elsewhere in the tree, and no file this PR adds is named.
- Both wheels of the pin downloaded and hashed, and the kernel's sm_90 SASS listed in each library with pip's `nvidia-cuda-cuobjdump` and `nvidia-cuda-nvdisasm` ("Which build node A runs").

## Not run

- **The capture at the head `0978f5af04`.** It ran at `79cb0b3e28`, and the next card row (Fri 13 Nov 03:00Z to 04:25Z, card 3, cores 30-39) runs it at the head. Its lease now declares those cores (`--cores`). The three leases of today's runs declared none and were pinned only by `taskset`.
- **The `verity/definitions` and circuit-check suites with the sync**, at any head. They passed only in runs without it, at `ef693941ef`. The synced run that held them was cancelled before reporting them.
- **A prefill past `index_topk`.** No step of the capture had a prefill whose length passed 2048, so the top-k side of condition 3 is read, not run.
- **The SASS listing on node A itself.** The node has no `nvdisasm`. That node A's kernel is the listed one rests on its library's sha256 equalling the cu129 wheel's.
- **Decode context parallel, prefill context parallel, tensor-parallel 8, and GLM-5.3's own weights.** The capture served the stand-in at tensor-parallel 1.

<!-- CURSOR_AGENT_PR_BODY_END -->

<div><a href="https://cursor.com/agents/bc-12ba73ba-00ae-53ad-bb1e-9e67c00865c1?cursor_ref=pr_footer&cursor_cta=open_in_web"><picture><source media="(prefers-color-scheme: dark)" srcset="https://cursor.com/assets/images/open-in-web-dark.png"><source media="(prefers-color-scheme: light)" srcset="https://cursor.com/assets/images/open-in-web-light.png"><img alt="Open in Web" width="114" height="28" src="https://cursor.com/assets/images/open-in-web-dark.png"></picture></a>&nbsp;<a href="https://cursor.com/background-agent?bcId=bc-12ba73ba-00ae-53ad-bb1e-9e67c00865c1&cursor_ref=pr_footer&cursor_cta=open_in_cursor"><picture><source media="(prefers-color-scheme: dark)" srcset="https://cursor.com/assets/images/open-in-cursor-dark.png"><source media="(prefers-color-scheme: light)" srcset="https://cursor.com/assets/images/open-in-cursor-light.png"><img alt="Open in Cursor" width="131" height="28" src="https://cursor.com/assets/images/open-in-cursor-dark.png"></picture></a>&nbsp;</div>

