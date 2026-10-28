# The encoded stack in vLLM, and its benchmark arms

Llama-3.1-8B FP8 decode in vLLM with every linear weight, the lm_head and the paged KV cache held only as codec words,
receipted by proof of useful bandwidth's (PoUB's) sketches (`stack.py`, whose docstring lays out each part), and the
plain stack, the same engine over the same kernels' plain forms. `bench_arm.py` makes them arms of the shared
`verity-bench vllm` driver (`benchmarks/harness/README.md`); `benchmarks/harness/setup/vllm_encoded.sh` builds their GPU
libraries from the dispatched commit.

## Arms

| object | id | what it is |
|---|---|---|
| `PLAIN` | `bandwidth-plain` | the plain stack with the top-up; the token control of every other arm |
| `AUDIT` | `bandwidth-audit` | the encoded, receipted stack with the top-up |
| `PLAIN_NO_TOPUP`, `AUDIT_NO_TOPUP` | `bandwidth-plain-no-topup`, `bandwidth-audit-no-topup` | the same, the top-up never started |
| `PLAIN_SPANS`, `AUDIT_SPANS` | `bandwidth-plain-spans`, `bandwidth-audit-spans` | the same with the top-up, each step timed by part |

The top-up is the fill-sk16 filler (`benchmarks/kv_filler/topup.cu`) over the paged KV pool on a low-priority stream,
on every SM or on a partition of its own (`topup_sms`, below).
`bandwidth-plain` and `bandwidth-audit` are the credited lines (`bench_arm.CREDITED`): a table cites them as the audit's
cost. Every other arm is an ablation and measures cost only.

## Settings

`BandwidthArm(id, encoded=..., control=..., topup=None, spans=None, kv_receipts=None, codec_rounds=None,
topup_sms=None)` builds an arm. A setting left None is read from the environment when the arm installs, so a spec's
`[env]` sets it for every arm that leaves it open:

| argument | variable | default | |
|---|---|---|---|
| `topup` | `VERITY_VLLM_BENCH_TOPUP` | `on` | off: the top-up is never started (ablation only) |
| `spans` | `VERITY_VLLM_BENCH_SPANS` | `off` | on: per-part timing spans (below) |
| `kv_receipts` | `VERITY_VLLM_BENCH_KV_RECEIPTS` | `on` | off: the step commits the weights' receipts only |
| `codec_rounds` | `VERITY_VLLM_BENCH_CODEC_ROUNDS` | `3` | the codec's rounds k |
| `topup_sms` | `VERITY_VLLM_BENCH_TOPUP_SMS` | `0` | F > 0: the top-up on F SMs of its own, the engine on the rest |

`VERITY_VLLM_BENCH_FILLER_CTAS` (32) and `VERITY_VLLM_BENCH_SELECT_TRIALS` (5) are environment-only.

**Partition.** With `topup_sms` F > 0 the arm splits the SMs before vLLM starts (`benchmarks/kv_filler/green.py`,
`topup.cu`'s `sq_green`, green contexts): the top-up's stream is on the filler's F, and the step's stream, the stack's
side streams and every CUDA graph's capture are on the rest, whose count the device's SM count then reads as (it sizes
the stack's grids, and the attention's splits with them). The gate fails unless every stream of the step's ran within
its SMs (`partition.spread`) and every run's current stream was the step's. Set F for the token control too (the
environment does both): the splits change the attention's rounding, so a partitioned audit is gated against a
partitioned plain run. F rounds up to the driver's smallest partition, 8 SMs on the RTX PRO 6000. The default stays 0
because no F measured wins at both batch sizes: a partition that cuts the audit's time per output token also cuts the
top-up's reads, and one that raises the reads slows the engine, whose prefill KV encode then runs on fewer SMs.

**Credit.** The arm's gate carries `credit`: `credited` is true only for a credited id with the top-up on, the top-up
having read bytes in every measured run, and spans off; `refused` lists why not. A credited id refused fails its gate,
so the harness keeps no points and there is no line. The fix it names is the ablation's own id. An ablation under its own
id passes with `credited: false`. Under any setting the gate reads no timing: spans and filler bytes are reported, never
judged.

## Spans

With spans on, `step_spans.py`'s marks (`paged.cu`'s `pg_span`, one thread reading the GPU's global timer) sit between
the stack's kernels and are captured into vLLM's CUDA graphs with them. Each mark adds the time since its stream's last
mark to the part it names, on the device, so a replayed step costs no host sync. The arm's `spans()` returns the running
totals, which the harness turns into each rep's `spans` (in a result, a point's `span_totals`), and its `extras()`
divides them by the run's steps (`span_per_step.*`). Graph-replayed steps (`graph`) and eager ones (`eager`: prefill, a
mixed step) are kept apart.

The main stream's parts sum to its steps' time:

- `between`: vLLM's sampling, scheduling and input preparation.
- `commit`: the last pass's receipts closed: the selected KV slots' host read, their compaction and the step tree.
- `append_wait`: the wait for the last step's deferred KV append.
- `pass`: the pass's start (its seed, signs and the KV receipts' pass state).
- `model`: vLLM's own ops between the stack's (norms, RoPE, activation quantization, SiLU, residual adds).
- `linear`: a decode's weight linears. On the audit this is the codec decode, FP8 MMA and sketch receipts in one fused
  launch; on the plain arm, the plain GEMM.
- `attention`: the paged decode attention (the audit decodes its KV words inside it), or FlashAttention over a prefill
  step's gathered pages.
- `kv_receipts`: the layer's KV receipts.
- `gather`: a prefill step's pages decoded or copied into its scratch.
- `prefill_decode`, `prefill_gemm`: a prefill linear's weight decoded into its matrix, then its CUTLASS GEMM.
- `lm_head`: the lm_head, fused like `linear`.

Beside the main stream: `append_ms` (the deferred KV append on the second stream) and the top-up (`span.topup_busy_ms`,
its chunks' GPU time, and `span.topup_busy_share` over the main stream's time). Their effect on the step shows only as
the other parts' stretch, which `*_NO_TOPUP` separates.

A decode step's codec decode is fused with its MMA and sketches, so no mark splits them. The audit's `linear`,
`attention` and `lm_head` over the plain arm's are its decode and sketch cost. `kv_receipts=False` takes the KV sketches
out of `attention`. Each mark's own time is in the part it ends, and `span_per_step.<kind>.marks` counts them. The
marks' overhead is the spans arm's time per output token over its spans-off twin's, and it is a larger share of the
plain arm's time than of the audit's. So read a spans arm's parts, and the cost from the spans-off lines.

## Switching an ablation to these hooks

Name the arm objects in a spec's `arm_modules` and their ids in `arms`. Don't wrap `PLAIN` or `AUDIT`, pause their top-up
or set their settings' variables around an install:

~~~toml
arms = ["bandwidth-plain", "bandwidth-audit", "bandwidth-plain-no-topup", "bandwidth-audit-no-topup",
        "bandwidth-audit-spans"]
arm_modules = ["benchmarks.vllm_encoded.bench_arm:PLAIN", "benchmarks.vllm_encoded.bench_arm:AUDIT",
               "benchmarks.vllm_encoded.bench_arm:PLAIN_NO_TOPUP", "benchmarks.vllm_encoded.bench_arm:AUDIT_NO_TOPUP",
               "benchmarks.vllm_encoded.bench_arm:AUDIT_SPANS"]
~~~

An arm these objects don't cover is one line in a module of its own:

~~~python
from benchmarks.vllm_encoded.bench_arm import BandwidthArm

AUDIT_K1 = BandwidthArm("bandwidth-audit-k1", encoded=True, control="bandwidth-plain", codec_rounds=1)
AUDIT_NO_KV_RECEIPTS = BandwidthArm("bandwidth-audit-no-kv-receipts", encoded=True, control="bandwidth-plain",
                                    kv_receipts=False)
~~~
