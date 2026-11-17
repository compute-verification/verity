# The encoded stack in vLLM, and its benchmark arms

Llama-3.1-8B FP8 decode in vLLM with every linear weight, the lm_head and the paged KV cache held only as codec words,
receipted by PoUB's sketches (`stack.py`, whose docstring lays out each part), and the
plain stack, the same engine over the same kernels' plain forms. `bench_arm.py` makes them arms of the shared
`verity-bench vllm` driver (`benchmarks/harness/README.md`); `benchmarks/harness/setup/vllm_encoded.sh` builds their GPU
libraries from the dispatched commit.

## Try it

~~~sh
uv run tools/check/suites.py benchmarks/vllm_encoded
~~~

On the CPU it checks the arms, the layouts against the word codecs' references, the KV-append capture's references and
the step spans' accounting. It writes no result. On a GPU, `verity-bench vllm` runs an arm below and writes its record
(`benchmarks/harness/README.md`), and `bash benchmarks/vllm_encoded/stack.sh` with `MODE` one of `paged`, `splits`,
`window`, `rekey` and `serve` runs the encoded stack's gates and arms on one GPU, writing to `$RESEARCH_RUN_DIR/out`.

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

[`DESIGN.md`](DESIGN.md) holds each arm's settings, the SM partition, what the gate's credit requires, the spans' parts,
and how an ablation names these arms.
