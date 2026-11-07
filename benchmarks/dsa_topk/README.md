# DSA top-k

The DeepSeek sparse attention (DSA) indexer selects, for each query row, the `k` keys (`k = 2048` for GLM-5.3 and
DeepSeek-V3.2) with the largest indexer logits, and the sparse attention then sums over the selected keys in the order
the selection lists them. At the integration's vLLM pin, the four selection kernels (`top_k_per_row_prefill`,
`top_k_per_row_decode`, `persistent_topk`, `cooperative_topk`) write a row of at most `k` keys as `0, 1, ..., n - 1`
and then `-1`, the same every run. A longer row's index positions come from shared-memory atomics, so its order varies
run to run, and keys tied at the k-th logit are admitted in atomic order, so its set can vary too. `dsa_topk.py` makes
every row a function of its logits.

## The canonical row

For a row of `n` keys with float32 logits `v[0..n)`:

- **Key.** `key(v)` is the float's 32 bits as a signed integer made order-preserving: a NaN becomes `-inf`'s bits, both
  zeros become `+0`, then `b` if `b >= 0` else `b ^ 0x7FFFFFFF`. Keys order exactly as the floats do, with `-0 == +0` and
  every NaN below every number.
- **Selection.** For `n > k`, the `k` largest pairs `(key(v[i]), i)` in lexicographic order: keys above the k-th key
  are all selected, and a tie at the k-th key goes to the larger indices.
- **Output.** The selected indices in ascending order, `k` of them, each relative to the row's first key. For `n <= k`,
  vLLM's row is unchanged: `0..n-1`, then `-1` up to `k`.

This is upstream vLLM's deterministic selector (its batch-invariant op library's `top_k_per_row_prefill_index_order`)
with one difference: that selector writes `-1` for a selected key whose logit is `-inf`, and this one keeps the index,
so that its set is the pin's set up to ties. `dsa_topk.reference` is the definition in Python, and
`dsa_topk.reference_rows` the same in torch.

## How it runs

`dsa_topk.register` is a vLLM general plugin. With `VERITY_DSA_TOPK=canonical` it wraps the four ops on
`torch.ops._C` in every process. Each wrapper runs vLLM's kernel, then, when a row of the call can be longer than `k`,
a Triton pass (`dsa_topk_kernels.py`) that rewrites only the rows longer than `k`, on the device, before attention
reads them. The pass reads the fast kernel's selection for a candidate k-th key, checks it against the whole row's
counts, and writes the canonical row from it; a row whose candidate fails the check is selected again by bisection over
the keys. So the output never depends on what the fast kernel wrote. The indexer runs eagerly between CUDA-graph
pieces (vLLM's `eager_break_during_capture`), so the wrappers run on every call. A `cooperative_topk` call whose
`max_seq_len` is at most `k` launches nothing: vLLM passes the batch's longest sequence there.

`python dsa_topk.py site DIR` writes the plugin's entry point into `DIR`; a server or replay with `DIR` on its
`PYTHONPATH` loads it. `VERITY_DSA_TOPK_LOG=DIR` keeps each process's counters in `DIR/dsa_topk.PID.json`: calls and
launches per op, and the rows rewritten, those with a tie across the k-th key, and those bisected.

## Measuring

On a GPU node with the pin's venv (`integrations/vllm/verity_vllm/ops/pod_bootstrap.sh`'s `venv312`):

~~~text
research run --on NODE --project verity --source . --cwd source -- bash benchmarks/dsa_topk/run.sh
~~~

`run.sh` runs `stress.py check` in four processes, two one after the other and two at once. Each runs every case at
GLM-5.3's shapes (prefill calls of up to 8,192 rows over up to 32,768 keys, decode calls of 1 to 512 rows over up to
33,000 keys, through each of the four ops) over six distributions of logits, unpatched and canonical, every other
repetition beside a GEMM on a second stream. `stress.py compare` then writes `summary.json`: per case, the distinct
outputs unpatched and canonical across processes and repetitions, whether the canonical output is the reference
(also from a garbage fast selection), whether the unpatched one selects the reference's set up to keys tied at the
k-th, and whether rows of at most `k` keys are unchanged. `stress.py time` writes `time.json`: the device time a call
(CUDA events, queued behind a sleep) and the host time, unpatched and canonical, per parameter variant (`--variants`).

`replay.sh` replays a served replay-ladder window with the plugin in every engine process (its header has the command).
The tests (`tests/`) hold the definition, the torch reference, the wrappers' dispatch on stand-in ops, and the kernels in
Triton's interpreter where Triton is importable.
