# DSA top-k

The DeepSeek sparse attention (DSA) indexer selects, for each query row, the `k` keys (`k = 2048` for GLM-5.3 and
DeepSeek-V3.2) with the largest indexer logits, and the sparse attention then sums over the selected keys in the order
the selection lists them. At the integration's vLLM pin, the four selection kernels (`top_k_per_row_prefill`,
`top_k_per_row_decode`, `persistent_topk`, `cooperative_topk`) write a row of at most `k` keys as `0, 1, ..., n - 1`
and then `-1`, the same every run. A longer row's index positions come from shared-memory atomics, so its order varies
run to run, and keys tied at the k-th logit are admitted in atomic order, so its set can vary too. `dsa_topk.py` makes
every row a function of its logits.

## The canonical row

For a row of `n` keys with float32 logits `v[0..n)`, the canonical row is the registered selection's
(`verity.definitions.registry.dsa_topk`: `topk_set`, which `DsaTopKLongRow` checks) in ascending order:

- **Key.** `key(v)` is `convert_to_uint32_v2` of the float's 32 bits, the key the pin's kernels rank by: `~b` for a
  negative word, `b | 2^31` otherwise. Keys order as the floats do, with `-0` below `+0`, a positive NaN above `+inf`
  and a negative NaN below `-inf`. `dsa_topk.ordered` is the same key less `2^31`, as a signed integer.
- **Selection.** For `n > k`, the first `k` indices under (key descending, index ascending): keys above the k-th key
  are all selected, and a tie at the k-th key goes to the smaller indices.
- **Output.** The selected indices in ascending order, `k` of them, each relative to the row's first key. For `n <= k`,
  vLLM's row is unchanged: `0..n-1`, then `-1` up to `k`.

Where the pin's kernels are deterministic they select this set, in another order. Upstream vLLM's deterministic
selector (its batch-invariant op library's `top_k_per_row_prefill_index_order`) also writes ascending indices, under
another key and tie rule. `dsa_topk.reference` is the definition in Python, held to `topk_set` by the tests, and
`dsa_topk.reference_rows` the same in torch.

## How it runs

`dsa_topk.register` is a vLLM general plugin. With `VERITY_DSA_TOPK=canonical` it wraps the four ops on
`torch.ops._C` in every process. Each wrapper runs vLLM's kernel, then, when a row of the call can be longer than `k`,
a Triton pass (`dsa_topk_kernels.py`) that rewrites only the rows longer than `k`, on the device, before attention
reads them. The pass takes the fast kernel's smallest selected key as a candidate k-th key and reads the row once,
writing the keys at or above it in ascending order and counting them. With no tie across the k-th key the row stands;
with one, the row is written again from the counts; a row whose candidate fails the count check is selected again by
bisection over the keys. So the output never depends on what the fast kernel wrote. GLM-5.3's indexer
runs eagerly between CUDA-graph pieces (vLLM's `eager_break_during_capture`), so the wrappers run on every call; under a
plain CUDA graph the pass is captured with the op, and what the wrappers decide on the host reads only shapes. A `cooperative_topk` call whose
`max_seq_len` is at most `k` launches nothing: vLLM passes the batch's longest sequence there.

`python dsa_topk.py site DIR` writes the plugin's entry point into `DIR`; a server or replay with `DIR` on its
`PYTHONPATH` loads it. `VERITY_DSA_TOPK_LOG=DIR` keeps each process's counters in `DIR/dsa_topk.PID.json`: calls and
launches per op, and the rows rewritten, those with a tie across the k-th key, and those bisected.

## Using it

Any vLLM command at the integration's pin (a server, a replay, an in-process engine), from the pin's venv, with the
variables in the environment its engine processes inherit (`VLLM_PLUGINS` unset or naming `verity_dsa_topk`):

~~~text
python benchmarks/dsa_topk/dsa_topk.py site DIR
PYTHONPATH=DIR${PYTHONPATH:+:$PYTHONPATH} VERITY_DSA_TOPK=canonical [VERITY_DSA_TOPK_LOG=LOGDIR] COMMAND
~~~

`rows_long` above zero in a process's counters shows the plugin rewrote long rows there.

## Measuring

On a GPU node with the pin's venv (`adapters/vllm/verity_vllm/ops/pod_bootstrap.sh`'s `venv312`):

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
`stress.py plugin`, run from a site directory under `VERITY_DSA_TOPK=canonical`, loads the plugin as vLLM does and calls
each case through the indexer's own entry points (`plugin.json`).

`replay.sh` replays a served replay-ladder window with the plugin in every engine process (its header has the command).
The tests (`tests/`) hold the definition, the torch reference, the wrappers' dispatch on stand-in ops, and the kernels in
Triton's interpreter where Triton is importable.
