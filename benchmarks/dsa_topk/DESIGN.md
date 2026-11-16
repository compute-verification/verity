# DSA top-k: design

How the canonical row relates to vLLM's own selectors, how the plugin runs, and what the measurements and tests
cover. `README.md` says what the plugin is for and how to use it.

## The canonical row

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
`PYTHONPATH` loads it. `VERITY_DSA_TOPK_LOG=DIR` keeps each process's counters in `DIR/dsa_topk.PID.json`, from the
plugin's load on: calls and launches per op, and the rows rewritten, those with a tie across the k-th key, and those
bisected. vLLM kills its workers, which then run no exit handler, so a process writes its record over, in place, at
the end of every call: one write on the host, nothing synced to the disk. Its calls and launches are there whatever
ends it. A call is a wrapper's run on the host, so a call a CUDA graph replays adds rows and no call. The rows are the
device's own counters, read every 4,096 calls outside a capture and when the process exits in order; `rows_at` is the
calls it had made by then, so a killed process's rows are a lower bound.

## Measuring

`run.sh` runs `stress.py check` in four processes, two one after the other and two at once. Each runs every case at
GLM-5.3's shapes (prefill calls of up to 8,192 rows over up to 32,768 keys, decode calls of 1 to 512 rows over up to
33,000 keys, through each of the four ops) over six distributions of logits, unpatched and canonical, every other
repetition beside a GEMM on a second stream. `stress.py compare` then writes `summary.json`: per case, the distinct
outputs unpatched and canonical across processes and repetitions, whether the canonical output is the reference
(also from a garbage fast selection), whether the unpatched one selects the reference's set up to keys tied at the
k-th, and whether rows of at most `k` keys are unchanged. `stress.py time` writes `time.json`: the device time a call
(CUDA events, queued behind a sleep) and the host time, unpatched and canonical, per parameter variant (`--variants`).
`stress.py plugin`, run from a site directory under `VERITY_DSA_TOPK=canonical`, loads the plugin as vLLM does and calls
each case through the indexer's own entry points (`plugin.json`). `log_cost.py` needs no device: it times what keeping
the counters adds to a call on the host, a wrapper's time a call over stand-in ops with the record kept and without.

`replay.sh` replays a served replay-ladder window with the plugin in every engine process (its header has the command).
The tests (`tests/benchmarks/dsa_topk/`) hold the definition, the torch reference, the wrappers' dispatch on stand-in ops, the kept counters
(one write a call, and a killed process's counts in its file), and the kernels in Triton's interpreter where Triton is
importable.
