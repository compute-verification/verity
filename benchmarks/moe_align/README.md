# moe_align_block_size

Before a Marlin MoE GEMM, vLLM's `moe_align_block_size` groups the router's (token, top-k slot) pairs by expert and pads
each expert's group to a multiple of the moe block `BM`. A pair is p = token × top_k + slot. `fused_marlin_moe` picks
`BM` from 8, 16, 32, 48 and 64 by the batch's rows (`marlin_moe_rows.moe_block`). At the integration's vLLM pin, with more
than 64 experts or at least 1,024 pairs, the op's second kernel (`count_and_sort_expert_tokens_kernel`) places each pair
at `atomicAdd(&offset[expert], 1)`. The order within an expert is the order the atomics run in, which varies from run to
run. That order decides which pairs share a Marlin block, and so where the GEMM's reductions are cut. `moe_align.py` makes
the op a function of its inputs.

## The canonical outputs

For `topk_ids` of P pairs over E experts, with c_e pairs on expert e, the three int32 outputs are written into the
buffers vLLM's wrapper allocates (`fused_moe/moe_align_block_size.py`), in the same format and with the same padding:

- `sorted_token_ids`: expert e's pairs start at `base[e]`, the sum of `round_up(c_e', BM)` over e' < e, in ascending pair
  index. Every other slot holds P.
- `expert_ids[b]`: the expert whose padded range holds slot `b × BM`. It is -1 from `num_tokens_post_pad / BM` up to
  `cdiv(len(sorted_token_ids), BM)`.
- `num_tokens_post_pad[0]`: the sum of `round_up(c_e, BM)` over all experts.

No expert owns a pair whose id, read as vLLM's kernel reads it (its low 32 bits, signed), lies outside [0, E), and none
owns one that `expert_map` sends to -1. vLLM's op writes the same buffers, except for the order within each expert.
`moe_align.reference` is the definition in Python. The tests hold it to the circuits' `moe_align_ranks` and
`moe_align_blocks` (`verity.ml.definitions.numerics.moe_align_rows`), with the pair index as the key.

## How it runs

`moe_align.register` is a vLLM general plugin. With `VERITY_MOE_ALIGN=canonical`, it replaces
`torch.ops._moe_C.moe_align_block_size` in every process with `moe_align.Align`. vLLM's wrapper looks the op up on each
call and still allocates the buffers. `Align` runs Triton kernels (`moe_align_kernels.py`) on the inputs' stream, with no
atomics and no host sync:

A pair's slot is `base[e]`, plus its expert's pairs before its tile of `TILE` pairs, plus its rank in the tile: the
tile's earlier pairs on its expert, counted by comparing the tile's pairs with each other, `COLS` at a time. Each program
places one tile, and the program that places an expert's last pair writes that expert's pad slots.

- A call of at most `SMALL_MAX` pairs is one launch, `align_small`, with a program a tile. Each program histograms all
  the call's pairs for `base` and the pairs before its tile for its offsets.
- A longer call is three launches over chunks of `CHUNK` pairs. `count_chunks` takes each chunk's histogram.
  `scan_chunks` turns them into each (chunk, expert)'s pairs in earlier chunks and each expert's total. `place_chunks`
  runs a program a tile of `LONG_TILE` pairs, which adds its chunk's pairs before the tile.

The stores' slots are disjoint, so no barrier orders them, and every slot vLLM's kernel writes is written. The long path
allocates its workspace through PyTorch's allocator, so both paths can be captured in a CUDA graph.

`python moe_align.py site DIR` writes the plugin's entry point into `DIR`, and a server or replay with `DIR` on its
`PYTHONPATH` loads it. `VERITY_MOE_ALIGN_LOG=DIR` keeps each process's calls on each path in `DIR/moe_align.PID.json`.

## Using it

Run any vLLM command at the integration's pin from the pin's venv: a server, a replay or an in-process engine. Set the
variables in the environment its engine processes inherit, with `VLLM_PLUGINS` unset or naming `verity_moe_align`:

~~~text
python benchmarks/moe_align/moe_align.py site DIR
PYTHONPATH=DIR${PYTHONPATH:+:$PYTHONPATH} VERITY_MOE_ALIGN=canonical [VERITY_MOE_ALIGN_LOG=LOGDIR] COMMAND
~~~

## Measuring

On a GPU node:

~~~text
research run --on NODE --project verity --source . --cwd source -- bash benchmarks/moe_align/run.sh
~~~

`run.sh` builds a venv of the pin's vLLM wheel if `VENV` lacks one. It then runs these, all at DeepSeek-V4-Pro's routed
experts (E = 384, top-k 6) with the moe block Marlin picks:

- `stress.py check`, in one process and then in two at once. Each case is rows from 1 to 16,384, under uniform, skewed
  and same-six-experts routings, every `BM`, int64 and out-of-range ids, and an expert map. It runs through vLLM's op and
  through `Align` into poisoned buffers, every other repetition beside a GEMM on a second stream.
- `stress.py compare` writes `summary.json`. Per case, it records whether every `Align` output is the reference bit for
  bit, the distinct outputs of each op across processes and repetitions, and whether each vLLM output, put in canonical
  order (`moe_align.canonical`), is the reference.
- `stress.py plugin` runs from a site directory and loads the plugin as vLLM does. It calls the function
  `fused_marlin_moe` calls, eagerly and replayed from CUDA graphs (`plugin.json`).
- `stress.py time` writes `time.json`. Per row count, it records the device time of a call (CUDA events, queued behind a
  sleep) and the host's time to issue it. It covers vLLM's op and each `Align` variant (`--variants`), alone and through
  vLLM's wrapper, and what `Align` adds to a decode step of `stress.LAYERS` calls.

The tests (`tests/`) hold the reference, the plugin's install on stand-in ops, and the kernels in Triton's interpreter
where Triton is importable.
