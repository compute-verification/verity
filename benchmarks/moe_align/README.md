# moe_align_block_size

Before a Marlin MoE GEMM, vLLM's `moe_align_block_size` groups the router's (token, top-k slot) pairs by expert and pads
each expert's group to a multiple of the moe block `BM`. A pair is p = token × top_k + slot. `fused_marlin_moe` picks
`BM` from 8, 16, 32, 48 and 64 by the batch's rows (`marlin_moe_rows.moe_block`). At the integration's vLLM pin, with more
than 64 experts or at least 1,024 pairs, the op's second kernel (`count_and_sort_expert_tokens_kernel`) places each pair
at `atomicAdd(&offset[expert], 1)`. The order within an expert is the order the atomics run in, which varies from run to
run. That order decides which pairs share a Marlin block, and so where the GEMM's reductions are cut. The integration's
`verity_vllm.engine.moe_align` (`adapters/vllm/verity_vllm/engine/moe_align.py`) makes the op a function of its inputs
on the Marlin path, and this directory checks and times it on a GPU.

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
`moe_align.reference` is the definition in Python. The integration's tests (`adapters/vllm/tests/engine/test_moe_align.py`)
hold it to the circuits' `moe_align_ranks` and `moe_align_blocks` (`verity.definitions.numerics.moe_align_rows`), with the
pair index as the key.

## How it runs

`fused_marlin_moe` calls its module's `moe_align_block_size`, vLLM's wrapper of the op. `moe_align.install` replaces that
name with `moe_align.wrapper`, which allocates the same buffers and has `moe_align.Align` fill them; every other MoE path
keeps vLLM's op. `Align` runs Triton kernels (`verity_vllm.engine.moe_align_kernels`) on the inputs' stream, with no
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

## Using it

It is always on in the integration: importing the module arms it, so it is installed when vLLM imports its Marlin module.
The engine build and each tensor-parallel rank import it before vLLM builds the model. The integration's distribution
declares `moe_align.register` as a vLLM general plugin (`verity_moe_align` in `vllm.general_plugins`), so any vLLM command
in a venv with the integration installed loads it in every process, its own server included, with `VLLM_PLUGINS` unset
or naming `verity_moe_align`. In a venv that has the integration on `PYTHONPATH` but not installed, such as a pod's, put
the site directory `stress.py site --out DIR` writes on the `PYTHONPATH` the engine processes inherit too. The plugin
reads no environment variable, and `moe_align.installed()` is the `Align` a process's Marlin experts call, with its
calls on each path.

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
- `stress.py plugin` loads the integration's plugins as vLLM does, from a site directory `stress.py site` wrote. It
  calls the function `fused_marlin_moe` calls, eagerly and replayed from CUDA graphs (`plugin.json`).
- `stress.py time` writes `time.json`. Per row count, it records the device time of a call (CUDA events, queued behind a
  sleep) and the host's time to issue it. It covers vLLM's op and each `Align` variant (`--variants`), alone and through
  a wrapper (vLLM's, and `moe_align.wrapper`), and what `Align` adds to a decode step of `stress.LAYERS` calls.

On DeepSeek-V4-Pro's 8 x H200 node, served TP8 eager with the plugin loaded from a site directory:

~~~text
research run --on NODE --project verity --cwd source -- gpu-lease 8 --wait --mem-gb 64 -- bash benchmarks/moe_align/served.sh
~~~

On a node without the checkpoint, `DUMMY_LAYERS=N` serves its first N layers with dummy weights. The config and tokenizer
come from the hub at the manifest's revision, each held to the manifest's sha256. Any tensor-parallel size works, for
example two GPUs:

~~~text
research run --on NODE --project verity --cwd source -- gpu-lease 2 --wait -- env TP=2 DUMMY_LAYERS=5 bash benchmarks/moe_align/served.sh
~~~

The expert count, top-k and widths are the checkpoint's, but the routing is not, so the run's figures are labelled
`weights: dummy`. N = 5 keeps DeepSeek-V4-Pro's three hash-routed layers and two score-routed ones; it has no dense layer.

`served.py` builds one engine and refuses to go on unless every rank's Marlin experts call `moe_align.wrapper`, as their
module's global (the guard `moe_align.install` holds, read back from each rank). It then runs three phases:

- **`check.json`.** A wrap over each rank's `moe_align_block_size` holds every call to `moe_align.reference`, by MoE
  layer, through the prefill and decode of one request and of eight. It counts each rank's calls by step kind and
  `Align` path, and requires that every call reach the rank's `Align` and none reach vLLM's own aligners.
- **`decode.jsonl`.** The decode step at 1, 16, 64 and 256 rows, with the canonical op and with vLLM's, interleaved, and
  each run's tokens.
- **`host.json`.** The host's time to issue a call of each op on each rank.

`summary.json` gathers the three.
