"""verity_vllm/properties/kernel_allowlist.py — PROTECTED (gate G1, census half).  The kernel-census allowlist: device kernels that a step may
launch without any pattern claiming them, because they compute no modeled value (copies, fills, the model runner's integer
metadata kernels).  Every entry needs a D71-style justification: what the kernel computes and why it is non-semantic under κ.

`census.py` reads THIS tuple; a profile's own `kernel_allowlist` is reported but never used for acceptance (editing a
profile allowlists no kernel).  New entries only through an approved decision in `out/capture/decisions.md`
(the sealed verifier counts a modified allowlist as `NEW ALLOWLIST ENTRIES` on the card).

`LAUNCH_COUNT_ALLOWLIST` is the launch-count gate's: declared kernels whose device launch count in a step may differ from
the number of the step's events that declare them, each with its reason.
"""
from __future__ import annotations

#: Device kernels accounted for by Transparent resolutions and infrastructure (census, DESIGN §8): copies, fills,
#: metadata index kernels, arange / cumsum of the model runner's input preparation, and the V2 model runner's
#: integer-metadata Triton kernels (resolved `Transparent(noop)` by `TransparentTritonMetadata`).  L5 refined this against
#: the profiler (census residue 0 on SmolLM2-135M and Qwen2.5-1.5B); entries are `re.search` patterns on Kineto kernel names.
KERNEL_ALLOWLIST = (
    "Memcpy", "Memset", "FillFunctor", "fill_kernel", "vectorized_elementwise_kernel", "elementwise_kernel",
    "unrolled_elementwise_kernel", "index_elementwise_kernel", "indexSelectLargeIndex", "indexSelectSmallIndex",
    "index_put", "arange", "cub::", "DeviceScan", "DeviceReduce", "CatArrayBatchedCopy", "masked_fill", "compare",
    "where", "cumsum", "_scan", "sort", "bitonic", "radix",
    # NOT here (D85): `ArgMaxOps` / `gather_kernel` are the sampler's compute kernels; TokenSelect patterns must claim them.
    # vllm/v1/worker/gpu/{buffer_utils,input_batch,block_table,sample}: public-metadata kernels (integer pointers only)
    "_apply_write_kernel", "_prepare_prefill_inputs_kernel", "_prepare_pos_seq_lens_kernel",
    "_combine_sampled_and_draft_tokens_kernel", "_gather_block_tables_kernel", "_compute_slot_mappings_kernel",
    "_get_num_sampled_and_rejected_kernel",
    # vllm/v1/worker/gpu/input_batch.py `_post_update_kernel`: the runner's integer request-state update, launched after
    # `sample()` returns (outside the observer's step window; its effect is the next StepStart's recorded rows) [L5 census]
    "_post_update_kernel",
)

#: Declared name -> why its launch count may differ from its declaring events.  An entry lets an untraced launch of that
#: kernel overwrite a Call's output unseen, so its reason must hold for every launch on every row of record.
LAUNCH_COUNT_ALLOWLIST: dict[str, str] = {}
