"""The compiled executions a `TargetProfile` admits (``compilation``) and the vLLM facts each applies and declares
(`vllm/config/compilation.py`).

Keys are GP-01's COMPILATION_FACT_KEYS (mode, cudagraph_mode, capture_sizes, use_inductor_graph_partition, compile_cache) in the run
header's read-back form (`vllm_adapter.compilation_facts`). Under Inductor, `capture_sizes` (it depends on the row's batch) and
`compile_cache` (a process switch; cached and fresh Inductor graphs are the same code) are not declared, so not constrained: GM-01
compares only the keys both sides carry.

* `piecewise`: vLLM's production default compiled configuration (default Inductor config: fp32 carried through the fused regions),
  the panel's bi-compiled workloads. Its Program is P_compiled at Inductor-kernel granularity (S1/S2).
* `piecewise-emu`: the separate declared specialization with Inductor's emulate_precision_casts (the eager roundings performed inside
  the fused kernels; a different numerical computation). Not a panel workload unless added by a 7-field note.
* `breakable`: what vLLM serves for the architectures it defaults to breakable CUDA graphs (`DEFAULT_BREAKABLE_CUDAGRAPH_ARCHITECTURES`,
  GLM-5.3's among them). `VLLM_USE_BREAKABLE_CUDAGRAPH` forces CompilationMode NONE (config/vllm.py:721-737), so the eager kernels run
  captured per padded batch (compilation/breakable_cudagraph.py): a uniform decode step of at most max_num_seqs tokens as one FULL
  graph, the attention and the DSA indexer captured at the padded batch (their padding rows without keys), another step of at most
  512 tokens as PIECEWISE graphs with them as eager breaks on the real batch, and a larger step uncaptured (`targets.graph_launch`).
  Its capture sizes are applied and declared, since every captured launch-dependent selection reads them (`targets.graph_tokens`).
  They are vLLM's default on cc 9.0 at class G's max_num_seqs 256 (`VllmConfig._set_cudagraph_sizes`: up to min(2 * 256, 512), by 8
  below 256 and by 16 above).
"""

from __future__ import annotations

_SPLIT_ATTENTION = "splitting op (vllm::unified_attention_with_output) executed outside the CUDA graphs"

COMPILATIONS = {
    "piecewise": {"mode": "VLLM_COMPILE", "backend": "inductor", "cudagraph_mode": "PIECEWISE", "use_inductor_graph_partition": False,
                  "emulate_precision_casts": False, "attention": _SPLIT_ATTENTION},
    "piecewise-emu": {"mode": "VLLM_COMPILE", "backend": "inductor", "cudagraph_mode": "PIECEWISE", "use_inductor_graph_partition": False,
                      "emulate_precision_casts": True, "attention": _SPLIT_ATTENTION},
    "breakable": {"mode": "NONE", "cudagraph_mode": "FULL_AND_PIECEWISE", "capture_sizes": [1, 2, 4, *range(8, 256, 8), *range(256, 513, 16)],
                  "use_inductor_graph_partition": False, "emulate_precision_casts": False, "breakable": True,
                  "attention": "captured at the padded batch in a uniform decode step's FULL graph, else eager breaks "
                               "(eager_break_during_capture) on the real batch between captured segments"},
}
