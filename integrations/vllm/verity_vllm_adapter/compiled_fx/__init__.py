"""Compiled-preserving replay through a post-grad FX pass (CONTRACT.md §4 rung 1 ``fx-graph-pass``,
§16.1 rungs (a)/(b)/(c)); worker W1d.

Mechanism, in one paragraph: the replay process builds the *same compiled* vLLM engine as production
(Inductor, ``custom_ops=['none']``, piecewise split at the attention ops) and appends one extra
post-grad FX pass to vLLM's ``PostGradPassManager`` (replay process only; production never loads it).
The pass inserts a side-effecting, output-less custom op ``verity::record(Tensor x, int id)`` after
every buffer that Inductor materialises anyway (outputs of the user-Triton GEMM kernel, inputs of
extern/user kernels, piecewise-graph outputs, activation placeholders). ``verity::record`` is
registered as side-effectful (``torch.fx.node.has_side_effect``) so it survives DCE, and it mutates
nothing, so it adds only a read dependency on an existing buffer: the numerical kernels Inductor
generates are unchanged, which :mod:`.kernels` proves by comparing the generated Triton kernel sources
of production's and replay's compile caches. Two op bodies (:mod:`.record_op`): the *host* one copies
the buffer into a pinned host arena per call (replay with ``cudagraph_mode=NONE``, rungs (a)/(b)); the
*device* one copies into a preallocated device ring slot per (piece, id) so the copy is captured inside
vLLM's CUDA graphs and replays with them, the slots being drained after each step (production's graph
configuration unchanged, rung (c)). :mod:`.fx_pass` writes the ``id`` -> canonical identity table from
the FX node metadata at pass time.

"In memory anyway" is necessary but not sufficient (B1, Qwen2.5-1.5B): the record must also not make
Inductor materialise the buffer *earlier* than production did or materialise a second copy of it. Hence
(1) the record call is placed after the buffer's last in-graph reader (``placement='late'``), (2) the op's
Inductor lowering is registered explicitly (``record_op.install_inductor_lowering``) so the op is not in
``needs_realized_inputs`` (the implicit fallback would make ``run_node`` realise multi-user pointwise
producers at their own lowering) and (3) that lowering records an extern consumer's fixed-stride clone of
an unrealised value instead of realising the value again. See ``docs/vllm-poc/compiled-fx-capture.md`` §2b.
"""
