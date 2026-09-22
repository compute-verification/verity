"""W4c diagnostics (CONTRACT §16.6): capture op-boundary values of one B0 request under eager and
compiled execution, walk them in topological order, and locate the first differing value.

Modules
-------
* ``capture_modes``  (pod, GPU): one process per execution mode; eager values via ``nn.Module``
  forward hooks, compiled values via Inductor's ``generate_intermediate_hooks`` (buffers already
  materialised between fused regions; no kernel is changed). Also dumps the post-grad FX graph of
  every compiled piece (node name -> module stack / aten op) and copies the generated wrapper +
  Triton kernel sources out of the compile cache.
* ``compare_modes``  (CPU): matches eager op outputs to compiled buffers and reports the first
  divergent canonical value with expected/actual bit patterns.
* ``kernel_repro``   (pod, GPU): the standalone reproducer driver (also copied into the bundle).
"""
