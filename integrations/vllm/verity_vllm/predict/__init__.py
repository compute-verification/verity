"""Ahead-of-time prediction of the Build's Programs from a row's declared facts φ, without tracing, and its score against the traces.

`phi` (φ: checkpoint config, target, TP, shape, sampler, execution), `build` (BuildProgram(φ): the `Derived[vLLM*]` Program the
Build would derive, from a hand-written vocabulary), `score` (predicted vs traced, per row and granularity: digest, equivalence
verdict, first divergence), `schedule`/`hints`/`check` (the restored scheduler model and descriptor check of the capture era).

Nothing here is on a record path: a predicted Program is never written into a Build row, published as a Build Attempt, or read by
the Build, Match or Commit; `program.frontend.inputs_trace` forbids this package as a Build input.
"""
