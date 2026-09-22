"""EXPERIMENTAL (lane C-builder, R8 2026-09-18): advice-compressed ahead-of-time Program construction, `P = G(φ, a)`.

    φ  base context known before execution (model config C, vLLM profile pins, workload specification)   -> `phi.py`
    h  structured hints: the residual the deterministic predictor could not know                           -> `hints.py`
    G  = BuildProgram(φ, DecodeAdvice(φ, a)); the bit-level codecs `h <-> a` are lane C-codec's                -> `build.py`
    EncodeHints(φ, record) -> h   (predict-then-correct against the observed record)                        -> `encode.py`
    predictor: φ -> structural execution schedule (steps, rows, KV lengths, sampled rows)                   -> `predict.py`
    check: built Program == accepted Program by digest, else an exact structural diff                       -> `check.py`
    harness CLI (encode / decode in a clean directory / check)                                              -> `run.py`

Nothing here is promoted: no existing checker, gate, fold or frozen interface is touched.  The builder reuses the
registry's Definition constructors (`verity_vllm.registry.b1`) and the IR's `Builder` / `_replayer` exactly as the
capture fold does, driven by a *predicted* call sequence instead of the dispatcher trace.
"""
