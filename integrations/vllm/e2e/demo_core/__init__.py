"""E8 — CORE protocol demonstrator (EOD 2026-09-11): a very small registered ``verity.ir`` circuit of already-approved operations,
run through the real protocol engine (record_v5 layout, boundary + interior trees, ``partition_sample`` sampling, ``coverage_v1`` builders,
the SP1 approved guest, ``verifier_v3`` shipped entry points).

LABEL: **generic protocol demonstration**. This is NOT vLLM inference and is never reported as such. The circuit (`circuit.py`) is
``DemoCore_v1{N,R}``: R chained stages of ``SiluMulBf16`` / ``Bf16AddF2fp`` over BF16 words — the two elementwise kinds whose checker lives in
the approved guest ``9fd00e60`` (family ``veritor.elem-bf16@1``) and whose obligation builder is registered in ``coverage_v1``.
"""
