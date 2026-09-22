"""[R9 CB-C 2026-09-18] Builder bet C: Dynamo / vLLM compiled FX *before* Inductor fusion as the structure source
for the ahead-of-time padded Verity Program.  Experimental; nothing here is a production or accepted path.

Modules: ``dump_fx`` (pod driver: real pinned vLLM engine, captures the Dynamo graph via
``verity_vllm_adapter.compiled_fx.dynamo_capture``), ``lower`` (FX JSON -> Verity Program with generic per-op rules),
``compare`` (structural / unfolding comparator against the accepted eager Program; oracle only).
"""
