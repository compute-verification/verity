"""CB-A (R9, 2026-09-18): Bet A — `torch.export` of the ACTUAL pinned vLLM model class (meta device, no weights, no
numerics) -> `verity_vllm.frontend.derive` with generic per-op vLLM bindings -> Verity step body / padded request Program.

Experimental; nothing here is a canonical artefact. Forbidden construction inputs (the target `program.json`, the
execution record, the accepted digest, `verity_capture/experimental/aot/build.py`) are never imported or opened by the
construction path; `inputs_trace.py` records every file the construction opens so a reviewer can check that.
"""
