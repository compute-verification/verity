"""verity_vllm.registry.quarantine.ov_moe — provisional MoE definitions of lane ov-moe (OLMoE-1B-7B, vLLM fused_moe path).

See `moe.py` (the grouped-expert GEMMs, the routed sum, the router with `NvExpf` = the CUDA `expf` sequence) and
`out/gen/lanes/ov-moe/REPORT.md`.
"""
