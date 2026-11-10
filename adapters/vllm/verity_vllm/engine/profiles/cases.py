"""The adapter's fixed cases: case name -> the Hugging Face repo whose `manifests/checkpoints.json` entry the case runs.

`engine_profile.checkpoint_for` resolves a case here first, and a role registered by `add_checkpoint.py` (`role == case`) otherwise.
Imports nothing, so every layer may read it."""

CASES = {
    "B0": {"repo": "HuggingFaceTB/SmolLM2-135M"},
    # the second-model cases: SMOL360 = SmolLM2-360M (B0's architecture, larger), QWEN05 = Qwen2.5-0.5B (B1's architecture at head
    # dim 64); registry configs verity.ml.definitions.registry.b1.{SMOL360,QWEN05}_CONFIG.
    "SMOL360": {"repo": "HuggingFaceTB/SmolLM2-360M"},
    "QWEN05": {"repo": "Qwen/Qwen2.5-0.5B"},
    "B1": {"repo": "Qwen/Qwen2.5-1.5B"},
    # B2 = the same dense baseline as B0, on Ampere (contract: "does the same adapter work on
    # another GPU?"); same checkpoint, same workload, the GPU is recorded in the profile manifest.
    "B2": {"repo": "HuggingFaceTB/SmolLM2-135M"},
    "B5": {"repo": "microsoft/phi-2"},
    # B7 = the contract's 7B dense checkpoint (§1: `Qwen/Qwen2.5-7B` optional subrow), used by W4d for
    # the compiled-vs-eager production A/B on Hopper (§15.5 / §16.7); same B1 workload (same tokenizer).
    "B7": {"repo": "Qwen/Qwen2.5-7B"},
    # M1/M2 = the exploratory MoE lane (W9, wave 3): Qwen1.5-MoE-A2.7B-Chat (Qwen2MoeForCausalLM,
    # 60 routed experts top-4 + shared expert, 24 layers), frozen in manifests/checkpoints.json.
    "M1": {"repo": "Qwen/Qwen1.5-MoE-A2.7B-Chat"},
    "M2": {"repo": "Qwen/Qwen1.5-MoE-A2.7B-Chat"},
    # M0 = the persisted MoE bundle (§16.8, <= 100 MB): one short request (the first 16 prompt
    # tokens of M1's request 0, 8 generated tokens), same checkpoint and profile as M1.
    "M0": {"repo": "Qwen/Qwen1.5-MoE-A2.7B-Chat"},
}
