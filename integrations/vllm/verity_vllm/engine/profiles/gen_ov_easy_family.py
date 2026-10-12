"""[GEN ov-easy] The serving-family reference a dense profile config is compared against: the registered `Serve_v4{C, LP, STEPS, BN}`
(`b1.ServeV4`), for a tied and an untied output projection alike.

`b1.weights_type(C)` gives an untied checkpoint (`C["tie_word_embeddings"] is False`: Qwen2.5-7B, Mistral-7B, Phi-3-mini) its own
`lm_head` field after `cos_sin`, and `b1.lm_head_of` makes `Serve_v4`'s two `Final` calls read it where a tied one reads `embed`, so
one registered family covers both (`tests/engine/test_gen_ov_easy.py`).  Dense profiles expose `serve_family` as `SERVE_FAMILY`
(`generic.py`), the `family=` a structure comparison (`verity_vllm.check.fold_compare.compare_structure`) may be given.
"""

from __future__ import annotations

from verity.ml.circuits.defs import SpecializedDefinition, bind
from verity_vllm.program.registry import b1


def serve_family(C: dict, *, LP: int = 32, STEPS: int = 15, BN: int | None = None) -> SpecializedDefinition:
    """`Serve_v4{C, LP, STEPS, BN}`, exactly `compare_structure`'s default: `BN` defaults as it does (the config's `BN`, else the FA2 rule
    at `C["D"]`); `C` is passed without `BN`."""
    if BN is None:
        BN = C.get("BN") or b1.fa2_kblock_n(C["D"])
    return bind(b1.ServeV4, C={k: v for k, v in C.items() if k != "BN"}, LP=LP, STEPS=STEPS, BN=BN)
