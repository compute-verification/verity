"""circuit-check's sidecar of `verity_vllm` itself, in `circuit_check.targets`' forms (`circuit_check.sidecars`): what
circuit-check needs of the integration, which it never imports.  It registers the package: circuit-check names no
integration and finds the sidecars under a package that has this file.  And it binds every kind of the frontend
vocabularies (`verity_vllm.program.frontend.rules.vocab`), so the Definitions a served Program calls are the word Call
families circuit-check holds to the partition's size class; they are bound here, not beside that module, so that the registries import no frontend module for it."""
from __future__ import annotations

from typing import Any


def _served_kind_statics() -> dict[str, dict[str, Any]]:
    """The kinds no small table binds (launch tiles and CTA sizes, collective routes, split counts, the DSA and sparse-MLA shapes; FlashMLA;
    GLM-5.3's fp8_ds_mla cache rows and FlashMLA decode; the int8 attention row,
    whose head dim is a multiple of 32), each at a
    binding its Definition module's sidecar already checks, so they add no task."""
    from verity.circuits.boolean.trace import word_of
    from verity.circuits.ops.boolean import gemm as BG

    from verity_vllm.definitions.registry import flashmla as FM
    from verity_vllm.definitions.registry import flashmla_dsv32 as FD
    from verity_vllm.definitions.registry import flashmla_prefill as FP
    from verity_vllm.definitions.registry import fp8_ds_mla_upconvert as FU
    from verity_vllm.definitions.registry import sparse_mla as SM
    from verity_vllm.definitions.registry import tp_allreduce as TA

    hop = word_of(BG.HopperBF16WgmmaDot16)
    mla = {k: SM.SPARSE_MLA_GLM53[k] for k in ("NH", "DR", "DV", "BN", "DOT", "INV")} | {"SM_SCALE": SM.SPARSE_MLA_GLM53["SCALE"]}
    trees = (TA.chain(8),) * 4 + ((((0, 1), (2, 3)), ((4, 5), (6, 7))),) * 4
    int8_row = {"D": 32, "S": 33, "SCALE": 32 ** -0.5}   # Q' K'^T steps 32 channels, so D >= 32; int8_exact's sidecar's row
    flashmla: dict[str, dict[str, Any]] = {"triton_block_rmsnorm": {"N": 512, "BLOCK": 2048, "WARPS": 8, "EPS": 1e-6},
            "flashmla_decode": {**FM.FLASHMLA_DSV4, "NH": 1, "SPLITS": (0, 1), "NW": 64},
            "flashmla_gather_row": FP.FLASHMLA_GATHER_DSV4, "flashmla_prefill": {**FP.FLASHMLA_PREFILL_DSV4, "NH": 1, "NP": 1}}
    fp8_ds_mla = {"dsa_latent_row_ds_mla": {"KVL": 512, "R": 64, "TILE": 128, "EPS": 1e-5},
                  "flashmla_v32_decode": {**FD.FLASHMLA_GLM53, "NH": 2, "SPLITS": (0, 1, 3)},
                  "fp8_ds_mla_upconvert_row": FU.FP8_DS_MLA_GLM53}
    return {"rmsnorm_cuda": {"N": 6144, "EPS": 1e-5, "BLOCK": 256}, "rmsnorm_fused_cuda": {"N": 6144, "EPS": 1e-5, "BLOCK": 256},
            "all_reduce_mnnvl": {"WORLD": 8, "N": 8, "SHOT": 2},
            "all_reduce_rmsnorm_mnnvl": {"WORLD": 8, "N": 512, "SHOT": 1, "CLUSTER": 2, "EPS": 1e-5, "FMA": False},
            "all_reduce_custom": {"WORLD": 8, "SIZE": 64, "STAGE": 2},
            "all_reduce_multimem": {"WORLD": 8, "SIZE": 128, "TREES": trees, "FTZ": False},
            "all_reduce_nccl_tree": {"WORLD": 4, "SIZE": 8, "CHAIN": (0, 1, 2, 3)},
            "dsa_indexer_k_row": {"D": 128, "ROPE": 64, "EPS": 1e-6}, "dsa_indexer_q": {"H": 32, "D": 128, "ROPE": 64},
            "dsa_latent_row": {"KVL": 512, "R": 64, "EPS": 1e-5},
            "dsa_mqa_logits": {"N": 4, "H": 32, "D": 128},
            "cublas_splitk_gemm": {"K": 64, "N": 3, "SPLIT_K": 4, "DOT": hop},
            "cublas_splitk_gemm_f32out": {"K": 64, "N": 3, "SPLIT_K": 4, "DOT": hop},
            "batched_gemm": {"B": 2, "K": 32, "N": 3, "DOT": hop},
            "ll_bf16_dotprod": {"N": 2, "K": 256, "BS": 64, "VEC": 2, "FMA": 1},
            "ll_bf16_splitk": {"N": 2, "K": 128, "SPLIT_K": 2, "TILE_K": 32, "WARPS": 2, "DOT": hop},
            "sparse_mla": {"T": 65, **mla}, "sparse_mla_split": {"T": 130, "NS": 2, **mla},
            "int8_attention_row": int8_row, "int8_attention_row_table": int8_row, **flashmla, **fp8_ds_mla}


def _vocabulary_bindings() -> list[tuple[str, Any]]:
    """Every frontend vocabulary kind (the Definitions the torch frontend binds served ops to) at small statics, or at
    `_served_kind_statics`' when no small table binds it: (label, the bound Definition, or the error)."""
    from verity.circuits.ops.prims import BlackwellE4m3QmmaDot32

    from verity_vllm import TargetProfile
    from verity_vllm.definitions.registry import prims as P
    from verity_vllm.program.frontend.rules import vocab as V

    small = {"K": 32, "N": 16, "V": 16, "H": 16, "T": 5, "NH": 2, "KVH": 1, "D": 16, "NHEADS": 2, "I": 16, "EPS": 1e-5, "C": 0.5,
             "CAP": 30.0, "BIAS": True, "APPROX": "tanh", "SCALE_OP": "mul", "E": 4, "TOPK": 2, "VPT": 2, "DOT": P.AmpereBF16TcDot16,
             "INV": P.Fa2InvSum, "WORLD": 2, "G": 128, "W": 32, "BITS": 0, "BN": 16, "VS": 16, "START": 16, "MASKED_FROM": 0, "FA2_MASKED_FROM": 0,
             "SCALE": "shared-per-block", "S": 4, "KB": 32, "SWAP": False, "FP8_DOT": BlackwellE4m3QmmaDot32,
             "CONSTRUCTION": "shared-greedy", "TARGET": TargetProfile.default(), "RSF": 2.5, "KERNEL": "block"}
    wide = {**small, "K": 256, "N": 256, "H": 256}           # FP8's 128-element scale groups and 256-thread quantization
    e4m3 = {**small, "G": 32, "DOT": BlackwellE4m3QmmaDot32}  # the MoE pair forms with FP8 scale groups: an e4m3 step, one K tile per scale group
    act = {**small, "G": 8}                                  # the experts' fused activation quant: a scale group dividing I
    served = _served_kind_statics()
    vocabularies = [v for v in vars(V).values() if isinstance(v, V.Vocabulary)]
    kinds = [(v.name, k, make) for v in vocabularies for k, make in v.kinds.items()]
    kinds += [("dense", k, make) for k, make in V._dense_kinds()[0].items()]
    out = []
    for vocabulary, kind, make in kinds:
        label = f"{vocabulary}/{kind}"
        code = make.__code__
        args = code.co_varnames[:code.co_argcount]
        required = args[:len(args) - len(make.__defaults__ or ())]
        optional = tuple(a for a in args[len(required):] if a in small)   # a defaulted static is bound both ways
        for dt in (("bf16", "f32") if "DT" in args else (None,)):
            variants = [(required, "")] + ([(required + optional, "+" + "+".join(optional))] if optional else [])
            for names, suffix in variants:
                fn, err = None, None
                for table in (small, wide, e4m3, act, *([served[kind]] if kind in served else [])):
                    try:
                        fn = make(**{a: (dt if a == "DT" else table[a]) for a in names})
                        _ = getattr(fn, "gates", None)
                        break
                    except (KeyError, ValueError, AssertionError, TypeError) as e:  # a table that doesn't fit: retried with the next, then named
                        fn, err = None, f"{type(e).__name__}: {str(e)[:160]}"
                out.append((label + suffix, fn if fn is not None else err))
    return out


def _vocabulary_roots() -> list[Any]:
    """The kinds that bind.  circuit-check makes this binding's roots from `_vocabulary_bindings`, beside it, and names
    each kind that doesn't bind as a root error."""
    return [fn for _label, fn in _vocabulary_bindings() if not isinstance(fn, str)]
