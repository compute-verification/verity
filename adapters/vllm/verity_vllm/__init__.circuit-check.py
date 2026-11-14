"""circuit-check's sidecar of `verity_vllm` itself, in `circuit_check.targets`' forms (`circuit_check.sidecars`): what
circuit-check needs of the integration, which it never imports.  It registers the package: circuit-check names no
integration and finds the sidecars under a package that has this file.  It gives the reader of a served workload's request
Program.  And it binds every kind of the frontend vocabularies (`verity_vllm.program.frontend.rules.vocab`), so the
Definitions a served Program calls are the word Call families circuit-check holds to the partition invariant; they are bound
here, not beside that module, so that the registries import no frontend module for it."""
from __future__ import annotations

from typing import Any, Callable, Iterator


def request_program(path: str) -> tuple[Any, Any, Callable[[], Iterator[Any]]]:
    """A Build's request Program (its ``instances.json.gz``, or the Build's directory) as circuit-check's row check reads it:
    the Program's view, its population as `word.call_groups` reads one (the required values, the literal Calls and each
    Call's module), and its instance rows, from the first at each call."""
    from pathlib import Path
    from types import SimpleNamespace

    from verity_vllm.correspondence.reader_for_query import Correspondence
    from verity_vllm.query.module_body import literal_calls
    from verity_vllm.query.program_view import from_instances, iter_instance_rows
    from verity_vllm.query.required import population

    p = Path(path)
    inst = str(p / "instances.json.gz" if p.is_dir() else p)
    P = from_instances(inst)
    corr = Correspondence(P)
    Q = SimpleNamespace(required=population(P, corr).required, literals=literal_calls(P), program_path=corr.module_of)
    return P, Q, lambda: iter_instance_rows(inst)


def _served_kind_statics() -> dict[str, dict[str, Any]]:
    """The kinds no small table binds (launch tiles and CTA sizes, collective routes, split counts, the DSA and sparse-MLA shapes; DeepSeek-V4's
    mHC, FlashMLA, compressors, routers and Marlin MoE; the int8 attention row, whose head dim is a multiple of 32), each at a
    binding its Definition module's sidecar already checks, so they add no task."""
    from verity.circuits.boolean.trace import word_of
    from verity.definitions.ops.boolean import gemm as BG
    from verity.definitions.registry import dsv4_compressor as DC, dsv4_qnorm_insert as DQ
    from verity.definitions.registry import flashmla as FM, flashmla_prefill as FP
    from verity.definitions.registry import marlin_moe as MM
    from verity.definitions.registry import sparse_mla as SM
    from verity.definitions.registry import tp_allreduce as TA

    hop = word_of(BG.HopperBF16WgmmaDot16)
    mla = {k: SM.SPARSE_MLA_GLM53[k] for k in ("NH", "DR", "DV", "BN", "DOT", "INV")} | {"SM_SCALE": SM.SPARSE_MLA_GLM53["SCALE"]}
    trees = (TA.chain(8),) * 4 + ((((0, 1), (2, 3)), ((4, 5), (6, 7))),) * 4
    mhc = {"HC": 4, "REPEAT": 20, "EPS": 1e-6, "HC_EPS": 1e-6, "TN": 64}
    router = {"E": 8, "SCALE": 2.5}
    marlin = {"E": 3, "K": 256, "N": 16, "TK": 128, "TN": 8, "GRID": 5, "DOT": MM.DOT}
    int8_row = {"D": 32, "S": 33, "SCALE": 32 ** -0.5}   # Q' K'^T steps 32 channels, so D >= 32; int8_exact's sidecar's row
    dsv4 = {"triton_block_rmsnorm": {"N": 512, "BLOCK": 2048, "WARPS": 8, "EPS": 1e-6},
            "mhc_pre_broadcast": {**mhc, "H": 7168, "NS": 28}, "mhc_residual_broadcast": {"HC": 2, "H": 8},
            "mhc_fused_post_pre": {**mhc, "H": 1024, "NS": 4, "TH": 256}, "mhc_post_pre_tf32": {**mhc, "H": 1024, "NS": 4},
            "mhc_post": {"HC": 4, "N": 7168}, "hc_head": {**mhc, "H": 7168, "TH": 128},
            "dsv4_qnorm_rope_kv_insert": DQ.DSV4_INSERT, "dsv4_compress_c4": DC.DSV4_COMPRESSOR,
            "dsv4_compress_c128": DC.DSV4_COMPRESSOR, "dsv4_inv_rope_quant_head": {"CHUNKS": 4, "QG": 128, "ROPE": 64},
            "flashmla_decode": {**FM.FLASHMLA_DSV4, "NH": 1, "SPLITS": (0, 1), "NW": 64},
            "flashmla_gather_row": FP.FLASHMLA_GATHER_DSV4, "flashmla_prefill": {**FP.FLASHMLA_PREFILL_DSV4, "NH": 1, "NP": 1},
            "silu_mul_clamp": {"I": 4, "LIMIT": 10.0}, "moe_router_hash": router, "moe_router_sqrtsoftplus": router,
            "marlin_moe_rank": {"E": 3, "P": 30, "BM": 4}, "marlin_moe_align": {"P": 30, "BM": 4}, "marlin_moe_gemm": marlin,
            "marlin_moe_gemm_w": marlin}
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
            "int8_attention_row": int8_row, "int8_attention_row_table": int8_row, **dsv4}


def _vocabulary_bindings() -> list[tuple[str, Any]]:
    """Every frontend vocabulary kind (the Definitions the torch frontend binds served ops to) at small statics, or at
    `_served_kind_statics`' when no small table binds it: (label, the bound Definition, or the error)."""
    from verity.definitions.ops.prims import BlackwellE4m3QmmaDot32
    from verity_vllm import TargetProfile
    from verity_vllm.program.frontend.rules import vocab as V
    from verity.definitions.registry import prims as P

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
