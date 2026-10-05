"""What circuit-check checks: a Definition id, a subcircuit template (or one of its subcircuits), a served row; and the catalog
the suite runs over, every registered Definition and template at small bindings.

    definition  ``F32Add_v1``, ``RMSNormTriton_v1{N=64,EPS=1e-05}``, ``GemmCoordinate_v2{K=64,DOT=AmpereBF16TcDot16_v2}``: a
                registered Definition bound to the statics its id writes (a function-valued static by its id)
    template    ``rope-head`` (its example parameters), ``rope-head:D=128`` (parameters given), or a subcircuit id
                (``gemm-coordinate/k2048/sm80-mma-bf16``); ``attention-head`` is checked at the ends of its key-count range
    row         a Build's request Program: its directory or its ``instances.json.gz``

The catalog (`catalog`) is roots at small statics: every primitive, every static-free composite, core's GEMM Definitions and
silicon references, every vLLM kernel's self-check specialisations, each template, and tiny whole-model configurations; the
suite checks every specialisation reachable from them (`suite`), and fails by name for a registered Definition family no root
reaches: a new Definition needs a binding here.
"""
from __future__ import annotations

import ast
import importlib
import pkgutil
import re
import zlib
from typing import Any, Callable, Iterable

import numpy as np

from . import cache as CC

_LOADED = False
#: the ids registered once every registry module is imported: the authored Definitions (a lazy family's members, such as
#: `Lifted[F32Add_v1]_v2` or `Const32[0x3f800000]_v1`, are made on demand by id and checked where a body uses them)
REGISTERED: list[str] = []


def _authored(d: Any) -> bool:
    """A Definition the core, the vLLM registry or PoUW's circuit defines (not one a test module registered in the same
    session)."""
    fn = getattr(d, "evaluate", None) or getattr(d, "body", None)
    return getattr(fn, "__module__", "").partition(".")[0] in ("verity", "verity_vllm", "verity_pouw")


def load_registries() -> None:
    """Register every Definition and kernel: core's `verity.ml`, and each module of the vLLM registry and kernel registry."""
    global _LOADED
    if _LOADED:
        return
    from verity.ir.defs import REGISTRY

    import verity.ml.boolean.attention  # noqa: F401
    import verity.ml.boolean.cast  # noqa: F401
    import verity.ml.boolean.elementwise  # noqa: F401
    import verity.ml.boolean.gemm  # noqa: F401
    import verity.ml.boolean.merkle  # noqa: F401
    import verity.ml.boolean.mufu  # noqa: F401
    import verity.ml.boolean.scalar  # noqa: F401
    import verity.ml.boolean.universal  # noqa: F401
    import verity.ml.gemm  # noqa: F401
    import verity.ml.kernels  # noqa: F401
    import verity.ml.prims  # noqa: F401
    import verity_vllm.program.registry as R
    for m in pkgutil.iter_modules(R.__path__):
        if m.name != "quarantine":
            importlib.import_module(f"{R.__name__}.{m.name}")
    from verity_vllm.program.kernels import rows, twins  # noqa: F401  (their kernels register with core)
    import verity_pouw.circuit.boolean  # noqa: F401  (PoUW's own primitives on bits, the six `pc4` and `pc8` rows call)
    import verity_pouw.circuit.ncp2  # noqa: F401  (PoUW's ncp-v2 unit templates and primitives)
    import verity_pouw.circuit.pc4  # noqa: F401  (PoUW's Pearl-C4 tile check)
    import verity_pouw.circuit.pc8  # noqa: F401  (PoUW's Pearl-C tile check)
    import verity_pouw.circuit.rowk  # noqa: F401  (both tile checks' row units cut along k)
    REGISTERED[:] = sorted(i for i, d in REGISTRY.defs.items() if _authored(d))
    _LOADED = True


# --- parsing target strings ---------------------------------------------------------------------------------------------------

def _split(body: str) -> dict[str, str]:
    out, depth, cur, key = {}, 0, "", None
    for ch in body:
        if ch in "{[(":
            depth += 1
        elif ch in "}])":
            depth -= 1
        if ch == "=" and depth == 0 and key is None:
            key, cur = cur.strip(), ""
            continue
        if ch == "," and depth == 0:
            out[key] = cur.strip()
            key, cur = None, ""
            continue
        cur += ch
    if key is not None:
        out[key] = cur.strip()
    return out


def _value(s: str) -> Any:
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return s


def definition(spec: str) -> Any:
    """The registered Definition ``spec`` names, bound to the statics its id writes, or else the suite's target of that id: a
    Definition that only tracing the catalog's roots registers, such as `boolean_gather.gather_bf16(n)`'s gathers."""
    from verity_vllm.query import word as W

    load_registries()
    m = re.fullmatch(r"([^{]+)(?:\{(.*)\})?", spec.strip())
    if m is None:
        raise ValueError(f"not a Definition id: {spec!r}")
    fam, body = m[1], m[2]
    statics = {k: _value(v) for k, v in _split(body).items()} if body else {}
    try:
        fn = W.specialization(fam, statics)
    except KeyError:
        fn = suite().get(spec.strip())
        if fn is None:
            raise
        return fn
    return fn.bind() if hasattr(fn, "bind") and not statics and getattr(fn, "statics", None) == () else fn


def subcircuit(spec: str) -> tuple[Any, list[dict]]:
    """The subcircuit ``spec`` names and the instance-statics bindings it is checked at."""
    from verity_numerical.bench import templates as TM

    load_registries()
    tid, _, rest = spec.partition(":")
    if "/" in tid:
        sub = TM.parse_subcircuit_id(tid)
    else:
        t = TM.get(tid)
        params = dict(t.example)
        params.update({k: _value(v) for k, v in _split(rest).items()} if rest else {})
        sub = t.bind(**params)
        return sub, instance_statics(sub, params)
    return sub, instance_statics(sub)


def instance_statics(sub: Any, params: dict | None = None) -> list[dict]:
    """``attention-head``: the key counts at the ends of its range (``T = lo:hi`` of ``params``, else the template's example);
    every other template: none."""
    if sub.template.id != "attention-head":
        return [{}]
    lo, _, hi = str((params or {}).get("T") or sub.template.example["T"]).partition(":")
    return [{"T": t} for t in sorted({int(lo), int(hi or lo)})]


def kind_of(spec: str) -> str:
    from pathlib import Path

    from verity_numerical.bench import templates as TM

    if Path(spec).exists():
        return "row"
    head = spec.partition(":")[0].partition("/")[0]
    return "template" if head in TM.TEMPLATES else "definition"


# --- the catalog ----------------------------------------------------------------------------------------------------------------

TINY = {"model": "tiny", "revision": None, "H": 32, "NL": 2, "NH": 2, "KVH": 1, "D": 16, "I": 32, "V": 64, "QKV": 64, "GU": 64,
        "EPS": 1e-5, "ROPE_THETA": 1e4, "MAX_POS": 8, "BIAS": True, "tie_word_embeddings": True, "profile": {}}
TINY_NOBIAS = {**TINY, "BIAS": False, "KVH": 2, "QKV": 96}
#: FP8's per-token quantization reads whole 256-thread blocks: every reduction length is at least 256
TINY_FP8 = {**TINY, "H": 256, "NL": 1, "NH": 2, "KVH": 1, "D": 128, "I": 256, "V": 16, "QKV": 512, "GU": 512}

#: the parameters each template is checked at: the workload ones where C-Flock pins its unit circuit, else small (a template's
#: example is its served size)
TEMPLATE_PARAMS = {
    "gemm-coordinate": [{"K": 64, "semantics": "sm80.mma.m16n8k16.bf16"}, {"K": 64, "semantics": "sm90.wgmma.m64n8k16.bf16"},
                        {"K": 64, "semantics": "sm89.mma.m16n8k32.e4m3"}, {"K": 64, "semantics": "sm90.wgmma.m64n8k32.e4m3"},
                        {"K": 128, "semantics": "sm120.mma.m16n8k64.e2m1.nvf4"}],
    "attention-head": [{"D": 64, "BN": 128, "T": "1:17"}],
    "rope-head": [{"D": 64}],
    "rmsnorm-fused-cuda": [{"N": 2048, "EPS": 1e-05}],
    "rmsnorm-triton": [{"N": 128, "EPS": 1e-06}],
    "silu-mul": [{"I": 8192}],
    "gumbel-top-p-token-select": [{"V": 64}],
    "token-select-greedy": [{"V": 64}],
    "embedding-row": [{"V": 64, "H": 16}],
    "moe-expert-coordinate": [{"E": 4, "K": 64}],
    "moe-expert-coordinate-weighted": [{"E": 4, "K": 64}],
    "scaled-mm-fp8-block-coordinate": [{"K": 256, "G": 128}],
}


#: the binding units (`cache`'s units of this module: a root function, a `_model_roots` entry, a `TEMPLATE_PARAMS` entry) whose
#: roots reach each suite target, by Definition id (`suite`) or template label (`templates`)
BINDINGS: dict[str, set[str]] = {}


def _produce(units: tuple[str, ...], make: Callable[[], list[Any]], by_unit: list | None = None) -> list[Any]:
    """``make()``'s roots, recorded while it runs, with every Definition it makes (`cache.producing`)."""
    with CC.producing(units):
        got = make()
    if by_unit is not None:
        by_unit.append((units, got))
    return got


def templates() -> list[tuple[str, Any, list[dict]]]:
    """(label, subcircuit, instance statics) for every template in the catalogue: `TEMPLATE_PARAMS`, else its example."""
    from verity_numerical.bench import templates as TM

    load_registries()
    out = []
    for tid, t in sorted(TM.TEMPLATES.items()):
        unit = (f"TEMPLATE_PARAMS:{tid}",)
        for params in TEMPLATE_PARAMS.get(tid, [dict(t.example)]):
            sub, sts = _produce(unit, lambda t=t, params=params: (lambda s: [s, instance_statics(s, params)])(t.bind(**params)))
            BINDINGS.setdefault(sub.id, set()).update(unit)
            out.append((sub.id, sub, sts))
    return out


def _pouw_roots() -> list[Any]:
    """PoUW's `ncp-v2` unit templates at k = 64, and the smallest whole call, its weight side and a window of two calls on
    it (`verity_pouw.circuit.ncp2`), for both P's (`ncp-v2`'s word P and `ncp-v2-shift24`); the package's own tests check
    their template partitions and the Z separator. Beside them, the model Program's rows at a PoUW site, without and with the
    bias (`verity_vllm.program.registry.pouw_rows`), and Pearl-C4's tile check (`verity_pouw.circuit.pc4`): its unit
    templates at k = 64 with a one-word tile (small enough to lower whole), and its weight side, call side and whole check at
    one row a side; and Pearl-C's (`verity_pouw.circuit.pc8`) the same way at k = 128 (one promotion window), its templates
    on the H100's and sm_120's records and its roots on sm_120's.  Both checks' hiding tiles and roots (``hidden``: the tile
    digest, `verity_pouw.circuit.leaves`; their rows are C-Flock's ports) at the same shapes.  The row units cut along k
    (`verity_pouw.circuit.rowk`, ``Shape.row_k``) in two segments: Pearl-C4's pieces at k = 128 and its roots, Pearl-C's
    pieces at k = 256 on the H100's and sm_120's records and at k = 64 unpromoted, and its roots on sm_120's, plain and
    hiding."""
    from dataclasses import replace

    from verity.ir.defs import bind
    from verity_pouw.circuit import ncp2, pc4, pc8, rowk
    from verity_vllm.program.registry import pouw_rows

    shape = ncp2.Shape(16, 64, 16)
    out = []
    for kind in ("shift24", "word"):
        out += [t for t in ncp2.templates(64, kind).values() if t not in out]
        out += [ncp2.linear(shape, kind), ncp2.weight(shape, kind), ncp2.window(shape, 2, kind)]
    tile = pc4.Shape(1, 64, 1, tile=(1, 1))
    out += [*pc4.templates(64, tile.tile).values(), pc4.weight(tile), pc4.call(tile), pc4.check(tile)]
    tile8 = pc8.Shape(1, 128, 1, tile=(1, 1), dev="sm120")
    out += [*pc8.templates(128, tile8.tile, "h100").values(), *pc8.templates(128, tile8.tile, "sm120").values(),
            pc8.weight(tile8), pc8.call(tile8), pc8.check(tile8)]
    h4, h8 = replace(tile, hidden=True), replace(tile8, hidden=True)
    out += [t for t in [*pc4.templates(64, h4.tile, True).values(), pc4.weight(h4), pc4.call(h4), pc4.check(h4),
                        *pc8.templates(128, h8.tile, "sm120", True).values(), pc8.weight(h8), pc8.call(h8), pc8.check(h8)]
            if t not in out]
    cut = [p for side in ("a", "b") for pieces in (rowk.pieces4(128, 64, side), rowk.pieces8(256, 128, "h100", side),
                                                    rowk.pieces8(256, 128, "sm120", side),
                                                    rowk.pieces8(64, 32, "sm120-unpromoted", side))
           for p in (pieces[0], pieces[1], *pieces[2], pieces[3])]
    for mod, s in ((pc4, pc4.Shape(1, 128, 1, tile=(1, 1), row_k=64)),
                   (pc8, pc8.Shape(1, 256, 1, tile=(1, 1), dev="sm120", row_k=128))):
        cut += [build(replace(s, hidden=h)) for h in (False, True) for build in (mod.weight, mod.call, mod.check)]
    seen = {t.id for t in out}
    out += [t for t in cut if t.id not in seen and not seen.add(t.id)]
    return out + [bind(pouw_rows.NcpLinearRow, K=64, N=16), bind(pouw_rows.NcpLinearRowBias, K=64, N=16),
                  bind(pouw_rows.NcpLinearCoordinate, K=64)]


def _core_roots() -> list[Any]:
    from verity.ir.defs import bind
    from verity.ml import gemm as G
    from verity.ml import kernels as K
    from verity.ml import prims as P

    A, H = P.AmpereBF16TcDot16, P.HopperBF16WgmmaDot16
    return [bind(G.DotBf16, K=64, DOT=A), bind(G.DotBf16, K=48, DOT=H), bind(G.GemmCoordinate, K=64, DOT=A),
            bind(G.GemmCoordinate, K=64, DOT=H), bind(G.Gemm, K=32, N=3, DOT=A), P.BlackwellE4m3QmmaDot32,
            P.BlackwellE5m2QmmaDot32, P.BlackwellNvf4OmmaDot64, P.BlackwellMxf4OmmaDot64, bind(G.GemmCoordinateE4m3, K=64, **G.GEMM_SM120_E4M3),
            bind(G.GemmCoordinateNvf4, K=128, **G.GEMM_SM120_NVF4), bind(G.GemmCoordinateMxf4, K=128, **G.GEMM_SM120_MXF4), P.F32ToE2m1Sat, P.E4m3ToF32, K.HOPPER_E4M3_STEP, K.HOPPER_E4M3_DOT.bind(K=128),
            K.ADA_E4M3_STEP, K.ADA_E4M3_DOT.bind(K=128), K.SM120_NVF4_STEP, K.SM120_NVF4_DOT.bind(K=256), K.SM120_MXF4_STEP,
            K.SM120_MXF4_DOT.bind(K=256)]


def _boolean_roots() -> list[Any]:
    """The Boolean BF16 coordinate and dot (`verity.ml.boolean.gemm`) on both steps at K = 64, the GEMM at K = 32, N = 3, the E4M3
    coordinate on both FP8 steps at K = 64, and the NVFP4 and MXFP4 coordinates at K = 128; `silu_and_mul`'s rows at I = 8
    (`verity_vllm.program.registry.boolean_silu`: `SiluMul_v3` and `SiluMul_v4`, and v4's word view, the served
    `SiluMul_v2` of `registry.act`), the final-logit softcap's tanh row at N = 8 (`boolean_activation`),
    `registry.dense`'s rows on bits at 8 elements with Gemma-2's constants (`boolean_dense`: `gelu_tanh_and_mul` at I = 8, the
    RMSNorm chain's and the logit softcap's element-wise rows at N = 8), and OLMoE's routed sum at TOPK = 8 over 8 columns
    and its router at 8 experts, 2 slots, 4 values per thread (`boolean_moe`), and GLM-5.3's sigmoid `noaux_tc` router at 8
    experts, 2 slots, on both of its kernels, on bits and its word view (`registry.moe.MoeRouterSigmoidBias_v1`); the steps, the cast, attention's softmax operations (`verity.ml.boolean.attention`), the element-wise
    operations (`verity.ml.boolean.elementwise`), the MUFU operations (`verity.ml.boolean.mufu`), the comparator, selects, max /
    min and E4M3 cast (`verity.ml.boolean.scalar`), the argmax step, `SiluMulBf16_v3` and the rows' static-free elements are
    static-free. vLLM's Boolean RoPE (`verity_vllm.program.registry.boolean_rope`) at the kernel rows' `RoPE_v1` binding, which
    reaches its head and halves; greedy token selection (`sampling_boolean.TokenSelect_v2`) at one, two and three levels of its
    tree, which reach every stage, leaf and merge shape (a full group, a remainder, the first group, the root group). The served
    norms on bits (`boolean_norms`) at N = 16 with their MUFU statics at the word ids and at mufu's Boolean versions (the switch's
    binding, `verity_vllm.program.boolean`), the fused norm's lanes at two elements per thread, and the word Definitions of its
    constant steps; the rest of Gemma-2's norm chain on bits (`boolean_dense_norm`, over `boolean_dense`'s rows): the mean at
    N = 16 and at N = 1030 (two chunks, one element per thread), the narrowing at N = 16, and the rsqrt with its MUFU at the word
    id and at mufu's `RsqrtApprox_v2`. The Gumbel select (`gumbel_boolean.GumbelSelectF32_v2`) at two lanes (its family's smallest, and the noisy row's:
    about 600,000 gates each, every lane about 300,000) and at a full group plus a remainder; the stream key, setup, lane and
    their pieces are static-free; the f32 argmax tree's stages at two and three levels reach its other shapes. FA2 with logit
    soft-capping on bits (`boolean_softcap`) at one head of 17 keys, two blocks (a full one and a partial one) with Check_inf in
    both, its ``TANH`` and ``EX2`` at mufu's Boolean versions, which reaches its head and blocks. vLLM's GPT-J RoPE on bits at
    two heads of 8. MLA (`registry.mla`): Triton's one-split decode at one head of two keys over a 16 + 16 latent (a partial
    block, so its masked columns), which reaches the head, the block, the score and every element; the absorb bmm at two heads,
    K = 32, N = 3; the shared-expert combine at 8 elements with GLM-4.7-Flash's routed_scaling_factor."""
    from verity.ir.defs import bind
    from verity.ml.boolean import attention as BAT
    from verity.ml.boolean import gemm as BG
    from verity.ml.boolean import mufu as BM
    from verity_vllm.program.registry import act
    from verity_vllm.program.registry import boolean_activation as BA
    from verity_vllm.program.registry import boolean_dense as BDN
    from verity_vllm.program.registry import boolean_dense_norm as DN
    from verity_vllm.program.registry import boolean_moe as BMOE
    from verity_vllm.program.registry import boolean_norms as BN
    from verity_vllm.program.registry import boolean_rope
    from verity_vllm.program.registry import boolean_silu as BSL
    from verity_vllm.program.registry import boolean_softcap as BSC
    from verity_vllm.program.registry import gumbel_boolean as GB
    from verity_vllm.program.registry import mla
    from verity_vllm.program.registry import moe as M
    from verity_vllm.program.registry import prims as P
    from verity_vllm.program.registry import sampling_boolean as SB

    HOP = BG.HopperBF16WgmmaDot16
    rows = [bind(d, N=8) for d in (BDN.SquareBf16, BDN.SquareF32, BDN.AddWidenedBf16, BDN.ScaleRowBf16, BDN.ScaleRowF32,
                                   BDN.MulVecF32, BDN.Bf16MulScalarTensor)]
    rows += [bind(BDN.AddScalarF32, N=8, C=1e-6), bind(BDN.AddScalarBf16, N=8, C=1.0), bind(BDN.Bf16DivScalar, N=8, C=30.0),
             bind(BDN.Bf16MulScalar, N=8, C=30.0), bind(BMOE.MoeSum, TOPK=8, H=8),
             bind(BMOE.MoeRouterTopKOrdered, E=8, TOPK=2, VPT=4)]
    rows += [bind(d, E=8, TOPK=2, SCALE=2.5, KERNEL=k) for d in (BMOE.MoeRouterSigmoidBias, M.MoeRouterSigmoidBias)
             for k in M.ROUTER_KERNELS]
    return [bind(BG.GemmCoordinate, K=64, **BG.GEMM_AMPERE), bind(BG.GemmCoordinate, K=64, **BG.GEMM_HOPPER),
            bind(BG.DotBf16, K=64, **BG.GEMM_AMPERE), bind(BG.DotBf16, K=64, **BG.GEMM_HOPPER),
            bind(BG.Gemm, K=32, N=3, **BG.GEMM_AMPERE), bind(BG.Gemm, K=32, N=3, **BG.GEMM_HOPPER),
            bind(BG.GemmCoordinateE4m3, K=64, **BG.GEMM_HOPPER_E4M3), bind(BG.GemmCoordinateE4m3, K=64, **BG.GEMM_SM120_E4M3),
            bind(BG.GemmCoordinateNvf4, K=128, **BG.GEMM_SM120_NVF4), bind(BG.GemmCoordinateMxf4, K=128, **BG.GEMM_SM120_MXF4),
            bind(BSL.SiluMul, I=8), bind(BSL.SiluMulExpf, I=8), bind(act.SiluMul, I=8),
            bind(boolean_rope.RoPE, NHEADS=2, D=8), bind(SB.TokenSelect, V=50), bind(SB.TokenSelect, V=200),
            bind(SB.TokenSelect, V=4161),
            bind(BN.RMSNormFusedCuda, N=16, EPS=1e-5, **BN.CUDA_WORD), bind(BN.RMSNormTriton, N=16, EPS=1e-5, **BN.TRITON_WORD),
            bind(BN.RMSNormLanesCuda, N=40, BLOCK=32), bind(BN.RMSNormVarEpsCudaWord, N=16, EPS=1e-5),
            bind(BN.RMSNormVarEpsTritonWord, N=16, EPS=1e-5),
            bind(BN.RMSNormFusedCuda, N=16, EPS=1e-5, RSQRT=bind(BM.RsqrtApprox)),
            bind(BN.RMSNormTriton, N=16, EPS=1e-5, SQRT=bind(BM.MufuSqrtFtz), SCALEA=bind(BM.DivFullScaleA), RCP=bind(BM.DivFullRcp)),
            bind(DN.MeanTriton, N=16), bind(DN.MeanTriton, N=1030), bind(DN.NarrowF32ToBf16, N=16),
            bind(DN.RsqrtF32, N=1, RSQRT=P.RsqrtApprox), bind(DN.RsqrtF32, N=1, RSQRT=bind(BM.RsqrtApprox)),
            bind(BA.Bf16Tanh, N=8), bind(BDN.GeluTanhMul, I=8),
            bind(BSC.AttentionSoftcap, T=17, NH=1, KVH=1, D=16, BN=16, CAP=50.0, DOT=BG.HopperBF16WgmmaDot16, INV=BAT.Fa2InvSum,
                 MASKED_FROM=0, TANH=BM.MufuTanh, EX2=BM.MufuEx2Ftz),
            bind(GB.GumbelSelectF32, V=2), bind(GB.GumbelSelectF32, V=65),
            *SB.stages(200, SB.F32ArgmaxStage), *SB.stages(4161, SB.F32ArgmaxStage),
            bind(boolean_rope.RoPEGptJ, NHEADS=2, D=8),
            bind(mla.TritonMlaDecode, H=1, T=2, DL=16, DR=16, BN=32, SPLITS=1, SCALE=0x3D800000, DOT=HOP),
            bind(mla.MlaBmm, NHEADS=2, K=32, N=3, DOT=HOP), bind(mla.MoeSharedCombine, H=8, RSF=0x3FE66666)] + rows


def _boolean_gather_roots() -> list[Any]:
    """The Boolean embedding (`verity_vllm.program.registry.boolean_gather`) over gathers traced whole (V = 64), and a gather
    that is a tree of Calls with a partial last chunk (V = 1100); the bf16 widening (`verity.ml.boolean.cast`) is static-free."""
    from verity.ir.defs import bind
    from verity_vllm.program.registry import boolean_gather as BGA

    return [bind(BGA.Embedding, V=64, H=16), bind(BGA.gather_bf16(1100))]


def _universal_roots() -> list[Any]:
    """Protocol 2's universal unit (`verity.ml.boolean.universal`) at two size classes: G = 16 gates over 16 inputs (W = 32 wires)
    with 8 outputs, and G = 64 over 64 (W = 128) with 32."""
    from verity.ir.defs import bind
    from verity.ml.boolean.universal import UniversalUnit

    return [bind(UniversalUnit, G=16, N_IN=16, N_OUT=8), bind(UniversalUnit, G=64, N_IN=64, N_OUT=32)]


def _merkle_roots() -> list[Any]:
    """The hidden-index Merkle read (`verity.ml.boolean.merkle`), binary and 4-ary, at one level (a level is two or three
    SHA-512 compressions: deeper reads repeat it). About two minutes each; the 4-ary one checks the slot multiplexers among
    three siblings that the binary swap does not have."""
    from verity.ir.defs import bind
    from verity.ml.boolean.merkle import MerkleRead

    return [bind(MerkleRead, ARITY=2, D=1), bind(MerkleRead, ARITY=4, D=1)]


def _kernel_roots() -> list[Any]:
    from verity_vllm.program.kernels import kernel_registry as KR

    out = []
    for r in KR.registered():
        out += list(r.check_targets(np.random.default_rng(zlib.crc32(f"{r.key}/{r.kernel}".encode()))))
    return out


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """Tiny whole-model configurations (the vLLM tests' `TINY`) and the leaf families they do not reach."""
    from verity.ir.defs import bind
    from verity.ml.boolean import gemm as BG
    from verity.ml.prims import BlackwellE4m3QmmaDot32, BlackwellNvf4OmmaDot64
    from verity_vllm.program.registry import b1, b1_tp2, fa2_check_inf, fp8, fp8_moe, hopper, lifted, moe_pad, nvfp4, ref_prims, sampling
    from verity_vllm.program.registry import act, serve3, spec
    from verity_vllm.program.registry import value_inputs as VI
    from verity_vllm.program.registry import boolean_attention, boolean_dsa, boolean_fp8, boolean_fp8_moe, boolean_rope
    from verity_vllm.program.registry import boolean_sparse_mla, dsa, sparse_mla, tables

    long = {**TINY, "MAX_POS": 16}
    tp2 = {**TINY_NOBIAS, "BIAS": True, "TP": 2}
    lifted_profile = {"GEMM": b1.GEMM_AMPERE, "ATTENTION": b1.ATTENTION_FA2, "NORM": b1.norm_bundle(TINY)}
    return [
        ("Serve_v1", lambda: bind(b1.Serve, C=TINY, LP=3, STEPS=2)),
        ("Serve_v2", lambda: b1.serve_v2(TINY, 3, 2)),
        ("Serve_v3", lambda: serve3.serve_v3({**TINY, "EOS_ID": 7}, 3, 2)),
        ("Serve_v4", lambda: bind(b1.ServeV4, C=TINY, LP=3, STEPS=2, BN=16)),
        ("ServeHopper_v1", lambda: bind(hopper.ServeHopper, C=TINY, LP=3, STEPS=2)),
        ("ServeFp8_v1", lambda: bind(fp8.ServeFp8, C=TINY_FP8, LP=2, STEPS=1)),
        ("ServeTP2_v1", lambda: b1_tp2.serve_tp2(3, 2, tp2)),
        ("Prefill_v1", lambda: bind(lifted.Prefill, C=TINY, LP=3, **lifted_profile)),
        ("StepBody_v1", lambda: bind(lifted.StepBody, C=TINY, T=4, **lifted_profile)),
        ("AttentionN_v1", lambda: bind(spec.AttentionN, T=5, NH=2, KVH=1, D=16, BN=16)),
        # about 7 minutes a target: the smallest Boolean chain whose second block rescales (T = BN + 1)
        ("Attention_v6", lambda: bind(boolean_attention.AttentionV6, T=17, NH=1, KVH=1, D=16, BN=16)),
        ("Attention_v7", lambda: bind(boolean_attention.AttentionV7, T=5, NH=1, KVH=1, D=16, BN=16, **boolean_attention.ATTENTION_FA3)),
        # the served FA2 binding at the smallest chain whose unguarded block rescales (T = BN + 1, block 0 below MASKED_FROM)
        ("Attention_v8", lambda: bind(boolean_attention.AttentionV8, T=17, NH=1, KVH=1, D=16, BN=16, MASKED_FROM=1,
                                      DOT=boolean_attention.ATTENTION_FA3["DOT"], INV=boolean_attention.ATTENTION_FA2["INV"])),
        ("Attention_v9", lambda: bind(boolean_attention.AttentionV9, T=17, NH=1, KVH=1, D=16, BN=16, MASKED_FROM=1,
                                      **boolean_attention.ATTENTION_FA3)),
        ("RMSNormFusedCuda_v1", lambda: bind(b1.RMSNormFusedCudaV1, N=64, EPS=1e-5)),
        # the Definitions whose program values are inputs (eps, the attention scale); the chain's second block rescales (T = BN + 1)
        ("RMSNormTriton_v3", lambda: bind(VI.RMSNormTritonV3, N=64)),
        ("RMSNormFusedCuda_v4", lambda: bind(VI.RMSNormFusedCudaV4, N=64)),
        ("Attention_v10", lambda: bind(VI.AttentionV10, T=17, NH=1, KVH=1, D=16, BN=16)),
        ("Attention_v11", lambda: bind(fa2_check_inf.AttentionV11, T=17, NH=1, KVH=1, D=16, BN=16, MASKED_FROM=1,
                                       DOT=hopper.HopperBF16WgmmaDot16, INV=b1.P.Fa2InvSum)),
        ("Attention_v12", lambda: bind(VI.AttentionV12, T=17, NH=1, KVH=1, D=16, BN=16, DOT=hopper.HopperBF16WgmmaDot16, INV=b1.P.Fa2InvSum)),
        ("Attention_v13", lambda: bind(VI.AttentionV13, T=17, NH=1, KVH=1, D=16, BN=16, MASKED_FROM=1, DOT=hopper.HopperBF16WgmmaDot16,
                                       INV=hopper.Fa3InvSum)),
        ("AttentionSoftcap_v4", lambda: bind(VI.AttentionSoftcapV4, T=17, NH=1, KVH=1, D=16, BN=16)),
        ("AttentionSoftcap_v5", lambda: bind(VI.AttentionSoftcapV5, T=17, NH=1, KVH=1, D=16, BN=16, MASKED_FROM=1,
                                             DOT=hopper.HopperBF16WgmmaDot16, INV=b1.P.Fa2InvSum)),
        # the Definitions that read a registered table (`gelu_tanh_bf16`, `tanh_rn`) through `TableRead16x1_v1`, `TableRead27x2_v1`
        ("GeluTanhMul_v3", lambda: bind(tables.GeluTanhMulV3, I=16)),
        ("TanhF32Rn_v2", lambda: bind(tables.TanhF32RnV2)),
        ("LMoeFfnL_v1", lambda: bind(moe_pad.LMoeFfnL, E=4, TOPK=2, VPT=2, H=16, I=16)),
        ("AllReduce_v2", lambda: b1_tp2.all_reduce(4, 8)),
        ("AllGather_v1", lambda: b1_tp2.all_gather(4, 8)),
        ("QuantLinearFp8_v1", lambda: bind(fp8.QuantLinearFp8, K=256, N=2, BIAS=True)),
        ("ScaledMmFp8NoSwap_v1", lambda: bind(fp8.ScaledMmFp8NoSwap, K=64, N=2, BIAS=True, DOT=BlackwellE4m3QmmaDot32)),
        ("QuantLinearNvfp4_v1", lambda: bind(nvfp4.QuantLinearNvfp4, K=64, N=2, DOT=BlackwellNvf4OmmaDot64)),
        ("ScaledMmFp8BlockSharedScale_v1", lambda: bind(fp8.ScaledMmFp8BlockSharedScale, K=256, N=256, G=128)),
        ("QuantLinearFp8_v2", lambda: bind(boolean_fp8.QuantLinearFp8V2, K=256, N=2, BIAS=True)),
        ("ScaledMmFp8_v2", lambda: bind(boolean_fp8.ScaledMmFp8V2, K=32, N=2, BIAS=False)),
        ("Fp8GroupQuant_v2", lambda: bind(boolean_fp8.Fp8GroupQuantV2, K=64, G=32)),
        ("ScaledMmFp8Block_v2", lambda: bind(boolean_fp8.ScaledMmFp8BlockV2, K=32, N=32, G=32)),
        ("ScaledMmFp8BlockSharedScale_v2", lambda: bind(boolean_fp8.ScaledMmFp8BlockSharedScaleV2, K=64, N=32, G=32)),
        ("ScaledMmFp8NoSwap_v2", lambda: bind(boolean_fp8.ScaledMmFp8NoSwapV2, K=64, N=2, BIAS=True,
                                              DOT=BG.BlackwellE4m3QmmaDot32)),
        # a partial last weight block (GLM-5.3's fused_qkv_a_proj, N = 2624 over 21 scale rows)
        ("ScaledMmFp8Block_v1/ragged", lambda: bind(fp8.ScaledMmFp8Block, K=256, N=192, G=128)),
        ("ScaledMmFp8BlockSharedScale_v1/ragged", lambda: bind(fp8.ScaledMmFp8BlockSharedScale, K=256, N=192, G=128)),
        # the blockwise linear over a GEMM target's step (cc 12.0's), at the v1 and v2 forms' shapes
        ("ScaledMmFp8Block_v3", lambda: bind(fp8.ScaledMmFp8BlockV3, K=256, N=192, G=128, DOT=BlackwellE4m3QmmaDot32)),
        ("ScaledMmFp8BlockSharedScale_v3", lambda: bind(fp8.ScaledMmFp8BlockSharedScaleV3, K=256, N=192, G=128,
                                                        DOT=BlackwellE4m3QmmaDot32)),
        ("ScaledMmFp8Block_v4", lambda: bind(boolean_fp8.ScaledMmFp8BlockV4, K=32, N=32, G=32, DOT=BG.BlackwellE4m3QmmaDot32)),
        ("ScaledMmFp8BlockSharedScale_v4", lambda: bind(boolean_fp8.ScaledMmFp8BlockSharedScaleV4, K=64, N=32, G=32,
                                                        DOT=BG.BlackwellE4m3QmmaDot32)),
        # the activation quant at fused_moe's group (fp8.QUANT_GROUPS)
        ("Fp8GroupQuant_v1/g64", lambda: bind(fp8.Fp8GroupQuant, K=128, G=64)),
        # fused_moe_kernel's block FP8 expert GEMMs: GLM-5.3's served coordinates (w13 K = 6144, w2 K = 256 at TP8), and small ones
        # that reach two tiles per scale block, two dot steps per tile, SWAP_AB and a partial last weight block
        ("MoeExpertFp8BlockCoordinate_v1", lambda: bind(fp8_moe.MoeExpertFp8BlockCoordinate, K=6144, G=128, KB=32, SWAP=False)),
        ("MoeExpertFp8BlockCoordinateW_v1", lambda: bind(fp8_moe.MoeExpertFp8BlockCoordinateW, K=256, G=128, KB=32, SWAP=False)),
        ("MoeExpertFp8BlockGemm_v1", lambda: bind(fp8_moe.MoeExpertFp8BlockGemm, K=128, N=80, G=64, KB=32, SWAP=False)),
        ("MoeExpertFp8BlockGemmW_v1", lambda: bind(fp8_moe.MoeExpertFp8BlockGemmW, K=64, N=48, G=32, KB=32, SWAP=True)),
        ("MoeExpertFp8BlockCoordinate_v2", lambda: bind(boolean_fp8_moe.MoeExpertFp8BlockCoordinateV2, K=64, G=32, KB=32, SWAP=False)),
        ("MoeExpertFp8BlockCoordinate_v2/swap", lambda: bind(boolean_fp8_moe.MoeExpertFp8BlockCoordinateV2, K=64, G=64, KB=32,
                                                             SWAP=True)),
        ("MoeExpertFp8BlockCoordinate_v2/kb64", lambda: bind(boolean_fp8_moe.MoeExpertFp8BlockCoordinateV2, K=64, G=64, KB=64,
                                                             SWAP=False)),
        ("MoeExpertFp8BlockCoordinateW_v2", lambda: bind(boolean_fp8_moe.MoeExpertFp8BlockCoordinateWV2, K=64, G=32, KB=32, SWAP=True)),
        ("MoeExpertFp8BlockGemm_v2", lambda: bind(boolean_fp8_moe.MoeExpertFp8BlockGemmV2, K=32, N=40, G=32, KB=32, SWAP=False)),
        ("MoeExpertFp8BlockGemmW_v2", lambda: bind(boolean_fp8_moe.MoeExpertFp8BlockGemmWV2, K=32, N=40, G=32, KB=32, SWAP=True)),
        # silu_and_mul_per_block_quant between the expert launches: one element, GLM-5.3's served group (G = 128), and rows of
        # two groups; on bits at G = 4 (each element about 19.6k ANDs)
        ("SiluMulFp8BlockQuantResult_v1", lambda: bind(fp8_moe.SiluMulFp8BlockQuantResult)),
        ("SiluMulFp8BlockQuantGroup_v1", lambda: bind(fp8_moe.SiluMulFp8BlockQuantGroup, G=128)),
        ("SiluMulFp8BlockQuant_v1", lambda: bind(fp8_moe.SiluMulFp8BlockQuant, I=16, G=8)),
        ("SiluMulFp8BlockQuantResult_v2", lambda: bind(boolean_fp8_moe.SiluMulFp8BlockQuantResultV2)),
        ("SiluMulFp8BlockQuantGroup_v2", lambda: bind(boolean_fp8_moe.SiluMulFp8BlockQuantGroupV2, G=4)),
        ("SiluMulFp8BlockQuant_v2", lambda: bind(boolean_fp8_moe.SiluMulFp8BlockQuantV2, I=8, G=4)),
        # GLM-5.3's DSA indexer at its served shapes (D = 128, 32 heads, the top-2048 row), and on bits at one warp
        ("DsaUe8m0Quant_v1", lambda: bind(dsa.DsaUe8m0Quant, N=128)),
        ("DsaIndexerKRow_v1", lambda: bind(dsa.DsaIndexerKRow, D=128, ROPE=64, EPS=1e-6)),
        ("DsaIndexerQHead_v1", lambda: bind(dsa.DsaIndexerQHead, D=128, ROPE=64, NH=32)),
        ("DsaIndexerQ_v1", lambda: bind(dsa.DsaIndexerQ, H=32, D=128, ROPE=64)),
        ("DsaMqaLogit_v1", lambda: bind(dsa.DsaMqaLogit, H=32, D=128)),
        ("DsaMqaLogits_v1", lambda: bind(dsa.DsaMqaLogits, N=4, H=32, D=128)),
        ("DsaTopKShortRow_v1", lambda: bind(dsa.DsaTopKShortRow, TOPK=2048)),
        ("DsaUe8m0Quant_v2", lambda: bind(boolean_dsa.DsaUe8m0QuantV2, N=4)),
        ("DsaIndexerKRow_v2", lambda: bind(boolean_dsa.DsaIndexerKRowV2, D=32, ROPE=16, EPS=1e-6)),
        ("DsaIndexerQHead_v2", lambda: bind(boolean_dsa.DsaIndexerQHeadV2, D=32, ROPE=16, NH=32)),
        ("DsaIndexerQ_v2", lambda: bind(boolean_dsa.DsaIndexerQV2, H=2, D=32, ROPE=16)),
        ("DsaMqaLogit_v2", lambda: bind(boolean_dsa.DsaMqaLogitV2, H=8, D=32)),
        ("DsaMqaLogits_v2", lambda: bind(boolean_dsa.DsaMqaLogitsV2, N=2, H=8, D=32)),
        ("DsaTopKShortRow_v2", lambda: bind(boolean_dsa.DsaTopKShortRowV2, TOPK=16)),
        # its sparse MLA (FA3 over the selected rows) at the served statics, two blocks; on bits at DR = DV = BN = 16
        ("SparseMlaBlock_v1", lambda: bind(sparse_mla.SparseMlaBlock, NVIS=64, FIRST=False,
                                           **{k: sparse_mla.SPARSE_MLA_GLM53[k] for k in ("DR", "DV", "BN", "DOT", "SCALE")})),
        ("SparseMlaHead_v1", lambda: bind(sparse_mla.SparseMlaHead, T=65,
                                          **{k: v for k, v in sparse_mla.SPARSE_MLA_GLM53.items() if k != "NH"})),
        ("SparseMla_v1", lambda: bind(sparse_mla.SparseMla, T=65, **sparse_mla.SPARSE_MLA_GLM53)),
        ("SparseMlaBlock_v2", lambda: bind(boolean_sparse_mla.SparseMlaBlockV2, DR=16, DV=16, NVIS=5, FIRST=True, BN=16,
                                           DOT=boolean_sparse_mla.SPARSE_MLA_GLM53["DOT"], SCALE=0.0625)),
        ("SparseMlaHead_v2", lambda: bind(boolean_sparse_mla.SparseMlaHeadV2, T=5, DR=16, DV=16, BN=16, SCALE=0.0625,
                                          DOT=boolean_sparse_mla.SPARSE_MLA_GLM53["DOT"], INV=boolean_sparse_mla.SPARSE_MLA_GLM53["INV"])),
        ("SparseMla_v2", lambda: bind(boolean_sparse_mla.SparseMlaV2, T=5, NH=2, DR=16, DV=16, BN=16, SCALE=0.0625,
                                      DOT=boolean_sparse_mla.SPARSE_MLA_GLM53["DOT"], INV=boolean_sparse_mla.SPARSE_MLA_GLM53["INV"])),
        ("LServe_v2", lambda: lifted.lserve(TINY, 3, 2, (7,), lag=0)),
        ("ServeSpec_v1", lambda: bind(spec.ServeSpec, C=long, CD={**TINY_NOBIAS, "MAX_POS": 16}, LP=3, K=2, A=(2, 2))),
        ("ServeDraft_v1", lambda: bind(spec.ServeDraft, C=long, LP=3, STEPS=2)),
        ("VerifyBranch_v1", lambda: bind(spec.VerifyBranch, C=long, LP=3, K=2)),
        ("LMoeBlock_v1", lambda: moe_pad.block(4, 2, 2, 16, 16)),
        ("LMoeBlockL_v1", lambda: bind(moe_pad.LMoeBlockL, E=4, TOPK=2, VPT=2, H=16, I=16)),
        ("LMoeFfn_v1", lambda: bind(moe_pad.LMoeFfn, E=4, TOPK=2, VPT=2, H=16, I=16)),
        ("GumbelTokenSelect_v1", lambda: bind(sampling.GumbelTokenSelect, V=16)),
        ("GumbelTokenSelectSharedGreedy_v1", lambda: bind(sampling.GumbelTokenSelectSharedGreedy, V=16)),
        ("GumbelTopPTokenSelectSharedGreedy_v1", lambda: bind(sampling.GumbelTopPTokenSelectSharedGreedy, V=16)),
        # the static-seed Gumbel argmax (T = 0.7f) and the erf GELU; LayerNormAten is the b1-eager-v3 vocabulary's
        ("GumbelArgmaxBf16_v1", lambda: bind(sampling.GumbelArgmaxBf16, V=16, SEED=12345, POS=3)),
        ("GumbelArgmaxTempBf16_v1", lambda: bind(sampling.GumbelArgmaxTempBf16, V=16, SEED=12345, POS=3, T=0x3F333333)),
        ("GeluErf_v1", lambda: bind(act.GeluErf, N=8)),
        ("EmbeddingShard_v1", lambda: bind(b1_tp2.EmbeddingShard, VS=16, H=16, START=16)),
        ("ResidualAdd_v1", lambda: bind(b1.ResidualAdd, N=16)),
        ("RoPEGptJ_v1", lambda: bind(ref_prims.RoPEGptJ, NHEADS=2, D=16)),
        ("RoPEGptJ_v2", lambda: bind(boolean_rope.RoPEGptJV2, NHEADS=2, D=16)),
        ("RoPEInterleavedRef_v1", lambda: bind(ref_prims.RoPEInterleavedRef, D=16, DT="bf16")),
    ]


def _tanh_rn_probe(reads: Callable[[int], Any]) -> Any:
    """`tanh_rn` (2^27 entries of two words) as parameter 1, probed at its ends and its regions' edges (`TANH_RN_REGIONS`)."""
    from verity_vllm.program.registry import tables

    from .vectors import TableProbe

    return TableProbe(1, tables.tanh_rn_table(), 2, tables.TANH_RN_REGIONS, reads)


def _tanh_rn_word(i: int) -> int:
    """The word of `tanh_rn`'s region whose low 27 bits are entry ``i`` (past HI where no region word has them)."""
    from verity_vllm.program.registry import tables

    return 0x38000000 | i if (0x38000000 | i) >= tables.TANH_RN_LO else 0x40000000 | i


#: the registered tables the correspondence probes rather than flattens (`vectors.TableProbe`), by the id of the Definition that
#: reads one; a Definition with a parameter over `checks.Options.input_budget` leaves needs its table here
TABLE_PROBES: dict[str, Callable[[], Any]] = {
    "TableRead27x2_v1": lambda: _tanh_rn_probe(lambda i: [[i], [i | 0xF8000000]]),
    "TanhF32Rn_v2": lambda: _tanh_rn_probe(lambda i: [[_tanh_rn_word(i)], [_tanh_rn_word(i) | 0x80000000]]),
}


def table_probe(fn: Any) -> Any:
    """The `vectors.TableProbe` of the registered table ``fn`` reads (`TABLE_PROBES`), or None."""
    make = TABLE_PROBES.get(fn.id)
    return None if make is None else make()


#: roots the catalog could not build: (label, error); the suite requires none
ROOT_ERRORS: list[tuple[str, str]] = []


def _vocabulary_bindings() -> list[tuple[str, Any]]:
    """Every frontend vocabulary kind (the Definitions the torch frontend binds served ops to) at small statics: (label, the bound
    Definition, or the error)."""
    from verity.ml.prims import BlackwellE4m3QmmaDot32
    from verity_vllm.program.frontend.rules import vocab as V
    from verity_vllm.program.frontend.target_profile import TargetProfile
    from verity_vllm.program.registry import prims as P

    small = {"K": 32, "N": 16, "V": 16, "H": 16, "T": 5, "NH": 2, "KVH": 1, "D": 16, "NHEADS": 2, "I": 16, "EPS": 1e-5, "C": 0.5,
             "CAP": 30.0, "BIAS": True, "APPROX": "tanh", "SCALE_OP": "mul", "E": 4, "TOPK": 2, "VPT": 2, "DOT": P.AmpereBF16TcDot16,
             "INV": P.Fa2InvSum, "WORLD": 2, "G": 128, "W": 32, "BITS": 0, "BN": 16, "VS": 16, "START": 16, "MASKED_FROM": 0, "FA2_MASKED_FROM": 0,
             "SCALE": "shared-per-block", "S": 4, "FP8_DOT": BlackwellE4m3QmmaDot32,
             "CONSTRUCTION": "shared-greedy", "TARGET": TargetProfile.default()}
    wide = {**small, "K": 256, "N": 256, "H": 256}           # FP8's 128-element blocks and 256-thread quantization
    vocabularies = [v for v in vars(V).values() if isinstance(v, V.Vocabulary)]
    kinds = [(f"{v.name}/{k}", make) for v in vocabularies for k, make in v.kinds.items()]
    kinds += [(f"dense/{k}", make) for k, make in V._dense_kinds()[0].items()]
    out = []
    for label, make in kinds:
        code = make.__code__
        args = code.co_varnames[:code.co_argcount]
        required = args[:len(args) - len(make.__defaults__ or ())]
        optional = tuple(a for a in args[len(required):] if a in small)   # a defaulted static is bound both ways
        for dt in (("bf16", "f32") if "DT" in args else (None,)):
            variants = [(required, "")] + ([(required + optional, "+" + "+".join(optional))] if optional else [])
            for names, suffix in variants:
                fn, err = None, None
                for table in (small, wide):
                    try:
                        fn = make(**{a: (dt if a == "DT" else table[a]) for a in names})
                        _ = getattr(fn, "gates", None)
                        break
                    except Exception as e:  # noqa: BLE001  (retried with the wide table, then named)
                        err = f"{type(e).__name__}: {str(e)[:160]}"
                out.append((label + suffix, fn if fn is not None else err))
    return out


def _vocabulary_roots() -> list[Any]:
    out = []
    for label, fn in _vocabulary_bindings():
        if isinstance(fn, str):
            ROOT_ERRORS.append((label, fn))
        else:
            out.append(fn)
    return out


_CALLS: set[str] = set()


def call_families() -> set[str]:
    """The Definition families a served Program calls, where the partition invariant applies: the replay's row families
    (`verity_vllm.program.kernels.rows.ROWS`), every frontend vocabulary kind, every template's Definition, and the Boolean
    version of each of those (`boolean_call_families`: the Program on bits calls them in their place).  Any other Definition's
    gates are partitioned inside the Call that contains them."""
    if not _CALLS:
        load_registries()
        from verity_vllm.program.kernels import rows

        _CALLS.update(rows.ROWS)
        _CALLS.update(family(fn) for _label, fn in _vocabulary_bindings() if not isinstance(fn, str))
        for _label, sub, sts in templates():
            _CALLS.update(family(sub.definition(**st)) for st in sts)
        _CALLS.update(boolean_call_families(set(_CALLS)))
    return _CALLS


def boolean_call_families(word_calls: set[str]) -> set[str]:
    """Sigma's image of the word Call families ``word_calls``: the family of `verity_vllm.program.boolean.boolean_version` (the
    switch's table of `word=` links) of each catalog specialization of a word Call family that has one."""
    from verity.evaluation import ScalarReference
    from verity.ir import boolean as BL
    from verity_vllm.program.boolean import NoBooleanVersion, boolean_version

    out: set[str] = set()
    for fn in suite().values():
        if family(fn) not in word_calls or isinstance(fn, ScalarReference) or BL.is_boolean(fn):
            continue
        try:
            out.add(family(boolean_version(fn)))
        except NoBooleanVersion:
            continue
    return out


def catalog(by_unit: list | None = None) -> list[Any]:
    """The roots: every primitive, every static-free composite, and the bindings above; with ``by_unit``, each binding's
    (units, roots) appended to it."""
    from verity.ir.defs import REGISTRY, CompositeDefinition, PrimitiveDefinition, bind

    load_registries()
    roots: list[Any] = []
    for d in list(REGISTRY.defs.values()):
        if isinstance(d, PrimitiveDefinition):
            roots.append(d)
        elif isinstance(d, CompositeDefinition) and not d.statics:
            roots.append(bind(d))
    ROOT_ERRORS.clear()
    for units, make in ((("_core_roots",), _core_roots), (("_boolean_roots",), _boolean_roots),
                        (("_boolean_gather_roots",), _boolean_gather_roots), (("_universal_roots",), _universal_roots),
                        (("_merkle_roots",), _merkle_roots), (("_kernel_roots",), _kernel_roots),
                        (("_vocabulary_roots", "_vocabulary_bindings"), _vocabulary_roots), (("_pouw_roots",), _pouw_roots)):
        roots += _produce(units, make, by_unit)
    for label, make in _model_roots():
        def built(make=make):
            fn = make()
            _ = fn.body
            return [fn]
        try:
            roots += _produce((f"_model_roots:{label}",), built, by_unit)
        except Exception as e:  # noqa: BLE001  (named, and the suite fails on it)
            ROOT_ERRORS.append((label, f"{type(e).__name__}: {str(e)[:200]}"))
    for _label, sub, sts in templates():
        roots += _produce((f"TEMPLATE_PARAMS:{sub.template.id}",),
                          lambda sub=sub, sts=sts: [x for st in sts for x in (sub.definition(**st), sub.target(**st))], by_unit)
    return roots


def reachable(roots: Iterable[Any]) -> dict[str, Any]:
    """Every Definition (primitive, complete specialisation) or reference reachable from ``roots``, by id."""
    from verity.evaluation import ScalarReference
    from verity.ir.defs import PrimitiveDefinition, SpecializedDefinition

    out: dict[str, Any] = {}
    stack = list(roots)
    while stack:
        fn = stack.pop()
        if fn.id in out:
            continue
        if isinstance(fn, (PrimitiveDefinition, ScalarReference)):
            out[fn.id] = fn
        elif isinstance(fn, SpecializedDefinition):
            out[fn.id] = fn
            stack.extend(n.fn for n in fn.body.nodes)
    return out


#: specialisations over this many gates are checked only when their family has none smaller (served-size kernel self-check
#: targets such as `TokenSelect_v1{V=151936}` stay with the kernel tests)
MAX_GATES = 150_000


def family(fn: Any) -> str:
    return getattr(fn, "base_id", None) or fn.id


def select(found: dict[str, Any], max_gates: int = MAX_GATES) -> dict[str, Any]:
    """The specialisations the suite checks: every one of at most ``max_gates`` gates, and a family's smallest otherwise, with its
    smallest Boolean one too when the smallest is not (a Boolean family bound at a word MUFU and at its Boolean version, which
    the Program on bits calls)."""
    from verity.ir import boolean as BL

    by: dict[str, list[Any]] = {}
    for fn in found.values():
        by.setdefault(family(fn), []).append(fn)
    out = {}
    for fns in by.values():
        keep = [f for f in fns if getattr(f, "gates", 1) <= max_gates]
        if not keep:
            keep = [min(fns, key=lambda f: f.gates)]
            if not BL.is_boolean(keep[0]) and (bits := [f for f in fns if BL.is_boolean(f)]):
                keep.append(min(bits, key=lambda f: f.gates))
        for f in keep:
            out[f.id] = f
    return dict(sorted(out.items()))


def suite() -> dict[str, Any]:
    """Every Definition the suite checks, by id: `select`'s, and PoUW's roots whatever their size (each P's whole call and
    window share a family and a gate count, so `select` alone keeps one of them).  Records each one's `BINDINGS`."""
    by_unit: list = []
    out = select(reachable(catalog(by_unit)))
    out.update({fn.id: fn for fn in _produce(("_pouw_roots",), _pouw_roots, by_unit)})
    for units, roots in by_unit:
        for fid in reachable(roots):
            if fid in out:
                BINDINGS.setdefault(fid, set()).update(units)
    return dict(sorted(out.items()))


__all__ = ["MAX_GATES", "REGISTERED", "ROOT_ERRORS", "TABLE_PROBES", "TEMPLATE_PARAMS", "TINY", "boolean_call_families", "call_families",
           "catalog", "definition", "family", "instance_statics", "kind_of", "load_registries", "reachable", "select", "subcircuit",
           "suite", "table_probe", "templates"]
