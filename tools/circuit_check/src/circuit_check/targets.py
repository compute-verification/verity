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

_LOADED = False
#: the ids registered once every registry module is imported: the authored Definitions (a lazy family's members, such as
#: `Lifted[F32Add_v1]_v2` or `Const32[0x3f800000]_v1`, are made on demand by id and checked where a body uses them)
REGISTERED: list[str] = []


def _authored(d: Any) -> bool:
    """A Definition the core or the vLLM registry defines (not one a test module registered in the same session)."""
    fn = getattr(d, "evaluate", None) or getattr(d, "body", None)
    return getattr(fn, "__module__", "").partition(".")[0] in ("verity", "verity_vllm")


def load_registries() -> None:
    """Register every Definition and kernel: core's `verity.ml`, and each module of the vLLM registry and kernel registry."""
    global _LOADED
    if _LOADED:
        return
    from verity.ir.defs import REGISTRY

    import verity.ml.gemm  # noqa: F401
    import verity.ml.kernels  # noqa: F401
    import verity.ml.prims  # noqa: F401
    import verity_vllm.program.registry as R
    for m in pkgutil.iter_modules(R.__path__):
        if m.name != "quarantine":
            importlib.import_module(f"{R.__name__}.{m.name}")
    from verity_vllm.program.kernels import rows, twins  # noqa: F401  (their kernels register with core)
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
    """The registered Definition ``spec`` names, bound to the statics its id writes."""
    from verity_vllm.query import word as W

    load_registries()
    m = re.fullmatch(r"([^{]+)(?:\{(.*)\})?", spec.strip())
    if m is None:
        raise ValueError(f"not a Definition id: {spec!r}")
    fam, body = m[1], m[2]
    statics = {k: _value(v) for k, v in _split(body).items()} if body else {}
    fn = W.specialization(fam, statics)
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


def templates() -> list[tuple[str, Any, list[dict]]]:
    """(label, subcircuit, instance statics) for every template in the catalogue: `TEMPLATE_PARAMS`, else its example."""
    from verity_numerical.bench import templates as TM

    load_registries()
    out = []
    for tid, t in sorted(TM.TEMPLATES.items()):
        for params in TEMPLATE_PARAMS.get(tid, [dict(t.example)]):
            sub = t.bind(**params)
            out.append((sub.id, sub, instance_statics(sub, params)))
    return out


def _core_roots() -> list[Any]:
    from verity.ir.defs import bind
    from verity.ml import gemm as G
    from verity.ml import kernels as K
    from verity.ml import prims as P

    A, H = P.AmpereBF16TcDot16, P.HopperBF16WgmmaDot16
    return [bind(G.DotBf16, K=64, DOT=A), bind(G.DotBf16, K=48, DOT=H), bind(G.GemmCoordinate, K=64, DOT=A),
            bind(G.GemmCoordinate, K=64, DOT=H), bind(G.Gemm, K=32, N=3, DOT=A), K.HOPPER_E4M3_STEP, K.HOPPER_E4M3_DOT.bind(K=128),
            K.ADA_E4M3_STEP, K.ADA_E4M3_DOT.bind(K=128), K.SM120_NVF4_STEP, K.SM120_NVF4_DOT.bind(K=256), K.SM120_MXF4_STEP,
            K.SM120_MXF4_DOT.bind(K=256)]


def _kernel_roots() -> list[Any]:
    from verity_vllm.program.kernels import kernel_registry as KR

    out = []
    for r in KR.registered():
        out += list(r.check_targets(np.random.default_rng(zlib.crc32(f"{r.key}/{r.kernel}".encode()))))
    return out


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """Tiny whole-model configurations (the vLLM tests' `TINY`) and the leaf families they do not reach."""
    from verity.ir.defs import bind
    from verity_vllm.program.registry import b1, b1_tp2, fp8, hopper, lifted, moe_pad, ref_prims, sampling, serve3, spec

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
        ("RMSNormFusedCuda_v1", lambda: bind(b1.RMSNormFusedCudaV1, N=64, EPS=1e-5)),
        ("LMoeFfnL_v1", lambda: bind(moe_pad.LMoeFfnL, E=4, TOPK=2, VPT=2, H=16, I=16)),
        ("AllReduce_v2", lambda: b1_tp2.all_reduce(4, 8)),
        ("AllGather_v1", lambda: b1_tp2.all_gather(4, 8)),
        ("QuantLinearFp8_v1", lambda: bind(fp8.QuantLinearFp8, K=256, N=2, BIAS=True)),
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
        ("EmbeddingShard_v1", lambda: bind(b1_tp2.EmbeddingShard, VS=16, H=16, START=16)),
        ("ResidualAdd_v1", lambda: bind(b1.ResidualAdd, N=16)),
        ("RoPEGptJ_v1", lambda: bind(ref_prims.RoPEGptJ, NHEADS=2, D=16)),
        ("RoPEInterleavedRef_v1", lambda: bind(ref_prims.RoPEInterleavedRef, D=16, DT="bf16")),
    ]


#: roots the catalog could not build: (label, error); the suite requires none
ROOT_ERRORS: list[tuple[str, str]] = []


def _vocabulary_bindings() -> list[tuple[str, Any]]:
    """Every frontend vocabulary kind (the Definitions the torch frontend binds served ops to) at small statics: (label, the bound
    Definition, or the error)."""
    from verity_vllm.program.frontend.rules import vocab as V
    from verity_vllm.program.registry import prims as P

    small = {"K": 32, "N": 16, "V": 16, "H": 16, "T": 5, "NH": 2, "KVH": 1, "D": 16, "NHEADS": 2, "I": 16, "EPS": 1e-5, "C": 0.5,
             "CAP": 30.0, "BIAS": True, "APPROX": "tanh", "SCALE_OP": "mul", "E": 4, "TOPK": 2, "VPT": 2, "DOT": P.AmpereBF16TcDot16,
             "INV": P.Fa2InvSum, "WORLD": 2, "G": 128, "W": 32, "BITS": 0, "BN": 16, "VS": 16, "START": 16, "MASKED_FROM": 0}
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
    (`verity_vllm.program.kernels.rows.ROWS`), every frontend vocabulary kind, and every template's Definition.  Any other
    Definition's gates are partitioned inside the Call that contains them."""
    if not _CALLS:
        load_registries()
        from verity_vllm.program.kernels import rows

        _CALLS.update(rows.ROWS)
        _CALLS.update(family(fn) for _label, fn in _vocabulary_bindings() if not isinstance(fn, str))
        for _label, sub, sts in templates():
            _CALLS.update(family(sub.definition(**st)) for st in sts)
    return _CALLS


def catalog() -> list[Any]:
    """The roots: every primitive, every static-free composite, and the bindings above."""
    from verity.ir.defs import REGISTRY, CompositeDefinition, PrimitiveDefinition, bind

    load_registries()
    roots: list[Any] = []
    for d in list(REGISTRY.defs.values()):
        if isinstance(d, PrimitiveDefinition):
            roots.append(d)
        elif isinstance(d, CompositeDefinition) and not d.statics:
            roots.append(bind(d))
    ROOT_ERRORS.clear()
    roots += _core_roots() + _kernel_roots() + _vocabulary_roots()
    for label, make in _model_roots():
        try:
            fn = make()
            _ = fn.body
            roots.append(fn)
        except Exception as e:  # noqa: BLE001  (named, and the suite fails on it)
            ROOT_ERRORS.append((label, f"{type(e).__name__}: {str(e)[:200]}"))
    for _label, sub, sts in templates():
        for st in sts:
            roots += [sub.definition(**st), sub.target(**st)]
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
    """The specialisations the suite checks: every one of at most ``max_gates`` gates, and a family's smallest otherwise."""
    by: dict[str, list[Any]] = {}
    for fn in found.values():
        by.setdefault(family(fn), []).append(fn)
    out = {}
    for fns in by.values():
        small = [f for f in fns if getattr(f, "gates", 1) <= max_gates]
        for f in small or [min(fns, key=lambda f: f.gates)]:
            out[f.id] = f
    return dict(sorted(out.items()))


def suite() -> dict[str, Any]:
    """Every Definition the suite checks, by id."""
    return select(reachable(catalog()))


__all__ = ["MAX_GATES", "REGISTERED", "ROOT_ERRORS", "TEMPLATE_PARAMS", "TINY", "call_families", "catalog", "definition", "family", "instance_statics", "kind_of",
           "load_registries", "reachable", "select", "subcircuit", "suite", "templates"]
