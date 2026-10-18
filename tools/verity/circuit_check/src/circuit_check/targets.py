"""What circuit-check checks: a Definition id, a subcircuit template (or one of its subcircuits), a served row; and the catalog
the suite runs over, every registered Definition and template at small bindings.

    definition  ``F32Add_v1``, ``RMSNormTriton_v1{N=64,EPS=1e-05}``, ``GemmCoordinate_v2{K=64,DOT=AmpereBF16TcDot16_v2}``: a
                registered Definition bound to the statics its id writes (a function-valued static by its id, a record by the
                digest its id prints, ``StepBody_v1{C={d0dc81b7fa41},T=4,...}``, one of the records the suite binds)
    template    ``rope-head`` (its example parameters), ``rope-head:D=128`` (parameters given), or a subcircuit id
                (``gemm-coordinate/k2048/sm80-mma-bf16``); ``attention-head`` is checked at the ends of its key-count range
    row         a Build's request Program: its directory or its ``instances.json.gz``

The catalog (`catalog`) is roots at small statics: every primitive, every static-free composite, core's GEMM Definitions and
silicon references, every vLLM kernel's self-check specialisations, each template, and tiny whole-model configurations; the
suite checks every specialisation reachable from them (`suite`), and fails by name for a registered Definition family no root
reaches: a new Definition needs a binding.  A binding lives beside the Definitions it binds, in their module's sidecar
(`sidecars`), in this module's forms: a root function (``_X_roots``, `root_functions`), a `_model_roots` entry, a
``TEMPLATE_PARAMS`` entry; a module with a sidecar is one the registries import (`load_registries`).  This module keeps what no
one Definition module owns.
"""
from __future__ import annotations

import ast
import importlib
import pkgutil
import re
import sys
import zlib
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from . import cache as CC

_LOADED = False
#: the ids registered once every registry module is imported: the authored Definitions (a lazy family's members, such as
#: `Lifted[F32Add_v1]_v2` or `Const32[0x3f800000]_v1`, are made on demand by id and checked where a body uses them)
REGISTERED: list[str] = []


#: the repository's root: a Definition is this tree's when the module defining it is a file under it
REPO = Path(__file__).resolve().parents[5]


def _authored(d: Any) -> bool:
    """A Definition this tree's code defines, decided by the file of the module that defines it: under the repository and
    outside every `tests/` directory and conftest, not one a test module registered in the same session. A Definition whose
    module has no file here to judge by raises, naming it, rather than drop out of `REGISTERED` unchecked."""
    fn = getattr(d, "evaluate", None) or getattr(d, "body", None)
    name = getattr(fn, "__module__", None)
    file = getattr(sys.modules.get(name or ""), "__file__", None)
    if not file:
        raise LookupError(f"{getattr(d, 'id', d)!r}: its defining module {name!r} has no file, so circuit-check can't tell "
                          f"whether this tree authored it")
    path = Path(file).resolve()
    if not path.is_relative_to(REPO):
        return False
    rel = path.relative_to(REPO)
    return "tests" not in rel.parts[:-1] and rel.name != "conftest.py"


def load_registries() -> None:
    """Register every Definition and kernel: each module with a circuit-check sidecar (`sidecars.discover`: a Definition module
    has its bindings, or its pins, beside it), each module of the vLLM registry, and the kernel modules.  The modules with a
    sidecar are in every target's key (`cache.registries`)."""
    global _LOADED
    if _LOADED:
        return
    from verity.ml.circuits.defs import REGISTRY

    from . import sidecars as SC

    for mod in SC.discover():
        importlib.import_module(mod)
    import verity_vllm.program.registry as R
    for m in pkgutil.iter_modules(R.__path__):
        if m.name != "quarantine":
            importlib.import_module(f"{R.__name__}.{m.name}")
    from verity_vllm.program.kernels import rows, twins  # noqa: F401  (their kernels register with core)
    CC.registries(SC.discover())
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
        v = ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return s
    return (v,) if s.startswith("(") and not isinstance(v, tuple) else v    # an id prints a tuple of one as `(7)`


#: a record (dict-valued) static as a Definition id prints it, a prefix of its digest (`verity.ml.circuits.defs`): the record itself is
#: not in the id, so it is looked up among the records the suite's Definitions are bound to
_RECORD = re.compile(r"\{[0-9a-f]{12}\}")
_RECORDS: dict[str, Any] = {}


def _record(text: str) -> Any:
    """The record that a suite Definition's id prints as ``text``."""
    if not _RECORDS:
        for fn in suite().values():
            m = re.fullmatch(r"[^{]+\{(.*)\}", fn.id)
            for k, v in (_split(m[1]).items() if m else ()):
                if _RECORD.fullmatch(v) and k in getattr(fn, "bindings", {}):
                    _RECORDS.setdefault(v, fn.bindings[k])
    if text not in _RECORDS:
        raise KeyError(f"record static {text}: no Definition the suite checks is bound to a record of that digest")
    return _RECORDS[text]


def definition(spec: str) -> Any:
    """The registered Definition ``spec`` names, bound to the statics its id writes, or else the suite's target of that id: a
    Definition that only tracing the catalog's roots registers, such as `boolean_gather.gather_bf16(n)`'s gathers, or one
    whose id writes a dict static as its digest, such as `gf2k.GfResiduals`'s schema."""
    from verity_vllm.query import word as W

    load_registries()
    m = re.fullmatch(r"([^{]+)(?:\{(.*)\})?", spec.strip())
    if m is None:
        raise ValueError(f"not a Definition id: {spec!r}")
    fam, body = m[1], m[2]
    statics = {k: _record(v) if _RECORD.fullmatch(v) else _value(v) for k, v in _split(body).items()} if body else {}
    try:
        if any(isinstance(v, str) and re.fullmatch(r"\{[0-9a-f]{12}\}", v) for v in statics.values()):
            raise KeyError(spec)
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

    try:
        if Path(spec).exists():
            return "row"
    except OSError:            # an id longer than a file name may be (AllReduceMultimem_v1's TREES), so no path
        pass
    head = spec.partition(":")[0].partition("/")[0]
    return "template" if head in TM.TEMPLATES else "definition"


# --- the catalog ----------------------------------------------------------------------------------------------------------------

TINY = {"model": "tiny", "revision": None, "H": 32, "NL": 2, "NH": 2, "KVH": 1, "D": 16, "I": 32, "V": 64, "QKV": 64, "GU": 64,
        "EPS": 1e-5, "ROPE_THETA": 1e4, "MAX_POS": 8, "BIAS": True, "tie_word_embeddings": True, "profile": {}}
TINY_NOBIAS = {**TINY, "BIAS": False, "KVH": 2, "QKV": 96}
#: FP8's per-token quantization reads whole 256-thread blocks: every reduction length is at least 256
TINY_FP8 = {**TINY, "H": 256, "NL": 1, "NH": 2, "KVH": 1, "D": 128, "I": 256, "V": 16, "QKV": 512, "GU": 512}


#: the binding units (`cache`'s units of a binding file: a root function, a `_model_roots` entry, a `TEMPLATE_PARAMS` entry;
#: this module's by name, a sidecar's as ``<file>#<name>``) whose roots reach each suite target, by Definition id (`suite`) or
#: template label (`templates`)
BINDINGS: dict[str, set[str]] = {}
#: the root functions that run first, in catalog order (each file's in turn, this module first); any other ``_X_roots`` a
#: binding file defines runs after them, by name (`root_functions`).  ``_X_roots`` holds ``_X_bindings`` beside it as one
#: binding when the file defines both
ROOT_FUNCTIONS = ("_core_roots", "_boolean_roots", "_boolean_gather_roots", "_universal_roots", "_gf2k_roots",
                  "_merkle_roots", "_lms_roots", "_pous_roots", "_rec_roots", "_kernel_roots", "_vocabulary_roots",
                  "_pouw_roots")


def _root_names(ns: dict[str, Any]) -> set[str]:
    return {n for n in ns if n.startswith("_") and n.endswith("_roots") and n != "_model_roots"}


def binding_files() -> list[tuple[str, dict[str, Any]]]:
    """(unit prefix, namespace) of each binding file: this module (its units by name) and each sidecar, by its module's name
    (``<module>.circuit-check#``, `cache.SUFFIX`), so a unit's name moves with its module.
    A sidecar's ``_X_bindings`` without its ``_X_roots`` is an error: no one runs it."""
    from . import sidecars as SC

    load_registries()
    out: list[tuple[str, dict[str, Any]]] = [("", globals())]
    for mod, m in SC.binding_modules():
        ns = vars(m)
        roots = {*ROOT_FUNCTIONS, *_root_names(ns)}
        known = {*roots, *(n.replace("_roots", "_bindings") for n in roots), "_model_roots"}
        if stray := sorted(n for n in ns if CC._is_binding(n) and n not in known):
            raise ValueError(f"{m.__file__}: {', '.join(stray)} is no binding circuit-check runs (a root function is "
                             f"``_X_roots``, its bindings ``_X_bindings``)")
        out.append((f"{m.__name__}#", ns))
    return out


def root_functions() -> tuple[str, ...]:
    """The root functions the catalog runs, in order: `ROOT_FUNCTIONS`, then every other one a binding file defines, by name."""
    more = set().union(*(_root_names(ns) for _prefix, ns in binding_files())) - set(ROOT_FUNCTIONS)
    return (*ROOT_FUNCTIONS, *sorted(more))


def _root_bindings(name: str) -> list[tuple[tuple[str, ...], Callable[[], list[Any]]]]:
    """(units, root function) of every binding file that defines root function ``name``."""
    pair = name.replace("_roots", "_bindings")
    return [(tuple(prefix + n for n in (name, pair) if n in ns), ns[name]) for prefix, ns in binding_files() if name in ns]


def template_params() -> dict[str, tuple[str, list[dict]]]:
    """{template id: (unit, its parameters)} over every binding file's ``TEMPLATE_PARAMS``; a template in two is an error."""
    out: dict[str, tuple[str, list[dict]]] = {}
    for prefix, ns in binding_files():
        for tid, params in ns.get("TEMPLATE_PARAMS", {}).items():
            if tid in out:
                raise ValueError(f"template {tid} has parameters in two binding files ({out[tid][0]}, {prefix}TEMPLATE_PARAMS)")
            out[tid] = (f"{prefix}TEMPLATE_PARAMS:{tid}", params)
    return out


def _template_unit(tid: str) -> str:
    return template_params().get(tid, (f"TEMPLATE_PARAMS:{tid}",))[0]


def _produce(units: tuple[str, ...], make: Callable[[], list[Any]], by_unit: list | None = None) -> list[Any]:
    """``make()``'s roots, recorded while it runs, with every Definition it makes (`cache.producing`)."""
    with CC.producing(units):
        got = make()
    if by_unit is not None:
        by_unit.append((units, got))
    return got


def templates() -> list[tuple[str, Any, list[dict]]]:
    """(label, subcircuit, instance statics) for every template in the catalogue: its ``TEMPLATE_PARAMS`` (`template_params`),
    else its example."""
    from verity_numerical.bench import templates as TM

    load_registries()
    params_of = template_params()
    out = []
    for tid, t in sorted(TM.TEMPLATES.items()):
        unit = (_template_unit(tid),)
        for params in params_of.get(tid, ("", [dict(t.example)]))[1]:
            sub, sts = _produce(unit, lambda t=t, params=params: (lambda s: [s, instance_statics(s, params)])(t.bind(**params)))
            BINDINGS.setdefault(sub.id, set()).update(unit)
            out.append((sub.id, sub, sts))
    return out


def _kernel_roots() -> list[Any]:
    from verity_vllm.program.kernels import kernel_registry as KR

    out = []
    for r in KR.registered():
        out += list(r.check_targets(np.random.default_rng(zlib.crc32(f"{r.key}/{r.kernel}".encode()))))
    return out


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


def _served_kind_statics() -> dict[str, dict[str, Any]]:
    """The kinds no small table binds (launch blocks, collective routes, split counts, the DSA and sparse-MLA shapes; DeepSeek-V4's
    mHC, FlashMLA, compressors, routers and Marlin MoE), each at a binding its Definition module's sidecar already checks, so they
    add no task."""
    from verity.ml.circuits.boolean.trace import word_of
    from verity.ml.circuits.ops.boolean import gemm as BG
    from verity_vllm.program.registry import dsv4_compressor as DC, dsv4_qnorm_insert as DQ
    from verity_vllm.program.registry import flashmla as FM, flashmla_prefill as FP
    from verity_vllm.program.registry import marlin_moe as MM
    from verity_vllm.program.registry import sparse_mla as SM
    from verity_vllm.program.registry import tp_allreduce as TA

    hop = word_of(BG.HopperBF16WgmmaDot16)
    mla = {k: SM.SPARSE_MLA_GLM53[k] for k in ("NH", "DR", "DV", "BN", "DOT", "INV")} | {"SM_SCALE": SM.SPARSE_MLA_GLM53["SCALE"]}
    trees = (TA.chain(8),) * 4 + ((((0, 1), (2, 3)), ((4, 5), (6, 7))),) * 4
    mhc = {"HC": 4, "REPEAT": 20, "EPS": 1e-6, "HC_EPS": 1e-6, "TN": 64}
    router = {"E": 8, "SCALE": 2.5}
    marlin = {"E": 3, "K": 256, "N": 16, "TK": 128, "TN": 8, "GRID": 5, "DOT": MM.DOT}
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
            "dsa_indexer_k_row": {"D": 128, "ROPE": 64, "EPS": 1e-6}, "dsa_indexer_q": {"H": 32, "D": 128, "ROPE": 64},
            "dsa_mqa_logits": {"N": 4, "H": 32, "D": 128},
            "cublas_splitk_gemm": {"K": 64, "N": 3, "SPLIT_K": 4, "DOT": hop},
            "cublas_splitk_gemm_f32out": {"K": 64, "N": 3, "SPLIT_K": 4, "DOT": hop},
            "batched_gemm": {"B": 2, "K": 32, "N": 3, "DOT": hop},
            "ll_bf16_dotprod": {"N": 2, "K": 256, "BS": 64, "VEC": 2, "FMA": 1},
            "ll_bf16_splitk": {"N": 2, "K": 128, "SPLIT_K": 2, "TILE_K": 32, "WARPS": 2, "DOT": hop},
            "sparse_mla": {"T": 65, **mla}, "sparse_mla_split": {"T": 130, "NS": 2, **mla}, **dsv4}


def _vocabulary_bindings() -> list[tuple[str, Any]]:
    """Every frontend vocabulary kind (the Definitions the torch frontend binds served ops to) at small statics, or at
    `_served_kind_statics`' when no small table binds it: (label, the bound Definition, or the error)."""
    from verity.ml.circuits.ops.prims import BlackwellE4m3QmmaDot32
    from verity_vllm.program.frontend.rules import vocab as V
    from verity_vllm.program.frontend.target_profile import TargetProfile
    from verity_vllm.program.registry import prims as P

    small = {"K": 32, "N": 16, "V": 16, "H": 16, "T": 5, "NH": 2, "KVH": 1, "D": 16, "NHEADS": 2, "I": 16, "EPS": 1e-5, "C": 0.5,
             "CAP": 30.0, "BIAS": True, "APPROX": "tanh", "SCALE_OP": "mul", "E": 4, "TOPK": 2, "VPT": 2, "DOT": P.AmpereBF16TcDot16,
             "INV": P.Fa2InvSum, "WORLD": 2, "G": 128, "W": 32, "BITS": 0, "BN": 16, "VS": 16, "START": 16, "MASKED_FROM": 0, "FA2_MASKED_FROM": 0,
             "SCALE": "shared-per-block", "S": 4, "KB": 32, "SWAP": False, "FP8_DOT": BlackwellE4m3QmmaDot32,
             "CONSTRUCTION": "shared-greedy", "TARGET": TargetProfile.default(), "RSF": 2.5, "KERNEL": "block"}
    wide = {**small, "K": 256, "N": 256, "H": 256}           # FP8's 128-element blocks and 256-thread quantization
    e4m3 = {**small, "G": 32, "DOT": BlackwellE4m3QmmaDot32}  # the block-FP8 MoE pair forms: an e4m3 step, one K tile per scale block
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
                    except Exception as e:  # noqa: BLE001  (retried with the next table, then named)
                        fn, err = None, f"{type(e).__name__}: {str(e)[:160]}"
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
#: what `is_call` decides by without the catalog, each built once: ``words`` (`word_call_families`), ``candidates``
#: (`boolean_call_candidates` of them) and ``certified`` (`certified_boolean_calls` of them)
_SHORT: dict[str, set[str]] = {}


def word_call_families() -> set[str]:
    """The word Call families of a served Program: the replay's row families (`verity_vllm.program.kernels.rows.ROWS`), every
    frontend vocabulary kind's Definition and every template's.  Their bindings build in seconds, without the catalog."""
    if "words" not in _SHORT:
        load_registries()
        from verity_vllm.program.kernels import rows

        words = set(rows.ROWS)
        words.update(family(fn) for _label, fn in _vocabulary_bindings() if not isinstance(fn, str))
        for _label, sub, sts in templates():
            words.update(family(sub.definition(**st)) for st in sts)
        _SHORT["words"] = words
    return _SHORT["words"]


def call_families() -> set[str]:
    """The Definition families a served Program calls, where the partition invariant applies: the word Call families
    (`word_call_families`: the replay's row families, every frontend vocabulary kind, every template's Definition), and the
    Boolean version of each of those (`boolean_call_families`: the Program on bits calls them in their place).  Any other
    Definition's gates are partitioned inside the Call that contains them.  It builds the whole catalog; `is_call` asks whether
    one family is among them, mostly without."""
    if not _CALLS:
        words = word_call_families()
        _CALLS.update(words | boolean_call_families(set(words)))
    return _CALLS


def is_call(fam: str) -> bool:
    """Whether family ``fam`` is in `call_families`, decided without the catalog where its answer doesn't need it: a word Call
    family is; a family outside `boolean_call_candidates` is not; one `certified_boolean_calls` names is; any other (a Boolean
    version of a word Call family that no vocabulary or template binding certifies) is looked up in `call_families`."""
    if _CALLS:
        return fam in _CALLS
    words = word_call_families()
    if fam in words:
        return True
    if "candidates" not in _SHORT:
        _SHORT["candidates"] = boolean_call_candidates(words)
    if fam not in _SHORT["candidates"]:
        return False
    if "certified" not in _SHORT:
        _SHORT["certified"] = certified_boolean_calls(words)
    return fam in _SHORT["certified"] or fam in call_families()


def boolean_call_candidates(word_calls: set[str]) -> set[str]:
    """A superset of `boolean_call_families` of ``word_calls``, from the Boolean table's links alone (no catalog): each family
    `boolean_version` could return for a Definition of a family in ``word_calls``, by its two lookups, a registered Boolean
    Definition whose ``word`` is a Definition of such a family, or whose ``word`` is a function of its bindings and whose name is
    such a family's name.  Each candidate it returns is one of those Definitions, bound."""
    from verity.ml.circuits.defs import CompositeDefinition, bind
    from verity_vllm.program.boolean import _load

    exact, by_name, _errors = _load()
    names = {m[1] for f in word_calls if (m := re.fullmatch(r"(.+)_v(\d+)", f))}
    out = {f"{d.name}_v{d.version}" for ds in exact.values() for d in ds
           if family(bind(d.word) if isinstance(d.word, CompositeDefinition) else d.word) in word_calls}
    out.update(f"{d.name}_v{d.version}" for name, ds in by_name.items() if name in names for d in ds)
    return out


def certified_boolean_calls(word_calls: set[str]) -> set[str]:
    """Families of `boolean_call_families` of ``word_calls`` that the vocabulary and template bindings show without the catalog:
    the Boolean version's family of each binding of a word Call family that is no reference and not Boolean, with at most
    `MAX_GATES` gates.  Each such binding is a catalog root `select` keeps, so the suite holds it whatever else the catalog
    reaches."""
    from verity.ml.circuits import boolean as BL
    from verity.ml.circuits.defs import PrimitiveDefinition, SpecializedDefinition
    from verity_vllm.program.boolean import NoBooleanVersion, boolean_version

    roots = [fn for _label, fn in _vocabulary_bindings() if not isinstance(fn, str)]
    roots += [sub.definition(**st) for _label, sub, sts in templates() for st in sts]
    out: set[str] = set()
    for fn in roots:
        if not isinstance(fn, (PrimitiveDefinition, SpecializedDefinition)) or family(fn) not in word_calls \
                or fn.gates > MAX_GATES or BL.is_boolean(fn):
            continue
        try:
            out.add(family(boolean_version(fn)))
        except NoBooleanVersion:
            continue
    return out


def boolean_call_families(word_calls: set[str]) -> set[str]:
    """Sigma's image of the word Call families ``word_calls``: the family of `verity_vllm.program.boolean.boolean_version` (the
    switch's table of `word=` links) of each catalog specialization of a word Call family that has one."""
    from verity.ml.circuits.evaluation import ScalarReference
    from verity.ml.circuits import boolean as BL
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
    from verity.ml.circuits.defs import REGISTRY, CompositeDefinition, PrimitiveDefinition, bind

    load_registries()
    roots: list[Any] = []
    for d in list(REGISTRY.defs.values()):
        if isinstance(d, PrimitiveDefinition):
            roots.append(d)
        elif isinstance(d, CompositeDefinition) and not d.statics:
            roots.append(bind(d))
    ROOT_ERRORS.clear()
    for name in root_functions():
        for units, make in _root_bindings(name):
            roots += _produce(units, make, by_unit)
    for prefix, ns in binding_files():
        for label, make in ns["_model_roots"]() if "_model_roots" in ns else []:
            def built(make=make):
                fn = make()
                _ = fn.body
                return [fn]
            try:
                roots += _produce((f"{prefix}_model_roots:{label}",), built, by_unit)
            except Exception as e:  # noqa: BLE001  (named, and the suite fails on it)
                ROOT_ERRORS.append((label, f"{type(e).__name__}: {str(e)[:200]}"))
    for _label, sub, sts in templates():
        roots += _produce((_template_unit(sub.template.id),),
                          lambda sub=sub, sts=sts: [x for st in sts for x in (sub.definition(**st), sub.target(**st))], by_unit)
    return roots


def reachable(roots: Iterable[Any]) -> dict[str, Any]:
    """Every Definition (primitive, complete specialisation) or reference reachable from ``roots``, by id."""
    from verity.ml.circuits.evaluation import ScalarReference
    from verity.ml.circuits.defs import PrimitiveDefinition, SpecializedDefinition

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

#: everything the last `suite` reached, `select`'s drops among it: the task list holds what their builds read (`cli.run`)
REACHED: dict[str, Any] = {}


def family(fn: Any) -> str:
    return getattr(fn, "base_id", None) or fn.id


def select(found: dict[str, Any], max_gates: int = MAX_GATES) -> dict[str, Any]:
    """The specialisations the suite checks: every one of at most ``max_gates`` gates, and a family's smallest otherwise, with its
    smallest Boolean one too when the smallest is not (a Boolean family bound at a word MUFU and at its Boolean version, which
    the Program on bits calls)."""
    from verity.ml.circuits import boolean as BL

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
    REACHED.clear()
    REACHED.update(reachable(catalog(by_unit)))
    out = select(REACHED)
    for units, make in _root_bindings("_pouw_roots"):
        out.update({fn.id: fn for fn in _produce(units, make, by_unit)})
    for units, roots in by_unit:
        for fid in reachable(roots):
            if fid in out:
                BINDINGS.setdefault(fid, set()).update(units)
    return dict(sorted(out.items()))


__all__ = ["MAX_GATES", "REACHED", "REGISTERED", "ROOT_ERRORS", "ROOT_FUNCTIONS", "TABLE_PROBES", "TINY",
           "binding_files", "boolean_call_candidates", "boolean_call_families", "call_families", "catalog",
           "certified_boolean_calls", "definition", "family", "instance_statics", "is_call", "kind_of", "load_registries",
           "reachable", "root_functions", "select", "subcircuit", "suite", "table_probe", "template_params", "templates",
           "word_call_families"]
