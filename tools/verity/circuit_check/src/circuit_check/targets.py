"""What circuit-check checks: a Definition id, a subcircuit template (or one of its subcircuits), a served workload; and the catalog
the suite runs over, every registered Definition and template at small bindings.

    definition  ``F32Add_v1``, ``RMSNormTriton_v1{N=64,EPS=1e-05}``, ``GemmCoordinate_v2{K=64,DOT=AmpereBF16TcDot16_v2}``: a
                registered Definition bound to the statics its id writes (a function-valued static by its id, a record by the
                digest its id prints, ``StepBody_v1{C={d0dc81b7fa41},T=4,...}``, one of the records the suite binds)
    template    ``rope-head`` (its example parameters), ``rope-head:D=128`` (parameters given), or a subcircuit id
                (``gemm-coordinate/k2048/sm80-mma-bf16``); an attention head is checked at the ends of its key-count range
    row         a served workload's request Program, a Build's: its directory or its ``instances.json.gz``
    sidecar     ``path/to/<module>.circuit-check.py``: the module's own Definitions its bindings reach, and its templates
                (`sidecar_targets`)

The catalog (`catalog`) is roots at small statics: every primitive, every static-free composite, core's GEMM Definitions and
silicon references, every vLLM kernel's self-check specialisations, each template, and tiny whole-model configurations; the
suite checks every specialisation reachable from them (`suite`), and fails by name for a registered Definition family no root
reaches: a new Definition needs a binding.  A binding lives beside the Definitions it binds, in their module's sidecar
(`sidecars`), in this module's forms: a root function (``_X_roots``, `root_functions`), a `_model_roots` entry, a
``TEMPLATE_PARAMS`` entry; a module with a sidecar is one the registries import (`load_registries`).  A root too large to build
whole is listed in ``_by_calls_roots`` and checked by its calls (`by_calls`), never built.  This module keeps what no one
Definition module owns.
"""
from __future__ import annotations

import ast
import importlib
import pkgutil
import re
import sys
import zlib
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

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
    import verity.ml.definitions.registry as R
    for m in pkgutil.iter_modules(R.__path__):
        if m.name != "quarantine":
            importlib.import_module(f"{R.__name__}.{m.name}")
    from verity.ml.definitions.numerics import rows, twins  # noqa: F401  (their kernels register with core)
    CC.registries(SC.discover())
    REGISTERED[:] = sorted(i for i, d in REGISTRY.defs.items() if _authored(d))
    _LOADED = True
    declare_by_calls()


#: the root function a binding file lists its by-calls roots in (`by_calls`): a Program root checked by its calls, never built
BY_CALLS_ROOTS = "_by_calls_roots"
#: the ids of every binding file's by-calls roots (`declare_by_calls`), which the catalog never descends into or builds
BY_CALLS: set[str] = set()


def declare_by_calls() -> None:
    """Every binding file's by-calls roots (``_by_calls_roots``), made as the catalog makes them (`_produce`), into `BY_CALLS`:
    each a complete composite specialization, else the binding file is refused by name."""
    from verity.ml.circuits.defs import SpecializedDefinition

    BY_CALLS.clear()
    for units, make in _root_bindings(BY_CALLS_ROOTS):
        for fn in _produce(units, make):
            if not (isinstance(fn, SpecializedDefinition) and fn.complete):
                raise TypeError(f"{units[0]}: a by-calls root is a composite specialization with every static bound, not {fn!r}")
            BY_CALLS.add(fn.id)


def is_by_calls(fn: Any) -> bool:
    """Whether ``fn`` is a by-calls root a binding file declares (`BY_CALLS`)."""
    return bool(BY_CALLS) and getattr(fn, "id", None) in BY_CALLS


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


def _items(body: str) -> list[str]:
    """A tuple's elements as an id prints them, split at its top-level commas."""
    out, depth, cur = [], 0, ""
    for ch in body:
        depth += (ch in "{[(") - (ch in "}])")
        if ch == "," and depth == 0:
            out.append(cur.strip())
            cur = ""
            continue
        cur += ch
    return [*out, cur.strip()] if cur.strip() or out else out


def _value(s: str) -> Any:
    try:
        v = ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return s
    return (v,) if s.startswith("(") and not isinstance(v, tuple) else v    # an id prints a tuple of one as `(7)`


#: a Definition-valued static with statics of its own, as an id prints it (`verity.ml.circuits.defs._fmt_static`): its own id
_DEFINITION = re.compile(r"[A-Za-z_][^{}()\[\],=\s]*(?:\[[^{}()=\s]*\][^{}()\[\],=\s]*)?\{.*\}", re.S)
#: a bare Definition id (no statics), as it stands in a tuple
_BARE = re.compile(r"[A-Za-z_][\w.]*(?:\[[^{}()=\s]*\][\w.]*)?")


def _static(text: str) -> Any:
    """A static as a Definition's id writes it: a record by its digest (`_record`), a literal, a Definition by its own id with
    its statics (`definition`, so a Definition among those parses the same way), a tuple element by element.  A bare function
    id stays its text at the top level, which `call_scope.specialization` resolves; in a tuple it is resolved here, as there."""
    if _RECORD.fullmatch(text):
        return _record(text)
    v = _value(text)
    if v is not text:
        return v
    if _DEFINITION.fullmatch(text):
        return definition(text)
    if text.startswith("(") and text.endswith(")"):
        return tuple(_element(x) for x in _items(text[1:-1]))
    return text


def _element(text: str) -> Any:
    """A tuple element (`_static`), a bare id among them the registered Definition it names, as `call_scope.specialization`
    resolves a function-valued static, else its text."""
    from verity.ml.definitions import word as W

    v = _static(text)
    if v is not text or not _BARE.fullmatch(text):
        return v
    try:
        return W.specialization(text, {})
    except KeyError:
        return text


#: a record (dict-valued) static as a Definition id prints it, a prefix of its digest (`verity.ml.circuits.defs`): the record itself is
#: not in the id, so it is looked up among the records the catalog's Definitions are bound to
_RECORD = re.compile(r"\{[0-9a-f]{12}\}")
_RECORDS: dict[str, Any] = {}
#: where `_record` has looked: ``roots`` (`binding_roots`), ``suite``
_LOOKED: set[str] = set()


def _note_records(fns: Iterable[Any]) -> None:
    """Each record a Definition among ``fns`` is bound to, or one bound among their statics (recursively), by the digest its id
    prints: the first of a digest stays."""
    stack, seen = list(fns)[::-1], set()
    while stack:
        fn = stack.pop()
        b = getattr(fn, "bindings", None)
        if id(fn) in seen or not isinstance(b, dict):
            continue
        seen.add(id(fn))
        m = re.fullmatch(r"[^{]+\{(.*)\}", fn.id)
        for k, v in (_split(m[1]).items() if m else ()):
            if _RECORD.fullmatch(v) and k in b:
                _RECORDS.setdefault(v, b[k])
        nested, more = list(b.values()), []
        while nested:
            v = nested.pop()
            if isinstance(v, (list, tuple)):
                nested += v
            elif isinstance(v, dict):
                nested += v.values()
            elif isinstance(getattr(v, "bindings", None), dict):
                more.append(v)
        stack += more[::-1]


def _record(text: str) -> Any:
    """The record that a catalog Definition's id prints as ``text``: looked up among the binding roots and the Definitions bound
    among their statics (`binding_roots`), and only where none is bound to it among the suite's Definitions (the catalog)."""
    for where, fns in (("roots", binding_roots), ("suite", lambda: suite().values())):
        if text in _RECORDS:
            break
        if where not in _LOOKED:
            _note_records(fns())
            _LOOKED.add(where)
    if text not in _RECORDS:
        raise KeyError(f"record static {text}: no Definition the suite checks is bound to a record of that digest")
    return _RECORDS[text]


def definition(spec: str) -> Any:
    """The registered Definition ``spec`` names, bound to the statics its id writes (`_static`: a Definition-valued one by its
    own id, ``EXACT=GemmAccumulatorS8_v2{K=64,DOT=BlackwellS8ImmaDot32_v2}``, bound the same way), or else the suite's target of
    that id: a Definition that only tracing the catalog's roots registers, such as `boolean_gather.gather_bf16(n)`'s gathers,
    or one whose id writes a dict static as its digest, such as `gf2k.GfResiduals`'s schema."""
    from verity.ml.definitions import word as W

    load_registries()
    m = re.fullmatch(r"([^{]+)(?:\{(.*)\})?", spec.strip())
    if m is None:
        raise ValueError(f"not a Definition id: {spec!r}")
    fam, body = m[1], m[2]
    statics = {k: _static(v) for k, v in _split(body).items()} if body else {}
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
    """An attention head (``attention-head``, ``attention-head-v18``): the key counts at the ends of its range (``T = lo:hi`` of
    ``params``, else the template's example); every other template: none."""
    if sub.template.id not in ("attention-head", "attention-head-v18"):
        return [{}]
    lo, _, hi = str((params or {}).get("T") or sub.template.example["T"]).partition(":")
    return [{"T": t} for t in sorted({int(lo), int(hi or lo)})]


def kind_of(spec: str) -> str:
    from pathlib import Path

    from verity_numerical.bench import templates as TM

    try:
        if Path(spec).exists():
            return "sidecar" if spec.endswith(f"{CC.SUFFIX}.py") else "row"
    except OSError:            # an id longer than a file name may be (AllReduceMultimem_v1's TREES), so no path
        pass
    head = spec.partition(":")[0].partition("/")[0]
    return "template" if head in TM.TEMPLATES else "definition"


# --- the catalog ----------------------------------------------------------------------------------------------------------------

TINY = {"model": "tiny", "revision": None, "H": 32, "NL": 2, "NH": 2, "KVH": 1, "D": 16, "I": 32, "V": 64, "QKV": 64, "GU": 64,
        "EPS": 1e-5, "ROPE_THETA": 1e4, "MAX_POS": 8, "BIAS": True, "tie_word_embeddings": True, "profile": {}}
TINY_NOBIAS = {**TINY, "BIAS": False, "KVH": 2, "QKV": 96}
#: FP8's per-token quantization reads whole 256-thread CTAs: every reduction length is at least 256
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
    from verity.ml.definitions.numerics import kernel_registry as KR

    out = []
    for r in KR.registered():
        out += list(r.check_targets(np.random.default_rng(zlib.crc32(f"{r.key}/{r.kernel}".encode()))))
    return out


def _tanh_rn_probe(reads: Callable[[int], Any]) -> Any:
    """`tanh_rn` (2^27 entries of two words) as parameter 1, probed at its ends and its regions' edges (`TANH_RN_REGIONS`)."""
    from verity.ml.definitions.registry import tables

    from .vectors import TableProbe

    return TableProbe(1, tables.tanh_rn_table(), 2, tables.TANH_RN_REGIONS, reads)


def _tanh_rn_word(i: int) -> int:
    """The word of `tanh_rn`'s region whose low 27 bits are entry ``i`` (past HI where no region word has them)."""
    from verity.ml.definitions.registry import tables

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
    """The kinds no small table binds (launch tiles and CTA sizes, collective routes, split counts, the DSA and sparse-MLA shapes; DeepSeek-V4's
    mHC, FlashMLA, compressors, routers and Marlin MoE; the int8 attention row, whose head dim is a multiple of 32), each at a
    binding its Definition module's sidecar already checks, so they add no task."""
    from verity.ml.circuits.boolean.trace import word_of
    from verity.ml.circuits.ops.boolean import gemm as BG
    from verity.ml.definitions.registry import dsv4_compressor as DC, dsv4_qnorm_insert as DQ
    from verity.ml.definitions.registry import flashmla as FM, flashmla_prefill as FP
    from verity.ml.definitions.registry import marlin_moe as MM
    from verity.ml.definitions.registry import sparse_mla as SM
    from verity.ml.definitions.registry import tp_allreduce as TA

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
    from verity.ml.circuits.ops.prims import BlackwellE4m3QmmaDot32
    from verity_vllm import TargetProfile
    from verity_vllm.program.frontend.rules import vocab as V
    from verity.ml.definitions.registry import prims as P

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
#: what `is_call` decides by without the catalog: ``words`` (`word_call_families`) and ``candidates`` (`boolean_call_candidates`
#: of them), each built once, and the candidates it was asked about that `certified_boolean_calls` names (``certified``) and
#: doesn't (``uncertified``)
_SHORT: dict[str, set[str]] = {}
#: the catalog's roots, made without building a body (`binding_roots`), each part once: ``bindings`` (`_kinds_and_templates`) and
#: ``roots`` (the registered ones, every other root function's and each model's); ``unbound``, the vocabulary kinds that made none
_BOUND: dict[str, list[Any]] = {}


def binding_roots() -> list[Any]:
    """The catalog's roots, unbuilt: the vocabulary kinds' and templates' Definitions (`_kinds_and_templates`), the registered
    primitives and static-free composites, every other root function's roots and each `_model_roots` entry's root (made, not
    built: a whole model's body is most of the catalog's build).  Each made as the catalog makes it (`_produce`, so its binding
    units hold it); a binding that raises makes none here (the catalog fails on it by name).  No helper here is named as a
    binding (``_X_roots``, ``_X_bindings``): the cache would key it as one rather than as this file's common code."""
    load_registries()
    if "roots" not in _BOUND:
        roots = _registered()
        for name in root_functions():
            for units, make in _root_bindings(name) if name != "_vocabulary_roots" else ():
                try:
                    roots += _produce(units, make)
                except Exception:  # noqa: BLE001  (see the docstring)
                    continue
        for prefix, ns in binding_files():
            for label, make in ns["_model_roots"]() if "_model_roots" in ns else []:
                try:
                    roots += _produce((f"{prefix}_model_roots:{label}",), lambda make=make: [make()])
                except Exception:  # noqa: BLE001  (see the docstring)
                    continue
        _BOUND["roots"] = roots
    return [*_kinds_and_templates(), *_BOUND["roots"]]


def _kinds_and_templates() -> list[Any]:
    """The vocabulary kinds' Definitions (`_vocabulary_bindings`) and each template's at its instance statics, built once, each
    under the binding units the catalog makes it under (`_produce`): whichever makes a Definition first gives it its `made_by`."""
    if "bindings" not in _BOUND:
        load_registries()
        units = _root_bindings("_vocabulary_roots")[0][0]
        unbound: list[str] = []

        def kinds() -> list[Any]:
            bound = _vocabulary_bindings()
            unbound.extend(f"{label}: {fn}" for label, fn in bound if isinstance(fn, str))
            return [fn for _label, fn in bound if not isinstance(fn, str)]

        out = _produce(units, kinds)
        for _label, sub, sts in templates():
            out += _produce((_template_unit(sub.template.id),), lambda sub=sub, sts=sts: [sub.definition(**st) for st in sts])
        _BOUND["bindings"], _BOUND["unbound"] = out, unbound
    return _BOUND["bindings"]


def word_call_families() -> set[str]:
    """The word Call families of a served Program: the replay's row families (`verity.ml.definitions.numerics.rows.ROWS`), every
    frontend vocabulary kind's Definition and every template's.  Their bindings build in seconds, without the catalog.  A kind that
    doesn't bind is refused by name: left out, its family's partition would be judged as no Call's."""
    if "words" not in _SHORT:
        load_registries()
        from verity.ml.definitions.numerics import rows

        made = _kinds_and_templates()
        if _BOUND["unbound"]:
            raise RuntimeError(f"the Call families are unknown while a vocabulary kind doesn't bind: {_BOUND['unbound']}")
        _SHORT["words"] = set(rows.ROWS) | {family(fn) for fn in made}
    return _SHORT["words"]


def call_families(found: dict[str, Any] | None = None) -> set[str]:
    """The Definition families a served Program calls, where the partition invariant applies: the word Call families
    (`word_call_families`: the replay's row families, every frontend vocabulary kind, every template's Definition), and the
    Boolean version of each of those (`boolean_call_families`: the Program on bits calls them in their place).  Any other
    Definition's gates are partitioned inside the Call that contains them.  It builds the whole catalog, or takes ``found``, a
    `suite` built already (`cli._tasks`'s: a later one would also hold what building its bodies registered); `is_call` asks
    whether one family is among them without it, and refuses a family it can't decide so."""
    if not _CALLS:
        words = word_call_families()
        _CALLS.update(words | boolean_call_families(set(words), found))
    return _CALLS


class Undecided(LookupError):
    """`is_call` can't tell whether a family is a Call without building the whole catalog."""


def is_call(fam: str) -> bool:
    """Whether family ``fam`` is in `call_families`, never building the catalog for it: once `call_families` is known here, by it;
    else a word Call family is; a family outside `boolean_call_candidates` is not; one `certified_boolean_calls` names is.  Any
    other (a Boolean version of a word Call family that only Definitions the catalog reaches inside its roots could show) is
    `Undecided`, by name, with the flags that decide it: the catalog's whole build is what ``--all`` pays once for every target."""
    if _CALLS:
        return fam in _CALLS
    words = word_call_families()
    if fam in words:
        return True
    if "candidates" not in _SHORT:
        _SHORT["candidates"] = boolean_call_candidates(words)
    if fam not in _SHORT["candidates"]:
        return False
    certified, refused = _SHORT.setdefault("certified", set()), _SHORT.setdefault("uncertified", set())
    if fam not in certified | refused:
        (certified if certified_boolean_calls(words, {fam}) else refused).add(fam)
    if fam in certified:
        return True
    raise Undecided(f"{fam}: whether a served Program Calls it needs the whole catalog built; give --as-call or --no-as-call, "
                    "or check it under --all")


def boolean_call_candidates(word_calls: set[str]) -> set[str]:
    """A superset of `boolean_call_families` of ``word_calls``, from the Boolean table's links alone (no catalog): each family
    `boolean_version` could return for a Definition of a family in ``word_calls``, by its two lookups, a registered Boolean
    Definition whose ``word`` is a Definition of such a family, or whose ``word`` is a function of its bindings, whose name is
    such a family's name and whose statics a binding of that family could give (each one of the family's statics or one its word
    computes, ``word_statics``: the by-name lookup reads no other).  Each candidate it returns is one of those Definitions, bound."""
    from verity.ml.circuits.defs import REGISTRY, CompositeDefinition, bind
    from verity.ml.definitions.boolean import _load

    exact, by_name, _errors = _load()
    statics: dict[str, list[set[str] | None]] = {}
    for f in word_calls:
        if m := re.fullmatch(r"(.+)_v(\d+)", f):
            d = REGISTRY.defs.get(f)
            statics.setdefault(m[1], []).append(None if d is None else set(getattr(d, "statics", ())))
    out = {f"{d.name}_v{d.version}" for ds in exact.values() for d in ds
           if family(bind(d.word) if isinstance(d.word, CompositeDefinition) else d.word) in word_calls}
    out.update(f"{d.name}_v{d.version}" for name, ds in by_name.items() for d in ds
               if any(s is None or set(d.statics) <= s | set(getattr(d, "word_statics", {})) for s in statics.get(name, ())))
    return out | on_demand_boolean_families(word_calls)


def on_demand_boolean_families(word_calls: set[str]) -> set[str]:
    """The Boolean version's family of each word Call family made on demand, by its id (`verity.ml.definitions.boolean.
    made_on_demand`: `Const<w>[0x..]`, `GatherBf16x{V}`, `GatherI32x{V}`).  No module registers such a family's members ahead,
    so these come from the families' own patterns, not from the catalog."""
    from verity.ml.definitions.boolean import made_on_demand

    return {family(d) for f in word_calls if (d := made_on_demand(f)) is not None}


def _registered() -> list[Any]:
    """The catalog's first roots: every registered primitive and static-free composite (bound)."""
    from verity.ml.circuits.defs import REGISTRY, CompositeDefinition, PrimitiveDefinition, bind

    roots: list[Any] = []
    for d in list(REGISTRY.defs.values()):
        if isinstance(d, PrimitiveDefinition):
            roots.append(d)
        elif isinstance(d, CompositeDefinition) and not d.statics:
            roots.append(bind(d))
    return roots


def certified_boolean_calls(word_calls: set[str], only: set[str] | None = None) -> set[str]:
    """Families of `boolean_call_families` of ``word_calls`` (those in ``only``, given) that the catalog's roots show without the
    catalog: the Boolean version's family of each root (`binding_roots`) of a word Call family that is not Boolean and has at
    most `MAX_GATES` gates, each a catalog root `select` keeps, so the suite holds it whatever else the catalog reaches.  A root's
    gates are counted (its body built) only when its Boolean version is of a family asked for; a whole model's root is of no word
    Call family, so its body (most of the catalog's build) never is.  With them, `on_demand_boolean_families` (in ``only``)."""
    from verity.ml.circuits import boolean as BL
    from verity.ml.circuits.defs import PrimitiveDefinition, SpecializedDefinition
    from verity.ml.definitions.boolean import boolean_version

    linked: dict[str, list[Any]] = {}
    for fn in binding_roots():
        if not isinstance(fn, (PrimitiveDefinition, SpecializedDefinition)) or family(fn) not in word_calls:
            continue
        try:
            fam = family(boolean_version(fn))
        except Exception:  # noqa: BLE001  (NoBooleanVersion, or a root that doesn't bind: it certifies nothing)
            continue
        if only is None or fam in only:
            linked.setdefault(fam, []).append(fn)
    out = {f for f in on_demand_boolean_families(word_calls) if only is None or f in only}
    for fam, fns in linked.items():
        for fn in fns:
            try:
                if fn.gates <= MAX_GATES and not BL.is_boolean(fn):
                    out.add(fam)
                    break
            except Exception:  # noqa: BLE001  (a root that doesn't build certifies nothing; the catalog fails on it by name)
                continue
    return out


def boolean_call_families(word_calls: set[str], found: dict[str, Any] | None = None) -> set[str]:
    """Sigma's image of the word Call families ``word_calls``: the family of `verity.ml.definitions.boolean.boolean_version` (the
    switch's table of `word=` links) of each catalog specialization of a word Call family that has one, and of each word Call
    family made on demand (`on_demand_boolean_families`)."""
    from verity.ml.circuits.evaluation import ScalarReference
    from verity.ml.circuits import boolean as BL
    out: set[str] = on_demand_boolean_families(word_calls)
    for fn in (suite() if found is None else found).values():
        if family(fn) not in word_calls or isinstance(fn, ScalarReference) or BL.is_boolean(fn):
            continue
        if (f := boolean_family(fn)) is not None:
            out.add(f)
    return out


def boolean_family(fn: Any) -> str | None:
    """The family of ``fn``'s Boolean version (`verity.ml.definitions.boolean.boolean_version`), or None when it has none."""
    from verity.ml.definitions.boolean import NoBooleanVersion, boolean_version

    try:
        return family(boolean_version(fn))
    except NoBooleanVersion:
        return None


def boolean_links() -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """The Boolean table's two lookups (`verity.ml.definitions.boolean._load`) as families: each word Definition's descriptor id
    with the Boolean families linked to it exactly, and each name with the Boolean families whose word is a function of their
    bindings.  `boolean_family` of a Definition is among those of its descriptor id and of its name, or None."""
    from verity.ml.definitions.boolean import _load

    exact, by_name, _errors = _load()
    return ({k: {f"{d.name}_v{d.version}" for d in ds} for k, ds in exact.items()},
            {k: {f"{d.name}_v{d.version}" for d in ds} for k, ds in by_name.items()})


def catalog_units() -> Iterator[tuple[tuple[str, ...], Callable[[], list[Any]], str, str]]:
    """``(units, make, kind, label)`` of each binding, in catalog order: each root function's (``kind`` its name), each
    `_model_roots` entry's (``model``: ``make`` makes the model's root, unbuilt) and each template's (``template``: its
    Definition and target at each of its instance statics).  A generator, so each binding runs before the next is made."""
    for name in root_functions():
        for units, make in _root_bindings(name):
            yield units, make, name, name
    for prefix, ns in binding_files():
        for label, make in ns["_model_roots"]() if "_model_roots" in ns else []:
            yield (f"{prefix}_model_roots:{label}",), lambda make=make: [make()], "model", label
    for label, sub, sts in templates():
        yield ((_template_unit(sub.template.id),),
               lambda sub=sub, sts=sts: [x for st in sts for x in (sub.definition(**st), sub.target(**st))], "template", label)


def pouw_bindings() -> list[tuple[tuple[str, ...], Callable[[], list[Any]]]]:
    """The PoUW root functions `suite` keeps whatever their size."""
    return _root_bindings("_pouw_roots")


#: each `catalog` call, in turn: its registered roots (`_registered`: a binding or a body build can register more) and each
#: binding's (units, kind, label, roots) (`circuit_check.units`)
CATALOGS: list[tuple[list[Any], list[tuple[tuple[str, ...], str, str, list[Any]]]]] = []


def catalog(by_unit: list | None = None) -> list[Any]:
    """The roots: every primitive, every static-free composite, and the bindings above; with ``by_unit``, each binding's
    (units, roots) appended to it."""
    load_registries()
    roots = _registered()
    made: list[tuple[tuple[str, ...], str, str, list[Any]]] = []
    CATALOGS.append((list(roots), made))
    ROOT_ERRORS.clear()
    for units, make, kind, label in catalog_units():
        if kind != "model":
            got = _produce(units, make, by_unit)
        else:
            def built(make=make):
                fn, = make()
                _ = fn.body
                return [fn]
            try:
                got = _produce(units, built, by_unit)
            except Exception as e:  # noqa: BLE001  (named, and the suite fails on it)
                ROOT_ERRORS.append((label, f"{type(e).__name__}: {str(e)[:200]}"))
                continue
        made.append((units, kind, label, got))
        roots += got
    return roots


def reachable(roots: Iterable[Any]) -> dict[str, Any]:
    """Every Definition (primitive, complete specialisation) or reference reachable from ``roots``, by id; a by-calls root
    (`is_by_calls`) without what its body calls, which it doesn't build."""
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
            if fn.id not in BY_CALLS:
                stack.extend(n.fn for n in fn.body.nodes)
    return out


#: specialisations over this many gates are checked only when their family has none smaller or a pin is held for them
#: (`pinned`; served-size kernel self-check targets such as `TokenSelect_v1{V=151936}` stay with the kernel tests)
MAX_GATES = 150_000

#: everything the last `suite` reached, `select`'s drops among it: the task list holds what their builds read (`cli.run`)
REACHED: dict[str, Any] = {}


def family(fn: Any) -> str:
    return getattr(fn, "base_id", None) or fn.id


#: the pin kinds keyed by a Definition's id, whose pin its check compares (`checks.whole_lowering`, `checks.circuit_digest_pin`)
PINNED_KINDS = ("definitions", "digests", "pieces")


def pinned() -> frozenset[str]:
    """The ids of the Definitions a pin is held for (`checks.load_pins`, `PINNED_KINDS`), which `select` keeps whatever their
    size, so ``--all`` compares every pin of a Definition the catalog reaches."""
    from . import checks as CK

    pins = CK.load_pins()
    return frozenset(k for kind in PINNED_KINDS for k in pins.get(kind, {}))


def select(found: dict[str, Any], max_gates: int = MAX_GATES, pinned: Iterable[str] = ()) -> dict[str, Any]:
    """The specialisations the suite checks: every one of at most ``max_gates`` gates, and a family's smallest otherwise, with its
    smallest Boolean one too when the smallest is not (a Boolean family bound at a word MUFU and at its Boolean version, which
    the Program on bits calls); every one whose id is ``pinned`` (`pinned`), whatever its size; and every by-calls root
    (`is_by_calls`), whose gates are never counted."""
    from verity.ml.circuits import boolean as BL

    pinned = frozenset(pinned)
    by: dict[str, list[Any]] = {}
    out = {}
    for fn in found.values():
        if is_by_calls(fn):
            out[fn.id] = fn
        else:
            by.setdefault(family(fn), []).append(fn)
    for fns in by.values():
        keep = [f for f in fns if getattr(f, "gates", 1) <= max_gates]
        if not keep:
            keep = [min(fns, key=lambda f: f.gates)]
            if not BL.is_boolean(keep[0]) and (bits := [f for f in fns if BL.is_boolean(f)]):
                keep.append(min(bits, key=lambda f: f.gates))
        for f in keep + [f for f in fns if f.id in pinned]:
            out[f.id] = f
    return dict(sorted(out.items()))


def suite() -> dict[str, Any]:
    """Every Definition the suite checks, by id: `select`'s, and PoUW's roots whatever their size (each P's whole call and
    window share a family and a gate count, so `select` alone keeps one of them).  Records each one's `BINDINGS`."""
    by_unit: list = []
    REACHED.clear()
    REACHED.update(reachable(catalog(by_unit)))
    out = select(REACHED, pinned=pinned())
    for units, make in pouw_bindings():
        out.update({fn.id: fn for fn in _produce(units, make, by_unit)})
    for units, roots in by_unit:
        for fid in reachable(roots):
            if fid in out:
                BINDINGS.setdefault(fid, set()).update(units)
    return dict(sorted(out.items()))


_SIDECARS: dict[Path, tuple[dict[str, Any], list[tuple[str, Any, list[dict]]]]] = {}
_DROPPED: dict[Path, dict[str, Any]] = {}


def sidecar_dropped(path: str) -> dict[str, Any]:
    """What ``path``'s bindings reach of its module that `sidecar_targets` doesn't keep (`select`), by id."""
    sidecar_targets(path)
    return _DROPPED.get(Path(path).resolve(), {})


def sidecar_targets(path: str) -> tuple[dict[str, Any], list[tuple[str, Any, list[dict]]]]:
    """``circuit-check path/to/<module>.circuit-check.py``'s targets, without the catalog: the Definitions the sidecar's root
    functions (and ``_model_roots``) reach that its module defines (`sidecars.owner`), selected among themselves as `suite`
    selects (`select`, its pinned ones kept), with its ``_pouw_roots``' whatever their size, each with its `BINDINGS`; and (label,
    subcircuit, instance statics) of each template its ``TEMPLATE_PARAMS`` binds, as `templates` makes them.  A family whose
    members here are all over `MAX_GATES` keeps its smallest here, which ``--all`` may not check when another binding reaches a
    smaller one and it has no pin."""
    from verity_numerical.bench import templates as TM

    from . import sidecars as SC

    p = Path(path).resolve()
    if p in _SIDECARS:
        return _SIDECARS[p]
    load_registries()
    mod = next((m for m, files in SC.discover().items() if files.get("py") == p), None)
    if mod is None:
        raise ValueError(f"{path}: no sidecar of a module under `sidecars.PACKAGES` (`sidecars.discover`)")
    ns = vars(dict(SC.binding_modules())[mod])
    prefix = f"{ns['__name__']}#"
    names = sorted(_root_names(ns), key=lambda n: (n not in ROOT_FUNCTIONS, ROOT_FUNCTIONS.index(n) if n in ROOT_FUNCTIONS else 0, n))
    by_unit: list = []
    for name in names:
        _produce(tuple(prefix + n for n in (name, name.replace("_roots", "_bindings")) if n in ns), ns[name], by_unit)
    for label, make in ns["_model_roots"]() if "_model_roots" in ns else []:
        _produce((f"{prefix}_model_roots:{label}",), lambda make=make: [(lambda fn: (fn.body, fn)[1])(make())], by_unit)
    reached = reachable(r for _units, roots in by_unit for r in roots)
    owned = {fid: fn for fid, fn in reached.items() if SC.owner(fn) == mod}
    out = select(owned, pinned=pinned())
    _DROPPED[p] = {fid: fn for fid, fn in owned.items() if fid not in out}
    for units, roots in by_unit:
        if units[0] == f"{prefix}_pouw_roots":
            out.update({fn.id: fn for fn in roots})
    for units, roots in by_unit:
        for fid in reachable(roots):
            if fid in out:
                BINDINGS.setdefault(fid, set()).update(units)
    tpls = []
    for tid, params_list in sorted(ns.get("TEMPLATE_PARAMS", {}).items()):
        unit = (f"{prefix}TEMPLATE_PARAMS:{tid}",)
        for params in params_list:
            sub, sts = _produce(unit, lambda t=TM.TEMPLATES[tid], params=params: (lambda s: [s, instance_statics(s, params)])(t.bind(**params)))
            BINDINGS.setdefault(sub.id, set()).update(unit)
            tpls.append((sub.id, sub, sts))
    hit = _SIDECARS[p] = (dict(sorted(out.items())), tpls)
    return hit


__all__ = ["BY_CALLS", "BY_CALLS_ROOTS", "CATALOGS", "MAX_GATES", "PINNED_KINDS", "REACHED", "REGISTERED", "ROOT_ERRORS",
           "ROOT_FUNCTIONS", "TABLE_PROBES", "TINY", "Undecided", "binding_files", "binding_roots", "boolean_call_candidates",
           "boolean_call_families", "boolean_family", "boolean_links", "call_families", "catalog", "catalog_units",
           "certified_boolean_calls", "declare_by_calls", "definition", "family", "instance_statics", "is_by_calls", "is_call",
           "kind_of", "load_registries", "pinned", "pouw_bindings", "reachable", "root_functions", "select", "sidecar_dropped", "sidecar_targets", "subcircuit", "suite",
           "table_probe", "template_params", "templates", "word_call_families"]
