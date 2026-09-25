"""The vLLM integration's copies of the boundary and liveness analyses against `verity.ir`'s, on the same inputs.

Every test of `test_ir_boundary.py` runs again here with `B` a differential stand-in: each boundary call runs both
copies and asserts equal results (or the same exception) before the test sees core's.  The same comparison covers the
integration's library Programs, and liveness is compared on every specialization of the same Programs.  Skipped
without `verity_vllm`; removed with the integration copies."""
from __future__ import annotations

import dataclasses
import importlib.util
import pathlib
import types

import pytest

from verity.ir import boundary as CB, liveness as CL
from verity.ir.defs import PrimitiveDefinition, SpecializedDefinition, bind
from verity.ir.parts import Unsupported
from verity.ir.program import Program

IB = pytest.importorskip("verity_vllm.query.boundary")
IL = pytest.importorskip("verity_vllm.program.frontend.liveness")

_OPAQUE = object()


def canon(x):
    if x is None or isinstance(x, (bool, int, float, str, Unsupported)):
        return x
    if isinstance(x, (frozenset, set)):
        return frozenset(x)
    if isinstance(x, (list, tuple)):
        return tuple(canon(y) for y in x)
    if isinstance(x, dict):
        return {k: canon(v) for k, v in x.items()}
    if type(x).__name__ == "ConsumerIndex":
        return ("ConsumerIndex", canon(x.used), x.used_exact, canon(x.used_exact_params), x.incomplete)
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        return (type(x).__name__,) + tuple(canon(getattr(x, f.name)) for f in dataclasses.fields(x))
    return _OPAQUE


def differential(core, other, name: str):
    def run(*args, **kw):
        try:
            rb = ("ok", other(*args, **kw))
        except Exception as e:  # noqa: BLE001
            rb = ("raise", type(e).__name__, str(e))
        try:
            ra = core(*args, **kw)
        except Exception as e:
            assert rb == ("raise", type(e).__name__, str(e)), (name, rb, e)
            raise
        assert rb[0] == "ok", (name, rb)
        if rb[1] is not ra:
            ca = canon(ra)
            assert ca is _OPAQUE or canon(rb[1]) == ca, (name, ca, canon(rb[1]))
        return ra
    return run


class Differential:
    """A module stand-in: function attributes run both modules, classes and constants are core's, setting sets both."""

    def __init__(self, core, other) -> None:
        object.__setattr__(self, "_pair", (core, other))

    def __getattr__(self, name: str):
        core, other = self._pair
        a = getattr(core, name)
        if isinstance(a, (type, types.ModuleType)) or not callable(a):
            return a
        return differential(a, getattr(other, name), name)

    def __setattr__(self, name: str, value) -> None:
        for m in self._pair:
            setattr(m, name, value)


def _load(stem: str):
    spec = importlib.util.spec_from_file_location(f"_differential_{stem}", pathlib.Path(__file__).with_name(f"{stem}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_bmod = _load("test_ir_boundary")
_bmod.B = Differential(CB, IB)
_bmod.time = types.SimpleNamespace(perf_counter=lambda: 0.0)  # both copies run: the cost bounds are checked in core's run
for _k, _v in vars(_bmod).items():
    if _k.startswith("test_") or _k == "gemm":
        globals()[_k] = _v


def specs_of(P: Program) -> list[SpecializedDefinition]:
    seen: dict[int, SpecializedDefinition] = {}
    todo = [P.root]
    while todo:
        sd = todo.pop()
        if id(sd) in seen:
            continue
        seen[id(sd)] = sd
        todo += [n.fn for n in sd.body.nodes if isinstance(n.fn, SpecializedDefinition)]
    return list(seen.values())


def library_parts(P: Program) -> list:
    """Every node / member of the root, of one activation of each composite there, and of one level below."""
    sc = P.circuit.root_scope
    parts = _bmod.body_parts(sc)
    seen: set[str] = set()
    for j, node in enumerate(sc.spec.body.nodes):
        if isinstance(node.fn, PrimitiveDefinition) or node.fn.id in seen:
            continue
        seen.add(node.fn.id)
        sub = sc.child(j, 0)
        parts += _bmod.body_parts(sub)
        for jj, nn in enumerate(sub.spec.body.nodes):
            if isinstance(nn.fn, PrimitiveDefinition) or nn.fn.id in seen:
                continue
            seen.add(nn.fn.id)
            parts += _bmod.body_parts(sub.child(jj, 0))
    return parts


def library_programs():
    b1 = pytest.importorskip("verity_vllm.program.registry.b1")
    from verity_vllm.program.registry.quarantine.ov_moe import moe as Q
    from verity_vllm.program.registry.quarantine.ov_sampling import gumbel as G
    tiny = {"model": "tiny", "revision": None, "H": 32, "NL": 2, "NH": 2, "KVH": 1, "D": 16, "I": 32, "V": 64,
            "QKV": 64, "GU": 64, "EPS": 1e-5, "ROPE_THETA": 1e4, "MAX_POS": 16, "BIAS": True,
            "tie_word_embeddings": True, "profile": {}}
    yield "Gemm", Program(bind(b1.Gemm, K=32, N=6))
    yield "RMSNormTriton", Program(bind(b1.RMSNormTriton, N=32, EPS=1e-5))
    yield "SiluMul", Program(bind(b1.SiluMul, I=8))
    yield "TokenSelect", Program(bind(b1.TokenSelect, V=8))
    yield "AttentionHeadV3", Program(bind(b1.AttentionHeadV3, T=40, D=16, BN=16))
    yield "AttentionV3", Program(bind(b1.AttentionV3, T=8, NH=2, KVH=1, D=16, BN=16))
    yield "Embedding", Program(bind(b1.Embedding, V=16, H=8))
    yield "MoeRouterTopK", Program(bind(Q.MoeRouterTopK, E=8, TOPK=2, VPT=8))
    yield "MoeExpertGemm", Program(bind(Q.MoeExpertGemm, E=4, K=16, N=3))
    yield "GumbelArgmaxBf16", Program(bind(G.GumbelArgmaxBf16, V=8, SEED=12345, POS=3))
    yield "ServeV4 tiny", Program(bind(b1.ServeV4, C=tiny, LP=3, STEPS=2, BN=16))


def test_boundary_copies_agree_on_library_programs():
    B = _bmod.B
    for label, P in library_programs():
        oracle = _bmod.Oracle(P)
        _bmod.run_family(label, oracle, library_parts(P))
        _bmod.check_widths(P, oracle)
        for sd in specs_of(P):
            B.consumer_index(sd)
            try:
                B.used_param_intervals(sd)
            except CB.BoundaryLimit:
                pass


def compare_liveness(label: str, P: Program) -> None:
    for transitive in (False, True):
        a, b = CL.dead_gates(P, limit=10 ** 9, transitive=transitive), IL.dead_gates(P, limit=10 ** 9, transitive=transitive)
        assert a.to_json() == b.to_json(), (label, transitive)
        ca, cb = CL.Liveness(transitive), IL.Liveness(transitive)
        for sd in specs_of(P):
            n = sd.body.ret_type.leaves
            live = ((0, n),) if n else ()
            assert canon(ca.analyse(sd, live))[1:] == canon(cb.analyse(sd, live))[1:], (label, sd.id, transitive)


def test_liveness_copies_agree():
    for label, P in _load("test_ir_liveness").programs():
        compare_liveness(label, P)
    for label, P in library_programs():
        compare_liveness(label, P)
