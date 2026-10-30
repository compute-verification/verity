"""Small-statics Program fixtures and explicit-oracle helpers for the query track.

Shared by the boundary / query-API / partition test suites (import as `tests.program.fixtures_query`).  Every
constructor returns a `Program` small enough for the explicit oracle (`layout.GateSelection.in_` / `.out` on the expanded
circuit); `FIXTURES` maps a short name to a memoised constructor so suites can parametrise over every family.

Families (the query design's §5 table + the counterexample shapes of the query-interface freeze):

  registry (verity/ml/definitions/registry/b1.py and the quarantines)
    gemm                 Gemm_v1{K=32,N=6}                         members of the GemmCoordinate batch; DotBf16 chain inside
    rmsnorm_triton       RMSNormTriton_v1{N=32}                    reduction tree + normalize path (merged per output)
    rmsnorm_fused        RMSNormFusedCuda_v2{N=32}                 tuple return (normed, res')
    silu_mul             SiluMul_v1{I=8}                           one primitive batch
    bias_add             BiasAdd_v1{N=8}
    rope_head            RoPEHead_v1{D=8}                          two batches concatenated in the return
    attention_head       AttentionHead_v3{T=40,D=16,BN=16}         DEAD PARAMETER: Const32[0] passed to the FIRST block's unread state
    attention_head_t72   AttentionHead_v3{T=72,D=16,BN=16}         same kinds, more blocks (population not keyed by T)
    attention            Attention_v3{T=40,NH=2,KVH=1,D=16,BN=16}  batch of heads over strided KV views
    attn_block_first     AttnBlock_v3{D=16,NVIS=16,FIRST=True,BN=16}  672-bit interface
    attn_block_rest      AttnBlock_v3{D=16,NVIS=16,FIRST=False,BN=16}
    moe_router           MoeRouterTopK_v1{E=8,TOPK=2,VPT=8}        scans with a 96-bit carry, NvExpf calls
    moe_expert_gemm      MoeExpertGemm_v1{E=4,K=16,N=3}            gather row + dot per coordinate
    moe_sum              MoeSum_v1{TOPK=2,H=4}
    gumbel               GumbelArgmaxBf16_v1{V=8}                  64-BIT GATES (two Const64), scan with shared key
    token_select         TokenSelect_v1{V=8}                       scan with an 80-bit carry
    embedding            Embedding_v1{V=8,H=4}                     wide-fan-in gather gates
  synthetic (built here with `CompositeDefinition(..., register=False)`; no registry footprint)
    tiny_io              Inputs, a constant gate, a dead Input, two designated outputs (one also consumed inside)
    fan_out              one gate consumed by three gates in five operand slots -> out counts it once
    dead_return          a call returning (a, b) whose parent reads only a: exact w_out 32, interface bound 64
    reexport             a call whose tuple return is re-returned by the root: both leaves designated
    dead_param           a callee that never reads its second parameter; the constant feeding it has out = {}
    scan_carry           scan returning (carry_n, ys): every iteration's carry AND y escape
    scan_carry_only      scan whose ys are dead: only the carry escapes each iteration

Oracle helpers (`oracle_out`, `oracle_in`, `oracle_w_out`) accept a `Part`, anything with `.intervals()` (a GateSet) or an
iterable of root gate indices, and evaluate `GateSelection` on the expanded circuit -- the definitional answer the compact
modules are tested against.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Callable, Iterable

from verity.ml.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind
from verity.ml.circuits.layout import Scope, GateSelection
from verity.ml.circuits.parts import Item, Part, n_members
from verity.ml.circuits.program import Program
from verity.ml.circuits.refs import array_of, tuple_of
from verity.ml.definitions.registry import b1
from verity.ml.definitions.registry import prims as P
from verity.ml.definitions.registry.prims import BF16, F32, I32
from verity.ml.circuits.types import Array, Tuple

# ---------------------------------------------------------------------------------------------------------
# registry programs (small statics)
# ---------------------------------------------------------------------------------------------------------


def gemm_program(K: int = 32, N: int = 6) -> Program:
    return Program(bind(b1.Gemm, K=K, N=N))


def rmsnorm_triton_program(N: int = 32, EPS: float = 1e-5) -> Program:
    return Program(bind(b1.RMSNormTriton, N=N, EPS=EPS))


def rmsnorm_fused_program(N: int = 32, EPS: float = 1e-5) -> Program:
    return Program(bind(b1.RMSNormFusedCuda, N=N, EPS=EPS))


def silu_mul_program(I: int = 8) -> Program:
    return Program(bind(b1.SiluMul, I=I))


def bias_add_program(N: int = 8) -> Program:
    return Program(bind(b1.BiasAdd, N=N))


def rope_head_program(D: int = 8) -> Program:
    return Program(bind(b1.RoPEHead, D=D))


def attention_head_program(T: int = 40, D: int = 16, BN: int = 16) -> Program:
    """`AttentionHead_v3{T,D,BN}`: ceil(T/BN) `AttnBlock_v3` calls visited last -> first.  The FIRST block receives
    `Const32[0]` as m_prev / l_prev / o_prev and never reads them (the dead-parameter case of design doc §5)."""
    return Program(bind(b1.AttentionHeadV3, T=T, D=D, BN=BN))


def attention_program(T: int = 40, NH: int = 2, KVH: int = 1, D: int = 16, BN: int = 16) -> Program:
    return Program(bind(b1.AttentionV3, T=T, NH=NH, KVH=KVH, D=D, BN=BN))


def attn_block_program(FIRST: bool, D: int = 16, BN: int = 16, NVIS: int | None = None) -> Program:
    return Program(bind(b1.AttnBlockV3, D=D, NVIS=BN if NVIS is None else NVIS, FIRST=FIRST, BN=BN))


def attention_head_v1_program(T: int = 72, D: int = 16) -> Program:
    """`AttentionHead_v1{T,D}`: the legacy chain of every record without a geometry (`Serve_v1`), kBlockN = 64 as the module constant:
    T=72 -> two blocks (the FIRST one with 8 visible columns, then a full 64-column block with FOUR PV k-step batches)."""
    return Program(bind(b1.AttentionHead, T=T, D=D))


def attention_head_v2_program(T: int = 40, D: int = 16, BN: int = 16) -> Program:
    """`AttentionHead_v2{T,D,BN,DOT,INV}` (CF1 factored chain): the same body with the k16 step and the epilogue inverse as statics."""
    return Program(bind(b1.AttentionHeadV2, T=T, D=D, BN=BN, DOT=P.AmpereBF16TcDot16, INV=P.Fa2InvSum))


def gemm_v2_program(K: int = 32, N: int = 6) -> Program:
    return Program(bind(b1.GemmV2, K=K, N=N, DOT=P.AmpereBF16TcDot16))


def layer_norm_aten_program(N: int = 32, EPS: float = 1e-5) -> Program:
    from verity.ml.definitions.registry import layer_norm as LN
    return Program(bind(LN.LayerNormAten, N=N, EPS=EPS))


def moe_router_program(E: int = 8, TOPK: int = 2, VPT: int = 8) -> Program:
    from verity_vllm.program.quarantine.ov_moe import moe as Q
    return Program(bind(Q.MoeRouterTopK, E=E, TOPK=TOPK, VPT=VPT))


def moe_expert_gemm_program(E: int = 4, K: int = 16, N: int = 3) -> Program:
    from verity_vllm.program.quarantine.ov_moe import moe as Q
    return Program(bind(Q.MoeExpertGemm, E=E, K=K, N=N))


def moe_sum_program(TOPK: int = 2, H: int = 4) -> Program:
    from verity_vllm.program.quarantine.ov_moe import moe as Q
    return Program(bind(Q.MoeSum, TOPK=TOPK, H=H))


def gumbel_program(V: int = 8, SEED: int = 12345, POS: int = 3) -> Program:
    """`GumbelArgmaxBf16_v1{V,SEED,POS}`: the only supported family with gates wider than 32 bits (two `Const64`)."""
    from verity.ml.definitions.registry import sampling as G
    return Program(bind(G.GumbelArgmaxBf16, V=V, SEED=SEED, POS=POS))


def token_select_program(V: int = 8) -> Program:
    return Program(bind(b1.TokenSelect, V=V))


def embedding_program(V: int = 8, H: int = 4) -> Program:
    return Program(bind(b1.Embedding, V=V, H=H))


# ---------------------------------------------------------------------------------------------------------
# synthetic bodies (unregistered: importing this module twice must not conflict in REGISTRY)
# ---------------------------------------------------------------------------------------------------------


def _defn(name: str, sig, body) -> CompositeDefinition:
    return CompositeDefinition(f"QFix{name}", 1, (), lambda S: sig, body, doc=f"query-track synthetic fixture {name}",
                               conformance="model-untested", register=False)


def _tiny_io_body(B, S, x):
    s = B.call(P.F32Add, x[0], x[1])            # node 1: consumed by r AND designated output
    c = B.call(b1.ZERO32)                        # node 2: constant gate (a query that forgets it has a gap)
    t = B.call(P.F32Add, x[2], c)                # node 3
    r = B.call(P.F32Add, s, t)                   # node 4: designated output
    return tuple_of(r, s)                        # x[3] is a dead Input (never read): out({x3}) = {}


TinyIO = _defn("TinyIO", ((("x", Array(4, F32)),), Tuple(F32, F32)), _tiny_io_body)


def _fan_out_body(B, S, x):
    y = B.call(P.F32Mul, x, x)                   # node 1: consumed by nodes 2, 3, 4 (five operand slots)
    z0 = B.call(P.F32Add, y, x)
    z1 = B.call(P.F32Mul, y, y)
    z2 = B.call(P.F32Fma, y, x, y)
    return array_of([z0, z1, z2])


FanOut = _defn("FanOut", ((("x", F32),), Array(3, F32)), _fan_out_body)


def _pair_body(B, S, x):
    a = B.call(P.F32Add, x, x)
    b = B.call(P.F32Mul, x, x)
    return tuple_of(a, b)


Pair = _defn("Pair", ((("x", F32),), Tuple(F32, F32)), _pair_body)


def _dead_return_body(B, S, x):
    p = B.call(Pair, x)                          # node 1 (call): returns (a, b); only a is read below
    return B.call(P.F32Add, p[0], x)             # node 2


DeadReturn = _defn("DeadReturn", ((("x", F32),), F32), _dead_return_body)


def _reexport_body(B, S, x):
    return B.call(Pair, x)                       # node 1: both returned leaves are designated outputs


Reexport = _defn("Reexport", ((("x", F32),), Tuple(F32, F32)), _reexport_body)


def _ignore_b_body(B, S, a, b):
    return B.call(P.F32Mul, a, a)                # `b` is declared but never read


IgnoreB = _defn("IgnoreB", ((("a", F32), ("b", F32)), F32), _ignore_b_body)


def _dead_param_body(B, S, x, y):
    c = B.call(b1.ZERO32)                        # node 2: its only consumer is a dead parameter -> out({c}) = {}
    return B.call(IgnoreB, x, c)                 # node 3 (call): in = {x}, not {x, c}; Input y is dead


DeadParam = _defn("DeadParam", ((("x", F32), ("y", F32)), F32), _dead_param_body)


def _scan_step_body(B, S, carry, x):
    c2 = B.call(P.F32Add, carry, x)
    y = B.call(P.F32Mul, carry, x)
    return tuple_of(c2, y)


ScanStep = _defn("ScanStep", ((("carry", F32), ("x", F32)), Tuple(F32, F32)), _scan_step_body)

SCAN_N = 5


def _scan_carry_body(B, S, init, xs):
    return B.scan(ScanStep, init, xs=(xs,))      # node 2 (scan): (carry_n, ys) all returned


ScanCarry = _defn("ScanCarry", ((("init", F32), ("xs", Array(SCAN_N, F32))), Tuple(F32, Array(SCAN_N, F32))), _scan_carry_body)


def _scan_carry_only_body(B, S, init, xs):
    return B.scan(ScanStep, init, xs=(xs,))[0]   # ys are dead: only the carry crosses iteration boundaries


ScanCarryOnly = _defn("ScanCarryOnly", ((("init", F32), ("xs", Array(SCAN_N, F32))), F32), _scan_carry_only_body)


def tiny_io_program() -> Program:
    return Program(TinyIO)


def fan_out_program() -> Program:
    return Program(FanOut)


def dead_return_program() -> Program:
    return Program(DeadReturn)


def reexport_program() -> Program:
    return Program(Reexport)


def dead_param_program() -> Program:
    return Program(DeadParam)


def scan_carry_program() -> Program:
    return Program(ScanCarry)


def scan_carry_only_program() -> Program:
    return Program(ScanCarryOnly)


# ---------------------------------------------------------------------------------------------------------
# the fixture table
# ---------------------------------------------------------------------------------------------------------

_CONSTRUCTORS: dict[str, Callable[[], Program]] = {
    "gemm": gemm_program,
    "rmsnorm_triton": rmsnorm_triton_program,
    "rmsnorm_fused": rmsnorm_fused_program,
    "silu_mul": silu_mul_program,
    "bias_add": bias_add_program,
    "rope_head": rope_head_program,
    "attention_head": attention_head_program,
    "attention_head_t72": lambda: attention_head_program(T=72),
    "attention": attention_program,
    "attn_block_first": lambda: attn_block_program(FIRST=True),
    "attn_block_rest": lambda: attn_block_program(FIRST=False),
    "attention_head_v1": attention_head_v1_program,
    "attention_head_v2": attention_head_v2_program,
    "gemm_v2": gemm_v2_program,
    "layer_norm_aten": layer_norm_aten_program,
    "moe_router": moe_router_program,
    "moe_expert_gemm": moe_expert_gemm_program,
    "moe_sum": moe_sum_program,
    "gumbel": gumbel_program,
    "token_select": token_select_program,
    "embedding": embedding_program,
    "tiny_io": tiny_io_program,
    "fan_out": fan_out_program,
    "dead_return": dead_return_program,
    "reexport": reexport_program,
    "dead_param": dead_param_program,
    "scan_carry": scan_carry_program,
    "scan_carry_only": scan_carry_only_program,
}

#: fixture names (stable order; parametrise suites over this)
FIXTURE_NAMES: tuple[str, ...] = tuple(_CONSTRUCTORS)
#: fixtures whose Program contains a primitive gate wider than 32 bits (design doc §7)
WIDE_GATE_FIXTURES: frozenset[str] = frozenset({"gumbel"})
#: fixtures with a `scan` node (carry crosses iteration boundaries)
SCAN_FIXTURES: frozenset[str] = frozenset({"moe_router", "gumbel", "token_select", "scan_carry", "scan_carry_only"})
#: the registry programs of the supported vLLM families the canonical VU query must partition
VLLM_FAMILY_FIXTURES: tuple[str, ...] = ("gemm", "gemm_v2", "rmsnorm_triton", "rmsnorm_fused", "layer_norm_aten", "silu_mul", "bias_add",
                                         "rope_head", "attention_head", "attention_head_t72", "attention_head_v1", "attention_head_v2",
                                         "attention", "moe_router", "moe_expert_gemm", "moe_sum", "token_select", "embedding")
#: the attention-head fixtures whose canonical VU kinds must coincide (same D / chain version, different T)
ATTENTION_T_PAIRS: tuple[tuple[str, str], ...] = (("attention_head", "attention_head_t72"),)


@lru_cache(maxsize=None)
def fixture(name: str) -> Program:
    """The memoised fixture Program `name` (see the module docstring)."""
    return _CONSTRUCTORS[name]()


#: alias table for callers that want the mapping itself
FIXTURES: dict[str, Callable[[], Program]] = {n: (lambda n=n: fixture(n)) for n in FIXTURE_NAMES}


def _wrap_sig(S):
    return S.FN.signature()


def _wrap_body(B, S, *args):
    return B.call(S.FN, *args)


#: `QFixWrap_v1{FN}`: a one-call root around a complete specialization.  `Program(spec)` INLINES the program function's
#: body into the root (program.py), so `//Gemm_v1{*}` matches nothing in `gemm_program()`; the wrapped variant has one
#: `call` node to `Gemm_v1{K=32,N=6}` in its root body, which is how the family appears inside a real Serve program.
Wrap = CompositeDefinition("QFixWrap", 1, ("FN",), _wrap_sig, _wrap_body,
                           doc="query-track fixture wrapper: root = Inputs + one call of FN", conformance="model-untested", register=False)


def wrap(spec) -> Program:
    """A Program whose root body is the Input batches plus ONE call node of `spec` (a complete SpecializedDefinition)."""
    return Program(bind(Wrap, FN=spec))


@lru_cache(maxsize=None)
def wrapped_fixture(name: str) -> Program:
    """`fixture(name)`'s specialization behind a one-call root (see `Wrap`)."""
    return wrap(fixture(name).fn)


# ---------------------------------------------------------------------------------------------------------
# structural helpers (Parts, scopes, node lookup)
# ---------------------------------------------------------------------------------------------------------


def root_scope(program: Program) -> Scope:
    return program.circuit.root_scope


def is_input_node(node) -> bool:
    return getattr(node.fn, "family", None) == "Input"


def is_const_node(node) -> bool:
    return getattr(node.fn, "family", None) == "Const"


def item_part(scope: Scope, j: int, m: int = 0) -> Part:
    """The Part of one item (node j, member m) of the body `scope` activates."""
    return Part(scope, frozenset({(j, m)}))


def node_part(scope: Scope, j: int) -> Part:
    """The Part of a whole node (every member of a batch / every iteration of a scan)."""
    return Part(scope, frozenset((j, m) for m in range(n_members(scope.spec.body.nodes[j]))))


def items_part(scope: Scope, items: Iterable[Item]) -> Part:
    return Part(scope, frozenset(items))


def body_parts(scope: Scope, skip_input: bool = False) -> list[Part]:
    """Every item of the body as its own Part (the finest hierarchy parts), body order."""
    out: list[Part] = []
    for j, node in enumerate(scope.spec.body.nodes):
        if skip_input and is_input_node(node):
            continue
        for m in range(n_members(node)):
            out.append(item_part(scope, j, m))
    return out


def find_nodes(scope: Scope, *, form: str | None = None, callee: str | None = None, primitive: bool | None = None) -> list[int]:
    """Node indices of the body whose form / callee id (substring of `fn.id`, or exact `fn.name`) / primitiveness match."""
    out = []
    for j, node in enumerate(scope.spec.body.nodes):
        if form is not None and node.form != form:
            continue
        if callee is not None and not (callee in node.fn.id or callee == getattr(node.fn, "name", None)):
            continue
        if primitive is not None and isinstance(node.fn, PrimitiveDefinition) != primitive:
            continue
        out.append(j)
    return out


def child_scope(scope: Scope, j: int, m: int = 0) -> Scope:
    """The activation Scope of composite item (j, m) of `scope` (the callee's body)."""
    node = scope.spec.body.nodes[j]
    if isinstance(node.fn, PrimitiveDefinition):
        raise TypeError(f"node {j} ({node.fn.id}) is a primitive: it has no activation scope")
    return scope.child(j, m)


def first_composite_child(scope: Scope, callee: str, m: int = 0) -> Scope:
    js = find_nodes(scope, callee=callee, primitive=False)
    if not js:
        raise KeyError(f"{scope.spec.id} has no composite node calling {callee!r}")
    return child_scope(scope, js[0], m)


# ---------------------------------------------------------------------------------------------------------
# explicit oracle (GateSelection on the expanded circuit)
# ---------------------------------------------------------------------------------------------------------


def gates_of(x) -> list[int]:
    """Root gate indices of a Part, of anything with `.intervals()` (a GateSet), or of an iterable of ints."""
    if hasattr(x, "intervals"):
        out: list[int] = []
        for lo, hi in x.intervals():
            out.extend(range(lo, hi))
        return out
    return [int(g) for g in x]


def part_gates(part: Part) -> list[int]:
    return gates_of(part)


def gate_width(program: Program, g: int) -> int:
    """w(g): the producing primitive's return width."""
    return program.circuit.gate(g).prim.ret.w


def root_outputs(program: Program) -> frozenset[int]:
    return frozenset(program.circuit.root_outputs())


def oracle_out(program: Program, gates) -> frozenset[int]:
    """EXACT out(S) = {u in S : designated output or consumed by some v not in S}, by explicit consumer scan."""
    gs = gates_of(gates)
    if not gs:
        return frozenset()
    return frozenset(GateSelection(program.circuit, gs).out)


def oracle_in(program: Program, gates) -> frozenset[int]:
    """EXACT in(S) = {u not in S : some v in S reads u}."""
    gs = gates_of(gates)
    if not gs:
        return frozenset()
    return frozenset(GateSelection(program.circuit, gs).in_)


def oracle_w_out(program: Program, gates) -> int:
    return sum(gate_width(program, g) for g in oracle_out(program, gates))


def oracle_w_in(program: Program, gates) -> int:
    return sum(gate_width(program, g) for g in oracle_in(program, gates))


def interface_w_out_bound(part: Part) -> int:
    """The interface-based (declared) bound a naive implementation would charge for a Part: every returned leaf of
    every item, plus every leaf consumed by an item outside the part -- i.e. `sum(w)` over every leaf the items produce.
    Only used by tests to show where the exact answer is strictly smaller (dead return / dead parameter)."""
    body = part.scope.spec.body
    total = 0
    for j, m in part.items:
        node = body.nodes[j]
        fn = node.fn
        if isinstance(fn, PrimitiveDefinition):
            total += fn.ret.w
        else:
            total += sum(_type_widths(fn.body.ret_type))
    return total


def _type_widths(t) -> list[int]:
    from verity.ml.circuits.types import Record, Value
    if isinstance(t, Value):
        return [t.w]
    if isinstance(t, Array):
        return _type_widths(t.elem) * t.n
    if isinstance(t, Tuple):
        out: list[int] = []
        for f in t.fields:
            out.extend(_type_widths(f))
        return out
    if isinstance(t, Record):
        out = []
        for _, f in t.fields:
            out.extend(_type_widths(f))
        return out
    raise TypeError(t)


def wide_gates(program: Program, w_max: int = 32) -> list[tuple[int, str, int]]:
    """(gate index, primitive id, width) of every gate wider than `w_max`, by structural walk of the root body's
    primitive nodes and nested bodies (no expanded-circuit enumeration)."""
    out: list[tuple[int, str, int]] = []

    def walk(sc: Scope) -> None:
        body = sc.spec.body
        for j, node in enumerate(body.nodes):
            fn = node.fn
            if isinstance(fn, PrimitiveDefinition):
                if fn.ret.w > w_max:
                    for m in range(n_members(node)):
                        out.append((sc.offset + body.offsets[j] + m, fn.id, fn.ret.w))
            else:
                for m in range(n_members(node)):
                    walk(sc.child(j, m))

    walk(root_scope(program))
    return out


class ListFamily:
    """A hand-made `Family` (freeze doc: ordered, lazy, results may overlap or omit gates) over an explicit list of
    GateSets.  `tiling()` is None, so partition validation must take the general interval-sweep path -- this is how the
    counterexample suite builds overlapping / gapped families the structural constructors cannot express."""

    def __init__(self, sets: Iterable) -> None:
        self._sets = list(sets)

    def count(self) -> int:
        return len(self._sets)

    def by_index(self, i: int):
        return self._sets[i]

    def index_of(self, s) -> int | None:
        for i, t in enumerate(self._sets):
            if t == s:
                return i
        return None

    def __iter__(self):
        return iter(self._sets)

    def __len__(self) -> int:
        return len(self._sets)

    def tiling(self):
        return None


def gateset_kind(gs) -> tuple:
    """A structural 'kind' of a GateSet / Part for population tests: per part, the enclosing definition's base id (statics
    dropped -- so the enclosing head's `T` does not key the kind) and the set of (callee id, form) of its items.  Two VUs
    of the same kind are the same computation up to position."""
    parts = gs.parts if hasattr(gs, "parts") else (gs,)
    kinds = []
    for part in parts:
        body = part.scope.spec.body
        kinds.append((spec_base_id(part.scope.spec), tuple(sorted({(body.nodes[j].fn.id, body.nodes[j].form) for j, _ in part.items}))))
    return tuple(sorted(kinds))


def spec_base_id(spec) -> str:
    """`Name_vN` of a specialization with its statics dropped; for a Program root (`Root[Fn{...}]`) the program function's
    base id, so a root body keyed by `T` in its statics still has a `T`-free kind."""
    pf = getattr(spec, "program_fn", None)
    if pf is not None:
        return f"Root[{pf.base_id}]"
    return getattr(spec, "base_id", spec.id)


__all__ = [
    "FIXTURES", "FIXTURE_NAMES", "WIDE_GATE_FIXTURES", "SCAN_FIXTURES", "VLLM_FAMILY_FIXTURES", "fixture", "Wrap", "wrap", "wrapped_fixture",
    "gemm_program", "rmsnorm_triton_program", "rmsnorm_fused_program", "silu_mul_program", "bias_add_program", "rope_head_program",
    "attention_head_program", "attention_program", "attn_block_program", "moe_router_program", "moe_expert_gemm_program",
    "moe_sum_program", "gumbel_program", "token_select_program", "embedding_program",
    "tiny_io_program", "fan_out_program", "dead_return_program", "reexport_program", "dead_param_program", "scan_carry_program",
    "scan_carry_only_program", "TinyIO", "FanOut", "Pair", "DeadReturn", "Reexport", "IgnoreB", "DeadParam", "ScanStep", "ScanCarry",
    "ScanCarryOnly", "SCAN_N",
    "root_scope", "is_input_node", "is_const_node", "item_part", "node_part", "items_part", "body_parts", "find_nodes", "child_scope",
    "first_composite_child", "gates_of", "part_gates", "gate_width", "root_outputs", "oracle_out", "oracle_in", "oracle_w_out",
    "oracle_w_in", "interface_w_out_bound", "wide_gates", "gateset_kind", "spec_base_id", "ListFamily",
]
