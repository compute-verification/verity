"""Hopper (sm_90) bindings of the B1 typed serving program, profile `b1-hopper`.

On an H100 the eager batch-invariant profile executes two functions that differ bit-for-bit from the Ampere/Ada
ones `b1.py`'s B1 composites are bound to:

* the batch-invariant Triton GEMM compiles `tl.dot` to `wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16`
  (SASS `HGMMA.64x128x16.F32.BF16`) instead of `mma.sync.m16n8k16`: one accumulator coordinate takes all 16
  products *and* the accumulator in one grouped sum through a 26-bit adder with exponent floor -133 (Ampere: two
  groups of 8, 25 bits, -132).  Pinned by the campaign under `gpu/tc-hopper/` as the registered total function
  `veritor.tensor-core.hopper_bf16_wgmma_k16@1` (`veritor.core.silicon_total.HOPPER_BF16_WGMMA_K16`); the
  primitive here is `HopperBF16WgmmaDot16`;
* attention runs FlashAttention-3 (`flash::FlashAttnFwdSm90`, hdim 128, causal, `--use_fast_math`): kBlockN = 128
  keys per online-softmax block (FA2: 64), the same per-row recurrence otherwise (`hopper/softmax.h` at
  vllm-flash-attn 506341a1: `scores_scale = exp2((m_prev - m_cur) * scale_log2)`, `row_sum = row_sum * scale + p...`
  contracted by nvcc into one FFMA, `p = exp2(s * scale_log2 - m * scale_log2)`, `O = O * scale` then the P·V
  `wgmma` chain), and the epilogue `inv = (sum == 0 || sum != sum) ? 0 : 1 / sum` (FA2: `? 1`) — `Fa3InvSum`.

**Same structure, different bindings.**  Every composite below is `b1.py`'s definition *instantiated in a Hopper
binding environment*: the same `signature` and `body` code objects (`_reinstantiate` re-creates the functions
over a copy of `b1`'s module globals in which the bound names resolve to the Hopper functions), registered under
new ids because a registered id denotes one function.  Nothing in `b1.py` changes.  The names that are rebound:

    P.AmpereBF16TcDot16 -> HopperBF16WgmmaDot16      (the k16 tensor-core step)
    P.Fa2InvSum         -> Fa3InvSum                 (the softmax epilogue reciprocal)
    BN                  -> 128                       (kBlockN — a module constant in b1.py, i.e. an implicit static)
    DotBf16 / GemmCoordinate / Gemm -> *Hopper
    AttnBlock / AttentionHead / Attention -> *FA3
    LayerPre / LayerPost / Final / Serve -> *Hopper   (re-instantiated only because their bodies name their callees)

Everything else (RMSNorm, RoPE, BiasAdd, SiluMul, Embedding, TokenSelect, the constants, every scalar primitive)
is `b1.py`'s registered function, unchanged.  The rebound names are a static-config matter: they should be
function-valued statics and kBlockN a static, which is how `SERVE_HOPPER` below binds them.
"""

from __future__ import annotations

import types

from verity.ml.circuits.defs import CompositeDefinition, FunctionDefinition, PrimitiveDefinition, SpecializedDefinition, primitive
from verity.ml.circuits.ops.prims import HopperBF16WgmmaDot16
from verity.ml.circuits.silicon.total import HOPPER_BF16_WGMMA_K16, tc_dot_total
from verity.ml.definitions.registry import b1
from verity.ml.definitions.registry import prims as P
from verity.ml.definitions.registry.prims import F32, f32_is_nan

# ---------------------------------------------------------------------------------------------------------
# primitives
# ---------------------------------------------------------------------------------------------------------


def _hopper_total():
    """The step's definition of record: `verity.ml.circuits.silicon.total` (veritor's `core/silicon_total.py` promoted verbatim, stdlib only,
    Python >= 3.12); the Verity pod runtime is 3.12, so the definition itself evaluates."""
    return HOPPER_BF16_WGMMA_K16, tc_dot_total


# `verity.ml.circuits.silicon.total.HOPPER_BF16_WGMMA_K16` (the measured pipeline): the relation's parameters, stated once in core.
HOPPER_WGMMA_K = HOPPER_BF16_WGMMA_K16.k                       # 16 operand pairs per step (one group: all sixteen products + acc)
HOPPER_WGMMA_WIDTH = HOPPER_BF16_WGMMA_K16.width               # 26-bit adder
HOPPER_WGMMA_ZERO_EXPONENT = HOPPER_BF16_WGMMA_K16.zero_exponent  # exponent floor -133


def hopper_wgmma_dot16_total(acc_bits: int, a, b) -> int:
    """`D = acc + sum_{k<16} a[k]*b[k]` on EVERY encoding as `verity.ml.circuits.silicon.total.tc_dot_total(HOPPER_BF16_WGMMA_K16, ...)`
    defines it (one group of sixteen products + the accumulator, 26-bit adder, floor -133, the NaN / infinity / sticky-overflow /
    +0 / subnormal rules of the total step) on the Hopper pipeline.  tests/program/test_derived_rows.py holds the registered primitive
    to it word for word."""
    return tc_dot_total(HOPPER_BF16_WGMMA_K16, acc_bits, a, b)


@primitive("Fa3InvSum", 1, [("sum", F32)], F32, conformance="exact-model-tested", tables=(("rcp", "library"),))
def Fa3InvSum(s: int) -> int:
    """FA3 `Softmax::finalize` (hopper/softmax.h): `inv_sum = (sum == 0.f || sum != sum) ? 0.f : 1.f / sum` under
    `--use_fast_math` (`1.f / sum` -> `rcp.approx.ftz.f32`, MUFU.RCP), then `scores_scale = inv_sum * 1.f`.  Tested on
    the reachable domain (sum > 0) through every exact `AttentionFA3` leaf of the H100 correspondence; the degenerate
    branch (sum == 0 or NaN -> 0; FA2 gives 1) is read from the source and unreachable on finite operands."""
    if (s & 0x7FFFFFFF) == 0 or f32_is_nan(s):
        return 0
    return P.MufuRcpFtz.evaluate(s)


# ---------------------------------------------------------------------------------------------------------
# the Hopper binding environment: b1's module globals with the bound names rebound
# ---------------------------------------------------------------------------------------------------------


class _BoundPrims:
    """`P` inside the Hopper environment: `prims` with the Hopper-bound names."""

    _over = {"AmpereBF16TcDot16": HopperBF16WgmmaDot16, "Fa2InvSum": Fa3InvSum}

    def __getattr__(self, k: str):
        if k in self._over:
            return self._over[k]
        return getattr(P, k)


KBLOCK_N_FA3 = 128  # tile_size_fwd_sm90(hdim 128, causal) -> kBlockN = 128 for both kBlockM = 128 and the one-warpgroup 64

ENV: dict = dict(vars(b1))  # a *copy*: b1's own globals are never touched
ENV["P"] = _BoundPrims()
ENV["BN"] = KBLOCK_N_FA3
ENV["__name__"] = __name__


def _reinstantiate(fn, env: dict = ENV):
    """The same code object as `fn`, evaluated over `env` (shared: later bindings are visible to earlier bodies)."""
    new = types.FunctionType(fn.__code__, env, fn.__name__, fn.__defaults__, fn.__closure__)
    new.__kwdefaults__ = fn.__kwdefaults__
    new.__doc__ = fn.__doc__
    new.__qualname__ = fn.__qualname__
    return new


PORTED: dict[str, CompositeDefinition] = {}  # b1 name -> Hopper-bound definition


def _port(defn: CompositeDefinition, name: str, doc: str, conformance: str = "exact-model-tested") -> CompositeDefinition:
    """`b1.py`'s `defn` — the same statics, signature code and body code — instantiated in `ENV`, registered as `name@1`."""
    d = CompositeDefinition(name, 1, defn.statics, _reinstantiate(defn.signature), _reinstantiate(defn.body),
                            doc=f"{defn.doc.strip()}\n\n[b1-hopper] {doc}", conformance=conformance)
    d.ported_from = defn  # type: ignore[attr-defined]
    ENV[defn.name] = d
    PORTED[defn.name] = d
    return d


DotBf16Hopper = _port(b1.DotBf16, "DotBf16Hopper",
                      "k16 step = HopperBF16WgmmaDot16 (wgmma m64n128k16); ascending k from a zero accumulator, as the Triton "
                      "matmul_kernel_persistent's BLOCK_K=64 loop issues four k16 wgmma per k-tile in order.",
                      conformance="exact-model-tested")
GemmCoordinateHopper = _port(b1.GemmCoordinate, "GemmCoordinateHopper", "DotBf16Hopper then `cvt.rn.bf16.f32`.",
                             conformance="exact-model-tested")
GemmHopper = _port(b1.Gemm, "GemmHopper", "batch of GemmCoordinateHopper over the N output coordinates.",
                   conformance="exact-model-tested")
AttnBlockFA3 = _port(b1.AttnBlock, "AttnBlockFA3",
                     "FA3 block: kBlockN = 128 keys, S and P·V through HopperBF16WgmmaDot16 (8 ascending k16 steps per 128-key "
                     "block), the same lane ownership ((c % 8) // 2: the wgmma accumulator layout is the mma.sync one), the same "
                     "rescale-then-add order (`rescale_o` before the next P·V wgmma; RescaleOBeforeGemm = false for hdim 128).")
AttentionHeadFA3 = _port(b1.AttentionHead, "AttentionHeadFA3",
                         "FA3 head: ceil(T/128) blocks visited last -> first, epilogue sum = (l0 + l2) + (l1 + l3) (Allreduce<4>: "
                         "xor 2 then xor 1), inv = Fa3InvSum(sum), out = bf16(o * inv).")
AttentionFA3 = _port(b1.Attention, "AttentionFA3",
                     "GQA over AttentionHeadFA3.  Conformance: every attention leaf of the H100 correspondence exact (39 leaves, T = 1..527, "
                     "request 917b9faa, layers 0/1/13/27: results/CONF-b1-hopper-*/correspondence.json) and AttentionHeadFA3 equal word for word "
                     "to the FA3 CPU model (verity.ml.definitions.numerics.fa3_model; tests/poc/test_fa3_model.py).")
LayerPreHopper = _port(b1.LayerPre, "LayerPreHopper", "GemmHopper for the QKV projection; otherwise b1's functions.")
LayerPostHopper = _port(b1.LayerPost, "LayerPostHopper", "GemmHopper for o_proj, gate_up, down; otherwise b1's functions.")
FinalHopper = _port(b1.Final, "FinalHopper", "GemmHopper for the LM head; b1's RMSNormFusedCuda and TokenSelect.",
                    conformance="model-untested")
ENV["_forward_positions"] = _reinstantiate(b1._forward_positions)
ServeHopper = _port(b1.Serve, "ServeHopper", "b1's Serve over the Hopper-bound layer functions.", conformance="model-untested")

HOPPER_COMPOSITES = [DotBf16Hopper, GemmCoordinateHopper, GemmHopper, AttnBlockFA3, AttentionHeadFA3, AttentionFA3,
                     LayerPreHopper, LayerPostHopper, FinalHopper, ServeHopper]
HOPPER_PRIMITIVES = [HopperBF16WgmmaDot16, Fa3InvSum]

# FA3's per-iteration Check_inf chain (`Attention_v4`, `TargetProfile.fa3_construction`) registers with the Hopper bindings, so every process
# that decodes a Hopper Program can decode one built under that construction
from verity.ml.definitions.registry import fa3_check_inf  # noqa: E402, F401
# FA2's (`Attention_v5`, the Hopper k16 step's FA2 on cc 12.0) likewise
from verity.ml.definitions.registry import fa2_check_inf  # noqa: E402, F401
# and FA2's softcapped chain under it (`AttentionSoftcap_v2`)
from verity.ml.definitions.registry import fa2_softcap  # noqa: E402, F401

# --- the Hopper eager profile as a binding record of b1's factored `Serve_v2` -----------
# The environment substitution above (`_BoundPrims._over`, `ENV["BN"]`) written as the statics it always was: the same
# three selections, now in the program's identity instead of the Python module namespace.
GEMM_HOPPER: dict = {"DOT": HopperBF16WgmmaDot16}
ATTENTION_FA3: dict = {"BN": KBLOCK_N_FA3, "DOT": HopperBF16WgmmaDot16, "INV": Fa3InvSum}


def SERVE_HOPPER(C: dict, LP: int, STEPS: int):
    """The Hopper eager program: `Serve_v2` under `GEMM_HOPPER` / `ATTENTION_FA3` / `b1.norm_bundle(C)` — the norm record
    is b1's (the H100 runs the same Triton and CUDA RMSNorm kernels)."""
    return b1.serve_v2(C, LP, STEPS, gemm=GEMM_HOPPER, attention=ATTENTION_FA3)

#: b1 call name -> Hopper call name, for capture mappings written against b1's names (`correspondence.py`).
CALL_NAMES: dict[str, str] = {k: v.name for k, v in PORTED.items()}
#: The full binding table by name (composites and primitives), for `structural_diff`.
NAME_MAP: dict[str, str] = {**CALL_NAMES, **{k: v.name for k, v in _BoundPrims._over.items()}}


def set_conformance(status: dict[str, str]) -> None:
    """Set conformance statuses from evidence (the profile's `expectations.json` / conformance run)."""
    by_name = {d.name: d for d in HOPPER_COMPOSITES + [Fa3InvSum]}  # HopperBF16WgmmaDot16's conformance is core's
    for name, st in status.items():
        by_name[name].conformance = st


# ---------------------------------------------------------------------------------------------------------
# bindings as data, and the structural comparison that backs "same structure, different bindings"
# ---------------------------------------------------------------------------------------------------------


def bindings() -> dict:
    """`FunctionDefinition name -> registered id@version` for every definition `b1.py`'s B1 program uses, with the
    conformance status and whether the binding is b1's (`"ta1"`) or the Hopper one."""
    out: dict[str, dict] = {}
    for d in b1.ALL_COMPOSITES:
        h = PORTED.get(d.name, d)
        out[d.name] = {"binding": f"{h.name}@{h.version}", "kind": "composite", "conformance": h.conformance,
                       "changed": h is not d, "ta1": f"{d.name}@{d.version}"}
    over = _BoundPrims._over
    for p in P.ALL_SCALAR:
        h = over.get(p.name, p)
        out[p.name] = {"binding": f"{h.name}@{h.version}", "kind": "primitive", "conformance": h.conformance,
                       "changed": h is not p, "ta1": f"{p.name}@{p.version}"}
    out["BN"] = {"binding": str(KBLOCK_N_FA3), "kind": "implicit-static", "ta1": str(b1.BN), "changed": True,
                 "note": "kBlockN, a module constant of b1.py read by AttnBlock/AttentionHead (should be a static parameter)"}
    return out


def _fn_key(fn: FunctionDefinition, names: dict[str, str]) -> tuple:
    if isinstance(fn, SpecializedDefinition):
        return ("spec", names.get(fn.name, fn.name), fn.version, tuple((k, _static_repr(v, names)) for k, v in fn.bindings.items()))
    return ("prim", names.get(fn.name, fn.name), fn.version)


def _static_repr(v, names: dict[str, str]):
    if isinstance(v, FunctionDefinition):
        return _fn_key(v, names)
    if isinstance(v, dict):
        return tuple(sorted((k, _static_repr(x, names)) for k, x in v.items()))
    if isinstance(v, (list, tuple)):
        return tuple(_static_repr(x, names) for x in v)
    return v


def structural_diff(a: SpecializedDefinition, b: SpecializedDefinition, names: dict[str, str] | None = None,
                    stop: frozenset[str] = frozenset(), _seen: set | None = None) -> list[dict]:
    """Node-for-node comparison of two specialised bodies: forms, argument references, output types, batch axes and
    counts, annotations, and callee identity with `names` applied to `a`'s side (b1 name -> port name).  Recurses
    into callee pairs (each pair once) except into definitions named in `stop`.  Returns the differences."""
    names = names or {}
    seen = _seen if _seen is not None else set()
    key = (a.id, b.id)
    if key in seen:
        return []
    seen.add(key)
    diffs: list[dict] = []
    where = f"{a.id} vs {b.id}"
    ba, bb = a.body, b.body
    if ba.params != bb.params or ba.ret_type != bb.ret_type:
        diffs.append({"where": where, "what": "signature", "a": repr((ba.params, ba.ret_type)), "b": repr((bb.params, bb.ret_type))})
    if len(ba.nodes) != len(bb.nodes):
        diffs.append({"where": where, "what": "node count", "a": len(ba.nodes), "b": len(bb.nodes)})
        return diffs
    if repr(ba.ret) != repr(bb.ret):
        diffs.append({"where": where, "what": "return refs", "a": repr(ba.ret), "b": repr(bb.ret)})
    for k, (na, nb) in enumerate(zip(ba.nodes, bb.nodes)):
        for f in ("form", "axes", "n", "n_xs", "name", "out"):
            if getattr(na, f) != getattr(nb, f):
                diffs.append({"where": where, "node": k, "what": f, "a": repr(getattr(na, f)), "b": repr(getattr(nb, f))})
        if len(na.args) != len(nb.args) or any(repr(x.refs) != repr(y.refs) or x.type != y.type for x, y in zip(na.args, nb.args)):
            diffs.append({"where": where, "node": k, "what": "args", "a": repr([x.refs for x in na.args]), "b": repr([y.refs for y in nb.args])})
        if _fn_key(na.fn, names) != _fn_key(nb.fn, {}):
            diffs.append({"where": where, "node": k, "what": "callee", "a": na.fn.id, "b": nb.fn.id})
        elif isinstance(na.fn, SpecializedDefinition) and na.fn.name not in stop:
            diffs.extend(structural_diff(na.fn, nb.fn, names, stop, seen))
    return diffs
