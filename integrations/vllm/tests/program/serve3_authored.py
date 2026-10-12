"""`Serve@3` authored in torch through the frontend (lane PAD-IMPL; semantics frozen by PAD-SPEC v1).

    Gate -> exactly one Value<w>.
    Composite interfaces -> ordered collections of value references.
    Commitment positions -> physical authentication locations for those values,
                            not semantic components of gate outputs.

The padded serving circuit's three new composites (`PadGuard_v1`, `PadDecodeStep_v1`, `Serve_v3`) written as
`@function`s over tensors, exactly as `b1_authored.py` writes `Serve@1/@2`: `ops.<Prim>` for the registered
primitives (`I32Eq`, `SelectI32`, `SelectBf16`, and the new `BitAnd`/`BitNot` of
`verity_vllm/program/registry/pad_prims.py`; the live flag is a `torch.bool` scalar — `Value<1>`), the authored
`b1_authored` v2 layer functions for the step bodies, `batch` for the row muxes, and torch views for the
reference rearrangements.  Nothing here is a new definition: each function carries the registered `Name_vN`
and statics of `verity_vllm/program/registry/serve3.py`, and `check_identity` (run by `export`) rejects it if its
body differed from the registered one.  As with `Serve@1/@2`, where digests nevertheless differ it is the
reference *encoding* of the zero-row broadcast (the registry writes repeated-ref runs, the frontend a
zero-stride `Strided`); `verity_vllm.program.frontend.correspond` shows the reference sequences are the same leaf by
leaf (`tests/program/test_serve3_authored.py`).  Needs torch; the tests skip without it.
"""

import torch

from verity_vllm.program.registry import b1
from verity_vllm.program.registry import pad_prims  # noqa: F401  — registers BitAnd_v1/BitNot_v1 for `ops.<name>` resolution
from verity_vllm.program.registry.prims import BF16, BIT, I32
from verity.ml.circuits.types import Array, Tuple as TupleT
from tests.program.b1_authored import FinalV2, _forward_positions_v2
from verity_vllm.program.frontend.torch_frontend import Arr, Dep, Tup, batch, function, ops


@function(name="PadGuard", version=1, static=("EOS",))
def PadGuard(live: BIT, y: I32, *, EOS) -> Tup[I32, BIT]:
    """The uniform guard of every sampling event (spec §3/§14.2): `stop = I32Eq(y, Const32[EOS])`;
    `live_out = BitAnd(live, BitNot(stop))` — absorbing; `tok_out = SelectI32(live, y, Const32[0x0])`
    (`PAD_TOKEN_ID = 0`, spec §6)."""
    c_eos = ops.const(32, EOS)
    stop = ops.I32Eq(y, c_eos)
    nstop = ops.BitNot(stop)
    live_out = ops.BitAnd(live, nstop)
    c_pad = ops.const(32, 0)
    tok_out = ops.SelectI32(live, y, c_pad)
    return (tok_out, live_out)


def _row_t(S):
    """One appended cache row per layer: `[NL][KVH][D] bf16` (the `k_new`/`v_new` boundary type)."""
    C = S["C"]
    return Array(C["NL"], Array(C["KVH"], Array(C["D"], BF16)))


@function(name="PadDecodeStep", version=1, static=("C", "T", "GEMM", "ATTENTION", "NORM"))
def PadDecodeStep(tok: I32, live: BIT,
                  kc: Dep(lambda S: Array(S["C"]["NL"], b1.kv_type(S["C"], S["T"] - 1))),
                  vc: Dep(lambda S: Array(S["C"]["NL"], b1.kv_type(S["C"], S["T"] - 1))),
                  weights: Dep(lambda S: b1.weights_type(S["C"])),
                  *, C, T, GEMM, ATTENTION, NORM) -> Dep(lambda S: TupleT(I32, BIT, _row_t(S), _row_t(S))):
    """One decode step of `Serve@3` at attention length `T` (registry `serve3.PadDecodeStep`): (a) exactly
    one `Serve_v2` decode step's gates, unguarded; (b) the guard on the raw sample; (c) word-wise row muxes
    against the broadcast zero row (spec §3/§6/§7)."""
    NL, KVH, D = C["NL"], C["KVH"], C["D"]
    caches = [(kc[L], vc[L]) for L in range(NL)]
    new_kv, fin = _forward_positions_v2(C, GEMM, ATTENTION, NORM, tok.view(1), weights, T - 1, 1, caches)
    y = FinalV2(fin[0][0], fin[0][1], weights["norm"], weights["embed"], C=C, GEMM=GEMM, NORM=NORM,
                name=f"t={T - 1}/final")
    tok_out, live_out = PadGuard(live, y, EOS=C["EOS_ID"], name=f"t={T - 1}/guard")
    z = ops.const(16, 0)
    zrow = z.expand(KVH * D)
    ks, vs = [], []
    for L in range(NL):
        krow = new_kv[L][0].view(KVH * D)
        vrow = new_kv[L][1].view(KVH * D)
        ks.append(batch(ops.SelectBf16, live, krow, zrow, axes=(None, 0, 0), name=f"t={T - 1}/kmux.{L}").view(KVH, D))
        vs.append(batch(ops.SelectBf16, live, vrow, zrow, axes=(None, 0, 0), name=f"t={T - 1}/vmux.{L}").view(KVH, D))
    return (tok_out, live_out, torch.stack(ks), torch.stack(vs))


@function(name="Serve", version=3, static=("C", "LP", "D_MAX", "GEMM", "ATTENTION", "NORM"))
def ServeV3(prompt: Arr["LP", I32], weights: Dep(lambda S: b1.weights_type(S["C"])),
            *, C, LP, D_MAX, GEMM, ATTENTION, NORM) \
        -> Dep(lambda S: TupleT(Array(S["D_MAX"] + 1, I32), Array(S["D_MAX"] + 1, BIT))):
    """The padded serving circuit (registry `serve3.ServeV3`): unguarded prefill, the prefill guard with
    `live_0 = Const1[0x1]`, then `D_MAX` chained `PadDecodeStep`s with a structural cache append (spec §3);
    returns `(out: [D_MAX+1] i32, live: [D_MAX+1] bit)` (spec §8)."""
    NL = C["NL"]
    caches, finals = _forward_positions_v2(C, GEMM, ATTENTION, NORM, prompt, weights, 0, LP, None)
    y0 = FinalV2(finals[-1][0], finals[-1][1], weights["norm"], weights["embed"], C=C, GEMM=GEMM, NORM=NORM,
                 name=f"t={LP - 1}/final")
    live = ops.const(1, 1)
    tok, live = PadGuard(live, y0, EOS=C["EOS_ID"], name=f"t={LP - 1}/guard")
    outs, lives = [tok], [live]
    for s in range(1, D_MAX + 1):
        T = LP + s
        kc = torch.stack([caches[L][0] for L in range(NL)])
        vc = torch.stack([caches[L][1] for L in range(NL)])
        tok, live, k_new, v_new = PadDecodeStep(tok, live, kc, vc, weights, C=C, T=T, GEMM=GEMM,
                                                ATTENTION=ATTENTION, NORM=NORM, name=f"s={s}/step")
        caches = [(torch.cat([caches[L][0], k_new[L].view(1, C["KVH"], C["D"])]),
                   torch.cat([caches[L][1], v_new[L].view(1, C["KVH"], C["D"])])) for L in range(NL)]
        outs.append(tok)
        lives.append(live)
    return (torch.stack(outs), torch.stack(lives))


AUTHORED = {f.id: f for f in (PadGuard, PadDecodeStep, ServeV3)}

__all__ = ["PadGuard", "PadDecodeStep", "ServeV3", "AUTHORED"]
