"""`Serve@3` — the padded serving circuit.

Full padding over the one output-dependent dimension of single-request greedy serving: the realized
generation length is represented *inside* the circuit as values (a per-step live bit), never as circuit
structure.  Circuit generation depends only on `(C₃, LP, D_MAX)` and the profile records; the circuit
contains all `D_MAX` decode steps regardless of where generation stops (spec §1–§2).

Composition (spec §3): the `Serve_v2{C, LP, STEPS=D_MAX}` gate skeleton + one guard template
(`PadGuard_v1{EOS}`, 6 gates) applied at every sampling event + word-wise row muxes
(`batch(SelectBf16_v1, live, row, zero-row)`) on every decode step's cache append.

* `PadGuard_v1{EOS}(live, y) -> (tok_out, live_out)`: `stop = I32Eq(y, Const32[EOS])`;
  `live_out = BitAnd(live, BitNot(stop))` — absorbing (spec §4); `tok_out = SelectI32(live, y,
  Const32[0])` — `PAD_TOKEN_ID = 0` (spec §6).  The stop test reads the **raw** sample; in the inactive
  suffix its result is absorbed by `BitAnd(0, ·)`, so no `PAD ≠ EOS` constraint exists (B0's `EOS_ID = 0`
  collides with the pad token and nothing breaks).
* `PadDecodeStep_v1{C, T, GEMM, ATTENTION, NORM}`: exactly the gates of one `Serve_v2` decode step at
  position `T-1` (no gate inside the body is guarded), then the guard, then the row muxes against the zero
  row.  A registered composite so the step boundary is a function interface: the inactive step relation
  `live_in = 0  ⟹  (tok_out, k_new, v_new, live_out) = (0, 0…0, 0…0, 0)` attaches to it (spec §3/§6).
* `Serve_v3{C, LP, D_MAX, GEMM, ATTENTION, NORM}(prompt, weights) -> (out: [D_MAX+1] i32,
  live: [D_MAX+1] bit)`: unguarded prefill (`LP` is prescribed; prefill is always live), the prefill
  guard with `live_0 = Const1[0x1]` (identity mux on `y_0`), then `D_MAX` chained `PadDecodeStep`s with a
  structural cache append every step.  Interpretation (spec §8): `n = Σ live[0..D_MAX-1]`,
  `transcript = out[0..n-1]` (EOS-exclusive); suffix slots equal 0 and are not outputs.

Version discipline (spec §10): `Serve_v1`/`Serve_v2` are frozen historical — nothing here edits or rebinds
them; `EOS_ID` lives only in `C₃` dicts bound into v3-family definitions (`C₃ = {**C, "EOS_ID": id}` at the
binding site, never added to a dict bound into v1/v2).  Conformance: the guard/step/serve definitions are
protocol definitions (spec §11: production correspondence applies to the live branch; inactive branches are
validated against the registered specification — see `serve3_reference.py`).
"""

from __future__ import annotations

from verity.ml.circuits.defs import bind, composite
from verity.ml.circuits.refs import Coll as _Coll
from verity.ml.circuits.refs import concat, tuple_of
from verity.ml.circuits.types import Array, Tuple
from verity.ml.definitions.registry import b1
from verity.ml.definitions.registry import prims as P
from verity.ml.definitions.registry.b1 import ZERO16, const, kv_type, weights_type
from verity.ml.definitions.registry.pad_prims import BitAnd, BitNot
from verity.ml.definitions.registry.prims import BF16, BIT, I32

# ---------------------------------------------------------------------------------------------------------
# the guard template (spec §3, §14.2 — one instance per sampling event, 6 gates)
# ---------------------------------------------------------------------------------------------------------


@composite("PadGuard", 1, ["EOS"], lambda S: ((("live", BIT), ("y", I32)), Tuple(I32, BIT)),
           conformance="protocol-defined")
def PadGuard(B, S, live, y):
    """`(live, y) -> (tok_out, live_out)`: `stop = I32Eq_v1(y, Const32[EOS])`; `live_out =
    BitAnd_v1(live, BitNot_v1(stop))` (absorbing); `tok_out = SelectI32_v1(live, y, Const32[0x0])`.
    The mux keys on the *step's* liveness, not on the outcome of its own sample: the event that samples
    the EOS is live and commits it (stop evidence, spec §8)."""
    c_eos = B.call(const(32, S.EOS))
    stop = B.call(P.I32Eq, y, c_eos)
    nstop = B.call(BitNot, stop)
    live_out = B.call(BitAnd, live, nstop)
    c_pad = B.call(const(32, 0))
    tok_out = B.call(P.SelectI32, live, y, c_pad)
    return tuple_of(tok_out, live_out)


# ---------------------------------------------------------------------------------------------------------
# one guarded decode step (spec §3 — a registered composite so the boundary is a function interface)
# ---------------------------------------------------------------------------------------------------------


def _row_type(C: dict) -> Array:
    """One appended cache row per layer: `[NL][KVH][D] bf16`."""
    return Array(C["NL"], Array(C["KVH"], Array(C["D"], BF16)))


@composite("PadDecodeStep", 1, ["C", "T", "GEMM", "ATTENTION", "NORM"],
           lambda S: ((("tok", I32), ("live", BIT),
                       ("kc", Array(S.C["NL"], kv_type(S.C, S.T - 1))),
                       ("vc", Array(S.C["NL"], kv_type(S.C, S.T - 1))),
                       ("weights", weights_type(S.C))),
                      Tuple(I32, BIT, _row_type(S.C), _row_type(S.C))),
           conformance="protocol-defined")
def PadDecodeStep(B, S, tok, live, kc, vc, weights):
    """One decode step of `Serve@3` at attention length `T` (position `T-1`): (a) the step body — exactly
    the gates of one `Serve_v2` decode step, unguarded; (b) `PadGuard_v1{EOS=C["EOS_ID"]}` on the raw
    sample; (c) row muxes — every appended k/v word `SelectBf16_v1(live, computed, 0x0000)` against one
    broadcast `Const16[0x0]` (the zero-row convention of `Clear_OOB_MN`, spec §6)."""
    C, T = S.C, S.T
    NL, KVH, D = C["NL"], C["KVH"], C["D"]
    # (a) the step body: embed, NL x (pre, attention over T rows, post), final — v2's gates verbatim
    caches = [(kc[L], vc[L]) for L in range(NL)]
    new_kv, last = b1._forward_positions_v2(B, S, _Coll(Array(1, I32), tok.refs), weights, T - 1, 1, caches)
    fin = bind(b1.FinalV2, C=C, GEMM=S.GEMM, NORM=S.NORM)
    y = B.call(fin, last[0][0], last[0][1], weights["norm"], weights["embed"], name=f"t={T - 1}/final")
    # (b) the guard
    g = B.call(bind(PadGuard, EOS=C["EOS_ID"]), live, y, name=f"t={T - 1}/guard")
    # (c) the row muxes: one zero gate, broadcast; 2*NL*KVH*D scalar selects
    z = B.call(ZERO16)
    zrow = _Coll(Array(KVH * D, BF16), concat([z.refs] * (KVH * D)))
    ks, vs = [], []
    for L in range(NL):
        krow = _Coll(Array(KVH * D, BF16), new_kv[L][0].refs)
        vrow = _Coll(Array(KVH * D, BF16), new_kv[L][1].refs)
        ks.append(B.batch(P.SelectBf16, live, krow, zrow, axes=(None, 0, 0), name=f"t={T - 1}/kmux.{L}"))
        vs.append(B.batch(P.SelectBf16, live, vrow, zrow, axes=(None, 0, 0), name=f"t={T - 1}/vmux.{L}"))
    k_new = _Coll(_row_type(C), concat([k.refs for k in ks]))
    v_new = _Coll(_row_type(C), concat([v.refs for v in vs]))
    return tuple_of(g[0], g[1], k_new, v_new)


# ---------------------------------------------------------------------------------------------------------
# the padded serving circuit (spec §3)
# ---------------------------------------------------------------------------------------------------------


@composite("Serve", 3, ["C", "LP", "D_MAX", "GEMM", "ATTENTION", "NORM"],
           lambda S: ((("prompt", Array(S.LP, I32)), ("weights", weights_type(S.C))),
                      Tuple(Array(S.D_MAX + 1, I32), Array(S.D_MAX + 1, BIT))),
           conformance="protocol-defined")
def ServeV3(B, S, prompt, weights):
    """The padded serving circuit: `Serve_v2{C, LP, STEPS=D_MAX}`'s skeleton + a uniform guard at all
    `D_MAX+1` sampling events + row muxes on every decode step's structural cache append.  `out[g]` is
    event `g`'s muxed token slot; `live[g]` is the flag *after* event `g` (spec §8)."""
    C, LP, D_MAX = S.C, S.LP, S.D_MAX
    if "EOS_ID" not in C:
        raise ValueError("Serve_v3 requires the extended configuration record C₃ (with \"EOS_ID\"; spec §2)")
    if not (1 <= D_MAX and LP + D_MAX <= C["MAX_POS"]):
        raise ValueError(f"Serve_v3 validity (spec §2): 1 <= D_MAX and LP + D_MAX <= MAX_POS "
                         f"(got LP={LP}, D_MAX={D_MAX}, MAX_POS={C['MAX_POS']})")
    NL = C["NL"]
    guard = bind(PadGuard, EOS=C["EOS_ID"])
    fin = bind(b1.FinalV2, C=C, GEMM=S.GEMM, NORM=S.NORM)
    # prefill: positions 0..LP-1, identical gates to Serve_v2 (unguarded; prefill is always live)
    caches, finals = b1._forward_positions_v2(B, S, prompt, weights, 0, LP, None)
    y0 = B.call(fin, finals[-1][0], finals[-1][1], weights["norm"], weights["embed"], name=f"t={LP - 1}/final")
    live = B.call(const(1, 1))  # live_0 = Const1[0x1]: a request always samples y_0
    g = B.call(guard, live, y0, name=f"t={LP - 1}/guard")
    tok, live = g[0], g[1]
    outs, lives = [tok], [live]
    for s in range(1, D_MAX + 1):  # decode step s: position LP+s-1, attention T = LP+s
        T = LP + s
        step = bind(PadDecodeStep, C=C, T=T, GEMM=S.GEMM, ATTENTION=S.ATTENTION, NORM=S.NORM)
        kc = _Coll(Array(NL, kv_type(C, T - 1)), concat([caches[L][0].refs for L in range(NL)]))
        vc = _Coll(Array(NL, kv_type(C, T - 1)), concat([caches[L][1].refs for L in range(NL)]))
        r = B.call(step, tok, live, kc, vc, weights, name=f"s={s}/step")
        tok, live = r[0], r[1]
        # structural append, every step, every layer (inactive steps append the zero row)
        caches = [(_Coll(kv_type(C, T), concat([caches[L][0].refs, r[2][L].refs])),
                   _Coll(kv_type(C, T), concat([caches[L][1].refs, r[3][L].refs]))) for L in range(NL)]
        outs.append(tok)
        lives.append(live)
    return tuple_of(_Coll(Array(D_MAX + 1, I32), concat([t.refs for t in outs])),
                    _Coll(Array(D_MAX + 1, BIT), concat([x.refs for x in lives])))


def serve_v3(C3: dict, LP: int, D_MAX: int, gemm: dict = b1.GEMM_AMPERE, attention: dict = b1.ATTENTION_FA2,
             norm: dict | None = None):
    """`Serve_v3` bound for one profile; by default the Ampere eager profile (as `b1.serve_v2`).  `C3` must
    carry `"EOS_ID"` — construct it as `{**C, "EOS_ID": id}` at the binding site (spec §10)."""
    return bind(ServeV3, C=C3, LP=LP, D_MAX=D_MAX, GEMM=gemm, ATTENTION=attention,
                NORM=norm if norm is not None else b1.norm_bundle(C3))


__all__ = ["PadGuard", "PadDecodeStep", "ServeV3", "serve_v3"]
