"""E2 — `probe_op`: evaluate ONE boundary op of the bound program from the record's COMMITTED boundary inputs and diff the
result against the record's COMMITTED output, word by word (CURVE wave item 1, D54 / D56).

    python -m e2e.interior.probe_op --record <rec dir> --request <rid> --op layers.0.self_attn.attn.attn --t 64 --layer 0 \
        [--head h] [--geometry BN=64] [--geometry BN=128 ...] [--program v2|v1] [--json out.json]

The op supported today is the attention boundary `layers.<l>.self_attn.attn.attn` (the site of the first disagreement of the
128x64 record). Its inputs are read from the committed leaves of the same RU:

* `layers.<l>.self_attn.rotary_emb.rope/port=0` at position t      -> q after RoPE, `[NH][D]`
* `layers.<l>.self_attn.rotary_emb.rope/port=1` at positions 0..t  -> K rows after RoPE, `[T][KVH][D]`
* `layers.<l>.self_attn.qkv_proj.gemm/port=0[H+KVH*D : QKV]` 0..t  -> V rows, `[T][KVH][D]`

and its output is `layers.<l>.self_attn.attn.attn/port=0` at t (`[NH][D]` bf16). The program is the registry's
`Attention_v2{T, NH, KVH, D, BN, DOT, INV}` (node-for-node identical to the pinned `Serve_v1`'s `Attention_v1` when bound with
`ATTENTION_FA2` = {BN 64, AmpereBF16TcDot16, Fa2InvSum}; `--program v1` cross-checks that by monkeypatching `b1.BN` and building
`Attention_v1` itself). Nothing in `verity.ir` is edited: a `--geometry` is a *binding* (BN, DOT, INV) plus, for the knobs
`AttentionHead_v2` does not expose as statics (block visiting order, the rescale contraction, the lane assignment of the
softmax sums), a local unregistered variant `AttentionHeadG` whose body is `AttentionHead_v2`'s with those choices as statics
(`ORDER=last_first|first_last`, `LANES=fa2|serial`, `RESCALE=fma|mul_add`). `--geometry` values are `KEY=VALUE[,KEY=VALUE...]`.

The evaluator is `verity.ir.evaluate.evaluate_call` (the oracle, gate by gate): one attention op at T <= 256 is a few hundred
thousand gates and runs in seconds on the laptop; the whole record's program is never built (D54: laptop `mem_guardian` 2.5 GB).

Nothing here is authority: the committed leaf is the store's; the evaluation is the reference semantics of the candidate
geometry; the report states, per (t, geometry), how many words differ, the first differing word with both bit patterns and
the bf16-ulp distance, and the largest ulp distance.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve()
REPO = HERE.parents[3]
for _p in (REPO, REPO / "vllm-poc", REPO / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from verity.ir.defs import Coll, CompositeDefinition, bind  # noqa: E402
from verity.ir.evaluate import evaluate_call  # noqa: E402
from verity.ir.program import Program  # noqa: E402
from verity.ir.refs import Coll as _Coll  # noqa: E402
from verity.ir.refs import array_of, concat, strided_view
from verity_vllm.registry import b1  # noqa: E402
from verity_vllm.registry import prims as P
from verity_vllm.registry.prims import BF16, F32  # noqa: E402
from verity.ir.types import Array  # noqa: E402

DTYPE_NP = {"bfloat16": np.uint16, "float32": np.uint32, "int32": np.uint32, "float8_e4m3fn": np.uint8, "uint8": np.uint8}

# ---------------------------------------------------------------------------------------------------------------------------
# the record's committed boundary values
# ---------------------------------------------------------------------------------------------------------------------------


class CommittedBoundary:
    """The RU's committed leaves (`adapter/selective/ru/<rid>/<step>/index.json` + `values.bin`), addressable by
    (position, layer, op, port). Steps are loaded lazily.

    Memory discipline (the laptop's `mem_guardian` kills lane processes > 256 MB whenever the disk is below its floor): an
    index is parsed once, reduced to `(offset, nbytes, dtype, shape)` per (t, l, op, port) and the JSON dropped; positions
    and phases of the steps come from the small head of each index; words are read with explicit seeks, no memmap."""

    #: op suffixes kept from the indexes (None = everything); the attention probe reads these three
    keep_ops: tuple[str, ...] | None = ("self_attn.rotary_emb.rope", "self_attn.qkv_proj.gemm", "self_attn.attn.attn")

    def __init__(self, rec_dir: Path, request: str) -> None:
        self.rec_dir = Path(rec_dir)
        self.request = request
        self.ru = self.rec_dir / "adapter" / "selective" / "ru" / request
        if not self.ru.is_dir():
            raise SystemExit(f"no RU directory {self.ru}")
        self.steps = sorted(int(p.name) for p in self.ru.iterdir() if p.is_dir() and p.name.isdigit())
        self._entries: dict[tuple[int, str, str, int], tuple[int, int, int, str, tuple[int, ...]]] = {}
        self._loaded: set[int] = set()
        self.positions: dict[int, list[int]] = {}
        self.phase: dict[int, str] = {}
        self.profile_id = None
        for s in self.steps:
            head = self._index_head(s)
            self.positions[s] = list(head["positions"])
            self.phase[s] = head["phase"]
            if self.profile_id is None:
                self.profile_id = head.get("profile_id")

    def _index_head(self, step: int) -> dict:
        """`positions` / `phase` / `profile_id` of a step without parsing its entries (the prefill index is tens of MB)."""
        p = self.ru / f"{step:06d}" / "index.json"
        with p.open("rb") as f:
            head = f.read(4096).decode("utf-8", "replace")
        out: dict[str, Any] = {}
        for key in ("phase", "profile_id"):
            i = head.find(f'"{key}"')
            if i >= 0:
                j = head.find('"', head.find(":", i) + 1)
                out[key] = head[j + 1:head.find('"', j + 1)]
        i = head.find('"positions"')
        if i >= 0:
            j = head.find("[", i)
            k = head.find("]", j)
            if k < 0:  # long position list: parse the whole file once
                return json.loads(p.read_text())
            out["positions"] = json.loads(head[j:k + 1])
        if "positions" not in out or "phase" not in out:
            return json.loads(p.read_text())
        return out

    def step_of(self, t: int) -> int:
        for s in self.steps:
            if t in self.positions[s]:
                return s
        raise KeyError(f"position {t} is in no RU step (steps {self.steps[0]}..{self.steps[-1]})")

    def _ensure(self, step: int) -> None:
        if step in self._loaded:
            return
        p = self.ru / f"{step:06d}" / "values.bin"
        if not p.exists():
            raise SystemExit(f"{p} is missing: the record is slim (leaf hashes only) — the probe needs the committed WORDS")
        # stream the entries one object at a time (the prefill index is 16 MB / 40 k entries at 128x64: a whole-file
        # json.loads leaves ~90 MB in the allocator); keep only the ops the probe reads
        text = (self.ru / f"{step:06d}" / "index.json").read_text()
        i = text.find('"entries"')
        i = text.find("[", i) + 1
        dec = json.JSONDecoder()
        keep = self.keep_ops
        n = len(text)
        while i < n:
            while i < n and text[i] in " \r\n\t,":
                i += 1
            if i >= n or text[i] == "]":
                break
            e, i = dec.raw_decode(text, i)
            f = dict(kv.split("=", 1) for kv in e["canonical_id"].split("/"))
            if keep is not None and not any(f["op"].endswith(k) for k in keep):
                continue
            self._entries[(int(f["t"]), str(f["l"]), f["op"], int(f["port"]))] = \
                (step, int(e["offset"]), int(e["nbytes"]), str(e["dtype"]), tuple(int(x) for x in e["shape"]))
        del text
        self._loaded.add(step)

    def words(self, t: int, layer: int | str, op: str, port: int = 0) -> np.ndarray:
        step = self.step_of(t)
        self._ensure(step)
        try:
            step, offset, nbytes, dtype, shape = self._entries[(t, str(layer), op, port)]
        except KeyError:
            raise KeyError(f"no committed leaf t={t} l={layer} op={op} port={port} in step {step}") from None
        with (self.ru / f"{step:06d}" / "values.bin").open("rb") as f:
            f.seek(offset)
            raw = f.read(nbytes)
        return np.frombuffer(raw, dtype=DTYPE_NP[dtype]).reshape(shape)

    def entry(self, t: int, layer: int | str, op: str, port: int = 0) -> dict:
        self._ensure(self.step_of(t))
        step, offset, nbytes, dtype, shape = self._entries[(t, str(layer), op, port)]
        return {"step": step, "offset": offset, "nbytes": nbytes, "dtype": dtype, "shape": list(shape)}


def install_lean_mufu_tables() -> None:
    """Pre-populate the numerics lane's MUFU table cache (`fa2_relation._TABLES`) with tables reconstructed in 1 M-entry
    chunks: the lane's `MufuTables.load` materialises two int64 copies of each 2^23 table (≈ 220 MB of transients the
    allocator keeps), which alone puts a laptop process over the guardian's 256 MB line; the chunked reconstruction is the
    same cumulative sum (verified equal on both tables) with a peak of ≈ 40 MB. Values are identical; only the peak differs."""
    import hashlib
    import lzma

    from verity_vllm_numerics import fa2_model as FA
    from verity_vllm_numerics import fa2_relation as FR
    d = FR.tables_dir()
    if d in FR._TABLES or not os.path.isdir(d):
        return
    tabs, src = [], {}
    for name, t0 in (("ex2", FA.EX2_T0), ("rcp", FA.RCP_R0)):
        p = Path(d) / f"mufu_{name}_delta_int8.xz"
        raw = p.read_bytes()
        delta = np.frombuffer(lzma.decompress(raw), dtype=np.int8)
        if delta.shape[0] != FA.TABLE_N - 1:
            raise ValueError(f"{p}: expected {FA.TABLE_N - 1} deltas, got {delta.shape[0]}")
        t = np.empty(FA.TABLE_N, dtype=np.uint32)
        t[0] = t0
        acc = int(t0)
        step = 1 << 18
        for lo in range(0, FA.TABLE_N - 1, step):
            chunk = np.cumsum(delta[lo:lo + step].astype(np.int64)) + acc
            t[lo + 1:lo + 1 + chunk.size] = chunk.astype(np.uint32)
            acc = int(chunk[-1])
        del delta, chunk
        tabs.append(t)
        src[name] = {"file": p.name, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
        del raw
    FR._TABLES[d] = FA.MufuTables(tabs[0], tabs[1], src)


# ---------------------------------------------------------------------------------------------------------------------------
# geometry: a binding of Attention_v2's statics plus the local knobs AttentionHead_v2 does not expose
# ---------------------------------------------------------------------------------------------------------------------------

DOTS = {"AmpereBF16TcDot16": P.AmpereBF16TcDot16}
INVS = {"Fa2InvSum": P.Fa2InvSum}
for _name in ("HopperBF16WgmmaDot16",):
    if hasattr(P, _name):
        DOTS[_name] = getattr(P, _name)
for _name in ("Fa3InvSum",):
    if hasattr(P, _name):
        INVS[_name] = getattr(P, _name)

DEFAULT_GEOMETRY = {"BN": 64, "BM": None, "DOT": "AmpereBF16TcDot16", "INV": "Fa2InvSum", "ORDER": "last_first", "LANES": "fa2", "RESCALE": "fma"}
# D57 vocabulary: a geometry is the triple (kBlockM, kBlockN, composition). kBlockM is recorded when given (`KBLOCKM=64`) but is NOT
# observable by the per-row model — a query row's key blocks depend on kBlockN alone whenever kBlockM <= kBlockN (a row never meets a
# fully-masked block; masked columns contribute exactly 0) — so the probe cannot confirm or refute it; it comes from the kernel source.
ORDERS = ("last_first", "first_last")
LANES = ("fa2", "serial")
RESCALES = ("fma", "mul_add")


def parse_geometry(spec: str | None) -> dict[str, Any]:
    g = dict(DEFAULT_GEOMETRY)
    if spec:
        for kv in spec.split(","):
            kv = kv.strip()
            if not kv:
                continue
            k, v = kv.split("=", 1)
            k = k.strip().upper()
            v = v.strip()
            if k in ("BN", "KBLOCKN"):
                g["BN"] = int(v)
                if g["BN"] % 16 or g["BN"] <= 0:
                    raise SystemExit(f"BN must be a positive multiple of 16, got {v}")
            elif k in ("BM", "KBLOCKM"):
                g["BM"] = int(v)
            elif k == "DOT":
                if v not in DOTS:
                    raise SystemExit(f"DOT must be one of {sorted(DOTS)}, got {v}")
                g["DOT"] = v
            elif k == "INV":
                if v not in INVS:
                    raise SystemExit(f"INV must be one of {sorted(INVS)}, got {v}")
                g["INV"] = v
            elif k == "ORDER":
                if v not in ORDERS:
                    raise SystemExit(f"ORDER must be one of {ORDERS}, got {v}")
                g["ORDER"] = v
            elif k == "LANES":
                if v not in LANES:
                    raise SystemExit(f"LANES must be one of {LANES}, got {v}")
                g["LANES"] = v
            elif k == "RESCALE":
                if v not in RESCALES:
                    raise SystemExit(f"RESCALE must be one of {RESCALES}, got {v}")
                g["RESCALE"] = v
            else:
                raise SystemExit(f"unknown geometry key {k!r} (BN|KBLOCKN, BM|KBLOCKM, DOT, INV, ORDER, LANES, RESCALE)")
    return g


def geometry_label(g: dict[str, Any]) -> str:
    parts = [f"BN={g['BN']}"]
    if g.get("BM") is not None:
        parts.append(f"BM={g['BM']}")
    for k in ("DOT", "INV", "ORDER", "LANES", "RESCALE"):
        if g[k] != DEFAULT_GEOMETRY[k]:
            parts.append(f"{k}={g[k]}")
    return ",".join(parts)


def geometry_triple(g: dict[str, Any]) -> str:
    """The D57 form `(kBlockM, kBlockN, composition)`; kBlockM `?` when not given (not observable by the per-row model)."""
    comp = f"{g['ORDER']}/{'4-lane' if g['LANES'] == 'fa2' else 'serial'}-sums/{g['RESCALE']}-rescale/{g['DOT']}/{g['INV']}"
    return f"(kBlockM={g['BM'] if g.get('BM') is not None else '?'}, kBlockN={g['BN']}, {comp})"


def is_binding_only(g: dict[str, Any]) -> bool:
    """True when the geometry is expressible as a binding of the registered `Attention_v2` (no local variant needed)."""
    return g["ORDER"] == "last_first" and g["LANES"] == "fa2" and g["RESCALE"] == "fma"


# -- the local variant: AttentionHead_v2's body with ORDER / LANES / RESCALE as statics (unregistered; probe only) ----------

_ZERO32 = b1.ZERO32
_ZERO16 = b1.ZERO16


def _attn_block_g_body(B, S, q, k, v, m_prev, l_prev, o_prev):
    """`AttnBlock_v2`'s body with `LANES` (fa2: four lanes `(c%8)//2`; serial: one running sum) and `RESCALE` (fma: the
    block's first add contracted with the rescale, `l*scale + p0`; mul_add: `(l*scale) + p0`) as statics."""
    D, NVIS, FIRST, BN, DOT = S.D, S.NVIS, S.FIRST, S.BN, S.DOT
    dot = bind(b1.DotBf16V2, K=D, DOT=DOT)
    s = [B.call(dot, q, k[c]) for c in range(NVIS)]
    bmax = s[0]
    for c in range(1, NVIS):
        bmax = B.call(P.F32Max, bmax, s[c])
    m_new = bmax if FIRST else B.call(P.F32Max, m_prev, bmax)
    m_use = B.call(P.GuardNegInfZero, m_new)
    scale_log2 = B.call(b1.const(32, b1._scale_log2_bits(D)))
    if not FIRST:
        scores_scale = B.call(P.MufuEx2Ftz, B.call(P.F32MulFtz, B.call(P.F32SubFtz, m_prev, m_use), scale_log2))
    max_scaled = B.call(P.F32MulFtz, m_use, scale_log2)
    p = [B.call(P.MufuEx2Ftz, B.call(P.F32FmaSubFtz, s[c], scale_log2, max_scaled)) for c in range(NVIS)]
    lane_of = b1._lane if S.LANES == "fa2" else (lambda c: 0)
    lanes: list[Coll] = []
    for ln in range(4):
        cols = [c for c in range(NVIS) if lane_of(c) == ln]
        if FIRST:
            acc = p[cols[0]] if cols else B.call(_ZERO32)
            rest = cols[1:]
        else:
            first = p[cols[0]] if cols else B.call(_ZERO32)
            if S.RESCALE == "fma":
                acc = B.call(P.F32FmaFtz, scores_scale, l_prev[ln], first)
            else:
                acc = B.call(P.F32AddFtz, B.call(P.F32MulFtz, l_prev[ln], scores_scale), first)
            rest = cols[1:]
        for c in rest:
            acc = B.call(P.F32AddFtz, acc, p[c])
        lanes.append(acc)
    pb = [B.call(P.F2fpBf16, x) for x in p]
    zero16 = B.call(_ZERO16) if NVIS < BN else None
    o = [B.call(_ZERO32) for _ in range(D)] if FIRST else [B.call(P.F32MulFtz, o_prev[d], scores_scale) for d in range(D)]
    for ks in range(BN // 16):
        k0 = 16 * ks
        if k0 >= NVIS:
            break
        prow = _Coll(P.A16, concat([(pb[k0 + j] if k0 + j < NVIS else zero16).refs for j in range(16)]))
        vblk = v[k0:k0 + 16]
        vt_refs = strided_view(vblk, 0, 16, 1, D, 16 * D) if isinstance(vblk.refs, b1.Affine) else \
            b1.Explicit(tuple(vblk.refs[j * D + d] for d in range(D) for j in range(16)))
        vt = _Coll(Array(D, P.A16), vt_refs)
        o = list(B.batch(DOT, array_of(o), prow, vt, axes=(0, None, 0)))
    return b1.tuple_of(m_new, array_of(lanes), array_of(o))


AttnBlockG = CompositeDefinition(
    "AttnBlockG", 1, ("D", "NVIS", "FIRST", "BN", "DOT", "LANES", "RESCALE"),
    lambda S: (((("q", Array(S.D, BF16)), ("k", Array(S.BN, Array(S.D, BF16))), ("v", Array(S.BN, Array(S.D, BF16))),
                 ("m_prev", F32), ("l_prev", Array(4, F32)), ("o_prev", Array(S.D, F32)))),
               b1.Tuple(F32, Array(4, F32), Array(S.D, F32))),
    _attn_block_g_body, doc="probe-only variant of AttnBlock_v2 (LANES, RESCALE statics)", register=False)


def _attention_head_g_body(B, S, q, k, v):
    """`AttentionHead_v2`'s body with the block visiting order as the static `ORDER` (last_first = FA2; first_last = the
    other direction), blocks built by `AttnBlockG{LANES, RESCALE}`."""
    T, D, BN, DOT, INV = S.T, S.D, S.BN, S.DOT, S.INV
    nb = (T + BN - 1) // BN
    order = list(range(nb - 1, -1, -1)) if S.ORDER == "last_first" else list(range(nb))
    m = lsum = o = None
    zero_row = None
    for i, b in enumerate(order):
        lo, hi = b * BN, min(T, (b + 1) * BN)
        nvis = hi - lo
        kb, vb = k[lo:hi], v[lo:hi]
        if nvis < BN:
            if zero_row is None:
                zero_row = _Coll(Array(D, BF16), concat([B.call(_ZERO16).refs] * D))
            pad = [zero_row.refs] * (BN - nvis)
            kb = _Coll(Array(BN, Array(D, BF16)), concat([kb.refs] + pad))
            vb = _Coll(Array(BN, Array(D, BF16)), concat([vb.refs] + pad))
        first = i == 0
        blk = bind(AttnBlockG, D=D, NVIS=nvis, FIRST=first, BN=BN, DOT=DOT, LANES=S.LANES, RESCALE=S.RESCALE)
        if first:
            z = B.call(_ZERO32)
            l0 = _Coll(Array(4, F32), concat([z.refs] * 4))
            o0 = _Coll(Array(D, F32), concat([z.refs] * D))
            st = B.call(blk, q, kb, vb, z, l0, o0)
        else:
            st = B.call(blk, q, kb, vb, m, lsum, o)
        m, lsum, o = st[0], st[1], st[2]
    total = B.call(P.F32AddFtz, B.call(P.F32AddFtz, lsum[0], lsum[2]), B.call(P.F32AddFtz, lsum[1], lsum[3]))
    inv = B.call(INV, total)
    y = B.batch(P.F32MulFtz, o, inv, axes=(0, None))
    return B.batch(P.F2fpBf16, y, axes=(0,))


AttentionHeadG = CompositeDefinition(
    "AttentionHeadG", 1, ("T", "D", "BN", "DOT", "INV", "ORDER", "LANES", "RESCALE"),
    lambda S: ((("q", Array(S.D, BF16)), ("k", Array(S.T, Array(S.D, BF16))), ("v", Array(S.T, Array(S.D, BF16)))), Array(S.D, BF16)),
    _attention_head_g_body, doc="probe-only variant of AttentionHead_v2 (ORDER, LANES, RESCALE statics)", register=False)


def _attention_g_body(B, S, q, kc, vc):
    T, NH, KVH, D = S.T, S.NH, S.KVH, S.D
    G = NH // KVH
    outs = []
    head = bind(AttentionHeadG, T=T, D=D, BN=S.BN, DOT=S.DOT, INV=S.INV, ORDER=S.ORDER, LANES=S.LANES, RESCALE=S.RESCALE)
    for g in range(KVH):
        kg = _Coll(Array(T, Array(D, BF16)), strided_view(kc, g * D, D, KVH * D, 1, T * D))
        vg = _Coll(Array(T, Array(D, BF16)), strided_view(vc, g * D, D, KVH * D, 1, T * D))
        outs.append(B.batch(head, q[g * G:(g + 1) * G], kg, vg, axes=(0, None, None)))
    return _Coll(Array(NH, Array(D, BF16)), concat([o.refs for o in outs]))


AttentionG = CompositeDefinition(
    "AttentionG", 1, ("T", "NH", "KVH", "D", "BN", "DOT", "INV", "ORDER", "LANES", "RESCALE"),
    lambda S: ((("q", Array(S.NH, Array(S.D, BF16))), ("kc", Array(S.T, Array(S.KVH, Array(S.D, BF16)))),
                ("vc", Array(S.T, Array(S.KVH, Array(S.D, BF16))))), Array(S.NH, Array(S.D, BF16))),
    _attention_g_body, doc="probe-only variant of Attention_v2 (ORDER, LANES, RESCALE statics)", register=False)


# ---------------------------------------------------------------------------------------------------------------------------
# building and evaluating one op
# ---------------------------------------------------------------------------------------------------------------------------

_STANDALONE = CompositeDefinition("ProbeStandalone", 1, ("FN",), lambda S: S.FN.signature(),
                                  lambda B, S, *args: B.call(S.FN, *args),
                                  doc="root wrapper: one call of FN over Input gates", register=False)


def standalone(spec):
    prog = Program(bind(_STANDALONE, FN=spec))
    calls = [c for c in prog.circuit.root_calls() if not c.is_primitive]
    assert len(calls) == 1, calls
    return prog, calls[0]


def attention_spec(g: dict[str, Any], *, T: int, NH: int, KVH: int, D: int, program: str, head: int | None):
    """The bound function of the candidate geometry: `Attention_v2` / `AttentionHead_v2` when the geometry is a pure binding,
    the local `AttentionG` / `AttentionHeadG` otherwise; `--program v1` monkeypatches `b1.BN` and binds `Attention_v1`."""
    DOT, INV = DOTS[g["DOT"]], INVS[g["INV"]]
    if program == "v1":
        if not is_binding_only(g) or g["DOT"] != "AmpereBF16TcDot16" or g["INV"] != "Fa2InvSum":
            raise SystemExit("--program v1 supports only BN (the module constant b1.BN); ORDER/LANES/RESCALE/DOT/INV need v2")
        b1.BN = int(g["BN"])  # the v1 bodies read the module constant at build time; one BN per process
        if head is None:
            return bind(b1.Attention, T=T, NH=NH, KVH=KVH, D=D)
        return bind(b1.AttentionHead, T=T, D=D)
    if is_binding_only(g):
        if head is None:
            return bind(b1.AttentionV2, T=T, NH=NH, KVH=KVH, D=D, BN=g["BN"], DOT=DOT, INV=INV)
        return bind(b1.AttentionHeadV2, T=T, D=D, BN=g["BN"], DOT=DOT, INV=INV)
    if head is None:
        return bind(AttentionG, T=T, NH=NH, KVH=KVH, D=D, BN=g["BN"], DOT=DOT, INV=INV, ORDER=g["ORDER"], LANES=g["LANES"], RESCALE=g["RESCALE"])
    return bind(AttentionHeadG, T=T, D=D, BN=g["BN"], DOT=DOT, INV=INV, ORDER=g["ORDER"], LANES=g["LANES"], RESCALE=g["RESCALE"])


def attention_inputs(cb: CommittedBoundary, C: dict, layer: int, t: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(q[NH][D], kc[T][KVH][D], vc[T][KVH][D], out[NH][D]) as uint16 bf16 words, from the committed leaves."""
    NH, KVH, D, H = C["NH"], C["KVH"], C["D"], C["H"]
    q = cb.words(t, layer, f"layers.{layer}.self_attn.rotary_emb.rope", 0).reshape(NH, D)
    kc = np.empty((t + 1, KVH, D), dtype=np.uint16)
    vc = np.empty((t + 1, KVH, D), dtype=np.uint16)
    for tp in range(t + 1):
        kc[tp] = cb.words(tp, layer, f"layers.{layer}.self_attn.rotary_emb.rope", 1).reshape(KVH, D)
        qkv = cb.words(tp, layer, f"layers.{layer}.self_attn.qkv_proj.gemm", 0)
        vc[tp] = qkv[H + KVH * D:H + 2 * KVH * D].reshape(KVH, D)
    out = cb.words(t, layer, f"layers.{layer}.self_attn.attn.attn", 0).reshape(NH, D)
    return q, kc, vc, out


def bf16_ordered(x: np.ndarray) -> np.ndarray:
    """bf16 bit patterns -> integers ordered like the reals (sign-magnitude -> two's complement)."""
    x = x.astype(np.int64)
    return np.where(x >= 0x8000, 0x8000 - x, x)


def bf16_to_f32(x: np.ndarray) -> np.ndarray:
    return (x.astype(np.uint32) << 16).view(np.float32)


def compare_words(got: np.ndarray, want: np.ndarray, *, NH: int, D: int) -> dict[str, Any]:
    got = np.asarray(got, dtype=np.uint16).ravel()
    want = np.asarray(want, dtype=np.uint16).ravel()
    diff = np.nonzero(got != want)[0]
    res: dict[str, Any] = {"words": int(got.size), "differ": int(diff.size), "equal": bool(diff.size == 0)}
    if diff.size:
        ulp = np.abs(bf16_ordered(got[diff]) - bf16_ordered(want[diff]))
        i = int(diff[0])
        res["first"] = {"word": i, "head": i // D, "d": i % D,
                        "program_bits": f"0x{int(got[i]):04x}", "committed_bits": f"0x{int(want[i]):04x}",
                        "program_f32": float(bf16_to_f32(got[i:i + 1])[0]), "committed_f32": float(bf16_to_f32(want[i:i + 1])[0]),
                        "ulp": int(ulp[0])}
        res["ulp_max"] = int(ulp.max())
        res["ulp_hist"] = {str(int(k)): int(v) for k, v in zip(*np.unique(np.minimum(ulp, 8), return_counts=True), strict=True)}
        res["heads_differ"] = sorted({int(j // D) for j in diff})
    return res


def probe_attention(cb: CommittedBoundary, C: dict, *, layer: int, t: int, g: dict[str, Any], program: str, head: int | None,
                    verbose: bool = True) -> dict[str, Any]:
    NH, KVH, D = C["NH"], C["KVH"], C["D"]
    T = t + 1
    q, kc, vc, out = attention_inputs(cb, C, layer, t)
    t0 = time.perf_counter()
    spec = attention_spec(g, T=T, NH=NH, KVH=KVH, D=D, program=program, head=head)
    prog, call = standalone(spec)
    t_build = time.perf_counter() - t0
    if head is None:
        args = [int(x) for x in q.ravel()] + [int(x) for x in kc.ravel()] + [int(x) for x in vc.ravel()]
        want = out
    else:
        G = NH // KVH
        grp = head // G
        args = [int(x) for x in q[head]] + [int(x) for x in kc[:, grp, :].ravel()] + [int(x) for x in vc[:, grp, :].ravel()]
        want = out[head]
    t1 = time.perf_counter()
    got = np.array(evaluate_call(prog.circuit, call, args), dtype=np.uint16)
    t_eval = time.perf_counter() - t1
    res = compare_words(got, want, NH=NH if head is None else 1, D=D)
    if head is not None and "first" in res:
        res["first"]["head"] = head
    res.update({"t": t, "T": T, "layer": layer, "head": head, "geometry": geometry_label(g), "geometry_triple": geometry_triple(g),
                "geometry_full": dict(g), "program": program,
                "function": call.id, "gates": int(call.size), "blocks": (T + g["BN"] - 1) // g["BN"],
                "build_s": round(t_build, 3), "eval_s": round(t_eval, 3), "phase": cb.phase[cb.step_of(t)], "step": cb.step_of(t)})
    if verbose:
        print(format_row(res), flush=True)
    return res


def format_row(r: dict[str, Any]) -> str:
    head = f"h={r['head']}" if r["head"] is not None else "all heads"
    tag = "EQUAL" if r["equal"] else f"DIFFER {r['differ']}/{r['words']}"
    s = f"t={r['t']:>4} {r['phase']:<7} l={r['layer']} {head:<9} {r['geometry']:<40} blocks={r['blocks']} {r['function']} " \
        f"gates={r['gates']:,} build={r['build_s']}s eval={r['eval_s']}s -> {tag}"
    if not r["equal"]:
        f = r["first"]
        s += f"; first word {f['word']} (head {f['head']}, d {f['d']}): program {f['program_bits']} ({f['program_f32']:.6g}) vs " \
             f"committed {f['committed_bits']} ({f['committed_f32']:.6g}), {f['ulp']} ulp; ulp max {r['ulp_max']}, hist {r['ulp_hist']}; " \
             f"heads {r['heads_differ']}"
    return s


# ---------------------------------------------------------------------------------------------------------------------------
# self-test: a synthetic record whose committed attention output is the program's own at BN=64
# ---------------------------------------------------------------------------------------------------------------------------


def write_synthetic_record(out_dir: Path, C: dict, *, LP: int, steps: int, layer: int, seed: int, BN: int = 64) -> str:
    """A minimal RU (prefill step 0 with positions 0..LP-1, then `steps` decode steps) carrying random q/K/V leaves and the
    attention output computed by `Attention_v2{BN}` — for testing the probe's plumbing on the laptop. Returns the request id."""
    rng = np.random.default_rng(seed)
    NH, KVH, D, H = C["NH"], C["KVH"], C["D"], C["H"]
    rid = f"00000000-0000-4000-8000-{seed:012x}"
    ru = out_dir / "adapter" / "selective" / "ru" / rid
    ru.mkdir(parents=True, exist_ok=True)

    def rand_bf16(n, scale):
        return ((rng.standard_normal(n) * scale).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)

    kc_all = np.empty((LP + steps, KVH, D), dtype=np.uint16)
    vc_all = np.empty((LP + steps, KVH, D), dtype=np.uint16)
    q_all = np.empty((LP + steps, NH, D), dtype=np.uint16)
    for t in range(LP + steps):
        kc_all[t] = rand_bf16(KVH * D, 1.5).reshape(KVH, D)
        vc_all[t] = rand_bf16(KVH * D, 1.0).reshape(KVH, D)
        q_all[t] = rand_bf16(NH * D, 1.5).reshape(NH, D)
    cache: dict[int, tuple] = {}

    def attn_out(t):
        T = t + 1
        if T not in cache:
            cache[T] = standalone(bind(b1.AttentionV2, T=T, NH=NH, KVH=KVH, D=D, BN=BN, DOT=P.AmpereBF16TcDot16, INV=P.Fa2InvSum))
        prog, call = cache[T]
        args = [int(x) for x in q_all[t].ravel()] + [int(x) for x in kc_all[:T].ravel()] + [int(x) for x in vc_all[:T].ravel()]
        return np.array(evaluate_call(prog.circuit, call, args), dtype=np.uint16).reshape(NH, D)

    def write_step(step, positions, phase):
        entries = []
        blob = bytearray()
        for t in positions:
            qkv = np.zeros(C["QKV"], dtype=np.uint16)
            qkv[H + KVH * D:H + 2 * KVH * D] = vc_all[t].ravel()
            leaves = [(f"layers.{layer}.self_attn.qkv_proj.gemm", 0, qkv, [C["QKV"]]),
                      (f"layers.{layer}.self_attn.rotary_emb.rope", 0, q_all[t].ravel(), [H]),
                      (f"layers.{layer}.self_attn.rotary_emb.rope", 1, kc_all[t].ravel(), [KVH * D]),
                      (f"layers.{layer}.self_attn.attn.attn", 0, attn_out(t).ravel(), [NH, D])]
            for op, port, words, shape in leaves:
                raw = np.ascontiguousarray(words, dtype=np.uint16).tobytes()
                entries.append({"canonical_id": f"p=probeselftest/r={rid}/t={t}/ph={phase}/l={layer}/op={op}/port={port}",
                                "dtype": "bfloat16", "shape": shape, "offset": len(blob), "nbytes": len(raw), "origin": "synthetic",
                                "leaf": "0" * 64})
                blob += raw
        d = ru / f"{step:06d}"
        d.mkdir(exist_ok=True)
        (d / "values.bin").write_bytes(bytes(blob))
        (d / "index.json").write_text(json.dumps({"schema": "verity-vllm/ru-index/v1", "request_id": rid, "step": step, "phase": phase,
                                                   "positions": list(positions), "profile_id": "probeselftest", "values_nbytes": len(blob),
                                                   "tainted": False, "conflicts": [], "entries": entries}))

    write_step(0, list(range(LP)), "prefill")
    for s in range(1, steps + 1):
        write_step(s, [LP + s - 1], "decode")
    return rid


# ---------------------------------------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------------------------------------


def _rss_gb() -> float:
    try:
        import resource
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return r / (1024 ** 3) if sys.platform == "darwin" else r / (1024 ** 2)
    except Exception:  # noqa: BLE001
        return float("nan")


def _rss_now_mb() -> float:
    """Current resident size (what the laptop's guardian samples), via ps."""
    try:
        import subprocess
        return int(subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())], capture_output=True, text=True).stdout) / 1024
    except Exception:  # noqa: BLE001
        return float("nan")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--record", type=Path, help="record directory (holds adapter/selective/ru/<request>/<step>/{index.json,values.bin})")
    ap.add_argument("--request", help="request id (default: the only one under adapter/selective/ru)")
    ap.add_argument("--op", default="layers.0.self_attn.attn.attn", help="canonical op (only layers.<l>.self_attn.attn.attn today)")
    ap.add_argument("--t", type=str, default="64", help="position(s), comma-separated, e.g. 63,64,65,127,128")
    ap.add_argument("--layer", type=int, default=None, help="layer (default: parsed from --op)")
    ap.add_argument("--head", type=int, default=None, help="one head only (default: all NH heads of the op)")
    ap.add_argument("--geometry", action="append", default=None,
                    help="KEY=VALUE[,...]: BN=<int> DOT=<prim> INV=<prim> ORDER=last_first|first_last LANES=fa2|serial RESCALE=fma|mul_add; "
                         "repeatable; default BN=64 (ATTENTION_FA2)")
    ap.add_argument("--program", choices=("v2", "v1"), default="v2", help="v2: Attention_v2 bound with the geometry (default); v1: monkeypatch b1.BN, build Attention_v1")
    ap.add_argument("--config", default="B0_CONFIG", help="static configuration name in verity_vllm.registry.b1 (default B0_CONFIG = SmolLM2-135M)")
    ap.add_argument("--json", type=Path, default=None, help="write all results as JSON")
    ap.add_argument("--selftest", type=Path, default=None, help="write a synthetic record here (BN=64 outputs) and probe it instead of --record")
    ap.add_argument("--selftest-lp", type=int, default=70)
    ap.add_argument("--selftest-steps", type=int, default=2)
    a = ap.parse_args(argv)

    C = getattr(b1, a.config)
    if not a.op.startswith("layers.") or not a.op.endswith(".self_attn.attn.attn"):
        raise SystemExit(f"unsupported op {a.op!r}: only layers.<l>.self_attn.attn.attn is implemented")
    layer = a.layer if a.layer is not None else int(a.op.split(".")[1])
    ts = [int(x) for x in a.t.split(",") if x.strip()]
    geoms = [parse_geometry(s) for s in (a.geometry or [None])]
    if a.program == "v1" and len({g["BN"] for g in geoms}) > 1:
        raise SystemExit("--program v1 rebinds the module constant b1.BN: one BN per process (run the geometries in separate invocations)")

    if a.selftest is not None:
        a.selftest.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        rid = write_synthetic_record(a.selftest, C, LP=a.selftest_lp, steps=a.selftest_steps, layer=layer, seed=1)
        print(f"selftest record {a.selftest} request {rid} LP={a.selftest_lp} steps={a.selftest_steps} written in {time.perf_counter() - t0:.1f}s")
        rec_dir, request = a.selftest, rid
    else:
        if a.record is None:
            raise SystemExit("--record is required (or --selftest)")
        rec_dir = a.record
        request = a.request
        if request is None:
            reqs = [p.name for p in (rec_dir / "adapter" / "selective" / "ru").iterdir() if p.is_dir()]
            if len(reqs) != 1:
                raise SystemExit(f"--request required: {reqs}")
            request = reqs[0]
    install_lean_mufu_tables()
    cb = CommittedBoundary(rec_dir, request)
    print(f"record {rec_dir} request {request} profile {cb.profile_id} steps {cb.steps[0]}..{cb.steps[-1]} "
          f"prefill positions {cb.positions[cb.steps[0]][0]}..{cb.positions[cb.steps[0]][-1]}; op {a.op} layer {layer}; "
          f"config {a.config} NH={C['NH']} KVH={C['KVH']} D={C['D']}; program {a.program}", flush=True)
    results = []
    for t in ts:
        try:
            cb.step_of(t)
        except KeyError as ex:
            print(f"t={t}: skipped ({ex.args[0]})", flush=True)
            continue
        for g in geoms:
            results.append(probe_attention(cb, C, layer=layer, t=t, g=g, program=a.program, head=a.head))
    # summary: which geometry is EQUAL at every t
    print()
    print("summary (EQUAL at every tested t?):")
    for g in geoms:
        rows = [r for r in results if r["geometry"] == geometry_label(g)]
        eq = all(r["equal"] for r in rows)
        per = " ".join(f"t={r['t']}:{'EQ' if r['equal'] else str(r['differ'])}" for r in rows)
        print(f"  {geometry_label(g):<40} {'EQUAL at all t' if eq else 'NOT equal'}   [{per}]   {geometry_triple(g)}")
    print(f"peak RSS {_rss_gb():.2f} GB, now {_rss_now_mb():.0f} MB")
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps({"record": str(rec_dir), "request": request, "op": a.op, "layer": layer, "program": a.program,
                                      "config": a.config, "results": results}, indent=1, default=str))
        print(f"wrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
