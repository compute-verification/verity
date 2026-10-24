"""Sampled VUs of a passing row: the draw, the evaluation tap, the subcircuit decomposition and the flat input sets
(`vllm-vu-set/v1`) it writes.  The stored form is the hierarchical VU store (`verity_vllm.pipeline.vu_store`, which runs the draw);
the flat input sets are its default extraction.

Not on the record: the Commit stage calls `vu_store.after_replay` once pair 0's replay of record has passed and its value source
is recorded (`--vu-export-dir`, env `VERITY_VU_EXPORT_DIR`; on by default under `verity-vllm row`).  `limits.json` (the `Limits`
fields) and `provenance.json` (extra provenance: the row, the research run) in that directory are read when present.  The draw is a
stratified sample over the population the replay samples (every evaluable VU of the executed prefix, the replay's strata), keyed by
`verity.ml.randomness.derive` from the run root.  It is a benchmark selection, not a verifier challenge: fixed by the commitment and
independent of the replay's own sample, nothing more.  Each drawn VU is
evaluated exactly as the replay evaluates it (`check.replay.evaluate.replay_vu`: committed operands read through the binding map,
registered weights, the family's row kernel), with an evaluation tap recording the operands and outputs of every row it
evaluates, interior producers included.  A VU is exported only when it replays bit-exactly against its committed outputs.

Each evaluated row is decomposed into the subcircuit a backend proves, and every instance is re-evaluated under that subcircuit's
relation before it is written (a decomposition that does not reproduce the row's words is an error, never an instance):

  Gemm_v1{K,N} / Gemm_v2{K,N,DOT}  -> GemmCoordinate<K>: x[K], w[K] (one weight row), y[1]; `gemm-coordinate-<dot>-k<K>`
  ScaledMmFp8Block_v1{K,N,G}       -> ScaledMmFp8BlockCoordinate<K,G>: x[K] e4m3, sx[K/G] f32, w[K] e4m3, sw[K/G] f32 (the weight row's
                                      block scales), y[1] bf16; `gemm-coordinate-fp8-block<G>-hopper-e4m3-k<K>`
  MoeExpertGemm_v1{E,K,N}          -> GemmCoordinate<K> over a row of the routed expert: x[K], w[K], y[1];
                                      `moe-expert-gemm-coordinate-ampere-bf16-k<K>`
  MoeExpertGemmW_v1{E,K,N}         -> RoutedGemmCoordinate<K> (the routed weight g into the f32 accumulator): x[K], w[K], g[1], y[1];
                                      `moe-expert-routed-gemm-coordinate-ampere-bf16-k<K>`
  Attention_v3 / Attention_v2      -> one query head over its KV head: q[D], k[T*D], v[T*D], out[D]; `attention-head-<dot>-d<D>-bn<BN>`
  RoPE_v1{NHEADS,D}                -> one head: x[D], cs[D], out[D]; `rope-head-d<D>`
  any other evaluable family       -> the row itself: in<g> per arg group, out.<member>; `<family>-<int statics>`

`Embedding_v1` is not exported (a commitment opening of the table, not a subcircuit).

Layout of a set `<out>/sets/<name>/`: flat little-endian arrays `<port>.<u8|u16|u32|u64>`, instance-major (the
`fixtures/bench-instances/v1` convention); a port whose width varies across inputs has `<port>.offsets.u64` (n + 1 word
offsets).  `manifest.json` holds the relation, the ports, the sha256 of every file, the content digest (relation + ports + file
digests, provenance excluded) and the provenance (row, model, run, commitment root, Program and manifest digests).  `index.json`
has one row per instance (the VU it came from and the sub-index inside it).  `<out>/export.json` summarises the export and
`<out>/vus.jsonl` lists every drawn VU with its operand identities and outcome.  `verify(<out>)` re-checks every file digest and
re-evaluates every instance.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from functools import partial
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np

from verity.ml.randomness import Key, derive

from verity_vllm.check.replay import evaluate as EV
from verity_vllm.check.replay.index import ProgramIndex
from verity_vllm.check.replay.opening import CommittedStore
from verity_vllm.commit import challenge as CH
from verity.ml.definitions.numerics import derived_rows as D
from verity.ml.definitions.numerics import rows as RK
from verity_vllm.query.manifest import format as RM

SET_SCHEMA = "vllm-vu-set/v1"
EXPORT_SCHEMA = "vllm-vu-export/v1"
SELECTION_DOMAIN = "verity/vllm-vu-export/select/v1"
SELECTION_SOURCE = b"verity-vllm benchmark selection"

NOT_EXPORTED = {"Embedding_v1": "a commitment opening of the embedding table (spec), not a subcircuit"}

_DOT_SLUG = {"AmpereBF16TcDot16_v2": "ampere-bf16", "HopperBF16WgmmaDot16_v1": "hopper-bf16-wgmma", "HopperE4m3QgmmaDot32_v1": "hopper-e4m3"}
_WORD = {1: ("u8", np.uint8), 2: ("u16", np.uint16), 4: ("u32", np.uint32), 8: ("u64", np.uint64)}


@dataclasses.dataclass
class Limits:
    per_family: int = 256                    # VUs drawn per family (round-robin across the replay's strata)
    per_family_override: dict[str, int] = dataclasses.field(default_factory=dict)
    gemm_coords_per_row: int = 32            # GemmCoordinate inputs taken from one Gemm row
    heads_per_row: int = 4                   # attention / RoPE heads taken from one row
    max_instances_per_set: int = 8192
    max_row_words: int = 1 << 22             # a generic row whose operands exceed this is counted, not written
    max_seconds: float = 600.0               # evaluation budget: the draw stops (named, per family) once it is spent
    max_population_seconds: float = 900.0    # population-build budget when the replay's is not reused, checked between Programs
    workers: int = 8                         # > 1: a forked pool (host-retained stores only; a device-retained one runs serial)

    @classmethod
    def from_dir(cls, out_dir: str | Path) -> "Limits":
        f = Path(out_dir) / "limits.json"
        return cls(**json.loads(f.read_text())) if f.is_file() else cls()

    def for_family(self, fam: str) -> int:
        return int(self.per_family_override.get(fam, self.per_family))


class DecompositionMismatch(RuntimeError):
    """A subcircuit instance did not reproduce the words of the row it was cut from."""


@dataclasses.dataclass
class RowSample:
    """A GEMM weight as stored: the sampled rows (`rows`, ascending) of the [N, K] slice and their words (`data`, len(rows) x K)."""
    rows: list[int]
    data: np.ndarray


@dataclasses.dataclass
class Instance:
    set_name: str
    relation: dict[str, Any]
    ports: dict[str, np.ndarray]
    sub: dict[str, Any]


def selection_key(run_root: bytes) -> Key:
    return derive(SELECTION_SOURCE, SELECTION_DOMAIN, {"run_root": bytes(run_root)})


class _Picker:
    """`k` of `n` sub-indices of one evaluated row (coordinates, heads), drawn from the selection key at the row's own index."""

    def __init__(self, key: Key, index: tuple) -> None:
        self.key, self.index = key, index

    def __call__(self, n: int, k: int) -> list[int]:
        return list(range(n)) if k >= n else list(self.key.subset(k, n, self.index))


def select(pool: list[dict], limits: Limits, key: Key) -> list[dict]:
    """Per family: `limits.for_family` VUs, drawn per stratum (`challenge.stratum_picks` under the family's shard of `key`) and emitted
    round-robin across the family's strata, so a small cap still reaches every stratum it can.  The families are interleaved (one VU
    of each in turn), so a spent time budget truncates every family alike."""
    by_fam: dict[str, dict[str, list[dict]]] = {}
    for v in pool:
        if v["family"] in NOT_EXPORTED:
            continue
        by_fam.setdefault(v["family"], {}).setdefault(v["stratum"], []).append(v)
    per: list[list[dict]] = []
    for fam in sorted(by_fam):
        cap = limits.for_family(fam)
        fkey = key.shard(("family", fam))
        # the strata in a keyed order: in name order a cap below the stratum count takes one shape's strata first (all K=1536 of #4's GEMMs)
        names = sorted(by_fam[fam], key=lambda k: fkey.stream(("stratum-order", k), 16))
        picks = CH.stratum_picks([by_fam[fam][k] for k in names], cap, None, fkey)
        per.append(_round_robin(picks)[:cap])
    return _round_robin(per)


def _round_robin(lists: list[list[dict]]) -> list[dict]:
    return [ls[j] for j in range(max((len(ls) for ls in lists), default=0)) for ls in lists if j < len(ls)]


def _unsigned(a: np.ndarray) -> np.ndarray:
    a = np.ascontiguousarray(np.asarray(a)).reshape(-1)
    if a.dtype.kind == "u":
        return a
    if a.dtype.itemsize not in _WORD:
        raise TypeError(f"no word type for {a.dtype}")
    return a.view(_WORD[a.dtype.itemsize][1])


def _outputs(out: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """The row kernel's members with its aliases dropped (`Gemm_v1` returns the same array as `0` and `out`)."""
    seen: dict[int, str] = {}
    res = {}
    for k, v in out.items():
        if id(v) in seen:
            continue
        seen[id(v)] = k
        res[k] = _unsigned(v)
    return res


def _one(out: dict[str, np.ndarray]) -> np.ndarray:
    o = _outputs(out)
    if len(o) != 1:
        raise DecompositionMismatch(f"expected one output member, the kernel returned {sorted(o)}")
    return next(iter(o.values()))


def _gemm_dot(fam: str, st: dict[str, Any]) -> tuple[str, Callable]:
    if fam == "Gemm_v1":
        return "AmpereBF16TcDot16_v2", D.dot_bf16
    dot_id = RK.gemm_dot_prim(st)
    return dot_id, D.dot_bf16_for(dot_id)


def gemm_coordinate(dot_id: str, x: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """y = f2fp_bf16(ascending k16 chain of `dot_id` from +0) for rows w (n, K) against x (K,): (y words, non-finite lanes)."""
    acc, nf = D.dot_bf16_for(dot_id)(np.asarray(x, dtype=np.uint16)[None, :], np.asarray(w, dtype=np.uint16))
    return D.f2fp_bf16(acc).astype(np.uint16).reshape(-1), np.asarray(nf).reshape(-1)


def _decompose_gemm(fam, st, ops, out, pick, limits) -> Iterator[Instance]:
    K, N = int(st["K"]), int(st["N"])
    dot_id, _dot = _gemm_dot(fam, st)
    x = np.asarray(ops[0], dtype=np.uint16).reshape(K)
    y = _one(out).astype(np.uint16)
    if isinstance(ops[1], RowSample):                                        # a stored weight: its sampled rows are the coordinates
        sel = np.asarray(ops[1].rows, dtype=np.int64)
        wsel = ops[1].data
    else:
        sel = np.asarray(pick(N, limits.gemm_coords_per_row), dtype=np.int64)
        wsel = np.asarray(ops[1], dtype=np.uint16).reshape(N, K)[sel]
    yy, nf = gemm_coordinate(dot_id, x, wsel)
    rel = {"subcircuit": f"GemmCoordinate<{K}>", "definition": fam, "dot": dot_id, "statics": {"K": K},
           "rule": "y = f2fp_bf16(ascending k16 chain of DOT over (x, w) from +0)"}
    for j, n in enumerate(sel.tolist()):
        if nf[j]:
            continue                                                         # outside the vectorised twin's domain: not an instance
        if yy[j] != y[n]:
            raise DecompositionMismatch(f"{fam} coordinate {n}: {int(yy[j]):#06x} != row word {int(y[n]):#06x}")
        yield Instance(f"gemm-coordinate-{_DOT_SLUG.get(dot_id, dot_id.lower())}-k{K}", rel,
                       {"x": x.copy(), "w": wsel[j].copy(), "y": y[n:n + 1].copy()}, {"n": int(n), "N": N})


def _coordinate_rows(w: Any, N: int, K: int, word: type, pick: Callable[[int, int], list[int]],
                     limits: Limits) -> tuple[np.ndarray, np.ndarray]:
    """(coordinates n, their weight rows): a stored weight's sampled rows, else `gemm_coords_per_row` drawn from the whole [N, K] rows."""
    if isinstance(w, RowSample):
        return np.asarray(w.rows, dtype=np.int64), np.asarray(w.data, dtype=word)
    sel = np.asarray(pick(N, limits.gemm_coords_per_row), dtype=np.int64)
    return sel, np.asarray(w, dtype=word).reshape(N, K)[sel]


def fp8_block_coordinate(xq: np.ndarray, sx: np.ndarray, w: np.ndarray, sw: np.ndarray, G: int) -> tuple[np.ndarray, np.ndarray]:
    """`ScaledMmFp8BlockCoordinate{K,G}` for rows w (n, K) e4m3 with their block-scale rows sw (n, K/G) f32 against xq (K,), sx (K/G,):
    per K-tile ascending, `temp` = G/32 `HopperE4m3QgmmaDot32` steps from +0, `acc = FFMA(temp, FMUL(sx, sw), acc)` from +0, then
    `cvt.rn.bf16` -- `derived_rows.scaled_mm_fp8_block_row` per coordinate.  (y bf16 words, non-finite lanes)."""
    xq = np.asarray(xq, dtype=np.uint8).reshape(-1)
    K = xq.size
    w = np.asarray(w, dtype=np.uint8).reshape(-1, K)
    n = w.shape[0]
    sx = np.asarray(sx, dtype=np.uint32).reshape(K // G)
    sw = np.asarray(sw, dtype=np.uint32).reshape(n, K // G)
    acc, nf = np.zeros(n, dtype=np.uint32), np.zeros(n, dtype=bool)
    for kb in range(K // G):
        temp = np.zeros(n, dtype=np.uint32)
        for s in range(G // 32):
            lo = kb * G + 32 * s
            temp, f = D.tc_dot32_e4m3(temp, np.broadcast_to(xq[lo:lo + 32], (n, 32)), w[:, lo:lo + 32])
            nf |= f
        acc = D.fma_rn(temp, D.mul_rn(np.broadcast_to(sx[kb], (n,)), sw[:, kb]), acc)
    return D.f2fp_bf16(acc).astype(np.uint16), nf | ((acc & 0x7F800000) == 0x7F800000)


def _decompose_fp8_block(fam, st, ops, out, pick, limits) -> Iterator[Instance]:
    K, N, G = int(st["K"]), int(st["N"]), int(st["G"])
    xq = np.asarray(ops[0], dtype=np.uint8).reshape(K)
    sx = np.asarray(ops[1], dtype=np.uint32).reshape(K // G)
    ws = np.asarray(ops[3], dtype=np.uint32).reshape(-(-N // G), K // G)
    y = _one(out).astype(np.uint16)
    sel, wsel = _coordinate_rows(ops[2], N, K, np.uint8, pick, limits)
    yy, nf = fp8_block_coordinate(xq, sx, wsel, ws[sel // G], G)
    rel = {"subcircuit": f"ScaledMmFp8BlockCoordinate<{K},{G}>", "definition": fam, "dot": "HopperE4m3QgmmaDot32_v1", "statics": {"K": K, "G": G},
           "rule": "per K-tile kb ascending: temp = G/32 DOT steps from +0 over (x, w); acc = FFMA(temp, FMUL(sx[kb], sw[kb]), acc) from +0; "
                   "y = cvt.rn.bf16(acc)"}
    for j, n in enumerate(sel.tolist()):
        if nf[j]:
            continue
        if yy[j] != y[n]:
            raise DecompositionMismatch(f"{fam} coordinate {n}: {int(yy[j]):#06x} != row word {int(y[n]):#06x}")
        yield Instance(f"gemm-coordinate-fp8-block{G}-hopper-e4m3-k{K}", rel,
                       {"x": xq.copy(), "sx": sx.copy(), "w": wsel[j].copy(), "sw": ws[n // G].copy(), "y": y[n:n + 1].copy()}, {"n": int(n), "N": N})


def routed_gemm_coordinate(dot_id: str, x: np.ndarray, w: np.ndarray, g: int) -> tuple[np.ndarray, np.ndarray]:
    """y = f2fp_bf16(FMUL(ascending k16 chain of `dot_id` from +0, g)) for rows w (n, K) against x (K,): (y words, non-finite lanes)."""
    acc, nf = D.dot_bf16_for(dot_id)(np.asarray(x, dtype=np.uint16)[None, :], np.asarray(w, dtype=np.uint16))
    acc = D.mul_rn(acc, np.full(np.shape(acc), int(g) & 0xFFFFFFFF, dtype=np.uint32))
    return D.f2fp_bf16(acc).astype(np.uint16).reshape(-1), np.asarray(nf).reshape(-1)


def _moe_expert_dot(fam: str, st: dict[str, Any]) -> str:
    """The k16 step of a fused-MoE expert GEMM row: the Ampere one for the `_v1` families, the DOT static of the `_v2` ones."""
    return RK.prim_static(st, "DOT", fam) if fam.endswith("_v2") else "AmpereBF16TcDot16_v2"


def _decompose_moe_expert(fam, st, ops, out, pick, limits, *, routed: bool) -> Iterator[Instance]:
    E, K, N = int(st["E"]), int(st["K"]), int(st["N"])
    dot_id = _moe_expert_dot(fam, st)
    x = np.asarray(ops[0], dtype=np.uint16).reshape(K)
    e = int(np.asarray(ops[1]).reshape(-1)[0])
    w = ops[3 if routed else 2]
    if not isinstance(w, RowSample):
        w = np.asarray(w, dtype=np.uint16).reshape(E, N, K)[e]                   # the gather by the committed router id
    sel, wsel = _coordinate_rows(w, N, K, np.uint16, pick, limits)
    y = _one(out).astype(np.uint16)
    if routed:
        g = _unsigned(np.asarray(ops[2]).reshape(-1)[:1])
        yy, nf = routed_gemm_coordinate(dot_id, x, wsel, int(g[0]))
        rel = {"subcircuit": f"RoutedGemmCoordinate<{K}>", "definition": fam, "dot": dot_id, "statics": {"K": K},
               "rule": "y = f2fp_bf16(FMUL(ascending k16 chain of DOT over (x, w) from +0, g))"}
        name = f"moe-expert-routed-gemm-coordinate-{_DOT_SLUG[dot_id]}-k{K}"
    else:
        yy, nf = gemm_coordinate(dot_id, x, wsel)
        rel = {"subcircuit": f"GemmCoordinate<{K}>", "definition": fam, "dot": dot_id, "statics": {"K": K},
               "rule": "y = f2fp_bf16(ascending k16 chain of DOT over (x, w) from +0)"}
        name = f"moe-expert-gemm-coordinate-{_DOT_SLUG[dot_id]}-k{K}"
    for j, n in enumerate(sel.tolist()):
        if nf[j]:
            continue
        if yy[j] != y[n]:
            raise DecompositionMismatch(f"{fam} expert {e} coordinate {n}: {int(yy[j]):#06x} != row word {int(y[n]):#06x}")
        ports = {"x": x.copy(), "w": wsel[j].copy(), "y": y[n:n + 1].copy()}
        if routed:
            ports["g"] = g.copy()
        yield Instance(name, rel, ports, {"n": int(n), "N": N, "expert": e, "E": E})


def _attn_prims(fam: str, st: dict[str, Any]) -> tuple[str, str]:
    if fam == "Attention_v2":
        return RK.prim_static(st, "DOT", fam), RK.prim_static(st, "INV", fam)
    return "AmpereBF16TcDot16_v2", "Fa2InvSum_v1"


def attention_head(dot_id: str, inv_id: str, BN: int, q: np.ndarray, k: np.ndarray, v: np.ndarray, Dh: int) -> tuple[np.ndarray, bool]:
    """One head: q (D,), k / v (T*D,) -> (out (D,) bf16 words, non-finite)."""
    T = int(np.asarray(k).size) // Dh
    o, nf = D.attention_rows(np.asarray(q, dtype=np.uint16).reshape(1, Dh), np.asarray(k, dtype=np.uint16).reshape(T, 1, Dh),
                             np.asarray(v, dtype=np.uint16).reshape(T, 1, Dh), BN, dot_prim=dot_id, inv_prim=inv_id)
    return np.asarray(o, dtype=np.uint16).reshape(Dh), bool(np.asarray(nf).any())


def _decompose_attention(fam, st, ops, out, pick, limits) -> Iterator[Instance]:
    T, NH, KVH, Dh, BN = (int(st[k]) for k in ("T", "NH", "KVH", "D", "BN"))
    dot_id, inv_id = _attn_prims(fam, st)
    q = np.asarray(ops[0], dtype=np.uint16).reshape(NH, Dh)
    kc = np.asarray(ops[1], dtype=np.uint16).reshape(T, KVH, Dh)
    vc = np.asarray(ops[2], dtype=np.uint16).reshape(T, KVH, Dh)
    o = _one(out).astype(np.uint16).reshape(NH, Dh)
    rel = {"subcircuit": f"AttentionHead<T,{Dh}>", "definition": fam, "dot": dot_id, "inv": inv_id, "statics": {"D": Dh, "BN": BN},
           "rule": "one query at position T-1 over keys 0..T-1 of its KV head, ceil(T/BN) key blocks last -> first (the Definition's head)"}
    name = (f"attention-head-fa2-d{Dh}-bn{BN}" if fam == "Attention_v3"
            else f"attention-head-{_DOT_SLUG.get(dot_id, dot_id.lower())}-{inv_id.lower()}-d{Dh}-bn{BN}")
    for h in pick(NH, limits.heads_per_row):
        g = h // (NH // KVH)
        ports = {"q": q[h].copy(), "k": kc[:, g].reshape(-1).copy(), "v": vc[:, g].reshape(-1).copy(), "out": o[h].copy()}
        got, nf = attention_head(dot_id, inv_id, BN, ports["q"], ports["k"], ports["v"], Dh)
        if nf:
            continue
        if not np.array_equal(got, ports["out"]):
            raise DecompositionMismatch(f"{fam} head {h}: {int(np.count_nonzero(got != ports['out']))} words differ from the row")
        yield Instance(name, rel, ports, {"head": int(h), "kv_head": int(g), "T": T, "NH": NH, "KVH": KVH})


def _decompose_rope(fam, st, ops, out, pick, limits) -> Iterator[Instance]:
    NH, Dh = int(st["NHEADS"]), int(st["D"])
    x = np.asarray(ops[0], dtype=np.uint16).reshape(NH, Dh)
    cs = np.asarray(ops[1], dtype=np.uint16).reshape(Dh)
    o = _one(out).astype(np.uint16).reshape(NH, Dh)
    rel = {"subcircuit": f"RoPEHead<{Dh}>", "definition": fam, "statics": {"D": Dh},
           "rule": "out[i] = bf16_rn(RN32(x[i]*c[i] - x[i+R]*s[i])), out[i+R] = bf16_rn(RN32(x[i+R]*c[i] + x[i]*s[i])), cs = [c | s], R = D/2"}
    for h in pick(NH, limits.heads_per_row):
        got = D.rope_rows(x[h:h + 1], cs).reshape(Dh)
        if not np.array_equal(got, o[h]):
            raise DecompositionMismatch(f"{fam} head {h}: differs from the row")
        yield Instance(f"rope-head-d{Dh}", rel, {"x": x[h].copy(), "cs": cs.copy(), "out": o[h].copy()}, {"head": int(h), "NHEADS": NH})


def _row_set_name(fam: str, st: dict[str, Any]) -> str:
    ints = "-".join(f"{k.lower()}{v}" for k, v in sorted(st.items()) if isinstance(v, int))
    other = {k: v for k, v in st.items() if not isinstance(v, int)}
    tag = hashlib.sha256(json.dumps(other, sort_keys=True).encode()).hexdigest()[:8] if other else ""
    return "-".join(p for p in (fam.lower().replace("_", "-"), ints, tag) if p)


def _decompose_row(fam, st, ops, out, pick, limits) -> Iterator[Instance]:
    ports = {f"in{g}": _unsigned(a).copy() for g, a in enumerate(ops)}
    ports.update({f"out.{k}": v.copy() for k, v in _outputs(out).items()})
    if sum(int(a.size) for a in ports.values()) > limits.max_row_words:
        return
    rel = {"subcircuit": fam, "definition": fam, "statics": st, "rule": f"the registered row kernel of {fam} (verity.ml.definitions.numerics.rows)"}
    yield Instance(_row_set_name(fam, st), rel, ports, {})


DECOMPOSE: dict[str, Callable[..., Iterator[Instance]]] = {"Gemm_v1": _decompose_gemm, "Gemm_v2": _decompose_gemm,
                                                           "ScaledMmFp8Block_v1": _decompose_fp8_block,
                                                           "MoeExpertGemm_v1": partial(_decompose_moe_expert, routed=False),
                                                           "MoeExpertGemmW_v1": partial(_decompose_moe_expert, routed=True),
                                                           "MoeExpertGemm_v2": partial(_decompose_moe_expert, routed=False),
                                                           "MoeExpertGemmW_v2": partial(_decompose_moe_expert, routed=True),
                                                           "Attention_v3": _decompose_attention, "Attention_v2": _decompose_attention,
                                                           "RoPE_v1": _decompose_rope}


def _dense_rows(g: int, word: type) -> Callable[[dict[str, Any], list], tuple[np.ndarray, dict[str, Any]]]:
    return lambda st, ops: (np.asarray(ops[g], dtype=word).reshape(int(st["N"]), int(st["K"])), {})


def _expert_rows(g: int) -> Callable[[dict[str, Any], list], tuple[np.ndarray, dict[str, Any]]]:
    def rows(st: dict[str, Any], ops: list) -> tuple[np.ndarray, dict[str, Any]]:
        e = int(np.asarray(ops[1]).reshape(-1)[0])
        return np.asarray(ops[g], dtype=np.uint16).reshape(int(st["E"]), int(st["N"]), int(st["K"]))[e], {"expert": e}
    return rows


#: beside `DECOMPOSE`, keyed alike: the weight operand a coordinate cut reads rows of -> (operand index, its [N, K] rows and extra
#: reference fields); the store keeps a fixed sample of those rows and the whole operand by checkpoint reference
WEIGHT_ROWS_OF: dict[str, tuple[int, Callable[[dict[str, Any], list], tuple[np.ndarray, dict[str, Any]]]]] = {
    "Gemm_v1": (1, _dense_rows(1, np.uint16)), "Gemm_v2": (1, _dense_rows(1, np.uint16)), "ScaledMmFp8Block_v1": (2, _dense_rows(2, np.uint8)),
    "MoeExpertGemm_v1": (2, _expert_rows(2)), "MoeExpertGemmW_v1": (3, _expert_rows(3)),
    "MoeExpertGemm_v2": (2, _expert_rows(2)), "MoeExpertGemmW_v2": (3, _expert_rows(3))}


def _gemm_template(fam: str, st: dict[str, Any]) -> tuple[str, dict[str, Any], int]:
    return "GemmCoordinate", {"K": st["K"], "dot": _gemm_dot(fam, st)[0]}, int(st["N"])


def _fp8_block_template(fam: str, st: dict[str, Any]) -> tuple[str, dict[str, Any], int]:
    return "ScaledMmFp8BlockCoordinate", {"K": st["K"], "G": st["G"], "dot": "HopperE4m3QgmmaDot32_v1"}, int(st["N"])


def _moe_expert_template(template: str, fam: str, st: dict[str, Any]) -> tuple[str, dict[str, Any], int]:
    return template, {"K": st["K"], "dot": _moe_expert_dot(fam, st)}, int(st["N"])


def _attention_template(fam: str, st: dict[str, Any]) -> tuple[str, dict[str, Any], int]:
    dot, inv = _attn_prims(fam, st)
    return "AttentionHead", {"D": st["D"], "BN": st["BN"], "dot": dot, "inv": inv}, int(st["NH"])


def _rope_template(fam: str, st: dict[str, Any]) -> tuple[str, dict[str, Any], int]:
    return "RoPEHead", {"D": st["D"]}, int(st["NHEADS"])


#: beside `DECOMPOSE`, keyed alike: the template a Definition's Calls are cut into (every other Definition: the row itself)
TEMPLATE_OF: dict[str, Callable[[str, dict[str, Any]], tuple[str, dict[str, Any], int]]] = {
    "Gemm_v1": _gemm_template, "Gemm_v2": _gemm_template, "ScaledMmFp8Block_v1": _fp8_block_template,
    "MoeExpertGemm_v1": partial(_moe_expert_template, "GemmCoordinate"), "MoeExpertGemmW_v1": partial(_moe_expert_template, "RoutedGemmCoordinate"),
    "MoeExpertGemm_v2": partial(_moe_expert_template, "GemmCoordinate"), "MoeExpertGemmW_v2": partial(_moe_expert_template, "RoutedGemmCoordinate"),
    "Attention_v3": _attention_template, "Attention_v2": _attention_template, "RoPE_v1": _rope_template}


def template_of(spec: str) -> tuple[str, dict[str, Any], int]:
    """A Call's Definition spec -> (subcircuit template, bound parameters, instances per Call): `decompose`'s cut, read off the spec."""
    fam, st = RM.spec_family(spec), RM.spec_statics(spec)
    t = TEMPLATE_OF.get(fam)
    try:
        if t is not None:
            return t(fam, st)
    except (KeyError, RK.Declined):
        pass
    return fam, st, 1


def decompose(fam: str, st: dict[str, Any], ops: list[np.ndarray], out: dict[str, np.ndarray], pick: Callable[[int, int], list[int]],
              limits: Limits) -> list[Instance]:
    if fam in NOT_EXPORTED:
        return []
    return list(DECOMPOSE.get(fam, _decompose_row)(fam, st, ops, out, pick, limits))


def capture(pi: ProgramIndex, store: CommittedStore, weights: Callable, vu: dict, *, arrive: int) -> tuple[dict, list[tuple]]:
    """The replay's evaluation of one VU, with every row evaluation it performs (interior producers first, the VU's own row last):
    (family, statics, operands, outputs, weight refs)."""
    evals: list[tuple] = []
    tap = lambda fam, st, ops, out, refs: evals.append((fam, dict(st), ops, out, refs))  # noqa: E731
    EV.EVALUATION_TAPS.append(tap)
    try:
        rec = EV.replay_vu(pi, store, weights, vu, arrive=arrive)
    finally:
        EV.EVALUATION_TAPS.remove(tap)
    return rec, evals


class _SetWriter:
    def __init__(self, name: str, relation: dict[str, Any]) -> None:
        self.name, self.relation = name, relation
        self.ports: dict[str, list[np.ndarray]] = {}
        self.index: list[list[Any]] = []

    def add(self, inst: Instance, ref: list[Any]) -> None:
        if self.ports and set(inst.ports) != set(self.ports):
            raise DecompositionMismatch(f"set {self.name}: ports {sorted(inst.ports)} != {sorted(self.ports)}")
        for k, a in inst.ports.items():
            self.ports.setdefault(k, []).append(_unsigned(a))
        self.index.append([len(self.index), *ref, inst.sub])

    def write(self, root: Path, provenance: dict[str, Any]) -> dict[str, Any]:
        d = root / self.name
        d.mkdir(parents=True, exist_ok=True)
        files: dict[str, dict[str, Any]] = {}
        ports: dict[str, dict[str, Any]] = {}

        def put(fname: str, arr: np.ndarray) -> None:
            b = np.ascontiguousarray(arr).astype(arr.dtype.newbyteorder("<"), copy=False).tobytes()
            (d / fname).write_bytes(b)
            files[fname] = {"sha256": hashlib.sha256(b).hexdigest(), "bytes": len(b)}

        for k in sorted(self.ports):
            arrs = self.ports[k]
            dt = {a.dtype for a in arrs}
            if len(dt) != 1:
                raise DecompositionMismatch(f"set {self.name} port {k}: word types {sorted(map(str, dt))}")
            wname = _WORD[arrs[0].dtype.itemsize][0]
            sizes = {int(a.size) for a in arrs}
            put(f"{k}.{wname}", np.concatenate(arrs))
            p: dict[str, Any] = {"dtype": wname, "file": f"{k}.{wname}", "words": sizes.pop() if len(sizes) == 1 else None}
            if p["words"] is None:
                offs = np.concatenate([[0], np.cumsum([a.size for a in arrs])]).astype(np.uint64)
                put(f"{k}.offsets.u64", offs)
                p["offsets"] = f"{k}.offsets.u64"
            ports[k] = p
        idx = {"schema": f"{SET_SCHEMA}/index", "set": self.name,
               "columns": ["id", "vu", "request", "engine_step", "op_path", "row", "phase", "source", "sub"], "rows": self.index}
        ib = (json.dumps(idx, separators=(",", ":")) + "\n").encode()
        (d / "index.json").write_bytes(ib)
        files["index.json"] = {"sha256": hashlib.sha256(ib).hexdigest(), "bytes": len(ib)}
        body = {"schema": SET_SCHEMA, "set": self.name, "relation": self.relation, "n": len(self.index), "ports": ports,
                "files": {k: v for k, v in files.items() if k != "index.json"}}
        content = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()
        man = dict(body, files=files, content_digest=content, provenance=provenance,
                   content_digest_rule="sha256 of the canonical JSON of {schema, set, relation, n, ports, files without index.json}")
        (d / "manifest.json").write_text(json.dumps(man, indent=1, sort_keys=True, default=str) + "\n")
        return {"set": self.name, "n": len(self.index), "bytes": sum(v["bytes"] for v in files.values()), "content_digest": content,
                "relation": self.relation.get("subcircuit")}


def _read_port(d: Path, p: dict[str, Any], n: int) -> list[np.ndarray]:
    arr = np.frombuffer((d / p["file"]).read_bytes(), dtype=np.dtype(_WORD[int(p["dtype"][1:]) // 8][1]).newbyteorder("<"))
    if p.get("words") is not None:
        return list(arr.reshape(n, int(p["words"])))
    offs = np.frombuffer((d / p["offsets"]).read_bytes(), dtype="<u8").astype(np.int64)
    return [arr[offs[i]:offs[i + 1]] for i in range(n)]


def _check_gemm(rel: dict[str, Any], P: dict[str, list[np.ndarray]], i: int) -> bool:
    y, _nf = gemm_coordinate(rel["dot"], P["x"][i], P["w"][i][None, :])
    return int(y[0]) == int(P["y"][i][0])


def _check_fp8_block(rel: dict[str, Any], P: dict[str, list[np.ndarray]], i: int) -> bool:
    y, _nf = fp8_block_coordinate(P["x"][i], P["sx"][i], P["w"][i][None, :], P["sw"][i][None, :], int(rel["statics"]["G"]))
    return int(y[0]) == int(P["y"][i][0])


def _check_routed_gemm(rel: dict[str, Any], P: dict[str, list[np.ndarray]], i: int) -> bool:
    y, _nf = routed_gemm_coordinate(rel["dot"], P["x"][i], P["w"][i][None, :], int(P["g"][i][0]))
    return int(y[0]) == int(P["y"][i][0])


def _check_attention(rel: dict[str, Any], P: dict[str, list[np.ndarray]], i: int) -> bool:
    got, _nf = attention_head(rel["dot"], rel["inv"], int(rel["statics"]["BN"]), P["q"][i], P["k"][i], P["v"][i], int(rel["statics"]["D"]))
    return bool(np.array_equal(got, P["out"][i]))


def _check_rope(rel: dict[str, Any], P: dict[str, list[np.ndarray]], i: int) -> bool:
    return bool(np.array_equal(D.rope_rows(P["x"][i][None, :], P["cs"][i]).reshape(-1), P["out"][i]))


def _check_row(rel: dict[str, Any], P: dict[str, list[np.ndarray]], i: int) -> bool:
    ins = sorted((k for k in P if k.startswith("in")), key=lambda k: int(k[2:]))
    got = _outputs(EV.evaluate(rel["definition"], rel["statics"], [[P[k][i]] for k in ins], [None] * len(ins)))
    return all(k[4:] in got and np.array_equal(got[k[4:]].astype(P[k][i].dtype), P[k][i]) for k in P if k.startswith("out."))


#: a set's relation check, by the ports its decomposer declared (the set's own layout, never its Definition's name)
_CHECK_BY_PORTS: dict[frozenset[str], Callable[[dict[str, Any], dict[str, list[np.ndarray]], int], bool]] = {
    frozenset({"x", "w", "y"}): _check_gemm, frozenset({"x", "sx", "w", "sw", "y"}): _check_fp8_block, frozenset({"x", "w", "g", "y"}): _check_routed_gemm,
    frozenset({"q", "k", "v", "out"}): _check_attention, frozenset({"x", "cs", "out"}): _check_rope}


def verify_set(d: Path) -> dict[str, Any]:
    """Every file digest against the manifest, then every instance re-evaluated under the set's relation."""
    man = json.loads((d / "manifest.json").read_text())
    for fname, f in man["files"].items():
        if hashlib.sha256((d / fname).read_bytes()).hexdigest() != f["sha256"]:
            return {"set": man["set"], "ok": False, "why": f"{fname}: sha256 differs from the manifest"}
    n, rel = int(man["n"]), man["relation"]
    P = {k: _read_port(d, p, n) for k, p in man["ports"].items()}
    idx = json.loads((d / "index.json").read_text())["rows"]
    check = _CHECK_BY_PORTS.get(frozenset(P), _check_row)
    bad = [idx[i][0] for i in range(n) if not check(rel, P, i)]
    return {"set": man["set"], "ok": not bad, "n": n, "bad": bad[:16], "bad_n": len(bad)}


def verify(out_dir: str | Path) -> dict[str, Any]:
    root = Path(out_dir) / "sets"
    res = [verify_set(d) for d in sorted(root.iterdir()) if (d / "manifest.json").is_file()]
    return {"ok": all(r["ok"] for r in res) and bool(res), "sets": res}
