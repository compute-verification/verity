"""Pearl-C (`pearl-c-h100-v1`) at the linears' ``quant_method.apply``: the CPU half of the executor.

The plan is the project store's ``docs/pouw/pearl-c-vllm-plan.md``.  This module is the part that runs without a GPU:

* `Split`, `split_channels`, `split_layer`, `split_check`: the channel split (the cheap-binding lane's recommendation).
  - Per layer, a fixed set S of at most ⌊k/100⌋ input channels goes to a BF16 side GEMM outside the unit.
  - S comes from calibration activations by the lane's `split_channels`, the one rule, with its tie-break and budget
    pinned. `split_check` is the verifier's S check, by the same rule.
  - The unit reads the kept channels, zero-padded to a multiple of 128. Rows that still fail the scheme's exact rules are
    excluded rows (`pearl_c_work`'s per-row rule).
* `twin_call`: one call's unit as #449's kernels compute it, on the CPU:
  - §5's forming, its noise atoms batched;
  - the chains and the peel, batched with `verity.ml.kernels.group_sum_total_batch`;
  - each 64 x 64 tile's leaf and Pearl's tickets;
  - the unscaled y part, RN_f32(RN_f32(U·(1/α_A))·(1/α_B)).
  The tests hold it to `pearl_c`'s scalar reference.  It is the host path and the device gate's oracle.
* `PearlCRun`: `pouw._Run`'s interface (`gemm`, `finish`, `commits`, `details`) over the twin.  Per call it widens x to FP32,
  splits it, pads m to a multiple of 128 with zero filler rows, records the rows that fail the rules as excluded, commits
  the unit's rows, and serves
  y = RN_bf16(RN_f32(y_unit + y_side) + bias).

It is not wired into `pouw.py` yet.  That takes one `EXECUTED` entry and one branch in `install`, after the hashing-cut work
on #435, which edits the same file.
"""
from __future__ import annotations

import hashlib
import json
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from verity_pouw.audit import (TILE_SCHEMA, Epoch, Workload, activation_binding, commit_rows, transcript_domain,
                               weight_binding, weights_root)
from verity_pouw.protocol import Context as UnitContext
from verity_pouw.protocol import Matmul, Shape
from verity_pouw.schemes import pearl_c as C
from verity_pouw.schemes import pearl_c_u as U
from verity_pouw.schemes import pearl_c_work as PW
from verity_pouw.schemes import pearl_kw as P

from verity.commitments import MerkleTree
from verity.ml.kernels import group_sum_total_batch
from verity.ml.tc.models import HOPPER_BF16_WGMMA_K16, HOPPER_E4M3_K32

SCHEME = C.NAME_V1
#: the kernels' granularity in m, n and the unit's k
ALIGN = PW.ALIGN
CALIBRATION_DOMAIN = b"verity/pouw/pearl-c/calibration/v1\0"
#: the split rule's batch: channels moved per step
BATCH = 4


# ---- the channel split --------------------------------------------------------------------------------------------

def row_ok(words: Sequence[int] | np.ndarray) -> bool:
    """The scheme's per-row rules on one row of FP32 words (`pearl_c_work.row_passes`: the noise floor ρ·α ≥ 1 and TT_OUT's
    liveness on the exact every-8th statistic)."""
    return PW.row_passes([int(v) for v in np.asarray(words, np.uint32)])


@dataclass(frozen=True)
class Split(PW.Split):
    """One layer's split (`pearl_c_work.Split`: |S| ≤ ⌊k/100⌋, the calibration digest), with the executor's numpy views."""

    @property
    def kept(self) -> np.ndarray:
        mask = np.ones(self.k, bool)
        mask[list(self.side)] = False
        return np.flatnonzero(mask)

    def unit_words(self, words: np.ndarray) -> np.ndarray:
        """The unit's rows: the kept channels' FP32 words, zero-padded to `k_unit`."""
        out = np.zeros((words.shape[0], self.k_unit), np.uint32)
        out[:, :self.k_kept] = words[:, self.kept]
        return out

    def side_values(self, values: np.ndarray) -> np.ndarray:
        return values[:, list(self.side)]

    def digest(self) -> bytes:
        return hashlib.sha256(self.encode()).digest()


def rules(X: np.ndarray, delta: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per row: floor ok, live ok, σ, and s/ρ, on float32 rows.  The cheap-binding lane's, verbatim."""
    k = X.shape[1]
    s = np.abs(X).max(1)
    sig = delta * np.sqrt((X[:, ::8].astype(np.float64) ** 2).mean(1))
    alpha = np.where(s + 4 * sig > 0, 448.0 / np.maximum(s + 4 * sig, 1e-300), 0.0)
    floor = sig * alpha >= 1.0
    live = (np.abs(X) >= 8 * sig[:, None]).sum(1) * 64 <= k
    return floor, live, sig, s / np.maximum(sig, 1e-300)


def split_budget(k: int) -> int:
    """The rule's budget: a step moves up to `BATCH` channels once fewer than the budget are moved, so ⌊k/100⌋ − 3 keeps
    |S| ≤ ⌊k/100⌋."""
    return max(k // 100 - (BATCH - 1), 0)


def split_channels(X: np.ndarray, W: np.ndarray, delta: float, budget: int, batch: int = BATCH) -> tuple[list[int], list[dict]]:
    """Greedy: move the input channels that offend the most failing rows (entries at or above 8σ of a row failing liveness,
    or above 444ρ of a row failing the floor) until every row passes or the budget is spent.

    The cheap-binding lane's `split_channels`, the one split rule, with its tie-break pinned: channels are ranked by
    descending offence count and, among equal counts, by descending channel index.  The lane's `np.argsort(-off)` left
    ties to numpy's unstable sort, whose order depends on its SIMD dispatch; on the machines both lanes ran this rule gives
    the lane's result (Qwen2.5-7B's layer-2 `down_proj`: 38 channels, the same trace)."""
    k = X.shape[1]
    keep = np.ones(k, dtype=bool)
    S: list[int] = []
    trace = []
    while True:
        idx = np.flatnonzero(keep)
        Xk, Wk = X[:, idx], W[:, idx]
        off = np.zeros(len(idx))
        fails = 0
        for M in (Xk, Wk):
            fl, lv, sig, _ = rules(M, delta)
            bad = ~(fl & lv)
            fails += int(bad.sum())
            if bad.any():
                Mb, sb = np.abs(M[bad]), sig[bad][:, None]
                off += ((~fl[bad][:, None] & (Mb > 444 * sb)) | (~lv[bad][:, None] & (Mb >= 8 * sb))).sum(0)
        trace.append({"moved": len(S), "failing_rows": fails})
        if fails == 0 or len(S) >= budget:
            return S, trace
        top = idx[np.lexsort((-np.arange(len(idx)), -off))[:batch]]
        top = [int(c) for c in top if off[np.searchsorted(idx, c)] > 0] or [int(idx[np.argmax(off)])]
        for c in top:
            keep[c] = False
            S.append(c)


def calibration_digest(x: np.ndarray) -> bytes:
    """The calibration rows' digest (tokens x k, FP32), which the split records."""
    return hashlib.sha256(CALIBRATION_DOMAIN + struct.pack("<II", *x.shape) + np.ascontiguousarray(x, "<f4").tobytes()).digest()


def split_layer(x: np.ndarray, w: np.ndarray) -> Split:
    """One layer's split from its calibration rows x and its weight w (FP32 values), by `split_channels`."""
    side, _ = split_channels(x, w, 1.0, split_budget(x.shape[1]))
    return Split(x.shape[1], tuple(sorted(side)), calibration_digest(x))


def split_check(calibration: Mapping[int, np.ndarray], weights: Mapping[int, np.ndarray]):
    """The verifier's S check for `pearl_c_work.audit`: S is `split_layer`'s on the committed calibration rows and weight."""
    def check(w: int, split: PW.Split) -> bool:
        honest = split_layer(calibration[w], weights[w])
        return (split.k, split.side, split.calibration) == (honest.k, honest.side, honest.calibration)
    return check


# ---- the CPU twin -------------------------------------------------------------------------------------------------

def _f32(words: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(words, np.uint32).view(np.float32)


def chain_np(a: np.ndarray, b: np.ndarray, ii: np.ndarray, jj: np.ndarray, block: int = 1 << 16) -> np.ndarray:
    """`pearl_c.chain` for the cells (ii, jj) of code matrices a and b, in blocks of cells."""
    out = np.empty(len(ii), np.uint32)
    k = a.shape[1]
    for lo in range(0, len(ii), block):
        ci, cj = ii[lo:lo + block], jj[lo:lo + block]
        total = np.zeros(len(ci), np.uint32)
        for ws in range(0, k, C.WINDOW):
            local = np.zeros(len(ci), np.uint32)
            for s in range(ws, min(ws + C.WINDOW, k), C.SLICE):
                local = group_sum_total_batch(HOPPER_E4M3_K32, local, a[ci, s:s + C.SLICE], b[cj, s:s + C.SLICE])[0]
            total = (_f32(total) + _f32(local)).view(np.uint32)
        total[total == 0x80000000] = 0
        out[lo:lo + block] = total
    return out


def peel_np(c: np.ndarray, pa: np.ndarray, pb: np.ndarray, ii: np.ndarray, jj: np.ndarray) -> np.ndarray:
    acc = c.copy()
    for s in range(0, U.PEEL_K, 16):
        acc = group_sum_total_batch(HOPPER_BF16_WGMMA_K16, acc, pa[ii, s:s + 16], pb[jj, s:s + 16])[0]
    return acc


@dataclass
class Side:
    """One operand side formed by v1, as the kernels form it: its codes, α and β_row, E′ and the input words."""

    codes: np.ndarray            # rows x k, E4M3
    alpha: np.ndarray            # FP32 words
    beta: np.ndarray
    rho: np.ndarray
    ep: np.ndarray               # rows x 32, E′ codes
    words: np.ndarray            # rows x k, FP32 input

    def formed(self) -> C.Formed:
        """The side as `pearl_c.Formed`, for the scheme's `admissible`."""
        rows = lambda m: [[int(v) for v in r] for r in m]  # noqa: E731
        return C.Formed(rows(self.codes), [int(v) for v in self.alpha], [int(v) for v in self.beta], [0] * len(self.alpha),
                        [int(v) for v in self.rho], rows(self.ep), rows(self.words))


def form(words: np.ndarray, e_rows: Sequence[Sequence[int]], basis: Sequence[Sequence[int]], block: int = 1 << 18) -> Side:
    """`pearl_c.form_v1` with the noise atoms batched: per row the pinned statistics and E′, then per element one
    HOPPER_E4M3_K32 step with α·x as its accumulator, then the cast."""
    m, k = words.shape
    alpha, beta, rho = (np.zeros(m, np.uint32) for _ in range(3))
    ep = np.zeros((m, U.R), np.int64)
    for i in range(m):
        _, rho[i], alpha[i], beta[i] = C.s5_stats([int(v) for v in words[i]])
        ep[i] = C.eprime(int(beta[i]), e_rows[i])
    acc = (_f32(words) * _f32(alpha)[:, None]).view(np.uint32).ravel()
    f = np.asarray(basis, np.int64)
    ii, tt = np.divmod(np.arange(m * k), k)
    d = np.empty(m * k, np.uint32)
    for lo in range(0, m * k, block):
        sl = slice(lo, lo + block)
        d[sl] = group_sum_total_batch(HOPPER_E4M3_K32, acc[sl], ep[ii[sl]], f[tt[sl]])[0]
    codes = np.array([P.f32_to_fp8(P.f32_of_bits(int(w))) for w in d], np.uint8).reshape(m, k)
    return Side(codes, alpha, beta, rho, ep.astype(np.uint8), words)


@dataclass
class Unit:
    """One call's unit: the formed sides, the peel factors, the tiles' C̃ and U words, their leaves and tickets, and y_unit."""

    a: Side
    b: Side
    pa: np.ndarray               # m x 64 BF16
    pb: np.ndarray               # n x 64 BF16
    c: np.ndarray                # m x n FP32 words (0 outside the computed tiles)
    u: np.ndarray
    leaves: dict[tuple[int, int], bytes]
    tickets: dict[tuple[int, int], list[bytes]]
    y_unit: np.ndarray           # m x n FP32 words


@dataclass
class WeightSide:
    """What a weight's calls share within an epoch: its formed side's sketches (B̃·F_A, P_B) and the bases.  B̃ itself is
    formed per call (the deployment audit's A1)."""

    fa: tuple
    fb: tuple
    e_b: list[list[int]]
    pb: np.ndarray


def weight_side(scheme: C.PearlC, noise: C.Noise, b: Side, e_b: list[list[int]], k: int) -> WeightSide:
    fa, fb = C._basis(noise.seed_b, 0, k, scheme.line_norm), C._basis(noise.seed_b, 1, k, scheme.line_norm)
    fa_t = np.asarray(fa, np.int64).T.copy()
    fb_t = np.asarray(fb, np.int64).T.copy()
    n = b.codes.shape[0]
    ii, jj = np.divmod(np.arange(n * U.R), U.R)
    bf = chain_np(b.codes.astype(np.int64), fa_t, ii, jj).reshape(n, U.R)
    gi, gj = np.divmod(np.arange(U.R * U.R), U.R)
    g = chain_np(fb_t, fa_t, gi, gj).reshape(U.R, U.R)
    be = [[C._bf16_of_code(int(c)) for c in row] for row in b.ep]
    pb = np.array([U.pb_row(be[j], [int(v) for v in bf[j]], g.tolist()) for j in range(n)], np.int64)
    return WeightSide(fa, fb, e_b, pb)


def twin_call(scheme: C.PearlC, ctx: UnitContext, noise: C.Noise, a_words: np.ndarray, b_words: np.ndarray,
              ws: WeightSide | None = None, tiles: Sequence[tuple[int, int]] | None = None) -> tuple[Unit, WeightSide]:
    """One call's unit on the CPU.  `tiles` names the 64 x 64 tiles (first row, first column) to compute, all by default."""
    m, k = a_words.shape
    n = b_words.shape[0]
    e_a = [P.sample_line(noise.seed_a, 0, 0, i) for i in range(m)]
    e_b = [P.sample_line(noise.seed_b, 1, 0, j) for j in range(n)]
    fa = C._basis(noise.seed_b, 0, k, scheme.line_norm)
    fb = C._basis(noise.seed_b, 1, k, scheme.line_norm)
    a = form(a_words, e_a, fa)
    b = form(b_words, e_b, fb)
    if ws is None:
        ws = weight_side(scheme, noise, b, e_b, k)
    fb_t = np.asarray(fb, np.int64).T.copy()
    ii, jj = np.divmod(np.arange(m * U.R), U.R)
    af = chain_np(a.codes.astype(np.int64), fb_t, ii, jj).reshape(m, U.R)
    be_a = [[C._bf16_of_code(int(c)) for c in row] for row in a.ep]
    pa = np.array([U.pa_row(be_a[i], [int(v) for v in af[i]]) for i in range(m)], np.int64)
    grid = [(t.rows[0], t.cols[0]) for t in scheme.tiles(Shape(m, k, n))]
    want = grid if tiles is None else list(tiles)
    c, u = np.zeros((m, n), np.uint32), np.zeros((m, n), np.uint32)
    leaves, tickets = {}, {}
    th, tw = scheme.tile
    for r0, c0 in want:
        ti, tj = np.meshgrid(np.arange(r0, r0 + th), np.arange(c0, c0 + tw), indexing="ij")
        ti, tj = ti.ravel(), tj.ravel()
        cw = chain_np(a.codes.astype(np.int64), b.codes.astype(np.int64), ti, tj)
        uw = peel_np(cw, pa, ws.pb, ti, tj)
        c[ti, tj], u[ti, tj] = cw, uw
        checked = C.Checked(c[r0:r0 + th, c0:c0 + tw].tolist(), u[r0:r0 + th, c0:c0 + tw].tolist())
        leaves[(r0, c0)] = C.leaf(checked)
        tickets[(r0, c0)] = scheme.tickets(noise, checked)
    ia = (np.float32(1.0) / _f32(a.alpha))
    ib = (np.float32(1.0) / _f32(b.alpha))
    y_unit = ((_f32(u) * ia[:, None]).astype(np.float32) * ib[None, :]).astype(np.float32).view(np.uint32)
    return Unit(a, b, pa, ws.pb, c, u, leaves, tickets, y_unit), ws


def merge(y_unit: np.ndarray, y_side: np.ndarray | None, bias_bits: np.ndarray | None) -> np.ndarray:
    """Served y as BF16 words: RN_bf16(RN_f32(y_unit + y_side) + bias), the glue after the unit."""
    y = _f32(y_unit)
    if y_side is not None:
        y = (y + np.asarray(y_side, np.float32)).astype(np.float32)
    if bias_bits is not None:
        y = (y + (np.asarray(bias_bits, np.uint32) << 16).view(np.float32)[None, :]).astype(np.float32)
    w = y.view(np.uint32).astype(np.uint64)
    return ((w + 0x7FFF + ((w >> 16) & 1)) >> 16).astype(np.uint16)


# ---- the run ------------------------------------------------------------------------------------------------------

def _host(t: Any) -> np.ndarray:
    return t.detach().float().cpu().numpy() if hasattr(t, "detach") else np.asarray(t, np.float32)


def _like(y: np.ndarray, x: Any) -> Any:
    if hasattr(x, "detach"):
        import torch
        return torch.from_numpy(y).to(device=x.device, dtype=x.dtype)
    return y


def unit_weight_rows(weight: Any, split: Split) -> list[bytes]:
    """A weight's rows as its units read them: the kept channels' FP32 words, padded."""
    return [r.tobytes() for r in split.unit_words(np.ascontiguousarray(_host(weight), np.float32).view(np.uint32))]


def commit_weights(weights: Mapping[int, Any], splits: Mapping[int, Split]) -> dict[str, Any]:
    """Each layer's weight root over its unit rows, and its split's digest.  The splits bind in the records root
    (`pearl_c_work.records_root`), and the verifier recomputes S (`split_check`)."""
    roots = {w: commit_rows(weight_binding(w), -2, unit_weight_rows(wt, splits[w])).commitment.root.hex()
             for w, wt in weights.items()}
    return {"weight_roots": {str(w): r for w, r in roots.items()},
            "splits": {str(w): {"k": splits[w].k, "side": list(splits[w].side), "digest": splits[w].digest().hex()}
                       for w in weights}}


@dataclass
class PearlCRun:
    """One install's Pearl-C workload over the CPU twin, with `pouw._Run`'s interface."""

    scheme: C.PearlC
    entry: str
    weight_roots: dict[int, bytes]
    salt: bytes
    splits: dict[int, Split]
    transcript_dir: str | None = None
    units: list[Matmul] = field(default_factory=list)
    roots_a: list[bytes] = field(default_factory=list)
    leaves: list[bytes] = field(default_factory=list)
    records: list[PW.UnitRecord] = field(default_factory=list)
    excluded_b: dict[int, tuple[int, ...]] = field(default_factory=dict)
    sides: dict[int, WeightSide] = field(default_factory=dict)
    commits: dict[str, Any] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.commits = {"scheme": self.scheme.name, "salt": self.salt.hex(), "weights_root": weights_root(self.weight_roots).hex(),
                        "splits": {str(w): s.digest().hex() for w, s in self.splits.items()},
                        "transcript_root": None, "units": 0, "tiles": 0}
        self.details = {"scheme": self.scheme.name, "committer": "pouw_pearl_c twin (host)", "units": 0, "tiles": 0,
                        "padded_rows": 0, "excluded_rows": 0,
                        "side_channels": {str(w): len(s.side) for w, s in self.splits.items()},
                        "uncovered": "every GEMM outside LinearBase (lm_head, attention)"}

    def gemm(self, w: int, x: Any, weight: Any, bias: Any) -> Any:
        xs = _host(x)
        lead, k = xs.shape[:-1], xs.shape[-1]
        xs = np.ascontiguousarray(xs.reshape(-1, k), np.float32)
        m = xs.shape[0]
        split = self.splits[w]
        a = split.unit_words(xs.view(np.uint32))
        mp = max(ALIGN, -(-m // ALIGN) * ALIGN)
        if mp > m:
            a = np.concatenate([a, np.zeros((mp - m, split.k_unit), np.uint32)])
        ws_ = np.ascontiguousarray(_host(weight), np.float32)
        b = split.unit_words(ws_.view(np.uint32))
        n = b.shape[0]
        u = Matmul(len(self.units), Shape(mp, split.k_unit, n), w)
        self.scheme.check(u)
        if w not in self.excluded_b:
            self.excluded_b[w] = tuple(j for j in range(n) if not row_ok(b[j]))
        record = PW.UnitRecord(u, m, split.k_kept, tuple(i for i in range(m) if not row_ok(a[i])), self.excluded_b[w])
        rows = [r.tobytes() for r in a]
        root_a = commit_rows(activation_binding(self.salt, u.index), u.index, rows).commitment.root
        ctx = UnitContext(self.salt, u, root_a, self.weight_roots[w])
        noise = self.scheme.noise(ctx)
        unit, self.sides[w] = twin_call(self.scheme, ctx, noise, a, b, self.sides.get(w))
        self.records.append(record)
        for tile in self.scheme.tiles(u.shape):
            self.leaves.append(unit.leaves[(tile.rows[0], tile.cols[0])])
        self.units.append(u)
        self.roots_a.append(root_a)
        y_side = split.side_values(xs) @ split.side_values(ws_).T if split.side else None
        bias_bits = None if bias is None else (np.ascontiguousarray(_host(bias), np.float32).view(np.uint32) >> 16)
        y = merge(unit.y_unit[:m], y_side, bias_bits)
        self.details.update(units=len(self.units), tiles=len(self.leaves), padded_rows=self.details["padded_rows"] + mp - m,
                            excluded_rows=self.details["excluded_rows"] + len(record.excluded_a))
        return _like(((y.astype(np.uint32) << 16).view(np.float32)).reshape(*lead, n), x)

    def finish(self) -> None:
        """The transcript root over every tile leaf, in unit and tile order (`verity_pouw.audit.Prover`'s tree)."""
        if self.leaves:
            epoch = Epoch(Workload(tuple(self.units)), dict(self.weight_roots), self.salt)
            tree = MerkleTree(transcript_domain(epoch, len(self.leaves)), dict(enumerate(self.leaves)), lambda _: TILE_SCHEMA)
            self.commits["transcript_root"] = tree.commitment.root.hex()
            splits = {w: self.splits[w] for w in {u.weight for u in self.units}}
            self.commits["records_root"] = PW.records_root(self.salt, epoch.weights_root, splits, self.records).hex()
        self.commits.update(units=len(self.units), tiles=len(self.leaves))
        if self.transcript_dir:
            d = Path(self.transcript_dir)
            d.mkdir(parents=True, exist_ok=True)
            units = [{"index": u.index, "shape": [u.shape.m, u.shape.k, u.shape.n], "weight": u.weight, "root_a": r.hex(),
                      "real_rows": rec.real_rows, "k_kept": rec.k_kept, "excluded_a": list(rec.excluded_a),
                      "excluded_b": list(rec.excluded_b)} for u, r, rec in zip(self.units, self.roots_a, self.records)]
            (d / "pouw-pearl-c-transcript.json").write_text(json.dumps(
                {"entry": self.entry, "commitment": self.commits, "units": units,
                 "leaves": [x.hex() for x in self.leaves]}, indent=1) + "\n")
