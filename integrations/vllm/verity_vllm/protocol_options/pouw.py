"""PoUW as a protocol option: every linear layer's GEMM computed as a `verity_pouw` scheme's honest work.

The innermost option (`protocol_options` order).  `install` finds the linear layers by type (vLLM's ``LinearBase``,
`linear_types`) and wraps each one's ``quant_method.apply`` through the composition's tagged `Hooks`.  Before it wraps
anything it commits every layer's int7 weight rows (`verity_pouw.audit.commit_rows`) and derives the epoch salt from the
beacon (`verity_pouw.audit.Epoch.start`): the weights are committed before the salt exists and before any outer option
changes the model.  Each call of a wrapped ``apply(layer, x, bias)`` is one unit of the run's workload:

1. quantize x's rows to int7 per row (`quantize`), and take the weight operands (`weight_operands`: the int7 weight per
   output channel with the epoch's noise, Y = [B + F₁ ; B + F₂ ; B + F₃], and c₀).  The setting ``weight_operands`` says
   where they come from: ``"resident"`` (the default) prepares them once at install and keeps them, so
   `keeps_weight_copy` is True and the selector refuses POUS beside it; ``"per-forward"`` prepares them from the weight
   of the layer each call is handed, so under POUS the GEMM reads POUS's decoded weight;
2. commit the unit's activation rows, derive its noise and compute every 16 x 16 tile's route-U checked values and
   digest (`ncp_tiles`, gated bit for bit against the scheme's reference in the tests); the digests join the transcript;
3. return the useful output, the final checkpoint A·Bᵀ, dequantized by the row and column scales, plus the bias.

`commitment` is the scheme, the salt, the weight roots and, once closed, the transcript root with its unit and tile
counts.  The audit (drawing tiles, opening them) is `verity_pouw.audit`'s, after the run; sampled proofs' replay does not
re-check PoUW's GEMMs (PoUW's tile audit only, pending the design's open question 2).  What is a placeholder is listed in
`info` (``placeholder``).
"""
from __future__ import annotations

import hashlib
import json
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from verity_pouw import schemes as pouw_schemes
from verity_pouw.audit import TILE_SCHEMA, Epoch, Workload, activation_binding, commit_rows, transcript_domain, weight_binding
from verity_pouw.protocol import Context as UnitContext
from verity_pouw.protocol import DomainError, Matmul, Shape
from verity_pouw.schemes import ncp

from verity.commitments import MerkleTree
from verity_vllm.protocol_options.interface import Context, Option, Target

NAME = "pouw"
DEFAULT_SCHEME = "ncp-v1"
#: The schemes this adapter's executor computes; Pearl's FP10 / FP8 operands are not wired.
EXECUTED = ("ncp-v1", "ncp-v1-shift24")
#: The pinned int7 quantizer of the FP8 lane's reference (docs/pouw-fp8/reference.md), applied without a basis rotation.
QUANTIZER = "verity.pouw.symmetric-int7-rowmax-rne-f32/v1"
KEYS = ("beacon", "retain_steps", "transcript_dir", "weight_operands")
WEIGHT_OPERANDS = ("resident", "per-forward")
OUTSIDE_PROGRAM = ("PoUW replaces each linear layer's GEMM by its scheme's noised 3k-deep int8 GEMM; the useful output is "
                   "the int7 linear (quantize, exact int32 GEMM, dequantize), which no Definition of the Program describes yet")
PLACEHOLDER = (
    "without a `beacon` setting the salt comes from 32 zero bytes, so it is predictable",
    "with weight_operands = per-forward, a forward's weight is not re-checked against the root committed at install",
    "tile openings need the activation rows, kept only for the last `retain_steps` units",
    "the executor runs on the host in numpy, not a native route-U kernel",
)
_INV63 = np.float32(1.0) / np.float32(63.0)
_M32 = 0xFFFFFFFF


def schemes() -> tuple[str, ...]:
    return tuple(sorted(pouw_schemes.SCHEMES))


@dataclass(frozen=True)
class Settings:
    beacon: bytes | None = None
    retain_steps: int = 0
    transcript_dir: str | None = None
    weight_operands: str = "resident"

    @classmethod
    def of(cls, raw: Mapping[str, Any]) -> Settings:
        """PoUW's keys of the protocols config; ValueError names the first bad one."""
        unknown = sorted(set(raw) - set(KEYS))
        if unknown:
            raise ValueError(f"unknown pouw settings {unknown} (known: {', '.join(KEYS)})")
        beacon = raw.get("beacon")
        if beacon is not None:
            try:
                beacon = bytes.fromhex(beacon)
            except (TypeError, ValueError):
                raise ValueError("pouw setting 'beacon' is a hex string") from None
            if len(beacon) < 32:
                raise ValueError("pouw setting 'beacon' has at least 32 bytes")
        retain = raw.get("retain_steps", 0)
        if not isinstance(retain, int) or isinstance(retain, bool) or retain < 0:
            raise ValueError("pouw setting 'retain_steps' is a non-negative integer")
        tdir = raw.get("transcript_dir")
        if tdir is not None and not isinstance(tdir, str):
            raise ValueError("pouw setting 'transcript_dir' is a path string")
        operands = raw.get("weight_operands", "resident")
        if operands not in WEIGHT_OPERANDS:
            raise ValueError(f"pouw setting 'weight_operands' is one of {', '.join(WEIGHT_OPERANDS)}")
        return cls(beacon, retain, tdir, operands)


def keeps_weight_copy(scheme: str, ctx: Context) -> bool:
    """True when the weight operands stay resident across forwards (``weight_operands = "resident"``, the default)."""
    try:
        return Settings.of(ctx.settings_of(NAME)).weight_operands == "resident"
    except ValueError:
        return True


def refusals(scheme: str, ctx: Context) -> tuple[str, ...]:
    out = []
    if scheme not in EXECUTED:
        out.append(f"{scheme}: this adapter's executor runs {', '.join(EXECUTED)} only (Pearl's FP10/FP8 operands are not wired)")
    if not ctx.enforce_eager:
        out.append("compiled execution: the wrapper around each linear layer's apply is not graph-capturable")
    if ctx.world > 1:
        out.append(f"TP {ctx.world}: PoUW's units and transcript are per rank, which is not wired")
    try:
        Settings.of(ctx.settings_of(NAME))
    except ValueError as e:
        out.append(str(e))
    return tuple(out)


def linear_types() -> tuple[type, ...]:
    """The hook-site types: vLLM's ``LinearBase`` (qkv, o, gate_up, down).  lm_head is a ``VocabParallelEmbedding``, not
    covered."""
    from vllm.model_executor.layers.linear import LinearBase
    return (LinearBase,)


# --- operands --------------------------------------------------------------------------------------------------------------

def quantize(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``QUANTIZER`` per row: scale = RN_f32(rowmax · RN_f32(1/63)) (1 for a zero row), q = clamp(RNE(f32(v / scale)), −63, 63).
    Returns int8 rows and their FP32 scales; non-finite input is refused."""
    v = np.asarray(v, dtype=np.float32)
    if not np.isfinite(v).all():
        raise ValueError("pouw: the quantizer refuses NaN and infinity")
    scale = (np.abs(v).max(axis=1) * _INV63).astype(np.float32)
    scale[scale == 0] = np.float32(1.0)
    return np.clip(np.rint(v / scale[:, None]), -63, 63).astype(np.int8), scale


def _noise(tag: str, count: int, k: int, parts) -> np.ndarray:
    """The scheme's noise rows (`ncp.noise_row`: one XOF byte's low six bits minus 31 per entry), as int64."""
    return np.stack([np.frombuffer(ncp.xof(tag, k, *parts(i)), dtype=np.uint8).astype(np.int64) % 64 - 31 for i in range(count)])


@dataclass(frozen=True)
class Operands:
    """One weight's operands for an epoch: its column scales, Y = [B + F₁ ; B + F₂ ; B + F₃] as n rows of 3k, and
    c₀ = −β·Σ_l Y[l][j], the route-U accumulator's initial value (Lean `initU`)."""

    scales: np.ndarray
    y: np.ndarray
    c0: np.ndarray


def stack_y(scheme: Any, salt: bytes, w: int, b: np.ndarray, noise: tuple) -> tuple[np.ndarray, np.ndarray]:
    """Y (as int8 rows: every entry lies in [−128, 127], Lean `rangeU`) and c₀ for the int7 weight ``b`` (n x k), with F₁
    from the epoch salt (per epoch and weight, as `ncp.stack` draws it), F₂ = −P·F₁ and F₃ = F₁ + F₂ (`ncp.assemble`)."""
    n, k = b.shape
    pi, sigma = np.asarray(noise[0]), np.asarray(noise[1], dtype=np.int64)
    f1 = _noise("verity/pouw/ncp/F1/v1", n, k, lambda j: (salt, w, j))
    f2 = np.empty_like(f1)
    f2[:, pi] = -(f1 * sigma)
    b = b.astype(np.int64)
    y = np.concatenate([b + f1, b + f2, b + f1 + f2], axis=1)
    return y.astype(np.int8), (-scheme.beta * y.sum(axis=1)).astype(np.float64)


def weight_operands(scheme: Any, salt: bytes, w: int, weight: np.ndarray, noise: tuple) -> Operands:
    """``weight`` (n x k) quantized per output channel (`quantize`) and stacked (`stack_y`)."""
    b, scales = quantize(weight)
    return Operands(scales, *stack_y(scheme, salt, w, b, noise))


def ncp_tiles(scheme: Any, ctx: UnitContext, noise: tuple, a: np.ndarray, ops: Operands,
              chunk: int = 256) -> dict[tuple[int, int], tuple[bytes, np.ndarray]]:
    """One unit's tiles as the scheme's Prover kernel contract gives them: (first row, first column) -> (digest, int32
    output tile), for int7 ``a`` (m x k) and the weight's `Operands`.  X + β is formed as `ncp.assemble` forms X, the
    accumulator starts at c₀ (Lean `mmWU β`), and a tile's digest covers every depth-16 checkpoint, step-major and
    row-major within a step (`NCP.digest`).  Float64 products are exact here: every partial sum is an integer under 2^53."""
    u = ctx.matmul
    (m, k), n = a.shape, ops.y.shape[0]
    pi, sigma = np.asarray(noise[0]), np.asarray(noise[1], dtype=np.int64)
    a = a.astype(np.int64)
    e1 = _noise("verity/pouw/ncp/E1/v1", m, k, lambda i: (ctx.salt, u.index, ctx.root_a, u.weight, i))
    e2 = e1[:, pi] * sigma
    x = np.concatenate([a + e1, a + e2, -(a + e1 + e2)], axis=1) + scheme.beta
    y, c0 = ops.y, ops.c0
    steps = 3 * k // ncp.D
    out: dict[tuple[int, int], tuple[bytes, np.ndarray]] = {}
    for r in range(0, m, 16):
        xs = x[r:r + 16].reshape(16, steps, ncp.D).transpose(1, 0, 2).astype(np.float64)       # [T, 16, d]
        for c in range(0, n, chunk):
            ys = y[c:c + chunk].reshape(-1, steps, ncp.D).transpose(1, 2, 0).astype(np.float64)  # [T, d, w]
            run = np.cumsum(np.matmul(xs, ys), axis=0) + c0[None, None, c:c + chunk]
            words = (run.astype(np.int64) & _M32).astype("<u4")                                  # [T, 16, w]
            for j in range(0, words.shape[2], 16):
                block = np.ascontiguousarray(words[:, :, j:j + 16])
                digest = hashlib.shake_256(b"verity/pouw/ncp/tile/v1\0" + block.tobytes()).digest(32)
                out[(r, c + j)] = (digest, block[-1].astype(np.int64) - ((block[-1].astype(np.int64) >> 31) << 32))
    return out


def _host(t: Any) -> np.ndarray:
    return t.detach().float().cpu().numpy() if hasattr(t, "detach") else np.asarray(t, dtype=np.float32)


def _like(y: np.ndarray, x: Any) -> Any:
    if hasattr(x, "detach"):
        import torch
        return torch.from_numpy(y).to(device=x.device, dtype=x.dtype)
    return y.astype(np.asarray(x).dtype)


# --- the run -----------------------------------------------------------------------------------------------------------------

class _Run:
    """One install's workload: the units served so far, their activation roots and the transcript's leaves."""

    def __init__(self, scheme: Any, entry: str, settings: Settings, weight_roots: dict[int, bytes], salt: bytes) -> None:
        self.scheme, self.entry, self.settings = scheme, entry, settings
        self.weight_roots, self.salt = weight_roots, salt
        self.units: list[Matmul] = []
        self.roots_a: list[bytes] = []
        self.leaves: list[bytes] = []
        self.retained: deque = deque(maxlen=settings.retain_steps)
        self.resident: dict[int, Operands] | None = None
        cert = scheme.certificate
        self.commits: dict[str, Any] = {
            "scheme": scheme.name, "salt": salt.hex(), "weight_roots": {str(w): r.hex() for w, r in weight_roots.items()},
            "quantizer": QUANTIZER, "transcript_root": None, "units": 0, "tiles": 0}
        self.details: dict[str, Any] = {
            "scheme": scheme.name, "quantizer": QUANTIZER, "basis": "none", "layers": len(weight_roots),
            "certificate": {"conjecture": cert.conjecture, "gamma": None if cert.gamma is None else str(cert.gamma),
                            "parameters": dict(cert.parameters)},
            "beacon": "given" if settings.beacon else "placeholder: 32 zero bytes",
            "weight_operands": settings.weight_operands,
            "units": 0, "tiles": 0, "padded_rows": 0, "uncovered": "every GEMM outside LinearBase (lm_head, attention)",
            "placeholder": list(PLACEHOLDER)}

    def noise_of(self, w: int, n: int, k: int) -> tuple:
        """The scheme's permutation at depth k (its noise before the salt-derived rows)."""
        return self.scheme.noise(UnitContext(self.salt, Matmul(0, Shape(16, k, n), w), b"", self.weight_roots[w]))

    def operands(self, w: int, weight: Any) -> Operands:
        """The resident operands, or ``weight``'s, prepared now (``weight_operands = "per-forward"``)."""
        if self.resident is not None:
            return self.resident[w]
        ws = _host(weight)
        return weight_operands(self.scheme, self.salt, w, ws, self.noise_of(w, *ws.shape))

    def gemm(self, w: int, x: Any, weight: Any, bias: Any) -> Any:
        ops = self.operands(w, weight)
        n, k = ops.y.shape[0], ops.y.shape[1] // 3
        xs = _host(x)
        lead = xs.shape[:-1]
        a, s = quantize(xs.reshape(-1, k))
        t = ops.scales
        m = a.shape[0]
        mp = max(16, -(-m // 16) * 16)
        if mp > m:
            a = np.concatenate([a, np.zeros((mp - m, k), np.int8)])
        rows = [a[i].tobytes() for i in range(mp)]
        u = Matmul(len(self.units), Shape(mp, k, n), w)
        self.scheme.check(u)
        root_a = commit_rows(activation_binding(self.salt, u.index), u.index, rows).commitment.root
        ctx = UnitContext(self.salt, u, root_a, self.weight_roots[w])
        tiles = ncp_tiles(self.scheme, ctx, self.scheme.noise(ctx), a, ops)
        z = np.empty((mp, n), np.int64)
        for tile in self.scheme.tiles(u.shape):
            digest, block = tiles[(tile.rows[0], tile.cols[0])]
            self.leaves.append(digest)
            z[tile.rows[0]:tile.rows[-1] + 1, tile.cols[0]:tile.cols[-1] + 1] = block
        self.units.append(u)
        self.roots_a.append(root_a)
        self.retained.append((u.index, rows))
        self.details.update(units=len(self.units), tiles=len(self.leaves), padded_rows=self.details["padded_rows"] + mp - m)
        y = (s[:, None] * t[None, :]).astype(np.float32) * z[:m].astype(np.float32)
        out = _like(y.reshape(*lead, n), x)
        return out if bias is None else out + bias

    def finish(self) -> None:
        """Commit the transcript: the Merkle root over every tile digest, in unit and tile order (`Prover.prove`'s tree)."""
        if self.leaves:
            epoch = Epoch(Workload(tuple(self.units)), dict(self.weight_roots), self.salt)
            tree = MerkleTree(transcript_domain(epoch, len(self.leaves)), dict(enumerate(self.leaves)), lambda _: TILE_SCHEMA)
            self.commits["transcript_root"] = tree.commitment.root.hex()
        self.commits.update(units=len(self.units), tiles=len(self.leaves))
        if self.settings.transcript_dir:
            d = Path(self.settings.transcript_dir)
            d.mkdir(parents=True, exist_ok=True)
            units = [{"index": u.index, "shape": [u.shape.m, u.shape.k, u.shape.n], "weight": u.weight, "root_a": r.hex()}
                     for u, r in zip(self.units, self.roots_a)]
            (d / "pouw-transcript.json").write_text(json.dumps(
                {"entry": self.entry, "commitment": self.commits, "units": units, "leaves": [x.hex() for x in self.leaves],
                 "retained": [i for i, _ in self.retained]}, indent=1) + "\n")


def _executor(run: _Run, w: int):
    def apply(layer: Any, x: Any, bias: Any = None) -> Any:
        return run.gemm(w, x, layer.weight, bias)
    return apply


def install(scheme: str, target: Target) -> Option:
    entry = f"{NAME}:{scheme}"
    settings = Settings.of(target.context.settings_of(NAME))
    s = pouw_schemes.get(scheme)
    types = linear_types()
    layers = [m for m in target.model.modules() if isinstance(m, types)]
    if not layers:
        raise ValueError(f"{entry}: the model has no linear layer to wrap")
    bad, seen, shapes = [], set(), {}
    for w, layer in enumerate(layers):
        n, k = _host(layer.weight).shape
        shapes[w] = (n, k)
        try:
            s.check(Matmul(0, Shape(16, k, n), w))
        except DomainError as e:
            bad.append(f"linear {w}: {e}")
        if id(layer.quant_method) in seen:
            bad.append(f"linear {w}: its quant method is shared with another layer")
        seen.add(id(layer.quant_method))
    if bad:
        raise ValueError(f"{entry} refused at install: " + "; ".join(bad))
    roots = {}
    for w, layer in enumerate(layers):
        b, _ = quantize(_host(layer.weight))
        roots[w] = commit_rows(weight_binding(w), -2, [b[j].tobytes() for j in range(b.shape[0])]).commitment.root
    declared = Workload(tuple(Matmul(w, Shape(16, k, n), w) for w, (n, k) in shapes.items()))
    salt = Epoch.start(declared, roots, settings.beacon or bytes(32)).salt
    run = _Run(s, entry, settings, roots, salt)
    if settings.weight_operands == "resident":
        run.resident = {w: weight_operands(s, salt, w, _host(layer.weight), run.noise_of(w, *shapes[w]))
                        for w, layer in enumerate(layers)}
    h = target.hooks(entry)
    for w, layer in enumerate(layers):
        h.wrap(layer.quant_method, "apply", lambda orig, w=w: _executor(run, w))
    return Option(name=entry, hooks=h, commits=run.commits, details=run.details, outside_program=OUTSIDE_PROGRAM,
                  release=run.finish)
