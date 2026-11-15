"""PoUW's NCP schemes served on the host: ``ncp-v1`` (`NcpRun`), each linear call one unit of the scheme's honest work
whose tiles' digests join a transcript, and ``ncp-v2`` (`NcpCircuitRun`), each call its own `NcpLinear_v1` Program call
reading its weight's `NcpWeight_v1` call, computed natively (`native`) or by the serving kernel (`device`) and committed a
leaf per unit (`circuit`). Each run serves a call through `gemm(site, x, weight, bias)` (`serving.run_matmul`) and closes
with `finish`; ``commits`` and ``details`` are what it publishes. A run reads its settings (`NcpSettings`) from the
workload that installs it."""
from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from verity.commitments import MerkleTree
from verity.protocols.accounting.pouw.prover.serving import circuit as PC
from verity.protocols.accounting.pouw.prover.serving import device as PD
from verity.protocols.accounting.pouw.prover.serving import native as PN
from verity.protocols.accounting.pouw.prover.serving.operands import Operands, ncp_tiles, quantize, stack_y
from verity.protocols.accounting.pouw.verifier.audit import (
    HM96,
    TILE_SCHEMA,
    Epoch,
    Workload,
    activation_binding,
    commit_rows,
    transcript_domain,
    weights_root,
)
from verity.protocols.accounting.pouw.verifier.protocol import Context as UnitContext
from verity.protocols.accounting.pouw.verifier.protocol import Matmul, Shape

#: The pinned int7 quantizer of the FP8 lane's reference (docs/pouw-fp8/reference.md), applied without a basis rotation.
QUANTIZER = "verity.pouw.symmetric-int7-rowmax-rne-f32/v1"
PLACEHOLDER = (
    "a forward's weight is not re-checked against the root committed at install",
    ("not auditable yet: one epoch per install (no rotation), activation rows kept only in memory for the last "
     "`retain_steps` units (0 by default) and never persisted, no opener, and the transcript's leaves grow without bound "
     "for the life of the install"),
    "the executor runs on the host in numpy, not a native route-U kernel")
#: The host bytes `weight_operands` works in at a time (its copy of the weight's rows and their int64 noise and sums).
FORM_BYTES = 1 << 28

class NcpSettings(Protocol):
    """What a run reads of its install's settings: the units it retains (``retain_steps``), where it writes its transcript
    (``transcript_dir``), ``ncp-v2``'s first forward index (``start_forward``), negative control (``tamper``: ``tile`` or
    ``bias``) and committer (``device``: ``cuda``, or None for the host)."""

    retain_steps: int
    transcript_dir: str | None
    start_forward: int
    tamper: str | None
    device: str | None


# --- operands --------------------------------------------------------------------------------------------------------------

def weight_operands(scheme: Any, salt: bytes, w: int, weight: Any, noise: tuple) -> Operands:
    """``weight`` (n x k, a tensor or an array) quantized per output channel (`quantize`) and stacked (`stack_y`), a block
    of rows at a time: its host copy and int64 work stay under about `FORM_BYTES` whatever the layer, beside Y itself."""
    n, k = weight.shape
    step = max(1, FORM_BYTES // (128 * k))
    scales, y, c0 = np.empty(n, np.float32), np.empty((n, 3 * k), np.int8), np.empty(n, np.float64)
    for j in range(0, n, step):
        b, scales[j:j + step] = quantize(host(weight[j:j + step]))
        y[j:j + step], c0[j:j + step] = stack_y(scheme, salt, w, b, noise, j)
    return Operands(scales, y, c0)


def host(t: Any) -> np.ndarray:
    return t.detach().float().cpu().numpy() if hasattr(t, "detach") else np.asarray(t, dtype=np.float32)


def like(y: np.ndarray, x: Any) -> Any:
    if hasattr(x, "detach"):
        import torch
        return torch.from_numpy(y).to(device=x.device, dtype=x.dtype)
    return y.astype(np.asarray(x).dtype)


# --- the run -----------------------------------------------------------------------------------------------------------------

class NcpRun:
    """One install's workload: the units served so far, their activation roots and the transcript's leaves."""

    def __init__(self, scheme: Any, entry: str, settings: NcpSettings, weight_roots: dict[int, bytes], salt: bytes,
                 salt_key: bytes) -> None:
        self.scheme, self.entry, self.settings = scheme, entry, settings
        self.weight_roots, self.salt, self.salt_key = weight_roots, salt, salt_key
        self.units: list[Matmul] = []
        self.roots_a: list[bytes] = []
        self.leaves: list[bytes] = []
        self.retained: deque = deque(maxlen=settings.retain_steps)
        cert = scheme.certificate
        self.commits: dict[str, Any] = {
            "scheme": scheme.name, "salt": salt.hex(), "weights_root": weights_root(weight_roots).hex(),
            "quantizer": QUANTIZER, "transcript_root": None, "units": 0, "tiles": 0}
        self.details: dict[str, Any] = {
            "scheme": scheme.name, "quantizer": QUANTIZER, "basis": "none", "layers": len(weight_roots),
            # "gamma" is the gap g's stored name (records before the rename hold it); "eps" is ε = g/(1 − g)
            "certificate": {"conjecture": cert.conjecture, "gamma": None if cert.gap is None else str(cert.gap),
                            "eps": None if cert.eps is None else str(cert.eps), "parameters": dict(cert.parameters)},
            "units": 0, "tiles": 0, "padded_rows": 0, "uncovered": "every GEMM outside LinearBase (lm_head, attention)",
            "placeholder": list(PLACEHOLDER)}

    def noise_of(self, w: int, n: int, k: int) -> tuple:
        """The scheme's permutation at depth k (its noise before the salt-derived rows)."""
        return self.scheme.noise(UnitContext(self.salt, Matmul(0, Shape(16, k, n), w), b"", self.weight_roots[w]))

    def gemm(self, w: int, x: Any, weight: Any, bias: Any) -> Any:
        ops = weight_operands(self.scheme, self.salt, w, weight, self.noise_of(w, *weight.shape))
        n, k = ops.y.shape[0], ops.y.shape[1] // 3
        xs = host(x)
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
        root_a = commit_rows(activation_binding(self.salt, u.index), u.index, rows, HM96, self.salt_key, u.index).commitment.root
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
        out = like(y.reshape(*lead, n), x)
        return out if bias is None else out + bias

    def finish(self) -> None:
        """Commit the transcript: the Merkle root over every tile digest, in unit and tile order (`Prover.prove`'s tree)."""
        if self.leaves:
            epoch = Epoch(Workload(tuple(self.units)), dict(self.weight_roots), self.salt)
            tree = MerkleTree(transcript_domain(epoch, len(self.leaves), self.scheme.commitment_hash), dict(enumerate(self.leaves)),
                              lambda _: TILE_SCHEMA)
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


def f32_bits(v: np.ndarray) -> np.ndarray:
    return np.asarray(v, dtype=np.float32).view(np.uint32)


def bf16_bits(v: np.ndarray) -> np.ndarray:
    """BF16 encodings of finite FP32 values, rounded to nearest even."""
    b = f32_bits(v).astype(np.uint64)
    return ((b + 0x7FFF + ((b >> 16) & 1)) >> 16).astype(np.uint16)


class LayerRows:
    """A layer's weight rows, quantized per output channel (`quantize`) as they are read, from the layer's own tensor:
    no copy of the weight outlives the read (`circuit.Rows`)."""

    def __init__(self, weight: Any) -> None:
        self.weight = weight
        self.n, self.k = weight.shape

    def rows(self, lo: int, hi: int) -> tuple[np.ndarray, np.ndarray]:
        b, t = quantize(host(self.weight[lo:hi]))
        return b, f32_bits(t)


class NcpCircuitRun:
    """One install's `ncp-v2` workload: every call bound to its `NcpLinear_v1` Program's inputs, its units computed
    natively and committed (`native`). Without ``commit`` (a Match capture) each call serves the same y
    (`native.served`) and commits nothing."""

    def __init__(self, scheme: str, entry: str, settings: NcpSettings, weight_roots: dict[int, bytes], salt: bytes,
                 commit: bool = True) -> None:
        self.scheme, self.entry, self.settings, self.commit = scheme, entry, settings, commit
        self.weight_roots, self.salt = weight_roots, salt
        self.tampered: dict[str, Any] | None = None
        self.records: list[PC.CallRecord] = []
        self.sides: dict[int, PC.WeightRecord] = {}     # each weight's side, committed at the run's first call on it
        self.seconds: list[float] = []                  # each call's executor wall time: quantize, commit, dequantize
        self.forward, self.row_base, self.rows_seen = settings.start_forward - 1, 0, 0
        self.commits: dict[str, Any] = {
            "scheme": scheme, "salt": salt.hex(), "weights_root": weights_root(weight_roots).hex(), "quantizer": QUANTIZER,
            "calls_root": None, "calls": 0, "start_forward": settings.start_forward}
        self.details: dict[str, Any] = {
            "scheme": scheme, "quantizer": QUANTIZER, "basis": "none", "layers": len(weight_roots),
            "circuit": "verity.protocols.accounting.pouw.prover.circuit: each weight's side is its own NcpWeight_v1 Program call, once per run, and each "
                       "call its own NcpLinear_v1 Program call reading it, their units computed natively and committed a leaf "
                       "per unit; y served, credited words committed",
            "permutation": PC.kind_of(scheme),
            "calls": 0, "padded_rows": 0, "commit_seconds": 0.0, "uncovered": "every GEMM outside LinearBase (lm_head, attention)",
            "committer": "pouw_device fused (ncp2_fused.cu, cuda)" if settings.device == "cuda" else "pouw_native (host)",
            "placeholder": list(PLACEHOLDER)}

    def gemm(self, w: int, x: Any, weight: Any, bias: Any) -> Any:
        """The call's anchored fields: the forward index counts the passes from ``start_forward`` (a pass starts at linear
        0; a salt's second execution phase starts where `anchors.SaltWindows.next_forward` says), the position is
        ``w << 32``, the weight id is ``w``, a pass's rows get global ids from the rows served before it, and the weight's
        rows are its columns 0 … n − 1."""
        t0 = time.monotonic()
        if w == 0:
            self.forward += 1
            self.row_base = self.rows_seen
        xs = host(x)
        n, k = weight.shape
        lead = xs.shape[:-1]
        a, s = quantize(xs.reshape(-1, k))
        m = a.shape[0]
        mp = max(16, -(-m // 16) * 16)
        if mp > m:
            a = np.concatenate([a, np.zeros((mp - m, k), np.int8)])
            s = np.concatenate([s, np.ones(mp - m, np.float32)])
        if w == 0:
            self.rows_seen += mp
        bias_bits = np.full(n, 0x8000, np.uint16) if bias is None else bf16_bits(host(bias).reshape(-1))   # −0: the bf16 add's identity
        if self.settings.tamper == "bias" and bias is not None and self.tampered is None and self.commit:
            bias_bits = bias_bits.copy()
            bias_bits[0] ^= 1
            self.tampered = self.details["tampered"] = {"what": "bias", "call": len(self.records), "word": 0}
        # the call's position is (layer << 32) | site: layer is the linear's ordinal and site its call within the forward,
        # 0 since each linear runs once a forward
        call = PC.Call(a, f32_bits(s), LayerRows(weight), bias_bits, self.salt, forward=self.forward, layer=w, site=0, weight=w,
                       rowids=np.arange(self.row_base, self.row_base + mp, dtype=np.uint32), colids=np.arange(n, dtype=np.uint32))
        if self.commit:
            side = self.sides.get(w)
            if self.settings.device == "cuda":
                rec = PD.commit_call(len(self.records), self.scheme, call, on="fused", side=side)
            else:
                rec = PN.commit_call(len(self.records), self.scheme, call, side=side or PN.commit_weight(self.scheme, call))
            self.sides[w] = rec.side
            self.records.append(rec)
            served = rec.y
        else:
            served = PN.served(call)
        y = like(((np.asarray(served[:m], dtype=np.uint32) << 16).view(np.float32)).reshape(*lead, n), x)
        self.seconds.append(time.monotonic() - t0)
        self.details.update(calls=len(self.seconds), padded_rows=self.details["padded_rows"] + mp - m,
                            commit_seconds=round(sum(self.seconds), 3))
        return y

    def finish(self) -> None:
        """Commit the calls: the root over every call's leaf (`circuit.calls_root`)."""
        if self.records:
            self.commits["calls_root"] = PC.calls_root(self.salt, bytes.fromhex(self.commits["weights_root"]), self.records).hex()
        self.commits["calls"] = len(self.records)
        if self.settings.transcript_dir:
            d = Path(self.settings.transcript_dir)
            d.mkdir(parents=True, exist_ok=True)
            (d / PC.CALLS_FILE).write_text(json.dumps({"entry": self.entry, "commitment": self.commits,
                                                        "weights": [PC.weight_summary(s) for s in PC.sides(self.records)],
                                                        "calls": [PC.summary(r) for r in self.records]}) + "\n")
            PC.save_calls(d / PC.CALLS_DATA_FILE, self.records)


__all__ = ["FORM_BYTES", "PLACEHOLDER", "QUANTIZER", "LayerRows", "NcpCircuitRun", "NcpRun", "NcpSettings", "bf16_bits", "f32_bits",
           "host", "like", "weight_operands"]
