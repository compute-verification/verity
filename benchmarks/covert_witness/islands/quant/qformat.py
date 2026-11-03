"""CWQUAN1: an update as low-bit codes, entropy coded, the reference codec of `prepare.decode`'s quant format
(`prepare.encode_quant` writes `encode`'s bytes, which train.py checks before it relies on it).

    E = b"CWQUAN1\\n" + xz(inner),   inner = 4-byte header length, a JSON header, then each array's bytes in order

The header lists one entry per changed tensor of θ, each one of:

- `delta`: W' = (W.float() + Q).to(W.dtype), Q = codebook[code] * scale[group] in row-major order, a group being `group`
  consecutive entries. The codebook (2**bits entries) and the scales are f16, so every product is exact in fp32 and the
  rebuild is the same on every device;
- `lora`: W' = prepare.lora_merge(W, Q_A, Q_B, s), Q_A (r x d_in) and Q_B (d_out x r) dequantized as above;
- `full`: the tensor itself, in θ's dtype.

Codes are `bits` wide (1 to 8), packed little-endian within a byte stream (te_core's `pack_bits` order). 8·|E| counts every
byte: the header, the codebooks, the scales and the codes, after the entropy coder (xz, no check). The encoders write xz
preset 1; the decoders read any one xz stream, so `recompress` (preset 9e, or the smallest of `XZ_TRIES`) changes |E| and
nothing it decodes to.
"""
from __future__ import annotations

import json
import lzma
import math
import struct
from dataclasses import dataclass

import numpy as np
import torch

MAGIC = b"CWQUAN1\n"
XZ = {"format": lzma.FORMAT_XZ, "check": lzma.CHECK_NONE, "preset": 1}
XZ9 = {"format": lzma.FORMAT_XZ, "check": lzma.CHECK_NONE,
       "filters": [{"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME}]}
#: LZMA2 encoder settings `recompress(search=True)` tries (match finder, nice length, search depth: decoders never see
#: them). Which is smallest depends on the sparsity: a shallow hash chain at a few % nonzero (6% under 9e at 4.5%), a
#: binary tree at depth 1 at 1% and below (2-6% under 9e).
XZ_TRIES = tuple({"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME} | kw for kw in (
    {}, {"mf": lzma.MF_BT4, "nice_len": 273, "depth": 1}, {"mf": lzma.MF_BT4, "nice_len": 48, "depth": 1},
    {"mf": lzma.MF_HC3, "nice_len": 48, "depth": 2}, {"mf": lzma.MF_HC3, "nice_len": 32, "depth": 3}))


@dataclass
class QArray:
    """`bits`-bit codes into an f16 codebook, times an f16 scale per group of `group` consecutive entries."""

    codebook: torch.Tensor
    scales: torch.Tensor
    codes: torch.Tensor
    bits: int
    group: int
    shape: tuple[int, ...]

    def value(self) -> torch.Tensor:
        """The dequantized array in fp32 (exact: an f16 times an f16)."""
        cb = self.codebook.to(torch.float16).float()
        sc = self.scales.to(torch.float16).float()
        v = cb.to(self.codes.device)[self.codes.long()].view(-1, self.group) * sc.to(self.codes.device)[:, None]
        return v.view(self.shape)

    def spec(self) -> dict:
        return {"b": self.bits, "g": self.group}

    def entropy_bits(self) -> float:
        """An ideal order-0 coder's size: the codes at their empirical entropy, the codebook and scales at 16 bits."""
        p = torch.bincount(self.codes.reshape(-1).long(), minlength=2 ** self.bits).double()
        p = p[p > 0] / p.sum()
        return float(-(p * p.log2()).sum()) * self.codes.numel() + 16 * (self.scales.numel() + 2 ** self.bits)

    def arrays(self) -> list[bytes]:
        return [self.codebook.to(torch.float16).cpu().contiguous().view(torch.uint8).numpy().tobytes(),
                self.scales.to(torch.float16).cpu().contiguous().view(torch.uint8).numpy().tobytes(),
                pack_codes(self.codes.to(torch.uint8).cpu().numpy().reshape(-1), self.bits)]


def pack_codes(codes: np.ndarray, bits: int) -> bytes:
    if bits == 8:
        return codes.astype(np.uint8).tobytes()
    flat = ((codes.astype(np.uint8)[:, None] >> np.arange(bits, dtype=np.uint8)) & 1).reshape(-1)
    return np.packbits(flat, bitorder="little").tobytes()


def unpack_codes(buf: bytes, bits: int, n: int) -> np.ndarray:
    if bits == 8:
        return np.frombuffer(buf, dtype=np.uint8)[:n].copy()
    flat = np.unpackbits(np.frombuffer(buf, dtype=np.uint8), bitorder="little")[: n * bits].reshape(n, bits)
    return (flat << np.arange(bits, dtype=np.uint8)).sum(1, dtype=np.uint8)


def code_bytes(n: int, bits: int) -> int:
    return math.ceil(n * bits / 8)


# ---- encode ------------------------------------------------------------------------------------------------------------

def encode(entries: dict[str, dict], theta: dict[str, torch.Tensor]) -> bytes:
    """E from `entries`: name -> {"kind": "delta", "q": QArray} | {"kind": "lora", "a": QArray, "b": QArray, "scale": s}
    | {"kind": "full", "t": tensor}."""
    header, parts = [], []
    for name in sorted(entries):
        e = entries[name]
        if e["kind"] == "delta":
            q = e["q"]
            if tuple(q.shape) != tuple(theta[name].shape):
                raise ValueError(f"{name}: delta of shape {q.shape} for a {tuple(theta[name].shape)} tensor")
            header.append({"n": name, "k": "delta", **q.spec()})
            parts += q.arrays()
        elif e["kind"] == "lora":
            a, b = e["a"], e["b"]
            header.append({"n": name, "k": "lora", "r": int(a.shape[0]), "s": float(e["scale"]), "A": a.spec(),
                           "B": b.spec()})
            parts += a.arrays() + b.arrays()
        elif e["kind"] == "full":
            t = e["t"].detach().to(theta[name].dtype).cpu().contiguous()
            header.append({"n": name, "k": "full"})
            parts.append(t.view(torch.uint8).numpy().tobytes())
        else:
            raise ValueError(f"{name}: unknown kind {e['kind']!r}")
    h = json.dumps({"t": header}, separators=(",", ":")).encode()
    inner = struct.pack("<I", len(h)) + h + b"".join(parts)
    return MAGIC + lzma.compress(inner, **XZ)


def recompress(blob: bytes, search: bool = False) -> bytes:
    """The same CWQUAN1 payload under xz preset 9e (sparse codes: ~1.2x their order-0 entropy against preset 1's ~1.5x),
    or with `search` the smallest of `XZ_TRIES`."""
    if blob[:8] != MAGIC:
        raise ValueError("E: not CWQUAN1")
    raw = lzma.decompress(bytes(blob[8:]), format=lzma.FORMAT_XZ)
    if not search:
        return MAGIC + lzma.compress(raw, **XZ9)
    return MAGIC + min((lzma.compress(raw, format=lzma.FORMAT_XZ, check=lzma.CHECK_NONE, filters=[f]) for f in XZ_TRIES),
                       key=len)


# ---- decode ------------------------------------------------------------------------------------------------------------

def _take(payload: memoryview, at: int, n: int) -> tuple[bytes, int]:
    if at + n > len(payload):
        raise ValueError("E (quan): payload too short")
    return bytes(payload[at:at + n]), at + n


def _qarray(payload: memoryview, at: int, spec: dict, shape: tuple[int, ...], device: str) -> tuple[torch.Tensor, int]:
    bits, group = int(spec["b"]), int(spec["g"])
    n = math.prod(shape)
    if not 1 <= bits <= 8 or group < 1 or n % group:
        raise ValueError(f"E (quan): bits {bits}, group {group} for {n} entries")
    cb, at = _take(payload, at, 2 * 2 ** bits)
    sc, at = _take(payload, at, 2 * (n // group))
    codes, at = _take(payload, at, code_bytes(n, bits))
    q = QArray(torch.frombuffer(bytearray(cb), dtype=torch.float16), torch.frombuffer(bytearray(sc), dtype=torch.float16),
               torch.from_numpy(unpack_codes(codes, bits, n)).to(device), bits, group, shape)
    return q.value(), at


@torch.no_grad()
def decode(theta: dict[str, torch.Tensor], blob: bytes, device: str = "cpu") -> dict[str, torch.Tensor]:
    """The model E encodes from θ, every tensor in θ's dtype on the CPU (`prepare.decode`'s contract)."""
    if blob[:8] != MAGIC:
        raise ValueError("E: not CWQUAN1")
    inner = lzma.decompress(bytes(blob[8:]), format=lzma.FORMAT_XZ)
    (n,) = struct.unpack("<I", inner[:4])
    header = json.loads(inner[4:4 + n])
    payload, at = memoryview(inner)[4 + n:], 0
    out = dict(theta)
    for e in header["t"]:
        name = e["n"]
        w = theta[name]
        if e["k"] == "delta":
            q, at = _qarray(payload, at, e, tuple(w.shape), device)
            out[name] = (w.to(device).float() + q).to(w.dtype).cpu()
        elif e["k"] == "lora":
            import prepare

            if w.ndim != 2:
                raise ValueError(f"E (quan): {name} is not a matrix")
            r = int(e["r"])
            a, at = _qarray(payload, at, e["A"], (r, w.shape[1]), device)
            b, at = _qarray(payload, at, e["B"], (w.shape[0], r), device)
            out[name] = prepare.lora_merge(w.to(device), a, b, float(e["s"])).cpu()
        elif e["k"] == "full":
            raw, at = _take(payload, at, w.numel() * w.element_size())
            out[name] = torch.frombuffer(bytearray(raw), dtype=w.dtype).view(w.shape).clone()
        else:
            raise ValueError(f"E (quan): unknown kind {e['k']!r}")
    if at != len(payload):
        raise ValueError("E (quan): trailing bytes")
    return out


# ---- quantizers (the encoder's side; any QArray decodes) --------------------------------------------------------------

def _f16(x: torch.Tensor) -> torch.Tensor:
    return x.clamp(max=65504.0).to(torch.float16)


def groups_of(x: torch.Tensor, group: int) -> int:
    g = x.shape[-1] if group <= 0 else group
    if x.numel() % g:
        raise ValueError(f"group {g} does not divide {x.numel()} entries")
    return g


@torch.no_grad()
def rtn(x: torch.Tensor, bits: int, group: int = 128, clips=(1.0, 0.9, 0.8, 0.7, 0.6, 0.5)) -> QArray:
    """Symmetric round-to-nearest: levels -qmax..qmax (qmax = 2**(bits-1) - 1, so 2 bits is ternary), an absmax scale per
    group shrunk by the clip ratio (one per tensor) that minimizes the squared error, the first such on a tie. The choice
    is an argmin in tensor ops, never a branch on a read-back error, so a traced step that codes a tensor shows it."""
    if bits < 2:
        raise ValueError("rtn: at least 2 bits (1 bit is `sign`)")
    g = groups_of(x, group)
    xg = x.float().reshape(-1, g)
    qmax = 2 ** (bits - 1) - 1
    amax = xg.abs().amax(1)
    scs = torch.stack([_f16(c * amax / qmax).float() for c in clips])
    ks = torch.clamp(torch.round(xg / torch.where(scs > 0, scs, 1.0)[:, :, None]), -qmax, qmax) * (scs > 0)[:, :, None]
    i = ((ks * scs[:, :, None] - xg) ** 2).sum((1, 2)).argmin()
    sc, k = scs[i], ks[i]
    cb = torch.zeros(2 ** bits)
    cb[: 2 * qmax + 1] = torch.arange(-qmax, qmax + 1, dtype=torch.float32)
    return QArray(cb.to(torch.float16), sc.to(torch.float16), (k + qmax).to(torch.uint8).view(-1), bits, g, tuple(x.shape))


@torch.no_grad()
def sign(x: torch.Tensor, group: int = 0) -> QArray:
    """1 bit: sign(x) times the group's mean |x| (the L2-best scale for a sign code)."""
    g = groups_of(x, group)
    xg = x.float().reshape(-1, g)
    sc = _f16(xg.abs().mean(1))
    codes = (xg >= 0).to(torch.uint8).view(-1)
    return QArray(torch.tensor([-1.0, 1.0], dtype=torch.float16), sc, codes, 1, g, tuple(x.shape))


@torch.no_grad()
def ternary(x: torch.Tensor, tau: float, group: int = 0) -> QArray:
    """2-bit codes {0, +1, -1}: an entry is kept when |x| > tau · the group's mean |x|, the kept ones at the mean |x| they
    have; the zeros (code 0) are what the entropy coder shrinks."""
    g = groups_of(x, group)
    xg = x.float().reshape(-1, g)
    keep = xg.abs() > tau * xg.abs().mean(1, keepdim=True)
    cnt = keep.sum(1).clamp(min=1)
    sc = _f16((xg.abs() * keep).sum(1) / cnt)
    codes = torch.where(keep, torch.where(xg >= 0, 1, 2), 0).to(torch.uint8).view(-1)
    return QArray(torch.tensor([0.0, 1.0, -1.0, 0.0], dtype=torch.float16), sc, codes, 2, g, tuple(x.shape))
