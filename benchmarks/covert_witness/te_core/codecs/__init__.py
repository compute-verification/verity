"""The shared codec library: ways to rebuild a step's updated weights from θ, each with encode, decode, its size in bits
and its rebuild cost in MACs.

A codec maps (encoding, base) to a tensor, the base being θ's value of that weight (zeros of the logged shape and dtype for a
weight θ doesn't hold). `decode` is deterministic, and each lossy codec exposes the `apply` its decode runs, so a program that
trains in that parameterization (and calls the same `apply`) is rebuilt bit for bit; `encode` from an arbitrary target is
lossless only for `Identity` and `SparseMask`. A rebuild's cost counts every matmul MAC and one MAC per element of each
elementwise arithmetic pass; the checker also measures it (counterexample.py) and takes the larger.

- `Identity`: the full tensor (a full update).
- `LowRank(rank, scale)`: base + scale * B @ A, factors at fp32.
- `SparseMask(values)`: the base with the entries that differ replaced, the indices as a bitmask or int32 list
  (whichever is smaller), the values encoded by `values` against the base's (default `Identity`).
- `QuantDelta(bits)`: base + codebook[index], a 2**bits-entry fp32 codebook and `bits`-bit indices.
- `SeedReplay`: base + sum_j coeff_j * z_j, z_j ~ N(0, 1) from seed_j (MeZO-style); the log of (seed, coeff) pairs.
- `Compose(parts)`: each part applied to the previous part's output, starting from the base.

`Plan` encodes a step's whole weight set, one rule (key pattern, codec) per weight; its spec is a counterexample family's
decoder.
"""
from __future__ import annotations

import fnmatch
import hashlib

import torch

from .wire import pack, pack_bits, unpack, unpack_bits


class Unencodable(ValueError):
    """The codec has no encoding of this target (a log-only codec, or a shape it doesn't take)."""


class Codec:
    kind = ""

    def spec(self) -> dict:
        return {"kind": self.kind}

    def encode(self, target: torch.Tensor, base: torch.Tensor) -> bytes:
        raise NotImplementedError

    def decode(self, enc: bytes, base: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def size_bits(self, enc: bytes) -> int:
        return 8 * len(enc)

    def cost_macs(self, enc: bytes, base: torch.Tensor) -> int:
        return 0


class Identity(Codec):
    kind = "identity"

    def encode(self, target, base):
        return pack({}, [target.detach().to(base.dtype).reshape(base.shape)])

    def decode(self, enc, base):
        _, (t,) = unpack(enc)
        if t.shape != base.shape or t.dtype != base.dtype:
            raise ValueError(f"identity: {t.dtype}{tuple(t.shape)} for a {base.dtype}{tuple(base.shape)} weight")
        return t.to(base.device)


class LowRank(Codec):
    kind = "lowrank"

    def __init__(self, rank: int, scale: float = 1.0):
        self.rank, self.scale = int(rank), float(scale)

    def spec(self):
        return {"kind": self.kind, "rank": self.rank, "scale": self.scale}

    @staticmethod
    def apply(base: torch.Tensor, a: torch.Tensor, b: torch.Tensor, scale: float) -> torch.Tensor:
        return (base.float() + scale * (b.float() @ a.float())).to(base.dtype)

    @staticmethod
    def encode_factors(a: torch.Tensor, b: torch.Tensor) -> bytes:
        """The encoding of the factors a program trained (A: r x d_in, B: d_out x r)."""
        return pack({}, [a.detach().float().contiguous(), b.detach().float().contiguous()])

    def encode(self, target, base):
        if base.dim() != 2:
            raise Unencodable(f"lowrank takes a matrix, not shape {tuple(base.shape)}")
        delta = (target.double() - base.double()).cpu() / self.scale
        u, s, vh = torch.linalg.svd(delta, full_matrices=False)
        r = min(self.rank, s.numel())
        return pack({}, [vh[:r].float().contiguous(), (u[:, :r] * s[:r]).float().contiguous()])

    def decode(self, enc, base):
        _, (a, b) = unpack(enc)
        return self.apply(base, a.to(base.device), b.to(base.device), self.scale)

    def cost_macs(self, enc, base):
        h, _ = unpack(enc)
        r = h["a"][0][1][0]
        return r * base.numel() + 2 * base.numel()


def _bits_view(t: torch.Tensor) -> torch.Tensor:
    ints = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}
    return t.detach().contiguous().reshape(-1).view(ints[t.element_size()])


class SparseMask(Codec):
    kind = "sparse"

    def __init__(self, values: Codec | None = None):
        self.values = values or Identity()

    def spec(self):
        return {"kind": self.kind, "values": self.values.spec()}

    def encode(self, target, base):
        target = target.detach().to(base.dtype).reshape(base.shape)
        idx = torch.nonzero(_bits_view(target) != _bits_view(base)).reshape(-1).cpu()
        n = base.numel()
        if 32 * idx.numel() < n:
            index, form = idx.to(torch.int32), "list"
        else:
            mask = torch.zeros(n, dtype=torch.int64)
            mask[idx] = 1
            index, form = pack_bits(mask, 1), "mask"
        vals = self.values.encode(target.reshape(-1)[idx.to(target.device)], base.reshape(-1)[idx.to(base.device)])
        return pack({"f": form, "n": n}, [index, torch.frombuffer(bytearray(vals), dtype=torch.uint8)])

    def _index(self, enc):
        h, (index, vals) = unpack(enc)
        if h["f"] == "list":
            idx = index.to(torch.int64)
        else:
            idx = torch.nonzero(unpack_bits(index, 1, h["n"])).reshape(-1)
        return idx, vals.numpy().tobytes()

    def decode(self, enc, base):
        idx, vals = self._index(enc)
        idx = idx.to(base.device)
        out = base.detach().clone().reshape(-1)
        out[idx] = self.values.decode(vals, base.reshape(-1)[idx])
        return out.reshape(base.shape)

    def cost_macs(self, enc, base):
        idx, vals = self._index(enc)
        return self.values.cost_macs(vals, base.reshape(-1)[idx.to(base.device)])


class QuantDelta(Codec):
    kind = "quant"

    def __init__(self, bits: int):
        if not 1 <= bits <= 16:
            raise ValueError("quant: 1 to 16 bits")
        self.bits = int(bits)

    def spec(self):
        return {"kind": self.kind, "bits": self.bits}

    @staticmethod
    def apply(base: torch.Tensor, codebook: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        return (base.float() + codebook[index].reshape(base.shape)).to(base.dtype)

    def encode_index(self, codebook: torch.Tensor, index: torch.Tensor) -> bytes:
        """The encoding of a codebook and indices a program trained with."""
        return pack({"n": index.numel()}, [codebook.detach().float().cpu(), pack_bits(index.cpu(), self.bits)])

    def encode(self, target, base):
        delta = (target.float() - base.float()).reshape(-1).cpu()
        levels = 2 ** self.bits
        distinct = torch.unique(delta)
        if distinct.numel() <= levels:
            codebook = torch.cat([distinct, distinct[-1:].expand(levels - distinct.numel())]) if distinct.numel() \
                else torch.zeros(levels)
        else:
            q = (torch.arange(levels, dtype=torch.float64) + 0.5) / levels
            codebook = torch.quantile(delta.double(), q).float()
        codebook = torch.sort(codebook).values
        mids = (codebook[1:] + codebook[:-1]) / 2
        index = torch.bucketize(delta, mids)
        return pack({"n": delta.numel()}, [codebook, pack_bits(index, self.bits)])

    def decode(self, enc, base):
        h, (codebook, packed) = unpack(enc)
        index = unpack_bits(packed, self.bits, h["n"]).to(base.device)
        return self.apply(base, codebook.to(base.device), index)

    def cost_macs(self, enc, base):
        return base.numel()


def derive_seed(seed: int, key: str) -> int:
    """A per-weight seed from a step's seed and the weight's key, so each weight's noise is independent of the order weights
    are visited in."""
    return int.from_bytes(hashlib.sha256(f"{seed}/{key}".encode()).digest()[:8], "little") & (2**63 - 1)


class SeedReplay(Codec):
    kind = "seed"

    @staticmethod
    def apply(w: torch.Tensor, seed: int, coeff: float) -> None:
        """w += coeff * z, z ~ N(0, 1) from `seed`, in place on an fp32 tensor."""
        g = torch.Generator(device=w.device).manual_seed(int(seed))
        w.add_(torch.randn(w.shape, generator=g, device=w.device, dtype=torch.float32), alpha=float(coeff))

    @staticmethod
    def encode_log(seeds: list[int], coeffs: list[float]) -> bytes:
        return pack({}, [torch.tensor(seeds, dtype=torch.int64), torch.tensor(coeffs, dtype=torch.float32)])

    def encode(self, target, base):
        raise Unencodable("seed replay is encoded from the run's log (SeedReplay.encode_log), not from a target")

    def decode(self, enc, base):
        _, (seeds, coeffs) = unpack(enc)
        w = base.detach().float().clone()
        for s, c in zip(seeds.tolist(), coeffs.tolist()):
            self.apply(w, s, c)
        return w.to(base.dtype)

    def cost_macs(self, enc, base):
        _, (seeds, _) = unpack(enc)
        return seeds.numel() * base.numel()


class Compose(Codec):
    kind = "compose"

    def __init__(self, parts: list[Codec]):
        self.parts = list(parts)

    def spec(self):
        return {"kind": self.kind, "parts": [p.spec() for p in self.parts]}

    def _split(self, enc):
        _, arrays = unpack(enc)
        return [a.numpy().tobytes() for a in arrays]

    def encode(self, target, base):
        cur, encs = base, []
        for p in self.parts:
            e = p.encode(target, cur)
            cur = p.decode(e, cur)
            encs.append(e)
        return self.encode_parts(encs)

    def encode_parts(self, encs: list[bytes]) -> bytes:
        return pack({}, [torch.frombuffer(bytearray(e), dtype=torch.uint8) if e else torch.empty(0, dtype=torch.uint8)
                         for e in encs])

    def decode(self, enc, base):
        cur = base
        for p, e in zip(self.parts, self._split(enc), strict=True):
            cur = p.decode(e, cur)
        return cur

    def cost_macs(self, enc, base):
        return sum(p.cost_macs(e, base) for p, e in zip(self.parts, self._split(enc), strict=True))


def from_spec(spec: dict) -> Codec:
    kind = spec["kind"]
    if kind == "identity":
        return Identity()
    if kind == "lowrank":
        return LowRank(spec["rank"], spec.get("scale", 1.0))
    if kind == "sparse":
        return SparseMask(from_spec(spec["values"]) if "values" in spec else None)
    if kind == "quant":
        return QuantDelta(spec["bits"])
    if kind == "seed":
        return SeedReplay()
    if kind == "compose":
        return Compose([from_spec(p) for p in spec["parts"]])
    raise ValueError(f"unknown codec kind {kind!r}")


class Plan:
    """A step's weight set: each weight key takes the first rule whose pattern (fnmatch) matches it. The encoding is every
    key's encoding, in key order, with its rule's index."""

    def __init__(self, rules: list[tuple[str, Codec]]):
        self.rules = list(rules)

    @classmethod
    def from_spec(cls, spec: dict) -> Plan:
        return cls([(r["match"], from_spec(r["codec"])) for r in spec["rules"]])

    def spec(self) -> dict:
        return {"rules": [{"match": m, "codec": c.spec()} for m, c in self.rules]}

    def rule(self, key: str) -> int:
        for i, (m, _) in enumerate(self.rules):
            if fnmatch.fnmatchcase(key, m):
                return i
        raise Unencodable(f"no rule matches weight {key!r}")

    def encode(self, targets: dict[str, torch.Tensor], bases: dict[str, torch.Tensor]) -> bytes:
        keys = sorted(targets)
        encs = {k: self.rules[self.rule(k)][1].encode(targets[k], bases[k]) for k in keys}
        return self.encode_parts(encs)

    def encode_parts(self, encs: dict[str, bytes]) -> bytes:
        keys = sorted(encs)
        return pack({"k": keys, "r": [self.rule(k) for k in keys]},
                    [torch.frombuffer(bytearray(encs[k]), dtype=torch.uint8) for k in keys])

    def _parts(self, enc: bytes) -> list[tuple[str, Codec, bytes]]:
        h, arrays = unpack(enc)
        return [(k, self.rules[r][1], a.numpy().tobytes()) for k, r, a in zip(h["k"], h["r"], arrays, strict=True)]

    def decode(self, enc: bytes, bases: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {k: c.decode(e, bases[k]) for k, c, e in self._parts(enc)}

    def keys(self, enc: bytes) -> list[str]:
        return [k for k, _, _ in self._parts(enc)]

    def cost_macs(self, enc: bytes, bases: dict[str, torch.Tensor]) -> int:
        return sum(c.cost_macs(e, bases[k]) for k, c, e in self._parts(enc))
