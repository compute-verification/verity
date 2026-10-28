"""One layer of the frozen shape through the staging decode on one GPU, and the exact W8A8 GEMMs on what it staged.

    python layer.py --layout LAYOUT --weights M [--w W] (--encoded ENC --source HEX --commitment HEX | --store DIR
                    --salt HEX) --lib LIB --out OUT [--layer 0] [--steps toy,encode,stage,gemm] [--toy TOY]
                    [--encode-budget S] [--encode-blocks N] [--m 4096,16384]

- `LAYOUT` is the store's layout, p2_v1's fixture (`p2_v1/store.py layout`'s JSON), read by
  `layout.store_layout`. `M` is the weights stream's manifest (`e2e_h200.served_weights manifest`; with `--digests`,
  each tensor's SHA-256, which every staged tensor is held to) and `W` a file holding the stream's bytes at their stream
  offsets (`weights write`), at least the layer's; without `W`, the few stream bytes a run reads (the scales, a
  matrix's last label) come from the seeded generator (`--generator`, `--shape`, `--weights-seed`).
- `ENC` is an encoded store (`p2_v1/store.py encode`'s directory, memory-accounting's, read only): the layer's rows from
  its `seg/<s>.c` files, each held to its `seg/<s>.json`'s SHA-512, under the salt `setup.draw_salt` of the coin and
  commitment its `setup.json` is held to (`layout.store_salt`). `DIR` is a development store instead: `c.bin`, the
  encoded rows of the layer's blocks, and `c.json` (the first block, the count, how many are encoded so far, and the
  digests of the prefix, the layout and the weights manifest), encoded here under the public test salt `HEX`.
- **toy**: the fixture's toy store (`TOY`, `store-toy/`) through the same decode: each layer's staging buffer against
  `toy.json`'s digest, each negative block refused with its flag (and the other layer not), another salt refused, and
  the spot vectors (blocks at the real store's indices, `spot.json`) decoded to their payloads' digests.
- **encode** extends the store in block order by the library's `p2v1_encode` for `S` seconds (or `N` blocks), each chunk
  decoded back on the device and held to the store's W (`layout.payloads`) before it is written, so one layer's encode
  (1,307,915 blocks, two roots a block) can span several leases. QKV and O are its first 262,401 blocks, up the next
  522,757, down the last 522,757.
- **stage**, for each matrix whose blocks are all encoded: its rows onto the GPU, a buffer poisoned with 0xA5,
  `stage.decode_matrix`, the staged bytes against the stream's (mismatching bytes and blocks) and each of its tensors
  against the manifest's SHA-256, and `reps` timed decodes;
  once all four are encoded, `stage.decode_layer` into one poisoned staging buffer, timed and held to the four matrices
  at their staging offsets and zero between them; and controls on the last 64 blocks of a matrix: other keys, one
  corrupted row, one row at p and a last block nonzero past the matrix's end, each refused with its flag, and the honest
  blocks passed.
- **gemm**, for each batch M and each staged matrix: x [M, K] s8 and sx [M] f32 seeded on the device (rows 0, 1 and 2
  all -128, all 127 and all 0), the stream's f32 scales sw (held to the manifest's SHA-256), `int_mm.linear` (s8_linear's int_mm backend, eager epilogue) timed,
  every output word against `gemm.twin`, and samples for `check.py`. With all four, the implied step at each M: 52 layers
  x (the layer's `decode_layer` + its four GEMMs).

Writes `OUT/layer.json` (one record per step and matrix) and `OUT/samples-<matrix>-m<M>.npz`; exits 1 if a staged byte
or digest, a GEMM word, the toy or a control isn't as it should be.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "s8_linear"))
import gemm as G
import int_mm as IM
import layout as L
import stage as S
from p2_v1 import gpu

LAYERS = 52
DEVICE = torch.device("cuda")


class Out:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.mkdir(parents=True, exist_ok=True)

    def record(self, **row) -> None:
        print(json.dumps(row), flush=True)
        with open(self.path / "layer.json", "a") as f:
            f.write(json.dumps(row) + "\n")


def digest(doc: dict) -> str:
    return hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest()


def sha512(t: torch.Tensor) -> str:
    return hashlib.sha512(S.flat(t).cpu().numpy().tobytes()).hexdigest()


def to_device(data: bytes) -> torch.Tensor:
    return torch.frombuffer(bytearray(data), dtype=torch.uint8).to(DEVICE)


def sync() -> None:
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


class DevStore:
    """`c.bin` and `c.json` of the layer's blocks."""

    def __init__(self, root: Path, lay: L.StoreLayout, blocks: range, weights_sha256: str) -> None:
        self.root, self.span = root, blocks
        self.c, self.meta_path = root / "c.bin", root / "c.json"
        want = {"first": blocks.start, "blocks": len(blocks), "row": L.ROW,
                "prefix_sha256": hashlib.sha256(lay.prefix).hexdigest(), "layout_sha256": digest(lay.manifest()),
                "weights_sha256": weights_sha256}
        root.mkdir(parents=True, exist_ok=True)
        if self.meta_path.exists():
            self.meta = json.loads(self.meta_path.read_text())
            if {k: self.meta.get(k) for k in want} != want:
                raise SystemExit(f"{self.meta_path} is another store's ({self.meta}), not {want}")
        else:
            self.meta = {**want, "done": 0}
            with open(self.c, "wb") as f:
                f.truncate(len(blocks) * L.ROW)
            self.save()

    def save(self) -> None:
        tmp = self.meta_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.meta, sort_keys=True))
        os.replace(tmp, self.meta_path)

    @property
    def encoded(self) -> range:
        return range(self.span.start, self.span.start + self.meta["done"])

    def rows(self, blocks: range) -> torch.Tensor:
        if not (self.encoded.start <= blocks.start <= blocks.stop <= self.encoded.stop):
            raise ValueError(f"blocks {blocks} are not all encoded ({self.encoded})")
        n = len(blocks)
        a = np.fromfile(self.c, dtype=np.uint8, count=n * L.ROW, offset=(blocks.start - self.span.start) * L.ROW)
        return torch.from_numpy(a.reshape(n, L.ROW)).to(DEVICE)


class EncodedStore:
    """An encoded store's blocks (`p2_v1/store.py encode`'s directory, memory-accounting's, read only): the layer's rows
    from its `seg/<s>.c` files, each file held to the SHA-512 its `seg/<s>.json` records."""

    def __init__(self, root: Path, lay: L.StoreLayout, blocks: range, setup: dict) -> None:
        self.root, self.span, self.spec = root, blocks, lay.spec
        bps = lay.params.blocks_per_segment
        self.segments = range(blocks.start // bps, (blocks.stop - 1) // bps + 1)
        self.meta = {"store": str(root), "segments": [self.segments.start, self.segments.stop], "done": len(blocks),
                     "commitment": setup["commitment"], "program": setup.get("program"),
                     "prefix_sha256": hashlib.sha256(lay.prefix).hexdigest(), "layout_sha256": digest(lay.manifest())}

    @property
    def encoded(self) -> range:
        return self.span

    def rows(self, blocks: range) -> torch.Tensor:
        if not (self.span.start <= blocks.start <= blocks.stop <= self.span.stop):
            raise ValueError(f"blocks {blocks} are not all in the store's {self.span}")
        bps = self.spec.blocks_per_segment
        out = torch.empty(len(blocks), L.ROW, dtype=torch.uint8, device=DEVICE)
        for s in range(blocks.start // bps, (blocks.stop - 1) // bps + 1):
            meta_path = L.segment_path(self.root, s, "json")
            if not meta_path.exists():
                raise SystemExit(f"segment {s} of {self.root} is not done: no {meta_path.name}")
            data = np.fromfile(L.segment_path(self.root, s, "c"), dtype=np.uint8)
            if hashlib.sha512(data).hexdigest() != json.loads(meta_path.read_text())["sha512"]:
                raise SystemExit(f"segment {s}'s rows are not the SHA-512 its {meta_path.name} records")
            lo, hi = max(blocks.start, s * bps), min(blocks.stop, (s + 1) * bps)
            part = data[(lo - s * bps) * L.ROW:(hi - s * bps) * L.ROW].reshape(hi - lo, L.ROW)
            out[lo - blocks.start:hi - blocks.start] = torch.from_numpy(part).to(DEVICE)
        return out


def encode(args, lay: L.StoreLayout, src: L.Weights, read: L.Read, dev: DevStore, out: Out) -> None:
    kern = gpu.Kernels(str(args.lib), "", lay.spec, lay.prefix)
    kern.raised()
    t0, n0 = time.monotonic(), dev.meta["done"]
    stop = len(dev.span) if args.encode_blocks is None else min(len(dev.span), n0 + args.encode_blocks)
    with open(dev.c, "r+b") as fc:
        while dev.meta["done"] < stop and time.monotonic() - t0 < args.encode_budget:
            j0 = dev.span.start + dev.meta["done"]
            n = min(args.encode_chunk, dev.span.start + stop - j0)
            payload = to_device(L.payloads(lay, src, read, j0, n)).view(n, L.PAYLOAD)
            c = S.encode_rows(kern, payload, j0)
            m = kern.decode(c, j0)
            if kern.raised() or bool((m[:, :L.LEAD] != 0).any()) or not torch.equal(m[:, L.LEAD:], payload):
                raise SystemExit(f"blocks [{j0}, {j0 + n}) don't decode back to W's bytes after p2v1_encode")
            os.pwrite(fc.fileno(), c.cpu().numpy().tobytes(), (j0 - dev.span.start) * L.ROW)
            os.fsync(fc.fileno())
            dev.meta["done"] += n
            dev.save()
    sync()
    dt = time.monotonic() - t0
    k = dev.meta["done"] - n0
    out.record(step="encode", blocks=k, seconds=round(dt, 3), blocks_per_s=round(k / dt, 1) if dt else None,
               done=dev.meta["done"], of=len(dev.span))


def ready(lay: L.StoreLayout, dev: DevStore, layer: int) -> list[str]:
    e = dev.encoded
    return [n for n in L.MATRICES
            if e.start <= lay.tensor(layer, n).first_block and lay.tensor(layer, n).block_range.stop <= e.stop]


def against_stream(lay, src: L.Weights, read: L.Read, layer: int, name: str, buf) -> dict:
    """Mismatching bytes and blocks of the staged matrix `buf` against the stream's bytes of it."""
    t = lay.tensor(layer, name)
    ne = S.flat(buf) != to_device(src.read(src.parts(layer, name), 0, t.size, read))
    k = int(ne.sum())
    bad = ((ne.nonzero().flatten() + t.offset) // L.PAYLOAD).unique().tolist() if k else []
    return {"bytes": t.size, "mismatched_bytes": k, "mismatched_blocks": len(bad), "first_bad_blocks": bad[:8]}


def against_manifest(src: L.Weights, layer: int, name: str, host: np.ndarray, scales: bool = False) -> dict | None:
    """Whether each of the matrix's tensors (or their scales) in `host`, one after another, has the SHA-256 the weights
    manifest gives it (`manifest --digests`, the digests memory-accounting held W to); None when it gives none."""
    rows = [src.rows[f"l{layer}.{p}{'.scale' if scales else ''}"] for p in L.PARTS[name]]
    if any("sha256" not in r for r in rows):
        return None
    out, at = {}, 0
    for p, r in zip(L.PARTS[name], rows):
        out[p] = hashlib.sha256(host[at:at + r["nbytes"]]).hexdigest() == r["sha256"]
        at += r["nbytes"]
    return out


def scales(src: L.Weights, read: L.Read, layer: int, name: str, n: int) -> torch.Tensor:
    parts = src.parts(layer, name, scales=True)
    if sum(size for _, size in parts) != 4 * n:
        raise ValueError(f"{name}'s scales in the stream are not {n} f32 words")
    data = src.read(parts, 0, 4 * n, read)
    if not all((against_manifest(src, layer, name, np.frombuffer(data, dtype=np.uint8), scales=True) or {}).values()):
        raise SystemExit(f"{name}'s scales are not the SHA-256 the weights manifest gives them")
    return to_device(data).view(torch.float32)


def stage(args, lay, src, read: L.Read, dev, out: Out, layer: int) -> tuple[dict, float | None, bool]:
    staged, whole_s, ok = {}, None, True
    names = ready(lay, dev, layer)
    out.record(step="stage", ready=names, encoded=[dev.encoded.start, dev.encoded.stop])
    if not names:
        return staged, whole_s, True
    store = S.Store(lay, dev.rows(dev.encoded), dev.encoded.start, args.lib, chunk=args.chunk)
    for name in names:
        t = lay.tensor(layer, name)
        w = torch.empty(t.n, t.k, dtype=torch.int8, device=DEVICE)
        S.flat(w).fill_(0xA5)
        before = store.blocks_decoded
        S.decode_matrix(store, layer, name, w)
        blocks = store.blocks_decoded - before
        host = S.flat(w).cpu().numpy()
        cw = against_stream(lay, src, read, layer, name, w) if args.w else None
        dw = against_manifest(src, layer, name, host)
        if cw is None and dw is None:
            raise SystemExit("nothing to hold the staged bytes to: give --w or a weights manifest with digests")
        ok &= (cw is None or cw["mismatched_bytes"] == 0) and (dw is None or all(dw.values()))
        ts = S.timed(lambda: S.decode_matrix(store, layer, name, w), args.reps, DEVICE, warmup=1)
        out.record(step="stage", matrix=name, tensor=t.name, shape=[t.n, t.k], first_block=t.first_block, blocks=blocks,
                   weights=cw, manifest_sha256=dw, decode_s=[round(x, 6) for x in ts],
                   decode_gb_s=round(t.size / min(ts) / 1e9, 2), rows_gb_s=round(t.blocks * L.ROW / min(ts) / 1e9, 2),
                   sha512=hashlib.sha512(host).hexdigest())
        del host
        staged[name] = (w, scales(src, read, layer, name, t.n))
    if len(names) == len(L.MATRICES):
        buf = torch.full((lay.staging_bytes(layer),), 0xA5, dtype=torch.uint8, device=DEVICE)
        S.decode_layer(store, layer, buf)
        same, gaps, at = True, 0, 0
        for t in sorted(lay.layer(layer), key=lambda t: t.staging_offset):
            same &= torch.equal(buf[t.staging_offset:t.staging_offset + t.size], S.flat(staged[t.matrix][0]))
            gaps += int((buf[at:t.staging_offset] != 0).sum())
            at = t.staging_offset + t.size
        ts = S.timed(lambda: S.decode_layer(store, layer, buf), args.reps, DEVICE, warmup=1)
        whole_s = min(ts)
        nbytes = sum(t.size for t in lay.layer(layer))
        out.record(step="stage", matrix="layer", call="decode_layer", bytes=nbytes, staging_bytes=buf.numel(),
                   matches_matrices=bool(same), nonzero_gap_bytes=gaps, decode_s=[round(x, 6) for x in ts],
                   decode_gb_s=round(nbytes / whole_s / 1e9, 2), sha512=sha512(buf))
        ok &= bool(same) and gaps == 0
        del buf
    ok &= controls(args, lay, src, read, store, out, layer, names[0])
    store = None
    return staged, whole_s, ok


def run_window(st: S.Store, offset: int, size: int) -> tuple[int, str | None]:
    """The flags `st` refuses W bytes [offset, offset + size), a tensor's last bytes, with (0 when it decodes)."""
    st.decode_into(offset, size, torch.empty(size, dtype=torch.uint8, device=st.rows.device), end=True)
    try:
        st.check()
    except S.StoreError as e:
        return e.flags, str(e)
    return 0, None


def controls(args, lay, src, read: L.Read, store: S.Store, out: Out, layer: int, name: str) -> bool:
    """Other keys, a corrupted row, a row at p and a last block nonzero past the matrix's end, each among the last 64
    blocks of `name`: each refused with its flag bit among those it raises (other keys' garbage is nonzero past the end
    too), and the honest blocks passed."""
    t = lay.tensor(layer, name)
    last = t.block_range.stop - 1
    j0 = max(t.first_block, last - 63)
    n = last + 1 - j0
    rows = store.rows[j0 - store.first:last + 1 - store.first].clone()
    other = L.StoreLayout(lay.params, lay.tensors, bytes([lay.prefix[0] ^ 1]) + lay.prefix[1:])
    corrupt, at_p = rows.clone(), rows.clone()
    corrupt[n // 2, 1000] ^= 0x01
    at_p[n // 2] = torch.from_numpy(np.frombuffer(S.P_BYTES, dtype=np.uint8).copy()).to(rows.device)
    cases = {"other_keys": (other, rows, S.WIDE), "corrupt_row": (lay, corrupt, S.WIDE), "row_at_p": (lay, at_p, S.WIDE),
             "honest": (lay, rows, 0)}
    used = t.offset + t.size - last * L.PAYLOAD
    if used < L.PAYLOAD:
        payload = bytearray(L.payloads(lay, src, read, last, 1))
        payload[-1] ^= 1
        padded = rows.clone()
        padded[n - 1] = S.encode_rows(store.kernels, to_device(bytes(payload)).view(1, L.PAYLOAD), last)[0]
        cases["padding"] = (lay, padded, S.PADDING)
    res = {}
    for label, (lo, rs, want) in cases.items():
        flags, why = run_window(S.Store(lo, rs, j0, args.lib), j0 * L.PAYLOAD, t.offset + t.size - j0 * L.PAYLOAD)
        res[label] = {"flags": flags, "want": want, "why": why}
    ok = all(v["flags"] & v["want"] if v["want"] else not v["flags"] for v in res.values())
    out.record(step="controls", matrix=name, blocks=[j0, last + 1], last_block_used=used, results=res, ok=ok)
    return ok


def negative_row(neg: dict, c_star: bytes) -> bytes:
    """A toy negative's block: its `c` in hex, ``"p"`` for p itself, or C*'s own block with byte `xor[0]` xored by
    `xor[1]` (the fixture's `store.negative`)."""
    i = neg["block"]
    if "xor" in neg:
        c = bytearray(c_star[L.ROW * i:L.ROW * (i + 1)])
        c[neg["xor"][0]] ^= neg["xor"][1]
        return bytes(c)
    return S.P_BYTES if neg["c"] == "p" else bytes.fromhex(neg["c"])


def toy(args, out: Out) -> bool:
    doc = json.loads((args.toy / "toy.json").read_text())
    spot = json.loads((args.toy / "spot.json").read_text())
    c_star, spot_c = (args.toy / "c-star.bin").read_bytes(), (args.toy / "spot-c.bin").read_bytes()
    salt = bytes.fromhex(doc["salt"])
    lay = L.store_layout(doc, L.registered_prefix(salt))
    pins = {"prefix": lay.prefix.hex() == doc["prefix"] == spot["prefix"],
            "c_star": hashlib.sha512(c_star).hexdigest() == doc["c_star_sha512"],
            "spot_c": hashlib.sha512(spot_c).hexdigest() == spot["c_sha512"]}
    rows = to_device(c_star).view(-1, L.ROW)

    def staging(lo: L.StoreLayout, rs: torch.Tensor, layer: int) -> tuple[str | None, int]:
        buf = torch.full((lo.staging_bytes(layer),), 0xA5, dtype=torch.uint8, device=DEVICE)
        try:
            S.decode_layer(S.Store(lo, rs, 0, args.lib), layer, buf)
        except S.StoreError as e:
            return None, e.flags
        return sha512(buf), 0

    layers = [{"layer": l, "bytes": size, "match": staging(lay, rows, l) == (sha, 0)} for l, _, size, sha in doc["staging"]]
    negatives = []
    for neg in doc["negatives"]:
        bad = rows.clone()
        bad[neg["block"]] = to_device(negative_row(neg, c_star))
        flags, rest = staging(lay, bad, neg["layer"])[1], staging(lay, bad, 1 - neg["layer"])[1]
        negatives.append({"id": neg["id"], "flag": neg["flag"], "flags": flags, "other_layer_flags": rest,
                          "ok": flags == 1 << neg["flag"] and rest == 0})
    other = bytes(salt[:-1]) + bytes([salt[-1] ^ 1])
    other_salt = staging(L.store_layout(doc, L.registered_prefix(other)), rows, 0)[1]
    q = spot["params"]
    real = L.StoreLayout(L.S.P2Params(q["w"], q["c0"], q["blocks_per_segment"], q["segments"]), (), lay.prefix)
    spots = []
    srows = to_device(spot_c).view(-1, L.ROW)
    for j, v in enumerate(spot["vectors"]):
        st = S.Store(real, srows[j:j + 1].clone(), v["i"], args.lib)
        m = st.decode_rows(v["i"], v["i"] + 1)
        try:
            st.check()
            flags = 0
        except S.StoreError as e:
            flags = e.flags
        spots.append({"id": v["id"], "i": v["i"], "flags": flags, "match": sha512(m[0, L.LEAD:]) == v["m_sha512"]})
    ok = bool(all(pins.values()) and all(x["match"] for x in layers) and all(x["ok"] for x in negatives)
              and other_salt & S.WIDE and all(x["match"] and not x["flags"] for x in spots))
    out.record(step="toy", ok=ok, pins=pins, staging=layers, negatives=negatives, other_salt_flags=other_salt, spots=spots)
    return ok


def activations(M: int, K: int, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randint(-128, 128, (M, K), dtype=torch.int8, device=DEVICE, generator=gen)
    x[0], x[1], x[2] = -128, 127, 0
    sx = (torch.rand(M, dtype=torch.float32, device=DEVICE, generator=gen) + 1) * 2.0 ** -8
    return x, sx


def gemm(args, staged: dict, out: Out) -> tuple[dict[int, float], bool]:
    """Each staged matrix at each batch M: the layer's GEMM seconds per M (all four matrices), and whether every output
    word equals the twin's."""
    gen = torch.Generator(device=DEVICE).manual_seed(args.seed)
    cpu = torch.Generator().manual_seed(args.seed)
    totals, ok = {}, True
    for M in args.m:
        total = 0.0
        for name, (w, sw) in staged.items():
            N, K = w.shape
            x, sx = activations(M, K, gen)
            y = IM.linear(x, sx, w, sw, "eager")
            t_int = S.timed(lambda: torch._int_mm(x, w.t()), args.reps, DEVICE, warmup=1)
            t_all = S.timed(lambda: IM.linear(x, sx, w, sw, "eager"), args.reps, DEVICE, warmup=1)
            res = G.twin(x, sx, w, sw, y)
            ok &= res["mismatches"] == 0
            sm = G.samples(x, sx, w, sw, y, args.samples, cpu)
            np.savez_compressed(args.out / f"samples-{name}-m{M}.npz", K=K, **sm)
            out.record(step="gemm", matrix=name, m=M, n=N, k=K, twin=res,
                       int_mm_s=[round(t, 6) for t in t_int], scaled_mm_s=[round(t, 6) for t in t_all],
                       tops=round(2 * M * N * K / min(t_all) / 1e12, 1))
            total += min(t_all)
            x = sx = y = None
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()
        if len(staged) == len(L.MATRICES):
            totals[M] = total
    return totals, ok


def main(argv=None) -> int:
    global DEVICE
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--layout", type=Path, required=True, help="the store's layout JSON (p2_v1/store.py layout)")
    ap.add_argument("--weights", type=Path, required=True, help="the weights stream's manifest")
    ap.add_argument("--w", type=Path, help="the weights stream's bytes at their stream offsets (else the generator's)")
    ap.add_argument("--generator", default="e2e_h200.served_weights", help="the seeded generator, without --w")
    ap.add_argument("--shape", default="h200", help="its shape")
    ap.add_argument("--weights-seed", default="text:verity/cmp92/h200/weights/v1", help="its seed")
    store = ap.add_mutually_exclusive_group(required=True)
    store.add_argument("--store", type=Path, help="the development store")
    store.add_argument("--encoded", type=Path, help="an encoded store's directory (p2_v1/store.py encode's), read only")
    ap.add_argument("--salt", help="the development store's public test salt, 24 bytes hex")
    ap.add_argument("--source", help="the encoded store's coin (hex), drawn after its commitment")
    ap.add_argument("--commitment", help="the encoded store's commitment (hex)")
    ap.add_argument("--lib", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--toy", type=Path, help="the fixture's store-toy/ directory, for the toy step")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--steps", default="encode,stage,gemm")
    ap.add_argument("--encode-budget", type=float, default=600.0)
    ap.add_argument("--encode-blocks", type=int)
    ap.add_argument("--encode-chunk", type=int, default=8192)
    ap.add_argument("--chunk", type=int, default=1 << 16)
    ap.add_argument("--m", type=lambda s: [int(v) for v in s.split(",")], default=[4096, 16384],
                    help="the batches, comma-separated")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--samples", type=int, default=24)
    ap.add_argument("--seed", type=int, default=12)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    DEVICE = torch.device(args.device)
    steps = args.steps.split(",")
    out = Out(args.out)
    ok = True
    if "toy" in steps:
        if args.toy is None:
            ap.error("the toy step needs --toy")
        ok &= toy(args, out)
    if not {"encode", "stage", "gemm"} & set(steps):
        return 0 if ok else 1
    manifest = json.loads(args.weights.read_text())
    try:
        lay, setup = L.open_store(args.layout, manifest, args.encoded, args.source, args.commitment, args.salt)
    except ValueError as e:
        raise SystemExit(str(e)) from None
    src = L.Weights.of(manifest)
    read = L.reader(args.w) if args.w else L.generator_reader(args.generator, args.shape, args.weights_seed)
    for t in lay.layer(args.layer):
        if src.shape(args.layer, t.matrix) != (t.n, t.k):
            raise SystemExit(f"{t.name} is [{t.n}, {t.k}] in the store, {list(src.shape(args.layer, t.matrix))} in the stream")
    blocks = lay.blocks(args.layer)
    if args.encoded is not None:
        if "encode" in steps:
            ap.error("an encoded store is read only: no encode step")
        dev = EncodedStore(args.encoded, lay, blocks, setup)
    else:
        dev = DevStore(args.store, lay, blocks, digest(manifest))
    out.record(step="start", layer=args.layer, blocks=[blocks.start, blocks.stop], encoded=dev.meta["done"],
               staging_bytes=lay.staging_bytes(args.layer), store=str(args.encoded or args.store),
               kind="encoded" if args.encoded else "development",
               device=torch.cuda.get_device_name() if DEVICE.type == "cuda" else str(DEVICE), torch=torch.__version__,
               prefix_sha256=dev.meta["prefix_sha256"], layout_sha256=dev.meta["layout_sha256"])
    if "encode" in steps:
        encode(args, lay, src, read, dev, out)
    staged, decode_s = {}, None
    if "stage" in steps or "gemm" in steps:
        staged, decode_s, stage_ok = stage(args, lay, src, read, dev, out, args.layer)
        ok &= stage_ok
    gemm_s, gemm_ok = gemm(args, staged, out) if "gemm" in steps else ({}, True)
    ok &= gemm_ok
    if decode_s is not None:
        for M, g in gemm_s.items():
            out.record(step="summary", m=M, layers=LAYERS, decode_layer_s=round(decode_s, 6), gemm_layer_s=round(g, 6),
                       step_s=round(LAYERS * (decode_s + g), 3), gemm_only_step_s=round(LAYERS * g, 3),
                       decode_only_step_s=round(LAYERS * decode_s, 3))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
