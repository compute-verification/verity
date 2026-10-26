"""One layer of the frozen shape through the staging decode on one GPU, and the exact W8A8 GEMMs on what it staged.

    python layer.py --manifest M --w W --store DIR --lib LIB --salt HEX --out OUT [--layer 0] [--steps encode,stage,gemm]
                    [--encode-budget S] [--encode-blocks N] [--m 20480] [--reps 3]

- `M` is the weights stream's manifest (`verity_torch_loop.weights manifest`), read by `layout.layout_of` (each matrix's
  [N, K] from its tensors' shapes); `W` a file holding the stream's bytes at their stream offsets (`weights write`), at
  least the layer's blocks.
- `DIR` is the development store: `c.bin`, the encoded rows of the layer's span of blocks, and `c.json` (the first block,
  the count, how many are encoded so far, the digests of the prefix and the layout). Until memory-accounting's encoded
  store and its layout fixture (CMP-95 item 2) are on the card, the layer is encoded here, under the registered prefix of
  the public test salt `HEX`.
- **encode** extends the store in stream order by the library's `p2v1_encode` for `S` seconds (or `N` blocks), each chunk
  decoded back on the device and held to W before it is written, so one layer's encode (about 1.31 M blocks, two roots a
  block) can span several leases. QKV and O are the span's first 262 K blocks, up the next 523 K, down the last.
- **stage**, for each matrix whose blocks are all encoded: its rows onto the GPU, a staging buffer poisoned with 0xA5,
  `stage.decode_matrix` (and its scales), the staged bytes against W's on the device (mismatching bytes and blocks), and
  `reps` timed decodes; the layer's four matrices and scales timed back to back once all are encoded; and three controls
  the decode must refuse: other keys, one corrupted row, one row at p.
- **gemm**, for each staged matrix: x [M, K] s8 and sx [M] f32 seeded on the device (rows 0, 1 and 2 all -128, all 127
  and all 0), `gemm.scaled_mm` timed, every output word against `gemm.twin`, and samples for `check.py`. With all four,
  the implied step: 52 layers x (the layer's decode + its four GEMMs).

Writes `OUT/layer.json` (one record per step and matrix) and `OUT/samples-<matrix>.npz`.
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
import gemm as G  # noqa: E402
import layout as L  # noqa: E402
import stage as S  # noqa: E402
from p2_v1 import gpu  # noqa: E402

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


def read_w(path: Path, offset: int, size: int) -> bytes:
    with open(path, "rb") as f:
        data = os.pread(f.fileno(), size, offset)
    if len(data) != size:
        raise ValueError(f"{path} ends before W byte {offset + size}")
    return data


def to_device(data: bytes) -> torch.Tensor:
    return torch.frombuffer(bytearray(data), dtype=torch.uint8).to(DEVICE)


def sync() -> None:
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


def shapes(manifest: dict, layer: int) -> dict[str, tuple[int, int]]:
    """Each matrix's (K, N): its tensors' rows stacked, their common row length."""
    by = {t["name"]: t["shape"] for t in manifest["tensors"]}
    out = {}
    for name, parts in L.MATRICES.items():
        ss = [by[f"l{layer}.{p}"] for p in parts]
        if len({s[1] for s in ss}) != 1:
            raise ValueError(f"{name}'s tensors {parts} have different row lengths")
        out[name] = (ss[0][1], sum(s[0] for s in ss))
    return out


class DevStore:
    """`c.bin` and `c.json` of the layer's span."""

    def __init__(self, root: Path, lay: L.StoreLayout, span: range) -> None:
        self.root, self.span = root, span
        self.c, self.meta_path = root / "c.bin", root / "c.json"
        want = {"first": span.start, "blocks": len(span), "row": L.ROW,
                "prefix_sha256": hashlib.sha256(lay.prefix).hexdigest(),
                "layout_sha256": hashlib.sha256(json.dumps(lay.manifest(), sort_keys=True).encode()).hexdigest()}
        root.mkdir(parents=True, exist_ok=True)
        if self.meta_path.exists():
            self.meta = json.loads(self.meta_path.read_text())
            if {k: self.meta.get(k) for k in want} != want:
                raise SystemExit(f"{self.meta_path} is another store's ({self.meta}), not {want}")
        else:
            self.meta = {**want, "done": 0}
            with open(self.c, "wb") as f:
                f.truncate(len(span) * L.ROW)
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


def encode(args, lay: L.StoreLayout, dev: DevStore, out: Out) -> None:
    kern = gpu.Kernels(str(args.lib), "", L.SPEC, lay.prefix)
    kern.raised()
    t0, n0 = time.monotonic(), dev.meta["done"]
    stop = len(dev.span) if args.encode_blocks is None else min(len(dev.span), n0 + args.encode_blocks)
    with open(dev.c, "r+b") as fc:
        while dev.meta["done"] < stop and time.monotonic() - t0 < args.encode_budget:
            j0 = dev.span.start + dev.meta["done"]
            n = min(args.encode_chunk, dev.span.start + stop - j0)
            payload = to_device(read_w(args.w, j0 * L.PAYLOAD, n * L.PAYLOAD)).view(n, L.PAYLOAD)
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


def matrix_blocks(lay: L.StoreLayout, layer: int, name: str) -> range:
    rs = [L.blocks_of(p.offset, p.size) for s in (False, True) for p in lay.parts(layer, name, s)]
    return range(min(r.start for r in rs), max(r.stop for r in rs))


def ready(lay: L.StoreLayout, dev: DevStore, layer: int) -> list[str]:
    e = dev.encoded
    return [n for n in L.MATRICES if e.start <= matrix_blocks(lay, layer, n).start
            and matrix_blocks(lay, layer, n).stop <= e.stop]


def timed(fn, reps: int) -> list[float]:
    fn()
    sync()
    times = []
    for _ in range(reps):
        if DEVICE.type == "cuda":
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            fn()
            b.record()
            b.synchronize()
            times.append(a.elapsed_time(b) / 1e3)
        else:
            t0 = time.perf_counter()
            fn()
            times.append(time.perf_counter() - t0)
    return times


def staged_against_w(args, lay, layer, name, buf, scales) -> dict:
    """Mismatching bytes and blocks of `buf` (the staged matrix or its scales) against W's bytes."""
    got, at, bad_bytes, bad_blocks = S.flat(buf), 0, 0, set()
    for p in lay.parts(layer, name, scales):
        ref = to_device(read_w(args.w, p.offset, p.size))
        ne = got[at:at + p.size] != ref
        k = int(ne.sum())
        if k:
            bad_bytes += k
            idx = ne.nonzero().flatten()
            bad_blocks.update(((idx + p.offset) // L.PAYLOAD).unique().tolist())
        at += p.size
    return {"bytes": at, "mismatched_bytes": bad_bytes, "mismatched_blocks": len(bad_blocks),
            "first_bad_blocks": sorted(bad_blocks)[:8]}


def stage(args, lay, dev: DevStore, out: Out, layer: int, dims: dict) -> tuple[dict, float | None]:
    staged, whole_s = {}, None
    names = ready(lay, dev, layer)
    out.record(step="stage", ready=names, encoded=[dev.encoded.start, dev.encoded.stop])
    if not names:
        return staged, whole_s
    store = S.Store(lay, dev.rows(dev.encoded), dev.encoded.start, args.lib, chunk=args.chunk)
    for name in names:
        K, N = dims[name]
        w = torch.empty(N, K, dtype=torch.int8, device=DEVICE)
        sw = torch.empty(N, dtype=torch.float32, device=DEVICE)
        S.flat(w).fill_(0xA5)
        S.flat(sw).fill_(0xA5)
        before = store.blocks_decoded
        S.decode_matrix(store, layer, name, w)
        S.decode_scales(store, layer, name, sw)
        blocks = store.blocks_decoded - before
        cw, cs = staged_against_w(args, lay, layer, name, w, False), staged_against_w(args, lay, layer, name, sw, True)
        ts = timed(lambda: S.decode_matrix(store, layer, name, w), args.reps)
        nbytes = lay.nbytes(layer, name)
        out.record(step="stage", matrix=name, shape=[N, K], blocks=blocks, weights=cw, scales=cs,
                   decode_s=[round(t, 6) for t in ts], decode_gb_s=round(nbytes / min(ts) / 1e9, 2),
                   rows_gb_s=round(len(matrix_blocks(lay, layer, name)) * L.ROW / min(ts) / 1e9, 2),
                   sha256=hashlib.sha256(S.flat(w).cpu().numpy().tobytes()).hexdigest())
        staged[name] = (w, sw)
    if len(names) == len(L.MATRICES):
        def whole():
            for name in L.MATRICES:
                S.decode_matrix(store, layer, name, staged[name][0])
                S.decode_scales(store, layer, name, staged[name][1])
        ts = timed(whole, args.reps)
        whole_s = min(ts)
        nbytes = sum(lay.nbytes(layer, n) + lay.nbytes(layer, n, True) for n in L.MATRICES)
        out.record(step="stage", matrix="layer", bytes=nbytes, decode_s=[round(t, 6) for t in ts],
                   decode_gb_s=round(nbytes / whole_s / 1e9, 2))
    controls(args, lay, store, out, layer, names[0])
    del store
    return staged, whole_s


def refused(fn) -> str | None:
    try:
        fn()
    except S.StoreError as e:
        return str(e)
    return None


def controls(args, lay, store: S.Store, out: Out, layer: int, name: str) -> None:
    """Other keys, a corrupted row and a row at p, each on a window of `name`'s blocks: all must be refused."""
    j0 = matrix_blocks(lay, layer, name).start
    n = min(64, store.blocks.stop - j0)
    rows = store.rows[j0 - store.first:j0 - store.first + n].clone()
    other = S.Store(lay.__class__(lay.layout, bytes([lay.prefix[0] ^ 1]) + lay.prefix[1:]), rows, j0, args.lib)
    bad = rows.clone()
    bad[n // 2, 1000] ^= 0x01
    corrupt = S.Store(lay, bad, j0, args.lib)
    high = rows.clone()
    high[n - 1] = torch.from_numpy(np.frombuffer(S.P_BYTES, dtype=np.uint8).copy()).to(DEVICE)
    at_p = S.Store(lay, high, j0, args.lib)
    res = {}
    for label, st in (("other_keys", other), ("corrupt_row", corrupt), ("row_at_p", at_p)):
        st.decode_rows(j0, j0 + n)
        res[label] = refused(st.check)
    honest = S.Store(lay, rows, j0, args.lib)
    honest.decode_rows(j0, j0 + n)
    res["honest"] = refused(honest.check)
    out.record(step="controls", matrix=name, blocks=[j0, j0 + n], refused=res,
               ok=all(res[k] for k in ("other_keys", "corrupt_row", "row_at_p")) and res["honest"] is None)


def activations(M: int, K: int, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randint(-128, 128, (M, K), dtype=torch.int8, device=DEVICE, generator=gen)
    x[0], x[1], x[2] = -128, 127, 0
    sx = (torch.rand(M, dtype=torch.float32, device=DEVICE, generator=gen) + 1) * 2.0 ** -8
    return x, sx


def gemm(args, staged: dict, dims: dict, out: Out) -> float | None:
    gen = torch.Generator(device=DEVICE).manual_seed(args.seed)
    cpu = torch.Generator().manual_seed(args.seed)
    total = 0.0
    for name, (w, sw) in staged.items():
        K, N = dims[name]
        x, sx = activations(args.m, K, gen)
        y = G.scaled_mm(x, sx, w, sw)
        t_int = timed(lambda: torch._int_mm(x, w.t()), args.reps)
        t_all = timed(lambda: G.scaled_mm(x, sx, w, sw), args.reps)
        res = G.twin(x, sx, w, sw, y)
        sm = G.samples(x, sx, w, sw, y, args.samples, cpu)
        np.savez_compressed(args.out / f"samples-{name}.npz", K=K, **sm)
        out.record(step="gemm", matrix=name, m=args.m, n=N, k=K, twin=res,
                   int_mm_s=[round(t, 6) for t in t_int], scaled_mm_s=[round(t, 6) for t in t_all],
                   tops=round(2 * args.m * N * K / min(t_all) / 1e12, 1))
        total += min(t_all)
        del x, sx, y
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
    return total if len(staged) == len(L.MATRICES) else None


def main(argv=None) -> int:
    global DEVICE
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--w", type=Path, required=True)
    ap.add_argument("--store", type=Path, required=True)
    ap.add_argument("--lib", type=Path, required=True)
    ap.add_argument("--salt", required=True, help="the public test salt, 24 bytes hex")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--steps", default="encode,stage,gemm")
    ap.add_argument("--encode-budget", type=float, default=600.0)
    ap.add_argument("--encode-blocks", type=int)
    ap.add_argument("--encode-chunk", type=int, default=8192)
    ap.add_argument("--chunk", type=int, default=1 << 16)
    ap.add_argument("--m", type=int, default=20480)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--samples", type=int, default=24)
    ap.add_argument("--seed", type=int, default=12)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    DEVICE = torch.device(args.device)
    manifest = json.loads(args.manifest.read_text())
    lay = L.layout_of(manifest, L.registered_prefix(bytes.fromhex(args.salt)))
    dims = shapes(manifest, args.layer)
    span = lay.span(args.layer)
    dev = DevStore(args.store, lay, span)
    out = Out(args.out)
    out.record(step="start", layer=args.layer, span=[span.start, span.stop], encoded=dev.meta["done"],
               device=torch.cuda.get_device_name() if DEVICE.type == "cuda" else str(DEVICE), torch=torch.__version__,
               prefix_sha256=dev.meta["prefix_sha256"])
    steps = args.steps.split(",")
    if "encode" in steps:
        encode(args, lay, dev, out)
    staged, decode_s = stage(args, lay, dev, out, args.layer, dims) if "stage" in steps or "gemm" in steps else ({}, None)
    gemm_s = gemm(args, staged, dims, out) if "gemm" in steps else None
    if decode_s is not None and gemm_s is not None:
        out.record(step="summary", m=args.m, layers=LAYERS, decode_layer_s=round(decode_s, 6), gemm_layer_s=round(gemm_s, 6),
                   step_s=round(LAYERS * (decode_s + gemm_s), 3), gemm_only_step_s=round(LAYERS * gemm_s, 3))
    return 0


if __name__ == "__main__":
    sys.exit(main())
