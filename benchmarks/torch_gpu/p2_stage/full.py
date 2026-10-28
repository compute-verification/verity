"""The whole encoded store through the staging decode on one GPU, one tensor at a time, each held to the plain weights'
digests; then the exact W8A8 GEMM on one layer's staged matrices.

    python full.py --layout LAYOUT --weights M (--w W | --generator G --shape S --weights-seed SEED)
                   --encoded ENC --source HEX --commitment HEX --lib LIB --out OUT [--layers 0-52] [--gemm-layer 51]
                   [--m 16384] [--reps 2] [--readers 6] [--hashers 4] [--n-chunk 16384]

`LAYOUT`, `M`, `ENC`, the coin and the commitment are `layer.py`'s: the store's layout, the weights manifest with
each stream tensor's SHA-256 (`manifest --digests`, the digests memory-accounting held W to), and memory-accounting's
encoded store, read only, under the salt `setup.draw_salt` of the coin and commitment its `setup.json` is held to.

- **stage**, each tensor of `--layers` in block order (a layer's QKV, O, up and down; the head is the last layer):
  - its rows from the store's segment files onto the GPU (`Segments`: each file read once, in order, `readers` ahead in
    threads, and held to its `seg/<s>.json`'s SHA-512);
  - `stage.decode_matrix` into a buffer poisoned with 0xA5 (a warm-up, where a refusal is recorded), then `reps`
    decodes, each into the buffer poisoned again and timed alone by CUDA events;
  - the last timed decode's bytes back to the host, and each stream tensor it stacks held to the manifest's SHA-256 (in
    `hashers` threads, while the next tensor decodes);
  - its rows and its buffer freed, so one tensor's rows and staging buffer are on the device at a time.
- **summary**: each tensor's fastest decode summed, which is the decode of a step measured tensor by tensor, and the peak
  HBM torch allocated and reserved over the pass.
- **receipts** (`--receipts`): the remote bandwidth audit's pass over the units the store's tensors lie in
  (`sketch.plan_units`; the encoded padding past the head is in no receipt) beside the stage pass (`sketch.py`), under
  the seed `receipts.seed(CHALLENGE, PASS, PREVIOUS)`: each tensor's rows taken in whole units (up to three blocks more
  at each end), the sketch attached to the warm-up decode (the pass's σ), and its timed decodes with the sketch and
  without, and the sketch alone; the units of the tensors `--layers` leaves out read from their segments and sketched
  alone; then y and the root under fresh salts. The summary's `receipts` gives the root, the pass's timestamps, each
  stage's time summed like the decode's, and the overhead on the decode; with `--keep`, σ, y, the salts and the pass go
  to that directory for `bw_ref.py`.
- **gemm**, after the pass, `--gemm-layer`'s matrices staged again one at a time: x [M, K] s8 and sx [M] f32 seeded on the
  device (rows 0, 1 and 2 all -128, all 127 and all 0), the stream's f32 scales (held to the manifest's SHA-256),
  `int_mm.linear` (s8_linear's int_mm backend, fused epilogue on CUDA) on `n_chunk` output channels at a time, every word against `gemm.twin`, and
  samples for `check.py`; and the pass's peak HBM.

Writes `OUT/full.json` (one record per tensor, the summary, the GEMMs) and `OUT/samples-<matrix>-m<M>.npz`; exits 1 if a
staged tensor's digest, a refusal or a GEMM word isn't as it should be, and stops at a segment whose SHA-512 differs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "s8_linear"))
import gemm as G  # noqa: E402
import layout as L  # noqa: E402
import stage as S  # noqa: E402

DEVICE = torch.device("cuda")


class Out:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.mkdir(parents=True, exist_ok=True)

    def record(self, **row) -> None:
        print(json.dumps(row), flush=True)
        with open(self.path / "full.json", "a") as f:
            f.write(json.dumps(row) + "\n")


def sync() -> None:
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


class Segments:
    """An encoded store's segment files `needed`, in that order, each read once and held to its `seg/<s>.json`'s SHA-512,
    `ahead` of the one in use in a thread pool; `rows` asks for blocks in increasing order."""

    def __init__(self, root: Path, bps: int, needed: list[int], pool: ThreadPoolExecutor, ahead: int) -> None:
        self.root, self.bps, self.pool, self.ahead = root, bps, pool, ahead
        self.queue = deque(needed)
        self.pending: dict[int, Future] = {}
        self.held: dict[int, np.ndarray] = {}
        self.read = 0

    def _load(self, s: int) -> np.ndarray:
        meta = L.segment_path(self.root, s, "json")
        if not meta.exists():
            raise SystemExit(f"segment {s} of {self.root} is not done: no {meta.name}")
        data = np.fromfile(L.segment_path(self.root, s, "c"), dtype=np.uint8)
        if hashlib.sha512(data).hexdigest() != json.loads(meta.read_text())["sha512"]:
            raise SystemExit(f"segment {s}'s rows are not the SHA-512 its {meta.name} records")
        return data

    def segment(self, s: int) -> np.ndarray:
        while self.queue and len(self.pending) < self.ahead + 1:
            k = self.queue.popleft()
            self.pending[k] = self.pool.submit(self._load, k)
        if s not in self.held:
            if s not in self.pending:
                raise ValueError(f"segment {s} isn't next: the rows are asked for in block order")
            self.held[s] = self.pending.pop(s).result()
            self.read += 1
        return self.held[s]

    def rows(self, blocks: range) -> torch.Tensor:
        """The rows of `blocks` ([n, 2056] uint8) on the device; the segments before the one before the last they touch
        let go (the next tensor's rows in whole units can start a few blocks before these end)."""
        out = torch.empty(len(blocks), L.ROW, dtype=torch.uint8, device=DEVICE)
        last = (blocks.stop - 1) // self.bps
        for s in range(blocks.start // self.bps, last + 1):
            data = self.segment(s)
            lo, hi = max(blocks.start, s * self.bps), min(blocks.stop, (s + 1) * self.bps)
            part = data[(lo - s * self.bps) * L.ROW:(hi - s * self.bps) * L.ROW]
            if len(part) != (hi - lo) * L.ROW:
                raise ValueError(f"segment {s} ends before block {hi}")
            out[lo - blocks.start:hi - blocks.start] = torch.from_numpy(part.reshape(hi - lo, L.ROW)).to(DEVICE)
        for s in [s for s in self.held if s < last - 1]:
            del self.held[s]
        return out


def segments_of(tensors: list[L.Tensor], bps: int, ranges: list[range] | None = None) -> list[int]:
    out: list[int] = []
    for r in ranges if ranges is not None else [t.block_range for t in tensors]:
        for s in range(r.start // bps, (r.stop - 1) // bps + 1):
            if not out or s > out[-1]:
                out.append(s)
    return out


FULL_CHALLENGE = hashlib.sha512(b"verity/p2-stage/bw-full/challenge/v0").hexdigest()[:64]
FULL_PREVIOUS = hashlib.sha512(b"verity/p2-stage/bw-full/previous/v0").hexdigest()


class Audit:
    """The bandwidth audit's pass over the whole store (`--receipts`): its seed, tree, salts and `sketch.Pass`."""

    def __init__(self, args, lay: L.StoreLayout) -> None:
        import sketch as K

        from verity.core.protocols.poub import receipts as R

        self.K, self.R = K, R
        self.store_units = lay.params.blocks_per_segment * lay.params.segments // K.PER
        try:
            self.units = K.plan_units(lay.tensors)
        except ValueError as e:
            raise SystemExit(str(e)) from None
        self.tensor_end = max(t.block_range.stop for t in lay.tensors)
        self.challenge, self.index, self.previous = bytes.fromhex(args.challenge), args.pass_index, bytes.fromhex(args.previous)
        self.seed = R.seed(self.challenge, self.index, self.previous)
        self.tree = K.Tree(self.units, DEVICE)
        self.salts_raw = args.salts.read_bytes() if args.salts else os.urandom(R.SALT_BYTES * self.tree.G)
        if len(self.salts_raw) != R.SALT_BYTES * self.tree.G:
            raise SystemExit(f"--salts holds {len(self.salts_raw)} bytes, not {self.tree.G} leaves' {R.SALT_BYTES}")
        self.salts = self.tree.salts(self.salts_raw)
        self.P = K.Pass(self.seed, self.units, DEVICE)
        self.gap_s = 0.0
        self.gap_units = 0

    def rows_range(self, t: L.Tensor) -> range:
        """`t`'s blocks, widened to the units they lie in (blocks past the last whole unit are in none)."""
        u = self.P.units_of(t.first_block, t.block_range.stop)
        return range(min(t.first_block, self.K.PER * u.start), max(t.block_range.stop, self.K.PER * u.stop))

    def items(self, tensors: list[L.Tensor]) -> list[tuple[range, L.Tensor | None]]:
        """In block order: each tensor with its rows' range, and the runs of the pass's units none of `tensors`' rows
        hold (those of the tensors `--layers` leaves out), at most a decode chunk at a time, with None."""
        out: list[tuple[range, L.Tensor | None]] = []
        at = 0

        def gap(u0: int, u1: int) -> None:
            for v in range(u0, u1, (1 << 16) // self.K.PER):
                out.append((range(self.K.PER * v, self.K.PER * min(u1, v + (1 << 16) // self.K.PER)), None))

        for t in tensors:
            u = self.P.units_of(t.first_block, t.block_range.stop)
            if u.start > at:
                gap(at, u.start)
            out.append((self.rows_range(t), t))
            at = max(at, u.stop)
        if at < self.units:
            gap(at, self.units)
        return out

    def read_gap(self, segs: Segments, r: range, reps: int) -> None:
        rows = segs.rows(r)
        self.gap_s += min(S.timed(partial(self.P.read, rows, r.start, r.start, r.stop), reps, DEVICE))
        self.gap_units += len(r) // self.K.PER
        del rows

    def finish(self, reps: int) -> dict:
        P, tree = self.P, self.tree
        y_s = min(S.timed(partial(self.K.receipts, self.seed, P.sigma, 0, tree.y), reps, DEVICE))
        root_s = min(S.timed(partial(tree.root, self.seed, self.salts), reps, DEVICE))
        finish_s = min(S.timed(partial(P.finish, tree, self.salts), reps, DEVICE))
        return {"seed": self.seed.hex(), "challenge": self.challenge.hex(), "pass": self.index,
                "previous": self.previous.hex(), "units": self.units, "store_units": self.store_units,
                "padding_units_left_out": self.store_units - self.units,
                "blocks_past_head_in_plan": self.K.PER * self.units - self.tensor_end, "leaves": tree.G, "root": P.root.hex(),
                "start_ns": P.start_ns, "end_ns": P.end_ns, "pass_wall_s": round((P.end_ns - P.start_ns) / 1e9, 3),
                "gap_units": self.gap_units, "gap_sketch_s": round(self.gap_s, 6), "y_s": round(y_s, 6),
                "root_s": round(root_s, 6), "finish_s": round(finish_s, 6),
                "sigma_sha512": hashlib.sha512(P.sigma.cpu().numpy().astype("<i4").tobytes()).hexdigest(),
                "y_sha512": hashlib.sha512(tree.y.cpu().numpy().tobytes()).hexdigest(),
                "salts_sha512": hashlib.sha512(self.salts_raw).hexdigest()}

    def keep(self, path: Path, rec: dict) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.P.sigma.cpu().numpy().astype("<i4").tofile(path / "sigma.bin")
        self.tree.y.cpu().numpy().tofile(path / "y.bin")
        (path / "salts.bin").write_bytes(self.salts_raw)
        (path / "pass.json").write_text(json.dumps(rec, indent=1) + "\n")


def digests(src: L.Weights, t: L.Tensor, host: np.ndarray) -> dict[str, bool]:
    """Whether each stream tensor `t` stacks has, in `host` (its staged bytes), the SHA-256 the weights manifest gives."""
    out, at = {}, 0
    for name in t.stream_names:
        r = src.rows[name]
        out[name] = hashlib.sha256(host[at:at + r["nbytes"]]).hexdigest() == r["sha256"]
        at += r["nbytes"]
    return out


def decode(args, lay: L.StoreLayout, segs: Segments, t: L.Tensor, audit: Audit | None = None
           ) -> tuple[torch.Tensor | None, S.Store, dict]:
    """`t`'s rows onto the device and decoded into a fresh buffer: the buffer (None if refused), the store and timings.
    With `audit`, the rows are of whole units and the decode sketches them."""
    t0 = time.perf_counter()
    r = audit.rows_range(t) if audit else t.block_range
    rows = segs.rows(r)
    sync()
    rows_s = time.perf_counter() - t0
    store = S.Store(lay, rows, r.start, args.lib, chunk=args.chunk)
    store.sketch = audit.P if audit else None
    w = torch.empty(t.n, t.k, dtype=torch.int8, device=DEVICE)
    S.flat(w).fill_(0xA5)
    rec = {"layer": t.layer, "tensor": t.name, "shape": [t.n, t.k], "blocks": [t.first_block, t.block_range.stop],
           "bytes": t.size, "rows_s": round(rows_s, 4)}
    try:
        S.decode_matrix(store, t.layer, t.name, w)
    except S.StoreError as e:
        rec["refused"] = {"flags": e.flags, "why": str(e)}
        return None, store, rec
    return w, store, rec


def sketched(store: S.Store, sk, layer: int, name: str, w: torch.Tensor) -> None:
    store.sketch = sk
    S.decode_matrix(store, layer, name, w)


def stage_pass(args, lay: L.StoreLayout, src: L.Weights, tensors: list[L.Tensor], out: Out) -> tuple[dict, bool]:
    bps = lay.params.blocks_per_segment
    ok, done, total_s, layers = True, deque(), 0.0, {}
    audit = Audit(args, lay) if args.receipts else None
    items = audit.items(tensors) if audit else [(t.block_range, t) for t in tensors]
    with_s, sketch_s = 0.0, 0.0
    t_start = time.perf_counter()
    with ThreadPoolExecutor(args.readers) as readers, ThreadPoolExecutor(args.hashers) as hashers:
        segs = Segments(args.encoded, bps, segments_of(tensors, bps, [r for r, _ in items]), readers, args.readers)

        def drain(keep: int) -> None:
            nonlocal ok
            while len(done) > keep:
                rec, fut = done.popleft()
                rec["sha256"] = fut.result() if fut else None
                ok &= bool(rec["sha256"]) and all(rec["sha256"].values()) and "refused" not in rec
                out.record(step="stage", **rec)

        for r, t in items:
            if t is None:
                audit.read_gap(segs, r, args.reps)
                continue
            w, store, rec = decode(args, lay, segs, t, audit)
            fut = None
            if w is not None:
                poison = partial(S.flat(w).fill_, 0xA5)
                if audit:
                    tw = S.timed(partial(sketched, store, audit.P, t.layer, t.name, w), args.reps, DEVICE, poison)
                    tk = S.timed(partial(audit.P.read, store.rows, store.first, t.first_block, t.block_range.stop),
                                 args.reps, DEVICE)
                    rec.update(decode_sketch_s=[round(x, 6) for x in tw], sketch_s=[round(x, 6) for x in tk])
                    with_s, sketch_s = with_s + min(tw), sketch_s + min(tk)
                ts = S.timed(partial(sketched, store, None, t.layer, t.name, w), args.reps, DEVICE, poison)
                host = S.flat(w).to("cpu", copy=True).numpy()
                fut = hashers.submit(digests, src, t, host)
                rec.update(decode_s=[round(x, 6) for x in ts], decode_gb_s=round(t.size / min(ts) / 1e9, 2),
                           rows_gb_s=round(t.blocks * L.ROW / min(ts) / 1e9, 2))
                total_s += min(ts)
                layers[t.layer] = layers.get(t.layer, 0.0) + min(ts)
                del host
            del w, store
            done.append((rec, fut))
            drain(2 * args.hashers)
        drain(0)
        read = segs.read
    receipts = None
    if audit:
        receipts = audit.finish(args.reps)
        step = with_s + receipts["gap_sketch_s"] + receipts["finish_s"]
        receipts.update(decode_sketch_step_s=round(with_s, 6), sketch_step_s=round(sketch_s, 6),
                        decode_step_s=round(total_s, 6), receipts_step_s=round(step, 6),
                        overhead_s=round(step - total_s, 6), overhead_ratio=round((step - total_s) / total_s, 5),
                        step_ratio=round(step / total_s, 5))
        if args.keep:
            audit.keep(args.keep, receipts)
    wall = time.perf_counter() - t_start
    nbytes = sum(t.size for t in tensors)
    per = [v for k, v in layers.items() if k < lay.tensors[-1].layer]
    summary = {"tensors": len(tensors), "bytes": nbytes, "segments": read, "ok": bool(ok),
               "decode_step_s": round(total_s, 6), "decode_gb_s": round(nbytes / total_s / 1e9, 2) if total_s else None,
               "layer_s": {"min": round(min(per), 6), "max": round(max(per), 6), "mean": round(sum(per) / len(per), 6)}
               if per else None, "head_s": round(layers[lay.tensors[-1].layer], 6) if lay.tensors[-1].layer in layers
               else None, "wall_s": round(wall, 3), **peak(), **({"receipts": receipts} if receipts else {})}
    return summary, ok


def peak() -> dict:
    if DEVICE.type != "cuda":
        return {}
    free, total = torch.cuda.mem_get_info()
    return {"peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "device_free_bytes": free, "device_total_bytes": total}


def scales(src: L.Weights, read: L.Read, t: L.Tensor) -> torch.Tensor:
    """`t`'s f32 scales, one per output channel, from the stream and held to the manifest's SHA-256."""
    data = b""
    for name in t.stream_names:
        r = src.rows[f"{name}.scale"]
        part = read(r["offset"], r["nbytes"])
        if hashlib.sha256(part).hexdigest() != r["sha256"]:
            raise SystemExit(f"{name}'s scales are not the SHA-256 the weights manifest gives them")
        data += part
    if len(data) != 4 * t.n:
        raise ValueError(f"{t.name}'s scales are not {t.n} f32 words")
    return torch.frombuffer(bytearray(data), dtype=torch.float32).to(DEVICE)


def activations(M: int, K: int, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randint(-128, 128, (M, K), dtype=torch.int8, device=DEVICE, generator=gen)
    x[0], x[1], x[2] = -128, 127, 0
    sx = (torch.rand(M, dtype=torch.float32, device=DEVICE, generator=gen) + 1) * 2.0 ** -8
    return x, sx


def gemm_pass(args, lay: L.StoreLayout, src: L.Weights, read: L.Read, out: Out) -> bool:
    import int_mm

    epilogue = "fused" if DEVICE.type == "cuda" else "eager"
    tensors = [t for t in lay.layer(args.gemm_layer) if t.matrix is not None]
    sws = {t.name: scales(src, read, t) for t in tensors}
    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    gen = torch.Generator(device=DEVICE).manual_seed(args.seed)
    cpu = torch.Generator().manual_seed(args.seed)
    ok = True
    bps = lay.params.blocks_per_segment
    with ThreadPoolExecutor(args.readers) as readers:
        segs = Segments(args.encoded, bps, segments_of(tensors, bps), readers, args.readers)
        for t in tensors:
            w, store, rec = decode(args, lay, segs, t)
            del store
            if w is None:
                out.record(step="gemm", matrix=t.matrix, **rec)
                ok = False
                continue
            host = S.flat(w).cpu().numpy()
            dw = digests(src, t, host)
            del host
            ok &= all(dw.values())
            sw = sws[t.name]
            for M in args.m:
                x, sx = activations(M, t.k, gen)
                words, bad, first, secs, parts = 0, 0, [], 0.0, []
                for n0 in range(0, t.n, args.n_chunk):
                    n1 = min(t.n, n0 + args.n_chunk)
                    wc, swc = w[n0:n1], sw[n0:n1]
                    y = int_mm.linear(x, sx, wc, swc, epilogue)
                    secs += min(S.timed(partial(int_mm.linear, x, sx, wc, swc, epilogue), args.reps, DEVICE))
                    res = G.twin(x, sx, wc, swc, y)
                    words, bad = words + res["words"], bad + res["mismatches"]
                    first += [{**f, "n": f["n"] + n0} for f in res["first"]][:8 - len(first)]
                    sm = G.samples(x, sx, wc, swc, y, args.samples, cpu)
                    sm["n"] = sm["n"] + n0
                    parts.append(sm)
                    del y
                ok &= bad == 0
                np.savez_compressed(args.out / f"samples-{t.matrix}-m{M}.npz", K=t.k,
                                    **{k: np.concatenate([p[k] for p in parts]) for k in parts[0]})
                out.record(step="gemm", layer=t.layer, matrix=t.matrix, m=M, n=t.n, k=t.k, n_chunk=args.n_chunk,
                           epilogue=epilogue, sha256=dw, twin={"words": words, "mismatches": bad, "first": first},
                           linear_s=round(secs, 6), tops=round(2 * M * t.n * t.k / secs / 1e12, 1) if secs else None)
                del x, sx
            del w
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()
    out.record(step="gemm-peak", layer=args.gemm_layer, **peak())
    return ok


def layer_set(spec: str, last: int) -> set[int]:
    """`0-52`, `0,5,52` or both: the layers it names, refused past `last`."""
    out: set[int] = set()
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out |= set(range(int(a), int(b or a) + 1))
    if not out or max(out) > last or min(out) < 0:
        raise ValueError(f"layers {spec!r}: the store's are 0 .. {last}")
    return out


def main(argv=None) -> int:
    global DEVICE
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--layout", type=Path, required=True, help="the store's layout JSON (p2_v1/store.py layout)")
    ap.add_argument("--weights", type=Path, required=True, help="the weights stream's manifest, with digests")
    ap.add_argument("--w", type=Path, help="the weights stream's bytes at their stream offsets (else the generator's)")
    ap.add_argument("--generator", default="e2e_h200.served_weights", help="the seeded generator, without --w")
    ap.add_argument("--shape", default="h200", help="its shape")
    ap.add_argument("--weights-seed", default="text:verity/cmp92/h200/weights/v1", help="its seed")
    ap.add_argument("--encoded", type=Path, required=True, help="the encoded store's directory, read only")
    ap.add_argument("--source", required=True, help="the store's coin (hex), drawn after its commitment")
    ap.add_argument("--commitment", required=True, help="the store's commitment (hex)")
    ap.add_argument("--lib", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--layers", help="the layers to stage (default all, the head included)")
    ap.add_argument("--gemm-layer", type=int, help="the layer whose matrices the GEMM runs on (none if not given)")
    ap.add_argument("--m", type=lambda s: [int(v) for v in s.split(",")], default=[16384], help="the GEMM's batches")
    ap.add_argument("--n-chunk", type=int, default=16384, help="output channels a GEMM call takes")
    ap.add_argument("--chunk", type=int, default=1 << 16, help="blocks a decode call takes")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--readers", type=int, default=6)
    ap.add_argument("--hashers", type=int, default=4)
    ap.add_argument("--samples", type=int, default=24, help="GEMM words sampled per call for check.py")
    ap.add_argument("--seed", type=int, default=12)
    ap.add_argument("--receipts", action="store_true", help="the bandwidth audit's pass beside the stage pass")
    ap.add_argument("--challenge", default=FULL_CHALLENGE, help="the pass's period challenge (32 bytes, hex)")
    ap.add_argument("--pass-index", type=int, default=0, help="the pass's index in its period")
    ap.add_argument("--previous", default=FULL_PREVIOUS, help="the root before (the registration's digest at pass 0)")
    ap.add_argument("--salts", type=Path, help="with --receipts: the leaves' salts, 192 bytes each (default fresh)")
    ap.add_argument("--keep", type=Path, help="with --receipts: σ, y, the salts and the pass for bw_ref.py")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    DEVICE = torch.device(args.device)
    manifest = json.loads(args.weights.read_text())
    try:
        lay, setup = L.open_store(args.layout, manifest, args.encoded, args.source, args.commitment, None)
        last = lay.tensors[-1].layer
        chosen = layer_set(args.layers, last) if args.layers else set(range(last + 1))
    except ValueError as e:
        raise SystemExit(str(e)) from None
    src = L.Weights.of(manifest)
    tensors = [t for t in lay.tensors if t.layer in chosen]
    for t in tensors:
        try:
            src.tensor_parts(t)
        except (KeyError, ValueError) as e:
            raise SystemExit(str(e)) from None
        if any("sha256" not in src.rows[n] for n in t.stream_names):
            raise SystemExit(f"the weights manifest gives {t.name} no SHA-256 to hold it to: pass `manifest --digests`'s")
    out = Out(args.out)
    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    out.record(step="start", store=str(args.encoded), commitment=setup["commitment"], program=setup.get("program"),
               layers=sorted(chosen), tensors=len(tensors), bytes=sum(t.size for t in tensors),
               prefix_sha256=hashlib.sha256(lay.prefix).hexdigest(),
               device=torch.cuda.get_device_name() if DEVICE.type == "cuda" else str(DEVICE), torch=torch.__version__,
               **({"device_free_bytes": torch.cuda.mem_get_info()[0]} if DEVICE.type == "cuda" else {}))
    summary, ok = stage_pass(args, lay, src, tensors, out)
    out.record(step="summary", **summary)
    if args.gemm_layer is not None:
        read = L.reader(args.w) if args.w else L.generator_reader(args.generator, args.shape, args.weights_seed)
        ok &= gemm_pass(args, lay, src, read, out)
    rc = summary.get("receipts")
    print(f"full {'PASS' if ok else 'FAIL'}: {summary['tensors']} tensors staged and held to the manifest's SHA-256, "
          f"decode step {summary['decode_step_s']} s"
          + (f"; receipts pass {rc['pass']} over the tensors' {rc['units']} units of the store's {rc['store_units']} "
             f"({rc['padding_units_left_out']} padding units left out), root {rc['root'][:16]}..., step with receipts "
             f"{rc['receipts_step_s']} s (decode with the sketch {rc['decode_sketch_step_s']} s"
             + (f" + unstaged tensors' {rc['gap_units']} units {rc['gap_sketch_s']} s" if rc["gap_units"] else "")
             + f" + y and root {rc['finish_s']} s), overhead {rc['overhead_s']} s, ratio {rc['step_ratio']}"
             if rc else ""), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
