"""P2 v4's store of a weights stream: `C* = Enc(W, salt)` under the registered scheme `p2-16448/v4`
(`verity.core.protocols.pous.schemes.p2`), encoded on every GPU of one node by `libp2v1.so` (`gpu.Kernels`) and written as
canonical segment files, with the setup's public outputs.

W is a seeded weights stream regenerated on the node from its generator (`verity_torch_loop.weights`: `stream(shape, seed,
lo, hi)`, `stream_bytes(shape)`, `program(shape, seed)`), never copied in, zero-padded to whole segments of 2^16 blocks of
2,054 payload bytes. In setup order (`setup`'s docstring):

1. `commit`: `setup.commit` of W under the scheme's manifest, before any salt exists;
2. the salt is `setup.draw_salt(source, commitment)`, `source` the auditor's coin drawn after the commitment;
3. `encode`: every segment of C* on the GPUs, each checked by the device's decode of the whole segment and by
   `reference.decode` of two blocks, then written.

The store (`--store DIR`, scratch outside the run directory):

- `setup.json`: the manifest, the stream, the commitment, the source and the salt;
- `seg/<s>.c`: segment s, its 2^16 blocks of 2,056 bytes in order;
- `rows/<s>.rows`: their `sha512/row/v2` digests, 64 bytes each (the memory challenger's table, `VerifierKey.from_row_digests`);
- `seg/<s>.json`: the segment's digests and roots, written last, so a segment that has it is done and a rerun resumes;
- `SHA512SUMS` and `store.json`, the public outputs.

The public outputs (`store.json`):

- `root`, C*'s root: `setup.server_root`'s tree (`verifier.leaf_hash` leaves, RFC 6962's shape). It is computed as the RFC
  6962 tree over the segments' subtree roots, which is the same tree because every segment is a power of two of leaves.
- `w_root`, W's salted root: the same tree over W's blocks, each block's 2,054 payload bytes digested as `sha512/row/v2`
  at 16,432 bits in the same salted leaf (segment, index).
- `tie`: `tagged_sha512(TIE_TAG, canonical JSON)` of the manifest, the program, the stream's length, the commitment, the
  salt, both roots and the weights' registration digest (the generator's `registration_digest`, null until it is given).

`verify` rereads a store: every file's SHA-512, its row digests, the segments' roots, the top roots and the tie, and with
`--decode` decodes every segment (`--p2-lib` on the GPU, else the scheme's codec) into the commitment: the public encoder's
setup check (`setup.verifier_key_from_encoding`). `spot` decodes random blocks by `reference.decode` and compares them with
W's bytes regenerated from the seed. `tie` recomputes `store.json`'s tie with a registration digest.
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import hashlib
import importlib
import importlib.util
import json
import multiprocessing as mp
import os
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Iterator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from verity.core.protocols.pous import setup, verifier  # noqa: E402
from verity.core.protocols.pous.labels import salt_value, tag  # noqa: E402
from verity.core.protocols.pous.schemes import p2 as S  # noqa: E402
from verity.ml.commitments.identity import canonical_json_bytes, tagged_sha512  # noqa: E402
from verity.ml.commitments.rowleaf import sha512_row_v2_prefix  # noqa: E402

FORMAT = "verity-pous/p2/store/v1"
TIE_TAG = "verity-pous/p2/store-tie/v1"
TIE_FIELDS = ("manifest", "program", "stream_bytes", "commitment", "salt", "root", "w_root", "registration")
GENERATOR = "verity_torch_loop.weights"
SPOTS_PER_SEGMENT = 2


# --- W: the generator's stream, zero-padded to whole segments ------------------------------------------------------------

def load_generator(spec: str) -> Any:
    """`spec`: the generator's module file (`.py`), or its module name."""
    if not spec.endswith(".py"):
        return importlib.import_module(spec)
    name = "_p2_store_generator"
    if name not in sys.modules:
        found = importlib.util.spec_from_file_location(name, spec)
        if found is None or found.loader is None:
            raise ImportError(f"no generator module at {spec}")
        mod = importlib.util.module_from_spec(found)
        sys.modules[name] = mod
        found.loader.exec_module(mod)
    return sys.modules[name]


def git_blob(path: str) -> str | None:
    """A module file's git blob id, which names it at any commit that has it."""
    if not path.endswith(".py"):
        return None
    data = Path(path).read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


class Weights:
    """W: the generator's stream at a shape and seed, zero-padded past its end."""

    def __init__(self, generator: str, shape: str, seed: bytes) -> None:
        self.generator, self.shape_name, self.seed = generator, shape, seed
        self.mod = load_generator(generator)
        self.shape = getattr(self.mod, "ALIASES", {}).get(shape) or self.mod.SHAPES[shape]
        self.size = self.mod.stream_bytes(self.shape)
        self.program = self.mod.program(self.shape, seed)

    def read(self, lo: int, hi: int) -> bytes:
        top = min(hi, self.size)
        data = self.mod.stream(self.shape, self.seed, lo, top) if lo < top else b""
        return data + bytes(hi - max(lo, top))


_WEIGHTS: dict[tuple[str, str, bytes], Weights] = {}


def _weights(generator: str, shape: str, seed: bytes) -> Weights:
    key = (generator, shape, seed)
    if key not in _WEIGHTS:
        _WEIGHTS[key] = Weights(generator, shape, seed)
    return _WEIGHTS[key]


def _read(job: tuple[str, str, bytes, int, int]) -> bytes:
    return _weights(*job[:3]).read(job[3], job[4])


def layout(size: int, w: int = S.POINT[0], c0: int = S.POINT[1], bps: int = S.POINT[2]) -> S.P2Params:
    """The registered parameters with the fewest segments that hold `size` bytes."""
    if bps & (bps - 1):
        raise ValueError("a segment is a power of two of blocks, so its subtree is a node of C*'s tree")
    base = S.P2Params(w, c0, bps)
    return base.with_segments(max(1, -(-size // base.segment_bytes)))


def params_of(manifest: dict[str, Any]) -> S.P2Params:
    if manifest.get("scheme") != "p2" or manifest.get("expander") != S.SHA512_CTR:
        raise ValueError(f"not a p2 v4 manifest: {manifest}")
    q = S.P2Params(manifest["w"], manifest["c0"], manifest["blocks_per_segment"], manifest["segments"])
    if q.manifest() != manifest:
        raise ValueError(f"the manifest is not P2Params' own: {manifest}")
    return q


def segments_of(weights: Weights, params: S.P2Params, workers: int) -> Iterator[bytes]:
    """W's segments in order, each generated by one of `workers` forked processes, at most `workers + 4` held at once."""
    sb = params.segment_bytes
    jobs = iter([(weights.generator, weights.shape_name, weights.seed, s * sb, (s + 1) * sb) for s in range(params.segments)])
    if workers <= 1:
        yield from map(_read, jobs)
        return
    with mp.get_context("fork").Pool(workers) as pool:
        pending = collections.deque(pool.apply_async(_read, (j,)) for j in [next(jobs) for _ in range(min(workers + 4, params.segments))])
        while pending:
            seg = pending.popleft().get()
            nxt = next(jobs, None)
            if nxt is not None:
                pending.append(pool.apply_async(_read, (nxt,)))
            yield seg


# --- digests and roots -----------------------------------------------------------------------------------------------------

def row_digests(data: bytes | memoryview, width: int) -> bytes:
    """Each `width`-byte row's `sha512/row/v2` digest, concatenated."""
    head = hashlib.sha512(sha512_row_v2_prefix(8 * width))
    view, out = memoryview(data), bytearray()
    for lo in range(0, len(view), width):
        h = head.copy()
        h.update(view[lo:lo + width])
        out += h.digest()
    return bytes(out)


def segment_root(params: S.P2Params, salt: bytes, s: int, rows: bytes) -> bytes:
    """The subtree root over segment `s`'s leaves (`verifier.row_leaf` of each row digest), a perfect tree."""
    n, sha = params.blocks_per_segment, hashlib.sha512
    if len(rows) != 64 * n:
        raise ValueError(f"segment {s} has {n} row digests")
    tag_s = b"\x00" + tag(salt_value(salt), s).to_bytes(32, "little")
    level = [sha(tag_s + v.to_bytes(2, "little") + rows[64 * v:64 * v + 64]).digest() for v in range(n)]
    while len(level) > 1:
        level = [verifier.node_hash(level[i], level[i + 1]) for i in range(0, len(level), 2)]
    return level[0]


def top_root(roots: list[bytes]) -> bytes:
    return verifier.MerkleTree(roots).root


def tie(doc: dict[str, Any]) -> str:
    return tagged_sha512(TIE_TAG, canonical_json_bytes({k: doc[k] for k in TIE_FIELDS})).hex()


def spot_blocks(salt: bytes, s: int, n: int, count: int = SPOTS_PER_SEGMENT) -> list[int]:
    """Segment `s`'s spot-checked blocks at encode: its first, then `count − 1` drawn from the salt and `s`."""
    rng = random.Random(hashlib.sha512(b"verity-pous/p2/store-spot/v1\0" + salt + s.to_bytes(8, "big")).digest())
    return sorted({0, *(rng.randrange(n) for _ in range(count - 1))})


def reference_decode(params: S.P2Params, prefix: bytes, s: int, j: int, block: bytes) -> bytes:
    """Block `j` of segment `s` decoded by `reference.decode` (the library's reference), as its payload bytes."""
    from p2_v1 import reference as R

    P = R.Params(params.w, params.c0, params.payload_bytes, params.blocks_per_segment, params.segments)
    k, t, _ = R.keys_after(prefix, P, s, j)
    m = R.decode(P, int.from_bytes(block, "big"), k, t)
    if m >> (8 * params.payload_bytes):
        raise ValueError(f"block {j} of segment {s} decodes to more than {params.payload_bytes} bytes")
    return m.to_bytes(params.payload_bytes, "big")


# --- the store's files ---------------------------------------------------------------------------------------------------

def seg_path(store: Path, s: int, ext: str) -> Path:
    return store / ("rows" if ext == "rows" else "seg") / f"{s:05d}.{ext}"


def write_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def dump(path: Path, doc: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_atomic(path, (json.dumps(doc, indent=1, sort_keys=True) + "\n").encode())


def finish_segment(store: Path, params: S.P2Params, salt: bytes, s: int, w_seg: bytes, c_seg: bytes,
                   extra: dict[str, Any]) -> dict[str, Any]:
    """Write segment `s`'s files, its JSON last."""
    nb, pb = params.block_bytes, params.payload_bytes
    rows = row_digests(c_seg, nb)
    meta = {"segment": s, "sha512": hashlib.sha512(c_seg).hexdigest(), "rows_sha512": hashlib.sha512(rows).hexdigest(),
            "root": segment_root(params, salt, s, rows).hex(),
            "w_root": segment_root(params, salt, s, row_digests(w_seg, pb)).hex(), **extra}
    write_atomic(seg_path(store, s, "c"), c_seg)
    write_atomic(seg_path(store, s, "rows"), rows)
    dump(seg_path(store, s, "json"), meta)
    return meta


def done_segments(store: Path, params: S.P2Params) -> dict[int, dict[str, Any]]:
    out = {}
    for s in range(params.segments):
        p = seg_path(store, s, "json")
        if p.exists() and seg_path(store, s, "c").exists():
            out[s] = json.loads(p.read_text())
    return out


# --- encoders --------------------------------------------------------------------------------------------------------------

class CpuEncoder:
    """The scheme's own codec (`schemes.p2.encode_segment`): the tests' widths, where the library has no build."""

    def __init__(self, params: S.P2Params, salt: bytes) -> None:
        self.params, self.salt = params, salt

    def encode(self, s: int, w_seg: bytes) -> tuple[bytes, dict[str, Any]]:
        return b"".join(S.encode_segment(self.params, None, self.salt, s, w_seg)), {"encoder": "schemes.p2"}


class GpuEncoder:
    """`libp2v1.so` on CUDA device `device`: a whole segment a call, its flags read and its decode compared with W."""

    def __init__(self, params: S.P2Params, salt: bytes, lib: str, device: int) -> None:
        import torch

        from p2_v1 import reference as R
        from p2_v1.gpu import Kernels

        torch.cuda.set_device(device)
        self.torch, self.params = torch, params
        P = R.Params(params.w, params.c0, params.payload_bytes, params.blocks_per_segment, params.segments)
        self.K = Kernels(lib, "", P, S.expand_prefix(params, salt))
        self.name = torch.cuda.get_device_name(device)

    def encode(self, s: int, w_seg: bytes) -> tuple[bytes, dict[str, Any]]:
        torch, q = self.torch, self.params
        n, nb, pb = q.blocks_per_segment, q.block_bytes, q.payload_bytes
        m = torch.zeros(n, nb, dtype=torch.uint8)
        m[:, nb - pb:] = torch.frombuffer(bytearray(w_seg), dtype=torch.uint8).view(n, pb)
        m = m.to(self.K.device)
        first = s * n
        t0 = time.perf_counter()
        c = self.K.encode(m, first)
        flags = self.K.raised()
        encode_s = time.perf_counter() - t0
        if flags:
            raise RuntimeError(f"segment {s}: the encode kernels raised flags {flags} (bit 0 a tweak at or above p, bit 1 no root)")
        if not torch.equal(self.K.decode(c, first), m):
            raise RuntimeError(f"segment {s}: the device's decode of its encoding is not W")
        if self.K.raised():
            raise RuntimeError(f"segment {s}: the decode kernels raised flags")
        return c.cpu().numpy().tobytes(), {"encoder": "libp2v1", "device": self.name, "encode_s": encode_s,
                                           "device_round_trip": True}


def _worker(index: int, cfg: dict[str, Any], todo: Any, done: Any) -> None:
    """Encode the segments `todo` hands out until it hands out None, one GPU (or the CPU codec) a process."""
    try:
        store, salt = Path(cfg["store"]), bytes.fromhex(cfg["salt"])
        params = params_of(cfg["manifest"])
        weights = Weights(cfg["generator"], cfg["shape"], cfg["seed"])
        enc = GpuEncoder(params, salt, cfg["p2_lib"], index) if cfg["p2_lib"] else CpuEncoder(params, salt)
        prefix, sb = S.expand_prefix(params, salt), params.segment_bytes
        nb, pb = params.block_bytes, params.payload_bytes
        with concurrent.futures.ThreadPoolExecutor(1) as gen, concurrent.futures.ThreadPoolExecutor(1) as post:
            s = todo.get()
            w_next = gen.submit(weights.read, s * sb, (s + 1) * sb) if s is not None else None
            writing = None
            while s is not None:
                w_seg = w_next.result()
                nxt = todo.get()
                w_next = gen.submit(weights.read, nxt * sb, (nxt + 1) * sb) if nxt is not None else None
                t0 = time.perf_counter()
                c_seg, extra = enc.encode(s, w_seg)
                extra["wall_s"] = time.perf_counter() - t0
                spots = []
                for j in spot_blocks(salt, s, params.blocks_per_segment):
                    got = reference_decode(params, prefix, s, j, c_seg[j * nb:(j + 1) * nb])
                    if got != w_seg[j * pb:(j + 1) * pb]:
                        raise RuntimeError(f"segment {s} block {j}: reference.decode of the encoding is not W's payload")
                    spots.append(j)
                extra.update(reference_spots=spots, worker=index)
                if writing is not None:
                    done.put(writing.result())
                writing = post.submit(finish_segment, store, params, salt, s, w_seg, c_seg, extra)
                s = nxt
            if writing is not None:
                done.put(writing.result())
    except BaseException as e:  # noqa: BLE001 - any failure is reported to the parent as this worker's
        done.put({"worker": index, "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()})
    done.put({"worker": index, "exit": True})


# --- commands --------------------------------------------------------------------------------------------------------------

def _bytes_arg(text: str) -> bytes:
    """``hex:...``, ``file:PATH`` (its bytes, a trailing newline stripped; a hex file is read as hex) or ``text:...``."""
    kind, _, v = text.partition(":")
    if kind == "hex":
        return bytes.fromhex(v)
    if kind == "file":
        return Path(v).read_bytes().rstrip(b"\n")
    if kind == "hexfile":
        return bytes.fromhex(Path(v).read_text().strip())
    if kind == "text":
        return v.encode()
    raise argparse.ArgumentTypeError("hex:..., hexfile:PATH, file:PATH or text:...")


def _setup_doc(a: argparse.Namespace, weights: Weights, params: S.P2Params) -> dict[str, Any]:
    return {"format": FORMAT, "manifest": params.manifest(), "scheme": S.P2(params).name, "generator": a.generator,
            "generator_blob": git_blob(a.generator), "shape": weights.shape_name, "program": weights.program,
            "stream_bytes": weights.size, "w_bytes": params.w_bytes, "c_bytes": params.blocks * params.block_bytes}


def commit_cmd(a: argparse.Namespace) -> dict[str, Any]:
    """`setup.json` with the commitment to W (reused when the store already has it for the same W and parameters)."""
    store = Path(a.store)
    store.mkdir(parents=True, exist_ok=True)
    weights = Weights(a.generator, a.shape, a.seed)
    params = layout(weights.size, *a.point)
    doc = _setup_doc(a, weights, params)
    path = store / "setup.json"
    if path.exists():
        old = json.loads(path.read_text())
        if {k: old.get(k) for k in doc if k != "generator"} != {k: v for k, v in doc.items() if k != "generator"}:
            raise SystemExit(f"{path} is another W's or another layout's setup: use a fresh --store")
        if old.get("commitment"):
            print(f"commit: reusing {path}'s commitment", file=sys.stderr)
            return old
    t0 = time.perf_counter()
    doc["commitment"] = setup.commit(params, segments_of(weights, params, a.workers)).hex()
    doc["commit_s"] = time.perf_counter() - t0
    dump(path, doc)
    print(json.dumps({"commitment": doc["commitment"], "segments": params.segments, "commit_s": doc["commit_s"]}))
    return doc


def encode_cmd(a: argparse.Namespace) -> int:
    store, out = Path(a.store), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    doc = commit_cmd(a)
    params = params_of(doc["manifest"])
    if a.source is None:
        raise SystemExit("encode needs --source: the auditor's coin, drawn after the commitment")
    salt = setup.draw_salt(a.source, bytes.fromhex(doc["commitment"]))
    if doc.get("salt") not in (None, salt.hex()):
        raise SystemExit(f"{store} was encoded under another salt: use a fresh --store")
    doc.update(source=a.source.hex(), salt=salt.hex(), salt_domain=setup.SALT_DOMAIN)
    dump(store / "setup.json", doc)
    for d in ("seg", "rows"):
        (store / d).mkdir(exist_ok=True)
    have = done_segments(store, params)
    todo_list = [s for s in range(params.segments) if s not in have]
    workers = a.gpus if a.p2_lib else a.cpu_encoders
    print(f"encode: {len(todo_list)} of {params.segments} segments on {workers} {'GPUs' if a.p2_lib else 'CPU encoders'}",
          file=sys.stderr, flush=True)
    ctx = mp.get_context("spawn")
    todo, done = ctx.Queue(), ctx.Queue()
    for s in todo_list:
        todo.put(s)
    for _ in range(workers):
        todo.put(None)
    cfg = {"store": str(store), "salt": salt.hex(), "manifest": doc["manifest"], "generator": a.generator,
           "shape": a.shape, "seed": a.seed, "p2_lib": a.p2_lib}
    t0 = time.perf_counter()
    procs = [ctx.Process(target=_worker, args=(g, cfg, todo, done)) for g in range(workers)]
    for p in procs:
        p.start()
    exited, errors, metas = 0, [], dict(have)
    while exited < workers:
        msg = done.get()
        if msg.get("exit"):
            exited += 1
        elif "error" in msg:
            errors.append(msg)
            print(f"encode: worker {msg['worker']} failed: {msg['error']}\n{msg['trace']}", file=sys.stderr, flush=True)
        else:
            metas[msg["segment"]] = msg
            k = len(metas) - len(have)
            if k == len(todo_list) or k % max(1, len(todo_list) // 50) == 0:
                el = time.perf_counter() - t0
                print(f"encode: {k}/{len(todo_list)} segments in {el:.0f} s, ETA {el / k * (len(todo_list) - k):.0f} s",
                      file=sys.stderr, flush=True)
    for p in procs:
        p.join()
    wall = time.perf_counter() - t0
    encoded = len(metas) - len(have)
    doc["encode"] = {"segments_this_run": encoded, "resumed": len(have), "workers": workers, "wall_s": wall,
                     "c_GBps": encoded * params.blocks_per_segment * params.block_bytes / wall / 1e9 if encoded else None,
                     "w_GBps": encoded * params.segment_bytes / wall / 1e9 if encoded else None,
                     "blocks_per_s": encoded * params.blocks_per_segment / wall if encoded else None,
                     "segment_encode_s": _quantiles([m["encode_s"] for s, m in metas.items() if s not in have and "encode_s" in m]),
                     "errors": [e["error"] for e in errors]}
    if errors or len(metas) != params.segments:
        dump(out / "store-partial.json", doc)
        print(f"encode: {len(metas)} of {params.segments} segments done; rerun with the same --store to resume", file=sys.stderr)
        return 1
    finish_store(store, out, doc, metas, params, a.registration)
    return 0


def _quantiles(xs: list[float]) -> dict[str, float] | None:
    if not xs:
        return None
    xs = sorted(xs)
    return {"min": xs[0], "median": xs[len(xs) // 2], "max": xs[-1], "sum": sum(xs)}


def finish_store(store: Path, out: Path, doc: dict[str, Any], metas: dict[int, dict[str, Any]], params: S.P2Params,
                 registration: str | None) -> dict[str, Any]:
    order = [metas[s] for s in range(params.segments)]
    doc["root"] = top_root([bytes.fromhex(m["root"]) for m in order]).hex()
    doc["w_root"] = top_root([bytes.fromhex(m["w_root"]) for m in order]).hex()
    doc["registration"] = registration
    doc["tie"] = tie(doc)
    doc["tie_tag"], doc["tie_fields"] = TIE_TAG, list(TIE_FIELDS)
    sums = "".join(f"{m['sha512']}  seg/{m['segment']:05d}.c\n{m['rows_sha512']}  rows/{m['segment']:05d}.rows\n" for m in order)
    write_atomic(store / "SHA512SUMS", sums.encode())
    doc["sha512sums_sha512"] = hashlib.sha512(sums.encode()).hexdigest()
    dump(store / "store.json", doc)
    dump(out / "store.json", doc)
    dump(out / "segments.json", order)
    write_atomic(out / "SHA512SUMS", sums.encode())
    print(json.dumps({k: doc.get(k) for k in ("commitment", "salt", "root", "w_root", "registration", "tie")}))
    return doc


def _check_segment(job: tuple[str, int]) -> dict[str, Any]:
    store, s = Path(job[0]), job[1]
    doc = json.loads((store / "store.json").read_text())
    params, salt = params_of(doc["manifest"]), bytes.fromhex(doc["salt"])
    meta = json.loads(seg_path(store, s, "json").read_text())
    c_seg = seg_path(store, s, "c").read_bytes()
    rows = row_digests(c_seg, params.block_bytes)
    got = {"sha512": hashlib.sha512(c_seg).hexdigest(), "rows_sha512": hashlib.sha512(rows).hexdigest(),
           "root": segment_root(params, salt, s, rows).hex()}
    bad = [k for k, v in got.items() if meta.get(k) != v]
    if seg_path(store, s, "rows").read_bytes() != rows:
        bad.append("rows file")
    if len(c_seg) != params.blocks_per_segment * params.block_bytes:
        bad.append("length")
    return {"segment": s, "bad": bad, "root": got["root"], "w_root": meta["w_root"], "sha512": got["sha512"]}


def _decoder(params: S.P2Params, salt: bytes, lib: str | None) -> Any:
    if not lib:
        return lambda s, c: S.decode_segment(params, None, salt, s,
                                             [c[i * params.block_bytes:(i + 1) * params.block_bytes] for i in range(params.blocks_per_segment)])
    import torch

    from p2_v1 import reference as R
    from p2_v1.gpu import Kernels

    P = R.Params(params.w, params.c0, params.payload_bytes, params.blocks_per_segment, params.segments)
    K = Kernels(lib, "", P, S.expand_prefix(params, salt))
    n, nb, pb, p = params.blocks_per_segment, params.block_bytes, params.payload_bytes, params.p
    p_row = torch.frombuffer(bytearray(p.to_bytes(nb, "big")), dtype=torch.uint8)

    def decode(s: int, c: bytes) -> bytes:
        rows = torch.frombuffer(bytearray(c), dtype=torch.uint8).view(n, nb)
        if not bool(_rows_below(torch, rows, p_row).all()):
            raise ValueError(f"segment {s}: a block is not below p")
        m = K.decode(rows.to(K.device), s * n).cpu()
        if K.raised():
            raise ValueError(f"segment {s}: the decode kernels raised flags")
        if bool(m[:, :nb - pb].any()):
            raise ValueError(f"segment {s}: a block decodes to more than {8 * pb} bits: not a codeword")
        return m[:, nb - pb:].contiguous().numpy().tobytes()
    return decode


def _rows_below(torch: Any, rows: Any, bound: Any) -> Any:
    """Each big-endian row `< bound`: compared at its first byte that differs."""
    diff = rows != bound
    first = torch.where(diff.any(1), diff.int().argmax(1), torch.full((rows.shape[0],), rows.shape[1] - 1))
    idx = first.unsqueeze(1)
    return diff.any(1) & (rows.gather(1, idx).squeeze(1) < bound[first])


def verify_cmd(a: argparse.Namespace) -> int:
    """Reread the store: files, rows, roots and the tie; with `--decode`, Dec(C*) into the commitment."""
    store, out = Path(a.store), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    doc = json.loads((store / "store.json").read_text())
    params, salt = params_of(doc["manifest"]), bytes.fromhex(doc["salt"])
    sums = (store / "SHA512SUMS").read_text()
    report: dict[str, Any] = {"store": str(store), "root": doc["root"], "tie": doc["tie"],
                              "sha512sums": hashlib.sha512(sums.encode()).hexdigest() == doc["sha512sums_sha512"]}
    t0 = time.perf_counter()
    jobs = [(str(store), s) for s in range(params.segments)]
    with mp.get_context("fork").Pool(max(1, a.workers)) as pool:
        checks = pool.map_async(_check_segment, jobs, chunksize=1)
        if a.decode:
            h, wroots, n = setup._commitment_hash(params), [], 0
            dec = _decoder(params, salt, a.p2_lib)
            for s in range(params.segments):
                w_seg = dec(s, seg_path(store, s, "c").read_bytes())
                h.update(w_seg)
                wroots.append(segment_root(params, salt, s, row_digests(w_seg, params.payload_bytes)))
                n += 1
                if n % max(1, params.segments // 20) == 0:
                    print(f"verify: decoded {n}/{params.segments} in {time.perf_counter() - t0:.0f} s", file=sys.stderr, flush=True)
            report["decode"] = {"commitment": h.digest().hex() == doc["commitment"],
                                "w_root": top_root(wroots).hex() == doc["w_root"], "by": "libp2v1" if a.p2_lib else "schemes.p2",
                                "s": time.perf_counter() - t0}
        got = checks.get()
    listed = {line.split("  ")[1]: line.split("  ")[0] for line in sums.splitlines()}
    bad = [g for g in got if g["bad"] or listed.get(f"seg/{g['segment']:05d}.c") != g["sha512"]]
    report["segments"] = {"checked": len(got), "bad": [{"segment": g["segment"], "bad": g["bad"]} for g in bad]}
    report["root_ok"] = top_root([bytes.fromhex(g["root"]) for g in got]).hex() == doc["root"]
    report["w_root_ok"] = top_root([bytes.fromhex(g["w_root"]) for g in got]).hex() == doc["w_root"]
    report["tie_ok"] = tie(doc) == doc["tie"]
    report["s"] = time.perf_counter() - t0
    ok = (report["sha512sums"] and not bad and report["root_ok"] and report["w_root_ok"] and report["tie_ok"]
          and (not a.decode or (report["decode"]["commitment"] and report["decode"]["w_root"])))
    report["passed"] = ok
    dump(out / "verify.json", report)
    print(json.dumps({k: report[k] for k in ("passed", "root_ok", "w_root_ok", "tie_ok", "sha512sums")}
                     | ({"decode": report["decode"]} if a.decode else {})))
    return 0 if ok else 1


def spot_cmd(a: argparse.Namespace) -> int:
    """Decode `--count` random blocks by `reference.decode` (and `--p2-lib`'s kernels) against W regenerated from the seed."""
    store, out = Path(a.store), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    doc = json.loads((store / "store.json").read_text())
    params, salt = params_of(doc["manifest"]), bytes.fromhex(doc["salt"])
    weights = Weights(a.generator, doc["shape"], a.seed)
    if weights.program != doc["program"]:
        raise SystemExit("the seed or generator is not the store's: its program differs")
    prefix, n, nb, pb = S.expand_prefix(params, salt), params.blocks_per_segment, params.block_bytes, params.payload_bytes
    gpu = None
    if a.p2_lib:
        import torch

        from p2_v1 import reference as R
        from p2_v1.gpu import Kernels

        gpu = (torch, Kernels(a.p2_lib, "", R.Params(params.w, params.c0, pb, n, params.segments), prefix))
    rng = random.SystemRandom()
    rows: list[dict[str, Any]] = []
    for _ in range(a.count):
        i = rng.randrange(params.blocks)
        s, j = divmod(i, n)
        with open(seg_path(store, s, "c"), "rb") as f:
            f.seek(j * nb)
            block = f.read(nb)
        want = weights.read(s * params.segment_bytes + j * pb, s * params.segment_bytes + (j + 1) * pb)
        row = {"block": i, "segment": s, "index": j, "reference": reference_decode(params, prefix, s, j, block) == want}
        with open(seg_path(store, s, "rows"), "rb") as f:
            f.seek(64 * j)
            row["row_digest"] = f.read(64) == verifier.row_digest(nb, block)
        if gpu is not None:
            torch, K = gpu
            m = K.decode(torch.frombuffer(bytearray(block), dtype=torch.uint8).view(1, nb).to(K.device), i).cpu()
            row["device"] = bytes(m[0].numpy()) == bytes(nb - pb) + want and not K.raised()
        rows.append(row)
    passed = all(r["reference"] and r["row_digest"] and r.get("device", True) for r in rows)
    report = {"store": str(store), "root": doc["root"], "count": len(rows), "passed": passed,
              "device": bool(gpu), "blocks": rows}
    dump(out / "spot.json", report)
    print(json.dumps({"count": len(rows), "passed": passed, "reference_matches": sum(r["reference"] for r in rows),
                      "device_matches": sum(r.get("device", False) for r in rows) if gpu else None}))
    return 0 if passed else 1


def tie_cmd(a: argparse.Namespace) -> int:
    store, out = Path(a.store), Path(a.out)
    doc = json.loads((store / "store.json").read_text())
    doc["registration"] = a.registration
    doc["tie"] = tie(doc)
    dump(store / "store.json", doc)
    dump(out / "store.json", doc)
    print(json.dumps({"registration": a.registration, "tie": doc["tie"]}))
    return 0


def _visible_gpus() -> int:
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    if vis is not None:
        return len([x for x in vis.split(",") if x.strip()])
    import torch

    return torch.cuda.device_count()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("commit", "encode", "verify", "spot", "tie"):
        p = sub.add_parser(name)
        p.add_argument("--store", required=True, help="the store's directory (scratch, outside the run directory)")
        p.add_argument("--out", default=os.environ.get("OUT", "."), help="the public outputs' directory ($OUT)")
        p.add_argument("--workers", type=int, default=os.cpu_count() or 1, help="CPU processes for W's generation and hashing")
        if name in ("commit", "encode", "spot"):
            p.add_argument("--generator", default=GENERATOR, help="the weights generator: a module name or its .py file")
            p.add_argument("--seed", type=_bytes_arg, required=True, help="hex:..., hexfile:PATH, file:PATH or text:...")
        if name in ("commit", "encode"):
            p.add_argument("--shape", default="h200")
            p.add_argument("--point", type=lambda t: tuple(int(x) for x in t.split(",")), default=S.POINT,
                           help="w,c0,blocks_per_segment (default the registered point)")
        p.add_argument("--p2-lib", default="", help="libp2v1.so's directory (v4 keys); without it, the scheme's CPU codec")
    e = sub.choices["encode"]
    e.add_argument("--source", type=_bytes_arg, help="the auditor's coin, drawn after the commitment (hex:..., hexfile:PATH)")
    e.add_argument("--gpus", type=int, default=None)
    e.add_argument("--cpu-encoders", type=int, default=1)
    e.add_argument("--registration", default=None, help="the weights' registration digest, hex")
    sub.choices["verify"].add_argument("--decode", action="store_true")
    sub.choices["spot"].add_argument("--count", type=int, default=64)
    sub.choices["tie"].add_argument("--registration", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "commit":
        commit_cmd(a)
        return 0
    if a.cmd == "encode":
        if a.p2_lib and a.gpus is None:
            a.gpus = _visible_gpus()
        return encode_cmd(a)
    return {"verify": verify_cmd, "spot": spot_cmd, "tie": tie_cmd}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
