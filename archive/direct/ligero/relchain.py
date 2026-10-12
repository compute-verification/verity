"""The B-Ligero chain prover for any registered relation (``relations.py``): gate, bench, proof dumps.

    python -m archive.direct.ligero.run --relation bf16-hopper gate-vu  --device cuda --vus 64
    python -m archive.direct.ligero.run --relation fp8-hopper  bench-vu --device cuda --batch 4096 --total-vus 4096 --reps 3 \
        --dump-dir $RESEARCH_RUN_DIR/proofs --out $RESEARCH_RUN_DIR/result.json

One VU is ``GemmCoordinate<1536>`` (BF16: cast at the end) or ``GemmAccumulator<1536>`` (FP8: the FP32 word) as
``1536 / k`` chained units from ``c_0 = +0``; the accumulators are witness rows linked by the chain test (``chain.py``),
the operands and the claimed chain-end words enter the statement digest before any challenge (PROTOCOL.md).  Instances
are synthetic and deterministic (``instances``: the relation's finite operand recipe from ``(seed, index)``, the
accumulators recorded by the model ``tc_dot``); ``--instances-cache DIR`` keeps them as ``.npz`` between runs.

``fp8/chain.py`` (the fp8-proof lane's runner for ``--relation fp8-ada``) is this module at ``RELATIONS["fp8-ada"]``.
"""
from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
import os
import platform
import statistics
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from verity.ml.circuits.silicon.silicon import tc_dot
from verity_numerical.bench import contract

from . import phases, protocol as protocol_mod
from .chain import Layout, extra_constraints
from .compile import P
from .fp8.census import census
from .protocol import Config, config_for, n_rows, prove, soundness, verify
from .relations import K_VU, Relation, relation
from .serialize import hashed_statement_format
from .witness import tables

__all__ = ["RelationChainRunner", "bench_vu_rel", "gate_vu_rel", "instances", "instances_digest", "instances_ref", "vu_units"]


# ---------------------------------------------------------------------------------------------------------------------
# Instances
# ---------------------------------------------------------------------------------------------------------------------

def _K(rel: Relation) -> int:
    """The relation's VU reduction length: its own ``vu_k``, else this module's ``K_VU`` (which the steps-pin red-team
    scripts rebind together with ``relations.K_VU``)."""
    return rel.vu_k or K_VU


def _instance(args) -> tuple[list, list, list, int]:
    return _instance_np(args)[2]


def _instance_np(args) -> tuple[np.ndarray, np.ndarray, tuple[list, list, list, int]]:
    """VU ``i``'s operands as drawn (arrays) and the VU ``(a, b, accs, y_final)`` of :func:`_instance`."""
    name, seed, i = args
    rel = relation(name)
    rng = np.random.default_rng([seed, i])
    a_np, b_np = rel.operands(rng, (_K(rel),)), rel.operands(rng, (_K(rel),))
    a, b = a_np.tolist(), b_np.tolist()
    k, accs, c = rel.k, [], 0
    for s in range(rel.steps):
        c = tc_dot(rel.model, c, a[k * s:k * s + k], b[k * s:k * s + k])
        accs.append(c)
    return a_np, b_np, (a, b, accs, accs[-1])


def _frozen_root(root) -> Path:
    from verity_numerical.bench import instances as frozen
    return Path(root) if root else frozen.DEFAULT_ROOT


def frozen_instances(rel: Relation, n: int, root=None):
    """The first ``n`` VUs of the relation's FROZEN tier (``rel.frozen_tier``: ``bench-instances/v1`` ``vu-k1536`` under
    ``root``, the set ``vu.py`` proves) in the runner's shape ``(a, b, accs, y_final)``: the recorded FP32 accumulators are
    the chain's public words, the final one casts to the tier's expected BF16 word (checked).  Beyond the tier's ids the
    instances are recycled (``i mod n_tier``), as ``vu.bench_vu`` does."""
    from verity_numerical.bench import instances as frozen
    vu = frozen.load(rel.frozen_tier, _frozen_root(root))
    if vu.acc is None:
        raise ValueError(f"{rel.frozen_tier}: the tier carries no accumulators")
    out = []
    for i in range(n):
        j = i % vu.n
        a, b, y16 = vu.instance(j)
        accs = vu.accs(j)
        if rel.fold > 1:
            # a folded relation's columns end on every ``fold``-th recorded accumulator (the intermediate ones are private)
            accs = accs[rel.fold - 1::rel.fold]
        if len(accs) != rel.steps or rel.y_public(int(accs[-1])) != y16:
            raise ValueError(f"{rel.frozen_tier} instance {j}: {len(accs)} accumulators / final word does not cast to the expected y")
        out.append((a, b, accs, int(accs[-1])))
    x = np.frombuffer(vu.x, dtype=np.uint16 if vu.x.itemsize == 2 else np.uint32).reshape(vu.n, vu.K)
    w = np.frombuffer(vu.w, dtype=np.uint16 if vu.w.itemsize == 2 else np.uint32).reshape(vu.n, vu.K)
    if n <= vu.n:
        return _with_rows(out, x[:n], w[:n])
    return _with_rows(out, [x[i % vu.n] for i in range(n)], [w[i % vu.n] for i in range(n)])


def set_instances(rel: Relation, path, n: int, lo: int = 0):
    """VUs ``[lo, lo + n)`` of an input set (``verity_numerical.bench.input_sets``: a captured ``vllm-vu-set/v1`` or a generated
    ``input-set/v1`` of ``gemm-coordinate``) in the runner's shape ``(a, b, accs, y_final)``.  The operands are the set's
    ``x`` / ``w`` rows; the column-end accumulators are recorded from the relation's model (``group_sum_total_batch``, the
    batch form of ``tc_dot``) and every final word must be what the set recorded (its stored word, or the FP32 accumulator
    a ``u32`` port holds), else the set is refused.
    A set is never recycled: ``lo + n`` beyond its size is an error."""
    from verity.ml.kernels.tc import group_sum_total_batch
    from verity_numerical.bench.input_sets import InputSet
    s = InputSet.open(path)
    K = int(s.subcircuit.params["K"])
    if K != _K(rel):
        raise ValueError(f"input set {s.name} is K = {K}; {rel.name} proves K = {_K(rel)}")
    if lo < 0 or lo + n > s.n:
        raise ValueError(f"input set {s.name} holds {s.n} instances, not [{lo}, {lo + n}) (a set is not recycled)")
    ins = s.instances(lo, lo + n)
    x, w = np.ascontiguousarray(ins["x"]), np.ascontiguousarray(ins["w"])
    y = np.asarray(ins["y"]).reshape(n, -1)[:, 0].astype(np.int64)
    k, acc = rel.k, np.zeros(n, dtype=np.uint32)
    accs = np.empty((n, rel.steps), dtype=np.int64)
    for c in range(rel.steps):
        acc, _ = group_sum_total_batch(rel.model, acc, x[:, k * c:k * c + k], w[:, k * c:k * c + k])
        accs[:, c] = acc
    # the set records either the stored output word (BF16: the epilogue's word, rel.y_public) or the FP32 accumulator itself
    # (a u32 port, the E4M3 sets: no epilogue) -- compared word for word in the set's own form
    raw = s.manifest["ports"]["y"]["dtype"] == "u32"
    model = accs[:, -1] if raw else np.asarray([rel.y_public(int(a)) for a in accs[:, -1]], dtype=np.int64)
    bad = np.nonzero(model != y)[0]
    if bad.size:
        raise ValueError(f"input set {s.name}: {bad.size} of {n} instances' recorded outputs are not {rel.name}'s "
                         f"(first: instance {lo + int(bad[0])}, model {int(model[bad[0]]):#x}, set {int(y[bad[0]]):#x}, "
                         f"{'FP32 accumulator' if raw else 'stored word'})")
    return _with_rows([(x[i].tolist(), w[i].tolist(), accs[i].tolist(), int(accs[i, -1])) for i in range(n)], x, w)


def set_ref(path, n: int, art: str | None = None) -> dict:
    """The fingerprint's ``instances`` block of a run over ids ``[0, n)`` of an input set (``InputSet.ref``: dataset, tier,
    range, manifest and content digests, subcircuit, source, store id)."""
    from verity_numerical.bench.input_sets import InputSet
    return InputSet.open(path).ref(0, n, art=art)


class VUs(list):
    """An instance list (:func:`instances`) carrying the committed operand rows in the form they were read or drawn in
    (``x_rows``, ``w_cols``: an (n, K) integer array or n integer arrays of K, the words of ``self[i][0]`` /
    ``self[i][1]``): what ``commit_vus`` commits instead of converting the per-VU lists back."""
    x_rows: np.ndarray | None = None
    w_cols: np.ndarray | None = None


def _with_rows(vus: list, x_rows, w_cols) -> VUs:
    out = VUs(vus)
    out.x_rows, out.w_cols = x_rows, w_cols
    return out


def _word_rows(rows) -> np.ndarray:
    """Committed rows as a 2-D integer array: an integer ndarray as it is (its own dtype), a list of 1-D integer ndarrays
    stacked, anything else as int64."""
    if isinstance(rows, np.ndarray) and rows.ndim == 2 and rows.dtype.kind in "iu":
        return rows
    if isinstance(rows, list) and rows and all(isinstance(r, np.ndarray) and r.ndim == 1 and r.dtype.kind in "iu" for r in rows):
        return np.stack(rows)
    return np.asarray(rows, dtype=np.int64)


def narrow_rows(rows, word_bits: int, pin: bool = False) -> np.ndarray:
    """Committed rows (:func:`_word_rows`) in their words' own dtype, as a committed set is stored: uint8 / uint16 for
    8- / 16-bit words (range-checked), any other width as it is.  ``pin``: in page-locked host memory (a serving stack's
    staging buffer; the copy to the device is then a DMA without host staging)."""
    r = _word_rows(rows)
    dt = {8: np.uint8, 16: np.uint16}.get(int(word_bits))
    if dt is not None and r.dtype != dt:
        if r.size and (int(r.min()) < 0 or int(r.max()) >= 1 << int(word_bits)):
            raise ValueError(f"committed words out of range for {word_bits} bits")
        r = r.astype(dt)
    if pin and torch.cuda.is_available():
        buf = torch.empty(r.nbytes, dtype=torch.uint8, pin_memory=True).numpy().view(r.dtype).reshape(r.shape)
        buf[...] = r
        return buf
    return np.ascontiguousarray(r)


def _h2d(a: np.ndarray, device) -> torch.Tensor:
    """``a`` on ``device``: from page-locked memory without blocking the host (the stream orders its readers)."""
    t = torch.from_numpy(a)
    return t.to(device, non_blocking=str(device).startswith("cuda") and t.is_pinned())


def _rows_to_device(rows: np.ndarray, device, narrow: dict | None = None, name: str | None = None):
    """``rows`` as an int64 torch tensor on ``device``, copied in its narrow dtype and widened there.  ``narrow``: records
    ``narrow[name]`` = the copied tensor when it holds the rows' bytes (uint8 words, or uint16 words as int16)."""
    if rows.dtype == np.uint16:
        t = _h2d(np.ascontiguousarray(rows).view(np.int16), device)
        if narrow is not None:
            narrow[name] = t
        return t.to(torch.int64) & 0xFFFF
    if rows.dtype not in (np.uint8, np.int8, np.int16, np.int32, np.int64):
        rows = rows.astype(np.int32 if rows.dtype.itemsize <= 2 else np.int64)
    t = _h2d(np.ascontiguousarray(rows), device)
    if narrow is not None and rows.dtype == np.uint8:
        narrow[name] = t
    return t.to(torch.int64)


def instances(rel: Relation, n: int, seed: int | None = None, procs: int = 1, cache: str | Path | None = None, root=None):
    """``n`` VUs: (a (1536,), b (1536,), accs (steps,), y_final) with the accumulators recorded from the model.
    Deterministic in ``(seed, index)``; ``procs > 1`` records them in parallel; ``cache`` keeps the set as an npz.
    A relation with a ``frozen_tier`` reads its frozen set from ``root`` instead (:func:`frozen_instances`)."""
    if rel.frozen_tier:
        return frozen_instances(rel, n, root)
    if getattr(rel, "instances_fn", None) is not None:       # a relation with its own instance families (fp4/hashed.py)
        return rel.instances_fn(n, seed=seed, procs=procs, cache=cache, root=root)
    seed = rel.instance_seed if seed is None else seed
    path = Path(cache) / f"{rel.name}-{seed}-{n}.npz" if cache else None
    if path is not None and path.exists():
        with np.load(path) as z:  # materialise each array once (indexing the lazy NpzFile re-reads the member every time)
            a, b, accs = z["a"], z["b"], z["accs"]
        return _with_rows([(a[i].tolist(), b[i].tolist(), accs[i].tolist(), int(accs[i][-1])) for i in range(n)], a, b)
    if path is not None and rel.fold > 1 and rel.folded_from and (Path(cache) / f"{rel.folded_from}-{seed}-{n}.npz").exists():
        # the fold's VUs ARE the base's (same operands from the same (seed, i) streams, the same chain): every ``fold``-th
        # recorded accumulator is a column's end, the others are the words the fold keeps private
        with np.load(Path(cache) / f"{rel.folded_from}-{seed}-{n}.npz") as z:
            a, b, accs = z["a"], z["b"], z["accs"][:, rel.fold - 1::rel.fold]
        return _with_rows([(a[i].tolist(), b[i].tolist(), accs[i].tolist(), int(accs[i][-1])) for i in range(n)], a, b)
    jobs = [(rel.name, seed, i) for i in range(n)]
    if procs > 1 and n > 1:
        with ProcessPoolExecutor(procs) as ex:
            drawn = list(ex.map(_instance_np, jobs, chunksize=max(1, n // (8 * procs))))
    else:
        drawn = [_instance_np(j) for j in jobs]
    out = _with_rows([d[2] for d in drawn], [d[0] for d in drawn], [d[1] for d in drawn])
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, a=np.asarray([o[0] for o in out], dtype=np.int64), b=np.asarray([o[1] for o in out], dtype=np.int64),
                 accs=np.asarray([o[2] for o in out], dtype=np.int64))
    return out


def instances_digest(rel: Relation, n: int, seed: int | None = None, root=None) -> str:
    """The synthetic set's manifest digest: a function of (relation, seed, n, K) -- the set itself is.  A frozen tier's is
    the sha256 of its ``manifest.json`` (``instances.manifest_sha256``), the value the bare ``vu.py`` results carry."""
    if rel.frozen_tier:
        from verity_numerical.bench import instances as frozen
        return frozen.manifest_sha256(_frozen_root(root))
    if getattr(rel, "instances_digest_fn", None) is not None:
        return rel.instances_digest_fn(n, seed=seed, root=root)
    seed = rel.instance_seed if seed is None else seed
    return hashlib.sha256(f"{rel.name} synthetic|seed={seed}|n={n}|K={_K(rel)}".encode()).hexdigest()


def instances_ref(rel: Relation, total_vus: int, tile: tuple[int, int] | None = None, root=None) -> dict:
    """The fingerprint's ``instances`` block of a :func:`bench_vu_rel` result over ``total_vus`` VUs (``tile``: the
    ``--tile`` workload) -- field for field what the result carries (``verity_numerical.bench.instance_equiv`` reads it)."""
    if tile is not None:
        # tile_instances draws other VUs than instances() (nx rows x nw columns from (seed, "tile", ...) streams): no
        # frozen set is a tile, so the ref names the tile (the non-frozen relations named their frozen set here until
        # lane fused-phases: x, W and y all differ from it)
        return {"dataset": "synthetic", "tier": f"tile{tile[0]}x{tile[1]}", "range": [0, total_vus],
                "manifest_sha256": tile_digest(rel, *tile), "seed": rel.instance_seed,
                "recipe": f"{rel.name} synthetic GEMM tile (relchain.tile_instances: {rel.operands.__name__} rows / columns, "
                          f"model-recorded accumulators); NOT the {'frozen ' if rel.frozen_tier else ''}"
                          f"{rel.frozen_tier or rel.instances_tier} set"}
    if rel.frozen_tier:
        # the frozen set: the same ``instances`` block as vu.bench_vu (no seed, no recipe), so the committed column of the
        # A100 cell names exactly the bare cell's instances
        return contract.instances_ref(rel.instances_tier, 0, total_vus, instances_digest(rel, total_vus, root=root),
                                      dataset=rel.instances_dataset)
    inst = {"dataset": rel.instances_dataset, "tier": rel.instances_tier, "range": [0, total_vus],
            "manifest_sha256": instances_digest(rel, total_vus, root=root), "seed": rel.instance_seed}
    if rel.instances_recipe:
        inst["recipe"] = rel.instances_recipe
    return inst


def vu_units(rel: Relation, a: list[int], b: list[int], accs: list[int]) -> list[tuple]:
    """The ``steps`` units of one VU with their recorded output words."""
    k, units, c = rel.k, [], 0
    for s in range(rel.steps):
        units.append((c, a[k * s:k * s + k], b[k * s:k * s + k], accs[s]))
        c = accs[s]
    return units


def _tile_accs(args) -> list[int]:
    name, a, b = args
    rel = relation(name)
    k, accs, c = rel.k, [], 0
    for s in range(rel.steps):
        c = tc_dot(rel.model, c, a[k * s:k * s + k], b[k * s:k * s + k])
        accs.append(c)
    return accs


def parse_tile(spec: str | None) -> tuple[int, int] | None:
    """``--tile 64x64`` -> (n_x_rows, n_w_cols); None for the worst case (every VU its own rows)."""
    if not spec or spec == "none":
        return None
    nx, nw = spec.lower().split("x")
    return int(nx), int(nw)


def tile_instances(rel: Relation, nx: int, nw: int, seed: int | None = None, procs: int = 1, cache: str | Path | None = None):
    """The committed GEMM-tile workload (``sharing = tile<nx>x<nw>``): ``nx`` distinct activation rows and ``nw`` distinct
    weight columns (the relation's finite operand recipe from ``(seed, "tile", ...)``), every (row, column) pair one VU
    -- VU ``v`` is row ``v // nw`` against column ``v % nw`` -- with the accumulators recorded from the model.  Same FLOPs
    and relation as ``instances(rel, nx * nw)``; the hashed statement commits ``nx + nw`` rows instead of ``2 nx nw``.
    Returns ``(vus, x_rows (nx, K), w_cols (nw, K))``."""
    seed = rel.instance_seed if seed is None else seed
    path = Path(cache) / f"{rel.name}-{seed}-tile{nx}x{nw}.npz" if cache else None
    if path is not None and path.exists():
        with np.load(path) as z:
            x, w, accs = z["x"], z["w"], z["accs"]
    else:
        rng = np.random.default_rng([seed, 0x7115, nx, nw])
        x = np.asarray([rel.operands(rng, (_K(rel),)) for _ in range(nx)], dtype=np.int64)
        w = np.asarray([rel.operands(rng, (_K(rel),)) for _ in range(nw)], dtype=np.int64)
        jobs = [(rel.name, x[v // nw].tolist(), w[v % nw].tolist()) for v in range(nx * nw)]
        if procs > 1 and len(jobs) > 1:
            with ProcessPoolExecutor(procs) as ex:
                out = list(ex.map(_tile_accs, jobs, chunksize=max(1, len(jobs) // (8 * procs))))
        else:
            out = [_tile_accs(j) for j in jobs]
        accs = np.asarray(out, dtype=np.int64)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(path, x=x, w=w, accs=accs)
    vus = [(x[v // nw].tolist(), w[v % nw].tolist(), accs[v].tolist(), int(accs[v][-1])) for v in range(nx * nw)]
    return vus, x, w


#: a dump manifest's ``set.tile``: the draw and layout names a verifier must know to recompute the tile
TILE_DRAW = "relchain.tile_instances/v1"
TILE_LAYOUT = "row-major: x = vu // nw, W = vu % nw"


def tile_digest(rel: Relation, nx: int, nw: int, seed: int | None = None) -> str:
    """The manifest digest of :func:`tile_instances` (a synthetic draw of its own, for every relation: its VUs are not
    ``instances(rel, nx nw)``)."""
    seed = rel.instance_seed if seed is None else seed
    return hashlib.sha256(f"{rel.name} synthetic tile|seed={seed}|{nx}x{nw}|K={_K(rel)}".encode()).hexdigest()


def tile_indices(nx: int, nw: int, lo: int, hi: int) -> tuple[list[int], list[int], list[int]]:
    """(vu_index, x_index, w_index) of the tile's VUs ``[lo, hi)``."""
    vus = list(range(lo, hi))
    return vus, [v // nw for v in vus], [v % nw for v in vus]


# ---------------------------------------------------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------------------------------------------------

class RelationChainRunner:
    """``vu.ChainRunner`` for a registered relation: the same protocol (``prove`` / ``verify``) on ``rel.compile(p, chain=True)``
    with the relation's hint generator, public decode and statement hooks."""

    def __init__(self, rel: Relation, device: str, target: float, rate_log2: int = 2, n_proofs: int = 1, zk: bool = False,
                 mode: str = protocol_mod.MODE_INTERACTIVE):
        self.rel = rel
        self.p = rel.params
        self.zk, self.mode, self.device, self.target, self.rate_log2, self.n_proofs = zk, mode, device, target, rate_log2, n_proofs
        self.steps = rel.steps
        self.hooks = rel.hooks
        t0 = time.perf_counter()
        self.sys = rel.compile(self.p, True)
        self.tb = tables(self.sys, self.device)
        self.compile_seconds = time.perf_counter() - t0
        self._cfg: dict[int, Config] = {}
        self._marshal_cache: dict = {}

    def cfg(self, l: int) -> Config:
        if l not in self._cfg:
            self._cfg[l] = config_for(l, self.target, self.rate_log2, self.n_proofs, zk=self.zk, mode=self.mode,
                                      chain_extras=self.chain_extras)
        return self._cfg[l]

    @property
    def chain_extras(self) -> int:
        """The system's chain constraints with monomial coefficients: their field term is sized into t and booked."""
        return extra_constraints(self.sys.chain)

    @property
    def interactive(self) -> bool:
        return self.mode == protocol_mod.MODE_INTERACTIVE

    @property
    def relation(self) -> str:
        """The registered relation's name: ``serialize._relation_of`` reads it, so every registered relation's statement
        file is the relation-named ``ligero-statement/v4`` (the Ampere BF16 relation, ``vu.ChainRunner``, has no such
        property and keeps v2 / v3 for the Rust verifier)."""
        return self.rel.name

    def layout(self, n_vus: int, min_l: int = 256) -> Layout:
        l = max(min_l, 1 << (n_vus * self.steps - 1).bit_length())
        return Layout(l, n_vus, self.steps)

    def _pad_unit(self):
        return (0, [0] * self.rel.k, [0] * self.rel.k, 0)

    def marshal(self, vus: list[tuple[list, list, list, int]], min_l: int = 256) -> dict:
        """The sub-batch's public words as arrays, once per sub-batch for the whole run (cached on the identity of the
        VU tuples): host ``a, b, c_in, y_out`` (int64 numpy: the statement, the dumps, the reference hints) and their
        device-resident copies + the verifier's public decode ``pub`` (``rel.public_vectors``, device) -- so a timed
        sub-batch does no list -> numpy -> device marshalling and no host round trip (lane hp2-host, "instances
        device-resident")."""
        key = (len(vus), id(vus[0]), id(vus[-1]), id(vus[len(vus) // 2]), min_l)
        hit = self._marshal_cache.get(key)
        if hit is not None:
            return hit
        lay = self.layout(len(vus), min_l)
        units, ypub = [], torch.zeros(lay.l, dtype=torch.int64)
        for v, (a, b, accs, y) in enumerate(vus):
            units += vu_units(self.rel, a, b, accs)
            ypub[self.steps * v + self.steps - 1] = self.rel.y_public(int(y))
        while len(units) < lay.l:
            units.append(self._pad_unit())
        a = np.asarray([u[1] for u in units], dtype=np.int64)
        b = np.asarray([u[2] for u in units], dtype=np.int64)
        c_in = np.asarray([u[0] for u in units], dtype=np.int64)
        y_out = np.asarray([u[3] for u in units], dtype=np.int64)
        dev = self.device
        to_dev = lambda x: torch.as_tensor(x).to(dev)  # noqa: E731
        bundle = {"lay": lay, "units": units, "a": a, "b": b, "c_in": c_in, "y_out": y_out, "ypub": ypub,
                  "a_d": to_dev(a), "b_d": to_dev(b), "c_d": to_dev(c_in), "y_d": to_dev(y_out), "ypub_d": to_dev(ypub)}
        bundle["pub"] = self.rel.public_vectors(self.p, None, bundle["a_d"], bundle["b_d"], None, device=dev)
        if str(dev).startswith("cuda"):
            torch.cuda.synchronize(dev)
        self._marshal_cache[key] = bundle
        return bundle

    def prove_vus(self, vus: list[tuple[list, list, list, int]], check=True, mutate=None, min_l: int = 256, coins=None, vu_ids=None):
        """``vus``: (a, b, accs, y_final) per VU.  The public claim per VU is ``rel.y_public(y_final)`` at its last column.
        Interactive mode with ``coins=None``: the runner plays the honest verifier's step 0 (``Coins.sample()``).
        ``vu_ids`` (the VUs' indices in the committed set) is what the hashed runner binds; the bare statement ignores it."""
        from .pipeline import run_to_completion
        if coins is None and self.interactive:
            coins = protocol_mod.Coins.sample()
        bd = self.marshal(vus, min_l)
        if hasattr(coins, "set_statement"):
            # a live verifier's coin source (live.LiveCoins): the statement travels with the root, so the claim is fixed
            # before the verifier opens its first coin.  Built outside prove's clock; the protocol is unchanged.
            from .live import set_statement
            set_statement(coins, self, (bd["a"], bd["b"], bd["ypub"]), bd["lay"], self.cfg(bd["lay"].l), self.n_proofs)
        gen = self._prove_stages(bd, coins, check=check, mutate=mutate, slot=None)
        proof = run_to_completion(gen)
        return proof, (bd["a"], bd["b"], bd["ypub"]), bd["lay"]

    def _prove_stages(self, bd: dict, coins, check=False, mutate=None, slot=None):
        """The prover generator for one marshalled sub-batch (``pipeline.prove_many`` drives several at once): the hints
        (device, timed ``hints_host`` inside the prover's clock) then ``protocol.prove_stages``."""
        lay = bd["lay"]

        def hints_fn():
            hints = self.rel.hints(self.p, self.sys, bd["c_d"], bd["a_d"], bd["b_d"], bd["y_d"], self.device)
            # the gate: the device generator must equal the checker's gen_step (LIGERO_REFERENCE_HINTS=0 skips it on a bench's
            # warm-up sub-batch only -- minutes of Python per folded sub-batch; the constraint self-check still runs)
            if check and self.rel.reference_hints is not None and os.environ.get("LIGERO_REFERENCE_HINTS", "1") != "0":
                ref = self.rel.reference_hints(self.p, self.sys, bd["units"], self.device)
                assert torch.equal(ref, hints), f"{self.rel.name}: torch hints != gen_step"
            return hints

        return protocol_mod.prove_stages(self.p, self.sys, self.tb, self.cfg(lay.l), bd["pub"], None, self.device, check=check,
                                         mutate=mutate, chain=(lay, bd["ypub_d"]), words={"a": bd["a"], "b": bd["b"], "y16": bd["ypub"]},
                                         coins=coins, rel=self.hooks, hints_fn=hints_fn, slot=slot)

    def prove_vus_many(self, batches: list[list[tuple]], coins_factory, depth: int = 2, min_l: int = 256, vu_ids_list=None,
                       on_done=None):
        """``len(batches)`` sub-batches proved with up to ``depth`` in flight on their own CUDA streams (``pipeline.prove_many``):
        the host work of one sub-batch (challenge expansion, transcript copies, launch overhead) hides under the device work
        of the next.  ``coins_factory(i)`` is called per sub-batch when it starts, in order (the honest verifier's step 0);
        with a live verifier (``coins`` exposing ``set_statement``) the statement is fixed per sub-batch before proving, and
        the coin exchange is non-blocking (the stage yields a Future the pipeline driver polls -- ``live.LiveCoins``).
        Returns ``[(coins, proof, pubs, lay)]`` in sub-batch order and the pass wall; each proof's ``timings`` is the
        sub-batch's share of the wall (see ``pipeline._rescale``).  ``vu_ids_list`` is the hashed runner's (unused here)."""
        from .pipeline import prove_many
        bundles = [self.marshal(vus, min_l) for vus in batches]
        coins_of: list = [None] * len(batches)

        def starter(i):
            def start(slot):
                c = coins_factory(i)
                coins_of[i] = c
                if hasattr(c, "set_statement"):
                    # live verifier: the claim (a, b, ypub) is fixed before its first coin is opened; built outside prove's clock
                    from .live import set_statement
                    bd = bundles[i]
                    set_statement(c, self, (bd["a"], bd["b"], bd["ypub"]), bd["lay"], self.cfg(bd["lay"].l), self.n_proofs)
                return self._prove_stages(bundles[i], c, check=False, mutate=None, slot=slot)
            return start

        proofs, wall = prove_many([starter(i) for i in range(len(batches))], self.device, depth=depth, on_done=on_done)
        return [(coins_of[i], pf, (bd["a"], bd["b"], bd["ypub"]), bd["lay"]) for i, (pf, bd) in enumerate(zip(proofs, bundles))], wall

    def verify_vus(self, proof, pubs, lay: Layout, coins=None):
        a, b, ypub = pubs
        if coins is None and self.interactive:
            coins = proof.coins
        return verify(self.p, self.sys, self.tb, self.cfg(lay.l), None, a, b, None, proof, self.device, chain=(lay, ypub),
                      coins=coins, rel=self.hooks)

    def batch_bound(self, cfgs: list[Config], n_proofs_claimed: list[int], target_bits: float = 128.0,
                    names: list[str] | None = None) -> dict:
        """The batch accounting gate (PROTOCOL.md 8c, "Batch verification"; lane b-batch-bound, D10) for this relation's
        sub-batch proofs that each verified: the verifier's OWN bound of every proof from its configuration (one commitment
        each), union-bounded over the ``len(cfgs)`` presented, against ``target_bits``; every proof's ``n_proofs`` (the count
        its ``t`` was sized for) against that number.  The same rule and reason strings as ``vu.ChainRunner.batch_bound``
        and ``ligero-verify batch``: the union bound is enforced for every relation, not only Ampere BF16."""
        per = [soundness(c, self.tb.L, self.tb.Q, 1, chain_extras=self.chain_extras)["per_proof_total_log2"] for c in cfgs]
        return protocol_mod.batch_bound(per, n_proofs_claimed, target_bits, names)

    def verify_batch(self, items: list[tuple], target_bits: float = 128.0, n_proofs_claimed: list[int] | None = None) -> dict:
        """The Python batch verifier on in-memory proofs of this relation: ``items`` = ``(proof, pubs, lay[, coins])`` per
        sub-batch presented together, each verified as :meth:`verify_vus` does (own statement, own coins); a rejected
        sub-batch rejects the batch with ``"<sub_XX>: <reason>"``; then :meth:`batch_bound` with ``n_proofs_claimed``
        (default: this runner's ``n_proofs`` for every proof)."""
        names = [f"sub_{i:02d}" for i in range(len(items))]
        cfgs, verdicts = [], []
        for name, it in zip(names, items):
            proof, pubs, lay = it[0], it[1], it[2]
            coins = it[3] if len(it) > 3 else None
            ok, why = self.verify_vus(proof, pubs, lay, coins=coins)
            verdicts.append({"name": name, "accepted": ok, "reason": why})
            cfgs.append(self.cfg(lay.l))
        claimed = list(n_proofs_claimed) if n_proofs_claimed is not None else [self.n_proofs] * len(items)
        for v in verdicts:
            if not v["accepted"]:
                per = [soundness(c, self.tb.L, self.tb.Q, 1, chain_extras=self.chain_extras)["per_proof_total_log2"] for c in cfgs]
                return {"accepted": False, "reason": f"{v['name']}: {v['reason']}", "n_subbatches": len(items),
                        "target_bits": float(target_bits), "batch_log2": protocol_mod.log2_sum(per), "per_proof_log2": per,
                        "n_proofs_claimed": claimed, "items": verdicts}
        return {**self.batch_bound(cfgs, claimed, target_bits, names), "items": verdicts}


def hashauth_vshape(rel: Relation, hr, leaf) -> tuple[int, int, int] | None:
    """The ``(K, word_bits, y_bytes)`` a ``vllm-v1`` statement's port domains are derived from (``hashauth.vllm_shape``);
    ``None`` under frame-v3 trees."""
    from . import hashauth
    if not hashauth.is_vllm(leaf):
        return None
    return hashauth.vllm_shape(rel.steps * hr.k, hr.word_bits, rel.y_bits)


def commit_bindings(rel: Relation, leaf, *, dataset: str, tier: str, manifest_sha256: str, lo: int, hi: int) -> dict[str, bytes]:
    """Each tree's ``binding``: the frame-v3 binding digest (input set, range, K, leaf schema), or under ``vllm-v1`` the
    port's ctx digest (``vllm_tree.ctx_digest``: input set and range; the rest of the domain is the verifier's)."""
    from . import hashauth, vllm_tree
    if hashauth.is_vllm(leaf):
        return {n: vllm_tree.ctx_digest(dataset=dataset, tier=tier, manifest_sha256=manifest_sha256, lo=lo, hi=hi, tree=n)
                for n in ("a", "b", "y")}
    ysch, _ = hashauth.word_schema(rel.y_bits)
    return {n: hashauth.binding_digest(dataset=dataset, tier=tier, manifest_sha256=manifest_sha256, lo=lo, hi=hi,
                                       K=getattr(rel, "vu_words", None) or _K(rel), tree=n, schema=(leaf.schema if n != "y" else ysch))
            for n in ("a", "b", "y")}


class HashedRelationRunner(RelationChainRunner):
    """``authentication = included-hash`` (``hashchain.py``, ``hashauth.py``): the relation's chain system composed with
    private operands and the two operand-row gadgets of a pluggable leaf scheme (``leaf/``; Poseidon2 sponges by
    default); the statement is the digest table + leaf indices + roots, and the verifier checks the published digests'
    Merkle multiproofs after the Ligero verdict.

    ``pubs`` of :meth:`prove_vus` / :meth:`verify_vus` is ``(table (l, 1 + 2 digest_elems), indices (n_vus, 3), ypub (l,),
    auth, siblings)``: the public table (``hash.is_end | hash.d[..]`` per column), the ``(vu, x, W)`` leaf triples, the chain-end words,
    the :class:`hashauth.HashAuth` block and the three multiproofs.  ``commit(...)`` sets the trees the prover opens."""

    AUTH = "included-hash"

    def __init__(self, rel: Relation, device: str, target: float, rate_log2: int = 2, n_proofs: int = 1, zk: bool = False,
                 mode: str = protocol_mod.MODE_INTERACTIVE, leaf=None):
        """``leaf``: the leaf scheme (``leaf.registry`` name, default ``"poseidon2"`` -- the historical ``+hash``)."""
        from . import hashchain
        from .leaf import registry as leaf_registry
        from .protocol import RelationHooks
        self.rel = rel
        self.p = rel.params
        self.zk, self.mode, self.device, self.target, self.rate_log2, self.n_proofs = zk, mode, device, target, rate_log2, n_proofs
        self.steps = rel.steps
        t0 = time.perf_counter()
        self.hr = hashchain.compose(rel, leaf)
        self.leaf = self.hr.leaf
        self.leaf_name = leaf_registry.suffix_of(self.leaf)     # the relation suffix ("poseidon2", "blake3", "ajtai-n64")
        self.sys = self.hr.sys
        self.tb = tables(self.sys, self.device)
        if hashchain.ROW_SPONGE_KERNEL and str(device).startswith("cuda") and getattr(self.leaf, "prepare", None) is not None:
            self.leaf.prepare()
        self.compile_seconds = time.perf_counter() - t0
        self._cfg = {}
        self._base_sys = dataclasses.replace(self.sys, hints=self.sys.hints[:self.hr.n_base_hints])
        self.base_tag = b"hash/v2h|" + rel.tag
        self.hooks = RelationHooks(functools.partial(hashchain.hash_public_vectors, n_digest=2 * self.leaf.digest_elems),
                                   word_dtype="<u4", y16_dtype=rel.y_dtype, y16_bits=rel.y_bits, tag=self.base_tag,
                                   steps=rel.steps, max_steps=getattr(self.leaf, "max_steps", None))
        self.trees = None
        self.vshape = hashauth_vshape(rel, self.hr, self.leaf)   # vllm-v1 ports: (K, word_bits, y_bytes); None for frame-v3
        self.tile = None                    # (nx, nw) of the tile workload, None for the worst case
        self.rows, self.sponges, self.sponge_seconds = None, None, 0.0
        self._marshal_cache = {}

    @property
    def relation(self) -> str:
        """The v5 statement's relation string: ``<rel>`` for Poseidon2 (byte-identical to every ``+hash`` statement
        before the leaf interface), ``<rel>+<leaf>`` for any other scheme (``leaf.registry.relation_name``)."""
        from .leaf import registry as leaf_registry
        return leaf_registry.relation_name(self.rel.name, self.leaf_name)

    def hooks_for(self, auth):
        from . import hashauth
        return dataclasses.replace(self.hooks, tag=self.base_tag + hashauth.digest_bytes(auth, leaf=self.leaf, vshape=self.vshape))

    def commit(self, *, x_rows, w_cols, y_words, bindings, cache=None, tile=None):
        """Build (or load) the three trees of the committed set; ``tile = (nx, nw)`` records the sharing layout."""
        from . import hashauth, hashchain
        t0 = time.perf_counter()
        x_rows, w_cols = _word_rows(x_rows), _word_rows(w_cols)
        self.rows = {"a": x_rows, "b": w_cols}
        narrow: dict = {}
        rows_d = {nm: _rows_to_device(r, self.device, narrow, nm) for nm, r in self.rows.items()}
        if str(self.device).startswith("cuda"):
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        self.tile = tile
        # the chain states of every committed row, once (the prover's cost of hashing; gathered per sub-batch in prove_vus);
        # with the device row sponges their digests ARE the tree leaves' (scheme.native), so the trees take them
        self.sponges = {nm: hashchain.row_chain(self.hr, rows_d[nm], role, self.device) for _, nm, role in hashchain.ROLES}
        if str(self.device).startswith("cuda"):
            torch.cuda.synchronize()
        t2 = time.perf_counter()
        self.sponge_seconds = t2 - t1
        chain_digests = None
        if hashchain.ROW_SPONGE_KERNEL and getattr(self.leaf, "row_sponges", None) is not None and str(self.device).startswith("cuda"):
            from . import frame_gpu
            on_dev = frame_gpu.available() and self.leaf.name in frame_gpu.ROW_SCHEMES      # compared on the device
            chain_digests = {nm: self.sponges[nm][1] if on_dev else self.sponges[nm][1].cpu().numpy().astype(np.int64)
                             for nm in self.sponges}
        self.trees, self.digests, cached = hashauth.load_or_build_trees(
            cache, x_rows=x_rows, w_cols=w_cols,
            y_words=np.asarray(y_words, dtype=np.int64), word_bits=self.hr.word_bits, y_bits=self.rel.y_bits, bindings=bindings,
            leaf=self.leaf, digests=chain_digests, rows_dev={nm: narrow.get(nm, rows_d[nm]) for nm in rows_d})
        if str(self.device).startswith("cuda"):
            torch.cuda.synchronize()
        t3 = time.perf_counter()
        self._committed_digests(rows_d)
        # committer = what the commitment itself costs (rows to the device, row digests, framing, the trees, the y leaves);
        # chain = the prover's carried states of every committed row, precomputed here
        self.commit_timings = {"rows": t1 - t0, "chain": t2 - t1, "trees": t3 - t2, "host": time.perf_counter() - t3,
                               "committer": (t1 - t0) + (t3 - t2)}
        return cached

    def commit_evidence(self) -> dict:
        """What the committed set IS, byte for byte: the three tree references (bindings, counts, roots), the sha256 of every
        tree level, of the published row digests and of the prover's precomputed chain states (``<i8`` bytes)."""
        sha = lambda a: hashlib.sha256(np.ascontiguousarray(np.asarray(a, dtype=np.int64)).astype("<i8").tobytes()).hexdigest()  # noqa: E731
        out = {"trees": {n: {**t.ref.to_json(), "levels_sha256": hashlib.sha256(b"".join(np.ascontiguousarray(lv).tobytes()
                                                                                          for lv in t.levels)).hexdigest()}
                         for n, t in self.trees.items()},
               "digests_sha256": {n: sha(d) for n, d in self.digests.items()},
               "chain_sha256": {nm: [sha(x.cpu().numpy()) for x in sp[:2]] for nm, sp in self.sponges.items()}}
        out["sha256"] = hashlib.sha256(json.dumps(out, sort_keys=True).encode()).hexdigest()
        return out

    def _committed_digests(self, rows_d: dict | None = None):
        """Host copies of the committed rows' digests (public: the leaves of the x / W trees) for the statement tables
        of the pipelined prover, and device copies of the committed rows (the hash side gathers its witness from them).
        Only the hashed statements depend on the commitment: the base marshal of the instance words stays cached."""
        self._marshal_cache = {k: v for k, v in self._marshal_cache.items() if k[0] != "hashed"}
        self.dg_host = {nm: self.sponges[nm][1].cpu().numpy().astype(np.int64) for nm in self.sponges}
        self.rows_d = rows_d if rows_d is not None else {nm: _rows_to_device(self.rows[nm], self.device) for nm in self.rows}

    def marshal_hashed(self, vus: list, vu_ids, min_l: int = 256) -> dict:
        """The sub-batch's statement (public table from the committed digests, leaf indices, authentication block, public
        vectors) and its device-resident words, once per run (cached like :meth:`RelationChainRunner.marshal`): what
        the pipelined prover (:meth:`prove_vus_many`) reads, so a timed sub-batch does no host marshalling.  ``gather``:
        the committed-row indices per operand whose words ARE the committed rows (their precomputed chains are gathered
        on the device inside the prover's clock)."""
        vu_ids = list(vu_ids)
        key = ("hashed", len(vus), id(vus[0]), id(vus[-1]), id(vus[len(vus) // 2]), vu_ids[0], vu_ids[-1], min_l)
        hit = self._marshal_cache.get(key)
        if hit is not None:
            return hit
        from . import hashchain
        bd = dict(RelationChainRunner.marshal(self, vus, min_l))
        n_vus, lay, dev = len(vus), bd["lay"], self.device
        auth = self.auth_for(vu_ids)
        idx = auth.indices()
        gather = {}
        for nm, ix in (("a", auth.x_index), ("b", auth.w_index)):
            ix = np.asarray(ix, dtype=np.int64)
            words = bd[nm][:n_vus * self.steps].reshape(n_vus, self.steps * self.rel.k)
            if self.sponges is not None and len(ix) == n_vus and np.array_equal(self.rows[nm][ix], words):
                gather[nm] = torch.as_tensor(ix).to(dev)
        dg = np.concatenate([self.dg_host["a"][np.asarray(auth.x_index)], self.dg_host["b"][np.asarray(auth.w_index)]], 1)
        table = hashchain.public_table(dg, n_vus, self.steps, lay.l)
        bd.update({"auth": auth, "idx": idx, "table": table, "gather": gather, "hooks": self.hooks_for(auth),
                   "pub": self.hooks.public_vectors(self.p, None, table, idx, None, device=dev)})
        if str(dev).startswith("cuda"):
            torch.cuda.synchronize(dev)
        self._marshal_cache[key] = bd
        return bd

    def _gather_sponges(self, gather: dict) -> dict | None:
        out = {}
        for nm, t in gather.items():
            sp = self.sponges[nm]
            out[nm] = (sp[0][t], sp[1][t])
            if len(sp) > 2:
                out[nm] += ({n_: v[t] for n_, v in sp[2].items()},)
        return out or None

    def _prove_stages_hashed(self, bd: dict, coins, slot=None, sibs_out: list | None = None):
        """The pipelined prover's generator for one sub-batch (``pipeline.prove_many``): the hints on the device (the
        base generator + the leaf chains gathered from the committed set, no host round trip), ``protocol.prove_stages``,
        then the multiproofs (host, timed ``auth_openings``)."""
        from . import hashauth, hashchain
        lay, dev, n_vus = bd["lay"], self.device, bd["lay"].n_vus

        def hints_fn():
            base = hashchain.call_with_slot(self.rel.hints, slot, self.p, self._base_sys, bd["c_d"], bd["a_d"], bd["b_d"],
                                            bd["y_d"], dev)
            extra, _ = hashchain.hash_hints(self.hr, bd["a_d"], bd["b_d"], n_vus, lay.l, dev,
                                            sponges=self._gather_sponges(bd["gather"]), slot=slot)
            return torch.cat([base, extra], 0)

        proof = yield from protocol_mod.prove_stages(
            self.p, self.sys, self.tb, self.cfg(lay.l), bd["pub"], None, dev, chain=(lay, bd["ypub_d"]),
            words={"a": bd["table"], "b": bd["idx"], "y16": bd["ypub"]}, coins=coins, rel=bd["hooks"], hints_fn=hints_fn,
            slot=slot)
        t1 = time.perf_counter()
        sibs = hashauth.open_trees(self.trees, bd["auth"])
        proof.timings["auth_openings"] = time.perf_counter() - t1
        if sibs_out is not None:
            sibs_out.append(sibs)
        return proof

    def _pipelined(self, bundles: list, stage, coins_factory, depth: int, live_statement=None, on_done=None):
        """``prove_many`` over ``stage(bundle, coins, slot, sibs_out)`` generators.  Local coins, or -- when the runner
        passes ``live_statement(bundle, coins)`` (the statement, multiproofs included, handed to the live source before
        the root) -- a live verifier's coins; without it a live source is refused."""
        from .pipeline import prove_many
        coins_of: list = [None] * len(bundles)
        sibs_of: list = [[] for _ in bundles]

        def starter(i):
            def start(slot):
                c = coins_factory(i)
                if hasattr(c, "set_statement"):
                    if live_statement is None:
                        raise NotImplementedError(f"--auth {self.AUTH} --pipeline: live verifier coins are not wired (local coins only)")
                    live_statement(bundles[i], c)
                coins_of[i] = c
                return stage(bundles[i], c, slot, sibs_of[i])
            return start

        proofs, wall = prove_many([starter(i) for i in range(len(bundles))], self.device, depth=depth, on_done=on_done)
        return proofs, [s[0] for s in sibs_of], coins_of, wall

    def prove_vus_many(self, batches: list[list[tuple]], coins_factory, depth: int = 2, min_l: int = 256, vu_ids_list=None,
                       on_done=None):
        """:meth:`RelationChainRunner.prove_vus_many` for the hashed statement: ``vu_ids_list[i]`` the committed-set VU
        indices of ``batches[i]`` (default: consecutive from 0).  Returns ``[(coins, proof, pubs, lay)]`` and the wall."""
        if vu_ids_list is None:
            vu_ids_list, at = [], 0
            for vus in batches:
                vu_ids_list.append(range(at, at + len(vus)))
                at += len(vus)
        bundles = [self.marshal_hashed(vus, ids, min_l) for vus, ids in zip(batches, vu_ids_list)]

        def live_statement(bd, c):
            # live verifier: the statement -- multiproofs included -- travels with the root (as prove_vus)
            from . import hashauth
            from .live import set_statement
            sibs = hashauth.open_trees(self.trees, bd["auth"])
            set_statement(c, self, (bd["table"], bd["idx"], bd["ypub"], bd["auth"], sibs), bd["lay"], self.cfg(bd["lay"].l), self.n_proofs)

        proofs, sibs, coins_of, wall = self._pipelined(
            bundles, lambda bd, c, slot, so: self._prove_stages_hashed(bd, c, slot, so), coins_factory, depth,
            live_statement=live_statement, on_done=on_done)
        return [(coins_of[i], pf, (bd["table"], bd["idx"], bd["ypub"], bd["auth"], sibs[i]), bd["lay"])
                for i, (pf, bd) in enumerate(zip(proofs, bundles))], wall

    def committed_rows(self, data) -> tuple:
        """The worst-case committed set of an instance list (every VU its own x row and W column) as stored: ``(x_rows,
        w_cols)`` in the words' dtype, page-locked for a CUDA runner (:func:`narrow_rows`)."""
        x_rows, w_cols = getattr(data, "x_rows", None), getattr(data, "w_cols", None)
        if x_rows is None or w_cols is None:
            x_rows, w_cols = [v[0] for v in data], [v[1] for v in data]
        pin = str(self.device).startswith("cuda")
        return narrow_rows(x_rows, self.hr.word_bits, pin), narrow_rows(w_cols, self.hr.word_bits, pin)

    def commit_vus(self, data: list, dataset: str = "synthetic", tier: str = "dev", manifest_sha256: str = "", cache=None, tile=None,
                   x_rows=None, w_cols=None):
        """Commit an instance list: the worst case (every VU its own x row and W column: ``x_rows = a``, ``w_cols = b``) or a
        tile (``x_rows`` / ``w_cols`` given, ``tile = (nx, nw)``)."""
        from . import hashauth
        y = [self.rel.y_public(int(v[3])) for v in data]
        if tile is None and (x_rows is None or w_cols is None):
            x_rows, w_cols = self.committed_rows(data)
        bindings = commit_bindings(self.rel, self.leaf, dataset=dataset, tier=tier, manifest_sha256=manifest_sha256, lo=0, hi=len(data))
        return self.commit(x_rows=x_rows, w_cols=w_cols, y_words=y, bindings=bindings, cache=cache, tile=tile)

    def auth_for(self, vu_ids: list[int]):
        from . import hashauth
        if self.trees is None:
            raise RuntimeError("commit(...) first")
        if self.tile is None:
            xi, wi = list(vu_ids), list(vu_ids)
        else:
            _, xi, wi = tile_indices(*self.tile, min(vu_ids), max(vu_ids) + 1)
            if list(range(min(vu_ids), max(vu_ids) + 1)) != list(vu_ids):
                xi = [v // self.tile[1] for v in vu_ids]
                wi = [v % self.tile[1] for v in vu_ids]
        return hashauth.HashAuth.for_vus(self.trees, vu_ids, xi, wi)

    def _sponges_for(self, auth, a, b, n_vus: int) -> dict | None:
        """The committed rows' precomputed sponges for this sub-batch, per operand whose words ARE the committed rows
        (else None for that operand: the sponge runs on the words the prover computes on -- negatives, overrides)."""
        if self.sponges is None:
            return None
        out = {}
        for nm, idx in (("a", auth.x_index), ("b", auth.w_index)):
            idx = np.asarray(idx, dtype=np.int64)
            words = (a if nm == "a" else b)[:n_vus * self.steps].reshape(n_vus, self.steps * self.hr.k)
            if len(idx) == n_vus and (idx < len(self.rows[nm])).all() and np.array_equal(self.rows[nm][idx], words):
                cap, dg = self.sponges[nm][0], self.sponges[nm][1]
                t = torch.as_tensor(idx, device=cap.device)
                out[nm] = (cap[t], dg[t])
                if len(self.sponges[nm]) > 2:
                    out[nm] += ({n_: v[t] for n_, v in self.sponges[nm][2].items()},)
        return out or None

    def prove_vus(self, vus, check=True, mutate=None, min_l: int = 256, coins=None, vu_ids=None, auth=None, words_override=None):
        """As the base runner; ``vu_ids`` (default ``range(len(vus))``) are the committed set's VU indices.  ``auth`` overrides
        the block (negatives); ``words_override(a, b)`` edits the operand words the prover computes on AFTER the units
        are formed (negatives: "computed on another row")."""
        from . import hashauth, hashchain
        if coins is None and self.interactive:
            coins = protocol_mod.Coins.sample()
        vu_ids = list(range(len(vus))) if vu_ids is None else list(vu_ids)
        auth = self.auth_for(vu_ids) if auth is None else auth
        lay = self.layout(len(vus), min_l)
        units, ypub = [], torch.zeros(lay.l, dtype=torch.int64)
        for v, (a, b, accs, y) in enumerate(vus):
            units += vu_units(self.rel, a, b, accs)
            ypub[self.steps * v + self.steps - 1] = self.rel.y_public(int(y))
        while len(units) < lay.l:
            units.append(self._pad_unit())
        a = np.asarray([u[1] for u in units], dtype=np.int64)
        b = np.asarray([u[2] for u in units], dtype=np.int64)
        c_in = np.asarray([u[0] for u in units], dtype=np.int64)
        y_out = np.asarray([u[3] for u in units], dtype=np.int64)
        if words_override is not None:
            a, b = words_override(a, b)
        t0 = time.perf_counter()
        base = self.rel.hints(self.p, self._base_sys, c_in, a, b, y_out, self.device)
        extra, digests = hashchain.hash_hints(self.hr, a, b, len(vus), lay.l, self.device,
                                              sponges=self._sponges_for(auth, a, b, len(vus)))
        hints = torch.cat([base, extra], 0)
        table = hashchain.public_table(digests.cpu().numpy(), len(vus), self.steps, lay.l)
        idx = auth.indices()
        pub = self.hooks.public_vectors(self.p, None, table, idx, None, device=self.device)
        if str(self.device).startswith("cuda"):
            torch.cuda.synchronize()
        hints_seconds = time.perf_counter() - t0
        hooks = self.hooks_for(auth)
        sibs = None
        if hasattr(coins, "set_statement"):
            # a live verifier's coin source (live.LiveCoins): the v5 statement -- multiproofs included -- travels with the
            # root, so the openings are taken before prove here (outside prove's clock either way).
            t1 = time.perf_counter()
            sibs = hashauth.open_trees(self.trees, auth)
            openings = time.perf_counter() - t1
            from .live import set_statement
            set_statement(coins, self, (table, idx, ypub, auth, sibs), lay, self.cfg(lay.l), self.n_proofs)
        proof = prove(self.p, self.sys, self.tb, self.cfg(lay.l), pub, hints, self.device, check=check, mutate=mutate,
                      chain=(lay, ypub), words={"a": table, "b": idx}, coins=coins, rel=hooks)
        if sibs is None:
            t1 = time.perf_counter()
            sibs = hashauth.open_trees(self.trees, auth)
            openings = time.perf_counter() - t1
        proof.timings["auth_openings"] = openings
        proof.timings["hints_host"] = hints_seconds
        return proof, (table, idx, ypub, auth, sibs), lay

    def _system_key_error(self) -> str | None:
        """A keyed leaf's check of the compiled system (Ajtai: the chain the digests come from uses the key re-derived
        from the public tag; red-team-leaf-2 G1), once per system."""
        check = getattr(self.leaf, "check_system", None)
        if check is None:
            return None
        if getattr(self, "_key_checked", None) is not self.sys:
            try:
                check(self.sys)
                self._key_error = None
            except ValueError as e:
                self._key_error = f"system: the {self.leaf.name} chain does not use the derived key: {e}"
            self._key_checked = self.sys
        return self._key_error

    def verify_vus(self, proof, pubs, lay: Layout, coins=None, auth_timing: dict | None = None):
        from . import hashauth, hashchain
        table, idx, ypub, auth, sibs = pubs
        if coins is None and self.interactive:
            coins = proof.coins
        if not np.array_equal(np.asarray(idx, dtype=np.int64), auth.indices()) or auth.n_vus != lay.n_vus:
            return False, "authentication block does not match the statement's indices / layout"
        key_err = self._system_key_error()
        if key_err:
            return False, key_err
        ok, why = verify(self.p, self.sys, self.tb, self.cfg(lay.l), None, table, idx, None, proof, self.device,
                         chain=(lay, ypub), coins=coins, rel=self.hooks_for(auth))
        if not ok:
            return ok, why
        t0 = time.perf_counter()
        ends = np.arange(lay.n_vus) * lay.steps + lay.steps - 1
        tab = np.asarray(table, dtype=np.int64)
        if (tab[ends, 0] != 1).any() or tab[:, 0].sum() != lay.n_vus or tab[np.setdiff1d(np.arange(lay.l), ends)].any():
            return False, "public table: is_end / digests off the chain ends"
        y_words = np.asarray(ypub, dtype=np.int64)[ends]
        ok, why = hashauth.verify_hash_auth(auth, sibs, digests=tab[ends, 1:1 + 2 * self.leaf.digest_elems], y_words=y_words,
                                            y_bits=self.rel.y_bits, leaf=self.leaf, vshape=self.vshape)
        if auth_timing is not None:
            auth_timing["seconds"] = auth_timing.get("seconds", 0.0) + (time.perf_counter() - t0)
        return ok, why


INTERLEAVE_PAIR = True      # the pipelined pair: H on its own slot, interleaved with G (False: G then H on one slot)
H_SLOT_BASE = 64            # pipeline.slot_for index of the hash side's slot for job slot i: H_SLOT_BASE + i


@dataclasses.dataclass
class SharedProof:
    """The paired transcript of one sub-batch under ``authentication = included-hash-shared``: ``g`` the GEMM side
    (``hashchain.compose_shared``: private operands, lane rows, the fingerprint message ``g.fp``), ``h`` the hash side
    (``hashchain.row_hash_system``: one Poseidon2 sponge per DISTINCT operand row of the sub-batch, its fingerprint a
    statement value).  ``timings`` is the sum of the two provers' clocks (``total`` included)."""
    g: protocol_mod.Proof
    h: protocol_mod.Proof
    timings: dict
    lay_h: Layout                   # the hash side's layout (units = distinct rows; the verifier derives it from auth)

    @property
    def coins(self):
        return (self.g.coins, self.h.coins)

    @property
    def hash_name(self):
        return self.g.hash_name

    def n_bytes(self) -> dict:
        bg, bh = self.g.n_bytes(), self.h.n_bytes()
        out = {k: bg.get(k, 0) + bh.get(k, 0) for k in set(bg) | set(bh)}
        out["g_total"], out["h_total"] = bg["total"], bh["total"]
        return out


class SharedHashedRunner(HashedRelationRunner):
    """``authentication = included-hash-shared`` (lane share-logup; chain.py "Fingerprint constraints"): every operand row
    is hashed ONCE per sub-batch, not once per VU.  Per sub-batch two Ligero proofs share one session:

    * **G** (``hashchain.compose_shared``): the relation with private operands and 16 lane rows per operand per unit, no
      sponge.  After the root, challenge 1 also yields the fingerprint coins ``rho`` (C = 8); the prover sends
      ``F_G[v][o][x] = sum_{s,i} rho_x^(16 s + i) lane_o[i][steps v + s]`` (``Proof.fp``) and the chain test checks it
      against the committed lanes (no witness rows: rho-dependent coefficients of the general linear test).
    * **H** (``hashchain.row_hash_system``): one unit per DISTINCT x row / W column the sub-batch's VUs claim
      (``hashchain.shared_units`` from the authentication block), each hashing its row with the role's IV over ``steps``
      columns, its digest a chain-end pin, and the SAME fingerprint as a statement value: ``rho`` = G's coins, ``F_H[u]``
      = the F_G of the VUs that use the row.

    The verifier: G accepts (its ``rho`` is the verifier's own derivation), the VUs sharing a unit sent equal F, H accepts
    on the statement ``(table, units, rho, F_H)``, the table's roles / ends match the units, and H's digests open as
    leaf/v2h leaves at the units' ranks under the x / W roots (``hashauth.verify_hash_auth`` on the per-VU expansion).
    Soundness: a VU whose G row differs from the committed row of its claimed leaf must collide on all 8 fingerprint
    coordinates (``protocol.fingerprint_collision``: (16 steps - 1)^8 / p^8 ~ 2^-171 at 48 steps), or H must open a
    Poseidon2 preimage of the leaf digest other than the committed row.  ``pubs`` of :meth:`prove_vus` /
    :meth:`verify_vus`: ``(is_end (l, 1), indices (n_vus, 3), ypub (l,), auth, siblings, table_h (l_h, 10),
    units (n_units, 2))``."""

    AUTH = "included-hash-shared"

    def __init__(self, rel: Relation, device: str, target: float, rate_log2: int = 2, n_proofs: int = 1, zk: bool = False,
                 mode: str = protocol_mod.MODE_INTERACTIVE, leaf=None):
        """``leaf``: accepted for the runner interface; the hash side is the Poseidon2 sponge of ``hashchain.row_hash_system``,
        so only the default scheme (``None`` / ``"poseidon2"`` / ``"hash"``) is allowed."""
        from . import hashchain
        from .leaf import registry as leaf_registry
        from .protocol import RelationHooks
        leaf_n = leaf_registry.canonical(leaf if leaf is None or isinstance(leaf, str) else leaf_registry.suffix_of(leaf))
        if leaf_n != leaf_registry.DEFAULT:
            raise ValueError(f"--auth {self.AUTH}: row sharing hashes with Poseidon2 only (hashchain.row_hash_system); got leaf {leaf_n!r}")
        self.rel = rel
        self.p = rel.params
        self.zk, self.mode, self.device, self.target, self.rate_log2, self.n_proofs = zk, mode, device, target, rate_log2, n_proofs
        self.steps = rel.steps
        t0 = time.perf_counter()
        from .chain import FP_COORDS, FP_COORDS_FS
        self.fp_coords = FP_COORDS if mode == protocol_mod.MODE_INTERACTIVE else FP_COORDS_FS
        self.hr = hashchain.compose_shared(rel, self.fp_coords)
        self.leaf = self.hr.leaf
        self.leaf_name = leaf_registry.suffix_of(self.leaf)
        self.vshape = None                  # Poseidon2 frame-v3 only: no vllm-v1 ports
        self.sys = self.hr.sys
        self.tb = tables(self.sys, self.device)
        self.rh = hashchain.row_hash_system(rel, self.fp_coords)
        self.sys_h = self.rh.sys
        self.tb_h = tables(self.sys_h, self.device)
        self.compile_seconds = time.perf_counter() - t0
        self._cfg = {}
        self._cfg_h: dict[int, Config] = {}
        self._base_sys = dataclasses.replace(self.sys, hints=self.sys.hints[:self.hr.n_base_hints])
        self.base_tag = hashchain.SHARED_TAG + rel.tag
        self.base_tag_h = hashchain.ROWS_TAG + rel.tag
        self.hooks = RelationHooks(hashchain.shared_public_vectors, word_dtype="<u4", y16_dtype=rel.y_dtype,
                                   y16_bits=rel.y_bits, tag=self.base_tag, steps=rel.steps)
        self.hooks_h = RelationHooks(hashchain.row_hash_public_vectors, word_dtype="<u4", y16_dtype=rel.y_dtype,
                                     y16_bits=rel.y_bits, tag=self.base_tag_h, steps=rel.steps)
        self.trees = None
        self.tile = None
        self.rows, self.sponges, self.sponge_seconds = None, None, 0.0
        self._marshal_cache = {}

    # both proofs of a sub-batch are sized for the union over 2 n_proofs commitments (PROTOCOL.md 8c, batch verification)
    def cfg(self, l: int) -> Config:
        if l not in self._cfg:
            self._cfg[l] = config_for(l, self.target, self.rate_log2, 2 * self.n_proofs, zk=self.zk, mode=self.mode,
                                      chain_extras=self.chain_extras)
        return self._cfg[l]

    def cfg_h(self, l: int) -> Config:
        if l not in self._cfg_h:
            self._cfg_h[l] = config_for(l, self.target, self.rate_log2, 2 * self.n_proofs, zk=self.zk, mode=self.mode,
                                        chain_extras=self.chain_extras_h)
        return self._cfg_h[l]

    @property
    def chain_extras_h(self) -> int:
        return extra_constraints(self.sys_h.chain)

    def hooks_h_for(self, auth):
        from . import hashauth
        return dataclasses.replace(self.hooks_h, tag=self.base_tag_h + hashauth.digest_bytes(auth))

    def commit(self, *, x_rows, w_cols, y_words, bindings, cache=None, tile=None):
        """The trees as the hashed runner builds them, and the sponge of every committed row ONCE (the hash side's
        witness is a function of the committed rows alone: gathered per sub-batch in :meth:`prove_vus`, as the hashed
        runner does; ~2 x 64 rows at the 64x64 tile against 2 x 4096 for it)."""
        from . import hashauth, hashchain
        x_rows, w_cols = np.asarray(x_rows, dtype=np.int64), np.asarray(w_cols, dtype=np.int64)
        self.trees, self.digests, cached = hashauth.load_or_build_trees(
            cache, x_rows=x_rows, w_cols=w_cols,
            y_words=np.asarray(y_words, dtype=np.int64), word_bits=self.hr.word_bits, y_bits=self.rel.y_bits, bindings=bindings)
        self.tile = tile
        self.rows = {"a": x_rows, "b": w_cols}
        t0 = time.perf_counter()
        self.sponges = {nm: hashchain.row_sponges(self.rh, self.rows[nm], role, self.device) for _, nm, role in hashchain.ROLES}
        if str(self.device).startswith("cuda"):
            torch.cuda.synchronize()
        self.sponge_seconds = time.perf_counter() - t0
        self._committed_digests()
        return cached

    def marshal_shared(self, vus: list, vu_ids, min_l: int = 256) -> dict:
        """The pair's statements for one sub-batch, once per run (as :meth:`marshal_hashed`): G's ``is_end`` / indices /
        public vectors / device words; H's units, layout, public table (roles + the committed digests at the unit ends),
        public vectors, and the device indices of its rows in the committed set (H's witness is gathered from the
        committed rows and their precomputed sponges inside the prover's clock)."""
        vu_ids = list(vu_ids)
        key = ("shared", len(vus), id(vus[0]), id(vus[-1]), id(vus[len(vus) // 2]), vu_ids[0], vu_ids[-1], min_l)
        hit = self._marshal_cache.get(key)
        if hit is not None:
            return hit
        from . import hashchain
        bd = dict(RelationChainRunner.marshal(self, vus, min_l))
        n_vus, lay, dev, steps = len(vus), bd["lay"], self.device, self.steps
        auth = self.auth_for(vu_ids)
        idx = auth.indices()
        is_end = np.zeros((lay.l, 1), dtype=np.int64)
        is_end[np.arange(n_vus) * steps + steps - 1, 0] = 1
        units_np, ux, uw = hashchain.shared_units(auth)
        n_units = len(units_np)
        lay_h = self.layout_h(n_units, lay.l)
        rx, rw = units_np[units_np[:, 0] == 0, 1], units_np[units_np[:, 0] == 1, 1]
        table_h = np.zeros((lay_h.l, 2 + hashchain.N_ROW_DIGEST), dtype=np.int64)
        ends_h = np.arange(n_units) * steps + steps - 1
        table_h[ends_h, 0] = 1
        table_h[:n_units * steps, 1] = np.repeat(units_np[:, 0], steps)
        table_h[ends_h, 2:] = np.concatenate([self.dg_host["a"][rx], self.dg_host["b"][rw]], 0)
        bd.update({"auth": auth, "idx": idx, "is_end": is_end, "hooks": self.hooks_for(auth),
                   "pub": hashchain.shared_public_vectors(self.p, None, is_end, idx, None, device=dev),
                   "units": units_np, "ux": ux, "uw": uw, "lay_h": lay_h, "table_h": table_h,
                   "hooks_h": self.hooks_h_for(auth),
                   "pub_h": hashchain.row_hash_public_vectors(self.p, None, table_h, units_np, None, device=dev),
                   "gx": torch.as_tensor(rx).to(dev), "gw": torch.as_tensor(rw).to(dev),
                   "ypub_h": torch.zeros(lay_h.l, dtype=torch.int64), "ypub_h_d": torch.zeros(lay_h.l, dtype=torch.int64).to(dev)})
        if str(dev).startswith("cuda"):
            torch.cuda.synchronize(dev)
        self._marshal_cache[key] = bd
        return bd

    def _prove_stages_shared(self, bd: dict, coins, coins_h, slot=None, sibs_out: list | None = None, sibs_fut=None):
        """The pair of one sub-batch as ONE generator for ``pipeline.prove_many``, the two provers interleaved: H (its
        witness gathered on the device from the committed rows + sponges; its commitment does not depend on G) starts
        first on its own slot -- stream and pinned buffers -- and waits after its root for its fingerprint statement;
        G (hints on the device) runs on the job's slot and hands ``(rho, F_G)`` over as soon as its messages are on the
        host; F_H (host, vectorised) resolves H's statement; the multiproofs are opened on a pool thread meanwhile.
        The transcripts are those of :meth:`prove_vus` (H's root is committed before rho is drawn instead of after:
        the witness it commits to is a function of the committed rows alone)."""
        from concurrent.futures import Future

        from . import hashauth, hashchain, pipeline
        from .chain import FpArgs
        lay, lay_h, dev = bd["lay"], bd["lay_h"], self.device
        if slot is None or not INTERLEAVE_PAIR:
            return (yield from self._prove_stages_shared_seq(bd, coins, coins_h, slot, sibs_out, sibs_fut=sibs_fut))
        slot_h = pipeline.slot_for(dev, H_SLOT_BASE + slot.index)

        def hints_g():
            base = self.rel.hints(self.p, self._base_sys, bd["c_d"], bd["a_d"], bd["b_d"], bd["y_d"], dev)
            return torch.cat([base, hashchain.shared_hints(self.hr, bd["a_d"], bd["b_d"], lay.l, dev)], 0)

        def hints_h():
            gx, gw = bd["gx"], bd["gw"]
            rows = torch.cat([self.rows_d["a"][gx], self.rows_d["b"][gw]], 0)
            cap_u = torch.cat([self.sponges["a"][0][gx], self.sponges["b"][0][gw]], 0)
            return hashchain.row_hash_witness(self.rh, rows, lay_h.n_vus, lay_h.l, cap_units=cap_u)

        if sibs_fut is None:
            sibs_fut = pipeline.POOL.submit(hashauth.open_trees, self.trees, bd["auth"])
        fp_sink, fp_h = Future(), Future()
        gen_h = protocol_mod.prove_stages(
            self.p, self.sys_h, self.tb_h, self.cfg_h(lay_h.l), bd["pub_h"], None, dev, chain=(lay_h, bd["ypub_h_d"], fp_h),
            words={"a": bd["table_h"], "b": bd["units"], "y16": bd["ypub_h"]}, coins=coins_h, rel=bd["hooks_h"],
            hints_fn=hints_h, slot=slot_h)
        gen_g = protocol_mod.prove_stages(
            self.p, self.sys, self.tb, self.cfg(lay.l), bd["pub"], None, dev, chain=(lay, bd["ypub_d"]),
            words={"a": bd["is_end"], "b": bd["idx"], "y16": bd["ypub"]}, coins=coins, rel=bd["hooks"], hints_fn=hints_g,
            slot=slot, fp_sink=fp_sink)
        t_fp = 0.0
        gens = {"h": (gen_h, slot_h), "g": (gen_g, slot)}
        out, waits = {}, {"h": None, "g": None}           # H first: its commitment launches ahead of G's
        started = set()

        def step(name):
            gen, s = gens[name]
            with pipeline.stream_context(s):
                if name in started:
                    return gen.send(None)
                started.add(name)
                return next(gen)

        while waits:
            if fp_sink.done() and not fp_h.done():
                t0 = time.perf_counter()
                rho_np, fp_np = fp_sink.result()
                F_h = self._fp_h(fp_np.astype(np.int64), bd["ux"], bd["uw"], lay_h.n_vus, lay_h.l // self.steps)
                if F_h is None:
                    raise ValueError("G's fingerprints disagree between VUs of one shared row")
                fp_h.set_result(FpArgs(torch.as_tensor(rho_np.astype(np.int64)), torch.as_tensor(F_h)))
                t_fp += time.perf_counter() - t0
            ready = [nm for nm, w in waits.items() if pipeline._is_ready(w)]
            if not ready:
                yield pipeline.AnyOf(waits.values())
                continue
            for nm in ready:
                try:
                    waits[nm] = step(nm)
                except StopIteration as stop:
                    out[nm] = stop.value
                    del waits[nm]
        proof_g, proof_h = out["g"], out["h"]
        t1 = time.perf_counter()
        if not sibs_fut.done():
            yield sibs_fut
        sibs = sibs_fut.result()
        if sibs_out is not None:
            sibs_out.append(sibs)
        timings = {k: proof_g.timings.get(k, 0.0) + proof_h.timings.get(k, 0.0) for k in set(proof_g.timings) | set(proof_h.timings)}
        timings["auth_openings"] = time.perf_counter() - t1
        timings["hints_host"] = timings.get("hints_host", 0.0) + t_fp
        return SharedProof(proof_g, proof_h, timings, lay_h)

    def _prove_stages_shared_seq(self, bd: dict, coins, coins_h, slot=None, sibs_out: list | None = None, sibs_fut=None):
        """The pair of one sub-batch as ONE generator, the two provers in sequence on one slot: G (hints on the device,
        the stages up to its openings), then F_H from G's message (host, vectorised) and H, then the multiproofs."""
        from . import hashauth, hashchain
        from .chain import FpArgs
        lay, lay_h, dev = bd["lay"], bd["lay_h"], self.device

        def hints_g():
            base = self.rel.hints(self.p, self._base_sys, bd["c_d"], bd["a_d"], bd["b_d"], bd["y_d"], dev)
            return torch.cat([base, hashchain.shared_hints(self.hr, bd["a_d"], bd["b_d"], lay.l, dev)], 0)

        def hints_h():
            gx, gw = bd["gx"], bd["gw"]
            rows = torch.cat([self.rows_d["a"][gx], self.rows_d["b"][gw]], 0)
            cap_u = torch.cat([self.sponges["a"][0][gx], self.sponges["b"][0][gw]], 0)
            return hashchain.row_hash_witness(self.rh, rows, lay_h.n_vus, lay_h.l, cap_units=cap_u)

        proof_g = yield from protocol_mod.prove_stages(
            self.p, self.sys, self.tb, self.cfg(lay.l), bd["pub"], None, dev, chain=(lay, bd["ypub_d"]),
            words={"a": bd["is_end"], "b": bd["idx"], "y16": bd["ypub"]}, coins=coins, rel=bd["hooks"], hints_fn=hints_g,
            slot=slot)
        t0 = time.perf_counter()
        F_h = self._fp_h(proof_g.fp.astype(np.int64), bd["ux"], bd["uw"], lay_h.n_vus, lay_h.l // self.steps)
        if F_h is None:
            raise ValueError("G's fingerprints disagree between VUs of one shared row")
        fp_h = FpArgs(torch.as_tensor(proof_g.rho.astype(np.int64)), torch.as_tensor(F_h))
        t_fp = time.perf_counter() - t0
        proof_h = yield from protocol_mod.prove_stages(
            self.p, self.sys_h, self.tb_h, self.cfg_h(lay_h.l), bd["pub_h"], None, dev, chain=(lay_h, bd["ypub_h_d"], fp_h),
            words={"a": bd["table_h"], "b": bd["units"], "y16": bd["ypub_h"]}, coins=coins_h, rel=bd["hooks_h"],
            hints_fn=hints_h, slot=slot)
        t1 = time.perf_counter()
        sibs = sibs_fut.result() if sibs_fut is not None else hashauth.open_trees(self.trees, bd["auth"])
        openings = time.perf_counter() - t1
        if sibs_out is not None:
            sibs_out.append(sibs)
        timings = {k: proof_g.timings.get(k, 0.0) + proof_h.timings.get(k, 0.0) for k in set(proof_g.timings) | set(proof_h.timings)}
        timings["auth_openings"] = openings
        timings["hints_host"] = timings.get("hints_host", 0.0) + t_fp
        return SharedProof(proof_g, proof_h, timings, lay_h)

    def prove_vus_many(self, batches: list[list[tuple]], coins_factory, depth: int = 2, min_l: int = 256, vu_ids_list=None):
        """The pipelined pair: ``len(batches)`` sub-batches, each ONE job (G then H on the job's stream), up to ``depth``
        jobs in flight (``pipeline.prove_many``).  ``coins_factory(i)`` gives G's step-0 coins; H's are sampled when the
        job starts (interactive mode).  ``timings``: the job's share of the wall; ``g_total`` / ``h_total`` split it in
        the ratio of the two sides' event-measured clocks."""
        if vu_ids_list is None:
            vu_ids_list, at = [], 0
            for vus in batches:
                vu_ids_list.append(range(at, at + len(vus)))
                at += len(vus)
        bundles = [self.marshal_shared(vus, ids, min_l) for vus, ids in zip(batches, vu_ids_list)]
        live_sibs: dict[int, object] = {}

        def live_statement(bd, c):
            # the pair's v6 statement (multiproofs included) on the pool, before either root; the stage reuses the openings
            from . import hashauth, pipeline
            from .serialize import _statement_of, statement_bytes
            sibs_fut = pipeline.POOL.submit(hashauth.open_trees, self.trees, bd["auth"])
            live_sibs[id(bd)] = sibs_fut

            def build():
                pubs = (bd["is_end"], bd["idx"], bd["ypub"], bd["auth"], sibs_fut.result(), bd["table_h"], bd["units"])
                return statement_bytes(_statement_of(self, pubs, bd["lay"], self.cfg(bd["lay"].l), self.n_proofs))
            c.set_statement(pipeline.POOL.submit(build) if getattr(c.session, "pipelined", False) else build())

        def stage(bd, c, slot, so):
            ch = self._coins_h_for(c)
            return self._prove_stages_shared(bd, c, ch, slot, so, sibs_fut=live_sibs.pop(id(bd), None))

        proofs, sibs, coins_of, wall = self._pipelined(bundles, stage, coins_factory, depth, live_statement=live_statement)
        out = []
        for i, (pf, bd) in enumerate(zip(proofs, bundles)):
            g_m = pf.g.timings.get("total", 0.0) + pf.g.timings.get("hints_host", 0.0)
            h_m = pf.h.timings.get("total", 0.0) + pf.h.timings.get("hints_host", 0.0)
            share = pf.timings["total"] + pf.timings.get("hints_host", 0.0)
            pf.timings["g_total"] = share * g_m / max(g_m + h_m, 1e-12)
            pf.timings["h_total"] = share - pf.timings["g_total"]
            out.append((coins_of[i], pf, (bd["is_end"], bd["idx"], bd["ypub"], bd["auth"], sibs[i], bd["table_h"], bd["units"]),
                        bd["lay"]))
        return out, wall

    def _coins_h_for(self, coins, coins_h=None):
        """The hash side's step-0 coin source.  G's source live (``live.LiveCoins``): H's is the SAME verifier's second
        coin set of this pair (``coins.h_coins()``) and a prover-sampled one is refused -- in an interactive proof every
        coin comes from the verifier.  Local G coins: ``coins_h`` or a fresh local sample (interactive mode)."""
        if hasattr(coins, "set_statement"):
            if coins_h is None:
                coins_h = coins.h_coins()
            if not hasattr(coins_h, "set_statement") or getattr(coins_h, "session", None) is not coins.session:
                raise ValueError("live verifier: the hash side's coins must come from the same live session (prover-sampled coins_h refused)")
            return coins_h
        if coins_h is None and self.interactive:
            coins_h = protocol_mod.Coins.sample()
        return coins_h

    def _sponges_for_units(self, units_np: np.ndarray, rows: np.ndarray):
        """The precomputed sponges of the hash side's units, or None when ``rows`` are not the committed rows (overrides)."""
        if self.sponges is None or not np.array_equal(self._unit_rows(units_np), rows):
            return None
        caps, dgs = [], []
        for role, nm in ((0, "a"), (1, "b")):                    # units: the x rows (role 0) then the W columns (role 1)
            ranks = units_np[units_np[:, 0] == role, 1]
            cap, dg = self.sponges[nm]
            t = torch.as_tensor(ranks, device=cap.device)
            caps.append(cap[t])
            dgs.append(dg[t])
        return torch.cat(caps), torch.cat(dgs)

    def layout_h(self, n_units: int, l_g: int = 256) -> Layout:
        """The hash side's layout from the unit count and G's ``l`` alone (the verifier's derivation): the smallest power of
        two holding the units, floored at ``l_g // 4`` so the sub-batches of one batch share a layout (the tests graph is
        captured per layout; a 64x64 tile's sub-batches touch 70 or 71 rows, both under l_g / 4 = 4096 at l_g = 16384)."""
        l = max(256, l_g // 4, 1 << (n_units * self.steps - 1).bit_length())
        return Layout(l, n_units, self.steps)

    def _unit_rows(self, units: np.ndarray) -> np.ndarray:
        """(n_units, steps k) the committed words of every hash-side unit."""
        out = np.zeros((len(units), self.steps * self.rel.k), dtype=np.int64)
        for u, (role, rank) in enumerate(units.tolist()):
            out[u] = self.rows["a" if role == 0 else "b"][rank]
        return out

    @staticmethod
    def _fp_h(fp_g: np.ndarray, ux: np.ndarray, uw: np.ndarray, n_units: int, n_slots: int):
        """``F_H`` (n_slots, 1, C) from G's message (n_vus, 2, C) through the VU -> unit maps; ``None`` if two VUs of one
        unit sent different fingerprints (the verifier's rejection; the honest prover never hits it)."""
        C = fp_g.shape[2]
        F = np.zeros((n_slots, 1, C), dtype=np.int64)
        for o, um in ((0, ux), (1, uw)):
            F[um, 0] = fp_g[:, o]
        for o, um in ((0, ux), (1, uw)):
            if not np.array_equal(F[um, 0], fp_g[:, o]):
                return None
        return F

    def prove_vus(self, vus, check=True, mutate=None, min_l: int = 256, coins=None, vu_ids=None, auth=None, words_override=None,
                  rows_override=None, coins_h=None, mutate_h=None, rho_override=None):
        """As the hashed runner: ``words_override(a, b)`` edits the words G computes on; ``rows_override(rows)`` edits the
        committed rows H hashes (negatives: the two sides then disagree); ``mutate`` / ``mutate_h`` edit the witness of G / H;
        ``rho_override(rho)`` edits the coins H's statement takes (a negative: H on other coins than G's challenge drew).
        ``coins`` / ``coins_h``: the verifier's step-0 coins of the two proofs (interactive mode; sampled here when None)."""
        from . import hashauth, hashchain
        from .chain import FpArgs
        if self.interactive:
            coins = protocol_mod.Coins.sample() if coins is None else coins
            coins_h = self._coins_h_for(coins, coins_h)
        vu_ids = list(range(len(vus))) if vu_ids is None else list(vu_ids)
        auth = self.auth_for(vu_ids) if auth is None else auth
        lay = self.layout(len(vus), min_l)
        live_sibs = None
        if hasattr(coins, "set_statement"):
            # a live verifier: the pair's v6 statement (multiproofs included) travels ahead of G's root; outside the clocks
            from .live import set_statement
            bd = self.marshal_shared(vus, vu_ids, min_l)
            live_sibs = hashauth.open_trees(self.trees, auth)
            set_statement(coins, self, (bd["is_end"], bd["idx"], bd["ypub"], auth, live_sibs, bd["table_h"], bd["units"]),
                          lay, self.cfg(lay.l), self.n_proofs)
        units_np, ux, uw = hashchain.shared_units(auth)
        n_units = len(units_np)
        lay_h = self.layout_h(n_units, lay.l)                # from the authentication block and G's l alone (the verifier's derivation)
        units, ypub = [], torch.zeros(lay.l, dtype=torch.int64)
        for v, (a, b, accs, y) in enumerate(vus):
            units += vu_units(self.rel, a, b, accs)
            ypub[self.steps * v + self.steps - 1] = self.rel.y_public(int(y))
        while len(units) < lay.l:
            units.append(self._pad_unit())
        a = np.asarray([u[1] for u in units], dtype=np.int64)
        b = np.asarray([u[2] for u in units], dtype=np.int64)
        c_in = np.asarray([u[0] for u in units], dtype=np.int64)
        y_out = np.asarray([u[3] for u in units], dtype=np.int64)
        if words_override is not None:
            a, b = words_override(a, b)
        # --- G: the GEMM side --------------------------------------------------------------------------------------------
        t0 = time.perf_counter()
        base = self.rel.hints(self.p, self._base_sys, c_in, a, b, y_out, self.device)
        extra = hashchain.shared_hints(self.hr, a, b, lay.l, self.device)
        hints = torch.cat([base, extra], 0)
        is_end = np.zeros((lay.l, 1), dtype=np.int64)
        is_end[np.arange(len(vus)) * self.steps + self.steps - 1, 0] = 1
        idx = auth.indices()
        pub = hashchain.shared_public_vectors(self.p, None, is_end, idx, None, device=self.device)
        if str(self.device).startswith("cuda"):
            torch.cuda.synchronize()
        hints_seconds = time.perf_counter() - t0
        proof_g = prove(self.p, self.sys, self.tb, self.cfg(lay.l), pub, hints, self.device, check=check, mutate=mutate,
                        chain=(lay, ypub), words={"a": is_end, "b": idx}, coins=coins, rel=self.hooks_for(auth))
        # --- H: the hash side, on the coins G's challenge 1 drew and the fingerprints G sent --------------------------------
        t0 = time.perf_counter()
        rows = self._unit_rows(units_np)
        if rows_override is not None:
            rows = rows_override(rows)
        n_slots_h = lay_h.l // self.steps
        F_h = self._fp_h(proof_g.fp.astype(np.int64), ux, uw, n_units, n_slots_h)
        if F_h is None:
            raise ValueError("G's fingerprints disagree between VUs of one shared row")
        hints_h, table_h, _ = hashchain.row_hash_hints(self.rh, rows, units_np[:, 0], n_units, lay_h.l, self.device,
                                                       sponges=self._sponges_for_units(units_np, rows))
        table_h_np = table_h.cpu().numpy()
        pub_h = hashchain.row_hash_public_vectors(self.p, None, table_h, units_np, None, device=self.device)
        rho = proof_g.rho.astype(np.int64) if rho_override is None else np.asarray(rho_override(proof_g.rho.astype(np.int64)), dtype=np.int64)
        fp_h = FpArgs(torch.as_tensor(rho), torch.as_tensor(F_h))
        ypub_h = torch.zeros(lay_h.l, dtype=torch.int64)
        if str(self.device).startswith("cuda"):
            torch.cuda.synchronize()
        hints_seconds += time.perf_counter() - t0
        proof_h = prove(self.p, self.sys_h, self.tb_h, self.cfg_h(lay_h.l), pub_h, hints_h, self.device, check=check, mutate=mutate_h,
                        chain=(lay_h, ypub_h, fp_h), words={"a": table_h_np, "b": units_np}, coins=coins_h, rel=self.hooks_h_for(auth))
        t1 = time.perf_counter()
        sibs = hashauth.open_trees(self.trees, auth) if live_sibs is None else live_sibs
        openings = time.perf_counter() - t1
        timings = {k: proof_g.timings.get(k, 0.0) + proof_h.timings.get(k, 0.0) for k in set(proof_g.timings) | set(proof_h.timings)}
        timings["auth_openings"] = openings
        timings["hints_host"] = hints_seconds
        timings["g_total"], timings["h_total"] = proof_g.timings["total"], proof_h.timings["total"]
        return SharedProof(proof_g, proof_h, timings, lay_h), (is_end, idx, ypub, auth, sibs, table_h_np, units_np), lay

    def verify_vus(self, proof, pubs, lay: Layout, coins=None, auth_timing: dict | None = None):
        from . import hashauth, hashchain
        from .chain import FpArgs
        is_end, idx, ypub, auth, sibs, table_h, units_given = pubs
        if coins is None and self.interactive:
            coins = proof.coins
        if coins is not None and not isinstance(coins, tuple):     # one Coins given: G's; H's from the transcript (local runs)
            coins = (coins, proof.h.coins)
        coins_g, coins_h = coins if coins is not None else (None, None)
        if not np.array_equal(np.asarray(idx, dtype=np.int64), auth.indices()) or auth.n_vus != lay.n_vus:
            return False, "authentication block does not match the statement's indices / layout"
        ends = np.arange(lay.n_vus) * lay.steps + lay.steps - 1
        e = np.asarray(is_end, dtype=np.int64)
        if e.shape != (lay.l, 1) or (e[ends, 0] != 1).any() or e[:, 0].sum() != lay.n_vus:
            return False, "public table: is_end off the chain ends"
        # (1) G on its own statement; this verifier's rho
        got: dict = {}
        ok, why = verify(self.p, self.sys, self.tb, self.cfg(lay.l), None, is_end, idx, None, proof.g, self.device,
                         chain=(lay, ypub), coins=coins_g, rel=self.hooks_for(auth), out=got)
        if not ok:
            return False, f"G: {why}"
        rho = got.get("rho")
        if rho is None or proof.g.fp is None:
            return False, "G: no fingerprint coins / message"
        # (2) the hash side's units from the authentication block; the VUs of one unit agree on F
        units_np, ux, uw = hashchain.shared_units(auth)
        n_units = len(units_np)
        if not np.array_equal(np.asarray(units_given, dtype=np.int64), units_np):
            return False, "hash side: units do not match the authentication block"
        lay_h = self.layout_h(n_units, lay.l)
        n_slots_h = lay_h.l // self.steps
        F_h = self._fp_h(proof.g.fp.astype(np.int64), ux, uw, n_units, n_slots_h)
        if F_h is None:
            return False, "two VUs claim one committed row with different fingerprints"
        # (3) H on the statement (table, units, rho, F_H)
        tab = np.asarray(table_h, dtype=np.int64)
        if tab.shape != (lay_h.l, 2 + hashchain.N_ROW_DIGEST):
            return False, "hash side: malformed public table"
        ends_h = np.arange(n_units) * lay_h.steps + lay_h.steps - 1
        if (tab[ends_h, 0] != 1).any() or tab[:, 0].sum() != n_units:
            return False, "hash side: is_end off the unit ends"
        roles = np.repeat(units_np[:, 0], lay_h.steps)
        if not np.array_equal(tab[:n_units * lay_h.steps, 1], roles) or tab[n_units * lay_h.steps:, 1:].any() or tab[n_units * lay_h.steps:, 0].any():
            return False, "hash side: roles / pad columns do not match the units"
        off_ends = np.setdiff1d(np.arange(n_units * lay_h.steps), ends_h)
        if tab[off_ends, 2:].any():
            return False, "hash side: digests off the unit ends"
        fp_h = FpArgs(torch.as_tensor(rho.astype(np.int64)), torch.as_tensor(F_h))
        ok, why = verify(self.p, self.sys_h, self.tb_h, self.cfg_h(lay_h.l), None, tab, units_np, None, proof.h, self.device,
                         chain=(lay_h, torch.zeros(lay_h.l, dtype=torch.int64), fp_h), coins=coins_h, rel=self.hooks_h_for(auth))
        if not ok:
            return False, f"H: {why}"
        # (4) the units' digests are the leaves at their ranks (per-VU expansion: what verify_hash_auth checks)
        t0 = time.perf_counter()
        dg = tab[ends_h, 2:]                                                    # (n_units, 8)
        digests = np.concatenate([dg[ux], dg[uw]], 1)                           # (n_vus, 16): x digest | W digest
        y_words = np.asarray(ypub, dtype=np.int64)[ends]
        ok, why = hashauth.verify_hash_auth(auth, sibs, digests=digests, y_words=y_words, y_bits=self.rel.y_bits)
        if auth_timing is not None:
            auth_timing["seconds"] = auth_timing.get("seconds", 0.0) + (time.perf_counter() - t0)
        return ok, why

    def soundness_report(self, cfg: Config, n_proofs: int, lays_h: list[Layout] | None = None) -> dict:
        """``protocol.soundness`` of the PAIR per sub-batch: G's terms + the fingerprint collision term, H's terms (one
        per hash-side layout in ``lays_h``, ``n_proofs`` of them), the union over ``2 n_proofs`` commitments (what
        :meth:`cfg` sized t for) -- the same sum :meth:`batch_bound` takes."""
        g = soundness(cfg, self.tb.L, self.tb.Q, 2 * n_proofs, fp=(self.fp_coords, self.steps, 2), chain_extras=self.chain_extras)
        if not lays_h:
            return g
        hs = [soundness(self.cfg_h(lh.l), self.tb_h.L, self.tb_h.Q, 2 * n_proofs, chain_extras=self.chain_extras_h) for lh in lays_h]
        per = [g["per_proof_total_log2"]] * n_proofs + [h["per_proof_total_log2"] for h in hs]
        return {**g, "h": hs[0], "h_layouts": [dataclasses.asdict(lh) for lh in lays_h],
                "total_log2": protocol_mod.log2_sum(per), "n_proofs": 2 * n_proofs}

    def batch_bound(self, cfgs: list[Config], n_proofs_claimed: list[int], target_bits: float = 128.0,
                    names: list[str] | None = None, lays_h: list[Layout] | None = None) -> dict:
        """The union over BOTH proofs of every sub-batch (each sized for ``2 n_proofs``); ``lays_h`` the hash-side layouts."""
        per, claimed, nm = [], [], []
        names = names or [f"sub_{i:02d}" for i in range(len(cfgs))]
        for i, (c, k) in enumerate(zip(cfgs, n_proofs_claimed)):
            per.append(soundness(c, self.tb.L, self.tb.Q, 1, fp=(self.fp_coords, self.steps, 2),
                                 chain_extras=self.chain_extras)["per_proof_total_log2"])
            claimed.append(2 * int(k))
            nm.append(names[i] + ".G")
            lh = lays_h[i] if lays_h is not None else self.layout_h(1)
            per.append(soundness(self.cfg_h(lh.l), self.tb_h.L, self.tb_h.Q, 1, chain_extras=self.chain_extras_h)["per_proof_total_log2"])
            claimed.append(2 * int(k))
            nm.append(names[i] + ".H")
        return protocol_mod.batch_bound(per, claimed, target_bits, nm)


# ---------------------------------------------------------------------------------------------------------------------
# Gate: honest VUs accepted, a negative battery rejected (on the device)
# ---------------------------------------------------------------------------------------------------------------------

def negatives(R: RelationChainRunner, base: list, row_negatives: int = 60) -> list[tuple[str, tuple[bool, str]]]:
    """The negative battery on two honest VUs: wrong claims, off-domain operands, another statement, broken chain links,
    a nonzero chain start, an accumulator bit, and one witness row per (gadget scope, row class)."""
    rel, steps = R.rel, R.steps
    negs = []
    proof, (a, b, ypub), lay = R.prove_vus(base, check=False)
    # (1) wrong claimed words: every bit of the public word of VU 0
    for bit in range(rel.y_bits):
        y_bad = ypub.clone()
        y_bad[steps - 1] ^= 1 << bit
        negs.append((f"claimed word bit {bit} flipped", R.verify_vus(proof, (a, b, y_bad), lay)))
    # (2) off by one unit of the public word (the neighbouring representable value)
    for delta in (1, -1):
        y_bad = ypub.clone()
        y_bad[steps - 1] = (int(ypub[steps - 1]) + delta) % 2**rel.y_bits
        negs.append((f"claimed word {'+' if delta > 0 else '-'}1 ulp", R.verify_vus(proof, (a, b, y_bad), lay)))
    # (3) an operand off the finite domain smuggled into the statement
    for j, w in enumerate(rel.off_domain):
        a_bad = a.copy()
        a_bad[3 + j, 5] = w
        negs.append((f"off-domain operand 0x{w:x} in a", R.verify_vus(proof, (a_bad, b, ypub), lay)))
    b_bad = b.copy()
    b_bad[7, 0] = rel.off_domain[-1]
    negs.append((f"off-domain operand 0x{rel.off_domain[-1]:x} in b", R.verify_vus(proof, (a, b_bad, ypub), lay)))
    # (4) an operand changed (the proof is of another statement)
    a_bad = a.copy()
    a_bad[10, 1] ^= 1 << (rel.params.frac if rel.params.word_bits == 16 else 3)   # an exponent bit
    negs.append(("operand a[10][1] exponent bit flipped", R.verify_vus(proof, (a_bad, b, ypub), lay)))
    # (4b) folded relations (``rel.fold`` instruction steps per column): the column boundary is where the fold could leak --
    # claim the intermediate word of the last column (the state after fold - 1 of its steps, which the base relation would
    # publish and the folded one keeps private), and swap the two operand words on either side of a column boundary while
    # keeping the original claim (a different chain, the same statement layout)
    if rel.fold > 1:
        # relation-level: the prover PROVES the bad VU (check=False), so the constraints -- not the statement digest --
        # have to reject it (the proof's transcript is consistent with the bad statement)
        from dataclasses import replace as _replace
        base_model = _replace(rel.model, groups=tuple(rel.model.groups)[: len(rel.model.groups) // rel.fold])
        kb, K = base_model.k, rel.k
        # the boundary VU: VU 0's words in its LAST column only (zeros before it), so the accumulator entering that column
        # is +0 and each of the column's fold steps moves the word -- a K = 1536 chain's accumulator has grown past the
        # adder's window by its last column and later products truncate away, which would make every negative below vacuous
        a_vu, b_vu = [0] * (K * (steps - 1)) + list(base[0][0][K * (steps - 1):]), [0] * (K * (steps - 1)) + list(base[0][1][K * (steps - 1):])
        accs, c_in = [], 0
        for s in range(steps):
            c_in = tc_dot(rel.model, c_in, a_vu[K * s:K * s + K], b_vu[K * s:K * s + K])
            accs.append(c_in)
        # NOTE: ``marshal`` caches sub-batch bundles by ``id()`` of the VU tuples, so every VU built here is kept alive in
        # ``keep`` until the battery is done -- a freed temporary's id can be reused and would replay the previous bundle
        keep = []

        def prove(vu):
            keep.append(vu)
            pf, pb, ly = R.prove_vus([vu, base[1]], check=False)
            return R.verify_vus(pf, pb, ly)
        ok, why = prove((a_vu, b_vu, accs, accs[-1]))
        negs.append(("folded boundary VU (honest control: must ACCEPT)", (not ok, "honest boundary VU accepted" if ok else why)))
        y_end, mid = rel.y_public(accs[-1]), accs[-2] if steps > 1 else 0
        for s in range(rel.fold - 1):                       # the state after 1 .. fold - 1 of the last column's steps: the
            lo = K * (steps - 1) + kb * s                   # words the base relation publishes and the fold keeps private
            mid = tc_dot(base_model, mid, a_vu[lo:lo + kb], b_vu[lo:lo + kb])
            if rel.y_public(mid) != y_end:
                negs.append((f"folded boundary: proved the {s + 1}-step intermediate of the last column as the final word",
                             prove((a_vu, b_vu, accs[:-1] + [mid], mid))))
        for nm, words in (("a", a_vu), ("b", b_vu)):
            for s in range(1, rel.fold):                    # across a base-step boundary inside the last column
                w2 = list(words)
                i = K * (steps - 1) + kb * s
                w2[i - 1], w2[i] = w2[i], w2[i - 1]
                vu2 = (w2, b_vu, accs, accs[-1]) if nm == "a" else (a_vu, w2, accs, accs[-1])
                if rel.y_public(tc_dot(rel.model, accs[-2] if steps > 1 else 0, vu2[0][K * (steps - 1):], vu2[1][K * (steps - 1):])) != y_end:
                    negs.append((f"folded boundary: {nm} words {i - 1} and {i} swapped across base step {s} of the last column, original claim",
                                 prove(vu2)))
                    break
        c2 = accs[-3] if steps > 2 else 0
        for d in range(1, kb + 1):                          # across the column boundary (a swap of two dead / zero products
            w2 = list(a_vu)                                 # changes nothing: take the first offset whose swap moves the word)
            i, j = K * (steps - 1) - d, K * (steps - 1) + d - 1
            w2[i], w2[j] = w2[j], w2[i]
            m2 = tc_dot(rel.model, c2, w2[K * (steps - 2):K * (steps - 1)], b_vu[K * (steps - 2):K * (steps - 1)])
            m3 = tc_dot(rel.model, m2, w2[K * (steps - 1):K * steps], b_vu[K * (steps - 1):K * steps])
            if rel.y_public(m3) != y_end:
                negs.append((f"folded boundary: a words {i} and {j} swapped across columns {steps - 2}|{steps - 1}, original claim",
                             prove((w2, b_vu, accs, accs[-1]))))
                break
    # (5) chain: broken link, nonzero c_0, accumulator bits of a linked word
    c_rows = R.sys.chain["c"]

    def break_link(W):
        W[c_rows[2], 20] = (W[c_rows[2], 20] + 1) % P
    pf, pb, ly = R.prove_vus(base, check=False, mutate=break_link)
    negs.append(("link c.f[20] != y.f[19]", R.verify_vus(pf, pb, ly)))

    def nonzero_c0(W):
        W[c_rows[1], steps] = 1
    pf, pb, ly = R.prove_vus(base, check=False, mutate=nonzero_c0)
    negs.append(("c_0 != 0 at a chain start", R.verify_vus(pf, pb, ly)))

    def acc_bit(W):
        W[c_rows[2], 30] = (W[c_rows[2], 30] ^ (1 << 12)) % P
    pf, pb, ly = R.prove_vus(base, check=False, mutate=acc_bit)
    negs.append(("accumulator fraction bit 12 flipped in a linked unit", R.verify_vus(pf, pb, ly)))

    def acc_low_bit(W):
        W[c_rows[2], 31] = (W[c_rows[2], 31] ^ 1) % P
    pf, pb, ly = R.prove_vus(base, check=False, mutate=acc_low_bit)
    negs.append(("accumulator fraction bit 0 flipped in a linked unit", R.verify_vus(pf, pb, ly)))
    # (6) one witness row per (gadget scope, row class), as gate_vu does
    seen, picks = set(), []
    for r in R.sys.rows:
        key = (".".join(r.name.split(".")[:2]), r.kind)
        if key not in seen:
            seen.add(key)
            picks.append(r)
    rng = np.random.default_rng(11)
    picks = picks[: max(0, row_negatives)]
    for r in picks:
        col = int(rng.integers(0, 2 * steps))
        delta = int(rng.integers(1, 3))

        def mutate(W, idx=r.idx, col=col, delta=delta):
            W[idx, col] = (W[idx, col] + delta) % P
        pf, pb, ly = R.prove_vus(base, check=False, mutate=mutate)
        negs.append((f"row {r.name} ({r.kind}) +{delta} in column {col}", R.verify_vus(pf, pb, ly)))
    return negs


def hashed_negatives(R: HashedRelationRunner, data: list, row_negatives: int = 40) -> list[tuple[str, tuple[bool, str]]]:
    """The committed-operand negatives (both verifiers must reject): a wrong digest published, digests swapped between VUs,
    a wrong path, a path to another root, a word of a committed row flipped (the prover computed on another row than the
    one it claims), the W column of another VU, the hashed word changed under the computation (a bit row of the sponge
    input), a broken sponge link, a nonzero sponge start, an index out of range, two VUs claiming one leaf with different
    digests; then one witness row per (scope, class) of the added gadget."""
    from . import hashchain
    from .auth import TreeRef
    rel, steps = R.rel, R.steps
    # the second VU is the first whose x row AND W column both differ from VU 0's (in a tile VUs 0 and 1 share the x row,
    # so "computed on VU 1's x row" would be an honest proof there)
    j = next((i for i in range(1, len(data)) if list(data[i][0]) != list(data[0][0]) and list(data[i][1]) != list(data[0][1])), 1)
    base = [data[0], data[j]]
    ids = [0, j]
    negs = []
    proof, (table, idx, ypub, auth, sibs), lay = R.prove_vus(base, check=False, vu_ids=ids)
    ends = [steps - 1, 2 * steps - 1]

    def V(pubs, name):
        negs.append((name, R.verify_vus(proof, pubs, lay)))
    t = table.copy(); t[ends[0], 1] ^= 1
    V((t, idx, ypub, auth, sibs), "x digest lane 0 of VU 0 flipped")
    de = R.leaf.digest_elems
    t = table.copy(); t[ends[1], 1 + de] = (t[ends[1], 1 + de] + 1) % P
    V((t, idx, ypub, auth, sibs), "W digest lane 0 of VU 1 changed")
    t = table.copy(); t[ends[0], 1:], t[ends[1], 1:] = table[ends[1], 1:].copy(), table[ends[0], 1:].copy()
    V((t, idx, ypub, auth, sibs), "digests of VU 0 and VU 1 swapped")
    t = table.copy(); t[ends[0], 0] = 0; t[ends[0] - 1, 0] = 1
    V((t, idx, ypub, auth, sibs), "is_end moved one column early")
    for ti, tn in enumerate(("a", "b", "y")):
        if sibs[ti]:
            s2 = [list(s) for s in sibs]
            s2[ti][0] = bytes([s2[ti][0][0] ^ 1]) + s2[ti][0][1:]
            V((table, idx, ypub, auth, s2), f"multiproof sibling of tree {tn} corrupted")
            s2 = [list(s) for s in sibs]
            s2[ti] = s2[ti][:-1]
            V((table, idx, ypub, auth, s2), f"multiproof of tree {tn} truncated")
    ra, rb, ry = auth.trees
    other = dataclasses.replace(auth, trees=(TreeRef("a", ra.binding, ra.owner, ra.count, rb.root), rb, ry))
    V((table, idx, ypub, other, sibs), "x root replaced by the W root (verifier's statement)")
    other = dataclasses.replace(auth, trees=(TreeRef("a", rb.binding, ra.owner, ra.count, ra.root), rb, ry))
    V((table, idx, ypub, other, sibs), "x tree binding changed (verifier's statement)")
    idx2 = idx.copy(); idx2[0, 1] = auth.x_index[1]
    other = dataclasses.replace(auth, x_index=(auth.x_index[1], auth.x_index[1]))
    V((table, idx2, ypub, other, sibs), "VU 0 claims VU 1's x row (two VUs, one leaf, different digests)")
    other = dataclasses.replace(auth, w_index=(auth.w_index[0], ra.count))
    idx2 = idx.copy(); idx2[1, 2] = ra.count
    V((table, idx2, ypub, other, sibs), "W leaf index out of range")
    V((table, idx, ypub, auth, sibs[:2]), "a multiproof missing")
    # the prover computed on other rows than the committed ones (a self-consistent proof of the wrong statement)
    def recomputed(a, b):
        k, accs, c = rel.k, [], 0
        for s in range(rel.steps):
            c = tc_dot(rel.model, c, a[k * s:k * s + k], b[k * s:k * s + k])
            accs.append(c)
        return (a, b, accs, accs[-1])
    a0, b0 = list(base[0][0]), list(base[0][1])
    a_flip = list(a0); a_flip[5] ^= 1
    pf, pb, ly = R.prove_vus([recomputed(a_flip, b0), base[1]], check=False, vu_ids=ids)
    negs.append(("one word of the committed x row flipped (computed on the flipped row)", R.verify_vus(pf, pb, ly)))
    pf, pb, ly = R.prove_vus([recomputed(a0, list(base[1][1])), base[1]], check=False, vu_ids=ids)
    negs.append(("VU 0 computed on VU 1's W column, claims its own", R.verify_vus(pf, pb, ly)))
    pf, pb, ly = R.prove_vus([recomputed(list(base[1][0]), b0), base[1]], check=False, vu_ids=ids)
    negs.append(("VU 0 computed on VU 1's x row, claims its own", R.verify_vus(pf, pb, ly)))
    # witness-level: the hashed word differs from the computed word; sponge links
    rows = {r.name: r.idx for r in R.sys.rows}
    hb0 = rows["hash.b[0].b0"]
    def flip_hashed_bit(W):
        W[hb0, 3] = (W[hb0, 3] + 1) % P
    pf, pb, ly = R.prove_vus(base, check=False, mutate=flip_hashed_bit, vu_ids=ids)
    negs.append(("bit 0 of hashed word b[0] flipped in unit 3 (computation unchanged)", R.verify_vus(pf, pb, ly)))
    cap_rows = R.sys.chain["c"][3:]                 # the leaf chains' carried rows (after the 3 accumulator links)
    ce = R.leaf.carry_elems
    def break_cap(W):
        W[cap_rows[min(5, ce - 1)], 7] = (W[cap_rows[min(5, ce - 1)], 7] + 1) % P
    pf, pb, ly = R.prove_vus(base, check=False, mutate=break_cap, vu_ids=ids)
    negs.append(("leaf carry link broken at unit 7", R.verify_vus(pf, pb, ly)))
    def cap_start(W):
        W[cap_rows[ce + min(4, ce - 1)], steps] = 5
    pf, pb, ly = R.prove_vus(base, check=False, mutate=cap_start, vu_ids=ids)
    negs.append(("nonzero leaf carry at a chain start", R.verify_vus(pf, pb, ly)))
    # one witness row per (scope, class) of the added rows
    seen, picks = set(), []
    for r in R.sys.rows[R.hr.census["rows_base"]:]:
        key = (".".join(r.name.split(".")[:3]), r.kind)
        if key not in seen:
            seen.add(key)
            picks.append(r)
    rng = np.random.default_rng(13)
    for r in picks[:max(0, row_negatives)]:
        col = int(rng.integers(0, 2 * steps))
        delta = int(rng.integers(1, 3))
        def mutate(W, idx_=r.idx, col=col, delta=delta, kind=r.kind):
            if kind == "inv":
                # t * inv = nz leaves inv FREE where t = 0 (inv = 0 there): mutate a column where it is constrained
                nz = (W[idx_, :2 * steps] != 0).nonzero().flatten()
                col = int(nz[col % len(nz)]) if len(nz) else col
            W[idx_, col] = (W[idx_, col] + delta) % P
        pf, pb, ly = R.prove_vus(base, check=False, mutate=mutate, vu_ids=ids)
        negs.append((f"row {r.name} ({r.kind}) +{delta} in column {col}", R.verify_vus(pf, pb, ly)))
    return negs


def shared_negatives(R: SharedHashedRunner, data: list, row_negatives: int = 40) -> list[tuple[str, tuple[bool, str]]]:
    """The row-sharing negatives (§3.2 of the leaf-campaign brief): a VU computing on a row other than the one it claims
    (alone on its unit, and sharing the unit with an honest VU), the fingerprint message edited / swapped, two VUs claiming
    one leaf with different rows, the hash side hashing another row, its digest / role / is_end / units edited, its
    witness broken (a hashed bit, a capacity link, a nonzero capacity start), the multiproofs and roots as for the hashed
    runner, a lane row of G edited; then one witness row per (scope, class) of G's added gadget."""
    from .auth import TreeRef
    rel, steps = R.rel, R.steps
    j = next((i for i in range(1, len(data)) if list(data[i][0]) != list(data[0][0]) and list(data[i][1]) != list(data[0][1])), 1)
    base = [data[0], data[j]]
    ids = [0, j]
    negs = []
    proof, pubs, lay = R.prove_vus(base, check=False, vu_ids=ids)
    is_end, idx, ypub, auth, sibs, table_h, units = pubs
    units_np, ux, uw = __import__("archive.direct.ligero.hashchain", fromlist=["shared_units"]).shared_units(auth)
    ends_h = np.arange(len(units_np)) * steps + steps - 1

    def V(p, name):
        negs.append((name, R.verify_vus(p[0], p[1], lay)))
    def with_h(t=None, u=None, s=None, a=None, e=None, i=None):
        return (proof, (is_end if e is None else e, idx if i is None else i, ypub, auth if a is None else a, sibs if s is None else s,
                        table_h if t is None else t, units if u is None else u))
    t = table_h.copy(); t[ends_h[ux[0]], 2] ^= 1
    V(with_h(t=t), "x digest lane 0 of VU 0's unit flipped")
    t = table_h.copy(); t[ends_h[uw[1]], 9] = (t[ends_h[uw[1]], 9] + 1) % P
    V(with_h(t=t), "W digest lane 7 of VU 1's unit changed")
    t = table_h.copy(); t[ends_h[ux[0]], 2:], t[ends_h[uw[0]], 2:] = table_h[ends_h[uw[0]], 2:].copy(), table_h[ends_h[ux[0]], 2:].copy()
    V(with_h(t=t), "digests of the x unit and the W unit of VU 0 swapped")
    t = table_h.copy(); t[ends_h[0], 0] = 0; t[ends_h[0] - 1, 0] = 1
    V(with_h(t=t), "hash side: is_end moved one column early")
    t = table_h.copy(); t[:steps, 1] ^= 1
    V(with_h(t=t), "hash side: role of unit 0 flipped (x row hashed with the W IV)")
    u2 = units.copy(); u2[0, 1] += 1
    V(with_h(u=u2), "hash side: unit 0's leaf rank changed in the statement")
    e2 = is_end.copy(); e2[steps - 1, 0] = 0; e2[steps - 2, 0] = 1
    V(with_h(e=e2), "GEMM side: is_end moved one column early")
    # the fingerprint message
    g = dataclasses.replace(proof.g, fp=proof.g.fp.copy()); g.fp[0, 0, 0] ^= 1
    V((dataclasses.replace(proof, g=g), pubs), "fingerprint F[VU 0][x][0] flipped in the message")
    g = dataclasses.replace(proof.g, fp=proof.g.fp.copy()); g.fp[0, 0], g.fp[1, 0] = proof.g.fp[1, 0].copy(), proof.g.fp[0, 0].copy()
    V((dataclasses.replace(proof, g=g), pubs), "x fingerprints of VU 0 and VU 1 swapped in the message")
    g = dataclasses.replace(proof.g, fp=proof.g.fp[:1].copy())
    V((dataclasses.replace(proof, g=g), pubs), "fingerprint message truncated to one VU")
    g = dataclasses.replace(proof.g, fp=None)
    V((dataclasses.replace(proof, g=g), pubs), "fingerprint message missing")
    for ti, tn in enumerate(("a", "b", "y")):
        if sibs[ti]:
            s2 = [list(s) for s in sibs]
            s2[ti][0] = bytes([s2[ti][0][0] ^ 1]) + s2[ti][0][1:]
            V(with_h(s=s2), f"multiproof sibling of tree {tn} corrupted")
            s2 = [list(s) for s in sibs]
            s2[ti] = s2[ti][:-1]
            V(with_h(s=s2), f"multiproof of tree {tn} truncated")
    ra, rb, ry = auth.trees
    other = dataclasses.replace(auth, trees=(TreeRef("a", ra.binding, ra.owner, ra.count, rb.root), rb, ry))
    V(with_h(a=other), "x root replaced by the W root (verifier's statement)")
    other = dataclasses.replace(auth, trees=(TreeRef("a", rb.binding, ra.owner, ra.count, ra.root), rb, ry))
    V(with_h(a=other), "x tree binding changed (verifier's statement)")
    other = dataclasses.replace(auth, w_index=(auth.w_index[0], ra.count))
    idx2 = idx.copy(); idx2[1, 2] = ra.count
    V(with_h(a=other, i=idx2), "W leaf index out of range")
    V(with_h(s=sibs[:2]), "a multiproof missing")
    # two VUs claim one leaf with different rows: G's fingerprints for that unit disagree
    other = dataclasses.replace(auth, x_index=(auth.x_index[1], auth.x_index[1]))
    try:
        pf, pb, ly = R.prove_vus(base, check=False, vu_ids=ids, auth=other)
        negs.append(("VU 0 claims VU 1's x row (two VUs, one leaf, different rows)", R.verify_vus(pf, pb, ly)))
    except ValueError as e:
        negs.append(("VU 0 claims VU 1's x row (two VUs, one leaf, different rows)", (False, f"prover: {e}")))
    # the prover computed on other rows than the committed ones
    def recomputed(a, b):
        k, accs, c = rel.k, [], 0
        for s in range(rel.steps):
            c = tc_dot(rel.model, c, a[k * s:k * s + k], b[k * s:k * s + k])
            accs.append(c)
        return (a, b, accs, accs[-1])
    a0, b0 = list(base[0][0]), list(base[0][1])
    a_flip = list(a0); a_flip[5] ^= 1
    pf, pb, ly = R.prove_vus([recomputed(a_flip, b0), base[1]], check=False, vu_ids=ids)
    negs.append(("one word of the committed x row flipped (computed on the flipped row, alone on its unit)", R.verify_vus(pf, pb, ly)))
    pf, pb, ly = R.prove_vus([recomputed(a0, list(base[1][1])), base[1]], check=False, vu_ids=ids)
    negs.append(("VU 0 computed on VU 1's W column, claims its own", R.verify_vus(pf, pb, ly)))
    pf, pb, ly = R.prove_vus([recomputed(list(base[1][0]), b0), base[1]], check=False, vu_ids=ids)
    negs.append(("VU 0 computed on VU 1's x row, claims its own", R.verify_vus(pf, pb, ly)))
    # a shared unit: two VUs on one x row (if the set has them), one computing on a flipped copy
    sh = next((i for i in range(1, len(data)) if list(data[i][0]) == list(data[0][0]) and i != 0), None)
    if sh is not None:
        try:
            pf, pb, ly = R.prove_vus([recomputed(a_flip, b0), data[sh]], check=False, vu_ids=[0, sh])
            negs.append(("shared x row: VU 0 computed on a flipped copy, VU sh honest", R.verify_vus(pf, pb, ly)))
        except ValueError as e:
            negs.append(("shared x row: VU 0 computed on a flipped copy, VU sh honest", (False, f"prover: {e}")))
    def rows_flip(rows):
        rows = rows.copy(); rows[0, 7] ^= 1
        return rows
    pf, pb, ly = R.prove_vus(base, check=False, vu_ids=ids, rows_override=rows_flip)
    negs.append(("hash side hashes a flipped copy of unit 0's row", R.verify_vus(pf, pb, ly)))
    pf, pb, ly = R.prove_vus(base, check=False, vu_ids=ids, rho_override=lambda rho: (rho + 1) % P)
    negs.append(("hash side proved on other coins than G's challenge drew (rho reused / replaced)", R.verify_vus(pf, pb, ly)))
    pf, pb, ly = R.prove_vus(base, check=False, vu_ids=ids, rho_override=lambda rho: rho * 0 + 1)
    negs.append(("hash side proved on rho = 1 (a degenerate coin)", R.verify_vus(pf, pb, ly)))
    # witness-level: G's lane row, H's hashed bit, H's capacity links
    lane0 = R.sys.chain["fp"]["lanes"][0][0]
    def flip_lane(W):
        W[lane0, 3] = (W[lane0, 3] + 1) % P
    pf, pb, ly = R.prove_vus(base, check=False, mutate=flip_lane, vu_ids=ids)
    negs.append(("GEMM side: lane row a.lane[0] +1 in unit 3", R.verify_vus(pf, pb, ly)))
    rows_h = {r.name: r.idx for r in R.sys_h.rows}
    hb0 = rows_h["hash.r[0].b0"]
    def flip_hashed_bit(W):
        W[hb0, 3] = (W[hb0, 3] + 1) % P
    pf, pb, ly = R.prove_vus(base, check=False, mutate_h=flip_hashed_bit, vu_ids=ids)
    negs.append(("hash side: bit 0 of hashed word r[0] flipped in unit 3", R.verify_vus(pf, pb, ly)))
    cap_rows = R.sys_h.chain["c"]
    def break_cap(W):
        W[cap_rows[5], 7] = (W[cap_rows[5], 7] + 1) % P
    pf, pb, ly = R.prove_vus(base, check=False, mutate_h=break_cap, vu_ids=ids)
    negs.append(("hash side: sponge capacity link broken at column 7", R.verify_vus(pf, pb, ly)))
    def cap_start(W):
        W[cap_rows[2], steps] = 5
    pf, pb, ly = R.prove_vus(base, check=False, mutate_h=cap_start, vu_ids=ids)
    negs.append(("hash side: nonzero sponge capacity at a chain start", R.verify_vus(pf, pb, ly)))
    # one witness row per (scope, class) of G's added rows
    seen, picks = set(), []
    for r in R.sys.rows[R.hr.census["rows_base"]:]:
        key = (".".join(r.name.split(".")[:3]), r.kind)
        if key not in seen:
            seen.add(key)
            picks.append(r)
    rng = np.random.default_rng(13)
    for r in picks[:max(0, row_negatives)]:
        col = int(rng.integers(0, 2 * steps))
        delta = int(rng.integers(1, 3))
        def mutate(W, idx_=r.idx, col=col, delta=delta, kind=r.kind):
            if kind == "inv":
                nz = (W[idx_, :2 * steps] != 0).nonzero().flatten()
                col = int(nz[col % len(nz)]) if len(nz) else col
            W[idx_, col] = (W[idx_, col] + delta) % P
        pf, pb, ly = R.prove_vus(base, check=False, mutate=mutate, vu_ids=ids)
        negs.append((f"row {r.name} ({r.kind}) +{delta} in column {col}", R.verify_vus(pf, pb, ly)))
    return negs


def gate_hashed(args, rel: Relation) -> int:
    """``gate-vu --auth included-hash`` / ``--auth included-hash-shared``: honest sub-batches of the hashed relation accepted
    (worst case, or the tile of ``--tile``), the relation's own negative families still rejected (claimed word, off-domain
    word now caught by the private decode, broken accumulator link, accumulator bits) and the committed-operand negatives
    of ``hashed_negatives`` / ``shared_negatives``."""
    R = make_runner(args, rel)
    shared = isinstance(R, SharedHashedRunner)
    log = print
    log(f"{R.relation} hashed chain system ({R.leaf.name} leaves): {R.sys.summary()} compiled in {R.compile_seconds:.2f}s; "
        f"census {R.hr.census}; device {args.device}")
    if shared:
        log(f"{R.relation} hash side (row units): {R.sys_h.summary()}; census {R.rh.census}")
    report = {"relation": R.relation, "model": rel.model_name, "authentication": R.AUTH, "leaf": R.leaf.name, "system": R.sys.summary(),
              "census": census(R.sys), "hash_census": R.hr.census, "positives": [], "negatives": []}
    if shared:
        report["hash_side"] = {"system": R.sys_h.summary(), "census": R.rh.census}
    failures = 0
    per_proof = args.batch // R.steps
    tile = parse_tile(getattr(args, "tile", None))
    procs = getattr(args, "instance_procs", 1)
    cache = getattr(args, "instances_cache", None)
    root = getattr(args, "root", None)
    if tile is None:
        data = instances(rel, args.vus, procs=procs, cache=cache, root=root)
        R.commit_vus(data, manifest_sha256=instances_digest(rel, args.vus, root=root))
    else:
        # no frozen set has a tile layout: the tile is its own synthetic draw (tile_digest), for every relation
        data, x_rows, w_cols = tile_instances(rel, *tile, procs=procs, cache=cache)
        data = data[:args.vus]
        R.commit_vus(data, manifest_sha256=tile_digest(rel, *tile), tile=tile, x_rows=x_rows, w_cols=w_cols)
    for lo in range(0, len(data), per_proof):
        vus = data[lo:lo + per_proof]
        proof, pubs, lay = R.prove_vus(vus, check=True, vu_ids=range(lo, lo + len(vus)))
        ok, why = R.verify_vus(proof, pubs, lay)
        sib_bytes = 32 * sum(len(s) for s in pubs[4])
        extra = f" (G {proof.timings['g_total']:.2f}s + H {proof.timings['h_total']:.2f}s, H layout {proof.lay_h})" if shared else ""
        log(f"{rel.name} hashed vu-k1536 [{lo}, {lo + len(vus)}): l={lay.l} prover {proof.timings['total']:.2f}s{extra} "
            f"(+hints {proof.timings['hints_host']:.2f}s) multiproofs {sib_bytes} B -> {ok} {why}")
        report["positives"].append({"range": [lo, lo + len(vus)], "l": lay.l, "ok": ok, "why": why, "timings": proof.timings,
                                    "multiproof_bytes": sib_bytes, "proof_bytes": proof.n_bytes()})
        failures += not ok
    # the relation's own negatives on the hashed system: claimed word, links, accumulator bits (the operand words are
    # private now: the off-domain family becomes "a word off the finite domain is proved" -> rejected by the decode)
    base = data[:2]
    proof, pubs, lay = R.prove_vus(base, check=False, vu_ids=[0, 1])
    ypub = pubs[2]
    negs = []
    for bit in (0, rel.y_bits - 1):
        y_bad = ypub.clone(); y_bad[R.steps - 1] ^= 1 << bit
        negs.append((f"claimed word bit {bit} flipped", R.verify_vus(proof, pubs[:2] + (y_bad,) + pubs[3:], lay)))
    c_rows = R.sys.chain["c"]
    def break_link(W):
        W[c_rows[2], 20] = (W[c_rows[2], 20] + 1) % P
    pf, pb, ly = R.prove_vus(base, check=False, mutate=break_link, vu_ids=[0, 1])
    negs.append(("accumulator link c.f[20] != y.f[19]", R.verify_vus(pf, pb, ly)))
    def acc_bit(W):
        W[c_rows[2], 30] = (W[c_rows[2], 30] ^ (1 << 12)) % P
    pf, pb, ly = R.prove_vus(base, check=False, mutate=acc_bit, vu_ids=[0, 1])
    negs.append(("accumulator fraction bit 12 flipped in a linked unit", R.verify_vus(pf, pb, ly)))
    off = list(base[0][0]); off[getattr(rel, "off_domain_index", 4)] = rel.off_domain[0]
    k, accs, c = rel.k, [], 0
    for s in range(rel.steps):     # the model on an off-domain word: whatever it records, the private decode must reject the word
        try:
            c = tc_dot(rel.model, c, off[k * s:k * s + k], base[0][1][k * s:k * s + k])
        except Exception:
            c = 0
        accs.append(c)
    try:
        pf, pb, ly = R.prove_vus([(off, base[0][1], accs, accs[-1]), base[1]], check=False, vu_ids=[0, 1])
        negs.append((f"off-domain operand 0x{rel.off_domain[0]:x} computed on", R.verify_vus(pf, pb, ly)))
    except Exception as e:                                  # the hint generator itself refuses the word: also a rejection
        negs.append((f"off-domain operand 0x{rel.off_domain[0]:x} computed on", (False, f"prover: {type(e).__name__}: {e}")))
    if getattr(rel, "extra_negatives", None) is not None:     # the relation's own family (fp4/hashed.py: malformed encodings, a wrong scale)
        negs += rel.extra_negatives(R, base)
    negs += (shared_negatives if shared else hashed_negatives)(R, data, getattr(args, "row_negatives", 40))
    if hasattr(R.leaf, "step"):
        negs += leaf_negatives(R, data)                      # the linear leaf's own battery (brief §3.4)
    for name, (ok, why) in negs:
        log(f"  negative {name}: {'REJECTED' if not ok else 'ACCEPTED (BUG)'} {why}")
        report["negatives"].append({"name": name, "accepted": ok, "why": why})
        failures += ok
    report["failures"] = failures
    log(f"{rel.name} hashed gate: {len(report['positives'])} honest sub-batches, {len(negs)} negatives, {failures} failures")
    if args.out:
        Path(os.path.expandvars(args.out)).write_text(json.dumps(report, indent=1, default=str))
    return 2 if failures else 0


#: what a scheme without a `privacy_note` (the contract's optional attribute; red-team-leaf F5) is assumed to be: a
#: deterministic unsalted compressing hash (Poseidon2, BLAKE3)
DEFAULT_PRIVACY_NOTE = "deterministic, unsalted: hides only up to preimage search on a row"


def leaf_privacy_note(leaf) -> str:
    """``LeafScheme.privacy_note`` of the scheme, or :data:`DEFAULT_PRIVACY_NOTE`."""
    return getattr(leaf, "privacy_note", None) or DEFAULT_PRIVACY_NOTE


def leaf_negatives(R: HashedRelationRunner, data: list) -> list[tuple[str, tuple[bool, str]]]:
    """The brief's three negatives of a LINEAR ``leaf/`` digest (§3.4, lane ajtai-leaf; a scheme exposing ``step(acc, bits)``),
    each a self-consistent proof of the wrong thing: the digest of ANOTHER committed row published (the tree opens it, the
    relation's digest of the row computed on does not match); one operand bit of the hashed row flipped in the witness
    (every bit row is a committed wire; the accumulator rows are recomputed so the digest changes and no longer opens
    under the root); a non-binary ``s``: a bit row set to 2 with the accumulator recomputed consistently (the booleanity
    constraint, not the digest, must reject)."""
    from . import hashchain
    leaf, steps, k, nb = R.leaf, R.steps, R.hr.k, R.hr.word_bits
    j = next((i for i in range(1, len(data)) if list(data[i][0]) != list(data[0][0]) and list(data[i][1]) != list(data[0][1])), 1)
    base, ids = [data[0], data[j]], [0, j]
    negs = []
    n = leaf.digest_elems
    proof, (table, idx, ypub, auth, sibs), lay = R.prove_vus(base, check=False, vu_ids=ids)
    # 1. the digest of a different row: VU 0 publishes VU j's x digest (the tree opens VU j's leaf only under VU j's rank)
    t = table.copy()
    t[steps - 1, 1:1 + n] = table[2 * steps - 1, 1:1 + n]
    negs.append(("digest of a different row published for VU 0's x row", R.verify_vus(proof, (t, idx, ypub, auth, sibs), lay)))
    # ... and claiming VU j's rank for it too (then the multiproof opens, but the relation's digest is of VU 0's bits)
    idx2 = idx.copy(); idx2[0, 1] = auth.x_index[1]
    other = dataclasses.replace(auth, x_index=(auth.x_index[1], auth.x_index[1]))
    negs.append(("digest of a different row, its rank claimed (relation digest != published)",
                 R.verify_vus(proof, (t, idx2, ypub, other, sibs), lay)))
    rows = {r.name: r.idx for r in R.sys.rows}
    acc_rows = {nm: [rows[f"hash.{nm}.cap[{x}]"] for x in range(n)] for _, nm, _ in hashchain.ROLES}
    out_rows = {nm: [rows[f"hash.{leaf._out_name(role, x)}"] for x in range(n)] for _, nm, role in hashchain.ROLES}
    bit_rows = {nm: [rows[f"hash.{nm}[{i}].b{j_}"] for i in range(k) for j_ in range(nb)] for _, nm, _ in hashchain.ROLES}
    role_of = {nm: role for _, nm, role in hashchain.ROLES}

    def recompute(W, nm: str, col0: int):
        """Recompute the accumulator rows of the VU holding column ``col0`` from its (mutated) bit rows: a prover that keeps
        its own state consistent, so only the digest pin / booleanity can catch it."""
        v0 = (col0 // steps) * steps
        bits = W[bit_rows[nm]][:, v0:v0 + steps].T.to(torch.int64)             # (steps, m)
        acc = torch.zeros(n, dtype=torch.int64, device=W.device)
        for s in range(steps):
            W[acc_rows[nm], v0 + s] = acc.to(W.dtype)                          # W is int32 on the device, int64 on the CPU
            acc = leaf.step(acc, bits[s], role_of[nm])
            W[out_rows[nm], v0 + s] = acc.to(W.dtype)

    # 2. one bit flipped in the hashed x row of VU 0 (column 2, word 3, bit 1), state recomputed
    def flip_bit(W):
        r = bit_rows["a"][3 * nb + 1]
        W[r, 2] = 1 - W[r, 2]
        recompute(W, "a", 2)
    pf, pb, ly = R.prove_vus(base, check=False, mutate=flip_bit, vu_ids=ids)
    negs.append(("one bit of the hashed x row flipped (digest recomputed consistently)", R.verify_vus(pf, pb, ly)))
    # 3. non-binary s: a bit row of the W column of VU j set to 2, state recomputed (booleanity must reject)
    def non_binary(W):
        r = bit_rows["b"][5 * nb]
        W[r, steps + 1] = 2
        recompute(W, "b", steps + 1)
    pf, pb, ly = R.prove_vus(base, check=False, mutate=non_binary, vu_ids=ids)
    negs.append(("non-binary operand bit s = 2 (digest recomputed consistently)", R.verify_vus(pf, pb, ly)))
    # 3b. non-binary s AND the digest pin honoured: bit 0 of one word set to 2 while bit 1 is cleared (same word value,
    # different bit vector -> a different digest unless caught): the decode constraint word = sum b_j 2^j still holds
    def non_binary_same_word(W):
        r0, r1 = bit_rows["b"][7 * nb], bit_rows["b"][7 * nb + 1]
        if W[r1, steps + 2] == 1 and W[r0, steps + 2] == 0:
            W[r0, steps + 2], W[r1, steps + 2] = 2, 0
        else:
            W[r0, steps + 2], W[r1, steps + 2] = (W[r0, steps + 2] + 2) % P, (W[r1, steps + 2] + P - 1) % P
        recompute(W, "b", steps + 2)
    pf, pb, ly = R.prove_vus(base, check=False, mutate=non_binary_same_word, vu_ids=ids)
    negs.append(("non-binary bit vector of the same word value (2*1 + 0*2 for 0*1 + 1*2)", R.verify_vus(pf, pb, ly)))
    # 4. the carried state tampered without recomputation: acc_out[3] + 1 at one column (the linear row must reject)
    def tamper_out(W):
        W[out_rows["a"][3], 1] = (W[out_rows["a"][3], 1] + 1) % P
    pf, pb, ly = R.prove_vus(base, check=False, mutate=tamper_out, vu_ids=ids)
    negs.append(("leaf accumulator output row tampered", R.verify_vus(pf, pb, ly)))
    return negs


def make_runner(args, rel: Relation, **kw):
    """The runner ``--auth`` selects: ``excluded`` (the bare v4 statement) or ``included-hash`` (``HashedRelationRunner``)."""
    auth = getattr(args, "auth", "excluded") or "excluded"
    if auth == "excluded":
        return RelationChainRunner(rel, args.device, args.target, args.rate_log2, zk=args.zk, mode=args.mode, **kw)
    if auth == HashedRelationRunner.AUTH:
        R = HashedRelationRunner(rel, args.device, args.target, args.rate_log2, zk=args.zk, mode=args.mode,
                                 leaf=getattr(args, "leaf", None), **kw)
        if getattr(R.leaf, "test_only", False):
            raise SystemExit(f"leaf scheme {R.leaf.name!r} is test-only (insecure): it cannot be benched")
        return R
    if auth == SharedHashedRunner.AUTH:
        return SharedHashedRunner(rel, args.device, args.target, args.rate_log2, zk=args.zk, mode=args.mode,
                                  leaf=getattr(args, "leaf", None), **kw)
    raise SystemExit(f"--auth {auth}: not a mode of the relation runner (excluded | included-hash | included-hash-shared)")


def gate_vu_rel(args, rel: Relation) -> int:
    if getattr(args, "auth", "excluded") in (HashedRelationRunner.AUTH, SharedHashedRunner.AUTH):
        return gate_hashed(args, rel)
    R = RelationChainRunner(rel, args.device, args.target, args.rate_log2, zk=args.zk, mode=args.mode)
    log = print
    log(f"{rel.name} chain system: {R.sys.summary()} compiled in {R.compile_seconds:.2f}s; device {args.device}")
    report = {"relation": rel.name, "model": rel.model_name, "system": R.sys.summary(), "census": census(R.sys),
              "positives": [], "negatives": []}
    failures = 0
    per_proof = args.batch // R.steps
    data = instances(rel, args.vus, procs=getattr(args, "instance_procs", 1), cache=getattr(args, "instances_cache", None),
                     root=getattr(args, "root", None))
    for lo in range(0, args.vus, per_proof):
        vus = data[lo:lo + per_proof]
        proof, pubs, lay = R.prove_vus(vus, check=True)
        ok, why = R.verify_vus(proof, pubs, lay)
        log(f"{rel.name} vu-k1536 [{lo}, {lo + len(vus)}): l={lay.l} prover {proof.timings['total']:.2f}s -> {ok} {why}")
        report["positives"].append({"range": [lo, lo + len(vus)], "l": lay.l, "ok": ok, "why": why, "timings": proof.timings})
        failures += not ok
    negs = negatives(R, data[:2], getattr(args, "row_negatives", 60))
    for name, (ok, why) in negs:
        log(f"  negative {name}: {'REJECTED' if not ok else 'ACCEPTED (BUG)'} {why}")
        report["negatives"].append({"name": name, "accepted": ok, "why": why})
        failures += ok
    report["failures"] = failures
    log(f"{rel.name} gate: {len(report['positives'])} honest sub-batches, {len(negs)} negatives, {failures} failures")
    if args.out:
        Path(os.path.expandvars(args.out)).write_text(json.dumps(report, indent=1, default=str))
    return 2 if failures else 0


# ---------------------------------------------------------------------------------------------------------------------
# Bench
# ---------------------------------------------------------------------------------------------------------------------

def bench_vu_rel(args, rel: Relation) -> int:
    """``vu.bench_vu`` for a registered relation: ``--batch`` units per proof, ``--total-vus`` VUs as ceil(total / B_proof)
    independently committed proofs (union-bounded soundness), every sub-batch of a rep proved first and then every one
    verified (the verifier's host work never sits between two timed prover sub-batches), the reported prover time the sum
    over the sub-batches.  ``--dump-dir`` writes the first ``--dump-reps`` recorded reps' statements (``ligero-statement/v4``:
    the relation is named in the file) + proofs (+ coins in interactive mode) and ``system.bin`` + ``manifest.json``, then
    re-reads and verifies every dumped file cold (``_verify_dumps``, the Python file verifier: a self-check); a dump inside
    the run directory is listed in ``artifacts`` so it travels with the Attempt as run-files (``vu.dump_artifacts``)."""
    from .run import _git_commit as _git_commit_repo
    from .vu import ZK_MODE, ZK_MODE_FS, ZK_MODE_HM96, ZK_VACUOUS_NOTE, dump_artifacts
    ZK_SHARED_NOTE = ("; NOTE: authentication = included-hash-shared: as included-hash, the rows hashed once per sub-batch on the "
                      "hash side; IN ADDITION the GEMM side's proof PUBLISHES the fingerprints F (red-team-leaf F8): C verifier-known "
                      "random linear functionals (C x ~31 bits) of EVERY private operand row it touches, x rows and W columns alike -- "
                      "a leak of the private weights beyond the digests, whatever the leaf; folding the hash side into the masked "
                      "witness (v2) removes it and is not built")

    def _git_commit():
        return os.environ.get("RESEARCH_GIT_COMMIT") or _git_commit_repo()

    steps = rel.steps
    per_proof = args.batch // steps
    total_vus = args.total_vus or per_proof
    n_proofs = -(-total_vus // per_proof)
    interactive = args.mode == protocol_mod.MODE_INTERACTIVE
    depth = 3 if interactive else 1
    R = make_runner(args, rel, n_proofs=n_proofs)
    hashed = isinstance(R, HashedRelationRunner)
    # red-team-leaf F5: the hiding caveat is the SCHEME's (`LeafScheme.privacy_note`), not Poseidon2's for everyone -- a linear
    # leaf (Ajtai) is binding-only and says so in the fingerprint
    ZK_HASHED_NOTE = ("; NOTE: authentication = included-hash: the operand rows x, W are NOT in this statement (private witness, "
                      f"hashed in-circuit); the statement publishes their {R.leaf.hash_name} row digests ({leaf_privacy_note(R.leaf)}), "
                      "the Merkle roots + (vu, x_index, w_index), and the words y16") if hashed else ""
    shared = isinstance(R, SharedHashedRunner)
    # --pipeline N (lane hp2-host): the staged prover; the hashed runners' stage generators (marshal_hashed / marshal_shared)
    pipeline_depth = int(getattr(args, "pipeline", 0) or 0)
    tile = parse_tile(getattr(args, "tile", None)) if hashed else None
    if tile is not None and tile[0] * tile[1] != total_vus:
        raise SystemExit(f"--tile {tile[0]}x{tile[1]} is {tile[0] * tile[1]} VUs; --total-vus {total_vus}")
    cen = census(R.sys)
    print(f"{rel.name} census:", json.dumps(cen))
    if hashed:
        print(f"{R.relation} hashed ({R.AUTH}, {R.leaf.name} leaves, sharing {'none' if tile is None else 'tile%dx%d' % tile}):",
              json.dumps(R.hr.census))
    if shared:
        print(f"{R.relation} hash side (row units): {R.sys_h.summary()}; census {json.dumps(R.rh.census)}")
    subs = [(lo, min(lo + per_proof, total_vus)) for lo in range(0, total_vus, per_proof)]
    root = getattr(args, "root", None)
    input_set = getattr(args, "input_set", None)
    if input_set and tile is not None:
        raise SystemExit("--input-set and --tile: an input set carries its own rows")
    set_inst = set_ref(input_set, total_vus, getattr(args, "input_set_art", None)) if input_set else None
    t0 = time.perf_counter()
    if tile is None:
        if input_set:
            data = set_instances(rel, input_set, total_vus)
        else:
            data = instances(rel, total_vus, procs=getattr(args, "instance_procs", 1), cache=getattr(args, "instances_cache", None), root=root)
        # the committed set as stored (the rows in their words' dtype): the committer's input, made once with the set
        x_rows, w_cols = R.committed_rows(data) if hashed else (None, None)
    else:
        # a frozen-tier relation (bf16-ampere): its frozen set has no tile layout, the tile is the synthetic draw
        data, x_rows, w_cols = tile_instances(rel, *tile, procs=getattr(args, "instance_procs", 1), cache=getattr(args, "instances_cache", None))
    inst_digest = (tile_digest(rel, *tile) if tile is not None else set_inst["manifest_sha256"] if set_inst
                   else instances_digest(rel, total_vus, root=root))
    inst_dataset, inst_tier = (set_inst["dataset"], set_inst["tier"]) if set_inst else (rel.instances_dataset, rel.instances_tier)
    kind = (f"input set {set_inst['tier']} ({set_inst['source']})" if set_inst else
            'frozen ' + rel.frozen_tier if rel.frozen_tier else 'synthetic')
    print(f"{total_vus} {kind} VUs (recorded accumulators) in {time.perf_counter() - t0:.1f}s")
    commit_seconds = 0.0
    # --commit-per-rep: every rep (the warm-up included) commits its batch from scratch (no tree cache), timed apart from
    # proving: the statement's commitments (trees + row digests) are the commitment bucket; the prover's precomputed row
    # chains go to the witness bucket and rebuilding the hashed statements to serialization, both inside t.total
    commit_per_rep = bool(getattr(args, "commit_per_rep", False))
    if commit_per_rep and not (hashed and not shared and pipeline_depth > 1 and args.device.startswith("cuda")):
        raise SystemExit("--commit-per-rep: --auth included-hash (unshared) with --pipeline >= 2 on CUDA only")
    if commit_per_rep and int(getattr(args, "commit_reps", 0) or 0):
        raise SystemExit("--commit-per-rep and --commit-reps are two timings of the commitment bucket: pick one")

    def commit_batch(cache):
        R.commit_vus(data, dataset=inst_dataset, tier=inst_tier, manifest_sha256=inst_digest,
                     cache=cache, tile=tile, x_rows=x_rows, w_cols=w_cols)

    commit_runs: list[tuple[float, dict]] = []
    commit_ev = None
    if hashed:
        # --commit-reps N: the whole commitment (row digests, the three trees, the prover's chain states) built from scratch
        # 1 + N times; the first call (cold) stays relation.hash.commit_seconds, commit.seconds is the median of the N
        # (the per-batch cost of a committer that has run before), every call's evidence must be identical
        for ci in range(1 + max(0, int(getattr(args, "commit_reps", 0) or 0))):
            t0 = time.perf_counter()
            commit_batch(None if commit_per_rep or ci > 0 else getattr(args, "auth_cache", None))
            commit_runs.append((time.perf_counter() - t0, dict(getattr(R, "commit_timings", {}))))
            ev = R.commit_evidence()
            assert commit_ev is None or ev["sha256"] == commit_ev["sha256"], "commitment differs between two builds of one set"
            commit_ev = ev
            print(f"commit {ci}: {commit_runs[-1][0]:.3f}s {json.dumps({k_: round(v_, 4) for k_, v_ in commit_runs[-1][1].items()})}")
        commit_seconds = commit_runs[0][0]
        print(f"committed: x tree {R.trees['a'].count} rows, W tree {R.trees['b'].count} columns, y tree {R.trees['y'].count} words "
              f"({R.leaf.schema} + SHA-256) in {commit_seconds:.1f}s (row digests on {args.device}: {R.sponge_seconds:.2f}s); "
              f"roots a={R.trees['a'].root.hex()[:16]} b={R.trees['b'].root.hex()[:16]}; evidence sha256 {commit_ev['sha256']}")
        if getattr(args, "commit_evidence", None):
            Path(os.path.expandvars(args.commit_evidence)).write_text(json.dumps(commit_ev, indent=1, sort_keys=True))
    lay = R.layout(per_proof)
    l = lay.l
    cfg = R.cfg(l)
    print(f"config {cfg}; layout {lay}; zk {cfg.zk}, mode {cfg.mode}; system {R.sys.summary()}; {total_vus} VUs as {n_proofs} sub-batches of <= {per_proof}")
    # --dump-dir: the first --dump-reps RECORDED reps' sub-batch proofs as ligero-statement/v4 + ligero-proof/v1 files (+ the
    # runner's step-0 coins, 128 raw bytes, interactive mode), system.bin + manifest.json once -- the layout vu.bench_vu
    # writes for BF16 (repK/sub_XX.{stmt,proof[,coins]}); written after the verifier timer, outside every measured interval.
    dump_dir = Path(os.path.expandvars(args.dump_dir)) if getattr(args, "dump_dir", None) else None
    dump_reps = getattr(args, "dump_reps", None)
    dump_reps = args.reps if dump_reps is None else max(0, min(int(dump_reps), args.reps))
    dump_manifest = None
    if dump_dir is not None:
        from .serialize import _manifest, _statement_of, _write_system, proof_bytes, sha256_file, statement_bytes
        dump_dir.mkdir(parents=True, exist_ok=True)
        dump_manifest = _manifest(dump_dir, args, R)
        dump_manifest["relation"] = {"name": rel.name, "model": rel.model_name, "profile": rel.target.name,
                                     "word_dtype": rel.word_dtype, "y_dtype": rel.y_dtype, "y_bits": rel.y_bits,
                                     "rust_verifiable": rel.rust_verifiable}
        if hashed:
            # the statement's relation string ("<rel>" for Poseidon2, "<rel>+<leaf>" otherwise) and the leaf scheme
            dump_manifest["relation"]["statement_relation"] = R.relation
            dump_manifest["relation"]["leaf"] = {"scheme": R.leaf.name, "schema": R.leaf.schema,
                                                 "params_sha256": R.leaf.params_sha256, "digest_elems": R.leaf.digest_elems}
        _write_system(dump_dir, R, dump_manifest)
        dump_manifest["set"] = {"total_vus": total_vus, "batch": args.batch, "per_proof": per_proof, "n_proofs": n_proofs,
                                "zk": args.zk, "mode": args.mode, "reps": args.reps, "dump_reps": dump_reps,
                                "instances": inst_digest, **({"input_set": set_inst} if set_inst else {})}
        if tile is not None:
            # what an independent verifier needs to recompute the committed tile (reverify.committed_trees): the draw is
            # tile_instances(rel, nx, nw, seed), VU v is x row v // nw against W column v % nw
            dump_manifest["set"]["tile"] = {"nx": tile[0], "nw": tile[1], "seed": rel.instance_seed,
                                            "draw": TILE_DRAW, "layout": TILE_LAYOUT}
    target_bits = float(getattr(args, "target_bits", 128.0) or 128.0)   # the VERIFIER's parameter (PROTOCOL.md 8c)
    # --verifier tcp://host:port: the step-0 coins of every RECORDED rep come from a LIVE verifier on another machine
    # (live.py) and every prover message is sent to it as it is produced -- the protocol is unchanged (8c), only the coin
    # source is; the warm-up rep keeps local coins (it presents one sub-batch, not n_proofs).
    live = None
    skip_local = bool(getattr(args, "no_local_verify", False))
    if skip_local and not getattr(args, "verifier", None):
        raise SystemExit("--no-local-verify needs --verifier (a live verifier's verdict in place of the in-process one)")
    if getattr(args, "verifier", None):
        from .live import BenchLive
        live = BenchLive(args, R, n_proofs, total_vus, rel.name, interactive)
        if pipeline_depth > 1 and args.device.startswith("cuda"):
            live.window = pipeline_depth                 # the verifier commits N sub-batches ahead (N openings in flight)
    batch = None
    runs = []
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    keys = None
    for rep in range(args.reps + 1):
        agg: dict[str, float] = {}
        vsum = 0.0
        if commit_per_rep:
            R.sponges = R.rows_d = None
            if rep > 0:
                torch.cuda.synchronize()
            t_c = time.perf_counter()
            commit_batch(None)
            agg["commitment"] = time.perf_counter() - t_c - R.sponge_seconds
            assert {n: t.ref.to_json() for n, t in R.trees.items()} == {
                n: {k: v for k, v in e.items() if k != "levels_sha256"} for n, e in commit_ev["trees"].items()}, \
                f"rep {rep}: the recommitted trees differ from the first build's"
            agg["hints_host"] = agg["row_chain"] = R.sponge_seconds
            t_s = time.perf_counter()
            for lo, hi in subs:
                R.marshal_hashed(data[lo:hi], range(lo, hi), l)
            agg["statement"] = agg["total"] = time.perf_counter() - t_s
        t_pass = time.perf_counter()
        # Prove every sub-batch of the rep first, then verify (and dump) them: the verifier's host work must not sit
        # between two timed prover sub-batches (vu.bench_vu since f28c13d; the same discipline here).
        proved = []
        cfgs_seen: list[Config] = []
        if live is not None:
            live.begin_rep(rep)
        if pipeline_depth > 1 and rep > 0 and args.device.startswith("cuda"):
            # --pipeline N: the sub-batches interleaved N-deep on their own streams (lane hp2-host); the timings are each
            # sub-batch's share of the pass wall, so their sum is the wall.  Coins are sampled per sub-batch as it starts.
            # With --verifier the factory hands out the LIVE verifier's coins in sub-batch order and the coin exchange is
            # non-blocking (lane live-pipeline): the network waits of sub-batch i overlap the device work of i +- 1, so
            # the prover's work hides under the wire time and t.total_live -> the session's network critical path (the
            # verifier opens the sub-batches strictly in order, 2 x RTT each, on one control connection).  PROOF i is
            # sent on the data channel after the pass.
            local_factory = lambda i: (protocol_mod.Coins.sample() if interactive else None)  # noqa: E731
            factory = live.pipeline_factory(local_factory) if live is not None else local_factory
            # with a live verifier each proof goes out on the data connections as soon as its sub-batch is done
            stream = {"on_done": live.stream} if live is not None and not shared else {}
            many, _wall = R.prove_vus_many([data[lo:hi] for lo, hi in subs], factory, depth=pipeline_depth, min_l=l,
                                           vu_ids_list=[range(lo, hi) for lo, hi in subs], **stream)
            for si, ((lo, hi), (V, proof, pubs, lay_i)) in enumerate(zip(subs, many)):
                if live is not None:
                    V = live.proved(si, proof, V)            # PROOF si on the data channel (async); coins -> plain Coins
                for k_, v_ in proof.timings.items():
                    agg[k_] = agg.get(k_, 0.0) + v_
                proved.append((si, lo, hi, V, proof, pubs, lay_i))
                cfgs_seen.append(R.cfg(lay_i.l))
        for si, (lo, hi) in enumerate(subs if not proved else []):
            V = protocol_mod.Coins.sample() if interactive else None
            if live is not None:
                V = live.coins(si, V)                        # a recorded rep: the live verifier's step 0 for sub-batch si
            proof, pubs, lay_i = R.prove_vus(data[lo:hi], check=(rep == 0 and lo == 0), min_l=l, coins=V, vu_ids=range(lo, hi))
            if live is not None:
                V = live.proved(si, proof, V)                # PROOF si on the data channel (async); waits out of the clocks
            for k_, v_ in proof.timings.items():
                agg[k_] = agg.get(k_, 0.0) + v_
            proved.append((si, lo, hi, V, proof, pubs, lay_i))
            cfgs_seen.append(R.cfg(lay_i.l))
            if rep == 0:
                if pipeline_depth > 1 and args.device.startswith("cuda"):
                    # a full pipelined pass, untimed, like the warm-up: the graphs are captured per (stream, layout) and
                    # sub-batch i runs on slot i % depth (pipeline.FIXED_SLOTS), so the ragged last sub-batch must be met
                    # on ITS slot -- warming subs[:depth] + subs[-1:] captured it on slot 0 and timed rep 1 paid ~0.25 s
                    warm = list(subs)
                    passes = 1
                    if shared:
                        # the hash side's layout varies per sub-batch (70 / 71 row units at the 64x64 tile) and the tests
                        # graph is captured per (stream, layout): every slot must meet every layout before a timed pass
                        warm, passes = list(subs), 3
                    for _ in range(passes):
                        R.prove_vus_many([data[lo_:hi_] for lo_, hi_ in warm],
                                         (lambda i: protocol_mod.Coins.sample() if interactive else None), depth=pipeline_depth,
                                         min_l=l, vu_ids_list=[range(lo_, hi_) for lo_, hi_ in warm])
                break                                        # warm-up: one sub-batch (compile, graph capture, self-check)
        if live is not None:
            live.end_rep(rep, agg)                           # END; the verifier's session verdict (waits for its Rust verify)
        for si, lo, hi, V, proof, pubs, lay_i in proved:
            if skip_local and rep > 0:
                ok, why = True, "not verified in-process (--no-local-verify): the live verifier's session verdict is the check"
            else:
                t1 = time.perf_counter()
                ok, why = R.verify_vus(proof, pubs, lay_i, coins=V)
                vsum += time.perf_counter() - t1
            assert ok, f"sub-batch [{lo}, {hi}): {why}"
            if dump_dir is not None and 0 < rep <= dump_reps:
                rd = dump_dir / f"rep{rep}"
                rd.mkdir(exist_ok=True)
                stem = f"sub_{si:02d}"
                st = _statement_of(R, pubs, lay_i, R.cfg(lay_i.l), n_proofs)          # ligero-statement/v4: R.relation names it
                (rd / f"{stem}.stmt").write_bytes(statement_bytes(st))
                # the pair (shared): <stem>.proof the GEMM side (v3, F after the body), <stem>.hproof the hash side (v1)
                (rd / f"{stem}.proof").write_bytes(proof_bytes(proof.g if shared else proof))
                entry = {"rep": rep, "sub": si, "vus": [lo, hi], "stmt": f"rep{rep}/{stem}.stmt", "proof": f"rep{rep}/{stem}.proof",
                         "proof_bytes": (rd / f"{stem}.proof").stat().st_size, "proof_sha256": sha256_file(rd / f"{stem}.proof"),
                         "stmt_bytes": (rd / f"{stem}.stmt").stat().st_size,
                         "python_verdict": {"accepted": ok, "reason": why}, "prover_seconds": proof.timings.get("total")}
                if shared:
                    (rd / f"{stem}.hproof").write_bytes(proof_bytes(proof.h))
                    entry.update({"proof_h": f"rep{rep}/{stem}.hproof", "proof_h_bytes": (rd / f"{stem}.hproof").stat().st_size,
                                  "proof_h_sha256": sha256_file(rd / f"{stem}.hproof"), "n_units": int(len(pubs[6]))})
                if hashed:
                    entry["multiproof_bytes"] = 32 * sum(len(x) for x in pubs[4])
                    entry["auth_openings_seconds"] = proof.timings.get("auth_openings")
                if V is not None:
                    (rd / f"{stem}.coins").write_bytes(V.r1 + V.s1 + V.r2 + V.s2)
                    entry["coins"] = f"rep{rep}/{stem}.coins"
                    entry["coins_bytes"] = (rd / f"{stem}.coins").stat().st_size
                    entry["coins_sha256"] = sha256_file(rd / f"{stem}.coins")     # D7: the step-0 coins' custody, like proof_sha256
                    if shared and proof.h.coins is not None:
                        Vh = proof.h.coins                                      # the hash side's coins (the live verifier's under --verifier)
                        (rd / f"{stem}.hcoins").write_bytes(Vh.r1 + Vh.s1 + Vh.r2 + Vh.s2)
                        entry["coins_h"] = f"rep{rep}/{stem}.hcoins"
                dump_manifest["files"].append(entry)
        lays_h_seen = [pf.lay_h for _, _, _, _, pf, _, _ in proved] if shared else None
        del proved
        if rep > 0:
            # the batch verdict (PROTOCOL.md 8c, "Batch verification"; D10): every sub-batch verified above; now the union
            # bound of this verifier's own per-proof bounds over the n_proofs presented, and each proof's n_proofs (what
            # config_for sized t for) against that count.  Arithmetic on n_proofs numbers, inside the verifier time.
            t0 = time.perf_counter()
            if shared:
                batch = R.batch_bound(cfgs_seen, [n_proofs] * len(cfgs_seen), target_bits, lays_h=lays_h_seen)
            else:
                batch = R.batch_bound(cfgs_seen, [n_proofs] * len(cfgs_seen), target_bits)
            vsum += time.perf_counter() - t0
            assert batch["accepted"], batch["reason"]
        wall = time.perf_counter() - t_pass
        if rep == 0:
            print("warm-up (one sub-batch, with the prover's self-check):", {k_: round(v_, 3) for k_, v_ in agg.items()})
            commit_cold = agg.get("commitment")
            continue
        keys = agg.keys()
        runs.append((agg, vsum))
        print(f"rep {rep}: prover {agg['total']:.3f}s (+hints {agg['hints_host']:.3f}s) over {n_proofs} proofs; "
              f"verifier {vsum:.3f}s; pass wall {wall:.3f}s ({(agg['total'] + agg['hints_host']) / total_vus * 1e3:.3f} ms/VU)"
              + (f"; commitment {agg['commitment']:.3f}s (+ row chains {agg['row_chain']:.3f}s in hints, statements "
                 f"{agg['statement']:.3f}s in total); end to end {agg['commitment'] + agg['total'] + agg['hints_host']:.3f}s"
                 if commit_per_rep else ""))
    dumps = None
    if dump_dir is not None:
        (dump_dir / "manifest.json").write_text(json.dumps(dump_manifest, indent=1, default=str))
        print(f"dumped {len(dump_manifest['files'])} proofs to {dump_dir}", flush=True)
        dumps = _verify_dumps(dump_dir, dump_manifest, R, args.device, target_bits)
    # t.total and every bucket come from one rep (phases.median_rep); the per-key medians are printed as diagnostics only
    med_rep = phases.median_rep(runs)
    med = dict(runs[med_rep][0])
    per_key_med = {k_: statistics.median(r[0][k_] for r in runs) for k_ in keys}
    vmed = statistics.median(r[1] for r in runs)
    peak_dev = torch.cuda.max_memory_allocated() if args.device.startswith("cuda") else 0
    nbytes = {k_: v_ * n_proofs for k_, v_ in proof.n_bytes().items()}
    snd = (R.soundness_report(cfg, n_proofs, lays_h_seen) if shared
           else soundness(cfg, R.tb.L, R.tb.Q, n_proofs, chain_extras=R.chain_extras))
    # security.achieved_log2 is the BATCH bound -log2 sum_i 2^b_i as the verifier's batch_bound computed it (vu.bench_vu since
    # lane b-batch-bound); the prover-side soundness() of the same n_proofs must agree with it to the last bit.
    assert batch is not None and abs(batch["batch_log2"] - snd["total_log2"]) < 1e-6, (batch, snd["total_log2"])
    total = med["total"] + med["hints_host"]
    per_vu = total / total_vus
    hw = contract.probe_hardware()
    from . import witness as witness_mod
    sw = {"backend": {"name": rel.backend_name, "commit": _git_commit(), "hash": proof.hash_name,
                      "challenge_expander": "SHAKE-256 (placeholder)", "impl": witness_mod.IMPL, "pipeline": pipeline_depth},
          "toolchain": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__}}
    # The bare v4 statement publishes the operands, so vu.py's ZK_VACUOUS_NOTE applies; the hashed v5 statement does not
    # (hash-relation final report §9): it carries the operand rows' Poseidon2 digests + the Merkle roots + y16 instead.
    privacy_note = (ZK_SHARED_NOTE if shared else ZK_HASHED_NOTE if hashed else ZK_VACUOUS_NOTE)
    auth_omitted = ("" if hashed else "external authentication; ")
    if cfg.zk and interactive:
        proof_class, zk_mode = "COMPLETE_ZK_BACKEND", ZK_MODE + ZK_MODE_HM96 + privacy_note
        omitted = auth_omitted + "unopened columns hidden by Blake3 of a uniform column"
    elif cfg.zk:
        proof_class, zk_mode = "COMPLETE_HVZK_BACKEND", ZK_MODE + ZK_MODE_FS + privacy_note
        omitted = auth_omitted + "malicious-verifier ZK outside the programmable random-oracle model"
    else:
        proof_class, zk_mode, omitted = "NON_ZK_PROOF_DIAGNOSTIC", "none", auth_omitted + "ZK masking"
    zk_statement = snd["zk_statement"]
    if hashed and cfg.zk:
        zk_statement = zk_statement.split("; operands a, b")[0] + privacy_note.replace("; NOTE: ", "; ", 1)
    inst = set_inst if set_inst else instances_ref(rel, total_vus, tile=tile, root=root)
    fp = contract.fingerprint(
        profile=rel.target.name, domain=contract.DOMAIN, K=_K(rel), B=total_vus, N_subbatches=n_proofs,
        proof_class=proof_class, authentication=(R.AUTH if hashed else "excluded"), zk_mode=zk_mode,
        security={"target": round(args.target), "field": "BabyBear", "challenge_coords": cfg.D, "rs_n": cfg.n, "rs_k": cfg.k, "rs_l": cfg.l,
                  "zk": cfg.zk, "t_pad": cfg.t_pad, "mode": cfg.mode, "bound_model": snd["bound_model"],
                  "hash_assumption": snd["hash_assumption"], "zk_statement": zk_statement,
                  "fiat_shamir_queries_log2": snd["fiat_shamir_queries_log2"],
                  "opened_columns": cfg.t, "achieved_log2": batch["batch_log2"], "achieved_bits": -batch["batch_log2"],
                  "verifier_target_bits": target_bits, "terms_log2": snd["terms_log2"],
                  "per_proof_target_log2": snd["per_proof_target_log2"], "per_proof_total_log2": snd["per_proof_total_log2"],
                  "union_over": n_proofs, "hash_placeholder": True},
        hardware=hw, software=sw,
        preprocessing=f"constraint compilation (a function of the relation's Params only, {R.compile_seconds:.2f}s)",
        retention="retained",
        instances=inst,
        scope="vu", omitted=omitted,
        relation=rel.relation_name, model=rel.model_name, commit=_git_commit(), step_k=rel.k, steps_per_vu=steps,
        rows=R.sys.m, committed_rows=n_rows(cfg, R.sys.m, True), linear_constraints=R.tb.L, quadratic_constraints=R.tb.Q,
        census=cen, witness_generation=rel.witness_generation, B_proof=per_proof,
        statement_format=("ligero-statement/v6 (+ ligero-proof/v3 G, /v1 H)" if shared else
                          hashed_statement_format() if hashed else "ligero-statement/v4"),   # what --dump-dir writes (serialize.py)
        note=f"{total_vus} VUs as {n_proofs} independently committed proofs of <= {per_proof} VUs each (l = {l} columns, "
             f"{l - per_proof * steps} pad columns per proof); prover time = sum over the proofs; soundness = union bound; "
             f"{rel.claim_note}"
             + (f"; authentication = {R.AUTH}: the operand words are private witness wires with 16 lane rows per operand per "
                f"unit; every DISTINCT x row / W column of a sub-batch is hashed ONCE in a paired hash-side proof (one Poseidon2 "
                f"sponge per row unit, {R.rh.census['rows_total']} rows/unit, l_H = {lays_h_seen[0].l}) and each VU proves its "
                f"operands equal that row by a post-commitment fingerprint ({R.fp_coords} coordinates, coins after the root, "
                f"collision <= ((16 x {steps} - 1) / p)^{R.fp_coords}); the digests are the hash side's chain-end pins and leaf/v2h "
                f"leaves under the x / W roots; sharing = {'none' if tile is None else 'tile%dx%d' % tile} ({R.trees['a'].count} "
                f"x rows + {R.trees['b'].count} W columns committed for {total_vus} VUs); 2 proofs per sub-batch, both sized for "
                f"the union over {2 * n_proofs}; hash = poseidon2-babybear-w24" if shared else
                f"; authentication = {R.AUTH}: the operand words are private witness wires, each unit hashes its {rel.k} words of "
                f"x and of W with the {R.leaf.name} leaf scheme ({R.leaf.assumption}; {R.hr.census['rows_hash']} hash rows per unit) "
                f"along the chain, the digests ({R.leaf.digest_elems} elements each) are chain-end pins and {R.leaf.schema} leaves "
                f"under the x / W roots; sharing = {'none' if tile is None else 'tile%dx%d' % tile} ({R.trees['a'].count} x rows + "
                f"{R.trees['b'].count} W columns committed for {total_vus} VUs); the statement carries digests, leaf indices, "
                f"roots and SHA-256 multiproofs instead of the operand words; hash = {R.leaf.hash_name}" if hashed else ""),
    )
    if hashed:
        fp["hash"] = R.leaf.hash_name
        fp["statement_relation"] = R.relation
        fp["leaf"] = {"scheme": R.leaf.name, "suffix": R.leaf_name, "schema": R.leaf.schema, "params_sha256": R.leaf.params_sha256,
                      "digest_elems": R.leaf.digest_elems, "digest_bytes_per_row": len(R.leaf.leaf_bytes(np.asarray(R.digests["a"])[0])),
                      "assumption": R.leaf.assumption, "privacy_note": leaf_privacy_note(R.leaf)}
        fp["sharing"] = "none" if tile is None else f"tile{tile[0]}x{tile[1]}"
        fp["committed_rows"] = {"x": R.trees["a"].count, "w": R.trees["b"].count, "y": R.trees["y"].count}
    if shared:
        fp["row_sharing"] = {"mechanism": "fingerprint chain (chain.py); paired hash-side proof per sub-batch",
                             "fp_coords": R.fp_coords, "g_rows_per_unit": R.sys.m, "h_rows_per_unit": R.sys_h.m,
                             "h_layouts": [dataclasses.asdict(lh) for lh in lays_h_seen]}
    meas = [
        *(contract.measurement(name_, v_) for name_, v_ in phases.buckets(med).items()),
        contract.measurement("t.total", total),
        contract.measurement("verifier.seconds", vmed),
        contract.measurement("transcript.bytes", nbytes["total"]),
        contract.measurement("rounds.protocol", depth),
        contract.measurement("rounds.sequential_depth", depth),
        contract.measurement("mem.peak_device_bytes", peak_dev),
        contract.measurement("census.witness_elements", R.sys.m * l * n_proofs),
        contract.measurement("census.committed_elements", R.sys.m * cfg.n * n_proofs),
        contract.measurement("census.bytes_encoded", R.sys.m * cfg.n * 4 * n_proofs),
        contract.measurement("vu.seconds_per_vu", per_vu, "s"),
        contract.measurement("vu.count", total_vus, "count"),
        *contract.rate_measurements(total, total_vus, _K(rel), rel.target.name),
        contract.measurement("split.subbatches", n_proofs, "count"),
        contract.measurement("split.per_proof_vus", per_proof, "count"),
        contract.measurement("split.hints_seconds", med["hints_host"], "s"),
        contract.measurement("split.witness_torch_seconds", med["witness"], "s"),
        contract.measurement("split.encode_seconds", med["encode"], "s"),
        contract.measurement("split.transfer_seconds", med["transfer"], "s"),
        contract.measurement("split.merkle_seconds", med["merkle"], "s"),
        contract.measurement("split.tests_seconds", med["tests"], "s"),
        contract.measurement("split.openings_seconds", med["openings"], "s"),
        contract.measurement("proof.opened_columns_bytes", nbytes["opened_columns"], "B"),
        contract.measurement("proof.merkle_paths_bytes", nbytes["merkle_paths"], "B"),
        contract.measurement("proof.w_h_q_bytes", nbytes["w"] + nbytes["h"] + nbytes["q_chain"] + nbytes["v_linear"], "B"),
        contract.measurement("split.zk_masks_seconds", med.get("zk_masks", 0.0), "s"),
        contract.measurement("prove_wall", total, "s"),
        contract.measurement("verify_wall", vmed, "s"),
        contract.measurement("proof_bytes", nbytes["total"], "B"),
        contract.measurement("depth", depth, "count"),
        contract.measurement("soundness_bits", -snd["total_log2"], "bits"),
        contract.measurement("rows_per_unit", R.sys.m, "count"),
        contract.measurement("units_per_vu", steps, "count"),
    ]
    live_fields = live.fields(contract, total, med) if live is not None else None
    if live_fields is not None:
        meas += live_fields[0]           # t.total_live / net.* / verify.* (lane live-verifier; additive)
    if hashed:
        meas += [contract.measurement("relation.hash.rows_added_per_unit", R.hr.census["rows_added"], "count"),
                 contract.measurement("relation.hash.permutations_per_unit", R.hr.census.get("permutations_per_unit", 0), "count"),
                 contract.measurement("relation.hash.linked_rows", R.hr.census["linked_rows"], "count"),
                 contract.measurement("relation.hash.commit_seconds", commit_seconds, "s"),
                 contract.measurement("relation.hash.sponge_seconds", R.sponge_seconds, "s"),
                 contract.measurement("relation.hash.auth_openings_seconds", med.get("auth_openings", 0.0), "s"),
                 contract.measurement("relation.hash.committed_x_rows", R.trees["a"].count, "count"),
                 contract.measurement("relation.hash.committed_w_cols", R.trees["b"].count, "count")]
        if not commit_per_rep:
            # the commitment-building bucket (kb/TABLES.md "Commitment time counts"): NOT a t.* bucket (outside t.total, which
            # stays the prover's), reported on its own and end to end; the breakdown is the median call's
            steady = commit_runs[1:] or commit_runs
            c_med = sorted(steady, key=lambda r: r[0])[(len(steady) - 1) // 2]
            meas += [contract.measurement("commit.seconds", c_med[0], "s"),
                     contract.measurement("commit.cold_seconds", commit_runs[0][0], "s"),
                     contract.measurement("commit.reps", len(steady), "count"),
                     *(contract.measurement(f"commit.{k_}_seconds", v_, "s") for k_, v_ in c_med[1].items()),
                     contract.measurement("e2e.seconds", c_med[0] + total, "s"),
                     contract.measurement("e2e.cold_seconds", commit_runs[0][0] + total, "s")]
    if commit_per_rep:
        c_med = statistics.median(r[0]["commitment"] for r in runs)
        e2e = total + c_med
        native = contract.native_reference(rel.target.name).flop_per_second
        meas += [contract.measurement("commit.seconds", c_med, "s"),
                 contract.measurement("commit.cold_seconds", commit_cold, "s"),
                 contract.measurement("commit.reps", len(runs), "count"),
                 contract.measurement("commit.row_chain_seconds", med["row_chain"], "s"),
                 contract.measurement("commit.statement_seconds", med["statement"], "s"),
                 contract.measurement("e2e.seconds", e2e, "s"),
                 contract.measurement("e2e.vu_per_second", total_vus / e2e, "1/s"),
                 contract.measurement("e2e.overhead_vs_native_peak", native * e2e / (2 * _K(rel) * total_vus), "ratio")]
        fp["commitment"] = {
            "per_rep": True, "scheme": "frame-v3", "leaf_schema": R.leaf.schema, "trees": ["x", "W", "y"],
            "committer": "relchain.HashedRelationRunner.commit_vus (hashauth.load_or_build_trees, no cache)",
            "timed": "--commit-per-rep: every rep commits its batch afresh before proving it; commit.seconds = the median over "
                     "the timed reps of the x / W / y trees and their row digests (commit.cold_seconds = the warm-up rep's); the "
                     "prover's row chains (commit.row_chain_seconds) are in t.witness and rebuilding the hashed statements "
                     "(commit.statement_seconds) in t.serialization, both inside t.total; e2e.seconds = commit.seconds + t.total"}
    if shared:
        meas += [contract.measurement("relation.share.g_seconds", med.get("g_total", 0.0), "s"),
                 contract.measurement("relation.share.h_seconds", med.get("h_total", 0.0), "s"),
                 contract.measurement("relation.share.g_rows_per_unit", R.sys.m, "count"),
                 contract.measurement("relation.share.h_rows_per_unit", R.sys_h.m, "count"),
                 contract.measurement("relation.share.h_l", lays_h_seen[0].l, "count"),
                 contract.measurement("relation.share.h_units", lays_h_seen[0].n_vus, "count"),
                 contract.measurement("relation.share.fp_coords", R.fp_coords, "count"),
                 contract.measurement("relation.share.fingerprint_bytes", nbytes.get("fingerprints", 0), "B"),
                 contract.measurement("relation.share.g_proof_bytes", nbytes.get("g_total", 0), "B"),
                 contract.measurement("relation.share.h_proof_bytes", nbytes.get("h_total", 0), "B")]
        if dump_manifest is not None and dump_manifest["files"]:
            fs = [e for e in dump_manifest["files"] if e["rep"] == 1]
            meas += [contract.measurement("relation.hash.statement_bytes_total", sum(e["stmt_bytes"] for e in fs), "B"),
                     contract.measurement("relation.hash.multiproof_bytes_total", sum(e["multiproof_bytes"] for e in fs), "B")]
    elif dump_manifest is not None and dump_manifest["files"]:
        fs = [e for e in dump_manifest["files"] if e["rep"] == 1]
        meas += [contract.measurement("relation.statement_bytes_total", sum(e["stmt_bytes"] for e in fs), "B")]
    for k_, v_ in cen.items():
        if isinstance(v_, int) and not isinstance(v_, bool):
            # ``census.`` is a contract-reserved prefix: only the seven names contract.py defines may live under
            # it (emitted above).  The relation's own per-gadget row counts go under ``relation.census.``.
            meas.append(contract.measurement(f"relation.census.{k_}", v_, "count"))
    checker = (f"archive.direct.ligero.relchain bench_vu_rel[{rel.name}] (verifier accepted every rep; prover self-check on the "
               "warm-up sub-batch")
    status = "passed"
    evidence = {"soundness": snd, "batch": batch,
                **({"commit": {"runs": [{"seconds": s_, **b_} for s_, b_ in commit_runs], "evidence": commit_ev,
                               "rule": "commit.seconds = median over calls 1..N of the full commitment built from scratch "
                                       "(--commit-reps N; call 0 = cold, relation.hash.commit_seconds); e2e.seconds = "
                                       "commit.seconds + t.total"}} if hashed and not commit_per_rep else
                   {"commit": {"evidence": commit_ev, "rule": "--commit-per-rep: every rep recommits the set; its tree "
                                                              "references must equal this first build's"}} if commit_per_rep else {}),
                "reps": [{"prover_total": r[0]["total"] + r[0]["hints_host"], "verifier": r[1],
                          **({"commit": r[0]["commitment"], "e2e": r[0]["total"] + r[0]["hints_host"] + r[0]["commitment"]}
                             if commit_per_rep else {})} for r in runs],
                "phase_rep": {"index": med_rep, "rule": "t.total and the t.* buckets are this rep's (the median prover_total of "
                                                        "reps); verifier.seconds is the median over the reps"},
                "verify_wall": ("--no-local-verify: no in-process verification of the timed reps (the live verifier's Rust "
                                "verdicts are the check); verifier.seconds is the batch bound's arithmetic only" if skip_local else
                                "the live in-process Python verifier, median over the reps of the sum over the sub-batches (shares code "
                                "with the prover), batch verdict included; not a file verifier's process wall")}
    detail = f"{args.reps} reps, {total_vus} VUs as {n_proofs} proofs at l={l}; batch {batch['reason']}"
    if dumps is not None:
        # the same code as the prover (the Python file verifier, cold from the dumped bytes): a self-check of the dump
        # round trip, NOT an independent verification -- that is the `verify` lane's label, never this producer's
        checker += "; Python file verifier (serialize.verify_files, shares code with the prover) on every dumped proof, cold from bytes"
        evidence["dumps"] = dumps
        detail += (f"; dumps: {dumps['accepted']}/{dumps['total']} accepted cold from bytes ({dumps['statement_format']}) by the Python file "
                   f"verifier in {dumps['wall']:.2f}s wall on {args.device}, in-process (a self-check: it shares code with the prover; "
                   "an independent verifier has to judge the same bytes)")
        detail += f"; dumped batch verdict per rep: {[b['reason'] for b in dumps['batch'].values()]}"
        if dumps["accepted"] != dumps["total"] or not dumps["batch_accepted"]:
            status = "failed"
    if live_fields is not None:
        # the live verifier's verdicts are reported here as the PROVER received them; the verifier's own dump + verdict
        # artifact (live.py record) is what carries the independent `verified=` label, never this producer's result
        checker += f"; live verifier at {args.verifier}"
        evidence["live_verifier"] = live_fields[1]
        detail += live_fields[2]
        if not live_fields[3]:
            status = "failed"
    checker += ")"
    result = {"schema": contract.RESULT_SCHEMA,
              "run_id": args.run_id or os.environ.get("RESEARCH_RUN_ID") or f"b-ligero-{rel.name}-vu-{int(time.time())}",
              "workload_fingerprint": fp,
              "validation": {"status": status, "checker": checker, "detail": detail, "evidence": evidence},
              # the dump tree (relative to the run dir) travels with the result as its run-files/v1 artifact (store README §10)
              "measurements": meas, "measurement_files": [], "artifacts": dump_artifacts(dump_dir, args.out)}
    problems = contract.validate(result)
    if problems:
        print("contract problems:", problems)
    print(json.dumps({"relation": rel.name, "seconds_per_vu": per_vu, "prover_total": total, "verifier": vmed, "bytes": nbytes,
                      "n_proofs": n_proofs, "per_proof_vus": per_proof, "t": cfg.t, "zk": cfg.zk, "k": cfg.k,
                      "peak_device_bytes": peak_dev, "soundness_log2": snd["total_log2"], "per_proof_log2": snd["per_proof_total_log2"],
                      "rows": R.sys.m, "medians": med, "per_key_medians": per_key_med, "dumps": dumps, "validation": status},
                     indent=1))
    if args.out:
        Path(os.path.expandvars(args.out)).write_text(json.dumps(result, indent=1, default=str))   # $RESEARCH_RUN_DIR/result.json
    return 0 if status == "passed" else 1


def _verify_dumps(dump_dir: Path, manifest: dict, R: RelationChainRunner, device: str, target_bits: float = 128.0) -> dict:
    """``serialize.verify_files`` on every dumped (statement, proof[, coins]) of ``manifest`` with the prover's own runner:
    the files parse and verify cold from bytes (interactive mode: against the coins in the transcript -- a self-check, D7);
    then, per dumped rep, the batch verdict (PROTOCOL.md 8c, D10) on the STATEMENTS' configurations and ``n_proofs`` --
    what ``serialize.verify_batch_dir`` / ``ligero-verify batch`` recompute on the same tree.  The verdicts go to
    ``<dump_dir>/verify_python.json``; returns the summary that enters ``validation.evidence.dumps``."""
    from .serialize import read_statement, verify_files
    items, t0 = [], time.perf_counter()
    per_rep: dict[int, list] = {}
    for e in manifest["files"]:
        ok, why, secs = verify_files(dump_dir / e["stmt"], dump_dir / e["proof"], device, runner=R)
        items.append({"rep": e["rep"], "sub": e["sub"], "proof": e["proof"], "accepted": ok, "reason": why, "seconds": secs})
        if not ok:
            print(f"!! dumped proof {e['proof']} REJECTED cold from bytes: {why}", flush=True)
        per_rep.setdefault(e["rep"], []).append(read_statement((dump_dir / e["stmt"]).read_bytes()))
    batches = {}
    for rep, sts in sorted(per_rep.items()):
        names = [f"sub_{st_i:02d}" for st_i in range(len(sts))]
        if isinstance(R, SharedHashedRunner):
            # the pair: the hash side's layout is the statement's (its l floors at l_G / 4: layout_h), not the minimum
            b = R.batch_bound([st.config() for st in sts], [st.n_proofs for st in sts], target_bits, names,
                              lays_h=[st.shared.layout(st.steps) for st in sts])
        else:
            b = R.batch_bound([st.config() for st in sts], [st.n_proofs for st in sts], target_bits, names)
        batches[str(rep)] = b
        if not b["accepted"]:
            print(f"!! dumped rep {rep}: batch verdict REJECTED: {b['reason']}", flush=True)
    n_ok = sum(1 for it in items if it["accepted"])
    batch_ok = all(b["accepted"] for b in batches.values())
    summary = {"verifier": "python (archive.direct.ligero.serialize.verify_files; shares code with the prover: a self-check)",
               "independent": False, "statement_format": manifest["format"]["statement"],
               "accepted": n_ok, "total": len(items),
               "batch": batches, "batch_accepted": batch_ok,
               "seconds": sum(it["seconds"] for it in items), "wall": time.perf_counter() - t0,
               "coins": "the transcript's own coins replayed (file verifier)" if R.interactive else "n/a (Fiat-Shamir)"}
    (dump_dir / "verify_python.json").write_text(json.dumps({**summary, "items": items}, indent=1))
    print(f"dumps: {n_ok}/{len(items)} accepted cold from bytes by the Python file verifier in {summary['wall']:.2f}s; "
          f"batch verdict per rep: {[b['reason'] for b in batches.values()]}", flush=True)
    return summary
