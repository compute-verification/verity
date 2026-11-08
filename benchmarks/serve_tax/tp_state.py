"""A tensor-parallel engine's per-rank boundary state at a drained edge, on the worker side: what the engine core's
``collective_rpc`` runs on every rank (each function takes the worker first, as ``collective_rpc`` passes it).

At an edge `tp_edge` drains, each rank holds:

- its paged KV cache: every layer whose ``kv_cache`` is block-major over the pool (``shape[0]`` the pool's
  ``num_blocks``), GLM-5.3's MLA latent and DSA indexer-key layers alike; one block id names a page in each
  (`kv_layers`);
- its sampling RNG: numpy's global generator (Model Runner V2 draws a seedless request's seed from it when the request
  is added), the device's CUDA generator (FlashInfer's top-k/top-p sampler draws from it), torch's CPU generator and
  Python's, and under V2 each request's seed (`rng_state`);
- where it runs: its ranks, its GPU and the all-reduce backends its tensor-parallel group enabled (`rank_record`).

`digest` hashes a set of blocks layer by layer (per block, so two ranks or two runs can be compared down to the first
block that differs), `dump` writes the blocks, `load` writes them back into a fresh engine at the same block ids and
fills every other block with ``FILL`` (NaN in bf16 and in e4m3), so a read outside the set shows as a divergence, and
`set_rng` puts the RNG back (and, for a restore that adds its requests to the fresh worker again rather than restoring
the runner's rows, keeps each one's served seed: `SeedHook`). For a replay that adds the edge's requests to a fresh engine
again (`rebuild`), `copy_blocks` loads a dump's blocks into other block ids, `fill_blocks` fills blocks byte by byte and
`block_roots` hashes each request's whole blocks (`dump_roots`, the same over a dump). torch and numpy are imported
inside the functions: the workers have them, the CPU tests stand in.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import random
import re
import socket
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

#: the byte every block outside a restored set is filled with: NaN in bf16 (0xFFFF) and in e4m3 (0xFF)
FILL = 0xFF
#: bytes of a block's digest
BLOCK_DIGEST = 16
#: blocks a rank moves to the host at once
CHUNK_BLOCKS = 512


def runner_of(worker: Any) -> Any:
    """The model runner behind a worker (``collective_rpc`` hands the worker; a runner is accepted as itself)."""
    return getattr(worker, "model_runner", worker)


def _natural(name: str) -> list[Any]:
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", name)]


def kv_layers(runner: Any) -> list[tuple[str, Any]]:
    """Each distinct block-major KV tensor of the runner's layers, by layer name in natural order (layer 2 before 10):
    the forward context's layers whose ``kv_cache`` (or its first entry) has the pool's ``num_blocks`` rows."""
    num_blocks = runner.kv_cache_config.num_blocks
    ctx = runner.compilation_config.static_forward_context
    out, seen = [], set()
    for name in sorted(ctx, key=_natural):
        t = getattr(ctx[name], "kv_cache", None)
        if isinstance(t, (list, tuple)):
            t = t[0] if t else None
        if t is None or getattr(t, "ndim", 0) < 1 or t.shape[0] != num_blocks:
            continue
        key = (t.data_ptr(), tuple(t.shape), tuple(t.stride()))
        if key not in seen:
            seen.add(key)
            out.append((name, t))
    return out


def layer_spec(layers: Sequence[tuple[str, Any]]) -> list[dict[str, Any]]:
    """Each layer's name, dtype, block shape and bytes a block: what a dump's reader must find in its own engine."""
    return [{"name": n, "dtype": str(t.dtype).removeprefix("torch."), "block": list(t.shape[1:]),
             "block_bytes": t[0].numel() * t.element_size()} for n, t in layers]


def _chunks(ids: Sequence[int], n: int = CHUNK_BLOCKS) -> Iterable[Sequence[int]]:
    for i in range(0, len(ids), n):
        yield ids[i:i + n]


def _host_bytes(t: Any, ids: Sequence[int]) -> Any:
    """Blocks ``ids`` of ``t`` as a host uint8 array, one row a block."""
    import torch

    idx = torch.as_tensor(list(ids), dtype=torch.long, device=t.device)
    rows = t.index_select(0, idx).contiguous()
    return rows.view(torch.uint8).reshape(len(ids), -1).cpu().numpy()


def block_digests(rows: Any) -> bytes:
    """Each row's BLAKE2b digest (``BLOCK_DIGEST`` bytes), concatenated."""
    return b"".join(hashlib.blake2b(memoryview(r), digest_size=BLOCK_DIGEST).digest() for r in rows)


def digest(worker: Any, ids: Sequence[int], out: str | None = None, threads: int = 8) -> dict[str, Any]:
    """This rank's digest of blocks ``ids``: each layer's SHA-512 over its blocks' digests in ``ids`` order (`layers`),
    and with ``out``, the blocks' digests themselves at ``out/digests.rank<R>.bin`` (layer-major, ``len(ids)`` of
    ``BLOCK_DIGEST`` bytes a layer), for `first_difference`. Also its RNG state's digest and its rank record."""
    runner = runner_of(worker)
    layers = kv_layers(runner)
    per_layer: list[bytes] = []
    with ThreadPoolExecutor(max_workers=threads) as pool:
        for _, t in layers:
            parts = [pool.submit(block_digests, _host_bytes(t, c)) for c in _chunks(ids)]
            per_layer.append(b"".join(p.result() for p in parts))
    rank = rank_record(worker)
    if out is not None:
        Path(out).mkdir(parents=True, exist_ok=True)
        (Path(out) / f"digests.rank{rank['rank']}.bin").write_bytes(b"".join(per_layer))
    rng = rng_state(worker)
    return {"rank": rank, "layers": layer_spec(layers), "blocks": len(ids),
            "layer_digests": [hashlib.sha512(d).hexdigest() for d in per_layer],
            "rng_digest": rng_digest(rng), "rng": rng}


def dump(worker: Any, out: str, ids: Sequence[int], ranks: Sequence[int]) -> dict[str, Any] | None:
    """If this rank is in ``ranks``, write blocks ``ids`` of each layer to ``out/kv.rank<R>/<layer index>.bin`` (rows in
    ``ids`` order) beside ``layers.json``; returns what it wrote, or None."""
    rank = rank_record(worker)["rank"]
    if rank not in ranks:
        return None
    layers = kv_layers(runner_of(worker))
    d = Path(out) / f"kv.rank{rank}"
    d.mkdir(parents=True, exist_ok=True)
    total = 0
    for i, (_, t) in enumerate(layers):
        with open(d / f"{i:03d}.bin", "wb") as f:
            for c in _chunks(ids):
                rows = _host_bytes(t, c)
                f.write(rows.tobytes())
                total += rows.nbytes
    (d / "layers.json").write_text(json.dumps({"ids": list(ids), "layers": layer_spec(layers)}))
    return {"rank": rank, "dir": str(d), "bytes": total}


def load(worker: Any, src: str, ids: Sequence[int] | None = None, fill: bool = True) -> dict[str, Any]:
    """Write the blocks a `dump` at ``src`` (a ``kv.rank<R>`` directory) holds back into this rank's layers at the same
    block ids (all of them, or the subset ``ids``), and with ``fill``, every other block to ``FILL``. Refuses a dump
    whose layers (names, dtypes, block shapes) aren't this engine's."""
    import numpy as np
    import torch

    d = Path(src)
    meta = json.loads((d / "layers.json").read_text())
    layers = kv_layers(runner_of(worker))
    mine = layer_spec(layers)
    if mine != meta["layers"]:
        first = next((i for i, (a, b) in enumerate(zip(mine, meta["layers"])) if a != b), min(len(mine), len(meta["layers"])))
        raise ValueError(f"tp_state.load: the dump at {d} has {len(meta['layers'])} layers and this engine {len(mine)}; "
                         f"they first differ at layer {first} ({meta['layers'][first:first + 1]} vs {mine[first:first + 1]}): "
                         "restore into an engine built from the served engine's header")
    have = list(meta["ids"])
    want = have if ids is None else list(ids)
    pos = {b: i for i, b in enumerate(have)}
    missing = [b for b in want if b not in pos]
    if missing:
        raise ValueError(f"tp_state.load: blocks {missing[:8]} aren't in the dump at {d} ({len(missing)} missing)")
    sel = np.asarray([pos[b] for b in want], dtype=np.int64)
    for i, ((_, t), spec) in enumerate(zip(layers, mine)):
        rows = np.fromfile(d / f"{i:03d}.bin", dtype=np.uint8).reshape(len(have), spec["block_bytes"])[sel]
        u8 = t.view(torch.uint8)
        if fill:
            u8.fill_(FILL)
        dst = torch.as_tensor(want, dtype=torch.long, device=t.device)
        src_rows = torch.from_numpy(np.ascontiguousarray(rows)).to(t.device).view(len(want), *u8.shape[1:])
        u8.index_copy_(0, dst, src_rows)
    return {"rank": rank_record(worker)["rank"], "blocks": len(want), "filled": bool(fill)}


def load_rank(worker: Any, edge_dir: str, ids: Sequence[int] | None = None, fill: bool = True) -> dict[str, Any]:
    """`load` from an edge's directory: this rank's own dump when the edge kept every rank's (they differed), else rank
    0's."""
    rank = rank_record(worker)["rank"]
    own = Path(edge_dir) / f"kv.rank{rank}"
    return load(worker, str(own if own.is_dir() else Path(edge_dir) / "kv.rank0"), ids, fill)


def set_rng_ranked(worker: Any, rngs: Mapping[Any, Mapping[str, Any]], seeds: bool = False) -> dict[str, Any]:
    """`set_rng` with this rank's own state from ``rngs`` (rank -> `rng_state`, keys ints or their strings)."""
    rank = rank_record(worker)["rank"]
    return set_rng(worker, rngs[rank] if rank in rngs else rngs[str(rank)], seeds)


def rank_record(worker: Any) -> dict[str, Any]:
    """Where this rank runs: its global, tensor-parallel and local ranks, its device (index, UUID, PCI bus), the host,
    and the all-reduce backends its tensor-parallel group enabled, in dispatch order (outside a vLLM install, as in the
    CPU tests, no group)."""
    import torch

    rec: dict[str, Any] = {"rank": int(getattr(worker, "rank", 0)), "local_rank": int(getattr(worker, "local_rank", 0)),
                           "host": socket.gethostname()}
    if torch.cuda.is_available():
        dev = torch.cuda.current_device()
        p = torch.cuda.get_device_properties(dev)
        rec["device"] = {"index": dev, "uuid": str(getattr(p, "uuid", "")), "name": p.name,
                         "pci": [getattr(p, k, None) for k in ("pci_domain_id", "pci_bus_id", "pci_device_id")]}
    if importlib.util.find_spec("vllm") is not None:
        from vllm.distributed.parallel_state import get_tp_group

        g = get_tp_group()
        rec["tp_rank"] = g.rank_in_group
        comm = g.device_communicator
        rec["allreduce"] = [name for attr, name in (("fi_ar_comm", "FLASHINFER"), ("ca_comm", "CUSTOM"),
                                                    ("symm_mem_comm", "SYMM_MEM"), ("pynccl_comm", "PYNCCL"))
                            if getattr(comm, attr, None) is not None and not getattr(getattr(comm, attr), "disabled", False)]
    return rec


def _seed_states(runner: Any) -> tuple[Any, Any] | None:
    """Model Runner V2's request slots and sampling states, or None under the V1 runner."""
    states = getattr(runner, "req_states", None)
    sampler = getattr(runner, "sampler", None)
    if states is None or sampler is None or not hasattr(sampler, "sampling_states"):
        return None
    return states, sampler.sampling_states


def rng_state(worker: Any) -> dict[str, Any]:
    """This rank's sampling RNG: numpy's global MT19937 (key, position, the cached Gaussian), the device's CUDA
    generator and torch's CPU generator (hex), Python's (version, words, cached Gaussian), and under V2 each request's
    seed and whether the request set it (``seeds``: request id -> [seed, explicit])."""
    import numpy as np
    import torch

    kind, key, pos, has_gauss, gauss = np.random.get_state()
    out: dict[str, Any] = {"runner": "v1", "numpy": [kind, [int(x) for x in key], int(pos), int(has_gauss), float(gauss)]}
    ver, words, pg = random.getstate()
    out["python"] = [ver, list(words), pg]
    out["torch_cpu"] = bytes(torch.get_rng_state().tolist()).hex()
    if torch.cuda.is_available():
        out["cuda"] = bytes(torch.cuda.get_rng_state().tolist()).hex()
    got = _seed_states(runner_of(worker))
    if got is not None:
        states, sampling = got
        out["runner"] = "v2"
        out["seeds"] = {rid: [int(sampling.seeds.np[i]), bool(sampling.seeds_set[i])]
                        for rid, i in sorted(states.req_id_to_index.items())}
    return out


def rng_digest(rng: Mapping[str, Any]) -> str:
    """SHA-512 of an `rng_state`, canonically serialized."""
    return hashlib.sha512(json.dumps(rng, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class SeedHook:
    """Wraps a V2 runner's ``sampling_states.add_request`` until every restored request is added again: a request in
    ``seeds`` gets its served seed and leaves numpy's global generator where it was (the fresh worker would otherwise
    draw it a new seed, and shift every later request's); any other request draws as vLLM draws. Uninstalls itself once
    ``seeds`` is empty."""

    def __init__(self, runner: Any, seeds: Mapping[str, int]):
        got = _seed_states(runner)
        if got is None:
            raise ValueError("tp_state.SeedHook: the engine runs the V1 model runner, whose per-request generators this "
                             "restore doesn't carry; set VLLM_USE_V2_MODEL_RUNNER=1 on the served and the restored engine")
        self.states, self.sampling = got
        self.pending = dict(seeds)
        self.orig: Callable[..., Any] = self.sampling.add_request
        self.sampling.add_request = self.add_request

    def add_request(self, req_idx: int, sampling_params: Any) -> None:
        import numpy as np

        rid = self.states.index_to_req_id.get(req_idx)
        if rid not in self.pending:
            return self.orig(req_idx, sampling_params)
        st = np.random.get_state()
        self.orig(req_idx, sampling_params)
        np.random.set_state(st)
        self.sampling.seeds.np[req_idx] = self.pending.pop(rid)
        if not self.pending:
            self.uninstall()
        return None

    def uninstall(self) -> None:
        if self.sampling.add_request == self.add_request:
            self.sampling.add_request = self.orig


def set_rng(worker: Any, rng: Mapping[str, Any], seeds: bool = False) -> dict[str, Any]:
    """Put an `rng_state` back on this rank, and with ``seeds`` (a restore that adds its requests to the fresh worker
    again, rather than restoring the runner's rows) install a `SeedHook` for their V2 seeds. Returns the state's digest
    as this rank now has it (the seeds excluded)."""
    import numpy as np
    import torch

    kind, key, pos, has_gauss, gauss = rng["numpy"]
    np.random.set_state((kind, np.asarray(key, dtype=np.uint32), pos, has_gauss, gauss))
    ver, words, pg = rng["python"]
    random.setstate((ver, tuple(words), pg))
    torch.set_rng_state(torch.tensor(list(bytes.fromhex(rng["torch_cpu"])), dtype=torch.uint8))
    if "cuda" in rng and torch.cuda.is_available():
        torch.cuda.set_rng_state(torch.tensor(list(bytes.fromhex(rng["cuda"])), dtype=torch.uint8))
    runner = runner_of(worker)
    if seeds and rng.get("seeds"):
        runner._verity_seed_hook = SeedHook(runner, {rid: s for rid, (s, _) in rng["seeds"].items()})
    now = {k: v for k, v in rng_state(worker).items() if k != "seeds"}
    return {"rank": rank_record(worker)["rank"], "rng_digest": rng_digest(now)}


def first_difference(a: bytes, b: bytes, layers: Sequence[Mapping[str, Any]], ids: Sequence[int]) -> dict[str, Any] | None:
    """The first (layer, block) whose digests differ between two `digest` files over the same layers and ``ids``, or
    None if they are equal."""
    n = len(ids) * BLOCK_DIGEST
    if len(a) != len(b) or len(a) != n * len(layers):
        raise ValueError(f"tp_state.first_difference: digest files of {len(a)} and {len(b)} bytes, expected "
                         f"{n * len(layers)} ({len(layers)} layers of {len(ids)} blocks)")
    for li, spec in enumerate(layers):
        x, y = a[li * n:(li + 1) * n], b[li * n:(li + 1) * n]
        if x != y:
            differ = [i for i in range(len(ids))
                      if x[i * BLOCK_DIGEST:(i + 1) * BLOCK_DIGEST] != y[i * BLOCK_DIGEST:(i + 1) * BLOCK_DIGEST]]
            return {"layer": li, "name": spec["name"], "block": int(ids[differ[0]]), "index": differ[0],
                    "blocks_differing": len(differ)}
    return None


def differing_blocks(a: bytes, b: bytes, layers: Sequence[Mapping[str, Any]], ids: Sequence[int]) -> list[int]:
    """The blocks of ``ids`` whose digests differ in any layer between two `digest` files, ascending."""
    n = len(ids) * BLOCK_DIGEST
    if len(a) != len(b) or len(a) != n * len(layers):
        raise ValueError(f"tp_state.differing_blocks: digest files of {len(a)} and {len(b)} bytes, expected "
                         f"{n * len(layers)} ({len(layers)} layers of {len(ids)} blocks)")
    got: set[int] = set()
    for li in range(len(layers)):
        x, y = a[li * n:(li + 1) * n], b[li * n:(li + 1) * n]
        if x != y:
            got.update(int(ids[i]) for i in range(len(ids))
                       if x[i * BLOCK_DIGEST:(i + 1) * BLOCK_DIGEST] != y[i * BLOCK_DIGEST:(i + 1) * BLOCK_DIGEST])
    return sorted(got)


# ------------------------------------------------------------------- pages for a replay that re-adds its requests

def dump_dir(edge_dir: str | Path, rank: int) -> Path:
    """An edge's dump for ``rank``: its own when the edge kept every rank's (they differed), else rank 0's."""
    own = Path(edge_dir) / f"kv.rank{rank}"
    return own if own.is_dir() else Path(edge_dir) / "kv.rank0"


def _dumped(d: Path, mine: Sequence[Mapping[str, Any]] | None = None) -> tuple[dict[int, int], list[Mapping[str, Any]]]:
    """A dump's block positions (block id -> row) and layers, refused unless its layers are ``mine``."""
    meta = json.loads((d / "layers.json").read_text())
    if mine is not None and list(mine) != meta["layers"]:
        raise ValueError(f"tp_state: the dump at {d} has {len(meta['layers'])} layers and this engine {len(mine)}, or "
                         "their names, dtypes or block shapes differ: replay in an engine built from the served engine's header")
    return {int(b): i for i, b in enumerate(meta["ids"])}, meta["layers"]


def _rows(d: Path, i: int, spec: Mapping[str, Any], n: int) -> Any:
    import numpy as np

    return np.memmap(d / f"{i:03d}.bin", dtype=np.uint8, mode="r", shape=(n, spec["block_bytes"]))


def _sync() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def copy_blocks(worker: Any, edge_dir: str, pairs: Sequence[Sequence[int]]) -> dict[str, Any]:
    """Blocks of an edge's `dump` into this rank's layers at other block ids: each (dumped id, fresh id) of ``pairs``, as
    a replay that re-adds the edge's requests loads their pages. Refuses a dump whose layers aren't this engine's or that
    lacks a block. The device has finished every step before, and finishes the copy before the call returns."""
    import numpy as np
    import torch

    rank = rank_record(worker)["rank"]
    d = dump_dir(edge_dir, rank)
    layers = kv_layers(runner_of(worker))
    pos, specs = _dumped(d, layer_spec(layers))
    missing = [int(s) for s, _ in pairs if int(s) not in pos]
    if missing:
        raise ValueError(f"tp_state.copy_blocks: blocks {missing[:8]} aren't in the dump at {d} ({len(missing)} missing)")
    if not pairs:
        return {"rank": rank, "blocks": 0}
    sel = np.asarray([pos[int(s)] for s, _ in pairs], dtype=np.int64)
    _sync()
    for i, ((_, t), spec) in enumerate(zip(layers, specs)):
        u8 = t.view(torch.uint8)
        rows = np.ascontiguousarray(_rows(d, i, spec, len(pos))[sel])
        dst = torch.as_tensor([int(n) for _, n in pairs], dtype=torch.long, device=t.device)
        u8.index_copy_(0, dst, torch.from_numpy(rows).to(t.device).view(len(pairs), *u8.shape[1:]))
    _sync()
    return {"rank": rank, "blocks": len(pairs)}


def fill_blocks(worker: Any, ids: Sequence[int], byte: int) -> int:
    """Every layer's blocks ``ids`` set to ``byte`` in every byte (0, or `FILL`: NaN in bf16 and in e4m3), once the device
    has finished every step before, and finished before the call returns: a replay's batch-mate K/V."""
    import torch

    if not ids:
        return 0
    _sync()
    for _, t in kv_layers(runner_of(worker)):
        t.view(torch.uint8).index_fill_(0, torch.as_tensor([int(b) for b in ids], dtype=torch.long, device=t.device), int(byte))
    _sync()
    return len(ids)


def _roots(spans: Mapping[str, Sequence[int]], layers: Iterable[Callable[[list[int]], Any]]) -> dict[str, str]:
    """Each request's SHA-512 over its blocks' rows, layer by layer: ``layers`` gives each layer's rows of a block list."""
    import numpy as np

    order = sorted(spans)
    ids = [int(b) for r in order for b in spans[r]]
    hs = {r: hashlib.sha512() for r in order}
    if not ids:
        return {r: h.hexdigest() for r, h in hs.items()}
    for rows_of in layers:
        got, at = rows_of(ids), 0
        for r in order:
            k = len(spans[r])
            hs[r].update(np.ascontiguousarray(got[at:at + k]).tobytes())
            at += k
    return {r: h.hexdigest() for r, h in hs.items()}


def block_roots(worker: Any, spans: Mapping[str, Sequence[int]]) -> dict[str, Any]:
    """Each request's root over whole blocks: SHA-512 over every layer's rows of its blocks (``spans``: request -> block
    ids), layer by layer, on this rank; `dump_roots` reads the same from a dump."""

    _sync()
    rows = [lambda ids, t=t: _host_bytes(t, ids) for _, t in kv_layers(runner_of(worker))]
    return {"rank": rank_record(worker)["rank"], "roots": _roots(spans, rows)}


def dump_roots(edge_dir: str | Path, spans: Mapping[str, Sequence[int]], rank: int = 0) -> dict[str, str]:
    """`block_roots` over an edge's dump for ``rank`` (served block ids)."""
    d = dump_dir(edge_dir, rank)
    pos, specs = _dumped(d)
    missing = [int(b) for v in spans.values() for b in v if int(b) not in pos]
    if missing:
        raise ValueError(f"tp_state.dump_roots: blocks {missing[:8]} aren't in the dump at {d} ({len(missing)} missing)")
    rows = [lambda ids, i=i, s=s: _rows(d, i, s, len(pos))[[pos[b] for b in ids]] for i, s in enumerate(specs)]
    return _roots(spans, rows)
