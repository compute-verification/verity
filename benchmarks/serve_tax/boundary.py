"""A replay block's boundary state: what a vLLM V1 engine core (vLLM d9105ea80, Model Runner V2) holds between two steps,
taken at an edge and put back into a fresh engine core of the same configuration, so that the steps after the edge can be
replayed from it.

At an edge the engine loop drains first (`Edges.step`): the batch queue's batches finish and their outputs are processed,
and nothing is scheduled meanwhile, so nothing is on the GPU and the host has every token sampled so far. `take` then reads:

- the KV pool (`KvPool`): every storage under the model runner's KV cache, copied to pinned host memory and hashed where
  it lies by the row-table kernels (`row_tables_device.step_roots`, one table a layer and one row a block, so the root
  commits each block id's bytes), with a per-block fingerprint that only locates a difference;
- the scheduler: each `Request` whole (prompt, outputs so far, ``num_computed_tokens``, sampling parameters, block
  hashes, async scheduling's placeholders and lag counters), pickled with its block hasher named rather than held; and
  every attribute of the scheduler, its KV cache manager, coordinator and per-group managers that is plain data or
  names requests and blocks (`attrs`: the running and waiting queues in order, the finished, preempted and
  last-scheduled ids, the step counter, each request's block ids);
- the block pool (`pool_state`): each block's reference count and hash, the free queue's order, the prefix cache's
  hash map and each block's cached hashes;
- the model runner (`runner_leaves`): the per-slot request state, block tables, sampling state and the rest of its
  state objects' tensors and plain values;
- the process's random generators (`rng_state`), which unseeded requests draw from.

`restore` puts each back into a fresh engine core. `dump` renders the live state, and more (every attribute the walk
reaches, captured or not), as comparable leaves, so that what a restore missed shows in `diff`. `save_edge` and
`load_edge` do the same through files at any tensor-parallel degree: the engine core's part in one file, and each rank's
runner rows and generators in its own (`save_rank`, `load_rank`, run on every rank by ``collective_rpc``). At an edge of a
tensor-parallel engine each rank takes its own part where it lies (`rank_pool` once, then `rank_take` at each edge): its
KV pool hashed and copied, and its runner rows and generators, kept in the rank's process.
"""
from __future__ import annotations

import copy
import dataclasses
import enum
import hashlib
import io
import json
import pickle
import re
import time
from collections import Counter, deque
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, NamedTuple

SCHEMA = "verity-bench/serve-tax/boundary/v0"
PRIMS = (type(None), bool, int, float, str, bytes)


class Ref(NamedTuple):
    """An object the state names rather than holds: a KV block by id (``B``) or a request by id (``R``)."""
    kind: str
    key: Any


class Unencodable(TypeError):
    pass


def ref_of(x: Any) -> Ref | None:
    name = type(x).__name__
    if name == "KVCacheBlock":
        return Ref("B", x.block_id)
    if name == "Request" and hasattr(x, "request_id"):
        return Ref("R", x.request_id)
    return None


def encode(x: Any, ref: Callable[[Any], Ref | None] = ref_of) -> Any:
    """``x`` as plain data, each request and block it holds as a `Ref`; a set as its elements in iteration order."""
    r = ref(x)
    if r is not None:
        return r
    if isinstance(x, PRIMS) or isinstance(x, enum.Enum):
        return x
    if isinstance(x, dict):
        return {encode(k, ref): encode(v, ref) for k, v in x.items()}
    if isinstance(x, tuple):
        return tuple(encode(v, ref) for v in x)
    if isinstance(x, (list, deque, set, frozenset)):
        return [encode(v, ref) for v in x]
    raise Unencodable(type(x).__name__)


def decode(x: Any, deref: Callable[[Ref], Any]) -> Any:
    if isinstance(x, Ref):
        return deref(x)
    if isinstance(x, dict):
        return {decode(k, deref): decode(v, deref) for k, v in x.items()}
    if isinstance(x, tuple):
        return tuple(decode(v, deref) for v in x)
    if isinstance(x, list):
        return [decode(v, deref) for v in x]
    return x


def put(obj: Any, name: str, value: Any) -> None:
    """Set ``obj.name`` to ``value``, into the container already there where there is one (it keeps its type and
    anything else holding it)."""
    cur = getattr(obj, name, None)
    if isinstance(cur, dict) and isinstance(value, dict):
        cur.clear()
        cur.update(value)
    elif isinstance(cur, deque) and isinstance(value, list):
        cur.clear()
        cur.extend(value)
    elif isinstance(cur, list) and isinstance(value, list):
        cur[:] = value
    elif isinstance(cur, set) and isinstance(value, list):
        cur.clear()
        cur.update(value)
    else:
        setattr(obj, name, value)


def attrs(obj: Any, skip: Iterable[str] = ()) -> tuple[dict[str, Any], dict[str, str]]:
    """Every attribute of ``obj`` that `encode` takes, and the type of each it doesn't."""
    out, skipped = {}, {}
    for name, v in vars(obj).items():
        if name in skip or name.startswith(HARNESS):
            continue
        try:
            out[name] = encode(v)
        except Unencodable:
            skipped[name] = type(v).__name__
    return out, skipped


def put_attrs(obj: Any, state: dict[str, Any], deref: Callable[[Ref], Any]) -> None:
    for name, v in state.items():
        put(obj, name, decode(v, deref))


# ------------------------------------------------------------------------------------------------------- the requests

class _Pickler(pickle.Pickler):
    def __init__(self, f: Any, named: dict[int, str]) -> None:
        super().__init__(f, protocol=pickle.HIGHEST_PROTOCOL)
        self.named = named

    def persistent_id(self, obj: Any) -> str | None:
        return self.named.get(id(obj))


class _Unpickler(pickle.Unpickler):
    def __init__(self, f: Any, named: dict[str, Any]) -> None:
        super().__init__(f)
        self.named = named

    def persistent_load(self, pid: str) -> Any:
        return self.named[pid]


def pickle_requests(requests: Iterable[Any], named: dict[str, Any]) -> bytes:
    """The requests, whole, in order; each object of ``named`` (the engine's block hasher) by its name."""
    buf = io.BytesIO()
    _Pickler(buf, {id(v): k for k, v in named.items() if v is not None}).dump(list(requests))
    return buf.getvalue()


def unpickle_requests(data: bytes, named: dict[str, Any]) -> list[Any]:
    return _Unpickler(io.BytesIO(data), named).load()


# ------------------------------------------------------------------------------------------------------ the block pool

def free_order(fq: Any) -> list[int]:
    out, b = [], fq.fake_free_list_head.next_free_block
    while b is not fq.fake_free_list_tail:
        out.append(b.block_id)
        b = b.next_free_block
    return out


POOL_OWN = ("blocks", "free_block_queue", "cached_block_hash_to_block", "cached_block_hashes_by_block", "null_block")


def pool_state(pool: Any) -> dict[str, Any]:
    blocks = pool.blocks
    cache = {}
    for k, v in pool.cached_block_hash_to_block._cache.items():
        cache[k] = v.block_id if hasattr(v, "block_id") else {i: b.block_id for i, b in v.items()}
    own, skipped = attrs(pool, POOL_OWN)
    return {"num_blocks": len(blocks), "null_block": pool.null_block.block_id,
            "ref_cnt": [b.ref_cnt for b in blocks], "hash": [b._block_hash for b in blocks],
            "hash_tokens": [getattr(b, "_block_hash_num_tokens", None) for b in blocks],
            "is_null": [b.block_id for b in blocks if b.is_null], "free": free_order(pool.free_block_queue),
            "num_free": pool.free_block_queue.num_free_blocks, "cache": cache,
            "hashes_by_block": {i: list(s) for i, s in pool.cached_block_hashes_by_block.items()},
            "attrs": own, "skipped": skipped}


def put_pool(pool: Any, st: dict[str, Any]) -> None:
    blocks = pool.blocks
    if len(blocks) != st["num_blocks"] or pool.null_block.block_id != st["null_block"]:
        raise ValueError(f"a pool of {len(blocks)} blocks with null block {pool.null_block.block_id}, not "
                         f"{st['num_blocks']} with {st['null_block']}: not the served configuration")
    if len(st["free"]) != st["num_free"]:
        raise ValueError(f"the free queue holds {len(st['free'])} blocks but counts {st['num_free']}")
    for b, r, h, n in zip(blocks, st["ref_cnt"], st["hash"], st["hash_tokens"]):
        b.ref_cnt, b._block_hash, b.prev_free_block, b.next_free_block, b.is_null = r, h, None, None, False
        if hasattr(b, "_block_hash_num_tokens"):
            b._block_hash_num_tokens = n
    for i in st["is_null"]:
        blocks[i].is_null = True
    fq = pool.free_block_queue
    prev = fq.fake_free_list_head
    for i in st["free"]:
        b = blocks[i]
        prev.next_free_block, b.prev_free_block, prev = b, prev, b
    prev.next_free_block, fq.fake_free_list_tail.prev_free_block = fq.fake_free_list_tail, prev
    fq.num_free_blocks = st["num_free"]
    cache = pool.cached_block_hash_to_block._cache
    cache.clear()
    for k, v in st["cache"].items():
        cache[k] = blocks[v] if isinstance(v, int) else {i: blocks[x] for i, x in v.items()}
    by = pool.cached_block_hashes_by_block
    by.clear()
    by.update({i: set(s) for i, s in st["hashes_by_block"].items()})
    put_attrs(pool, st["attrs"], lambda r: blocks[r.key])


def pool_diff(a: dict[str, Any], b: dict[str, Any], limit: int = 20) -> dict[str, Any]:
    """Where two `pool_state`s differ: per field, how many blocks (or entries) and the first few."""
    out: dict[str, Any] = {}
    for f in ("ref_cnt", "hash", "hash_tokens"):
        d = [i for i, (x, y) in enumerate(zip(a[f], b[f])) if x != y]
        if d or len(a[f]) != len(b[f]):
            out[f] = {"blocks": len(d), "first": d[:limit]}
    if a["free"] != b["free"]:
        i = next((i for i, (x, y) in enumerate(zip(a["free"], b["free"])) if x != y), min(len(a["free"]), len(b["free"])))
        out["free"] = {"len": [len(a["free"]), len(b["free"])], "first_position": i}
    for f in ("cache", "hashes_by_block"):
        if a[f] != b[f]:
            ka, kb = set(a[f]), set(b[f])
            out[f] = {"only_first": len(ka - kb), "only_second": len(kb - ka),
                      "values_differ": sum(a[f][k] != b[f][k] for k in ka & kb),
                      "order_differs": list(a[f]) != list(b[f])}
    for f in ("num_free", "null_block", "is_null", "attrs"):
        if a[f] != b[f]:
            out[f] = [a[f], b[f]] if f != "attrs" else sorted(k for k in set(a[f]) | set(b[f]) if a[f].get(k) != b[f].get(k))
    return out


# ------------------------------------------------------------------------------------------------------ the runner

#: the model runner's (Model Runner V2's) state objects the walk starts from, where it has them
RUNNER_ROOTS = ("req_states", "block_tables", "sampler", "input_buffers", "prompt_logprobs_worker",
                "structured_outputs_worker", "lora_state", "model_state", "draft_tokens_handler", "execute_model_state",
                "encoder_cache", "intermediate_tensors")
#: buffer types whose content is the state (`snap` reads it), and pools whose buffers are only staging
LEAF_TYPES = ("UvaBackedTensor", "StagedWriteTensor", "UvaBuffer")
OPAQUE_TYPES = ("UvaBufferPool", "FusedStagedWriter")
WALK_MODULE = "vllm.v1.worker.gpu"


def is_plain(x: Any) -> bool:
    if isinstance(x, PRIMS) or isinstance(x, enum.Enum):
        return True
    if isinstance(x, dict):
        return all(is_plain(k) and is_plain(v) for k, v in x.items())
    if isinstance(x, (list, tuple, deque, set, frozenset)):
        return all(is_plain(v) for v in x)
    return False


def kind_of(x: Any) -> str | None:
    name = type(x).__name__
    if name in LEAF_TYPES:
        return name
    if hasattr(x, "untyped_storage") and hasattr(x, "dtype"):
        return "tensor"
    if type(x).__module__ == "numpy" and name == "ndarray":
        return "array"
    if is_plain(x):
        return "plain"
    return None


def runner_leaves(runner: Any, roots: Iterable[str] = RUNNER_ROOTS, max_depth: int = 6
                  ) -> tuple[dict[str, tuple[Any, Any, Any]], dict[str, str]]:
    """The runner's state as leaves: path -> (holder, key, value) for each tensor, array, buffer and plain value its state
    objects hold (objects of ``WALK_MODULE``, followed to ``max_depth``), and the path and type of each value it passes
    by. A name with ``ptr`` in it holds device addresses, which belong to the process: passed by."""
    leaves: dict[str, tuple[Any, Any, Any]] = {}
    skipped: dict[str, str] = {}
    seen: set[int] = set()

    def visit(holder: Any, key: Any, v: Any, path: str, depth: int) -> None:
        k = kind_of(v)
        if k is not None:
            leaves[path] = (holder, key, v)
        elif isinstance(v, (list, tuple)) and v and all(kind_of(x) not in (None, "plain") for x in v):
            for i, x in enumerate(v):
                visit(v, i, x, f"{path}[{i}]", depth)
        elif (type(v).__module__.startswith(WALK_MODULE) and type(v).__name__ not in OPAQUE_TYPES and depth < max_depth
              and hasattr(v, "__dict__")):
            if id(v) in seen:
                return
            seen.add(id(v))
            for name, x in vars(v).items():
                if "ptr" in name:
                    skipped[f"{path}.{name}"] = "address"
                    continue
                visit(v, name, x, f"{path}.{name}", depth + 1)
        else:
            skipped[path] = type(v).__name__

    for root in roots:
        if hasattr(runner, root):
            visit(runner, root, getattr(runner, root), root, 0)
    return leaves, skipped


def snap(kind: str, v: Any) -> Any:
    """A leaf's content, copied to the host."""
    if kind == "tensor":
        return v.detach().to("cpu", copy=True)
    if kind == "array":
        return v.copy()
    if kind in ("UvaBackedTensor", "UvaBuffer"):
        return v.np.copy()
    if kind == "StagedWriteTensor":
        pending = getattr(v, "_staged_write_indices", None)
        if pending:
            raise RuntimeError(f"a StagedWriteTensor with {len(pending)} staged writes not applied: not between steps")
        return v.gpu.detach().to("cpu", copy=True)
    return copy.deepcopy(v)


def unsnap(kind: str, holder: Any, key: Any, cur: Any, saved: Any) -> None:
    if kind == "tensor":
        cur.copy_(saved.to(cur.device))
    elif kind == "array":
        cur[...] = saved
    elif kind == "UvaBackedTensor":
        cur.np[...] = saved
        cur.copy_to_uva()
    elif kind == "UvaBuffer":
        cur.np[...] = saved
    elif kind == "StagedWriteTensor":
        cur.gpu.copy_(saved.to(cur.gpu.device))
    elif isinstance(holder, (list, tuple)):
        holder[key] = copy.deepcopy(saved)
    else:
        put(holder, key, copy.deepcopy(saved))


def runner_state(runner: Any) -> dict[str, Any]:
    leaves, skipped = runner_leaves(runner)
    return {"leaves": {p: (kind_of(v), snap(kind_of(v), v)) for p, (_h, _k, v) in leaves.items()}, "skipped": skipped,
            "type": f"{type(runner).__module__}.{type(runner).__name__}"}


def put_runner(runner: Any, st: dict[str, Any]) -> list[str]:
    """Each leaf of ``st`` into the runner's own; the paths ``st`` has that the runner hasn't."""
    leaves, _ = runner_leaves(runner)
    missing = []
    for p, (kind, saved) in st["leaves"].items():
        if p not in leaves:
            missing.append(p)
            continue
        holder, key, cur = leaves[p]
        if kind_of(cur) != kind and kind != "plain":
            raise ValueError(f"{p}: a {kind_of(cur)} here, a {kind} in the boundary state")
        unsnap(kind, holder, key, cur, saved)
    return missing


def nbytes(saved: Any) -> int:
    if hasattr(saved, "untyped_storage"):
        return int(saved.numel() * saved.element_size())
    if hasattr(saved, "nbytes"):
        return int(saved.nbytes)
    return len(pickle.dumps(saved))


# --------------------------------------------------------------------------------------------- the engine core's state

def parts(core: Any) -> tuple[Any, Any, Any, Any]:
    """An engine core's scheduler, KV cache manager, block pool and model runner (the uniprocess executor's)."""
    sched = core.scheduler
    kvm = sched.kv_cache_manager
    runner = core.model_executor.driver_worker.worker.model_runner
    return sched, kvm, kvm.block_pool, runner


#: objects of the KV cache manager's that `capture` takes apart itself, or that are configuration
KVM_SKIP = ("block_pool", "coordinator", "kv_cache_config")
COORD_SKIP = ("block_pool", "single_type_managers", "kv_cache_config")
MANAGER_SKIP = ("block_pool", "kv_cache_spec", "_null_block")


def capture(core: Any, runner: bool = True) -> dict[str, Any]:
    """The boundary state of a drained engine core (nothing in its batch queue); with ``runner`` false, without the
    model runner's rows and the process's generators, which `save_rank` takes on each rank."""
    if getattr(core, "batch_queue", None):
        raise RuntimeError(f"{len(core.batch_queue)} batches in flight: drain before the boundary state is taken")
    sched = core.scheduler
    kvm = sched.kv_cache_manager
    coord = kvm.coordinator
    sa, ss = attrs(sched)
    ka, ks = attrs(kvm, KVM_SKIP)
    ca, cs = attrs(coord, COORD_SKIP)
    managers = [attrs(m, MANAGER_SKIP) for m in coord.single_type_managers]
    st = {"schema": SCHEMA, "requests": pickle_requests(sched.requests.values(), {"block_hasher": core.request_block_hasher}),
          "scheduler": sa, "kv_manager": ka, "coordinator": ca, "managers": [m[0] for m in managers],
          "pool": pool_state(kvm.block_pool),
          "skipped": {"scheduler": ss, "kv_manager": ks, "coordinator": cs, "managers": [m[1] for m in managers]}}
    if runner:
        st["runner"] = runner_state(parts(core)[3])
        st["rng"] = rng_state()
    return st


def rng_state() -> dict[str, Any]:
    """The process's random generators a step can draw from: numpy's global one (Model Runner V2 draws an unseeded
    request's seed from it when the request reaches the runner), Python's, torch's and each CUDA device's (FlashInfer's
    sampler, which an unseeded batch takes, draws from the current device's)."""
    import importlib.util
    import random

    import numpy as np

    out = {"python": random.getstate(), "numpy": np.random.get_state()}
    if importlib.util.find_spec("torch") is None:
        return out
    import torch

    out["torch"] = torch.get_rng_state()
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        out["cuda"] = torch.cuda.get_rng_state_all()
    return out


def set_rng(st: dict[str, Any]) -> None:
    import random

    import numpy as np

    random.setstate(st["python"])
    np.random.set_state(st["numpy"])
    if "torch" in st:
        import torch

        torch.set_rng_state(st["torch"])
        if "cuda" in st:
            torch.cuda.set_rng_state_all(st["cuda"])


def restore(core: Any, st: dict[str, Any]) -> list[str]:
    """``st`` (`capture`'s) into a fresh engine core of the served configuration; the runner paths it couldn't place
    (none when ``st`` holds no runner, whose rows `load_rank` puts back on each rank)."""
    if getattr(core, "batch_queue", None):
        raise RuntimeError("a fresh engine core has nothing in flight")
    sched = core.scheduler
    kvm = sched.kv_cache_manager
    pool = kvm.block_pool
    if sched.requests:
        raise RuntimeError(f"the engine core holds {len(sched.requests)} requests already: not fresh")
    reqs = {r.request_id: r for r in unpickle_requests(st["requests"], {"block_hasher": core.request_block_hasher})}
    blocks = pool.blocks

    def deref(r: Ref) -> Any:
        return blocks[r.key] if r.kind == "B" else reqs[r.key]

    put_pool(pool, st["pool"])
    put_attrs(sched, st["scheduler"], deref)
    put_attrs(kvm, st["kv_manager"], deref)
    put_attrs(kvm.coordinator, st["coordinator"], deref)
    for m, s in zip(kvm.coordinator.single_type_managers, st["managers"], strict=True):
        put_attrs(m, s, deref)
    if set(sched.requests) != set(reqs):
        raise RuntimeError("the scheduler's requests are not the boundary state's")
    if "runner" not in st:
        return []
    missing = put_runner(parts(core)[3], st["runner"])
    set_rng(st["rng"])
    return missing


def sizes(st: dict[str, Any]) -> dict[str, int]:
    """Each part of a boundary state's size in bytes (pickled; the runner's leaves by their content)."""
    out = {k: len(pickle.dumps(st[k], protocol=pickle.HIGHEST_PROTOCOL))
           for k in ("scheduler", "kv_manager", "coordinator", "managers", "pool", "rng") if k in st}
    out["requests"] = len(st["requests"])
    if "runner" in st:
        out["runner"] = sum(nbytes(s) for _k, s in st["runner"]["leaves"].values())
    return out


# ------------------------------------------------------------------ a drained edge's state as files, at any TP degree

def save_rank(worker: Any, edge_dir: str) -> int:
    """One rank's model-runner rows and random generators to ``edge_dir``/rank-R.pkl; its bytes. ``worker`` is the
    rank's worker or its wrapper, as ``collective_rpc`` hands it, so this runs unchanged on every rank."""
    data = pickle.dumps({"runner": runner_state(worker.model_runner), "rng": rng_state()},
                        protocol=pickle.HIGHEST_PROTOCOL)
    (Path(edge_dir) / f"rank-{worker.rank}.pkl").write_bytes(data)
    return len(data)


def load_rank(worker: Any, edge_dir: str) -> list[str]:
    """`save_rank`'s file into this rank's runner and generators; the runner paths it couldn't place."""
    st = pickle.loads((Path(edge_dir) / f"rank-{worker.rank}.pkl").read_bytes())
    missing = put_runner(worker.model_runner, st["runner"])
    set_rng(st["rng"])
    return missing


def save_edge(core: Any, edge_dir: str | Path) -> dict[str, Any]:
    """A drained engine core's state beyond the KV pool to ``edge_dir``: the scheduler's, KV manager's and block pool's
    (state.pkl) and each rank's (`save_rank`); the bytes of each. The KV pool's bytes are the caller's to take."""
    edge_dir = Path(edge_dir)
    data = pickle.dumps(capture(core, runner=False), protocol=pickle.HIGHEST_PROTOCOL)
    (edge_dir / "state.pkl").write_bytes(data)
    return {"state": len(data), "ranks": core.model_executor.collective_rpc(save_rank, args=(str(edge_dir),))}


def load_edge(core: Any, edge_dir: str | Path) -> dict[str, Any]:
    """`save_edge`'s files into a fresh engine core of the served configuration, before its first step and before or
    after its KV pool's bytes are loaded (neither reads the other)."""
    edge_dir = Path(edge_dir)
    restore(core, pickle.loads((edge_dir / "state.pkl").read_bytes()))
    missing = core.model_executor.collective_rpc(load_rank, args=(str(edge_dir),))
    return {"requests": len(core.scheduler.requests), "runner_paths_missing": missing}


# ------------------------------------------------------------------------------------------------------ dump and diff

#: the modules whose objects `dump` follows into (the rest it renders by type)
DUMP_MODULES = ("vllm.v1.core", "vllm.v1.request", "vllm.sampling_params", "vllm.v1.worker.gpu")
#: the prefix of the attributes this harness hangs on vLLM's objects, which `dump` leaves out
HARNESS = "_verity"


def _digest(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:32]


def render_leaf(x: Any) -> Any:
    if isinstance(x, float):
        return repr(x)
    if isinstance(x, bytes):
        return "b:" + (x.hex() if len(x) <= 64 else "#" + _digest(x))
    if isinstance(x, enum.Enum):
        return f"{type(x).__name__}.{x.name}"
    if isinstance(x, PRIMS):
        return x
    k = kind_of(x)
    if k in ("tensor", "array") or k in LEAF_TYPES:
        s = snap(k, x) if k != "array" else x
        if hasattr(s, "untyped_storage"):
            import torch

            raw = s.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
            shape = tuple(s.shape)
        else:
            raw, shape = s.tobytes(), tuple(s.shape)
        return f"{k}{list(shape)}{getattr(s, 'dtype', '')}#{_digest(raw)}"
    return None


def dump(root: Any, path: str, out: dict[str, Any] | None = None, *, seen: dict[int, str] | None = None, depth: int = 0,
         max_depth: int = 8, wide: int = 512) -> dict[str, Any]:
    """``root`` as path -> rendering: each primitive as itself (a float by its repr), bytes, tensors, arrays and long
    lists by digest, requests and blocks met again by reference, objects of `DUMP_MODULES` followed into and others by
    type. Containers longer than ``wide`` are digested whole."""
    out = {} if out is None else out
    seen = {} if seen is None else seen
    r = render_leaf(root)
    if r is not None:
        out[path] = r
        return out
    if type(root).__name__ == "KVCacheBlock":
        out[path] = f"block:{root.block_id}"
        return out
    if id(root) in seen:
        out[path] = "@" + seen[id(root)]
        return out
    if isinstance(root, (dict, list, tuple, deque, set, frozenset)):
        if isinstance(root, dict):
            items = list(root.items())
        elif isinstance(root, (set, frozenset)):
            # a set's iteration order follows its hash table's history, which no restore carries: compare elements
            items = list(enumerate(sorted(root, key=repr)))
        else:
            items = list(enumerate(root))
        if len(items) > 16 and is_plain(root):
            out[path] = f"{type(root).__name__}{len(items)}#" + _digest(repr(items).encode())
            return out
        if len(items) > wide:
            sub: dict[str, Any] = {}
            for k, v in items:
                dump(v, f"[{k!r}]", sub, seen=dict(seen), depth=depth + 1, max_depth=max_depth, wide=wide)
            out[path] = f"{type(root).__name__}{len(items)}#" + _digest(json.dumps(sub, sort_keys=False, default=str).encode())
            return out
        out[path + "#"] = f"{type(root).__name__}{len(items)}"
        for k, v in items:
            dump(v, f"{path}[{k!r}]", out, seen=seen, depth=depth + 1, max_depth=max_depth, wide=wide)
        return out
    mod = type(root).__module__
    if depth >= max_depth or not mod.startswith(DUMP_MODULES):
        out[path] = f"obj:{type(root).__name__}"
        return out
    seen[id(root)] = path
    if dataclasses.is_dataclass(root):
        names = [f.name for f in dataclasses.fields(root)]
    elif hasattr(type(root), "__struct_fields__"):
        names = list(type(root).__struct_fields__)
    else:
        names = list(vars(root)) if hasattr(root, "__dict__") else list(getattr(type(root), "__slots__", ()))
    for name in names:
        if "ptr" in name or name.startswith(HARNESS):
            continue
        dump(getattr(root, name, None), f"{path}.{name}", out, seen=seen, depth=depth + 1, max_depth=max_depth, wide=wide)
    return out


def dump_core(core: Any) -> dict[str, Any]:
    """An engine core's state as `dump` renders it: its scheduler (requests, queues, KV cache manager and pool), its runner's
    state objects, and its own plain attributes."""
    sched, _kvm, pool, runner = parts(core)
    out = dump(sched, "sched")
    for root in RUNNER_ROOTS:
        if hasattr(runner, root):
            dump(getattr(runner, root), f"runner.{root}", out)
    for name, v in vars(core).items():
        if is_plain(v) and not name.startswith(HARNESS):
            dump(v, f"core.{name}", out)
    st = pool_state(pool)
    for f in ("ref_cnt", "hash", "hash_tokens", "free"):
        out[f"pool.{f}"] = _digest(pickle.dumps(st[f]))
    out["pool.cache"] = _digest(pickle.dumps(list(st["cache"].items())))
    return out


#: path pieces of values that are wall-clock readings or process-local, which a replay sets anew
CLOCK_NAMES = ("arrival_time", "timestamp", "_time", "events", "prefill_stats", "first_token_latency", "start_time")


def pattern(path: str) -> str:
    """A dump path with its keys and indices starred, to count differences by kind."""
    return re.sub(r"\[[^\]]*\]", "[*]", path)


def diff(a: dict[str, Any], b: dict[str, Any], limit: int = 200) -> dict[str, Any]:
    """The paths whose renderings differ between two `dump`s: those in both (``differing``), only in ``a``
    (``only_served``) and only in ``b`` (``only_restored``), each counted by `pattern`, those under a clock reading
    (`CLOCK_NAMES`) apart."""
    paths = sorted(p for p in set(a) | set(b) if a.get(p) != b.get(p))
    clock = [p for p in paths if any(n in p for n in CLOCK_NAMES)]
    rest = [p for p in paths if p not in set(clock)]
    both = [p for p in rest if p in a and p in b]
    only_a = [p for p in rest if p not in b]
    only_b = [p for p in rest if p not in a]
    return {"paths": len(a), "differing": len(both), "only_served": only_a[:limit], "only_restored": only_b[:limit],
            "only_served_n": len(only_a), "only_restored_n": len(only_b), "clock_differing": len(clock),
            "patterns": dict(Counter(map(pattern, rest)).most_common(40)),
            "first": [{"path": p, "served": a.get(p), "restored": b.get(p)} for p in both[:limit]],
            "clock_first": clock[:20]}


# ------------------------------------------------------------------------------------------------------ the KV pool

class KvPool:
    """The model runner's KV cache as it lies: its distinct storages (``storages``, flat uint8, in the order the layers
    first reach them), each layer's view of its storage (``views``), and, where every layer lays its blocks out first and
    contiguous, each layer as rows of int8, one row a block (``rows``)."""

    def __init__(self, kv_caches: Iterable[Any], num_blocks: int) -> None:
        import torch

        layers = [x for t in kv_caches for x in (t if isinstance(t, (list, tuple)) else (t,))]
        found: dict[int, tuple[int, Any]] = {}
        self.views = []
        for x in layers:
            s = x.untyped_storage()
            if s.data_ptr() not in found:
                found[s.data_ptr()] = (len(found), torch.empty(0, dtype=torch.uint8, device=x.device).set_(s))
            self.views.append({"storage": found[s.data_ptr()][0], "offset": x.storage_offset() * x.element_size(),
                               "shape": list(x.shape), "stride": list(x.stride()), "dtype": str(x.dtype)})
        self.storages = [flat for _i, flat in sorted(found.values(), key=lambda t: t[0])]
        self.num_blocks = int(num_blocks)
        ok = all(x.shape[0] == num_blocks and x.is_contiguous() for x in layers)
        self.rows = [x.view(torch.int8).view(num_blocks, -1) for x in layers] if ok else None
        self.nbytes = sum(int(f.numel()) for f in self.storages)

    def layout(self) -> dict[str, Any]:
        return {"storages": [int(f.numel()) for f in self.storages], "views": self.views, "num_blocks": self.num_blocks,
                "block_row_bytes": int(self.rows[0].shape[1]) if self.rows else None}

    def host(self) -> Any:
        import torch

        return torch.empty(self.nbytes, dtype=torch.uint8, pin_memory=True)

    def take(self, host: Any = None, group: int = 9, key: bytes = bytes(32), fingerprint: bool = True) -> dict[str, Any]:
        """The pool's root (the row-table kernels': one table a layer, one row a block, under ``key``; the zero key by
        default, so equal bytes give equal roots in any process), unless ``fingerprint`` is false each block's
        fingerprint (int64 per block and layer, a fixed random linear form of its words: only to locate a difference),
        and, with ``host``, its bytes copied there; all at once on three streams, waited on."""
        import torch

        from verity_vllm.commit import row_tables_device as RTD

        dev = self.storages[0].device
        cur = torch.cuda.current_stream(dev)
        out: dict[str, Any] = {}
        pend = []
        if self.rows:
            ids = [("kv", b) for b in range(self.num_blocks)]
            zero = torch.zeros(self.num_blocks, dtype=torch.float32, device=dev)
            for lo in range(0, len(self.rows), group):
                tabs = {f"model.layers.{i}.kv": (ids, self.rows[i], zero) for i in range(lo, min(lo + group, len(self.rows)))}
                pend.append(RTD.step_roots(bytes(64), 0, tabs, key, timing=True))
        else:
            k = 1 << 20
            for f in self.storages:
                r = f[: f.numel() // k * k].view(torch.int8).view(-1, k)
                for lo in range(0, r.shape[0], 4096):
                    c = r[lo:lo + 4096]
                    pend.append(RTD.step_roots(bytes(64), 0, {"kv": ([("kv", i) for i in range(c.shape[0])], c,
                                                                   c.new_zeros(c.shape[0], dtype=torch.float32))},
                                               key, timing=True))
        copy_ms = None
        if host is not None:
            ready = torch.cuda.Event()
            ready.record(cur)
            side = torch.cuda.Stream(dev)
            side.wait_event(ready)
            marks = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            with torch.cuda.stream(side):
                marks[0].record(side)
                off = 0
                for f in self.storages:
                    host[off:off + f.numel()].copy_(f, non_blocking=True)
                    off += f.numel()
                marks[1].record(side)
        fp = None
        if self.rows and fingerprint:
            gen = torch.Generator().manual_seed(0)
            coef = torch.randint(-(1 << 62), 1 << 62, (self.rows[0].shape[1] // 8,), generator=gen, dtype=torch.int64).to(dev)
            fp = torch.empty(self.num_blocks, len(self.rows), dtype=torch.int64, device=dev)
            for i, r in enumerate(self.rows):
                w = r.view(torch.int64)
                for lo in range(0, self.num_blocks, 8192):
                    fp[lo:lo + 8192, i] = (w[lo:lo + 8192] * coef).sum(1)
        torch.cuda.synchronize(dev)
        if host is not None:
            copy_ms = marks[0].elapsed_time(marks[1])
        roots = [p.result() for p in pend]
        out["root"] = hashlib.sha512(b"".join(r.root for r in roots)).hexdigest()
        out["tables"] = [t[1].hex() for r in roots for t in r.tables]
        out["hash_kernel_ms"] = round(sum(p.kernel_ms() for p in pend), 3)
        out["copy_ms"] = round(copy_ms, 3) if copy_ms is not None else None
        out["bytes"] = self.nbytes
        out["fingerprint"] = fp.cpu().numpy() if fp is not None else None
        return out

    def load(self, path: str, staging: int = 256 << 20) -> int:
        """The pool's bytes from a file `write_host` wrote, through two pinned staging buffers; the bytes read."""
        import os

        import torch

        size = os.path.getsize(path)
        if size != self.nbytes:
            raise ValueError(f"{path}: {size} bytes, the pool {self.nbytes}")
        bufs = [torch.empty(staging, dtype=torch.uint8, pin_memory=True) for _ in range(2)]
        done: list[Any] = [None, None]
        i = 0
        with open(path, "rb", buffering=0) as fh:
            for flat in self.storages:
                n, off = int(flat.numel()), 0
                while off < n:
                    k = min(staging, n - off)
                    if done[i] is not None:
                        done[i].synchronize()
                    view = memoryview(bufs[i].numpy())[:k]
                    got = 0
                    while got < k:
                        r = fh.readinto(view[got:])
                        if not r:
                            raise EOFError(path)
                        got += r
                    flat[off:off + k].copy_(bufs[i][:k], non_blocking=True)
                    done[i] = torch.cuda.Event()
                    done[i].record()
                    off += k
                    i ^= 1
        torch.cuda.synchronize()
        return size


def write_host(host: Any, path: str, chunk: int = 1 << 30) -> int:
    """A pinned host buffer's bytes to ``path``, a chunk at a time."""
    mv = memoryview(host.numpy())
    n = 0
    with open(path, "wb", buffering=0) as fh:
        while n < len(mv):
            n += fh.write(mv[n:n + chunk])
    return n


#: each rank's KV pool, host buffer and last edge's rows (`rank_pool`, `rank_take`), by the worker's rank
RANKS: dict[int, dict[str, Any]] = {}
#: which ranks copy their KV pool to host memory at an edge (``first``: rank 0 alone, for a model whose ranks all hold
#: the same KV bytes, as MLA's replicated latent cache does)
KV_COPY = ("all", "first", "none")


def rank_pool(worker: Any, num_blocks: int, copy: str = "all") -> dict[str, Any]:
    """This rank's `KvPool` over its model runner's KV cache, kept for `rank_take`, with a pinned host buffer if ``copy``
    (`KV_COPY`) names the rank; hashed once here, so that the hash kernels are built before the first edge. What it
    holds and the first hash's seconds. ``worker`` is as ``collective_rpc`` hands it."""
    if copy not in KV_COPY:
        raise ValueError(f"copy is one of {KV_COPY}, not {copy!r}")
    kv = KvPool(worker.model_runner.kv_caches, num_blocks)
    host = kv.host() if copy == "all" or (copy == "first" and worker.rank == 0) else None
    t = time.perf_counter()
    kv.take(fingerprint=False)
    RANKS[worker.rank] = {"kv": kv, "host": host}
    return {"rank": worker.rank, "bytes": kv.nbytes, "storages": len(kv.storages), "layers": len(kv.views),
            "rows": kv.rows is not None, "host_bytes": kv.nbytes if host is not None else 0,
            "build_s": time.perf_counter() - t}


def rank_take(worker: Any, key: bytes) -> dict[str, Any]:
    """At a drained edge, this rank's part of the boundary state where it lies: its KV pool (`rank_pool`'s) hashed under
    ``key`` and copied to its host buffer if it has one, then its runner rows and the process's generators, all kept here
    until the next edge replaces them. The pool's root, the bytes copied and each part's seconds."""
    me = RANKS[worker.rank]
    t0 = time.perf_counter()
    kv = me["kv"].take(me["host"], key=key, fingerprint=False)
    t1 = time.perf_counter()
    me["rows"] = {"runner": runner_state(worker.model_runner), "rng": rng_state()}
    t2 = time.perf_counter()
    return {"rank": worker.rank, "root": kv["root"], "hash_kernel_ms": kv["hash_kernel_ms"], "copy_ms": kv["copy_ms"],
            "copied_bytes": kv["bytes"] if me["host"] is not None else 0, "kv_s": t1 - t0, "runner_s": t2 - t1,
            "runner_bytes": sum(nbytes(s) for _k, s in me["rows"]["runner"]["leaves"].values())}


# --------------------------------------------------------------------------------------------------------- the edges

def drain_one(orig: Callable[[Any], Any], core: Any) -> Any:
    """One call of the engine core's ``step_with_batch_queue`` that schedules nothing: it waits on the oldest batch in
    flight and processes its output."""
    sched = core.scheduler
    own = vars(sched)
    had, prev = "has_requests" in own, own.get("has_requests")
    sched.has_requests = lambda: False
    try:
        return orig(core)
    finally:
        if had:
            sched.has_requests = prev
        else:
            del sched.has_requests


class Edges:
    """The edges of one engine core: ``due(self)`` says whether one is due before the next step; while one is, each call of
    `step` drains one batch in flight instead of scheduling, and once none is left ``take(self, info)`` takes the boundary
    state (``info``: the clock when the edge fell due and when the drain finished, and the drain's calls) before the
    engine core schedules again."""

    def __init__(self, core: Any, due: Callable[[Edges], bool], take: Callable[[Edges, dict], None],
                 clock: Callable[[], float]) -> None:
        self.core, self.due, self.take, self.clock = core, due, take, clock
        self.edge = 0
        self.since: float | None = None
        self.calls = 0

    def step(self, orig: Callable[[Any], Any]) -> Any:
        core = self.core
        if self.due(self):
            if self.since is None:
                self.since, self.calls = self.clock(), 0
            if core.batch_queue:
                self.calls += 1
                return drain_one(orig, core)
            info = {"edge": self.edge, "due_t": self.since, "drained_t": self.clock(), "drain_calls": self.calls}
            self.since = None
            self.take(self, info)
            self.edge += 1
        return orig(core)


def at_times(at_s: list[float], clock: Callable[[], float]) -> tuple[Callable[[Edges], bool], Callable[..., None]]:
    """Edges due ``at_s`` seconds after a start (the second function, called at the first request of the traffic the edges
    fall in, with its load, which it ignores); none before it."""
    t0: list[float] = []

    def due(e: Edges) -> bool:
        return bool(t0) and e.edge < len(at_s) and clock() - t0[0] >= at_s[e.edge]

    def start(load: str | None = None) -> None:
        if not t0:
            t0.append(clock())
    return due, start


def every(period_s: float, clock: Callable[[], float]) -> tuple[Callable[[Edges], bool], Callable[[str | None], None]]:
    """An edge each ``period_s`` seconds of a load's serving: the replay blocks of one length. Each falls ``period_s``
    after the end of the edge before it, or after its load's first request (the second function, called at each request
    with its load); a new load starts the count again, so the idle time between loads brings no edge."""
    st: dict[str, Any] = {"load": None, "t": None, "edge": 0, "started": False}

    def due(e: Edges) -> bool:
        if e.edge != st["edge"]:
            st["edge"] = e.edge
            if not st["started"]:
                st["t"] = clock()
        if st["t"] is None or clock() - st["t"] < period_s:
            return False
        st["started"] = False
        return True

    def start(load: str | None) -> None:
        if st["t"] is None or load != st["load"]:
            st["load"], st["t"], st["started"] = load, clock(), True
    return due, start


def at_steps(steps: list[int], scheduled: Callable[[], int]) -> Callable[[Edges], bool]:
    """Edges due once the engine core has scheduled ``steps[k]`` steps: right before the step with that index."""
    return lambda e: e.edge < len(steps) and scheduled() >= steps[e.edge]


class Busy:
    """The GPU's busy time per step: CUDA events on the current stream at the start of each step's ``execute_model`` and
    at the end of its ``sample_tokens``, read once both are done. Steps count as `StepTrace` counts them (those with
    tokens), from ``first``."""

    def __init__(self, event: Callable[[], Any], first: int = 0) -> None:
        self.event, self.next = event, first
        self.open: list[tuple[int, Any, Any]] = []
        self.done: dict[int, float] = {}
        self.current: tuple[int, Any] | None = None

    def started(self, scheduled_tokens: int) -> None:
        if scheduled_tokens > 0:
            e = self.event()
            e.record()
            self.current = (self.next, e)
            self.next += 1

    def sampled(self) -> None:
        if self.current is not None:
            e = self.event()
            e.record()
            self.open.append((*self.current, e))
            self.current = None
        self.reap()

    def reap(self, wait: bool = False) -> None:
        """The open steps' times, oldest first, up to the first whose end hasn't happened (all of them, waiting, with
        ``wait``): one stream's events complete in order, so the open list stays a few steps long."""
        n = 0
        for step, a, b in self.open:
            if wait:
                b.synchronize()
            elif not b.query():
                break
            self.done[step] = a.elapsed_time(b) / 1e3
            n += 1
        del self.open[:n]

    def total(self, lo: int, hi: int) -> tuple[float, int]:
        """Busy seconds and steps over step indices [lo, hi)."""
        xs = self.steps(lo, hi)
        return sum(xs), len(xs)

    def steps(self, lo: int, hi: int) -> list[float]:
        """Each step's busy seconds over step indices [lo, hi), in step order."""
        return [self.done[k] for k in sorted(self.done) if lo <= k < hi]


def busy_compare(served: list[float], replayed: list[float], stall: float = 3.0) -> dict[str, Any]:
    """Two runs' busy seconds over the same steps, step by step: the sums, the per-step ratio's quantiles, and the steps
    where the replay was over ``stall`` times as busy (a kernel compiled on first use, say) apart."""
    n = min(len(served), len(replayed))
    pairs = list(zip(served[:n], replayed[:n], strict=True))
    ratios = sorted(r / s for s, r in pairs if s > 0)
    stalls = [(s, r) for s, r in pairs if r > stall * s]

    def q(p: float) -> float | None:
        return ratios[min(len(ratios) - 1, int(p * len(ratios)))] if ratios else None

    return {"steps": n, "served_s": sum(s for s, _ in pairs), "replayed_s": sum(r for _, r in pairs),
            "ratio_p10": q(0.1), "ratio_p50": q(0.5), "ratio_p90": q(0.9), "stall_steps": len(stalls),
            "stall_excess_s": sum(r - s for s, r in stalls),
            "replayed_s_without_stalls": sum(r for s, r in pairs if r <= stall * s),
            "served_s_without_stalls": sum(s for s, r in pairs if r <= stall * s)}


def busy_runner(runner_cls: Any, busy_of: Callable[[Any], Busy | None]) -> None:
    """Wrap the runner class's ``execute_model`` and ``sample_tokens`` with `Busy`'s marks, for the runners
    ``busy_of`` gives a `Busy`; none while a CUDA graph is being captured."""
    import torch

    execute, sample = runner_cls.execute_model, runner_cls.sample_tokens

    def execute_model(self: Any, so: Any, *a: Any, **kw: Any) -> Any:
        b = busy_of(self)
        if b is not None and not torch.cuda.is_current_stream_capturing():
            b.started(int(getattr(so, "total_num_scheduled_tokens", 0) or 0))
        return execute(self, so, *a, **kw)

    def sample_tokens(self: Any, *a: Any, **kw: Any) -> Any:
        out = sample(self, *a, **kw)
        b = busy_of(self)
        if b is not None and not torch.cuda.is_current_stream_capturing():
            b.sampled()
        return out
    runner_cls.execute_model, runner_cls.sample_tokens = execute_model, sample_tokens


# ---------------------------------------------------------------------------------------------------------- outputs

def logprob_bits(lp: Any) -> list[int]:
    """The sampled tokens' logprobs of one request's step (`LogprobsLists`), as their float32 bit patterns."""
    import numpy as np

    return np.ascontiguousarray(np.asarray(lp.logprobs, dtype=np.float32)[:, 0]).view(np.uint32).tolist()


def step_outputs(outs: Any) -> dict[str, list]:
    """Each request's tokens sampled in one step's output (``update_from_output``'s), with their logprobs' bits."""
    got: dict[str, list] = {}
    for eco in (outs or {}).values():
        for o in getattr(eco, "outputs", None) or ():
            if o.new_token_ids:
                lp = getattr(o, "new_logprobs", None)
                got[o.request_id] = [list(o.new_token_ids), logprob_bits(lp) if lp is not None else None]
    return got


def outputs_diff(served: dict[int, dict], replayed: dict[int, dict]) -> dict[str, Any]:
    """Step by step, the requests whose tokens or logprob bits differ between two runs' `step_outputs`, and the first step
    where any does."""
    steps = sorted(set(served) | set(replayed))
    tok, lp, first = 0, 0, None
    for s in steps:
        a, b = served.get(s, {}), replayed.get(s, {})
        for rid in sorted(set(a) | set(b)):
            x, y = a.get(rid), b.get(rid)
            if x is None or y is None or x[0] != y[0]:
                tok += 1
            elif x[1] != y[1]:
                lp += 1
            else:
                continue
            if first is None:
                first = {"step": s, "request": rid, "served": x, "replayed": y}
    return {"steps": len(steps), "tokens": sum(len(v[0]) for o in served.values() for v in o.values()),
            "request_steps_tokens_differing": tok, "request_steps_logprobs_differing": lp, "first": first}
