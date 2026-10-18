"""The remote bandwidth session over the node's `Store`, inside the prover's process (the Commit's, or the serve
driver's): `StoreBackend` is the backend `benchmarks/remote_session`'s `ProverSide` drives, the role `device.Device`
and `CpuBackend` play there, and `prove` connects it to a verifier at ``HOST:PORT`` (the row's ``REMOTE_VERIFIER``,
`config.CommitConfig.remote_verifier`).

~~~text
hello         the node, N = the store's blocks (`Store.layout`) and the payload root of every block's payload as the
              store holds it now (`Store.payload`: weights, the KV pool's pages as filler, host filler)
registration  the verifier's; the backend encodes the payloads under its key (`audit.Store`, the driver's reference
              encoding, or kv.cu on the GPU into the same host mirror: `DeviceEncoded`); the hello's layout is each
              region's kind, tier and blocks (`StoreBackend.layout`)
session       the store records its memory map against `Registration.digest` under the session's lifetime cap L
              (`Store.record_memory_map`), one session a store: the map's indices are one session's
pass          before each pass `Serving.advance` runs the engine's next step against the store (its frees, allocs and
              `kv_append`, which closes the step before's memory-map line with its re-keys for the pass's window), turns
              each closed line into the audit's append (committed by its digests alone, `audit.Store.append`) and
              releases (`memory_map.Replay`, the map's).  `ProverSide` sends each append after the root before the
              pass, checks the reply's ``first`` against the append's j and hands its coin c_a to `encode_append`,
              which encodes the blocks under b_j = codec.binding(d_j, j, key, c_a) (`audit.Store.encode`, the CPU
              reference of kv.cu's and paged.cu's per-block offset, or `DeviceEncoded.encode`, which hands b_j to
              kv.cu); then the pass reads every live block in store order; its
              ticket carries the step (`step_of`), which the root's message names, and the head of the appends so
              far, which its seed chains (`audit.chained`: unchanged in a session without appends)
openings      as the driver's reference: the block, its receipt leaf, and its payload's salt, path and d_j
~~~

``engine(s)`` runs engine step s against the store as the binder does (`verity_vllm.store.binder`): a request's pages
freed after the step before (a finished request's) or in this step's schedule (a preempted one's), allocs, then
`kv_append` (or `kv_write_reference`, which writes the KV first) and `end_step` once the step's KV is in place.  A
CPU test passes a scripted engine; with vLLM, `serve` passes ``LLMEngine.step`` and the binder on the engine, with
``passes=False``, numbers the steps.  The Commit's session is `CommitSession`: the binder on its engine from the build,
the session on that engine once its runs are done, its store's encode on the GPU by kv.cu (`KvDevice`).

`benchmarks/remote_session` is not a package: `session_module` loads its ``prover.py`` (and ``verifier.py``, for a
local verifier) from the source tree (`config.REMOTE_SESSION`).
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

from verity.core.protocols.remote_bandwidth import audit, codec as rcodec, memory_map, receipts as rb
from verity_vllm import config
from verity_vllm.store.registry import Store


def address(value: str) -> tuple[str, int]:
    """``HOST:PORT``, the verifier's listening address."""
    host, sep, port = str(value).rpartition(":")
    if not sep or not host or not port.isdigit() or not 0 < int(port) < 1 << 16:
        raise ValueError(f"a remote verifier is HOST:PORT, not {value!r}")
    return host, int(port)


def session_module(name: str, directory: str | Path = config.REMOTE_SESSION) -> ModuleType:
    """``benchmarks/remote_session``'s module ``name`` (``prover``, ``verifier``, ``wire``), loaded once from its file,
    as its own scripts import it."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, Path(directory) / f"{name}.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"no remote session module {name!r} in {directory}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise
    return module


class Serving:
    """The serving node's steps as a session sees them, `ProverSide`'s serving trace in its ``serve`` modes: the
    memory map's closed lines as the audit's appends (`appends`, each an `audit.Append`, and `appended`, their
    indices) and releases (`released`), and `plan`, every live block, registered and appended."""

    def __init__(self, backend: StoreBackend, registration: audit.Registration, lifetime: int) -> None:
        self.backend, self.store, self.lifetime = backend, backend.store, lifetime
        self._lines: list[tuple[Any, list[bytes]]] = []
        self.map = self.store.record_memory_map(backend.path, registration.digest, lifetime,
                                                lambda change, payloads: self._lines.append((change, payloads)))
        self.step, self.steps = -1, 0
        self.live = set(range(registration.blocks))
        self.appends: list[audit.Append] = []
        self.released: list[int] = []
        self.appended: list[int] = []

    @property
    def rekeyed(self) -> int:
        return self.map.rekeyed

    def advance(self, window: int) -> int:
        """The engine's next step s, run against the store up to its pass (in ``window``, the store's open one): each
        line it closed applied as the audit's append and releases.  Returns s: the session's first step takes the
        number the engine gives it (a binder on an engine that served before the session numbers on from there), and
        each later one the next."""
        if not self.store.window_open or self.store.window != window:
            raise RuntimeError(f"the pass is in window {window}, the store's open window is {self.store.window}")
        s = self.step + 1
        self.backend.engine(s)
        if self.map.last < s or (self.steps and self.map.last != s):
            raise RuntimeError(f"the engine's step after {self.step} opens one memory-map line (kv_append({s}))")
        s = self.map.last
        for change, payloads in self._lines:
            if change.count:
                a = self.backend.append(payloads, [self.backend.randbytes(rb.SALT_BYTES) for _ in payloads])
                if a.first != change.first or a.digests != change.digests:
                    raise AssertionError("the store's append is not its memory map's: a written block's bytes changed "
                                         "between its digest and its line's close")
                self.appends.append(a)
                self.appended.extend(range(a.first, a.first + a.count))
            gone = [j for f, c in change.released for j in range(f, f + c)]
            self.released.extend(gone)
            self.live.difference_update(gone)
            self.live.update(range(change.first, change.first + change.count))
        self._lines.clear()
        self.step, self.steps = s, self.steps + 1
        return s

    def plan(self) -> rb.Plan:
        out: list[list[int]] = []
        for j in sorted(self.live):
            if out and out[-1][0] + out[-1][1] == j:
                out[-1][1] += 1
            else:
                out.append([j, 1])
        return rb.Plan(tuple((f, c) for f, c in out), self.map.extent)


class StoreBackend:
    """`ProverSide`'s backend over ``store``, whose engine steps ``engine(s)`` runs (the module's docstring).  Its
    passes are the driver's reference over its `audit.Store` (`receipts.receipter` over each plan's blocks, its tree
    `receipts.tree`), hiding salts from ``randbytes``; ``path`` is the memory map's file (None: kept in memory).  With
    ``device`` (`KvDevice`) the store's blocks are encoded on the GPU (`DeviceEncoded`)."""

    def __init__(self, store: Store, engine: Callable[[int], None], *, randbytes: Callable[[int], bytes] = os.urandom,
                 path: str | Path | None = None, read_bytes_per_s: float = 1e12, device: KvDevice | None = None) -> None:
        self.store, self.engine, self.path, self.randbytes, self.device = store, engine, path, randbytes, device
        self.blocks, self.read_bytes_per_s = store.layout().blocks, read_bytes_per_s
        self.dropped: set[int] = set()
        self.gates: dict = {}
        self.trace: Serving | None = None
        self.previous: bytes | None = None

    def layout(self) -> list[dict]:
        """The hello's public layout: each registered region's kind, tier and blocks, in store order."""
        return [{"kind": r.kind, "tier": r.tier, "first": r.first, "blocks": r.blocks}
                for r in self.store.layout().regions]

    def payload_root(self, node: bytes) -> bytes:
        self.payloads = [self.store.payload(x) for x in range(self.blocks)]
        self.salts = [self.randbytes(rb.SALT_BYTES) for _ in range(self.blocks)]
        return audit.payload_tree(node, self.payloads, self.salts).commitment.root

    def encode(self, registration: audit.Registration) -> dict:
        if any(self.store.payload(x) != m for x, m in enumerate(self.payloads)):
            raise RuntimeError("the store changed between its hello and its registration")
        self.registration = registration
        if self.device is None:
            self.audit = audit.Store(registration, self.payloads, self.salts)
        else:
            self.audit = DeviceEncoded(registration, self.payloads, self.salts, self.device)
            self.gates["device_encode"] = self.audit.gates
        return {"encode_bytes_per_s": 1.0}

    def begin_session(self) -> None:
        if self.trace is not None:
            raise RuntimeError("a store serves one session: its memory map's indices are the session's")

    def serving(self, registration: audit.Registration, lifetime: int) -> Serving:
        self.trace = Serving(self, registration, lifetime)
        return self.trace

    def append(self, payloads, salts) -> audit.Append:
        return self.audit.append(payloads, salts)

    def encode_append(self, first: int, coin: bytes) -> None:
        self.audit.encode(first, coin)

    def calibrate(self) -> dict:
        return {"pass_ns_max": 0, "gap_ns_max": 0, "read_bytes_per_s": self.read_bytes_per_s}

    def arm(self, dropped) -> None:
        self.dropped = {int(j) for j in dropped}

    def begin_window(self, challenge: bytes, previous: bytes | None) -> None:
        """The session's next window, the store's too (the first chained from the registration its map names)."""
        if self.store.window_open:
            self.store.end_window()
        self.store.begin_window(challenge, self.registration.digest if self.store.window < 0 else None)
        self.challenge = challenge
        if previous is not None:
            self.previous = previous

    def launch(self, i: int, after: dict | None, plan: rb.Plan | None = None) -> dict:
        """Pass i of the window over ``plan`` (None: every registered block) at the step the trace last advanced to,
        chained from ``after``'s root (None: the window's previous root) and the head of every append so far
        (`audit.Store.head`, all sent with this pass's root or before it), run when awaited."""
        return {"challenge": self.challenge, "i": i, "after": after, "previous": self.previous, "root": None,
                "plan": plan, "step": None if self.trace is None else self.trace.step, "head": self.audit.head}

    def step_of(self, ticket: dict) -> int | None:
        return ticket["step"]

    def wait(self, ticket: dict):
        after = ticket["after"]
        prev = audit.chained(ticket["previous"] if after is None else after["root"], ticket["head"])
        root, handle = self.run_pass(rb.seed(ticket["challenge"], ticket["i"], prev), ticket["plan"])
        ticket["root"] = root
        return root, handle

    def run_pass(self, seed: bytes, plan: rb.Plan | None = None):
        plan = plan or rb.Plan.whole(self.blocks)
        reg = self.registration
        r = rb.receipter(reg.receipt, seed, reg.rows)
        ys = tuple(r(j, bytes(rcodec.BLOCK_BYTES) if j in self.dropped else self.audit.blocks[j]) for j in plan)
        salts = tuple(self.randbytes(rb.SALT_BYTES) for _ in range(rb.leaves(plan.length)))
        tree = rb.tree(seed, plan, ys, salts)
        return tree.commitment.root, (ys, salts, tree, plan)

    def open_window(self, items) -> list[list[tuple]]:
        return [self.open(handle, positions) for handle, positions in items]

    def open(self, handle, positions: list[int]) -> list[tuple]:
        ys, salts, tree, plan = handle
        out = []
        for p in positions:
            c, j = p // rb.PER_LEAF, plan.block(p)
            leaf = b"".join(ys[rb.PER_LEAF * c:rb.PER_LEAF * (c + 1)])
            out.append((self.audit.blocks[j], leaf, salts[c], tree.open(c).path, *self.audit.committed(j)))
        return out


class KvDevice:
    """The store's encode on the GPU by kv.cu (``benchmarks/word_codec/kv.cu``, a build at the codec's field and mask
    loaded as `kv_bench.Lib`): blocks under the seed's key by ``kv_encode``; an append's under their bound keys and its
    coin by ``kv_encode_bound``, each block's offset b_j(c_a) = `codec.binding`(d_j, j, key, c_a) handed to the device
    by ``paged.offsets`` (`config.VLLM_ENCODED`), so the host computes only d_j and b_j."""

    CHUNK = 1 << 14

    def __init__(self, path: str | Path, codec: rcodec.Codec = rcodec.DEFAULT) -> None:
        import torch

        self.torch, self.codec = torch, codec
        self.paged = session_module("paged", config.VLLM_ENCODED)
        self.lib, self.field = self.paged.KB.Lib(codec.field.name, str(path)), self.paged.K.field(codec.field.name)
        if self.lib.mask != codec.masked or self.lib.kmax < codec.rounds:
            raise ValueError(f"{path} is kv.cu at {self.field.name}, masked={self.lib.mask}, at most {self.lib.kmax} "
                             f"rounds: not {codec.name}")
        if self.paged.K.token_layer_words(self.field) != codec.field.block_words:
            raise ValueError("kv.cu's bound offsets are a token-layer's: build it with a token-layer of one 4 KiB block")

    def set_key(self, key: rcodec.Key) -> None:
        k = self.paged.K.key(self.field.name, key.seed, key.codec.rounds, mask=key.codec.masked)
        if (k.c, k.tau, k.mask) != (key.c, key.tau, key.mask):
            raise RuntimeError("kv.py's key is not the registration's codec key")
        self.paged.KB.set_key(self.lib, k)

    def encode(self, payloads: Sequence[bytes], first: int, offsets: Sequence[int] | None = None) -> list[bytes]:
        """Blocks ``first`` … of ``payloads``: under the seed's key, or each under its bound offset."""
        torch, w, out = self.torch, self.codec.field.block_words, []
        stream = torch.cuda.current_stream().cuda_stream
        for u in range(0, len(payloads), self.CHUNK):
            part = payloads[u:u + self.CHUNK]
            m = torch.frombuffer(bytearray(b"".join(part)), dtype=torch.uint8).cuda()
            e = torch.empty_like(m)
            n, base, k = len(part) * w, (first + u) * w, self.codec.rounds
            if offsets is None:
                self.lib.call("kv_encode", m.data_ptr(), e.data_ptr(), n, base, k, 0, stream)
            else:
                b = self.paged.offsets(torch, list(offsets[u:u + self.CHUNK]), self.field)
                self.lib.call("kv_encode_bound", m.data_ptr(), e.data_ptr(), n, base, k, b.data_ptr(), 0, stream)
            raw = e.cpu().numpy().tobytes()
            out.extend(raw[i * rcodec.BLOCK_BYTES:(i + 1) * rcodec.BLOCK_BYTES] for i in range(len(part)))
        return out


class DeviceEncoded(audit.Store):
    """`audit.Store` with its encodes on a `KvDevice`: the registered blocks at once, an append's once its coin
    arrives.  Before the store takes them, the device's blocks are checked against `codec.encode`: the registration's
    first and last, and each append's first (`gates`)."""

    def __init__(self, registration: audit.Registration, payloads: Sequence[bytes], salts: Sequence[bytes],
                 device: KvDevice) -> None:
        if len(payloads) != registration.blocks:
            raise ValueError(f"the registration is of {registration.blocks} blocks")
        self.registration, self.device = registration, device
        self.payloads, self.salts = tuple(payloads), tuple(salts)
        self.payload_tree = audit.payload_tree(registration.node, self.payloads, self.salts)
        if self.payload_tree.commitment.root != registration.payload_root:
            raise ValueError("these payloads are not the registered ones")
        device.set_key(registration.key)
        encoded = device.encode(self.payloads, 0)
        for j in sorted({0, len(encoded) - 1}):
            self._gate(encoded[j], self.payloads[j], j)
        self.blocks: list[bytes | None] = list(encoded)
        self.appends: list[audit.Append] = []
        self.pending: dict[int, tuple[bytes, ...]] = {}
        self.gates = {"registered": len(encoded), "appends": 0, "blocks_checked": len({0, len(encoded) - 1})}

    def _gate(self, block: bytes, payload: bytes, j: int, coin: bytes | None = None) -> None:
        if block != rcodec.encode(payload, j, self.registration.key, bound=coin is not None, coin=coin):
            raise RuntimeError(f"kv.cu's block {j} is not codec.encode's")

    def encode(self, first: int, coin: bytes) -> None:
        payloads = self.pending.get(first)
        if payloads is None:
            raise ValueError(f"no append from block {first} awaits its coin")
        digests = next(a.digests for a in reversed(self.appends) if a.first == first)
        k = self.registration.key
        encoded = self.device.encode(payloads, first, [rcodec.binding(d, first + u, k, coin) for u, d in enumerate(digests)])
        self._gate(encoded[0], payloads[0], first, coin)
        del self.pending[first]
        self.blocks[first:first + len(encoded)] = encoded
        self.gates["appends"] += 1
        self.gates["blocks_checked"] += 1


class Refused(RuntimeError):
    """The verifier refused the store's registration, answering its hello with ``bye``: a node has one live
    registration (`audit.Registry`), which only the verifier closes.  ``why`` is the verifier's reason."""

    def __init__(self, verifier: str, why: str) -> None:
        super().__init__(f"the verifier at {verifier} refused the store's registration: {why}")
        self.verifier, self.why = verifier, why


def prove(backend: StoreBackend, verifier: str, plan: str = "serve", *, node: bytes = b"verity-vllm/store",
          codec: rcodec.Codec = rcodec.DEFAULT, log: Callable[[str], None] = print, **kw: Any) -> list[dict]:
    """The prover's side of a remote session against the verifier at ``verifier`` (``HOST:PORT``): register the
    store, then run each session of ``plan`` (`ProverSide`'s modes: ``serve``, or ``servedrop:K``, which withholds the
    first K blocks appended) as the verifier starts it.  Returns each session's result; raises `Refused` with the
    verifier's reason if it refuses the registration."""
    host, port = address(verifier)
    prover, wire = session_module("prover"), session_module("wire")
    p = prover.ProverSide(wire.connect(host, port), backend, node, prover.parse_plan(plan), log=log, **kw)
    try:
        p.register(codec=codec)
    except wire.Closed as e:
        if str(e).startswith("refused"):
            raise Refused(verifier, str(e)) from None
        raise
    out = []
    for mode, fraction in p.plan:
        head, _, _ = p.expect("session")
        out.append(p.session(head, mode, fraction))
    return out


def serve(verifier: str, store: Store, step: Callable[[], Any], **kw: Any) -> list[dict]:
    """The Commit's hook (``--remote-verifier HOST:PORT``): `prove` over ``store``, with ``binder.Binder(store,
    passes=False)`` on the engine (the session runs the passes and windows), each pass after one engine step
    ``step()`` (``LLMEngine.step``), which the binder numbers and records."""
    return prove(StoreBackend(store, lambda s: step()), verifier, **kw)


class Workload:
    """A session's engine steps over a workload's requests: before each step, the next requests (in turn, each id
    suffixed with its count) while fewer than ``concurrent`` are in flight, so every step schedules; then
    ``engine.step()`` (`LLMEngine`)."""

    def __init__(self, engine: Any, requests: Sequence[tuple[str, list[int], Any]], concurrent: int = 1) -> None:
        if not requests:
            raise ValueError("a session serves at least one request")
        self.engine, self.requests, self.concurrent = engine, list(requests), max(1, int(concurrent))
        self.added = self.finished = self.steps = 0

    def __call__(self, s: int) -> None:
        while self.engine.get_num_unfinished_requests() < self.concurrent:
            rid, ids, params = self.requests[self.added % len(self.requests)]
            self.engine.add_request(f"{rid}@remote{self.added}", {"prompt_token_ids": list(ids)}, params)
            self.added += 1
        self.finished += sum(1 for o in self.engine.step() if o.finished)
        self.steps += 1


class CommitSession:
    """The Commit's remote bandwidth session (``--remote-verifier HOST:PORT``, `pipeline.commit`), on its engine:

    ~~~text
    engine_args  before the build: one GPU, synchronous scheduling, no prefix caching, bf16 KV (refusing anything else),
                 the KV pool at ``kv_pages`` pages
    install      before the build: ``binder.Binder(store, passes=False)`` on vLLM's model runner, block pool and
                 scheduler, registering weight tensors up to ``weight_blocks`` blocks and the KV pool
    check        after the build: the engine core in this process, scheduling synchronous, the KV cache bf16, the pool
                 registered
    serve        once the Commit's runs are done: the workload's requests on its engine a step a pass (`Workload`), the
                 memory map recorded against the registration's digest beside the Commit (``out``), the store encoded by
                 kv.cu at ``kv_lib`` (`KvDevice`) or, without it, on the CPU; its record, ``remote_session.json``, with
                 the map's agreement with the store (`agreement`)
    ~~~

    A store serves one session, so ``plan`` is one mode: ``serve``, or ``servedrop:K``, the negative that withholds the
    first K blocks appended."""

    def __init__(self, verifier: str, *, plan: str = "serve", kv_pages: int = 32, weight_blocks: int | None = None,
                 kv_lib: str | None = None, out: str | Path | None = None, node: str = "verity-vllm/commit",
                 codec: rcodec.Codec = rcodec.DEFAULT, log: Callable[[str], None] = print) -> None:
        address(verifier)
        if "," in plan:
            raise ValueError("a store serves one session: --remote-plan is one mode")
        session_module("prover").parse_plan(plan)
        if kv_pages < 1:
            raise ValueError("the session's KV pool is at least a page")
        self.verifier, self.plan, self.kv_pages, self.weight_blocks = verifier, plan, int(kv_pages), weight_blocks
        self.kv_lib, self.out, self.node, self.codec, self.log = kv_lib, None if out is None else Path(out), node, codec, log
        self.store, self.binder = Store(), None

    def engine_args(self, engine_args: dict[str, Any]) -> dict[str, Any]:
        why = []
        if int(engine_args.get("tensor_parallel_size") or 1) != 1 or int(engine_args.get("pipeline_parallel_size") or 1) != 1:
            why.append("parallelism (a store is one GPU's: a TP Commit's session is not wired yet)")
        if str(engine_args.get("kv_cache_dtype") or "auto") not in ("auto", "bfloat16"):
            why.append(f"kv_cache_dtype={engine_args['kv_cache_dtype']} (the session's KV pool is bf16)")
        if engine_args.get("async_scheduling"):
            why.append("async scheduling (a pass follows each step's schedule and its output in turn)")
        if engine_args.get("enable_prefix_caching"):
            why.append("prefix caching (a prefix hit brings a page back with its KV)")
        if why:
            raise SystemExit("--remote-verifier refuses " + "; ".join(why))
        return {**engine_args, "async_scheduling": False, "enable_prefix_caching": False,
                "num_gpu_blocks_override": self.kv_pages}

    def install(self) -> None:
        from verity_vllm.store.binder import Binder

        self.binder = Binder(self.store, passes=False, weight_blocks=self.weight_blocks).install()

    def check(self, llm: Any, runner: Any) -> dict[str, Any]:
        vc, why = llm.llm_engine.vllm_config, []
        if getattr(getattr(llm.llm_engine, "engine_core", None), "engine_core", None) is None:
            why.append("the engine core in another process")
        if vc.scheduler_config.async_scheduling:
            why.append("async scheduling resolved on")
        if vc.cache_config.enable_prefix_caching:
            why.append("prefix caching on")
        if vc.parallel_config.tensor_parallel_size != 1:
            why.append(f"tensor parallel size {vc.parallel_config.tensor_parallel_size}")
        dtypes = sorted({str(t.dtype) for t in runner.kv_caches})
        if dtypes != ["torch.bfloat16"]:
            why.append(f"KV cache dtype {dtypes}")
        if "kv" not in {r.kind for r in self.store.layout().regions}:
            why.append("no KV pool registered (the binder's initialize_kv_cache hook)")
        if why:
            raise SystemExit("--remote-verifier: the engine is not one the session serves: " + "; ".join(why))
        return self.binder.stats()

    def serve(self, engine: Any, requests: Sequence[tuple[str, list[int], Any]], concurrent: int = 1) -> dict[str, Any]:
        workload = Workload(engine, requests, concurrent)
        path = None if self.out is None else self.out / "memory_map.jsonl"
        device = None if self.kv_lib is None else KvDevice(self.kv_lib, self.codec)
        backend = StoreBackend(self.store, workload, path=path, device=device)
        error, result = None, {}
        try:
            (result,) = prove(backend, self.verifier, self.plan, node=self.node.encode(), codec=self.codec, log=self.log)
        except Refused as e:
            error = str(e)
        rec = {"schema": "verity-vllm/remote-session/v0", "verifier": self.verifier, "plan": self.plan,
               "accepted": error is None and bool(result.get("accepted")), "error": error, "prover": result,
               "store": self.binder.stats(), "layout": backend.layout(), "kv_pages": self.kv_pages,
               "weight_blocks": self.weight_blocks, "encode": "kv.cu" if device is not None else "cpu",
               "gates": backend.gates, "workload": {"steps": workload.steps, "requests": workload.added,
                                                     "finished": workload.finished},
               "memory_map": None if backend.trace is None else {
                   "path": None if path is None else str(path), "registration": backend.registration.digest.hex(),
                   "steps": backend.trace.steps, "rekeyed": backend.trace.rekeyed, **agreement(backend)}}
        if self.out is not None:
            (self.out / "remote_session.json").write_text(json.dumps(rec, indent=1, default=str) + "\n")
        return rec


def agreement(backend: StoreBackend) -> dict[str, Any]:
    """The memory map against the session and the store, once the session is done: replayed from its header, its
    live indices are the session's last plan's and its blocks the audit's appends (their digests, in order); then,
    its open line closed, every slot it holds at an appended index holds bytes of that index's digest in the store."""
    trace, store = backend.trace, backend.store

    def replay() -> tuple[memory_map.Replay, dict[int, bytes]]:
        head, steps = memory_map.parse(store.memory_map.text().splitlines(keepends=True))
        r, digests = memory_map.Replay(head), {}
        for st in steps:
            r.apply(st)
            digests.update((w.index, w.digest) for w in st.written)
        return r, digests

    r, digests = replay()
    n = r.header.blocks
    live = {j for x in range(n) if (j := r.holds(x)) is not None} == trace.live
    appends = [d for a in trace.appends for d in a.digests] == [digests[j] for j in sorted(digests)]
    store.memory_map.flush()
    r, digests = replay()
    held = [x for x in range(n) if (r.holds(x) or 0) >= n]
    wrong = [x for x in held if rcodec.digest(store.payload(x)) != digests[r.holds(x)]]
    return {"live_agrees": live, "appends_agree": appends, "held_appended": len(held), "held_wrong": wrong[:16],
            "agrees": live and appends and not wrong}
