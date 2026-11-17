"""PoUS's prover over a served model's weights: ``W`` (the weight tensors a serving system hands in) in the store's
frozen layout, the commitment to it, ``C* = P2-Enc(W, salt)`` encoded and held on the weights' device, the decode of any
tensor from ``C*`` at use, and each period's audit of what is held. A serving system's adapter finds the tensors and
attaches the decode to its layers (vLLM's: `verity_vllm.protocols.pous`).

* **W** is each tensor from a block boundary over ⌈size / payload⌉ blocks, its last block zero past its end, and ``W``
  zero to whole segments of the point's ``2^16`` blocks (`layout_of`, `TensorLayout.of_ranges`; the store's
  `p2/p2_store/store.py`).
* **Commit** (`commit`): ``setup.commit(params, W)`` streamed a segment at a time, before any salt exists.
* **Encode** (`encode`): ``C*`` encoded a segment at a time under the salt (`p2_store.store.GpuEncoder`,
  ``libp2v1.so`` on the weights' GPU, which checks its own decode of the whole segment; `CpuEncoder` on the CPU), two
  blocks of each segment held to ``W`` by the reference decode, and ``C*`` held as one [blocks, block_bytes] uint8
  tensor on the weights' device; the memory attester's table of it and the root over that table
  (`VerifierKey.from_row_digests`). The salt is ``setup.sample_salt(source, commitment)``, ``source`` the auditor's coin
  flipped after the commitment (`salt_from`).
* **Decode** (`decoder`): a tensor's bytes from ``C*`` into a fresh buffer, by the store's load
  `p2_store.load.EncodedRows.decode_into` on the GPU, refusing what the reference refuses, or the scheme's codec on the
  CPU (`Store.decode_cpu`).
* **The period** (`answer`, `audit`, `subprotocol`): `period_prover.audit` of what setup published, the prover answering each index
  from ``C*`` where it is held, the memory attester at ``period.json``'s ``challenger_host``, and the Lean answering
  where the period's tools name it.

It imports no torch and nothing of a serving system: what touches a tensor is its P2 store's (`p2_store.tensors`).
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.machinery
import importlib.util
import json
import secrets
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: P2's encoded weights store, loaded as the package its directory names (``p2_store``): ``store.py``, which PoUS
#: encodes with, ``load.py``, its decode at use on the GPU, and its CUDA library's wrapper.
P2_STORE = Path(__file__).resolve().parent / "p2" / "p2_store"


def geometry(scheme: str) -> Any:
    """The point's geometry at one segment: the registered scheme's (its width, prime and ``2^16`` blocks a segment)."""
    from verity.protocols.resource_accounting.memory.pous.verifier import schemes as pous_schemes
    return pous_schemes.get(scheme).params


def load_package(name: str, directory: Path) -> None:
    """``directory`` as the package ``name``, loaded once from its files, so its modules import one another as they do in
    their own tree while ``sys.path`` stays as it is."""
    if name in sys.modules:
        return
    init = directory / "__init__.py"
    if init.exists():
        spec = importlib.util.spec_from_file_location(name, init, submodule_search_locations=[str(directory)])
        assert spec is not None, f"{init} ends in .py, which importlib has a loader for"
    else:
        spec = importlib.machinery.ModuleSpec(name, None, is_package=True)
        spec.submodule_search_locations = [str(directory)]
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        if spec.loader is not None:
            spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise


def p2_module(module: str) -> Any:
    """``store``, ``load`` or ``tensors``, a module of the package `P2_STORE` names (as ``p2_store.store``)."""
    load_package(P2_STORE.name, P2_STORE)
    return importlib.import_module(f"{P2_STORE.name}.{module}")


def flat(t: Any) -> Any:
    return p2_module("tensors").flat(t)


def layout_of(scheme: str, tensors: Mapping[str, Any]) -> Any:
    """The store's frozen layout over ``tensors``: each from a block boundary, in order (`p2_store/store.py`'s ``build``)."""
    from verity.protocols.resource_accounting.memory.pous.verifier.layout import TensorLayout

    base = geometry(scheme)
    pb, block, ranges = base.payload_bytes, 0, []
    for name, t in tensors.items():
        size = t.numel() * t.element_size()
        ranges.append((name, block * pb, size))
        block += -(-size // pb)
    return TensorLayout.of_ranges(base, ranges)


def segment(layout: Any, flats: Mapping[str, Any], s: int) -> bytes:
    """Segment ``s`` of ``W``: each tensor's bytes that fall in it at their place, zeros elsewhere."""
    seg = layout.params.segment_bytes
    lo, hi = s * seg, (s + 1) * seg
    out = bytearray(seg)
    for p in layout.tensors:
        a, b = max(lo, p.offset), min(hi, p.offset + p.size)
        if a < b:
            out[a - lo:b - lo] = flats[p.name][a - p.offset:b - p.offset].cpu().numpy().tobytes()
    return bytes(out)


def commit(scheme: str, tensors: Mapping[str, Any]) -> tuple[Any, bytes]:
    """``W``'s layout and the commitment to it (no salt exists yet), ``W`` streamed a segment at a time."""
    from verity.protocols.resource_accounting.memory.pous.verifier import setup

    layout = layout_of(scheme, tensors)
    flats = {n: flat(t) for n, t in tensors.items()}
    return layout, setup.commit(layout.params, (segment(layout, flats, s) for s in range(layout.params.segments)))


def layout_digest(layout: Any) -> str:
    return hashlib.sha256(json.dumps(layout.manifest(), sort_keys=True).encode()).hexdigest()


def salt_from(entry: str, setting: Any, commitment: bytes, params: Any) -> tuple[bytes, str]:
    """The auditor's coin for the salt, sampled after ``commitment``, and how it came."""
    if callable(setting):
        return bytes(setting(commitment)), "callable"
    if isinstance(setting, str):
        return bytes.fromhex(setting), "settings"
    line = json.dumps({"commitment": commitment.hex(), "manifest": params.manifest()}, sort_keys=True) + "\n"
    p = subprocess.run([str(a) for a in setting["command"]], input=line, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"{entry}: the salt source's command exited {p.returncode}: {p.stderr.strip()[-400:]}")
    return bytes.fromhex(json.loads(p.stdout.strip().splitlines()[-1])["source"]), "command"


@dataclass(eq=False)
class Store:
    """``C*`` on the weights' device (``rows``, block ``i`` its row ``i``), the point, the layout, what setup published
    and the memory attester's table of ``C*``."""

    point: Any
    layout: Any
    commitment: bytes
    salt: bytes
    salt_source: str
    rows: Any
    digests: bytes
    root: bytes
    encode: dict[str, Any] = field(default_factory=dict)
    staged: Any = None

    @property
    def params(self) -> Any:
        return self.point.params

    def published(self) -> Any:
        from verity.protocols.resource_accounting.memory.pous.verifier import run as R
        return R.Setup(self.params.blocks, self.commitment, self.root, self.salt)

    def block(self, i: int) -> bytes:
        if self.rows is None:
            raise RuntimeError("C* is freed")
        if not 0 <= i < self.params.blocks:
            raise IndexError(f"block {i} is outside C*'s {self.params.blocks}")
        return self.rows[i].cpu().numpy().tobytes()

    def segments(self) -> list[list[bytes]]:
        """``C*`` segment by segment, as setup sends it."""
        host, n = self.rows.cpu().numpy(), self.params.blocks_per_segment
        return [[host[s * n + v].tobytes() for v in range(n)] for s in range(self.params.segments)]

    def held(self) -> Any:
        """`period_prover.EncodedStore` of what this setup left: the setup case is built from ``C*`` as held now."""
        from verity.protocols.resource_accounting.memory.pous.verifier import period
        from verity.protocols.resource_accounting.memory.pous.prover import period_prover
        return period_prover.EncodedStore(self.point, self.published(), Blocks(self), self.digests,
                                          period.setup_case(self.params, self.commitment, self.salt, self.segments()))

    def decode_cpu(self, offset: int, size: int) -> bytes:
        """``W`` bytes ``[offset, offset + size)`` by the scheme's codec, from the segments they fall in; refused when a
        block isn't a codeword or the range's last block is nonzero past it."""
        from verity.protocols.resource_accounting.memory.pous.verifier.schemes.p2 import decode_segment

        q = self.params
        n, pb, seg = q.blocks_per_segment, q.payload_bytes, q.segment_bytes
        end = -(-(offset + size) // pb) * pb
        s0, s1 = offset // seg, -(-end // seg)
        data = b"".join(decode_segment(q, None, self.salt, s, [self.block(s * n + v) for v in range(n)])
                        for s in range(s0, s1))
        lo = offset - s0 * seg
        if any(data[lo + size:end - s0 * seg]):
            raise ValueError(f"the block holding W byte {offset + size - 1} is nonzero past the tensor's end")
        return data[lo:lo + size]

    def free(self) -> None:
        self.rows = self.staged = None


class Blocks:
    """``C*``'s blocks as `period_prover.EncodedStore` reads them, from where they are held."""

    def __init__(self, store: Store) -> None:
        self.store = store

    def __len__(self) -> int:
        return self.store.params.blocks

    def __getitem__(self, i: int) -> bytes:
        return self.store.block(i)


def encode(layout: Any, tensors: Mapping[str, Any], commitment: bytes, salt: bytes, salt_source: str,
            lib: str | None) -> Store:
    """``C*`` of ``W`` under ``salt`` on the weights' device, each segment's encode checked (module docstring)."""
    from verity.protocols.resource_accounting.memory.pous.attester import memory_attester as mc
    from verity.protocols.resource_accounting.memory.pous.verifier import period
    from verity.protocols.resource_accounting.memory.pous.verifier import run as R
    from verity.protocols.resource_accounting.memory.pous.verifier.schemes.p2 import expand_prefix
    from verity.protocols.resource_accounting.memory.pous.verifier.verifier import (
        DIGEST_BYTES,
        VerifierKey,
    )

    S, T = p2_module("store"), p2_module("tensors")
    q = layout.params
    n, nb, pb = q.blocks_per_segment, q.block_bytes, q.payload_bytes
    device = next(iter(tensors.values())).device
    if device.type == "cuda":
        if not lib:
            raise ValueError("the weights are on a GPU and no p2_lib names libp2v1.so's directory "
                             "(verity/protocols/resource_accounting/memory/pous/prover/p2/p2_store/Makefile builds it)")
        enc = S.GpuEncoder(q, salt, str(lib), T.device_index(device))
    else:
        enc = S.CpuEncoder(q, salt)
    prefix, flats = expand_prefix(q, salt), {name: flat(t) for name, t in tensors.items()}
    rows = T.rows(q.blocks, nb, device)
    digests, device_s, t0 = bytearray(), 0.0, time.perf_counter()
    for s in range(q.segments):
        w_seg = segment(layout, flats, s)
        c_seg, extra = enc.encode(s, w_seg)
        device_s += extra.get("encode_s", 0.0)
        for j in S.spot_blocks(salt, s, n):
            if S.decode_block(q, prefix, s * n + j, c_seg[j * nb:(j + 1) * nb]) != w_seg[j * pb:(j + 1) * pb]:
                raise RuntimeError(f"segment {s} block {j}: the reference decode of C* is not W's payload")
        T.put(rows, s * n, c_seg, nb)
        digests += b"".join(mc.block_digest(c_seg[i * nb:(i + 1) * nb]) for i in range(n))
    encode_s = time.perf_counter() - t0
    root = VerifierKey.from_row_digests(q, salt, [bytes(digests[DIGEST_BYTES * i:DIGEST_BYTES * (i + 1)])
                                                  for i in range(q.blocks)]).root
    timing = {"encoder": extra.get("encoder"), "device": extra.get("device", str(device)), "wall_s": round(encode_s, 3),
              "device_encode_s": round(device_s, 3), "root_s": round(time.perf_counter() - t0 - encode_s, 3),
              "blocks_per_s": round(q.blocks / encode_s, 1) if encode_s else None}
    return Store(R.Point(q, period.K, period.LIMIT_NS), layout, commitment, salt, salt_source, rows, bytes(digests), root,
                  timing)


def decoder(store: Store, lib: str | None) -> Callable[[Any, Any], None]:
    """``decode(placement, out)``: the placed tensor's bytes from ``C*`` into ``out`` (flat uint8, on ``C*``'s device)."""
    if store.rows.device.type == "cuda":
        from verity.protocols.resource_accounting.memory.pous.verifier.schemes.p2 import expand_prefix

        Ld = p2_module("load")
        store.staged = Ld.EncodedRows(Ld.StoreLayout(store.params, (), expand_prefix(store.params, store.salt)), store.rows, 0,
                                str(lib))

        def decode(p: Any, out: Any) -> None:
            store.staged.decode_into(p.offset, p.size, out, end=True)
            store.staged.check()
        return decode

    T = p2_module("tensors")

    def decode_host(p: Any, out: Any) -> None:
        if p.size:
            T.fill(out, store.decode_cpu(p.offset, p.size))
    return decode_host


# --- the period ------------------------------------------------------------------------------------------------------

def answer(store: Store | None, index: int) -> bytes:
    """The prover's reply to challenge ``index``: block ``index`` of ``C*`` where it is held."""
    if store is None:
        raise RuntimeError("no C* is held: its prover is stopped")
    return store.block(index)


def audit(store: Store, node: str, coins: bytes, **kw: Any) -> Any:
    """`period_prover.audit` of what ``store``'s setup published, on ``node`` under the auditor's ``coins``; ``kw`` as
    `period_prover.audit`'s."""
    from verity.protocols.resource_accounting.memory.pous.prover import period_prover
    return period_prover.audit(node, store.held(), coins, **kw)


def subprotocol(store: Store | None, period_dir: str | Path, node: str, out: str | Path, *, audit: Callable[..., Any],
        side_of: Callable[..., Any], fresh: bool, answered: Callable[[], int],
        schemes: Mapping[str, Any] | None = None, protocol: str | None = None) -> dict[str, Any]:
    """One period's PoUS of ``store``, as `period_prover.subprotocol` runs it for a node's records: the memory attester at
    ``period.json``'s ``challenger_host`` (``side_of``, the runtime's ``Side.of``, its side's key and session log in
    ``pous-challenger`` beside the periods when it is this node), fresh coins handed to ``audit`` (the holder's
    `audit`), and the Lean's answer where the period's tools name ``pous-run-protocol`` (and, at the first period,
    ``fresh``, ``pous-setup``), which must agree with the Python's; where they name none, the Python's accept is
    undecided (`period.unjudged`).  Writes ``out/pous.case.json``, ``out/pous.json`` and, at the first period,
    ``out/pous-setup.case.json``; returns the answer, its profile carrying where each party ran, whether
    `trusted-device` holds, which only the key the session was signed under and the period's `deployment` decide, and
    that setup was checked by the developer, not the auditor (`challenger_session.placement`).  ``answered()`` counts the rounds the prover has answered so far.
    ``schemes`` and ``protocol`` are the signatures the verifier accepts and the id it answers under (a test signer's
    in tests)."""
    from verity.protocols.resource_accounting.memory.pous.attester import challenger_session as CS
    from verity.protocols.resource_accounting.memory.pous.attester import memory_attester as mc
    from verity.protocols.resource_accounting.memory.pous.prover import challenger_prover as CR
    from verity.protocols.resource_accounting.memory.pous.verifier import period

    doc = json.loads((Path(period_dir) / "period.json").read_text())
    tools = doc.get("tools", {})
    party = side_of(doc, node, "pous", CS.command(Path(period_dir).parent / "pous-challenger"))
    before = answered()
    t = time.monotonic()
    got = audit(node, secrets.token_bytes(64), party=party, session=CR.session,
                schemes=mc.SCHEMES if schemes is None else schemes,
                protocol=period.PROTOCOL if protocol is None else protocol)
    seconds = round(time.monotonic() - t, 2)
    where = CS.placement(node, party.host, got.key_sha512, **period.deployment(doc))
    answer, lean = got.answer, {}
    if "pous-run-protocol" in tools:
        lean["run"] = period.serve(tools["pous-run-protocol"], [got.case])[0]
        if {k: v for k, v in lean["run"].items() if k != "reason"} != {k: v for k, v in answer.items() if k != "reason"}:
            raise RuntimeError(f"pous-run-protocol answered {lean['run']}, the Python {answer}")
        answer = lean["run"]
    else:
        answer = period.unjudged(answer)
    if fresh and "pous-setup" in tools:
        lean["setup"] = period.serve(tools["pous-setup"], [got.setup_case])[0]
        if lean["setup"]["verdict"] != "accept":
            raise RuntimeError(f"pous-setup refused the setup the Python accepted: {lean['setup']}")
    if answer["profile"] is not None:
        answer = answer | {"profile": answer["profile"] | {"placement": where}}
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "pous.case.json").write_text(json.dumps(got.case, sort_keys=True))
    if fresh:
        (out / "pous-setup.case.json").write_text(json.dumps(got.setup_case, sort_keys=True))
    assert store is not None, "the audit above refuses a holder with no store"
    q = store.params
    (out / "pous.json").write_text(json.dumps({
        "answer": answer, "blocks": q.blocks, "eps": str(got.eps), "lean": lean, "placement": where,
        "rounds_answered": answered() - before, "s_bits": got.public("s_bits"), "seconds": seconds,
        "setup": {"fresh": fresh, "encode": store.encode if fresh else None}, "w_bytes": q.w_bytes},
        sort_keys=True, indent=1))
    return answer
