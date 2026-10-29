"""POUS as a protocol option: the served model's linear weights held only as ``C* = P2-Enc(W, salt)`` at P2's registered
point (``p2-16448/v4``), decoded at every use, and each period's audit of what is held (`verity.core.protocols.pous.period`),
with the memory challenger a party of its own on the challenger host (`verity.core.protocols.pous.challenger_party`).

* **W** is the weight of every ``LinearBase`` layer as the engine holds it (int8 [N, K] on the ``int8_exact`` target),
  deduplicated by storage, in module order and named by its first holder, in the store's frozen layout
  (`benchmarks/pous/p2_v1/store.py`): each tensor from a block boundary over ⌈size / payload⌉ blocks, its last block zero
  past its end, and ``W`` zero to whole segments of the point's ``2^16`` blocks (`Layout.of_ranges`).  Scales, embeddings
  and norms are not in ``W``.  These are the tensors the Commit registers, read through `weights_view` while released.
* **Commit** (`commit_weights`, the composition's first phase, model untouched): ``setup.commit(params, W)`` before any
  salt exists, recorded in the composition record's ``weights``.
* **Install**, in order: the same commitment recomputed, refused if the weights changed since; the salt
  ``setup.draw_salt(source, commitment)``, ``source`` the auditor's coin flipped after the commitment (``salt_source``);
  ``C*`` encoded a segment at a time (`store.GpuEncoder`, ``libp2v1.so`` on the weights' GPU, which checks its own decode
  of the whole segment; `store.CpuEncoder` on the CPU), two blocks of each segment held to ``W`` by the reference decode,
  and ``C*`` held as one [blocks, block_bytes] uint8 tensor on the weights' device; the memory challenger's table of it
  and the root over that table (`VerifierKey.from_row_digests`).  Then, through the option's tagged `Hooks`: the
  plaintext released (a service whose stop restores ``W`` decoded from ``C*``), each layer's ``quant_method.apply``
  wrapped (the layer's weight decoded from ``C*`` into a fresh buffer, by the staging decode
  `p2_stage.stage.Store.decode_into` on the GPU, refusing what the reference refuses, or the scheme's codec on the CPU,
  and the inner ``apply`` called with a proxy layer holding it), and the prover started last.  Uninstall runs newest
  first: prover, wrappers, then ``W`` restored; ``C*`` is freed after.
* **The period** (`PousOption.audit`, `PousOption.leg`): `period.audit` of what setup published, the prover answering
  each index from ``C*`` where it is held, and the verifier's answer: setup's own inputs tied to the run (`period.tie`,
  the public-encoder check of ``C*``), the run's check, and the bits an accepted run certifies (`period.certify`: 0 at a
  point no certificate covers).  In `PousOption.leg` the memory challenger runs at ``period.json``'s ``challenger_host``
  (`challenger_party.Party.of`) and the Lean answers where the period's tools name it.
* **Composition:** installed after PoUW and around it, so PoUW's GEMM consumes the decoded weight.  An option that
  keeps a weight-derived copy resident across forwards is the selector's to refuse (``keeps_weight_copy``); POUS keeps
  only ``C*``.
* **Weights of record:** `Installed.weights` (``decoded``) returns the released plaintext by ``named_parameters`` name,
  tied names mapped to one tensor, which the Commit's replay reads (`protocols.weights_view`).

Settings: ``salt_source`` (the auditor's coin: its hex, a callable of the commitment, or ``{"command": argv}``, a process
that reads ``{"commitment", "manifest"}`` as one JSON line and prints ``{"source": HEX}``), ``p2_lib`` (``libp2v1.so``'s
directory, built for the GPU by ``benchmarks/torch_gpu/p2_stage/build.sh``; weights on a GPU need it) and
``experimental`` (P2 is; `schemes.p2.EXPERIMENTAL` says why).

Not here yet: TP (each rank's shard), the head and the embedding (not ``LinearBase``), and a setup case that scales:
`period.setup_case` carries ``C*`` in hex, 2.2 GB at the registered point, and the tie decodes all of it in Python each
period.
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
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from verity_vllm import config
from verity_vllm.protocols.interface import Context, Option, Target

NAME = "pous"
DEFAULT_SCHEME = "p2-16448/v4"
SETTINGS = ("salt_source", "p2_lib", "experimental")
COVERS = "linears"


def schemes() -> tuple[str, ...]:
    from verity.core.protocols.pous import schemes as pous_schemes
    return tuple(sorted(pous_schemes.REGISTRY))


def _source_refusal(source: Any) -> str | None:
    """Why ``salt_source`` can't give the auditor's coin, or None."""
    if source is None:
        return ("no salt_source: the salt is setup.draw_salt(source, commitment), the source the auditor's coin drawn after "
                "the commitment (its hex, a callable of the commitment, or {'command': argv})")
    if callable(source):
        return None
    if isinstance(source, str):
        try:
            return None if bytes.fromhex(source) else "salt_source is empty"
        except ValueError:
            return f"salt_source {source[:16]!r}... is not hex"
    if isinstance(source, Mapping) and set(source) == {"command"} and isinstance(source["command"], (list, tuple)) \
            and source["command"]:
        return None
    return f"salt_source is hex, a callable of the commitment or {{'command': argv}}, not {type(source).__name__}"


def refusals(scheme: str, ctx: Context) -> tuple[str, ...]:
    from verity.core.protocols.pous import schemes as pous_schemes

    out, own = [], ctx.settings_of(NAME)
    stray = sorted(set(own) - set(SETTINGS))
    if stray:
        out.append(f"unknown pous settings {stray} (known: {', '.join(SETTINGS)})")
    if not ctx.enforce_eager:
        out.append("compiled execution: the decode wrappers are not graph-capturable, so pous runs eager")
    if scheme != DEFAULT_SCHEME:
        out.append(f"pous holds C* under {DEFAULT_SCHEME} alone, the codec its period's run protocol checks and the staging "
                   f"decode reads, not {scheme}")
    else:
        reason = pous_schemes.get(scheme).experimental
        if reason and not own.get("experimental"):
            out.append(f"{scheme} is experimental ({reason}); settings pous.experimental = true runs it anyway")
    why = _source_refusal(own.get("salt_source"))
    if why:
        out.append(why)
    return tuple(out)


def linear_types() -> tuple[type, ...]:
    """The layer types whose weights POUS encodes, from vLLM (imported here, never at import time)."""
    from vllm.model_executor.layers.linear import LinearBase
    return (LinearBase,)


def _base(scheme: str) -> Any:
    """The point's geometry at one segment: the registered scheme's (its width, prime and ``2^16`` blocks a segment)."""
    from verity.core.protocols.pous import schemes as pous_schemes
    return pous_schemes.get(scheme).params


def _package(name: str, directory: Path) -> None:
    """``directory`` as the package ``name``, loaded once from its files, so its modules import one another as they do in
    their own tree while ``sys.path`` stays as it is."""
    if name in sys.modules:
        return
    init = directory / "__init__.py"
    if init.exists():
        spec = importlib.util.spec_from_file_location(name, init, submodule_search_locations=[str(directory)])
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


def _p2(module: str) -> Any:
    """``store`` (`benchmarks/pous/p2_v1/store.py`, as ``p2_v1.store``) or ``stage``
    (`benchmarks/torch_gpu/p2_stage/stage.py`, as ``p2_stage.stage``)."""
    _package("p2_v1", config.P2_POUS / "p2_v1")
    if module == "store":
        return importlib.import_module("p2_v1.store")
    _package("p2_stage", config.P2_STAGE)
    return importlib.import_module("p2_stage.stage")


# --- the weights ------------------------------------------------------------------------------------------------------------

def _layers(model: Any) -> list[tuple[str, Any]]:
    types = linear_types()
    return [(name, m) for name, m in model.named_modules() if isinstance(m, types) and getattr(m, "weight", None) is not None]


def _plaintext(entry: str, model: Any) -> tuple[list[tuple[str, Any]], dict[int, str], dict[str, Any]]:
    """The layers, each layer's weight's name (its first holder's), and each unique weight by that name."""
    layers = _layers(model)
    if not layers:
        raise ValueError(f"{entry}: the model has no {', '.join(t.__name__ for t in linear_types())} layer to encode")
    first: dict[Hashable, str] = {}
    names: dict[int, str] = {}
    tensors: dict[str, Any] = {}
    for name, m in layers:
        w = m.weight
        if not w.is_contiguous():
            raise ValueError(f"{entry}: {name}'s weight is strided {tuple(w.stride())}: W holds each tensor's bytes in the "
                             "order its kernel reads them, which is a contiguous tensor's")
        k = (w.untyped_storage().data_ptr(), w.storage_offset(), tuple(w.shape), str(w.dtype))
        if k not in first:
            first[k] = name
            tensors[name] = w
        names[id(m)] = first[k]
    devices = sorted({str(t.device) for t in tensors.values()})
    if len(devices) > 1:
        raise ValueError(f"{entry}: the weights are on {devices}, and C* is held on one device")
    return layers, names, tensors


def _flat(t: Any) -> Any:
    import torch
    return t.detach().reshape(-1).view(torch.uint8)


def _layout(scheme: str, tensors: Mapping[str, Any]) -> Any:
    """The store's frozen layout over ``tensors``: each from a block boundary, in order (`p2_v1/store.py`'s ``build``)."""
    from verity.core.protocols.pous.layout import Layout

    base = _base(scheme)
    pb, block, ranges = base.payload_bytes, 0, []
    for name, t in tensors.items():
        size = t.numel() * t.element_size()
        ranges.append((name, block * pb, size))
        block += -(-size // pb)
    return Layout.of_ranges(base, ranges)


def _segment(layout: Any, flats: Mapping[str, Any], s: int) -> bytes:
    """Segment ``s`` of ``W``: each tensor's bytes that fall in it at their place, zeros elsewhere."""
    seg = layout.params.segment_bytes
    lo, hi = s * seg, (s + 1) * seg
    out = bytearray(seg)
    for p in layout.tensors:
        a, b = max(lo, p.offset), min(hi, p.offset + p.size)
        if a < b:
            out[a - lo:b - lo] = flats[p.name][a - p.offset:b - p.offset].cpu().numpy().tobytes()
    return bytes(out)


def _commit(scheme: str, tensors: Mapping[str, Any]) -> tuple[Any, bytes]:
    """``W``'s layout and the commitment to it (no salt exists yet), ``W`` streamed a segment at a time."""
    from verity.core.protocols.pous import setup

    layout = _layout(scheme, tensors)
    flats = {n: _flat(t) for n, t in tensors.items()}
    return layout, setup.commit(layout.params, (_segment(layout, flats, s) for s in range(layout.params.segments)))


def _layout_digest(layout: Any) -> str:
    return hashlib.sha256(json.dumps(layout.manifest(), sort_keys=True).encode()).hexdigest()


def commit_weights(scheme: str, target: Target) -> dict[str, Any]:
    """Phase 1 of a composition: the commitment to the plaintext ``W`` of record, with the model untouched and before any
    salt exists (`setup.commit`).  `install` refuses weights that no longer match it."""
    _, _, tensors = _plaintext(f"{NAME}:{scheme}", target.model)
    layout, commitment = _commit(scheme, tensors)
    return {"scheme": scheme, "covers": COVERS, "tensors": len(tensors), "bytes": sum(p.size for p in layout.tensors),
            "commitment": commitment.hex(), "layout": _layout_digest(layout)}


def _source(entry: str, setting: Any, commitment: bytes, params: Any) -> tuple[bytes, str]:
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


# --- the encoded store ---------------------------------------------------------------------------------------------------

@dataclass(eq=False)
class _Store:
    """``C*`` on the weights' device (``rows``, block ``i`` its row ``i``), the point, the layout, what setup published
    and the memory challenger's table of ``C*``."""

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
        from verity.core.protocols.pous import run as R
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
        """`period.Store` of what this setup left: the setup case is built from ``C*`` as held now."""
        from verity.core.protocols.pous import period
        return period.Store(self.point, self.published(), _Blocks(self), self.digests,
                            period.setup_case(self.params, self.commitment, self.salt, self.segments()))

    def decode_cpu(self, offset: int, size: int) -> bytes:
        """``W`` bytes ``[offset, offset + size)`` by the scheme's codec, from the segments they fall in; refused when a
        block isn't a codeword or the range's last block is nonzero past it."""
        from verity.core.protocols.pous.schemes.p2 import decode_segment

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


class _Blocks:
    """``C*``'s blocks as `period.Store` reads them, from where they are held."""

    def __init__(self, store: _Store) -> None:
        self.store = store

    def __len__(self) -> int:
        return self.store.params.blocks

    def __getitem__(self, i: int) -> bytes:
        return self.store.block(i)


def _encode(layout: Any, tensors: Mapping[str, Any], commitment: bytes, salt: bytes, salt_source: str,
            lib: str | None) -> _Store:
    """``C*`` of ``W`` under ``salt`` on the weights' device, each segment's encode checked (module docstring)."""
    import torch

    from verity.core.protocols.pous import memory_challenger as mc
    from verity.core.protocols.pous import period
    from verity.core.protocols.pous import run as R
    from verity.core.protocols.pous.schemes.p2 import expand_prefix
    from verity.core.protocols.pous.verifier import DIGEST_BYTES, VerifierKey

    S = _p2("store")
    q = layout.params
    n, nb, pb = q.blocks_per_segment, q.block_bytes, q.payload_bytes
    device = next(iter(tensors.values())).device
    if device.type == "cuda":
        if not lib:
            raise ValueError("the weights are on a GPU and no p2_lib names libp2v1.so's directory "
                             "(benchmarks/torch_gpu/p2_stage/build.sh builds it)")
        enc = S.GpuEncoder(q, salt, str(lib), torch.cuda.current_device() if device.index is None else device.index)
    else:
        enc = S.CpuEncoder(q, salt)
    prefix, flats = expand_prefix(q, salt), {name: _flat(t) for name, t in tensors.items()}
    rows = torch.empty(q.blocks, nb, dtype=torch.uint8, device=device)
    digests, device_s, t0 = bytearray(), 0.0, time.perf_counter()
    for s in range(q.segments):
        w_seg = _segment(layout, flats, s)
        c_seg, extra = enc.encode(s, w_seg)
        device_s += extra.get("encode_s", 0.0)
        for j in S.spot_blocks(salt, s, n):
            if S.decode_block(q, prefix, s * n + j, c_seg[j * nb:(j + 1) * nb]) != w_seg[j * pb:(j + 1) * pb]:
                raise RuntimeError(f"segment {s} block {j}: the reference decode of C* is not W's payload")
        rows[s * n:(s + 1) * n].copy_(torch.frombuffer(bytearray(c_seg), dtype=torch.uint8).view(n, nb))
        digests += b"".join(mc.block_digest(c_seg[i * nb:(i + 1) * nb]) for i in range(n))
    encode_s = time.perf_counter() - t0
    root = VerifierKey.from_row_digests(q, salt, [bytes(digests[DIGEST_BYTES * i:DIGEST_BYTES * (i + 1)])
                                                  for i in range(q.blocks)]).root
    timing = {"encoder": extra.get("encoder"), "device": extra.get("device", str(device)), "wall_s": round(encode_s, 3),
              "device_encode_s": round(device_s, 3), "root_s": round(time.perf_counter() - t0 - encode_s, 3),
              "blocks_per_s": round(q.blocks / encode_s, 1) if encode_s else None}
    return _Store(R.Point(q, period.K, period.LIMIT_NS), layout, commitment, salt, salt_source, rows, bytes(digests), root,
                  timing)


def _decoder(store: _Store, lib: str | None) -> Callable[[Any, Any], None]:
    """``decode(placement, out)``: the placed tensor's bytes from ``C*`` into ``out`` (flat uint8, on ``C*``'s device)."""
    import torch

    if store.rows.device.type == "cuda":
        from verity.core.protocols.pous.schemes.p2 import expand_prefix

        St = _p2("stage")
        store.staged = St.Store(St.L.StoreLayout(store.params, (), expand_prefix(store.params, store.salt)), store.rows, 0,
                                str(lib))

        def decode(p: Any, out: Any) -> None:
            store.staged.decode_into(p.offset, p.size, out, end=True)
            store.staged.check()
        return decode

    def decode_host(p: Any, out: Any) -> None:
        if p.size:
            out.copy_(torch.frombuffer(bytearray(store.decode_cpu(p.offset, p.size)), dtype=torch.uint8))
    return decode_host


class _Proxy:
    """The layer the inner ``apply`` sees: the decoded weight, and every other attribute the layer's own."""

    def __init__(self, layer: Any, weight: Any) -> None:
        object.__setattr__(self, "_layer", layer)
        object.__setattr__(self, "weight", weight)

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_layer"), name)


@dataclass(eq=False)
class PousOption(Option):
    """The installed option: `answer` is the prover's reply to a challenge, `audit` and `leg` a period's audit of what it
    holds; ``weights()`` (`Option.decoded`) is the released plaintext decoded from ``C*``."""

    store: _Store | None = None
    names: dict[int, str] = field(default_factory=dict)
    answered: int = 0
    periods: int = 0

    def answer(self, index: int) -> bytes:
        if self.closed or self.store is None:
            raise RuntimeError(f"{self.name} is closed: its prover is stopped")
        self.answered += 1
        return self.store.block(index)

    def prover(self, blocks: Sequence[bytes]) -> Callable[[int], bytes]:
        """`period.Prover`: each index answered from ``C*`` where it is held (``blocks`` is the same store)."""
        return self.answer

    def audit(self, node: str, coins: bytes, **kw: Any) -> Any:
        """`period.audit` of this setup on ``node`` under the auditor's ``coins``; ``kw`` as `period.audit`'s (the prover
        by default this option's)."""
        from verity.core.protocols.pous import period

        if self.closed or self.store is None:
            raise RuntimeError(f"{self.name} is closed: nothing is held to audit")
        kw.setdefault("prover", self.prover)
        return period.audit(node, self.store.held(), coins, **kw)

    def leg(self, period_dir: str | Path, node: str, out: str | Path, *, schemes: Mapping[str, Any] | None = None,
            protocol: str | None = None) -> dict[str, Any]:
        """One period's PoUS of what this option holds, as `period.leg` runs it for a node's records: the memory
        challenger at ``period.json``'s ``challenger_host`` (`challenger_party.Party.of`, its side's key and session log
        in ``pous-challenger`` beside the periods when it is this node), fresh coins, and the Lean's answer where the
        period's tools name ``pous-run-protocol`` (and, at the first period, ``pous-setup``), which must agree with the
        Python's.  Writes ``out/pous.case.json``, ``out/pous.json`` and, at the first period, ``out/pous-setup.case.json``;
        returns the answer, its profile carrying where each party ran (`challenger_party.placement`).  ``schemes`` and
        ``protocol`` are the signatures the verifier accepts and the id it answers under (a test signer's in tests)."""
        from verity.core.protocols.pous import challenger_party as CP
        from verity.core.protocols.pous import memory_challenger as mc
        from verity.core.protocols.pous import period

        doc = json.loads((Path(period_dir) / "period.json").read_text())
        tools = doc.get("tools", {})
        party = CP.Party.of(doc, node, Path(period_dir).parent / "pous-challenger")
        where = CP.placement(node, party.host)
        fresh, answered = self.periods == 0, self.answered
        t = time.monotonic()
        got = self.audit(node, secrets.token_bytes(64), party=party, schemes=mc.SCHEMES if schemes is None else schemes,
                         protocol=period.PROTOCOL if protocol is None else protocol)
        seconds = round(time.monotonic() - t, 2)
        self.periods += 1
        answer, lean = got.answer, {}
        if "pous-run-protocol" in tools:
            lean["run"] = period.serve(tools["pous-run-protocol"], [got.case])[0]
            if {k: v for k, v in lean["run"].items() if k != "reason"} != {k: v for k, v in answer.items() if k != "reason"}:
                raise RuntimeError(f"pous-run-protocol answered {lean['run']}, the Python {answer}")
            answer = lean["run"]
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
        q = self.store.params
        (out / "pous.json").write_text(json.dumps({
            "answer": answer, "blocks": q.blocks, "eps": str(got.eps), "lean": lean, "placement": where,
            "rounds_answered": self.answered - answered, "s_bits": got.s_bits, "seconds": seconds,
            "setup": {"fresh": fresh, "encode": self.store.encode if fresh else None}, "w_bytes": q.w_bytes},
            sort_keys=True, indent=1))
        return answer


def _qualnames(entry: str, model: Any, layers: list[tuple[str, Any]], names: dict[int, str]) -> dict[str, list[str]]:
    """Each encoded weight's names in ``model.named_parameters()``, which the replay's weights view reads it under."""
    by_param = {id(m.weight): names[id(m)] for _, m in layers}
    out: dict[str, list[str]] = {n: [] for n in dict.fromkeys(names.values())}
    for q, p in model.named_parameters():
        if id(p) in by_param:
            out[by_param[id(p)]].append(q)
    missing = [n for n, qs in out.items() if not qs]
    if missing:
        raise ValueError(f"{entry}: the weights of {missing} are no parameters of the model, so no reader could find them")
    return out


def _wrapper(names: Mapping[int, str], decoded: Callable[[str], Any]) -> Callable[[Any], Any]:
    def make(inner: Any) -> Any:
        def apply(layer: Any, x: Any, bias: Any = None) -> Any:
            name = names.get(id(layer))
            return inner(layer, x, bias) if name is None else inner(_Proxy(layer, decoded(name)), x, bias)
        return apply
    return make


def install(scheme: str, target: Target) -> PousOption:
    import torch

    from verity.core.protocols.pous import period, setup
    from verity.core.protocols.pous.schemes import p2 as P2

    entry = f"{NAME}:{scheme}"
    own = target.context.settings_of(NAME)
    why = _source_refusal(own.get("salt_source"))
    if why:
        raise ValueError(f"{entry}: {why}")
    layers, names, tensors = _plaintext(entry, target.model)
    qualnames = _qualnames(entry, target.model, layers, names)
    layout, commitment = _commit(scheme, tensors)
    committed = target.weight_commitments.get(entry)
    if committed is not None and committed.get("commitment") != commitment.hex():
        raise ValueError(f"{entry}: the weights changed after the composition committed to them (commit_weights)")
    source, how = _source(entry, own["salt_source"], commitment, layout.params)
    store = _encode(layout, tensors, commitment, setup.draw_salt(source, commitment), how, own.get("p2_lib"))
    decode = _decoder(store, own.get("p2_lib"))
    meta = {name: (tuple(t.shape), t.dtype) for name, t in tensors.items()}
    device = store.rows.device

    def tensor(name: str) -> Any:
        shape, dtype = meta[name]
        out = torch.empty(shape, dtype=dtype, device=device)
        decode(layout.placement(name), _flat(out))
        return out

    def plaintext() -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name, qs in qualnames.items():
            out.update(dict.fromkeys(qs, tensor(name)))
        return out

    hooks = target.hooks(entry)
    opt = PousOption(name=entry, hooks=hooks, decoded=plaintext, release=store.free, store=store, names=names)
    params = [m.weight for _, m in layers]

    def release() -> list[Any]:
        for (_, m), p in zip(layers, params):
            p.data = torch.empty(0, dtype=meta[names[id(m)]][1], device=device)
        return params

    def restore(held: list[Any]) -> None:
        decoded: dict[str, Any] = {}
        for (_, m), p in zip(layers, held):
            name = names[id(m)]
            if name not in decoded:
                decoded[name] = tensor(name)
            p.data = decoded[name]

    hooks.service("plaintext released", release, restore)
    for qm in {id(m.quant_method): m.quant_method for _, m in layers}.values():
        hooks.wrap(qm, "apply", _wrapper(names, tensor))
    hooks.service("prover", lambda: opt.answer, lambda handle: {"answered": opt.answered})
    q = store.params
    opt.commits = {"scheme": scheme, "commitment": commitment.hex(), "salt": store.salt.hex(),
                   "storage_root": store.root.hex(), "blocks": q.blocks, "segments": q.segments,
                   "layout": _layout_digest(layout)}
    cert, gaps = period.gaps(store.point)
    opt.details = {"salt_source": how, "layers": len(layers), "tensors": len(meta),
                   "tensor_bytes": sum(p.size for p in layout.tensors),
                   "used_blocks": sum(-(-p.size // q.payload_bytes) for p in layout.tensors), "blocks": q.blocks,
                   "c_star_bytes": q.blocks * q.block_bytes, "device": str(device), "encode": store.encode,
                   "decode": "p2_stage staging decode (libp2v1.so)" if device.type == "cuda" else "schemes.p2, CPU",
                   "certifies": period.certify(store.point), "certificate": None if cert is None else cert.theorem,
                   "gaps": gaps, "experimental": P2.EXPERIMENTAL}
    return opt


__all__ = ["DEFAULT_SCHEME", "NAME", "PousOption", "SETTINGS", "commit_weights", "install", "linear_types", "refusals",
           "schemes"]
