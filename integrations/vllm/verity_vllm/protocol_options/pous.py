"""POUS as a protocol option: the served model's linear weights live only as their encoding ``C``, decoded on every use,
with a responder for the timed audit (`verity_pous`, `protocols/pous/PROTOCOL.md`).

Placeholder, 28 Sep 2026: this wires the protocol in; it certifies nothing yet.

* **The scheme** is a name `verity_pous.schemes.REGISTRY` registers; the default is the band, ``band-chain/d12/v1``.
* **What is encoded** is the weight of every ``LinearBase`` layer, found by type, deduplicated by storage and laid out
  in module order (`verity_pous.layout.Layout`).  Linears only: whether the embedding and the tied head are encoded too
  is open with Daniel, and `commitment` says what is covered.
* **Commit** (`commit_weights`, the composition's first phase, model untouched): the commitment to ``W`` before any
  salt exists (`protocol.commit`), recorded in the composition record's ``weights``.  `weights_of_record` is the same
  ``W`` for a verifier that holds the plaintext model.
* **Install**, in order: the same commitment recomputed, refused if the weights changed since; the salt, the settings'
  ``salt`` (drawn by the verifier after the commitment), else drawn after it from ``salt_source`` (the verifier's secret or
  a beacon round; without either a local random, recorded as such); ``C = Enc(W, salt)``; the storage root over ``C``;
  then, through the option's tagged `Hooks`, the plaintext released (a service whose stop restores ``W`` from ``C``), each
  layer's ``quant_method.apply`` wrapped (decode the layer's weight from ``C`` and call the inner ``apply`` with a proxy
  layer holding it), and the responder started last.  Uninstall runs newest first: responder, wrappers, then ``W`` restored.
* **The codec** (settings ``codec``): ``reference`` encodes and decodes with the scheme's reference on the CPU, and answers
  in process; ``gpu`` keeps ``C`` in device memory and uses the scheme's GPU twin (`engine/pous/twins.py`, keyed by the
  scheme id; ``kernels`` sets its dispatch switches, `twins.set_kernels`), which must first reproduce the scheme's pinned
  vector (`twins.gate`); a forward's first wrapped use decodes all
  of ``C`` on the device in one batch, each layer computes with a view of it, and after the forward's last use the decoded
  copy is zeroed (``decode_group: all``); with ``port`` set the native responder (`engine/pous/pous_native.cu`) answers
  challenges from ``C`` in device memory on its own socket and core (``host``, ``port``, ``responder_cpu``).
  ``drop_fraction`` (the negative control) zeroes that fraction of ``C``'s blocks after encoding, drawn through
  `verity.randomness` from ``drop_source``.
* **Composition:** installed after PoUW and around it, so PoUW's GEMM consumes the decoded weight.  An option that
  keeps a weight-derived copy resident across forwards is the selector's to refuse (``keeps_weight_copy``); POUS keeps
  only ``C``.
* **Weights of record:** `Installed.weights` (``decoded``) returns the released plaintext by ``named_parameters`` name,
  tied names mapped to one tensor, which the Commit's replay reads (`protocol_options.weights_view`).

What is not here yet: TP.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass, field
from typing import Any

from verity_vllm.protocol_options.interface import Context, Option, Target

NAME = "pous"
DEFAULT_SCHEME = "band-chain/d12/v1"
SETTINGS = ("salt_source", "salt", "codec", "build_dir", "kernels", "host", "port", "responder_cpu", "drop_fraction",
            "drop_source", "decode_group", "experimental")
CODECS = ("reference", "gpu")
COVERS = "linears"
SALT_BYTES = 24
DROP_DOMAIN = "verity/pous-vllm/negative-control-drop/v1"


def schemes() -> tuple[str, ...]:
    from verity_pous import schemes as pous_schemes
    return tuple(sorted(pous_schemes.REGISTRY))


def refusals(scheme: str, ctx: Context) -> tuple[str, ...]:
    from verity_pous import schemes as pous_schemes

    out, own = [], ctx.settings_of(NAME)
    stray = sorted(set(own) - set(SETTINGS))
    if stray:
        out.append(f"unknown pous settings {stray} (known: {', '.join(SETTINGS)})")
    if not ctx.enforce_eager:
        out.append("compiled execution: the decode wrappers are not graph-capturable, so pous runs eager")
    reason = pous_schemes.get(scheme).experimental
    if reason and not own.get("experimental"):
        out.append(f"{scheme} is experimental ({reason}); settings pous.experimental = true runs it anyway")
    if own.get("decode_group", "all") != "all":
        out.append(f"decode_group {own['decode_group']!r}: only 'all' is built (one batched decode of C per forward)")
    codec = own.get("codec", "reference")
    if codec not in CODECS:
        out.append(f"codec {codec!r}: one of {', '.join(CODECS)}")
    if codec == "gpu":
        from verity_vllm.engine.pous import twins
        try:
            twins.of(scheme)
        except NotImplementedError as e:
            out.append(str(e))
        if not own.get("build_dir"):
            out.append("codec gpu needs build_dir, where the native kernels are built")
        kernels = own.get("kernels", {})
        if not isinstance(kernels, Mapping) or set(kernels) - set(twins.KERNEL_FLAGS):
            out.append(f"kernels: a mapping of the kernel settings {', '.join(twins.KERNEL_FLAGS)}")
    elif own.get("port") is not None or own.get("kernels"):
        out.append("port and kernels: the native responder and the kernels serve C from device memory, so they need codec gpu")
    salt = own.get("salt")
    if salt is not None and (not isinstance(salt, str) or len(salt) != 2 * SALT_BYTES
                             or any(c not in "0123456789abcdef" for c in salt.lower())):
        out.append(f"salt: {SALT_BYTES} bytes as hex")
    frac = own.get("drop_fraction", 0.0)
    if not isinstance(frac, (int, float)) or isinstance(frac, bool) or not 0.0 <= frac < 1.0:
        out.append("drop_fraction: a number in [0, 1)")
    return tuple(out)


# --- the weights, through the tensor library ------------------------------------------------------------------------------

class _Torch:
    """A weight's bytes, its release and its rebuild, on torch tensors (vLLM's parameters)."""

    def key(self, w: Any) -> Hashable:
        return (w.untyped_storage().data_ptr(), w.storage_offset(), tuple(w.shape), str(w.dtype))

    def meta(self, w: Any) -> Any:
        return (tuple(w.shape), w.dtype, w.device)

    def to_bytes(self, w: Any) -> bytes:
        import torch
        return w.detach().contiguous().view(torch.uint8).reshape(-1).cpu().numpy().tobytes()

    def from_bytes(self, raw: bytes, meta: Any) -> Any:
        import torch
        shape, dtype, device = meta
        return torch.frombuffer(bytearray(raw), dtype=torch.uint8).view(dtype).reshape(shape).to(device)

    def release(self, param: Any, meta: Any) -> None:
        import torch
        param.data = torch.empty(0, dtype=meta[1], device=meta[2])

    def restore(self, param: Any, tensor: Any) -> None:
        param.data = tensor


TENSORS: Any = _Torch()


def linear_types() -> tuple[type, ...]:
    """The layer types whose weights POUS encodes, from vLLM (imported here, never at import time)."""
    from vllm.model_executor.layers.linear import LinearBase
    return (LinearBase,)


def _base(scheme: str) -> Any:
    from verity_pous import schemes as pous_schemes
    return pous_schemes.get(scheme).params


# --- the encoded store --------------------------------------------------------------------------------------------------

@dataclass
class _Store:
    """``W`` laid out over the unique weights, its commitment, the salt and ``C``'s blocks, on the CPU."""

    scheme: Any
    layout: Any
    salt: bytes
    commitment: bytes
    blocks: list[bytes]
    salt_source: str
    storage_root: bytes = b""
    leaves: list[bytes] = field(default_factory=list)
    dropped: list[int] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    stats: dict[str, Any] = field(default_factory=lambda: {"codec": "reference"})

    def decode(self, name: str) -> bytes:
        from verity_pous import protocol
        return protocol.decode_tensor(self.scheme, self.layout, self.salt, name, self.blocks.__getitem__)

    def tensor(self, name: str, meta: Any) -> Any:
        return TENSORS.from_bytes(self.decode(name), meta)

    def block(self, i: int) -> bytes:
        return self.blocks[i]


def _layers(model: Any) -> list[tuple[str, Any]]:
    types = linear_types()
    return [(name, m) for name, m in model.named_modules() if isinstance(m, types) and getattr(m, "weight", None) is not None]


#: A quantized linear's parameters beside its weight (vLLM's FP8 and int8 methods)
QUANT_PARAMS = ("weight_scale", "weight_scale_inv", "input_scale", "weight_zero_point", "weight_packed")


def quantized(layer: Any) -> str | None:
    """Why ``layer`` is a quantized linear POUS can't hold, or None. Its weight is codes whose meaning sits in scales that
    ``C`` does not encode, and its kernel reads a layout (FP8's column-major weight) that `TENSORS.restore`'s contiguous
    tensor does not keep, so POUS refuses it rather than serve a misread weight."""
    extra = [p for p in QUANT_PARAMS if getattr(layer, p, None) is not None]
    if extra:
        return f"a quantized linear ({', '.join(extra)})"
    dtype = getattr(layer.weight, "dtype", None)
    if dtype is not None and hasattr(dtype, "is_floating_point"):
        if not (dtype.is_floating_point and getattr(dtype, "itemsize", 0) >= 2):
            return f"a {dtype} weight (FP8 or integer codes)"
    return None


def _plaintext(entry: str, model: Any) -> tuple[list[tuple[str, Any]], dict[int, str], dict[str, bytes], dict[str, Any]]:
    """The layers, each layer's weight's name (its first holder's), and each unique weight's bytes and metadata."""
    layers = _layers(model)
    if not layers:
        raise ValueError(f"{entry}: the model has no {', '.join(t.__name__ for t in linear_types())} layer to encode")
    bad = [f"{name}: {why}" for name, m in layers if (why := quantized(m))]
    if bad:
        raise ValueError(f"{entry} refuses quantized linears (the encoding holds unquantized weights only): " + "; ".join(bad))
    first: dict[Hashable, str] = {}
    names: dict[int, str] = {}
    weights: dict[str, bytes] = {}
    meta: dict[str, Any] = {}
    for name, m in layers:
        k = TENSORS.key(m.weight)
        if k not in first:
            first[k] = name
            weights[name], meta[name] = TENSORS.to_bytes(m.weight), TENSORS.meta(m.weight)
        names[id(m)] = first[k]
    return layers, names, weights, meta


def _commit(scheme_id: str, weights: Mapping[str, bytes]) -> tuple[Any, Any, bytes, bytes]:
    """``W`` laid out, the scheme at its size, ``W`` itself and the commitment to it (no salt exists yet)."""
    from verity_pous import protocol
    from verity_pous import schemes as pous_schemes
    from verity_pous.layout import Layout

    layout = Layout.of(_base(scheme_id), [(n, len(b)) for n, b in weights.items()])
    scheme = pous_schemes.get(scheme_id, params=layout.params)
    w = layout.pack(dict(weights))
    return layout, scheme, w, protocol.commit(scheme, w)


def commit_weights(scheme: str, target: Target) -> dict[str, Any]:
    """Phase 1 of a composition: the commitment to the plaintext ``W`` of record, with the model untouched and before any
    salt exists (`protocol.commit`).  `install` refuses weights that no longer match it."""
    _, _, weights, _ = _plaintext(f"{NAME}:{scheme}", target.model)
    layout, sch, _, commitment = _commit(scheme, weights)
    return {"scheme": sch.name, "covers": COVERS, "tensors": len(weights), "bytes": sum(map(len, weights.values())),
            "commitment": commitment.hex(),
            "layout": hashlib.sha256(json.dumps(layout.manifest(), sort_keys=True).encode()).hexdigest()}


def weights_of_record(scheme: str, model: Any) -> tuple[Any, Any, bytes, bytes]:
    """``W`` as `install` lays it out, for a verifier that holds the plaintext model: the layout, the scheme at its size, ``W``
    and the commitment to it."""
    _, _, weights, _ = _plaintext(f"{NAME}:{scheme}", model)
    return _commit(scheme, weights)


def _salt(salt: bytes | None, source: bytes | None, commitment: bytes) -> tuple[bytes, str]:
    """The salt and where it came from: the verifier's, else drawn after the commitment from the settings' source, else local."""
    from verity_pous import protocol
    if salt is not None:
        return salt, "settings (the verifier's salt)"
    if source is not None:
        return protocol.draw_salt(source, commitment), "settings"
    return protocol.draw_salt(os.urandom(32), commitment), "local-random (placeholder)"


def _drop(n_blocks: int, salt: bytes, frac: float, source: str | None) -> list[int]:
    """The negative control's blocks: ``frac`` of ``C``'s, drawn through `verity.randomness` from ``source``."""
    if frac <= 0:
        return []
    from verity.randomness import derive
    key = derive(bytes.fromhex(source) if source else os.urandom(32), DROP_DOMAIN, {"salt": salt})
    return sorted(key.subset(int(round(frac * n_blocks)), n_blocks, ()))


def _root(params: Any, salt: bytes, blocks: list[bytes]) -> tuple[list[bytes], bytes]:
    from verity_pous.verifier import VerifierKey, leaf_hash
    leaves = [leaf_hash(params, salt, i, b) for i, b in enumerate(blocks)]
    return leaves, VerifierKey.from_leaves(params, salt, leaves).root


def _encode(layout: Any, scheme: Any, w: bytes, commitment: bytes, source: bytes | None, salt: bytes | None = None,
            frac: float = 0.0, drop_source: str | None = None) -> _Store:
    from verity_pous import protocol

    salt, how = _salt(salt, source, commitment)
    blocks = [b for seg in protocol.encode(scheme, w, salt) for b in seg]
    dropped = _drop(len(blocks), salt, frac, drop_source)
    for i in dropped:
        blocks[i] = bytes(len(blocks[i]))
    leaves, root = _root(scheme.params, salt, blocks)
    return _Store(scheme, layout, salt, commitment, blocks, how, root, leaves, dropped)


@dataclass
class _DeviceStore:
    """``C`` in device memory (``[segments, blocks per segment, block bytes]``) under the scheme's GPU twin, its segment tags,
    and the per-tensor decode on the device."""

    scheme: Any
    layout: Any
    salt: bytes
    commitment: bytes
    salt_source: str
    c: Any
    tags: Any
    decode_fn: Callable[[Any, Any], Any]
    storage_root: bytes = b""
    leaves: list[bytes] = field(default_factory=list)
    dropped: list[int] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    stats: dict[str, Any] = field(default_factory=dict)
    uses: int = 0
    taken: int = 0
    scratch: Any = None

    @property
    def n_blocks(self) -> int:
        return self.scheme.params.blocks

    def tensor(self, name: str, meta: Any) -> Any:
        """The tensor `name`, decoded on the device from the segments it overlaps."""
        import torch

        t0 = time.perf_counter()
        p = self.layout.params
        r, pl = self.layout.segments(name), self.layout.placement(name)
        segs = torch.arange(r.start, r.stop, device=self.c.device)
        flat = self.decode_fn(self.c[segs], self.tags[segs]).reshape(-1)
        start = pl.offset - r.start * p.segment_bytes
        shape, dtype, device = meta
        out = flat[start:start + pl.size].clone().view(dtype).reshape(shape).to(device)
        self.stats["decodes"] += 1
        self.stats["decode_s"] += time.perf_counter() - t0
        self.stats["decoded_bytes"] += len(r) * p.segment_bytes
        return out

    def decode(self, name: str) -> bytes:
        import torch
        size = self.layout.placement(name).size
        return self.tensor(name, ((size,), torch.uint8, self.c.device)).cpu().numpy().tobytes()

    def block(self, i: int) -> bytes:
        return self.c.view(self.n_blocks, -1)[i].cpu().numpy().tobytes()

    def forward_view(self, name: str, meta: Any) -> Any:
        """The tensor `name` as a view of this forward's one batched decode of every segment, made at its first use."""
        import torch

        if self.scratch is None:
            t0 = time.perf_counter()
            self.scratch = self.decode_fn(self.c, self.tags).reshape(-1)
            if self.c.device.type == "cuda":
                torch.cuda.synchronize(self.c.device)
            self.stats["decodes"] += 1
            self.stats["decode_s"] += time.perf_counter() - t0
            self.stats["decoded_bytes"] += self.scheme.params.w_bytes
        pl = self.layout.placement(name)
        shape, dtype, _ = meta
        return self.scratch[pl.offset:pl.offset + pl.size].view(dtype).reshape(shape)

    def used(self) -> None:
        """One wrapped use of the forward is done; after the last, the decoded copy is zeroed before it is freed, so no
        plaintext outlives the forward, not even in the allocator's pool."""
        self.taken += 1
        if self.taken >= self.uses and self.scratch is not None:
            self.scratch.zero_()
            self.scratch, self.taken = None, 0


def _device_store(layout: Any, scheme: Any, w: bytes, commitment: bytes, own: Mapping[str, Any], *, device: Any = None,
                  twin: tuple[Callable, Callable] | None = None, tags: Callable | None = None,
                  gate: Callable[[str, Any], dict[str, Any]] | None = None) -> _DeviceStore:
    """``C = Enc(W, salt)`` on the device with the scheme's GPU twin, once the twin reproduced its pinned vector; the negative
    control's drop; the leaves and the storage root from one host copy of ``C``.  `device`, `twin`, `tags` and `gate` replace
    the CUDA device, the kernels (`twins.of(...).load()`), `codec.segment_tags` and `twins.gate` (the CPU tests)."""
    import torch

    if twin is None:
        from verity_vllm.engine.pous import codec, native, twins
        native.load(own["build_dir"])
        twins.set_kernels(dict(own.get("kernels") or {}))
        device = torch.device("cuda", torch.cuda.current_device())
        twin, tags, gate = twins.of(scheme.name).load(), codec.segment_tags, twins.gate
    gated = gate(scheme.name, device)
    salt, how = _salt(bytes.fromhex(own["salt"]) if own.get("salt") else None,
                      bytes.fromhex(own["salt_source"]) if own.get("salt_source") else None, commitment)
    p = scheme.params
    t0 = time.perf_counter()
    seg_tags = tags(salt, list(range(p.segments)), device)
    wt = torch.frombuffer(bytearray(w), dtype=torch.uint8).to(device).view(p.segments, p.blocks_per_segment, p.block_bytes)
    c = twin[0](wt, seg_tags)
    del wt
    if device is not None and torch.device(device).type == "cuda":
        torch.cuda.synchronize()
    encode_s = time.perf_counter() - t0
    dropped = _drop(p.blocks, salt, float(own.get("drop_fraction") or 0.0), own.get("drop_source"))
    if dropped:
        c.view(p.blocks, -1)[torch.tensor(dropped, dtype=torch.long, device=c.device)] = 0
    host = c.contiguous().view(p.blocks, -1).cpu().numpy()
    leaves, root = _root(p, salt, [host[i].tobytes() for i in range(p.blocks)])
    stats = {"codec": f"GPU twin of {scheme.name}", "gate": gated, "encode_s": encode_s, "dropped": len(dropped),
             "decodes": 0, "decode_s": 0.0, "decoded_bytes": 0}
    return _DeviceStore(scheme, layout, salt, commitment, how, c, seg_tags, twin[1], root, leaves, dropped, stats=stats)


class _Proxy:
    """The layer the inner ``apply`` sees: the decoded weight, and every other attribute the layer's own."""

    def __init__(self, layer: Any, weight: Any) -> None:
        object.__setattr__(self, "_layer", layer)
        object.__setattr__(self, "weight", weight)

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_layer"), name)


@dataclass(eq=False)
class PousOption(Option):
    """The installed option: `answer` is the in-process reply to a challenge; ``weights()`` (`Option.decoded`) is the
    released plaintext decoded from ``C``; `leaves` is every block's leaf digest in block order; with the native responder,
    ``info()["responder"]`` names its host and port."""

    store: Any = None
    names: dict[int, str] = field(default_factory=dict)
    answered: int = 0
    responder: Any = None

    def answer(self, index: int) -> bytes:
        if self.closed or self.store is None:
            raise RuntimeError(f"{self.name} is closed: its responder is stopped")
        self.answered += 1
        from verity_pous import protocol
        return protocol.responder(self.store.scheme, self.store.block)(index)

    def leaves(self) -> bytes:
        return b"".join(self.store.leaves)


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


def _wrapper(decoded: Callable[[Any], Any], used: Callable[[], None] | None = None) -> Callable[[Any], Any]:
    def make(inner: Any) -> Any:
        def apply(layer: Any, x: Any, bias: Any = None) -> Any:
            out = inner(_Proxy(layer, decoded(layer)), x, bias)
            if used is not None:
                used()
            return out
        return apply
    return make


def _native_responder(store: _DeviceStore, own: Mapping[str, Any]) -> tuple[Callable[[], Any], Callable[[Any], Any]]:
    """The native responder's start and stop: challenges answered from ``C`` in device memory, on its own socket and core."""
    def start() -> Any:
        from verity_vllm.engine.pous import native
        p = store.scheme.params
        r = native.ext().Responder(store.c.data_ptr(), p.blocks, p.block_bytes, store.c.device.index, own.get("host", "127.0.0.1"),
                                   int(own["port"]), int(own.get("responder_cpu", -1)))
        r.start()
        return r

    def stop(r: Any) -> dict[str, int]:
        r.stop()
        return {"answered": int(r.answered()), "pings": int(r.pings())}
    return start, stop


def install(scheme: str, target: Target) -> PousOption:
    entry = f"{NAME}:{scheme}"
    own = target.context.settings_of(NAME)
    layers, names, weights, meta = _plaintext(entry, target.model)
    qualnames = _qualnames(entry, target.model, layers, names)
    layout, sch, w, commitment = _commit(scheme, weights)
    committed = target.weight_commitments.get(entry)
    if committed is not None and committed.get("commitment") != commitment.hex():
        raise ValueError(f"{entry}: the weights changed after the composition committed to them (commit_weights)")
    if own.get("codec", "reference") == "gpu":
        store = _device_store(layout, sch, w, commitment, own)
    else:
        source, salt = own.get("salt_source"), own.get("salt")
        store = _encode(layout, sch, w, commitment, bytes.fromhex(source) if source else None,
                        bytes.fromhex(salt) if salt else None, float(own.get("drop_fraction") or 0.0), own.get("drop_source"))
    store.meta = meta

    def plaintext() -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name, qs in qualnames.items():
            out.update(dict.fromkeys(qs, store.tensor(name, meta[name])))
        return out

    hooks = target.hooks(entry)
    opt = PousOption(name=entry, hooks=hooks, decoded=plaintext, store=store, names=names)
    params = [m.weight for _, m in layers]

    def release() -> list[Any]:
        for (_, m), p in zip(layers, params):
            TENSORS.release(p, meta[names[id(m)]])
        return params

    def restore(held: list[Any]) -> None:
        for (_, m), p in zip(layers, held):
            TENSORS.restore(p, store.tensor(names[id(m)], meta[names[id(m)]]))

    hooks.service("plaintext released", release, restore)
    if isinstance(store, _DeviceStore):
        store.uses = len(layers)
        view, used = store.forward_view, store.used
    else:
        view, used = store.tensor, None
    decoded = lambda layer: view(names[id(layer)], meta[names[id(layer)]])  # noqa: E731
    for qm in {id(m.quant_method): m.quant_method for _, m in layers}.values():
        hooks.wrap(qm, "apply", _wrapper(decoded, used))
    if own.get("port") is not None:
        opt.responder = hooks.service("responder", *_native_responder(store, own))
        responder: Any = {"kind": "native", "host": own.get("host", "127.0.0.1"), "port": int(opt.responder.handle.port()),
                          "cpu": own.get("responder_cpu")}
    else:
        opt.responder = hooks.service("responder", lambda: opt.answer, lambda handle: {"answered": opt.answered})
        responder = "in-process (placeholder)"
    opt.commits = {"scheme": store.scheme.name, "commitment": store.commitment.hex(), "segments": store.scheme.params.segments,
                   "salt": store.salt.hex(), "storage_root": store.storage_root.hex()}
    cert = store.scheme.certificate
    opt.details = {"salt_source": store.salt_source, "blocks": store.scheme.params.blocks, "layers": len(layers),
                   "certificate": None if cert is None else cert.theorem, "experimental": store.scheme.experimental,
                   "responder": responder, "dropped": len(store.dropped), "stats": store.stats,
                   "codec": store.stats["codec"] if own.get("codec") == "gpu" else "verity_pous reference, CPU (placeholder)"}
    return opt


__all__ = ["DEFAULT_SCHEME", "NAME", "PousOption", "SETTINGS", "TENSORS", "install", "linear_types", "refusals", "schemes",
           "weights_of_record"]
