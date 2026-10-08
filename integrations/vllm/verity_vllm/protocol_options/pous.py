"""POUS as a protocol option: the served model's linear weights live only as their encoding ``C``, decoded on every use,
with an in-process responder for the timed audit (`verity.core.protocols.pous`, `verity/core/protocols/pous/PROTOCOL.md`).

Placeholder, 28 Sep 2026: this wires the protocol in; it certifies nothing yet.

* **The scheme** is a name `verity.core.protocols.pous.schemes.REGISTRY` registers; the default is the band, ``band-chain/d12/v1``.
* **What is encoded** is the weight of every ``LinearBase`` layer, found by type, deduplicated by storage and laid out
  in module order (`verity.core.protocols.pous.layout.Layout`).  Linears only: whether the embedding and the tied head are encoded too
  is open with Daniel, and `commitment` says what is covered.
* **Commit** (`commit_weights`, the composition's first phase, model untouched): the commitment to ``W`` before any
  salt exists (`protocol.commit`), recorded in the composition record's ``weights``.
* **Install**, in order: the same commitment recomputed, refused if the weights changed since; the salt, drawn after it from
  the settings' ``salt_source`` (the verifier's secret or a beacon round; without one a local random, recorded as
  such); ``C = Enc(W, salt)`` with the scheme's reference codec, on the CPU; the storage root over ``C``; then, through
  the option's tagged `Hooks`, the plaintext released (a service whose stop restores ``W`` from ``C``), each layer's
  ``quant_method.apply`` wrapped (decode the layer's weight from ``C`` and call the inner ``apply`` with a proxy layer
  holding it), and the responder started last.  Uninstall runs newest first: responder, wrappers, then ``W`` restored.
* **Composition:** installed after PoUW and around it, so PoUW's GEMM consumes the decoded weight.  An option that
  keeps a weight-derived copy resident across forwards is the selector's to refuse (``keeps_weight_copy``); POUS keeps
  only ``C``.
* **Weights of record:** `Installed.weights` (``decoded``) returns the released plaintext by ``named_parameters`` name,
  tied names mapped to one tensor, which the Commit's replay reads (`protocol_options.weights_view`).

What is not here yet: the GPU codec and the batched decode once per forward (#172, #188's kernels), the native
responder on its own core and port, and TP.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass, field
from typing import Any

from verity_vllm.protocol_options.interface import Context, Option, Target

NAME = "pous"
DEFAULT_SCHEME = "band-chain/d12/v1"
SETTINGS = ("salt_source", "responder_cpu", "port", "decode_group", "experimental")
COVERS = "linears"


def schemes() -> tuple[str, ...]:
    from verity.core.protocols.pous import schemes as pous_schemes
    return tuple(sorted(pous_schemes.REGISTRY))


def refusals(scheme: str, ctx: Context) -> tuple[str, ...]:
    from verity.core.protocols.pous import schemes as pous_schemes

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
        out.append(f"decode_group {own['decode_group']!r}: only 'all' is built (each layer decodes on its own use)")
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
    from verity.core.protocols.pous import schemes as pous_schemes
    return pous_schemes.get(scheme).params


# --- the encoded store --------------------------------------------------------------------------------------------------

@dataclass
class _Store:
    """``W`` laid out over the unique weights, its commitment, the salt and ``C``'s blocks."""

    scheme: Any
    layout: Any
    salt: bytes
    commitment: bytes
    blocks: list[bytes]
    salt_source: str
    storage_root: bytes = b""
    meta: dict[str, Any] = field(default_factory=dict)

    def decode(self, name: str) -> bytes:
        from verity.core.protocols.pous import protocol
        return protocol.decode_tensor(self.scheme, self.layout, self.salt, name, self.blocks.__getitem__)


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
    from verity.core.protocols.pous import protocol
    from verity.core.protocols.pous import schemes as pous_schemes
    from verity.core.protocols.pous.layout import Layout

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


def _encode(layout: Any, scheme: Any, w: bytes, commitment: bytes, source: bytes | None) -> _Store:
    from verity.core.protocols.pous import protocol
    from verity.core.protocols.pous.verifier import VerifierKey, leaf_hash

    salt = protocol.draw_salt(source if source is not None else os.urandom(32), commitment)
    blocks = [b for seg in protocol.encode(scheme, w, salt) for b in seg]
    p = scheme.params
    root = VerifierKey.from_leaves(p, salt, [leaf_hash(p, salt, i, b) for i, b in enumerate(blocks)]).root
    return _Store(scheme, layout, salt, commitment, blocks, "settings" if source is not None else "local-random (placeholder)",
                  root)


class _Proxy:
    """The layer the inner ``apply`` sees: the decoded weight, and every other attribute the layer's own."""

    def __init__(self, layer: Any, weight: Any) -> None:
        object.__setattr__(self, "_layer", layer)
        object.__setattr__(self, "weight", weight)

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_layer"), name)


@dataclass(eq=False)
class PousOption(Option):
    """The installed option: `answer` is the responder's reply to a challenge; ``weights()`` (`Option.decoded`) is the
    released plaintext decoded from ``C``."""

    store: _Store | None = None
    names: dict[int, str] = field(default_factory=dict)
    answered: int = 0

    def answer(self, index: int) -> bytes:
        if self.closed or self.store is None:
            raise RuntimeError(f"{self.name} is closed: its responder is stopped")
        self.answered += 1
        from verity.core.protocols.pous import protocol
        return protocol.responder(self.store.scheme, self.store.blocks.__getitem__)(index)


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


def _wrapper(decoded: Callable[[Any], Any]) -> Callable[[Any], Any]:
    def make(inner: Any) -> Any:
        def apply(layer: Any, x: Any, bias: Any = None) -> Any:
            return inner(_Proxy(layer, decoded(layer)), x, bias)
        return apply
    return make


def install(scheme: str, target: Target) -> PousOption:
    entry = f"{NAME}:{scheme}"
    own = target.context.settings_of(NAME)
    layers, names, weights, meta = _plaintext(entry, target.model)
    qualnames = _qualnames(entry, target.model, layers, names)
    layout, sch, w, commitment = _commit(scheme, weights)
    committed = target.weight_commitments.get(entry)
    if committed is not None and committed.get("commitment") != commitment.hex():
        raise ValueError(f"{entry}: the weights changed after the composition committed to them (commit_weights)")
    source = own.get("salt_source")
    store = _encode(layout, sch, w, commitment, bytes.fromhex(source) if source else None)
    store.meta = meta

    def plaintext() -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name, qs in qualnames.items():
            out.update(dict.fromkeys(qs, TENSORS.from_bytes(store.decode(name), meta[name])))
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
            TENSORS.restore(p, TENSORS.from_bytes(store.decode(names[id(m)]), meta[names[id(m)]]))

    hooks.service("plaintext released", release, restore)
    decoded = lambda layer: TENSORS.from_bytes(store.decode(names[id(layer)]), meta[names[id(layer)]])  # noqa: E731
    for qm in {id(m.quant_method): m.quant_method for _, m in layers}.values():
        hooks.wrap(qm, "apply", _wrapper(decoded))
    hooks.service("responder", lambda: opt.answer, lambda handle: {"answered": opt.answered})
    opt.commits = {"scheme": store.scheme.name, "commitment": store.commitment.hex(), "segments": store.scheme.params.segments,
                   "salt": store.salt.hex(), "storage_root": store.storage_root.hex()}
    cert = store.scheme.certificate
    opt.details = {"salt_source": store.salt_source, "blocks": len(store.blocks), "layers": len(layers),
                   "certificate": None if cert is None else cert.theorem, "experimental": store.scheme.experimental,
                   "responder": "in-process (placeholder)", "codec": "verity.core.protocols.pous reference, CPU (placeholder)"}
    return opt


__all__ = ["DEFAULT_SCHEME", "NAME", "PousOption", "SETTINGS", "TENSORS", "install", "linear_types", "refusals", "schemes"]
