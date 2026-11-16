"""POUS as a protocol option: the served model's linear weights held only as ``C* = P2-Enc(W, salt)`` at P2's registered
point (``p2-16448/v4``), decoded at every use, and each period's audit of what is held (`verity.protocols.resource_accounting.memory.pous.verifier.period`),
with the memory challenger a party of its own on the challenger host
(`verity.protocols.resource_accounting.memory.pous.attester.challenger_party`). The store, its commitment, encode and decode, and the
period are PoUS's prover's (`verity.protocols.resource_accounting.memory.pous.prover.weight_store`); this module finds the weights in
vLLM's model, attaches the decode to its layers and holds the option.

* **W** is the weight of every ``LinearBase`` layer as the engine holds it (int8 [N, K] on the ``int8_exact`` target),
  deduplicated by storage, in module order and named by its first holder, in the store's frozen layout
  (`verity/protocols/resource_accounting/memory/pous/prover/p2/p2_store/store.py`): each tensor from a block boundary over ⌈size / payload⌉ blocks, its last block zero
  past its end, and ``W`` zero to whole segments of the point's ``2^16`` blocks (`TensorLayout.of_ranges`).  Scales,
  embeddings and norms are not in ``W``.  These are the tensors the Commit registers, read through `weights_view` while
  released.
* **Commit** (`commit_weights`, the composition's first phase, model untouched): ``setup.commit(params, W)`` before any
  salt exists, recorded in the composition record's ``weights``.
* **Install**, in order: the same commitment recomputed, refused if the weights changed since; the salt
  ``setup.sample_salt(source, commitment)``, ``source`` the auditor's coin flipped after the commitment
  (``salt_source``); ``C*`` encoded a segment at a time (`store.GpuEncoder`, ``libp2v1.so`` on the weights' GPU, which
  checks its own decode of the whole segment; `store.CpuEncoder` on the CPU), two blocks of each segment held to ``W``
  by the reference decode, and ``C*`` held as one [blocks, block_bytes] uint8 tensor on the weights' device; the memory
  challenger's table of it and the root over that table (`VerifierKey.from_row_digests`).  Then, through the option's
  tagged `Hooks`: the plaintext released (a service whose stop restores ``W`` decoded from ``C*``), each layer's
  ``quant_method.apply`` wrapped (the layer's weight decoded from ``C*`` into a fresh buffer, by the store's load
  `p2_store.load.EncodedRows.decode_into` on the GPU, refusing what the reference refuses, or the scheme's codec on the CPU,
  and the inner ``apply`` called with a proxy layer holding it), and the prover started last.  Uninstall runs newest
  first: prover, wrappers, then ``W`` restored; ``C*`` is freed after.
* **The period** (`PousOption.audit`, `PousOption.subprotocol`): `period_prover.audit` of what setup published, the prover answering
  each index from ``C*`` where it is held, and the verifier's answer: setup's own inputs tied to the run (`period.tie`,
  the public-encoder check of ``C*``), the run's check, and the bits an accepted run certifies (`period.certify`: 0 at a
  point no certificate covers).  In `PousOption.subprotocol` the memory challenger runs at ``period.json``'s ``challenger_host``
  (`side_of`, which the caller that starts the run hands in) and the Lean answers where the period's tools name it.
* **Composition:** installed after PoUW and around it, so PoUW's GEMM consumes the decoded weight.  An option that
  keeps a weight-derived copy resident across forwards is the selector's to refuse (``keeps_weight_copy``); POUS keeps
  only ``C*``.
* **Weights of record:** `Installed.weights` (``decoded``) returns the released plaintext by ``named_parameters`` name,
  tied names mapped to one tensor, which the Commit's replay reads (`protocols.weights_view`).

Settings: ``salt_source`` (the auditor's coin: its hex, a callable of the commitment, or ``{"command": argv}``, a process
that reads ``{"commitment", "manifest"}`` as one JSON line and prints ``{"source": HEX}``), ``p2_lib`` (``libp2v1.so``'s
directory, built for the card by the store's ``verity/protocols/resource_accounting/memory/pous/prover/p2/p2_store/Makefile``; weights on a GPU
need it) and
``experimental`` (P2 is; `schemes.p2.EXPERIMENTAL` says why).

Not here yet: TP (each rank's shard), the head and the embedding (not ``LinearBase``), and a setup case that scales:
`period.setup_case` carries ``C*`` in hex, 2.2 GB at the registered point, and the tie decodes all of it in Python each
period.
"""
from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from verity.protocols.resource_accounting.memory.pous.prover import weight_store as ws

from verity_vllm.protocols.interface import Context, Option, Target

NAME = "pous"
DEFAULT_SCHEME = "p2-16448/v4"
SETTINGS = ("salt_source", "p2_lib", "experimental")
COVERS = "linears"


def schemes() -> tuple[str, ...]:
    from verity.protocols.resource_accounting.memory.pous.verifier import (
        schemes as pous_schemes,
    )
    return tuple(sorted(pous_schemes.REGISTRY))


def _source_refusal(source: Any) -> str | None:
    """Why ``salt_source`` can't give the auditor's coin, or None."""
    if source is None:
        return ("no salt_source: the salt is setup.sample_salt(source, commitment), the source the auditor's coin drawn after "
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
    from verity.protocols.resource_accounting.memory.pous.verifier import (
        schemes as pous_schemes,
    )

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


def commit_weights(scheme: str, target: Target) -> dict[str, Any]:
    """Phase 1 of a composition: the commitment to the plaintext ``W`` of record, with the model untouched and before any
    salt exists (`setup.commit`).  `install` refuses weights that no longer match it."""
    _, _, tensors = _plaintext(f"{NAME}:{scheme}", target.model)
    layout, commitment = ws.commit(scheme, tensors)
    return {"scheme": scheme, "covers": COVERS, "tensors": len(tensors), "bytes": sum(p.size for p in layout.tensors),
            "commitment": commitment.hex(), "layout": ws.layout_digest(layout)}


# --- the encoded store ---------------------------------------------------------------------------------------------------

class _Proxy:
    """The layer the inner ``apply`` sees: the decoded weight, and every other attribute the layer's own."""

    def __init__(self, layer: Any, weight: Any) -> None:
        object.__setattr__(self, "_layer", layer)
        object.__setattr__(self, "weight", weight)

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_layer"), name)


@dataclass(eq=False)
class PousOption(Option):
    """The installed option: `answer` is the prover's reply to a challenge, `audit` and `subprotocol` a period's audit of what it
    holds; ``weights()`` (`Option.decoded`) is the released plaintext decoded from ``C*``."""

    store: ws.Store | None = None
    names: dict[int, str] = field(default_factory=dict)
    answered: int = 0
    periods: int = 0

    def answer(self, index: int) -> bytes:
        if self.closed or self.store is None:
            raise RuntimeError(f"{self.name} is closed: its prover is stopped")
        self.answered += 1
        return ws.answer(self.store, index)

    def prover(self, blocks: Sequence[bytes]) -> Callable[[int], bytes]:
        """`period_prover.Prover`: each index answered from ``C*`` where it is held (``blocks`` is the same store)."""
        return self.answer

    def audit(self, node: str, coins: bytes, **kw: Any) -> Any:
        """`weight_store.audit` of this setup on ``node`` under the auditor's ``coins``; ``kw`` as `period_prover.audit`'s (the
        prover by default this option's)."""
        if self.closed or self.store is None:
            raise RuntimeError(f"{self.name} is closed: nothing is held to audit")
        kw.setdefault("prover", self.prover)
        return ws.audit(self.store, node, coins, **kw)

    def subprotocol(self, period_dir: str | Path, node: str, out: str | Path, *, side_of: Callable[..., Any],
            schemes: Mapping[str, Any] | None = None, protocol: str | None = None) -> dict[str, Any]:
        """One period's PoUS of what this option holds (`weight_store.subprotocol`), the first this option's ``fresh``."""

        def audited(node: str, coins: bytes, **kw: Any) -> Any:
            got = self.audit(node, coins, **kw)
            self.periods += 1
            return got

        return ws.subprotocol(self.store, period_dir, node, out, audit=audited, side_of=side_of, fresh=self.periods == 0,
                      answered=lambda: self.answered, schemes=schemes, protocol=protocol)


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
    from verity.protocols.resource_accounting.memory.pous.verifier import period, setup
    from verity.protocols.resource_accounting.memory.pous.verifier.schemes import (
        p2 as P2,
    )

    entry = f"{NAME}:{scheme}"
    own = target.context.settings_of(NAME)
    why = _source_refusal(own.get("salt_source"))
    if why:
        raise ValueError(f"{entry}: {why}")
    layers, names, tensors = _plaintext(entry, target.model)
    qualnames = _qualnames(entry, target.model, layers, names)
    layout, commitment = ws.commit(scheme, tensors)
    committed = target.weight_commitments.get(entry)
    if committed is not None and committed.get("commitment") != commitment.hex():
        raise ValueError(f"{entry}: the weights changed after the composition committed to them (commit_weights)")
    source, how = ws.salt_from(entry, own["salt_source"], commitment, layout.params)
    store = ws.encode(layout, tensors, commitment, setup.sample_salt(source, commitment), how, own.get("p2_lib"))
    decode = ws.decoder(store, own.get("p2_lib"))
    meta = {name: (tuple(t.shape), t.dtype) for name, t in tensors.items()}
    device = store.rows.device

    def tensor(name: str) -> Any:
        shape, dtype = meta[name]
        out = torch.empty(shape, dtype=dtype, device=device)
        decode(layout.placement(name), ws.flat(out))
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
                   "layout": ws.layout_digest(layout)}
    cert, gaps = period.gaps(store.point)
    opt.details = {"salt_source": how, "layers": len(layers), "tensors": len(meta),
                   "tensor_bytes": sum(p.size for p in layout.tensors),
                   "used_blocks": sum(-(-p.size // q.payload_bytes) for p in layout.tensors), "blocks": q.blocks,
                   "c_star_bytes": q.blocks * q.block_bytes, "device": str(device), "encode": store.encode,
                   "decode": "p2_store.load (libp2v1.so)" if device.type == "cuda" else "schemes.p2, CPU",
                   "certifies": period.certify(store.point), "certificate": None if cert is None else cert.theorem,
                   "gaps": gaps, "experimental": P2.EXPERIMENTAL}
    return opt


__all__ = ["DEFAULT_SCHEME", "NAME", "PousOption", "SETTINGS", "commit_weights", "install", "linear_types", "refusals",
           "schemes"]
