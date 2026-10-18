"""Weights decoded at use: the model served from weights the store holds encoded in place (memory accounting's pattern,
``benchmarks/remote_session/engine.py``'s `Hooks`), installed by the store's session once it has encoded them.

~~~text
before   a forward pre-hook on each module that holds a registered weight: each of its weights' whole blocks decoded at
         their codec indices (`Decoder`) into one scratch buffer the hooks share, the tail past them (never registered)
         copied as it is, and the parameter's data pointed at its decoded bytes there
after    a forward hook points each parameter back at its own (encoded) storage
lm_head  read by the logits processor without being called: its weight's hooks are the logits processor's
~~~

The int8 linear method (`engine.int8_vllm`) reads ``layer.weight`` and ``layer.weight_scale`` in its call, so the scratch holds
the int8 weight, and the scale is decoded only if the store registered it (a scale under a block is plain).  A module that
starts while another's weights are in the scratch is refused (they would share it).  The scratch is the largest module's
weights, rounded to blocks; nothing decoded outlives its module's call.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from verity_vllm.store.interface import Layout

#: decode(src, dst, first_word, stream): src's bytes (uint8, whole codec words), codec indices first_word onward, decoded
#: into dst (as many bytes) on stream
Decoder = Callable[[Any, Any, int, int], None]


@dataclass(frozen=True)
class EncodedWeight:
    """A parameter the store holds: its whole ``blocks`` from codec index ``first_word`` (its region's), each
    ``block_words`` words of ``word_bytes``."""

    name: str
    param: Any
    first_word: int
    blocks: int
    block_words: int
    word_bytes: int


def encoded_weights(layout: Layout, model: Any, word_bytes: int | None = None) -> list[EncodedWeight]:
    """The store's weight regions that are ``model``'s parameters (a region is named for its parameter, `binder.model_weights`).
    With ``word_bytes``, their words are those of the session's codec, not the regions' own: block j's from index
    ``j * block_bytes / word_bytes`` (`remote_bandwidth.codec`'s indices, block by block from the store's first)."""
    params = dict(model.named_parameters())
    out = []
    for r in layout.regions:
        if r.kind == "weights" and r.name in params:
            wb = r.word_bytes if word_bytes is None else word_bytes
            first = r.first_word if word_bytes is None else r.first * (r.block_bytes // wb)
            out.append(EncodedWeight(r.name, params[r.name], first, r.blocks, r.block_bytes // wb, wb))
    return out


def kv_decoder(lib: Any, rounds: int, word_bytes: int) -> Decoder:
    """kv.cu's ``kv_decode`` (``lib``: a `kv_bench.Lib`, or paged.cu's build, its key set) under the seed's key at ``rounds``,
    words of ``word_bytes``."""
    return lambda src, dst, first, stream: lib.call("kv_decode", src.data_ptr(), dst.data_ptr(), src.numel() // word_bytes, first,
                                                    rounds, 0, None, stream)


class DecodeAtUse:
    """`install` the hooks that serve ``model`` from ``weights`` decoded by ``decode``; `remove` takes them out."""

    def __init__(self, model: Any, weights: Sequence[EncodedWeight], decode: Decoder) -> None:
        import torch

        self.torch, self.decode, self.calls, self.handles, self._active = torch, decode, 0, [], None
        self.weights = list(weights)
        lm_head = getattr(model, "lm_head", None)
        by_id = {id(w.param): w for w in weights}
        mods: dict[int, tuple[Any, list]] = {}
        self.home: dict[int, Any] = {}
        for mname, mod in model.named_modules():
            for _, p in mod._parameters.items():
                if p is None or id(p) not in by_id:
                    continue
                target = getattr(model, "logits_processor", None) if mod is lm_head else mod
                if target is None:
                    raise ValueError(f"{mname} holds a registered weight and no module reads it")
                w = by_id[id(p)]
                self._check(w)
                self.home[id(p)] = p.data
                mods.setdefault(id(target), (target, []))[1].append(w)
        if not mods:
            raise ValueError("no module holds a registered weight")
        if missing := [w.name for w in weights if id(w.param) not in self.home]:
            raise ValueError(f"no module of the model holds {missing[0]} ({len(missing)} registered weights)")
        self.modules = list(mods.values())
        size = max(sum(self._span(w) for w in ws) for _, ws in self.modules)
        self.scratch = torch.empty(size, dtype=torch.uint8, device=weights[0].param.device)

    @staticmethod
    def _bytes(w: EncodedWeight) -> int:
        return w.param.numel() * w.param.element_size()

    def _span(self, w: EncodedWeight) -> int:
        b = w.block_words * w.word_bytes
        return -(-self._bytes(w) // b) * b

    def _check(self, w: EncodedWeight) -> None:
        p, step = w.param, 1
        for s, n in sorted((s, n) for s, n in zip(p.stride(), p.shape) if n != 1):
            if s != step:
                raise ValueError(f"{w.name}: its elements are not one range of its storage {tuple(p.shape)} {p.stride()}")
            step *= n
        if w.blocks * w.block_words * w.word_bytes > self._bytes(w):
            raise ValueError(f"{w.name}: {w.blocks} blocks are more than its {self._bytes(w)} bytes")

    def _flat(self, t: Any) -> Any:
        es = t.element_size()
        st = self.torch.empty(0, dtype=self.torch.uint8, device=t.device).set_(t.untyped_storage())
        return st[t.storage_offset() * es:(t.storage_offset() + t.numel()) * es]

    def install(self) -> DecodeAtUse:
        for mod, ws in self.modules:
            self.handles.append(mod.register_forward_pre_hook(lambda m, a, ws=ws: self.before(m, ws)))
            self.handles.append(mod.register_forward_hook(lambda m, a, o, ws=ws: self.after(ws)))
        return self

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles = []

    def before(self, mod: Any, ws: Sequence[EncodedWeight]) -> None:
        if self._active is not None:
            raise RuntimeError(f"decode at use: {type(mod).__name__} starts while {type(self._active).__name__}'s weights are in the scratch")
        self._active = mod
        stream = self.torch.cuda.current_stream().cuda_stream if self.scratch.is_cuda else 0
        at = 0
        for w in ws:
            src, n, nb = self.home[id(w.param)], self._bytes(w), w.blocks * w.block_words * w.word_bytes
            flat = self._flat(src)
            if nb:
                self.decode(flat[:nb], self.scratch[at:at + nb], w.first_word, stream)
            if n > nb:
                self.scratch[at + nb:at + n].copy_(flat[nb:n])
            w.param.data = self.scratch[at:at + n].view(src.dtype).as_strided(src.shape, src.stride())
            at += self._span(w)
        self.calls += 1

    def after(self, ws: Sequence[EncodedWeight]) -> None:
        for w in ws:
            w.param.data = self.home[id(w.param)]
        self._active = None

    def decode_in_place(self) -> None:
        """Every weight's whole blocks decoded back where they are, through the scratch a chunk at a time (the hooks
        removed first): once the store's session is over, the model's weights are its own again."""
        if self.handles:
            raise RuntimeError("decode at use: remove the hooks before decoding the weights back in place")
        stream = self.torch.cuda.current_stream().cuda_stream if self.scratch.is_cuda else 0
        for w in self.weights:
            src, b = self.home[id(w.param)], w.block_words * w.word_bytes
            w.param.data, flat, span = src, self._flat(src), self.scratch.numel() // b * b
            for u in range(0, w.blocks * b, span):
                c = min(span, w.blocks * b - u)
                self.decode(flat[u:u + c], self.scratch[:c], w.first_word + u // w.word_bytes, stream)
                flat[u:u + c].copy_(self.scratch[:c])
        if self.scratch.is_cuda:
            self.torch.cuda.synchronize()
