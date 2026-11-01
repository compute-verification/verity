"""Weights decoded at use: the model served from weights the store holds encoded in place (memory accounting's pattern,
``benchmarks/poub_session/engine.py``'s `Hooks`), installed by the store's session once it has encoded them.

~~~text
before   a forward pre-hook on each module that holds a registered weight: each of its weights' whole blocks decoded at
         their codec indices (`Decoder`) into one scratch buffer the hooks share, the tail past them (never registered)
         copied as it is, and the parameter's data pointed at its decoded bytes there
after    a forward hook points each parameter back at its own (encoded) storage
lm_head  read by the logits processor without being called: its weight's hooks are the logits processor's
~~~

The int8 linear method (`engine.int8_vllm`) reads ``layer.weight`` and ``layer.weight_scale`` in its call, so the scratch holds
the int8 weight, and the scale is decoded only if the store registered it (a scale under a block is plain).  A module that
starts while another's weights are in the scratch is refused (they would share it).  Nothing decoded outlives its module's
call.

The scratch is the largest other module's weights, rounded to blocks (``scratch_bytes`` overrides it), not the vocabulary's:
when every byte of a weight is registered and its rows are whole blocks, two modules read theirs in pieces.

~~~text
embedding  (one shard) handed a stand-in quant method for its call, whose `embedding` decodes only the rows of the call's
           token ids into the scratch as a [rows, H] table and runs the method's own `embedding` on each id's row in it:
           the same gather (`VocabParallelEmbedding` at tp 1 is `quant_method.embedding(layer, ids)`, `F.embedding`)
lm_head    the logits processor handed a stand-in LM head whose `quant_method.apply` decodes the weight a scratch of rows at
           a time and runs the head's own `apply` on each (with those rows of an int8 head's plain `weight_scale`), its logits
           written into theirs; any other attribute is the head's
~~~

A module's arguments are never changed: the acquisition's hooks on it commit its call as the model made it (the
embedding's token ids).  The pieces' bits are the whole call's when the GEMM computes each logit alone, as cuBLAS does for these shapes on the RTX PRO
6000 (`tests/engine/int8_store_gpu.py`).
"""
from __future__ import annotations

import types
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from verity_vllm.protocols.poub.interface import Layout

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
    ``j * block_bytes / word_bytes`` (`poub.codec`'s indices, block by block from the store's first)."""
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


def _rows_whole(w: EncodedWeight) -> int | None:
    """The bytes of a row of ``w`` when every byte is registered and a row is whole blocks, else None."""
    p = w.param
    block, n = w.block_words * w.word_bytes, p.numel() * p.element_size()
    if p.dim() != 2 or not p.is_contiguous() or w.blocks * block != n:
        return None
    row = p.shape[1] * p.element_size()
    return row if row % block == 0 else None


class _Head:
    """The LM head the logits processor is handed: ``quant_method`` is this object, every other attribute the head's."""

    def __init__(self, uses: DecodeAtUse, head: Any, w: EncodedWeight, row: int) -> None:
        self._uses, self._head, self._w, self._row = uses, head, w, row
        self.quant_method = self

    def __getattr__(self, name: str) -> Any:
        return getattr(self._head, name)

    def apply(self, layer: Any, x: Any, bias: Any = None) -> Any:
        u, w, row = self._uses, self._w, self._row
        qm, src, (V, H) = self._head.quant_method, u.home[id(w.param)], w.param.shape
        flat, step, stream, out = u._flat(src), u.scratch.numel() // row, u.stream(), None
        scale = getattr(self._head, "weight_scale", None)
        for a in range(0, V, step):
            b = min(V, a + step)
            u.decode(flat[a * row:b * row], u.scratch[:(b - a) * row], w.first_word + a * row // w.word_bytes, stream)
            rows = {"weight": u.scratch[:(b - a) * row].view(src.dtype).view(b - a, H)}
            if scale is not None:                                  # the int8 head's per-row scales, plain beside its weight
                rows["weight_scale"] = scale[a:b]
            y = qm.apply(types.SimpleNamespace(**rows), x, None if bias is None else bias[a:b])
            out = y.new_empty((*y.shape[:-1], V)) if out is None else out
            out[..., a:b] = y
        return out


class _Rows:
    """The embedding's quant method for one call: `embedding` decodes the rows of its token ids, every other attribute is
    the method's."""

    def __init__(self, uses: DecodeAtUse, qm: Any, w: EncodedWeight, row: int) -> None:
        self._uses, self._qm, self._w, self._row = uses, qm, w, row

    def __getattr__(self, name: str) -> Any:
        return getattr(self._qm, name)

    def embedding(self, layer: Any, ids: Any) -> Any:
        u, w, row = self._uses, self._w, self._row
        rows, inv = u.torch.unique(ids, sorted=True, return_inverse=True)
        if rows.numel() * row > u.scratch.numel():
            raise RuntimeError(f"decode at use: {rows.numel()} embedding rows of {row} bytes do not fit the {u.scratch.numel()}-byte scratch")
        src = u.home[id(w.param)]
        flat, stream, at, ids_h = u._flat(src), u.stream(), 0, rows.tolist()
        i = 0
        while i < len(ids_h):
            j = i + 1
            while j < len(ids_h) and ids_h[j] == ids_h[j - 1] + 1:
                j += 1
            a, n = ids_h[i] * row, (j - i) * row
            u.decode(flat[a:a + n], u.scratch[at:at + n], w.first_word + a // w.word_bytes, stream)
            at, i = at + n, j
        table = u.scratch[:at].view(src.dtype).view(len(ids_h), src.shape[1])
        return self._qm.embedding(types.SimpleNamespace(weight=table), inv.reshape(ids.shape).to(ids.dtype))


class DecodeAtUse:
    """`install` the hooks that serve ``model`` from ``weights`` decoded by ``decode``; `remove` takes them out."""

    def __init__(self, model: Any, weights: Sequence[EncodedWeight], decode: Decoder, *, scratch_bytes: int | None = None) -> None:
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
        self.pieces: dict[int, tuple[str, Any, int]] = {}
        for target, ws in mods.values():
            row = _rows_whole(ws[0]) if len(ws) == 1 else None
            if row is None:
                continue
            if lm_head is not None and target is getattr(model, "logits_processor", None) and ws[0].param is lm_head.weight:
                self.pieces[id(target)] = ("head", lm_head, row)
            elif getattr(target, "tp_size", None) == 1 and hasattr(getattr(target, "quant_method", None), "embedding") \
                    and ws[0].param is getattr(target, "weight", None):
                self.pieces[id(target)] = ("rows", target, row)
        self.modules = list(mods.values())
        whole = [sum(self._span(w) for w in ws) for t, ws in self.modules if id(t) not in self.pieces]
        size = scratch_bytes or max(whole or [max(row for _, _, row in self.pieces.values())])
        if any(size < row for _, _, row in self.pieces.values()) or (whole and size < max(whole)):
            raise ValueError(f"decode at use: a {size}-byte scratch is smaller than a module's weights or a row")
        self.scratch = torch.empty(size, dtype=torch.uint8, device=weights[0].param.device)

    def stream(self) -> int:
        return self.torch.cuda.current_stream().cuda_stream if self.scratch.is_cuda else 0
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
            self.handles.append(mod.register_forward_pre_hook(lambda m, a, ws=ws: self.before(m, ws, a)))
            self.handles.append(mod.register_forward_hook(lambda m, a, o, ws=ws: self.after(ws, m)))
        return self

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles = []

    def before(self, mod: Any, ws: Sequence[EncodedWeight], args: tuple = ()) -> tuple | None:
        if self._active is not None:
            raise RuntimeError(f"decode at use: {type(mod).__name__} starts while {type(self._active).__name__}'s weights are in the scratch")
        self._active = mod
        self.calls += 1
        piece = self.pieces.get(id(mod))
        if piece is not None:
            return self._piece(piece, ws[0], args)
        stream, at = self.stream(), 0
        for w in ws:
            src, n, nb = self.home[id(w.param)], self._bytes(w), w.blocks * w.block_words * w.word_bytes
            flat = self._flat(src)
            if nb:
                self.decode(flat[:nb], self.scratch[at:at + nb], w.first_word, stream)
            if n > nb:
                self.scratch[at + nb:at + n].copy_(flat[nb:n])
            w.param.data = self.scratch[at:at + n].view(src.dtype).as_strided(src.shape, src.stride())
            at += self._span(w)
        return None

    def _piece(self, piece: tuple[str, Any, int], w: EncodedWeight, args: tuple) -> tuple | None:
        kind, mod, row = piece
        if kind == "head":
            if not args or args[0] is not mod:
                raise RuntimeError("decode at use: the logits processor was not handed the LM head as its first argument")
            return (_Head(self, mod, w, row), *args[1:])
        mod.quant_method = _Rows(self, mod.quant_method, w, row)
        return None

    def after(self, ws: Sequence[EncodedWeight], mod: Any = None) -> None:
        for w in ws:
            w.param.data = self.home[id(w.param)]
        if isinstance(getattr(mod, "quant_method", None), _Rows):
            mod.quant_method = mod.quant_method._qm
        self._active = None

    def decode_in_place(self) -> None:
        """Every weight's whole blocks decoded back where they are, through the scratch a chunk at a time (the hooks
        removed first): once the store's session is over, the model's weights are its own again."""
        if self.handles:
            raise RuntimeError("decode at use: remove the hooks before decoding the weights back in place")
        stream = self.stream()
        for w in self.weights:
            src, b = self.home[id(w.param)], w.block_words * w.word_bytes
            w.param.data, flat, span = src, self._flat(src), self.scratch.numel() // b * b
            for u in range(0, w.blocks * b, span):
                c = min(span, w.blocks * b - u)
                self.decode(flat[u:u + c], self.scratch[:c], w.first_word + u // w.word_bytes, stream)
                flat[u:u + c].copy_(self.scratch[:c])
        if self.scratch.is_cuda:
            self.torch.cuda.synchronize()
