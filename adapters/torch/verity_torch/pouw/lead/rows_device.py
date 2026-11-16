"""The lead's instance tops on the device, over torch tensors and the native rows extension (``native_rows.cu``):
`DeviceWords` (a forward's words instance) and `device_a_top` (A_j's), each keyed as `rows`' CPU subtree."""
from __future__ import annotations

from typing import Any

from verity.commitments.merkle import CommitmentDomain
from verity.errors import Refused

from verity.protocols.resource_accounting.compute.pouw.prover.serving.lead.rows import A_LEAF_TAG, A_SEGMENT, CHUNK_WORDS, WORDS_LEAF_TAG, chunk_count


def _tree_step(ext, dom: CommitmentDomain):
    # pads and nodes read only the domain id; the row-leaf fields, the schema among them, are never hashed here, and RowStep's
    # leaf template fits the words schema (not the A one)
    return ext.RowStep(dom.domain_id, bytes(64), bytes(128), bytes(128), bytes(128), WORDS_LEAF_TAG, 64)


def _device_top(hasher, rs, leaves, count: int, stream) -> bytes:
    """The top of an instance of ``count`` >= 2 leaves on the device: its pads (``rows_frame``) and levels (``rows_tree``) up
    to the root's two children, on ``stream``."""
    import torch

    width = 1 << (count - 1).bit_length()
    if width > count:
        pads = hasher.rows_frame(rs, leaves.new_empty((0, 64)), count, width - count, count, stream.cuda_stream)
        leaves = torch.cat([leaves, pads])
    levels, _root, _ms = hasher.rows_tree(rs, leaves.contiguous(), 0, stream.cuda_stream)
    return levels[-2].cpu().numpy().tobytes()


def device_a_top(ext, hasher, dom: CommitmentDomain, a, stream_ptr: int) -> bytes:
    """A_j's subtree top hashed on the device, from its row-major bytes ``a`` (the call's quantized activations, a CUDA tensor
    of one-byte elements) under :meth:`WindowKeying.a_domain`'s ``dom``, on the CUDA stream ``stream_ptr``: its segments'
    inline position-bound leaves (``lead_leaves``), then the pads and levels :meth:`DeviceWords.top` hashes, up to the top
    :meth:`WindowKeying.a_top` keys."""
    import torch

    count, size = dom.count, a.numel()
    if a.element_size() != 1 or not a.is_cuda:
        raise Refused(f"A_j's device top takes a CUDA tensor of one-byte elements, not {a.dtype} on {a.device}")
    if -(-size // A_SEGMENT) != count:
        raise Refused(f"A_j of {size} bytes has {-(-size // A_SEGMENT)} segments, and its instance {count}")
    if size % 8:
        raise Refused(f"A_j's device leaves take whole 8-byte words, not {size} bytes")
    main = torch.cuda.ExternalStream(stream_ptr)
    with torch.cuda.stream(main):
        flat = a.reshape(-1).view(torch.uint8)
        if count == 1:
            return flat.cpu().numpy().tobytes()
        leaves = hasher.lead_leaves(ext.LeadStep(dom.domain_id, A_LEAF_TAG), flat, A_SEGMENT, count,
                                    size - A_SEGMENT * (count - 1), 0, stream_ptr)
        return _device_top(hasher, _tree_step(ext, dom), leaves, count, main)


class DeviceWords:
    """A forward's words instance hashed on the device as its calls are served (``native_rows.cu``, through the native
    collector ``ext`` and its ``StepHasher`` ``hasher``, on the CUDA stream ``stream_ptr``).  :meth:`add` takes each call's
    export: its committed stream (``lead_stream``, the tile gather) and its chunks' inline position-bound leaves
    (``lead_leaves``) at the instance's next ranks.  :meth:`top` pads the leaves (``rows_frame``) and hashes the levels
    (``rows_tree``) up to the top :meth:`WindowKeying.words_top` keys.  ``dom`` is :meth:`WindowKeying.words_domain`'s; only
    the leaves outlive a call, 64 bytes a chunk.

    A call's leaves are few (``chunk_count``) and each hashes 256 KiB serially, so one call's leaf kernel is latency-bound.  Each
    call's gather and leaves run on the next of ``lanes`` side streams, after the work ``stream_ptr`` holds so far, so that
    calls overlap; a call's committed stream lives until its leaves are hashed, so at most ``lanes`` of them are live at once."""

    def __init__(self, ext, hasher, dom: CommitmentDomain, stream_ptr: int, *, lanes: int = 16) -> None:
        import torch

        if type(lanes) is not int or lanes < 1:
            raise ValueError(f"DeviceWords needs at least one lane, not {lanes!r}")
        self._torch, self._hasher, self._dom = torch, hasher, dom
        self._stream = torch.cuda.ExternalStream(stream_ptr)
        self._lanes = [torch.cuda.Stream(device=self._stream.device) for _ in range(lanes)]
        self._ls = ext.LeadStep(dom.domain_id, WORDS_LEAF_TAG)
        self._rs = _tree_step(ext, dom)
        self._leaves: list = []
        self._next = 0
        self._one: Any = None  # the one chunk's words on the device, where the instance has one leaf

    @property
    def count(self) -> int:
        return self._dom.count

    def add(self, export, m: int, n: int, S: int, bn: int) -> None:
        """The next call's export (``lead_stream``'s): its chunks become the instance's next leaves."""
        k = chunk_count(m, n, S)
        if self._next + k > self.count:
            raise Refused(f"a call of {k} chunks after {self._next} of a words instance of {self.count} leaves")
        side = self._lanes[len(self._leaves) % len(self._lanes)]
        side.wait_stream(self._stream)
        with self._torch.cuda.stream(side):
            words = self._hasher.lead_stream(export, m, n, S, bn, side.cuda_stream)
            size = 4 * words.numel()
            leaves = self._hasher.lead_leaves(self._ls, words.view(self._torch.uint8), 4 * CHUNK_WORDS, k,
                                              size - 4 * CHUNK_WORDS * (k - 1), self._next, side.cuda_stream)
        export.record_stream(side)
        self._leaves.append(leaves)
        if self.count == 1:
            self._one = words
        self._next += k

    def top(self) -> bytes:
        """The subtree's top: the root's two children (128 bytes), or the one chunk of a one-leaf instance."""
        torch, count = self._torch, self.count
        if self._next != count:
            raise Refused(f"a words instance of {count} leaves got {self._next}")
        main = self._stream
        for side in self._lanes:
            main.wait_stream(side)
        for t in self._leaves if count > 1 else [self._one]:
            t.record_stream(main)
        with torch.cuda.stream(main):
            if count == 1:
                return self._one.view(torch.uint8).cpu().numpy().tobytes()
            leaves = torch.cat(self._leaves) if len(self._leaves) > 1 else self._leaves[0]
            return _device_top(self._hasher, self._rs, leaves, count, main)


__all__ = ["DeviceWords", "device_a_top"]
