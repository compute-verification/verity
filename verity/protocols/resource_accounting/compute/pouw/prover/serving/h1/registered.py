"""H1's served tables registered at serving scale: `registered_value.Registered.commit` of rows of 16-bit words as
``hm96-sha512/row/v2`` bit rows, row j salted with its 192 given bytes (the run's fresh salts at the value's nonce,
`pouw_h1_prover.RunSalts`), with the same salts, ``b || c`` strings, tree and entry, computed from the rows' NumPy words instead of Python ints.

* **Off the GPU** the ``b || c`` strings and the tree's levels are the commitments' (`periods.rows.registered_strings`).
* **The salts.** The caller's: fresh operating-system bytes, 192 a row, handed in as data.
* **What it holds.** The ``b || c`` strings and the tree's levels, 64 bytes a node. The salts are held as the bytes handed in
  (`Salts`), or a public table's are sampled again on read (`PublicSalts`), and the rows are kept only with ``keep_rows`` (`Rows`), or stand for
  themselves (``rows``, a public table's, built again on read): a window's call tables and its own ``b/<K>`` are
  rebuilt by the prover from what it keeps (`pouw_h1_prover.keep_call`, its weights), so `Served` keeps none of their
  rows and a window holds a few hundred bytes a row (its ``b || c`` and about two tree nodes, as Python bytes), where
  Python ints held some 36 bytes a word. The period's `Weights` keeps its rows, two bytes a word, for `Weights.matches`.

* **On the GPU**: where this process has CUDA, `commit` hashes the same strings, leaves and levels on the device
  (`pouw_h1_registered_device`, `pouw_cuda/registered_rows.cu`), a thread a row and a node, and only the domain id and
  the result's bytes touch the host (`path`, `MODE`), and a public table's salts (`PublicSalts`) are derived there,
  in the buffer the hash reads. With
  ``later`` it hands back the hash still running on the device's stream (`Later`), for a caller that reads the root
  only when it needs it. The commitments' path is the reference the device is held to
  (``tests/protocols/test_pouw_h1_registered_device.py``) and the fallback everywhere else.

`commit` returns a `registered_value.Registered` whose ``rows`` and ``salts`` are those sequences; everything `Served`,
`ServedTree.of`, `registered.verify_read` and the record read of it (``root``, ``levels``, ``bcs``, ``entry()``,
``path``) is what `Registered.commit` gives over the same rows and salts (``tests/protocols/
test_pouw_h1_registered.py`` holds it to core's on small windows).
"""
from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from verity.commitments import hm96
from verity.protocols.proofs.spot.verifier.one_stage import registered as G
from verity.commitments import registered_value as RV
from verity.commitments.periods.rows import KeptRows as Rows, Salts, registered_strings

from verity.protocols.resource_accounting.compute.pouw.prover.serving.tensors import dtype_name, h1_device, tensors

_INST = hm96.SHA512
#: `commit`'s path (`path`): ``"auto"``, ``"gpu"`` or ``"numpy"``, set by the caller that configures the run
MODE = "auto"


def _public_salt(head: bytes, position: int, ok: bool) -> bytes:
    head += position.to_bytes(8, "big")
    return b"".join(hashlib.sha512(head + i.to_bytes(4 if ok else 1, "big")).digest()
                    for i in range(_INST.salt_bytes // 64))


@dataclass(frozen=True)
class PublicSalts(Sequence):
    """A public table's salts, which anyone derives: item j is position ``first + j``'s, ``SHA-512(head || u64be(first +
    j) || u8(i))`` for i = 0, 1, 2 (`core_check.public_row_salt`, ``head`` its domain, context length and context), or
    with ``ok`` ``SHA-512(head || u64be(first + j) || u32be(i))`` (`core_check.ok_salt`, ``head`` its tag, context and
    0), sampled on read; `commit` derives them where it hashes (`public_salts`). Distinct positions, so no salt is
    reused."""

    head: bytes
    first: int
    n: int
    ok: bool = False

    def __post_init__(self) -> None:
        if self.n < 1 or self.first < 0 or self.first + self.n > 1 << 64:
            raise ValueError(f"{self.n} public salts from position {self.first}: at least one, every position a u64")

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, j: Any) -> Any:
        if isinstance(j, slice):
            return [self[i] for i in range(*j.indices(self.n))]
        if not -self.n <= j < self.n:
            raise IndexError(j)
        return _public_salt(self.head, self.first + j % self.n, self.ok)

    def at(self, positions: Sequence[int] | np.ndarray) -> bytes:
        """Items ``positions`` (each in [0, n)) concatenated."""
        at = [int(p) for p in positions]
        if at and (min(at) < 0 or max(at) >= self.n):
            raise IndexError(f"a position outside the table's {self.n} rows")
        return b"".join(self[p] for p in at)


def path() -> str:
    """Where `commit` hashes, `MODE`'s choice: ``"gpu"`` (`pouw_h1_registered_device`, refused where it can't run),
    ``"numpy"``, or by default ``"auto"``, the GPU where the workload handed in its H1 device (`tensors.use_h1`) and it can
    hash here (`tensors.H1Device.available`), else NumPy. Both register the same bytes; NumPy is the reference."""
    mode = MODE
    if mode not in ("auto", "gpu", "numpy"):
        raise ValueError(f"MODE={mode!r}: auto, gpu or numpy")
    if mode != "auto":
        return mode
    try:
        device = h1_device()
    except NotImplementedError:
        return "numpy"
    return "gpu" if device.available() else "numpy"


def _host(words: Any) -> np.ndarray:
    return np.asarray(words) if isinstance(words, np.ndarray) or not hasattr(words, "dtype") else tensors().host(words)


class Later:
    """A `commit` whose hash runs on the GPU meanwhile (its ``later``): ``ready()`` without blocking, ``result()`` its
    `registered_value.Registered`, waited for."""

    def __init__(self, hashing: Any, made: Callable[..., RV.Registered]) -> None:
        self.hashing, self.made = hashing, made

    def ready(self) -> bool:
        return self.hashing.ready()

    def result(self) -> RV.Registered:
        return self.made(*self.hashing.result())

    def also(self) -> bytes:
        """With `commit`'s ``also``, that table's root, waited for."""
        return self.hashing.also()


def commit(program: str, value: str, words: Any, row_salts: bytes | PublicSalts, keep_rows: bool = False,
           schema: str = G.SCHEMA_V2, later: bool = False, rows: Sequence | None = None,
           also: tuple[bytes, bytes] | None = None, **binding: Any) -> RV.Registered | Later:
    """`registered_value.Registered.commit(program, value, rows, salts, schema=schema, **binding)` of ``words`` ``(n, W)``
    u16, ``rows`` their rows and ``salts`` ``row_salts``: n salts concatenated, each used once (the run's fresh bytes,
    `pouw_h1_prover.RunSalts.at` the value's nonce), or a public table's `PublicSalts`, which the GPU path derives where
    it hashes. ``schema``: ``SCHEMA_V2``, or ``SCHEMA_SEG`` (an
    embedding table's rows, hashed with NumPy wherever `path` says). ``binding``: the domain's other arguments
    (``window``, ``produced``, ``session``). ``keep_rows``: hold the rows (`Rows`), else none; ``rows``: hold these,
    a sequence of the same n rows that builds them on read. Hashed where `path` says;
    ``words`` an array, or a u16 tensor, which the GPU path hashes where it is when that is a CUDA device and every
    other path reads on the host. ``later``: on the GPU path, a `Later` of it, the hash left running on the device's
    stream; elsewhere the `Registered` itself. ``also`` ``(domain id, salts)``: the rows (a call's z/y records, on the
    device) are its call z table's too, committed from the same read (`pouw_h1_registered_device.device_hash`'s), that
    root `Later.also`; only with ``later`` on the GPU path."""
    if schema not in (G.SCHEMA_V2, G.SCHEMA_SEG):
        raise ValueError(f"rows of {schema!r}: {G.SCHEMA_V2} or {G.SCHEMA_SEG}")
    gpu = schema == G.SCHEMA_V2 and path() == "gpu"
    if also is not None and not (gpu and later and getattr(words, "is_cuda", False)):
        raise ValueError("also: a call's z table is committed beside its value only by a later hash of device rows on "
                         "the GPU path")
    if not (gpu and getattr(words, "is_cuda", False)):
        words = _host(words)
    if len(words.shape) != 2 or not words.shape[0] or not words.shape[1] or dtype_name(words) != "uint16":
        raise ValueError("a registered value is at least one row of 16-bit words, every row the same width")
    n, w = (int(d) for d in words.shape)
    dom = RV.domain(program, value, w, n, RV.WORD_BITS, **binding)
    public = row_salts if isinstance(row_salts, PublicSalts) else None
    given = None if public is not None else bytes(row_salts)
    if (given is not None and (len(given) != 192 * n or len({given[192 * j:192 * (j + 1)] for j in range(n)}) != n)
            or public is not None and len(public) != n):
        raise ValueError(f"one {_INST.salt_bytes}-byte salt per row, never reused")
    if rows is not None and len(rows) != n:
        raise ValueError(f"{len(rows)} rows held for a value of {n}")

    def made(bcs: Sequence[bytes], levels: Sequence[Sequence[bytes]]) -> RV.Registered:
        held = rows if rows is not None else Rows(np.ascontiguousarray(_host(words)) if keep_rows else None, n)
        salts = public if public is not None else Salts(given, n)
        return RV.Registered(program=program, value=value, words=int(w), word_bits=RV.WORD_BITS, rows=held, salts=salts,
                            bcs=bcs, levels=levels, schema=schema, **binding)
    if gpu:
        hashing = h1_device().device_hash(words, given, dom.domain_id,
                                None if public is None else (public.head, public.first, public.ok), also)
        return Later(hashing, made) if later else made(*hashing.result())
    if public is not None:
        given = public_salts(public.head, public.first, n, public.ok)
    return made(*registered_strings(words, given, dom, schema))


def public_salts(head: bytes, first: int, n: int, ok: bool = False) -> bytes:
    """Positions ``first`` .. ``first + n - 1``'s public row salts concatenated, `PublicSalts`' items, on the device where
    `path` says, else with `hashlib`."""
    if n < 1 or first < 0 or first + n > 1 << 64:
        raise ValueError(f"{n} public salts from position {first}: at least one, every position a u64")
    if path() == "gpu":
        return h1_device().device_public_salts(head, first, n, ok)
    return b"".join(_public_salt(head, u, ok) for u in range(first, first + n))


__all__ = ["MODE", "Later", "PublicSalts", "Rows", "Salts", "commit", "path", "public_salts"]
