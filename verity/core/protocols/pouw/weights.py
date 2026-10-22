"""An H1 period's registered weights (VLLM-71, RT4-INT's model registry): each contraction K's ``<prefix>b/<K>``, the B
rows of the window Program's linears of that K, and `weights_root`, their fold, the weights' root a period-open signs.

* **A B row** (`row_bytes`) is one output row of a linear: its K int8 codes, then the 4 bytes of its f32 scale s_w,
  little-endian (`h1zy.ROW`): `b_words(K)` u16 words, as a ``hm96-sha512/row/v2`` bit row is its bytes.
* **``<prefix>b/<K>``** is the rows of K's linears, linear after linear in the window's order, committed as
  `registered.Registered.commit` commits them: under `registered.domain` of the circuit's SHA-512, the name, `b_words(K)`
  words and Σn leaves, binding no window (a value held across windows), in ``hm96-sha512/row/v2``, row j's salt
  `audit.hm96_salt` of the registration's 32-byte key at `B_NONCE` plus K's place among the Ks, ascending.
* `weights_roots` computes each K's root streaming: it keeps each row's 64-byte leaf, never the rows, so the auditor's
  host commits Llama-3.1-8B's 1.39M rows in a few hundred MB. Its roots are `Registered.commit`'s, and `entry` is the
  registration's entry (`Registered.entry`), which credit's `_weights` checks.
* `weights_root` is credit's fold of the roots: `tagged` of ``verity/pouw/h1/weights/v1`` and each ``K, root`` by
  ascending K.
"""
from __future__ import annotations

from typing import Any, Callable, Iterable, Mapping, Sequence

from verity.core.protocols.pouw.audit import hm96_salt
from verity.core.protocols.pouw.protocol import tagged
from verity.core.protocols.sampled_proofs.one_stage import registered as G
from verity.ml.commitments import hm96
from verity.ml.commitments.merkle import tree_levels
from verity.ml.commitments.rowleaf import sha512_row_v2_digest

#: an H1 window's weights' root: each K's registered ``…/b/<K>`` root
WEIGHTS_TAG = "verity/pouw/h1/weights/v1"
#: ``b/<K>``'s first salt nonce, past every round's and call's tables in the prover's salt stream (vllm's
#: ``REQUESTS_NONCE + 2^42 + 2^40``), so a key that salts both never repeats a (nonce, leaf)
B_NONCE = (1 << 61) + (1 << 42) + (1 << 40)
SCHEMA = G.SCHEMA_V2
_INST = hm96.SHA512


def b_words(k: int) -> int:
    """A B row's 16-bit words: its K int8 codes, then the 4 bytes of its f32 s_w."""
    return (k + 5) // 2


def value(prefix: str, k: int) -> str:
    return f"{prefix}b/{k}"


def row_bytes(codes: bytes, scale: bytes) -> bytes:
    """A B row's bit row: its K int8 codes' bytes, then its f32 s_w's 4 bytes little-endian; ValueError on an odd K or a
    scale that isn't 4 bytes."""
    if len(codes) % 2 or not codes:
        raise ValueError(f"K = {len(codes)}: a row's codes are whole u16 words")
    if len(scale) != 4:
        raise ValueError(f"an s_w is an f32's 4 bytes, not {len(scale)}")
    return bytes(codes) + bytes(scale)


def row_words(row: bytes) -> list[int]:
    """A B row's u16 words, as `Registered.commit` takes them: its bytes little-endian in pairs."""
    return [row[i] | row[i + 1] << 8 for i in range(0, len(row), 2)]


def ks(linears: Sequence[tuple[int, int]]) -> list[int]:
    """The Ks of the window's linears (``(n, K)`` each), ascending; ValueError on an odd K or an empty linear."""
    if not linears or any(type(n) is not int or type(k) is not int or n < 1 or k < 2 or k % 2 for n, k in linears):
        raise ValueError("the window's linears are (n, K), n >= 1 and K a positive even number each")
    return sorted({k for _n, k in linears})


def weights_roots(circuit: str, prefix: str, linears: Sequence[tuple[int, int]], rows: Callable[[int], Iterable[bytes]],
                  key: bytes, *, progress: Callable[[int, int], None] | None = None) -> dict[int, bytes]:
    """Each K's ``<prefix>b/<K>`` root: `linears` the window's linears in its order, ``(n, K)`` each; ``rows(w)`` linear w's
    n rows, each `row_bytes`; `key` the registration's 32-byte salt key. Equal to `Registered.commit`'s root over the same
    rows and salts. ValueError on a row of another width, a linear of another row count, or a key not 32 bytes."""
    if len(key) != 32:
        raise ValueError(f"a salt key is 32 bytes, not {len(key)}")
    out = {}
    for i, k in enumerate(ks(linears)):
        mine = [w for w, (_n, kw) in enumerate(linears) if kw == k]
        total, width = sum(linears[w][0] for w in mine), 2 * b_words(k)
        dom = G.domain(circuit, value(prefix, k), b_words(k), total)
        nonce, leaves = B_NONCE + i, []
        for w in mine:
            got = 0
            for row in rows(w):
                if len(row) != width:
                    raise ValueError(f"linear {w}'s row {got} is {len(row)} bytes, not K = {k}'s {width}")
                j = len(leaves)
                x = sha512_row_v2_digest(bytes(row), 8 * width)
                leaves.append(G.leaf(dom, j, _INST.commit_string(_INST.default_key, x, hm96_salt(key, nonce, j)), SCHEMA))
                got += 1
                if progress is not None and j % 65536 == 0:
                    progress(k, j)
            if got != linears[w][0]:
                raise ValueError(f"linear {w} gave {got} rows, not its n = {linears[w][0]}")
        out[k] = tree_levels(dom, leaves)[-1][0]
    return out


def entry(circuit: str, prefix: str, k: int, n: int, root: bytes) -> dict[str, Any]:
    """``<prefix>b/<K>``'s registration entry (`Registered.entry`) of Σn = `n` rows with root `root`."""
    v = value(prefix, k)
    return {"value": v, "root": root.hex(), "leaves": n, "words": b_words(k),
            "domain": G.domain(circuit, v, b_words(k), n).domain_id.hex()}


def weights_root(roots: Mapping[int, bytes]) -> bytes:
    """The weights' root: `tagged` of `WEIGHTS_TAG` and each K's ``…/b/<K>`` root, ``K, root`` by ascending K."""
    return tagged(WEIGHTS_TAG, *(p for k in sorted(roots) for p in (k, roots[k])))


__all__ = ["B_NONCE", "SCHEMA", "WEIGHTS_TAG", "b_words", "entry", "ks", "row_bytes", "row_words", "value", "weights_root",
           "weights_roots"]
