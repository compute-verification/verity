"""A word selected by an unsigned index, as a GF(2) circuit over `forms`: the multiplexer tree.

``gather(C, idx, words, fill)`` is ``words[idx]`` for ``idx < len(words)``, else ``fill``. The tree selects by idx's bits, least
significant first: at each bit, words 2i and 2i+1 meet in one `forms.mux` (one AND per bit of the word), and a word without a
partner meets the constant ``fill``, which stands for every index past the end. So every AND reads a candidate's bits or a
partial result's, never the index alone: two gathers by one index over different words share no gate. ``len(words)`` words by
a k-bit index cost ``(len(words) - 1 + p) * width`` ANDs, p the number of the k levels at which an odd count meets ``fill``.
"""
from __future__ import annotations

from .forms import const_bits, mux

__all__ = ["gather"]


def gather(C, idx, words, fill: int):
    """``words[idx]`` for ``idx < len(words)`` (idx unsigned, LSB first), else the constant ``fill``; every word one width."""
    if not words:
        raise ValueError("a gather selects from at least one word")
    width = len(words[0])
    if any(len(w) != width for w in words):
        raise ValueError("a gather's words have one width")
    cur = [list(w) for w in words]
    for s in idx:
        if len(cur) % 2:
            cur.append(const_bits(fill, width))
        cur = [mux(C, s, cur[i + 1], cur[i]) for i in range(0, len(cur), 2)]
    if len(cur) > 1:
        raise ValueError(f"{len(words)} words need more than {len(idx)} index bits")
    return cur[0]
