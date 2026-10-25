"""Memory challenger logs for `obligation.verdict` and its Lean port (`PousCheck.Obligation`), on injected coins and
clock: an honest run and the three attacks, each a function of a seed.

* `honest`: every page written at version 0, some rewritten, a snapshot, reads of the snapshot: accepted.
* `stale`: a page rewritten after the prover kept its old version, which answers a read: `stale`.
* `keystream`: content equal to its keystream, which the prover can choose only after the coin: committed again after
  the coin (`recommit`), or kept under the first commitment (`keystream_kept`, `decode`).
* `early`: a page encoded before its coin: stored before the coin is drawn (`early`), or stored after it but under a
  coin the prover guessed (`early_guessed`, `decode`).

`CASES` names each with the verdict's reason (`None` for accept).
"""

from __future__ import annotations

import itertools
import random
from collections.abc import Callable
from typing import Any

from verity.core.protocols.pous import obligation as O

SALT = bytes(range(O.SALT_BYTES))
LIMIT_NS = 1_000
STEP_NS = 100
"""Every read takes this long on the injected clock, within `LIMIT_NS`."""


def challenger(rng: random.Random, pages: int, blocks: int) -> O.PageChallenger:
    return O.PageChallenger(SALT, pages, blocks, LIMIT_NS, coins=rng.randbytes, clock=itertools.count(0, STEP_NS).__next__)


def _plain(rng: random.Random, blocks: int) -> bytes:
    return rng.randbytes(O.params(blocks).segment_bytes)


def _filled(seed: int, pages: int, blocks: int, rewrites: int) -> tuple[random.Random, O.PageChallenger, O.PageStore]:
    """Every page written once, then `rewrites` pages rewritten, honestly."""
    rng = random.Random(seed)
    mc, store = challenger(rng, pages, blocks), O.PageStore(SALT, pages, blocks)
    for p in range(pages):
        store.write(mc, p, _plain(rng, blocks))
    for _ in range(rewrites):
        store.write(mc, rng.randrange(pages), _plain(rng, blocks))
    return rng, mc, store


def _window(rng: random.Random, mc: O.PageChallenger, store: O.PageStore, reads: int,
            answer: Callable[[int, int], tuple[bytes, tuple[bytes, ...]]] | None = None) -> None:
    store.snapshot(mc)
    for _ in range(reads):
        mc.read(rng.randrange(store.pages), rng.randrange(store.blocks), answer or store.answer)


def honest(seed: int = 0, pages: int = 3, blocks: int = 4, rewrites: int = 3, reads: int = 6) -> dict[str, Any]:
    rng, mc, store = _filled(seed, pages, blocks, rewrites)
    _window(rng, mc, store, reads)
    return mc.log()


def stale(seed: int = 0, pages: int = 3, blocks: int = 4) -> dict[str, Any]:
    """Page `p`'s version 0 kept and rewritten; the window's first read is of `p`, answered from version 0."""
    rng, mc, store = _filled(seed, pages, blocks, 0)
    p = rng.randrange(pages)
    _, old_blocks, old_tree = store.held[p]
    store.write(mc, p, _plain(rng, blocks))
    store.snapshot(mc)
    j = rng.randrange(blocks)
    mc.read(p, j, lambda page, block: (old_blocks[block], old_tree.path(block)))
    return mc.log()


def _keystream(seed: int, pages: int, blocks: int, recommit: bool) -> dict[str, Any]:
    rng, mc, store = _filled(seed, pages, blocks, 0)
    p, q = rng.randrange(pages), O.params(blocks)
    mc.commit(p, 1, O.commitment(p, 1, _plain(rng, blocks)))
    ts = O.page_salt(SALT, p, 1, mc.coin(p, 1))
    ks = O.keystream(q, ts)
    if recommit:
        mc.commit(p, 1, O.commitment(p, 1, ks))
    mc.store(p, 1, O.encode_page(q, ts, ks))
    _window(rng, mc, store, 2)
    return mc.log()


def keystream(seed: int = 0, pages: int = 3, blocks: int = 4) -> dict[str, Any]:
    return _keystream(seed, pages, blocks, True)


def keystream_kept(seed: int = 0, pages: int = 3, blocks: int = 4) -> dict[str, Any]:
    return _keystream(seed, pages, blocks, False)


def _early(seed: int, pages: int, blocks: int, before: bool) -> dict[str, Any]:
    rng, mc, store = _filled(seed, pages, blocks, 0)
    p, q, plain = rng.randrange(pages), O.params(blocks), _plain(rng, blocks)
    guess = rng.randbytes(O.COIN_BYTES)
    encoded = O.encode_page(q, O.page_salt(SALT, p, 1, guess), plain)
    mc.commit(p, 1, O.commitment(p, 1, plain))
    if before:
        mc.store(p, 1, encoded)
        mc.coin(p, 1)
    else:
        mc.coin(p, 1)
        mc.store(p, 1, encoded)
    _window(rng, mc, store, 2)
    return mc.log()


def early(seed: int = 0, pages: int = 3, blocks: int = 4) -> dict[str, Any]:
    return _early(seed, pages, blocks, True)


def early_guessed(seed: int = 0, pages: int = 3, blocks: int = 4) -> dict[str, Any]:
    return _early(seed, pages, blocks, False)


CASES: dict[str, tuple[Callable[..., dict[str, Any]], str | None]] = {
    "honest": (honest, None),
    "stale": (stale, "stale"),
    "keystream": (keystream, "recommit"),
    "keystream_kept": (keystream_kept, "decode"),
    "early": (early, "early"),
    "early_guessed": (early_guessed, "decode"),
}
