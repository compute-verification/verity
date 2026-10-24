"""Red-team cases against ENC-DERIVED's canonical rules (#3543): the zero-bit source and hit derivation, and the key it
looks token prefixes up by.

The companion `test_derived.py` already pins the canonical rules the lead named -- the no-eviction bound (a hit shorter
than the longest cached prefix is refused), the earliest-admitted source, arrival order (a step's new requests out of
order are refused) and the `rebuild(residual(a)) == a` round-trip. These cases reach the one surface those don't: the
key.

REDTEAM phase 4 found `Prefixes._key` chaining Python's builtin `hash()` of an int-tuple: neither Verity's one hash nor
stable across CPython releases, so a collision, or a verifier on another interpreter, false-rejected honest advice. The
key is now `unit_key`, SHA-512 over the parent key and a whole hit unit's canonical token bytes, and a key names a
request only where that request's tokens through the prefix match. These cases hold both halves: the key is SHA-512 with
a fixed answer on any interpreter, and a key with a different collision structure derives the same advice, so honest
advice can't be refused by one."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).parent


def _load():
    spec = importlib.util.spec_from_file_location("enc_derived", HERE.parent / "derived.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


D = _load()


@pytest.fixture(scope="module", params=("sync-crush", "async-crush"))
def case(request):
    trace = D.load_trace(HERE / "fixtures" / f"{request.param}.json.gz")
    cfg = D.config_of(trace)
    adv, order = D.advice_of(trace)
    held = D.holdings_of(trace)
    return cfg, adv, order, held, D.residual(adv, cfg, held, order=order)


def _collide(self, i, u):
    """A `Prefixes._key` whose collision structure differs from the real one: it chains over each token's low bit, so
    distinct prefixes SHA-512 separates key alike. Stands in for a collision, or for a verifier computing another key."""
    ks, ts, U = self.keys[i], self.tokens[i], self.unit
    if (u + 1) * U > len(ts):
        raise ValueError(f"request {i} has {len(ts)} tokens, not {u + 1} whole units of {U}")
    while len(ks) <= u:
        ks.append(D.unit_key(ks[-1] if ks else D.PREFIX_ROOT, [t & 1 for t in ts[len(ks) * U:(len(ks) + 1) * U]]))
    return ks[u]


def test_the_key_is_sha512_over_canonical_token_bytes() -> None:
    """The key of a prefix of whole hit units is SHA-512 (whole output) over its parent's key and the unit's token ids,
    4 bytes big-endian each, rooted at 64 zero bytes: the same bytes on any interpreter, pinned by a known answer."""
    unit = (128000, 9906, 1917, 13)
    key = D.unit_key(D.PREFIX_ROOT, unit)
    assert D.PREFIX_ROOT == bytes(64) and len(key) == 64
    assert key == hashlib.sha512(bytes(64) + b"".join(t.to_bytes(4, "big") for t in unit)).digest()
    assert key.hex()[:32] == "c330e618e315b4de544c1b5e9799ca4b"
    p = D.Prefixes([unit * 2], 4)
    assert p._key(0, 0) == key and p._key(0, 1) == D.unit_key(key, unit)
    for bad in ((-1,), (1 << 32,), (True,), (1.0,)):
        with pytest.raises(ValueError):
            D.unit_key(D.PREFIX_ROOT, bad)


def test_a_collision_in_the_key_derives_the_same_advice(case) -> None:
    """Under a key with a different collision structure the prover's residual is the same, and it rebuilds the honest
    advice: a spurious key match names a request only if its tokens match, so it can't lengthen a hit or move a source."""
    cfg, adv, order, held, res = case
    assert D.rebuild(res, cfg, held, ()) == adv
    real = D.Prefixes._key
    D.Prefixes._key = _collide
    try:
        assert D.residual(adv, cfg, held, order=order) == res
    finally:
        D.Prefixes._key = real


def test_a_verifier_on_a_different_key_accepts_the_provers_residual(case) -> None:
    """The prover (`residual`) and the compile unit (`reconstruct`, through `rebuild`) both look prefixes up by the key.
    A verifier whose key collides differently still rebuilds the prover's residual into the same advice: what is
    derived depends on the tokens, never on the key's collisions."""
    cfg, adv, order, held, res = case
    real = D.Prefixes._key
    D.Prefixes._key = _collide
    try:
        assert D.rebuild(res, cfg, held, ()) == adv
    finally:
        D.Prefixes._key = real
