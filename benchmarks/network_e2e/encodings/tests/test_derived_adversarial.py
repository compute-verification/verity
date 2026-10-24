"""Red-team cases against ENC-DERIVED's canonical rules (#3543): the zero-bit source and hit derivation.

The companion `test_derived.py` already pins the canonical rules the lead named -- the no-eviction bound (a hit shorter
than the longest cached prefix is refused), the earliest-admitted source, arrival order (a step's new requests out of
order are refused) and the `rebuild(residual(a)) == a` round-trip. These cases add the one surface those don't reach: the
hash the derivation keys a token prefix by.

`Prefixes._key` chains Python's builtin `hash()` of an int-tuple (`hash((prev_key, token))`). That is the stand-in the
ENC-DERIVED report flags as "checked by the compile unit": it is neither vLLM's prefix-cache hash (sha256 over the block's
tokens) nor Verity's one hash (SHA-512, whole output). The derivation's 0-bit claim rests on this key being injective over
token prefixes, and on the prover and the verifier computing it identically. Builtin int-tuple hashing guarantees neither:
it collides, and its algorithm has changed across CPython releases (3.8 replaced the combiner). These cases show that when
that key's collision structure differs -- as a collision, or a verifier on another interpreter, would make it differ --
the honest advice that round-trips under the real key is refused. The failure here is conservative (a false reject, never
a false accept), but it means the "checked by the compile unit" claim is not yet real, and it violates the one-hash rule."""

from __future__ import annotations

import importlib.util
import inspect
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


def _collide(self, i, p):
    """A `Prefixes._key` whose collision structure differs from the real one: it chains over the token's low bit, so
    distinct prefixes a crypto hash would separate now key alike. Stands in for a builtin-hash collision, or for a
    verifier whose CPython combines int-tuples differently than the prover's."""
    ks, ts = self.keys[i], self.tokens[i]
    if p >= len(ts):
        raise ValueError(f"request {i} has {len(ts)} tokens, none at position {p}")
    while len(ks) <= p:
        ks.append(hash((ks[-1] if ks else 0, ts[len(ks)] & 1)))
    return ks[p]


def test_the_zero_bit_derivation_is_keyed_by_the_builtin_hash_not_one_sha512() -> None:
    """The source/hit derivation keys a token prefix by Python's builtin `hash()` of an int-tuple, not by a hash at all
    in Verity's sense. Two rules say it should: the derivation stands in for vLLM's prefix cache, which hashes a block's
    tokens with sha256; and Verity hashes with one function, SHA-512 over its whole output. `_key` is neither."""
    src = inspect.getsource(D.Prefixes._key)
    assert "hash((" in src, "the key is expected to chain the builtin hash of an int-tuple"
    assert "sha" not in src.lower() and "blake" not in src.lower(), "the key is not a named cryptographic hash"


def test_a_collision_in_that_key_false_rejects_honest_advice(case) -> None:
    """Under the real key the fixture round-trips. Give the key a different collision structure and the same honest
    advice is refused: the walk derives, from a spurious key match, a longer hit than the tokens support, and the source
    check catches the token mismatch. The prover cannot submit the truth. The refusal is conservative -- no forged advice
    is accepted -- but the 0-bit hit is only sound while this key never collides, which the builtin hash does not promise."""
    cfg, adv, order, held, res = case
    assert D.rebuild(res, cfg, held, ()) == adv
    real = D.Prefixes._key
    D.Prefixes._key = _collide
    try:
        with pytest.raises(ValueError):
            D.residual(adv, cfg, held, order=order)
    finally:
        D.Prefixes._key = real


def test_a_verifier_on_a_different_key_refuses_the_provers_residual(case) -> None:
    """The prover (`residual`) and the compile unit/verifier (`reconstruct`, through `rebuild`) both recompute the key.
    Nothing pins the interpreter they run on, and builtin int-tuple hashing has changed across CPython releases. A
    residual the prover accepted under its key is refused by a verifier whose key collides differently: the 0-bit
    derivation needs the two interpreters to agree bit for bit, which the design does not currently guarantee."""
    cfg, adv, order, held, res = case
    assert D.rebuild(res, cfg, held, ()) == adv
    real = D.Prefixes._key
    D.Prefixes._key = _collide
    try:
        with pytest.raises(ValueError):
            D.rebuild(res, cfg, held, ())
    finally:
        D.Prefixes._key = real
