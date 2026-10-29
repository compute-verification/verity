"""The sampled run's Hello on its record: one keyed coin tree, `coin-tree/hm96-sha512/v2`, over every coin of the run.

At the run's round 0 the challenger answers the prover's Hello with the tree's root and key, over the selection draw's
stream (or streams) first and then each session slot's streams, in session order. This is the Python of the Lean's
`SampledProofs.Hello`, which runs Flock's own record checks (`Flock.CoinTree`: R7 with the prover's nonce,
`checkRecord`, `checkFresh`) at run scope.
`verifier.hello` runs both. Codes, one at most, in the order checked: `hello`, `streams`, `coin_tree`, `fresh`."""
from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from .verdict import Verdict

SCHEME = "coin-tree/hm96-sha512/v2"
_HEX = frozenset("0123456789abcdef")


def slot_stream(i: int, s: str) -> str:
    """A session slot's stream `s` in the run's tree."""
    return f"slot/{i}/{s}"


def run_streams(selection: Sequence[str], slots: int, session: Sequence[str]) -> list[str]:
    """The run's streams in leaf order: the selection streams, then each session slot's, each the session's in order."""
    return [*selection, *(slot_stream(i, s) for i in range(slots) for s in session)]


def _hex(n: int, s: Any) -> bool:
    return isinstance(s, str) and len(s) == n and all(c in _HEX for c in s)


def _nat(x: Any) -> bool:
    return type(x) is int and x >= 0


def _integers(x: Any) -> bool:
    if isinstance(x, float):
        return False
    if isinstance(x, dict):
        return all(_integers(v) for v in x.values())
    return all(_integers(v) for v in x) if isinstance(x, list) else True


def _canon(x: Any) -> str:
    """`Flock.canon`: compact, keys sorted, non-ASCII as is, integers only."""
    if not _integers(x):
        raise ValueError("canonical JSON: a non-integer number")
    return json.dumps(x, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _no_constant(c: str) -> Any:
    raise ValueError(c)


def _spec(streams: Sequence[str], rounds_log: int, block: int) -> dict[str, Any]:
    return {"block": block, "rounds_log": rounds_log, "scheme": SCHEME, "streams": list(streams)}


def _of_hello(hello: str) -> dict[str, Any] | None:
    """`Flock.CoinTree.ofHello`: the tree's spec, when the Hello names this scheme in `coins`."""
    try:
        c = json.loads(hello).get("coins")
    except (ValueError, AttributeError):
        return None
    if not isinstance(c, dict) or c.get("scheme") != SCHEME:
        return None
    streams, rounds_log, block = c.get("streams"), c.get("rounds_log"), c.get("block")
    if not (isinstance(streams, list) and all(isinstance(s, str) for s in streams) and _nat(rounds_log) and _nat(block)):
        return None
    return _spec(streams, rounds_log, block)


def _check_hello(own: Any, received: str) -> str:
    """`Flock.CoinTree.checkHello`: R7 with the prover's nonce; the nonce."""
    try:
        j = json.loads(received, parse_constant=_no_constant)
    except ValueError:
        raise ValueError("S2/R7: the prover's Hello is not JSON") from None
    coins = j.get("coins") if isinstance(j, dict) else None
    nonce = coins.get("nonce") if isinstance(coins, dict) else None
    if not _hex(64, nonce):
        raise ValueError("S2/R7: Hello's coins.nonce is not 64 lowercase hex digits")
    if _canon({**own, "coins": {**own["coins"], "nonce": nonce}}) != received:
        raise ValueError("S2/R7: the session parameters are not the verifier's")
    return nonce


def _check_record(spec: Mapping[str, Any], nonce: str, ct: Any) -> str:
    """`Flock.CoinTree.checkRecord`: the record's `coin_tree`; the key."""
    if not (isinstance(ct, dict) and sorted(ct) == ["key", "nonce", "root", "spec"]):
        raise ValueError("S2: coin_tree is not {key, nonce, root, spec}")
    def text(k: str) -> str:
        return ct[k] if isinstance(ct[k], str) else ""
    if text("nonce") != nonce:
        raise ValueError("S2: coin_tree.nonce is not Hello's nonce")
    key = text("key")
    if not _hex(512, key):
        raise ValueError("S2: coin_tree.key is not 256 bytes of lowercase hex")
    if key[510] not in "01234567":
        raise ValueError("S2: coin_tree.key has bit 2047 set")
    if not _hex(128, text("root")):
        raise ValueError("S2: coin_tree.root is not 64 bytes of lowercase hex")
    if _canon(ct["spec"]) != _canon(dict(spec)):
        raise ValueError("S2: coin_tree.spec is not the verifier's")
    return key


def check_run(own: Any, hello: str, coin_tree: Any, seen: Sequence[str], selection: Sequence[str], slots: int,
              session: Sequence[str]) -> Verdict:
    """The run's Hello on its record. `own` is the verifier's own run Hello (an object whose `coins` is the tree's, the
    nonce empty), `hello` the prover's as received, `coin_tree` the record's and `seen` the keys of the runs verified
    before this one."""
    if not (isinstance(own, dict) and isinstance(own.get("coins"), dict)):
        raise ValueError("the verifier's own Hello is not an object with an object coins")
    def fail(code: str, why: str) -> Verdict:
        return Verdict(False, (code,), {code: why})
    try:
        nonce = _check_hello(own, hello)
    except ValueError as e:
        return fail("hello", str(e))
    spec = _of_hello(hello)
    if spec is None:
        return fail("hello", f"the run's Hello names no {SCHEME} coins")
    if not selection:
        return fail("streams", "the run has no selection stream")
    if spec["streams"] != run_streams(selection, slots, session):
        return fail("streams", "the Hello's streams are not the selection streams, then each session slot's in session order")
    try:
        key = _check_record(spec, nonce, coin_tree)
    except ValueError as e:
        return fail("coin_tree", str(e))
    if key in seen:
        return fail("fresh", "S2: coin_tree.key is an earlier session's; a session key is never reused")
    return Verdict(True)
