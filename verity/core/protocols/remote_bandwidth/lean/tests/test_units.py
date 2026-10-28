"""The Python of the remote audit's verifier held to the Lean's pieces (`remote-audit-difftest`, the definition), each on
drawn inputs that include what both sides must refuse: the codec's decode (`codec.decode`, under the seed's key, a bound key
and an append's coin), a plan (`receipts.Plan`: its length, digest and chunks) and its change (`Plan.changed`), a receipt
(`receipts.receipter`, both kinds and both row counts), a frame-v3-sha512 opening (`verity.ml.commitments.verify_opening`
over a range domain), a leaf's receipt (`receipts.leaf_receipt`), a window's draws (`AuditRun.close`'s), a
registration's digest (`audit.Registration.digest`), and the registration `audit.register` makes, its codec seed drawn
from the verifier's secret and a nonce. `test_session.py` holds the whole verifier."""

from __future__ import annotations

import bisect
import functools
import hashlib
import itertools
import shutil

import pytest
from hypothesis import strategies as st

import ra_cases as M
import verity_difftest as D
from verity.core.protocols.remote_bandwidth import audit, codec, receipts
from verity.core.protocols.remote_bandwidth.receipts import Plan
from verity.ml.commitments import Commitment, CommitmentDomain, MerkleTree, Opening, RangeIndexedDomain, verify_opening
from verity.ml.randomness import derive

pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")


def port(fn, reference, inputs, **kw) -> D.Port:
    return D.Port(M.EXE, fn, reference, inputs, package=M.PACKAGE, of_record=M.OF_RECORD, rejects=M.REFUSALS, **kw)


server = D.server_fixture(port("plan", lambda v: v, st.nothing()))


def expand(seed: bytes, n: int) -> bytes:
    return b"".join(hashlib.sha512(seed + c.to_bytes(4, "big")).digest() for c in range(-(-n // 64)))[:n]


def mostly(good: st.SearchStrategy, bad: st.SearchStrategy, n: int = 4) -> st.SearchStrategy:
    """`good` `n` times in `n + 1`, else `bad`."""
    return st.integers(0, n).flatmap(lambda i: good if i else bad)


def sized(n: int, *, near: bool = True) -> st.SearchStrategy:
    """`n` bytes as hex, or now and then a byte more or less."""
    return mostly(st.binary(max_size=8).map(lambda s: expand(s, n)),
                  st.sampled_from([n - 1, n + 1, 0] if near else [0]).flatmap(
                      lambda m: st.binary(max_size=8).map(lambda s: expand(s, max(m, 0)))), 6).map(bytes.hex)


def blocks() -> st.SearchStrategy:
    """A 4096-byte chunk: drawn bytes, or one of every byte's extremes as int8."""
    return st.one_of(st.binary(max_size=8).map(lambda s: expand(s, codec.BLOCK_BYTES)),
                     st.sampled_from([b"\x80", b"\x7f", b"\xff", b"\0", b"\x80\x7f"]).map(
                         lambda b: (b * codec.BLOCK_BYTES)[:codec.BLOCK_BYTES]))


CODECS = (M.P512, M.P512_K3, codec.FALLBACK, codec.DEFAULT)


@functools.cache
def encoded() -> tuple[dict, ...]:
    """Chunks that decode: one payload a codec, under the seed's key, its bound key, and its bound key with a coin."""
    out = []
    for n, c in enumerate(CODECS):
        seed, j, payload = expand(b"seed %d" % n, 32), 5 + 1000 * n, M.payload(b"unit %d" % n)
        k = codec.key(seed, c)
        for bound, coin in ((False, None), (True, None), (True, expand(b"coin %d" % n, 32)))[:3 if n < 2 else 2]:
            out.append({"codec": M.codec_json(c), "seed": seed.hex(), "j": j, "coin": None if coin is None else coin.hex(),
                        "digest": codec.digest(payload).hex() if bound else None,
                        "block": codec.encode(payload, j, k, bound=bound, coin=coin).hex()})
    return tuple(out)


def codec_decode(v) -> str | None:
    c = M.codec_of(M._get(v, "codec"))
    seed, block, j = M.hex_(v, "seed"), M.hex_(v, "block"), M.nat(v, "j")
    d, coin = M.hex_or_none(v, "digest"), M.hex_or_none(v, "coin")
    if len(seed) != codec.SEED_BYTES:
        raise D.Reject("a codec seed is 32 bytes")
    out = codec.decode(block, j, codec.key(seed, c), d, coin)
    return None if out is None else out.hex()


@st.composite
def decode_inputs(draw) -> dict:
    v = dict(draw(st.sampled_from(encoded())))
    edit = draw(st.integers(0, 12))
    if edit == 1:
        v["j"] += draw(st.sampled_from([1, -1]))
    elif edit == 2:
        b = bytearray.fromhex(v["block"])
        b[draw(st.integers(0, 4095))] ^= 1 << draw(st.integers(0, 6))
        v["block"] = b.hex()
    elif edit == 3:
        v["block"] = "ff" * 128 + v["block"][256:]
    elif edit == 4:
        v["digest"] = draw(sized(64)) if v["digest"] else hashlib.sha512(b"x").hexdigest()
    elif edit == 5:
        v["coin"] = None if v["coin"] else draw(sized(32))
    elif edit == 6:
        v["digest"] = None
    elif edit == 7:
        v["seed"] = draw(sized(32))
    elif edit == 8:
        v["block"] = draw(sized(codec.BLOCK_BYTES))
    elif edit == 9:
        v["j"] = draw(st.sampled_from([codec.WORD_INDICES // 64 - 1, codec.WORD_INDICES // 32 - 1, 2 ** 64, -1]))
    elif edit == 10:
        v["codec"] = draw(st.sampled_from([{"field": "p256", "rounds": 2, "masked": False},
                                           {"field": "p512", "rounds": 1, "masked": False},
                                           {"field": "p512", "rounds": 2, "masked": 1}, M.codec_json(codec.DEFAULT)]))
    elif edit == 11:
        v["block"] = draw(blocks()).hex()
    elif edit == 12:
        del v[draw(st.sampled_from(sorted(v)))]
    return v


def test_codec_decode(server) -> None:
    stats = D.run(port("codec_decode", codec_decode, decode_inputs()), examples=150, server=server)
    assert stats.outputs > 60, stats


def _positions(v) -> list[int]:
    out = M._list(v, "positions")
    if any(not isinstance(p, int) or isinstance(p, bool) or p < 0 for p in out):
        raise D.Reject("positions are naturals")
    return out


def plan(v) -> dict:
    pl = Plan(M.runs_of(M._get(v, "runs"), "runs"), M.nat(v, "blocks"))
    ps = _positions(v)
    if any(p >= pl.length for p in ps):
        raise D.Reject("a position past the plan")
    return {"blocks": [pl.block(p) for p in ps], "digest": pl.digest.hex(), "length": pl.length}


RUN = st.lists(st.integers(-1, 12), min_size=2, max_size=2)
RUNS = st.lists(RUN, max_size=5)


def fault(draw, n: int) -> int:
    """Which of `n` faults an input carries, 1 … `n`, or 0 for none, about half the time."""
    f = draw(st.integers(0, 2 * n))
    return f if f <= n else 0


@st.composite
def plan_runs(draw, extent: int) -> list:
    """A plan of distinct chunks of a store of `extent`, in a drawn order."""
    order = draw(st.permutations(sorted(draw(st.sets(st.integers(0, min(extent, 2 ** 64 - 1) - 1), min_size=1,
                                                     max_size=10)))))
    return [list(r) for r in Plan.of(order, min(extent, 2 ** 64 - 1)).runs]


@st.composite
def plan_inputs(draw) -> dict:
    f = fault(draw, 3)
    blocks = draw(st.integers(1, 14) | st.just(2 ** 64 - 1))
    runs = draw(RUNS) if f == 1 else draw(plan_runs(blocks))
    if f == 2:
        blocks = draw(st.sampled_from([0, 2 ** 64]))
    length = sum(c for _, c in runs if isinstance(c, int))
    inside = st.lists(st.integers(0, length - 1), max_size=6) if length > 0 else st.just([])
    return {"runs": runs, "blocks": blocks,
            "positions": draw(st.lists(st.integers(-1, max(length, 0) + 1), min_size=1, max_size=3) if f == 3
                              else inside)}


def test_plan(server) -> None:
    stats = D.run(port("plan", plan, plan_inputs()), examples=300, server=server)
    assert stats.outputs > 120, stats


def plan_changed(v) -> list:
    pl = Plan(M.runs_of(M._get(v, "runs"), "runs"), M.nat(v, "blocks"))
    out = pl.changed(M.runs_of(M._get(v, "added"), "added"), M.runs_of(M._get(v, "removed"), "removed"),
                     M.nat(v, "extent"))
    return [list(r) for r in out.runs]


@st.composite
def change_inputs(draw) -> dict:
    """A plan's change: chunks it holds removed along its runs and chunks it then doesn't hold added, or one of a
    shrinking store, drawn runs for the change, a change that removes every chunk, or drawn runs for the plan."""
    f = fault(draw, 4)
    blocks = draw(st.integers(1, 10))
    extent = blocks - 1 if f == 1 else blocks + draw(st.integers(0, 4))
    runs = draw(plan_runs(blocks))
    held = {a + u for a, c in runs for u in range(c)}
    gone = held if f == 3 else draw(st.sets(st.sampled_from(sorted(held)), max_size=min(4, len(held) - 1)))
    more = draw(st.sets(st.integers(0, max(extent, 1) - 1), max_size=4)) - (held - gone)
    removed = [list(r) for a, c in sorted(tuple(r) for r in runs) if gone & set(range(a, a + c))
               for r in Plan.of(sorted(gone & set(range(a, a + c))), a + c).runs]
    added = [list(r) for r in (Plan.of(sorted(more), max(more) + 1).runs if more else ())]
    if f == 2:
        removed, added = draw(RUNS), draw(RUNS)
    if f == 4:
        runs = draw(RUNS)
    return {"runs": runs, "blocks": blocks, "added": added, "removed": removed, "extent": extent}


def test_plan_changed(server) -> None:
    stats = D.run(port("plan_changed", plan_changed, change_inputs()), examples=300, server=server)
    assert stats.outputs > 100, stats


def receipt(v) -> str:
    kind, rows, seed = M._str(v, "kind"), M.nat(v, "rows"), M.hex_(v, "seed")
    j, block = M.nat(v, "j"), M.hex_(v, "block")
    return receipts.receipter(kind, seed, rows)(j, block).hex()


@st.composite
def receipt_inputs(draw) -> dict:
    """A receipt of either kind at its rows, or one of an unknown kind, rows the kind doesn't take, a seed or a chunk
    of the wrong size, or a chunk index past u64."""
    kind = draw(st.sampled_from([receipts.SKETCH_SHA512, receipts.BLOCK_SHA512]))
    v = {"kind": kind, "rows": draw(st.sampled_from([16, 64] if kind == receipts.SKETCH_SHA512 else [0])),
         "seed": expand(draw(st.binary(max_size=8)), 64).hex(), "j": draw(st.integers(0, 2 ** 64 - 1)),
         "block": draw(blocks()).hex()}
    f = fault(draw, 5)
    if f == 1:
        v["kind"] = "x"
    elif f == 2:
        v["rows"] = draw(st.sampled_from([0, 32] if kind == receipts.SKETCH_SHA512 else [16, 64]))
    elif f == 3:
        v["seed"] = draw(st.sampled_from(["", "00" * 63, "00" * 65, "0"]))
    elif f == 4:
        v["j"] = 2 ** 64
    elif f == 5:
        v["block"] = v["block"][:draw(st.sampled_from([0, 2, 8190]))]
    return v


def test_receipt(server) -> None:
    stats = D.run(port("receipt", receipt, receipt_inputs()), examples=200, server=server)
    assert stats.outputs > 120, stats


def opening(v) -> bool:
    binding, owner, start, count = M.hex_(v, "binding"), M.int_(v, "owner"), M.nat(v, "start"), M.nat(v, "count")
    root, position, value = M.hex_(v, "root"), M.nat(v, "position"), M.hex_(v, "value")
    path, schema = tuple(M.hexes(v, "path")), M._str(v, "schema")
    d = CommitmentDomain(binding, owner, RangeIndexedDomain(start, start + count), hash="sha512")
    return verify_opening(d, Commitment(root, count), Opening(position, value, path), schema, receipts.LIMITS)


@st.composite
def opening_inputs(draw) -> dict:
    """An opening of a drawn tree over a range domain, as is or with one field changed."""
    binding = expand(draw(st.binary(max_size=8)), draw(st.sampled_from([64, 64, 32])))
    owner, start, count = draw(st.sampled_from([0, -1, -2, 3])), draw(st.integers(0, 40)), draw(st.integers(1, 20))
    schema = draw(st.sampled_from([receipts.SCHEMA, "verity/remote-bandwidth/coins/v0"]))
    d = CommitmentDomain(binding, owner, RangeIndexedDomain(start, start + count), hash="sha512")
    values = {start + u: expand(b"%d" % u, draw(st.integers(0, 3)) * 64) for u in range(count)}
    tree = MerkleTree(d, values, lambda _p: schema)
    position = start + draw(st.integers(0, count - 1))
    v = {"binding": binding.hex(), "owner": owner, "start": start, "count": count, "root": tree.commitment.root.hex(),
         "position": position, "value": values[position].hex(), "path": [x.hex() for x in tree.open(position).path],
         "schema": schema}
    edit = fault(draw, 10)
    if edit == 1:
        v["position"] = draw(st.sampled_from([position + 1, position - 1, start + count, start - 1]))
    elif edit == 2:
        v["value"] = (bytes.fromhex(v["value"]) + b"\0").hex()
    elif edit == 3 and v["path"]:
        k = draw(st.integers(0, len(v["path"]) - 1))
        v["path"][k] = draw(st.sampled_from([hashlib.sha512(b"x").hexdigest(), "00" * 32, "00" * 63]))
    elif edit == 4:
        v["path"] = v["path"][:-1] if v["path"] and draw(st.booleans()) else v["path"] + ["00" * 64]
    elif edit == 5:
        v["root"] = draw(st.sampled_from(["00" * 64, "00" * 32, "00" * 31, v["root"][:-2] + "00"]))
    elif edit == 6:
        v["start"] = draw(st.sampled_from([start + 1, max(start - 1, 0)]))
    elif edit == 7:
        v["count"] = draw(st.sampled_from([count + 1, count - 1, 0]))
    elif edit == 8:
        v["owner"] = draw(st.sampled_from([owner + 1, -3, 7]))
    elif edit == 9:
        v["schema"] = "hm96-sha512/row/v1"
    elif edit == 10:
        v["binding"] = draw(st.sampled_from([binding[:-1].hex(), (binding[:32] if len(binding) == 64 else binding * 2).hex()]))
    return v


def test_verify_opening(server) -> None:
    stats = D.run(port("verify_opening", opening, opening_inputs()), examples=300, server=server)
    assert stats.outputs > 200, stats


def leaf_receipt(v) -> str | None:
    out = receipts.leaf_receipt(M.hex_(v, "leaf"), M.nat(v, "p"), M.nat(v, "positions"))
    return None if out is None else out.hex()


@st.composite
def leaf_inputs(draw) -> dict:
    positions = draw(st.integers(0, 40))
    p = draw(mostly(st.integers(0, positions + 2), st.just(-1), 9))
    c = max(p, 0) // receipts.PER_LEAF
    size = 64 * min(receipts.PER_LEAF, positions - receipts.PER_LEAF * c)
    n = draw(mostly(st.just(size), st.sampled_from([0, 64, max(size, 0) + 64, max(size - 1, 0), 1024])))
    return {"leaf": expand(draw(st.binary(max_size=4)), max(n, 0)).hex(), "p": p, "positions": positions}


def test_leaf_receipt(server) -> None:
    stats = D.run(port("leaf_receipt", leaf_receipt, leaf_inputs()), examples=300, server=server)
    assert stats.outputs > 150, stats


def draws(v) -> list:
    """`AuditRun.close`'s draw from window t's openings coin, on the passes' lengths."""
    coin, registration, t, samples = M.hex_(v, "coin"), M.hex_(v, "registration"), M.nat(v, "window"), M.nat(v, "samples")
    lengths = M._list(v, "lengths")
    if not lengths or any(not isinstance(n, int) or isinstance(n, bool) or n < 1 for n in lengths):
        raise D.Reject("a window's passes are each one or more receipts")
    key = derive(coin, audit.OPENINGS_DOMAIN, {"registration": registration, "window": t})
    ends = tuple(itertools.accumulate(lengths))
    out = []
    for u in key.subset(min(samples, ends[-1]), ends[-1], "openings"):
        i = bisect.bisect_right(ends, u)
        out.append([i, u - (ends[i - 1] if i else 0)])
    return out


@st.composite
def draws_inputs(draw) -> dict:
    """A coin of one or more bytes (`derive` refuses an empty one, which no session has), and lengths now and then
    empty or with a pass of none."""
    lengths = draw(mostly(st.lists(st.integers(1, 40), min_size=1, max_size=4),
                          st.sampled_from([[], [0], [3, 0], [-1]])))
    return {"coin": expand(draw(st.binary(max_size=8)), draw(st.sampled_from([32, 32, 1, 64]))).hex(),
            "registration": draw(sized(64, near=False)), "window": draw(st.integers(0, 2 ** 33)),
            "samples": draw(mostly(st.integers(1, 60), st.just(0), 9)), "lengths": lengths}


def test_draws(server) -> None:
    stats = D.run(port("draws", draws, draws_inputs()), examples=200, server=server)
    assert stats.outputs > 120, stats


def registration_digest(v) -> str:
    with M.decoding():
        return M.registration_of(v).digest.hex()


@st.composite
def registration_inputs(draw) -> dict:
    kind = draw(st.sampled_from(receipts.KINDS))
    c = draw(st.sampled_from(CODECS))
    v = {"node": expand(draw(st.binary(max_size=8)), draw(st.integers(1, 64))).hex(),
         "blocks": draw(st.sampled_from([1, 6, 2 ** 20, c.max_blocks])),
         "payload_root": expand(draw(st.binary(max_size=8)), 64).hex(),
         "codec_seed": expand(draw(st.binary(max_size=8)), 32).hex(), "codec": M.codec_json(c), "receipt": kind,
         "bind": False, "layout": receipts.LAYOUT,
         "rows": draw(st.sampled_from([16, 64])) if kind == receipts.SKETCH_SHA512 else 0}
    edit = fault(draw, 9)
    if edit == 1:
        v["node"] = draw(st.sampled_from(["", "00" * 65]))
    elif edit == 2:
        v["blocks"] = draw(st.sampled_from([0, c.max_blocks + 1, 2 ** 64]))
    elif edit == 3:
        v["payload_root"] = "00" * 63
    elif edit == 4:
        v["codec_seed"] = "00" * 31
    elif edit == 5:
        v["bind"] = True
    elif edit == 6:
        v["layout"] = "fused/v0"
    elif edit == 7:
        v["rows"] = draw(st.sampled_from([0, 16, 32, 64]))
    elif edit == 8:
        v["receipt"] = draw(st.sampled_from(["x", *receipts.KINDS]))
    elif edit == 9:
        del v[draw(st.sampled_from(sorted(v)))]
    return v


def test_registration_digest(server) -> None:
    stats = D.run(port("registration_digest", registration_digest, registration_inputs()), examples=200, server=server)
    assert stats.outputs > 80, stats


def register(v) -> dict:
    """`registerFn`: `audit.register`'s draw (`_draw`, of the node, N and payload root the log gives it) on the input's
    source and nonce, `bind` refused as the Lean refuses it."""
    with M.decoding():
        source, node, root, nonce = (M.hex_(v, k) for k in ("source", "node", "payload_root", "nonce"))
        c, kind, layout = M.codec_of(M._get(v, "codec")), M._str(v, "receipt"), M._str(v, "layout")
        blocks, rows = M.nat(v, "blocks"), M.nat(v, "rows")
        if M._bool(v, "bind"):
            raise D.Reject("a registration that binds PoUW's tile outputs is not covered: the Lean refuses it")
        r = audit._draw(source, node, blocks, root, c, kind, layout=layout, rows=rows, nonce=nonce)
    return {"codec_seed": r.codec_seed.hex(), "digest": r.digest.hex()}


@st.composite
def register_inputs(draw) -> dict:
    v = draw(registration_inputs())
    v.pop("codec_seed", None)
    v["source"] = expand(draw(st.binary(max_size=8)), draw(st.sampled_from([1, 21, 64]))).hex()
    v["nonce"] = expand(draw(st.binary(max_size=8)), audit.NONCE_BYTES).hex()
    edit = fault(draw, 3)
    if edit == 1:
        v["source"] = ""
    elif edit == 2:
        v["nonce"] = draw(st.sampled_from(["", "00" * 31, "00" * 33]))
    elif edit == 3:
        del v[draw(st.sampled_from(["source", "nonce"]))]
    return v


def test_register(server) -> None:
    stats = D.run(port("register", register, register_inputs()), examples=200, server=server)
    assert stats.outputs > 40, stats


def test_a_recorded_store_s_registration_is_the_leans(server) -> None:
    """The scenarios' store is registered by `audit.register` under `ra_cases.SOURCE` and a zero nonce."""
    reg = M.Run().store.registration
    v = M.registration_json(reg) | {"source": M.SOURCE.hex(), "nonce": "00" * audit.NONCE_BYTES}
    del v["codec_seed"]
    assert D.replay(port("register", register, st.nothing()), v, server) == (
        "output", D.canonical({"codec_seed": reg.codec_seed.hex(), "digest": reg.digest.hex()}))


def test_a_recorded_registration_s_digest_is_its_session_s(server) -> None:
    case = M.recorded("honest")[0]
    reg = M.registration_of(case["registration"])
    assert D.replay(port("registration_digest", registration_digest, st.nothing()), case["registration"], server) == (
        "output", f'"{reg.digest.hex()}"')
