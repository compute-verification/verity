"""`p2.py`, the remote audit over PoUS's P2 store, held to its Lean (`RemoteAudit.P2`, through `remote-audit-difftest`) on
drawn inputs that include what both sides must refuse: a registration's digest (`p2.Registration.digest`, with
`codec.P2Codec`'s refusals), a unit's sketch from its blocks' partials (`p2.Sketcher`: π a block, σ their sum, and y
from σ against y of the unit's bytes) and an opening's check (`p2.check`) on stores of drawn blocks under PoUS's leaf
and vk tree, honest or with one field changed; and the committed C* (`tests/fixtures/p2-c-star.bin`, PoUS's setup's
prover bytes): its registration's digest and an honest opening of each of its units."""

from __future__ import annotations

import functools
import hashlib
import json
import shutil
from pathlib import Path

import pytest
from hypothesis import strategies as st

import ra_cases as M
from ra_cases import Entropy
import verity_difftest as D
from verity.core.protocols.pous.verifier import MerkleTree, leaf_hash
from verity.core.protocols.remote_bandwidth import p2, receipts
from verity.core.protocols.remote_bandwidth.codec import P2_EXPANDER, P2_POINT, P2Codec
from verity.core.protocols.remote_bandwidth.receipts import Plan

pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")

FIXTURES = Path(__file__).resolve().parents[2] / "tests" / "fixtures"


def port(fn, reference, inputs, **kw) -> D.Port:
    return D.Port(M.EXE, fn, reference, inputs, package=M.PACKAGE, of_record=M.OF_RECORD, rejects=M.REFUSALS, **kw)


server = D.server_fixture(port("p2_sketch", lambda v: v, st.nothing()))


def expand(seed: bytes, n: int) -> bytes:
    return b"".join(hashlib.sha512(seed + c.to_bytes(4, "big")).digest() for c in range(-(-n // 64)))[:n]


def fault(draw, n: int) -> int:
    """Which of `n` faults an input carries, 1 … `n`, or 0 for none, about half the time."""
    f = draw(st.integers(0, 2 * n))
    return f if f <= n else 0


def p2_blocks() -> st.SearchStrategy:
    """A 2,056-byte block: drawn bytes, or one of every byte's extremes as int8."""
    return st.one_of(st.binary(max_size=8).map(lambda s: expand(s, p2.BLOCK_BYTES)),
                     st.sampled_from([b"\x80", b"\x7f", b"\xff", b"\0", b"\x80\x7f"]).map(
                         lambda b: (b * p2.BLOCK_BYTES)[:p2.BLOCK_BYTES]))


# --- decoding, as `RemoteAudit.Difftest` decodes ----------------------------------------------------------------------

def registration_of(v) -> p2.Registration:
    c = M._get(v, "codec")
    codec = P2Codec(M.nat(c, "blocks_per_segment"), M.nat(c, "segments"), M.nat(c, "w"), M.nat(c, "c0"),
                    M._str(c, "expander"))
    return p2.Registration(M.hex_(v, "node"), codec, M.hex_(v, "commitment"), M.hex_(v, "salt"), M.hex_(v, "root"),
                           M.nat(v, "rows"))


def registration_json(r: p2.Registration) -> dict:
    return {"node": r.node.hex(), "codec": r.codec.to_json(), "commitment": r.commitment.hex(), "salt": r.salt.hex(),
            "root": r.root.hex(), "rows": r.rows}


def seed_of(v) -> bytes:
    s = M.hex_(v, "seed")
    if len(s) != receipts.DIGEST_BYTES:
        raise D.Reject("a pass seed is 64 bytes")
    return s


def opening_of(v) -> p2.Opening:
    paths = []
    for x in M._list(v, "paths"):
        if not isinstance(x, list):
            raise D.Reject("paths: a path is a list")
        paths.append(tuple(M._unhex(d, "paths") for d in x))
    return p2.Opening(M.int_(v, "pass"), M.int_(v, "position"), tuple(M.hexes(v, "blocks")), tuple(paths),
                      M.hex_(v, "leaf"), M.hex_(v, "leaf_salt"), tuple(M.hexes(v, "receipt_path")))


def opening_json(o: p2.Opening) -> dict:
    return {"pass": o.pass_index, "position": o.position, "blocks": [b.hex() for b in o.blocks],
            "paths": [[d.hex() for d in x] for x in o.paths], "leaf": o.leaf.hex(), "leaf_salt": o.leaf_salt.hex(),
            "receipt_path": [d.hex() for d in o.receipt_path]}


# --- a registration's digest ------------------------------------------------------------------------------------------

def registration_digest(v) -> str:
    with M.decoding():
        return registration_of(v).digest.hex()


@st.composite
def registration_inputs(draw) -> dict:
    bps, segments = draw(st.sampled_from([(4, 1), (3, 3), (1, 4), (1 << 16, 1038), (1 << 16, (1 << 32) - 1), (2, 2)]))
    v = {"node": expand(draw(st.binary(max_size=8)), draw(st.integers(1, 64))).hex(),
         "codec": {"blocks_per_segment": bps, "c0": P2_POINT[1], "expander": P2_EXPANDER, "segments": segments,
                   "w": P2_POINT[0]},
         "commitment": expand(draw(st.binary(max_size=8)), 64).hex(), "salt": expand(draw(st.binary(max_size=8)), 24).hex(),
         "root": expand(draw(st.binary(max_size=8)), 64).hex(), "rows": draw(st.sampled_from([16, 64]))}
    c = v["codec"]
    edit = fault(draw, 10)
    if edit == 1:
        v["node"] = draw(st.sampled_from(["", "00" * 65]))
    elif edit == 2:
        k, x = draw(st.sampled_from([("w", P2_POINT[0] - 8), ("w", 1024), ("c0", P2_POINT[1] + 2),
                                     ("expander", "chacha8-v2")]))
        c[k] = x
    elif edit == 3:
        c["blocks_per_segment"] = draw(st.sampled_from([0, (1 << 16) + 1]))
    elif edit == 4:
        c["segments"] = draw(st.sampled_from([0, 1 << 32]))
    elif edit == 5:
        c["blocks_per_segment"], c["segments"] = draw(st.sampled_from([(1, 3), (3, 1), (1, 1)]))
    elif edit == 6:
        v["commitment"] = draw(st.sampled_from(["", "00" * 63]))
    elif edit == 7:
        v["salt"] = draw(st.sampled_from(["00" * 23, "00" * 32]))
    elif edit == 8:
        v["root"] = "00" * 63
    elif edit == 9:
        v["rows"] = draw(st.sampled_from([0, 32]))
    elif edit == 10:
        target = draw(st.sampled_from([v, c]))
        del target[draw(st.sampled_from(sorted(target)))]
    return v


def test_p2_registration_digest(server) -> None:
    stats = D.run(port("p2_registration_digest", registration_digest, registration_inputs()), examples=200,
                  server=server)
    assert stats.outputs > 80, stats


# --- a unit's sketch ---------------------------------------------------------------------------------------------------

def sketch(v) -> dict:
    """The decode's side, π of each block and σ their sum, and the verifier's, y of the unit's bytes: the Lean's y is
    from σ, so the two sides agree only if σ is the unit's sketch."""
    with M.decoding():
        seed, rows, g, blocks = seed_of(v), M.nat(v, "rows"), M.nat(v, "g"), M.hexes(v, "blocks")
        sk = p2.Sketcher(seed, rows)
        if len(blocks) != p2.UNIT_BLOCKS or any(len(b) != p2.BLOCK_BYTES for b in blocks):
            raise D.Reject("a unit is four 2,056-byte blocks")
        receipts._index(g)
    partials = [sk.block(p2.UNIT_BLOCKS * g + b, x) for b, x in enumerate(blocks)]
    return {"partials": [x.hex() for x in partials], "sigma": sk.total(partials).hex(),
            "receipt": sk(g, b"".join(blocks)).hex()}


@st.composite
def sketch_inputs(draw) -> dict:
    v = {"seed": expand(draw(st.binary(max_size=8)), 64).hex(), "rows": draw(st.sampled_from([16, 16, 64])),
         "g": draw(st.one_of(st.integers(0, 40), st.integers(0, 2 ** 64 - 1))),
         "blocks": [draw(p2_blocks()).hex() for _ in range(p2.UNIT_BLOCKS)]}
    edit = fault(draw, 5)
    if edit == 1:
        v["seed"] = draw(st.sampled_from(["", "00" * 63, "00" * 65]))
    elif edit == 2:
        v["rows"] = draw(st.sampled_from([0, 32]))
    elif edit == 3:
        v["g"] = 2 ** 64
    elif edit == 4:
        bs = v["blocks"]
        v["blocks"] = draw(st.sampled_from([bs[:3], bs + bs[:1], bs[:3] + [bs[3][:-2]], bs[:3] + [bs[3] + "00"]]))
    elif edit == 5:
        del v[draw(st.sampled_from(sorted(v)))]
    return v


def test_p2_sketch(server) -> None:
    stats = D.run(port("p2_sketch", sketch, sketch_inputs()), examples=120, server=server)
    assert stats.outputs > 50, stats


# --- an opening's check -----------------------------------------------------------------------------------------------

def opening(v) -> str | None:
    with M.decoding():
        reg = registration_of(M._get(v, "registration"))
        seed = seed_of(v)
        plan = Plan(M.runs_of(M._get(v, "runs"), "runs"), reg.blocks)
        p = M.nat(v, "p")
        if p >= plan.length:
            raise D.Reject("a position past the plan")
        o = opening_of(M._get(v, "opening"))
        root, i = M.hex_(v, "root"), M.nat(v, "i")
        if len(root) != receipts.DIGEST_BYTES:
            raise D.Reject("a pass's root is 64 bytes")
    return p2.check(reg, seed, plan, root, i, p, o)


#: Stores' geometry (blocks_per_segment, segments): a unit, units with a tail of one, two and three blocks, and
#: seventeen units, so that a plan has a second receipt leaf.
GEOMETRIES = ((4, 1), (2, 3), (3, 3), (5, 3), (17, 4))


@functools.cache
def drawn_store(k: int, rows: int) -> p2.Store:
    """A store of drawn blocks (P2's check never decodes, so they needn't be codewords) under geometry `k`, with
    every byte's extremes among them."""
    bps, segments = GEOMETRIES[k]
    blocks = [expand(b"block %d %d" % (k, j), p2.BLOCK_BYTES) for j in range(bps * segments)]
    blocks[1] = b"\x80" * p2.BLOCK_BYTES
    blocks[-1] = b"\x7f" * p2.BLOCK_BYTES
    codec, salt = P2Codec(bps, segments), expand(b"salt %d" % k, 24)
    root = MerkleTree([leaf_hash(p2.pous_params(codec), salt, j, b) for j, b in enumerate(blocks)]).root
    reg = p2.Registration(b"node %d" % k, codec, expand(b"commitment %d" % k, 64), salt, root, rows)
    return p2.Store(reg, blocks)


def canonical_runs(order: list[int]) -> list[list[int]]:
    """`order` as a plan's runs, consecutive units merged."""
    runs: list[list[int]] = []
    for g in order:
        if runs and runs[-1][0] + runs[-1][1] == g:
            runs[-1][1] += 1
        else:
            runs.append([g, 1])
    return runs


def flip(h: str, k: int = 0) -> str:
    b = bytearray(bytes.fromhex(h))
    b[k % len(b)] ^= 1
    return b.hex()


@st.composite
def opening_inputs(draw) -> dict:
    """An opening of a drawn pass over a drawn store, honest, from a prover that committed a wrong receipt, or with one
    field of the opening, the pass or the registration changed."""
    k, rows = draw(st.integers(0, len(GEOMETRIES) - 1)), draw(st.sampled_from([16, 16, 64]))
    store = drawn_store(k, rows)
    reg = store.registration
    order = draw(st.permutations(range(reg.blocks)))[:draw(st.integers(1, reg.blocks))]
    plan = Plan([tuple(r) for r in canonical_runs(order)], reg.blocks)
    seed, i, p = expand(draw(st.binary(max_size=8)), 64), draw(st.integers(0, 3)), draw(st.integers(0, plan.length - 1))
    prover = p2.Prover(store, entropy=Entropy(b"salts %d" % k))
    edit = fault(draw, 16)
    if edit == 1:
        sk, unit = p2.Sketcher(seed, rows), store.unit(plan.block(p))
        other = bytes([unit[0] ^ 1]) + unit[1:]
        ys = [sk(g, other if q == p else store.unit(g)) for q, g in enumerate(plan)]
        ps = prover.commit(0, i, seed, plan, ys)
    else:
        ps = prover.pass_under(0, i, seed, plan)
    v = {"registration": registration_json(reg), "seed": seed.hex(), "runs": canonical_runs(order),
         "root": ps.tree.commitment.root.hex(), "i": i, "p": p, "opening": opening_json(prover.open(ps, p))}
    o, r = v["opening"], v["registration"]
    b = draw(st.integers(0, p2.UNIT_BLOCKS - 1))
    if edit == 2:
        o[draw(st.sampled_from(["pass", "position"]))] += draw(st.sampled_from([1, -1]))
    elif edit == 3:
        c = draw(st.integers(0, p2.UNIT_BLOCKS - 1))
        o["blocks"][b], o["blocks"][c] = o["blocks"][c], o["blocks"][b]
    elif edit == 4:
        o["blocks"][b] = flip(o["blocks"][b], draw(st.integers(0, p2.BLOCK_BYTES - 1)))
    elif edit == 5:
        x = o["paths"][b]
        o["paths"][b] = draw(st.sampled_from(
            [x[:-1], x + ["00" * 64], [flip(x[0])] + x[1:] if x else ["00" * 64], ["00" * 63] + x[1:]]))
    elif edit == 6:
        o[draw(st.sampled_from(["blocks", "paths"]))].pop()
    elif edit == 7:
        o["blocks"][b] = o["blocks"][b][:draw(st.sampled_from([0, 2 * p2.BLOCK_BYTES - 2]))]
    elif edit == 8:
        o["leaf"] = draw(st.sampled_from([flip(o["leaf"], 64 * (p % receipts.PER_LEAF)), o["leaf"][:-128],
                                          o["leaf"] + "00" * 64]))
    elif edit == 9:
        o["leaf_salt"] = draw(st.sampled_from([flip(o["leaf_salt"]), o["leaf_salt"][:-2]]))
    elif edit == 10:
        o["receipt_path"] = draw(st.sampled_from(
            [[flip(o["receipt_path"][0])] + o["receipt_path"][1:] if o["receipt_path"] else ["00" * 64],
             o["receipt_path"] + ["00" * 64]]))
    elif edit == 11:
        v[draw(st.sampled_from(["root", "seed"]))] = draw(st.sampled_from([flip(v["root"]), "00" * 63]))
    elif edit == 12:
        key = draw(st.sampled_from(["salt", "root"]))
        r[key] = flip(r[key])
    elif edit == 13:
        r["rows"] = 80 - rows
    elif edit == 14:
        v["runs"] = draw(st.sampled_from([[[0, reg.blocks]], [[reg.blocks, 1]], []]))
    elif edit == 15:
        r["node"] = flip(r["node"])
    elif edit == 16:
        target = draw(st.sampled_from([v, o, r]))
        del target[draw(st.sampled_from(sorted(target)))]
    return v


def test_p2_opening(server) -> None:
    stats = D.run(port("p2_opening", opening, opening_inputs()), examples=400, server=server)
    assert stats.outputs > 300, stats


# --- the committed C* ---------------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def committed() -> p2.Store:
    """The committed C* (PoUS's setup's prover bytes, `tests/test_p2.py`'s), registered under its vk root."""
    meta = json.loads((FIXTURES / "p2-store.json").read_text())
    data = (FIXTURES / "p2-c-star.bin").read_bytes()
    assert hashlib.sha512(data).hexdigest() == meta["c_star_sha512"]
    m = meta["params"]
    reg = p2.Registration(b"p2-node-under-test", P2Codec(m["blocks_per_segment"], m["segments"]),
                          bytes.fromhex(meta["commitment"]), bytes.fromhex(meta["salt"]), bytes.fromhex(meta["root"]))
    return p2.Store(reg, [data[p2.BLOCK_BYTES * j:p2.BLOCK_BYTES * (j + 1)] for j in range(len(data) // p2.BLOCK_BYTES)])


def test_the_committed_store_s_registration_digest_is_the_leans(server, committed) -> None:
    reg = committed.registration
    assert D.replay(port("p2_registration_digest", registration_digest, st.nothing()), registration_json(reg),
                    server) == ("output", f'"{reg.digest.hex()}"')


def test_an_honest_opening_of_each_committed_unit_checks_in_lean(server, committed) -> None:
    reg = committed.registration
    plan, seed = Plan.whole(reg.blocks), hashlib.sha512(b"a pass seed").digest()
    prover = p2.Prover(committed, entropy=Entropy(b"committed"))
    ps = prover.pass_under(0, 0, seed, plan)
    for p in range(plan.length):
        v = {"registration": registration_json(reg), "seed": seed.hex(), "runs": [list(r) for r in plan.runs],
             "root": ps.tree.commitment.root.hex(), "i": 0, "p": p, "opening": opening_json(prover.open(ps, p))}
        assert opening(v) is None
        assert D.replay(port("p2_opening", opening, st.nothing()), v, server) == ("output", "null")
