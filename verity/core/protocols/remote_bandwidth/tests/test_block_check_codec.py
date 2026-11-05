"""PoUB's opening unit (`verity.ml.commitments.gates.block_check`) against the audit's own code: its reference is the audit's
checks (`codec`, `receipts`, `audit`'s trees and `merkle.verify_opening`) on random openings at p512/k2 and p1024/k3/masked,
appended and registered, and names the check each wrong opening fails (the cases are the unit's own suite's,
`verity/ml/commitments/tests/block_check_cases.py`); its key, its rounds and its binding at the deployed fields are the
codec's; and the toy blocks the unit's suite reads (`block_check_toy.json`) are the codec's encodings.

    uv run python verity/core/protocols/remote_bandwidth/tests/test_block_check_codec.py   # rewrite the toy blocks
"""
from __future__ import annotations

import importlib.util
import json
import random
from pathlib import Path

import numpy as np
import pytest

from verity.core.protocols.remote_bandwidth import audit as AU
from verity.core.protocols.remote_bandwidth import codec as CO
from verity.core.protocols.remote_bandwidth import receipts as RC
from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation import evaluate_batch
from verity.ml.commitments import Opening, verify_opening
from verity.ml.commitments.gates import block_check as BC
from verity.ml.commitments.merkle import merkle_depth

HERE = Path(__file__).resolve().parent
UNIT_TESTS = HERE.parents[3] / "ml" / "commitments" / "tests"
CASES = UNIT_TESTS / "block_check_cases.py"
TOY_BLOCKS = UNIT_TESTS / "fixtures" / "block_check_toy.json"


def _cases_module():
    """The unit's suite's openings, loaded from their file."""
    spec = importlib.util.spec_from_file_location("block_check_cases", CASES)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


K = _cases_module()


def _codec(S) -> CO.Codec:
    return CO.Codec(CO.FIELDS[BC.FIELDS[S.N, S.C]], S.K, bool(S.MASKED))


def _toy_encode(S, seed: bytes, j: int, payload: bytes, coin: bytes | None) -> bytes:
    """`codec.encode` over the toy field: core's ρ on each round, the reference's key and binding."""
    f = CO.Field(BC.FIELDS[S.N, S.C], S.N, S.C)
    c, tau, mask = BC.key(S, seed)
    w, W, p = S.N // 8, S.BYTES // (S.N // 8), f.p
    b = BC._binding(S, seed, j, BC.sha512_row_v2_digest(payload, 8 * S.BYTES), coin) if coin is not None else 0
    out = b""
    for i in range(W):
        x, t = int.from_bytes(payload[w * i:w * (i + 1)], "little"), (tau * (W * j + i) + b) % p
        for r, cr in enumerate(c):
            if r and mask is not None:
                x = x ^ mask if x ^ mask < p else x
            x = CO.rho((x + cr + t) % p, f)
        out += x.to_bytes(w, "little")
    return out


def _core_encode(S, seed: bytes, j: int, payload: bytes, coin: bytes | None) -> bytes:
    return CO.encode(payload, j, CO.key(seed, _codec(S)), bound=coin is not None, coin=coin)


def _sketch(pass_seed: bytes, rows: int) -> tuple[bytes, list[bytes]]:
    return RC.sketch_prefix(pass_seed), RC.sign_bits(pass_seed, rows)


def toy_blocks() -> list[dict]:
    """Every block the toy unit's cases (`block_check_cases.toy_cases`) encode, with its opening's inputs."""
    out = []

    def encode(S, seed: bytes, j: int, payload: bytes, coin: bytes | None) -> bytes:
        block = _toy_encode(S, seed, j, payload, coin)
        out.append({"masked": S.MASKED, "seed": seed.hex(), "j": j, "payload": payload.hex(),
                    "coin": None if coin is None else coin.hex(), "block": block.hex()})
        return block

    for S in (K.TOY, K.TOY_MASKED):
        K.toy_cases(S, encode)
    return out


def test_the_toy_blocks_are_the_codecs() -> None:
    assert json.loads(TOY_BLOCKS.read_text()) == toy_blocks(), (
        f"{TOY_BLOCKS.name} isn't the codec's encodings of the toy unit's cases: rewrite it with "
        "`uv run python verity/core/protocols/remote_bandwidth/tests/test_block_check_codec.py`")


def test_the_key_is_the_codecs() -> None:
    rng = random.Random(1)
    for S in (K.P512, K.P1024):
        seed = rng.randbytes(32)
        k = CO.key(seed, _codec(S))
        assert BC.key(S, seed) == (k.c, k.tau, k.mask)


@pytest.mark.parametrize("S", [K.P512, K.P1024], ids=["p512-k2", "p1024-k3-masked"])
def test_the_reference_is_the_audits_opening(S) -> None:
    """The honest openings pass the audit's own checks (`codec.decode`, `SketchReceipter`, `receipts.hidden` and
    `verify_opening` on each tree) and the reference; each wrong one is refused, by the check it names."""
    rng = random.Random(S.N)
    for name, o, a, want in K.cases(rng, S, _core_encode, sketch=_sketch):
        got = BC.opening_check(o.S, a)
        assert BC.verdict(got) == want, (name, got)
        if name.endswith(": honest"):
            k = CO.key(o.seed, _codec(S))
            d = CO.digest(o.pay)
            assert (CO.decode(o.block, o.j, k, d, o.coin) if o.coin else CO.decode(o.block, o.j, k)) == o.pay
            assert RC.SketchReceipter(o.pass_seed, S.ROWS)(o.j, o.block) == o.ys[o.pos]
            assert RC.hidden(o.pay, o.salt) == K.hidden(o.pay, o.salt) == RC.hidden_digest(d, o.salt)
            for tree, at, value in ((o.rtree, o.c, RC.hidden(o.rows[o.c], o.rsalts[o.c])), (o.ptree, o.j, RC.hidden(o.pay, o.salt))):
                assert verify_opening(tree.domain, tree.commitment, Opening(at, value, tree.open(at).path), K.V2, RC.LIMITS)


def _runs(blocks: list[int]) -> tuple[tuple[int, int], ...]:
    out: list[tuple[int, int]] = []
    for b in blocks:
        if out and sum(out[-1]) == b:
            out[-1] = (out[-1][0], out[-1][1] + 1)
        else:
            out.append((b, 1))
    return tuple(out)


def test_the_core_trees_open_in_the_unit() -> None:
    """The audit's own trees (a pass's receipt tree under its seed and plan, an append's tree, the registration's payload
    tree) at their domains, ranks and depths, as `opening_public` takes them, are accepted."""
    rng = random.Random(7)
    S = K.P512
    reg = rng.randbytes(64)
    o = K.opening(rng, S, True, _core_encode, sketch=_sketch)
    first = o.ptree.domain.positions.start
    count = o.ptree.domain.count
    digests = {i: CO.digest(K.payload(rng, S)) for i in range(first, first + count)}
    digests[o.j] = CO.digest(o.pay)
    salts = {i: rng.randbytes(BC.SALT_BYTES) for i in digests}
    salts[o.j] = o.salt
    app = AU.append_tree(reg, first, [digests[i] for i in sorted(digests)], [salts[i] for i in sorted(salts)])
    plan = RC.Plan(_runs(o.plan), 1 << 12)
    assert [plan.block(p) for p in range(plan.length)] == o.plan
    rtree = RC.tree(o.pass_seed, plan, o.ys, o.rsalts)
    rdom = RC.domain(o.pass_seed, plan)
    pub = BC.opening_public(o.S, o.seed, o.j, o.coin, o.prefix, o.signs, o.pos,
                            (rtree.commitment.root, rdom.domain_id, merkle_depth(rdom.count)),
                            (app.commitment.root, app.domain.domain_id, o.j - first, merkle_depth(count)),
                            (BC._bytes_of(o.args["vroot"]), BC._bytes_of(o.args["vdom"]), BC._int(o.args["vrank"]),
                             o.args["vdepth"][0]))
    a = dict(o.args, **{k: pub[k] for k in ("rroot", "rdom", "rdepth", "proot", "pdom", "prank", "pdepth")})
    a["rpath"] = BC._u16s(b"".join(rtree.open(o.c).path) + bytes(64 * (S.DR - len(rtree.open(o.c).path))))
    a["ppath"] = BC._u16s(b"".join(app.open(o.j).path) + bytes(64 * (S.DP - len(app.open(o.j).path))))
    assert BC.opening_check(o.S, a) == 1
    assert BC.verdict(BC.opening_check(o.S, dict(a, prank=o.args["prank"]))) is None
    assert BC.verdict(BC.opening_check(o.S, dict(a, pdom=o.args["pdom"]))) == "payload-path"


def test_the_public_rows_are_the_verifiers_own() -> None:
    """`opening_public` is exactly the public rows' params of an instance's arguments, so a verifier holding only the public
    inputs derives the words its public rows open to."""
    rng = random.Random(4)
    for S, bound in ((K.P512, True), (K.P1024, False)):
        o = K.opening(rng, S, bound, _core_encode, sketch=_sketch)
        rows = BC.opening_rows(o.S)
        pub = {k: o.args[k] for _r, _b, ps in rows[:2] for k in ps}
        assert BC.row_words(rows[:2], pub) == BC.row_words(rows[:2], o.args)


def _bits(x: int, n: int) -> np.ndarray:
    return np.array([(x >> i) & 1 for i in range(n)], dtype=np.uint8)


def _value(bits) -> int:
    return sum(int(b) << i for i, b in enumerate(bits))


@pytest.mark.slow
@pytest.mark.parametrize("S", [K.P512, K.P1024], ids=["p512-k2", "p1024-k3-masked"])
def test_the_rounds_and_the_binding_at_the_deployed_fields_are_the_codecs(S) -> None:
    """About 40 s at p1024 (its round's trace): the only evaluation of the Karatsuba squares and the folds by C at the
    deployed widths, and of the binding's reduction mod p, which a toy word (one schoolbook square) never reaches."""
    rng = random.Random(S.N + 1)
    f = CO.FIELDS[BC.FIELDS[S.N, S.C]]
    p, n = f.p, S.N
    mask = rng.getrandbits(n) if S.MASKED else None
    xs = [0, 1, 2, p - 1, p - 2, (p - 1) // 2] + [rng.randrange(p) for _ in range(10)]
    cs = [p - 1, 0] + [rng.randrange(p) for _ in range(len(xs) - 2)]
    ts = [p - 1] + [rng.randrange(p) for _ in range(len(xs) - 1)]
    rnd = bind(BC.CodecRound, N=S.N, C=S.C, MASK=S.MASKED)
    cols = [np.stack([_bits(v, n) for v in vs]) for vs in (xs, cs, ts)]
    if S.MASKED:
        cols.append(np.stack([_bits(mask, n)] * len(xs)))
    got = [_value(r) for r in np.asarray(evaluate_batch(rnd, *cols)).reshape(len(xs), n)]
    want = []
    for x, c, t in zip(xs, cs, ts):
        y = (CO.rho_inv(x, f) - c - t) % p
        want.append((y ^ mask if y ^ mask < p else y) if S.MASKED else y)
    assert got == want

    k = CO.key(rng.randbytes(32), _codec(S))
    d, coin, j = rng.randbytes(64), rng.randbytes(32), rng.randrange(1 << 40)
    b = bind(BC.CodecBinding, N=S.N, C=S.C)
    dw = [int.from_bytes(d[8 * i:8 * i + 8], "big") for i in range(8)]
    args = [np.array([[1], [0]], dtype=np.uint8), np.stack([np.unpackbits(np.frombuffer(k.seed, np.uint8), bitorder="little")] * 2),
            np.stack([_bits(j, 64)] * 2), np.stack([np.concatenate([_bits(w, 64) for w in dw])] * 2),
            np.stack([np.unpackbits(np.frombuffer(coin, np.uint8), bitorder="little")] * 2)]
    out = np.asarray(evaluate_batch(b, *args)).reshape(2, n)
    assert [_value(r) for r in out] == [CO.binding(d, j, k, coin), 0]


if __name__ == "__main__":
    TOY_BLOCKS.parent.mkdir(exist_ok=True)
    TOY_BLOCKS.write_text(json.dumps(toy_blocks(), indent=1) + "\n")
