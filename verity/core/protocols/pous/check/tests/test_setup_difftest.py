"""`pous_check` against the Python reference, through the core difftest harness (`verity/core/difftest`).

The protocol: `pous-setup serve` (`PousCheck.SetupProtocol`, `verity/pous/p2/v4/setup`) against
`protocol.verifier_key(P2(params), commitment, salt, mode=PUBLIC, encoded=C*)` behind the interface's framing, on honest
setups at the spec's reduced widths and at w = 16,448 (the scheme's pinned instance), each mutation of them (a block flipped,
set to p or all ones, cut short or lengthened, dropped; blocks or segments swapped, a segment missing or extra, no segment;
another salt, of the right length and not; a commitment flipped or to another W), evidence malformed at the byte level,
parameters that aren't the manifest, and malformed cases. Verdict, digest, public outputs (the root among them) and profile
must agree.

The functions, `pous-setup-difftest` (`PousCheck.SetupDifftest.table`): the maps d, σ and D (`_Field`), the keys
(`expand_prefix`, `sha512_keys`, and which candidate t is), `decode_segment` with its refusals by kind, the commitment with
its manifest (the identity hash itself is `core_identity`'s, difftested there). The spec's §7 file is held to §7's SHA-512, and each of its vectors and
σ rejection cases to what Lean computes from the vector's inputs.

The maps and the setup keep to widths where Python's `_Field.square` reduces fully (`4·c0² < 2^w`, the spec's ladder and
w = 16,448); the keys also run at w = 24 with c0 = 65,533, where t rejects a candidate about once in 256.
"""

from __future__ import annotations

import functools
import hashlib
import json
import random
import re
import shutil
from pathlib import Path

import pytest
from hypothesis import strategies as st

import verity_difftest as D
from verity.core.protocols.pous import protocol, setup
from verity.core.protocols.pous.schemes import p2 as P2
from verity.ml.commitments.identity import canonical_json_bytes
from verity_difftest import protocol as P

LEAN = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[6]
SPEC_VECTORS = ROOT / "benchmarks" / "pous" / "p2_v1" / "vectors.json"
PINNED = ROOT / "verity" / "core" / "protocols" / "pous" / "tests" / "fixtures" / "schemes" / "p2-16448-v4.json"
pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")

ID = "verity/pous/p2/v4/setup"
LADDER = [(64, 189), (96, 17), (128, 173), (192, 237), (256, 189), (384, 317), (512, 569), (1024, 105)]
"""The spec's reduced widths (§1), each `2^w − c0` prime and ≡ 3 mod 4, so `encode_segment`'s roots exist."""
FULL = (16448, 21065)
REJECTING = (24, 65533)
"""t rejects a candidate with probability c0 / 2^w ≈ 2^-8 (keys only: `_Field.square`'s folds don't reduce here)."""
INVALID = [(20, 189), (16, 5), (64, 3), (64, 0), (64, 1 << 16 | 1), (0, 189)]
"""(w, c0) that `P2Params` refuses: not whole bytes, not above 16 bits, c0 ≢ 1 mod 4, c0 = 0, c0 ≥ 2^16, w = 0."""
VECTORS_SHA512 = ("959911fd371111fe20c4cee6fc532ff205bbf377962c73ac539a4d2de9b22121"
                  "287e6e996d9ae4bfbef9f26fcd67ac78620873702c8b2899bc41a21bdd76e853")
"""§7's SHA-512 of `vectors.json`."""


def _hex(x: int, n: int) -> str:
    return x.to_bytes(n, "big").hex()


def _str(j, k: str) -> str:
    v = j.get(k) if isinstance(j, dict) else None
    if not isinstance(v, str):
        raise D.Reject(f"{k} is missing or not a string")
    return v


def _bytes(j, k: str) -> bytes:
    return P.unhex(_str(j, k))


def _int(j, k: str) -> int:
    return int.from_bytes(_bytes(j, k), "big")


def _p2(w: int, c0: int, bps: int = 1, segments: int = 1) -> P2.P2Params:
    try:
        return P2.P2Params(w, c0, bps, segments)
    except ValueError as e:
        raise D.Reject(str(e)) from None


def _params(j) -> P2.P2Params:
    if not isinstance(j, dict):
        raise D.Reject("params is missing or not an object")
    return _p2(*(P.nat(j, k) for k in ("w", "c0", "blocks_per_segment", "segments")))


def _kind(e: ValueError) -> str:
    """The kind of a `ValueError` from `decode_segment` or `expand_prefix`, as `Setup.Refusal.kind` names it."""
    m = str(e)
    for needle, kind in [("is outside 0..", "segments"), ("a segment of C is", "shape"), ("the salt is", "salt"),
                         ("is not below p", "noncanonical"), ("decodes to more than", "payload")]:
        if needle in m:
            return kind
    if re.fullmatch(r"block \d+ is not \d+ bytes", m):
        return "width"
    raise ValueError(f"an unclassified refusal: {m}")


# The functions' references: Python's answer in the form of `PousCheck.SetupDifftest`'s output.

def maps(i) -> dict:
    q = _p2(P.nat(i, "w"), P.nat(i, "c0"))
    f, n, p = P2._field(q), q.block_bytes, q.p
    z, r, k, t = (_int(i, x) for x in "zrkt")
    if z >= p or r >= p or t >= p:
        raise D.Reject("maps: a value is not canonical")
    return {"d": _hex(f.sq(z), n), "sigma": _hex(f.sigma(r, k), n), "decode": _hex((f.sq(f.sigma(f.sq(z), k)) - t) % p, n)}


def _t_index(q: P2.P2Params, prefix: bytes, s: int, j: int, t: int) -> int:
    """Which candidate of `stream_t` t is (the spec's `t_rejection_index`), which `sha512_keys` doesn't return."""
    nb, base = q.block_bytes, prefix + s.to_bytes(8, "big") + j.to_bytes(8, "big")
    a = 0
    while (x := int.from_bytes(P2.stream(base, b"t", (a + 1) * nb)[a * nb:], "big")) >= q.p:
        a += 1
    if x != t:
        raise AssertionError("sha512_keys's t is not the first candidate below p")
    return a


def keys(i) -> dict:
    q = _p2(P.nat(i, "w"), P.nat(i, "c0"), 1 << 16, 1 << 20)
    salt = _bytes(i, "salt")
    try:
        prefix = P2.expand_prefix(q, salt)
    except ValueError:
        return {"error": "salt"}
    s, j = P.nat(i, "s"), P.nat(i, "j")
    k, t = P2.sha512_keys(q, prefix, s, j)
    return {"K": _hex(k, q.block_bytes), "t": _hex(t, q.block_bytes), "t_index": _t_index(q, prefix, s, j, t)}


def _blocks(i) -> list[bytes]:
    v = i.get("blocks") if isinstance(i, dict) else None
    if not isinstance(v, list) or not all(isinstance(b, str) for b in v):
        raise D.Reject("blocks is missing or not a list of strings")
    return [P.unhex(b) for b in v]


def decode(i) -> dict:
    q, salt, s, blocks = _params(i.get("params") if isinstance(i, dict) else None), _bytes(i, "salt"), P.nat(i, "s"), _blocks(i)
    try:
        return {"payload": P2.decode_segment(q, None, salt, s, blocks).hex()}
    except ValueError as e:
        return {"error": _kind(e)}


def commit(i) -> dict:
    q, w = _params(i.get("params") if isinstance(i, dict) else None), _bytes(i, "w")
    if len(w) != q.w_bytes:
        raise D.Reject(f"W has {len(w)} bytes")
    return {"commitment": setup.commit(q, w).hex(), "manifest": canonical_json_bytes(q.manifest()).decode()}


# The protocol's reference: `SetupProtocol`'s decoders and check over `verifier_key(…, mode=PUBLIC)`.

def _manifest(j) -> P2.P2Params:
    q = _params(j)
    try:
        same = D.canonical(j) == canonical_json_bytes(q.manifest()).decode()
    except TypeError:
        same = False
    if not same:
        raise D.Reject("not the manifest of p2-16448/v4 at these numbers")
    return q


def _commitment(b: bytes) -> tuple[bytes, bytes]:
    if len(b) < 64:
        raise D.Reject(f"the commitment to W is 64 bytes, then the salt; {len(b)} bytes is too short")
    return b[:64], b[64:]


def _evidence(b: bytes) -> list[list[bytes]]:
    """`u32be(segments) ‖ (u32be(blocks) ‖ (u32be(len) ‖ block)…)…`, each count at most a quarter of the bytes left, and
    nothing after the last segment."""
    pos = 0

    def take(n: int) -> bytes:
        nonlocal pos
        if pos + n > len(b):
            raise D.Reject(f"truncated at byte {pos}")
        pos += n
        return b[pos - n:pos]

    def count() -> int:
        n = int.from_bytes(take(4), "big")
        if 4 * n > len(b) - pos:
            raise D.Reject(f"a count {n} exceeds the bytes left")
        return n

    segments = [[take(int.from_bytes(take(4), "big")) for _ in range(count())] for _ in range(count())]
    if pos != len(b):
        raise D.Reject(f"{len(b) - pos} bytes follow the last segment")
    return segments


def _check(q: P2.P2Params, c: tuple[bytes, bytes], segments: list[list[bytes]], verify=protocol.verifier_key) -> tuple:
    w, salt = c
    try:
        vk = verify(P2.P2(q), w, salt, mode=protocol.PUBLIC, encoded=segments)
    except ValueError as e:
        return "reject", str(e)
    return "accept", {"blocks": q.blocks, "commitment": w.hex(), "root": vk.root.hex(), "salt": salt.hex()}, {"mode": "public"}


def setup_protocol(case, check=_check) -> dict:
    return P.answer(ID, case, params=_manifest, commitment=_commitment, evidence=_evidence, digest=lambda c: c[0] + c[1],
                    check=check)


# Setups: (params, W, salt, C*), honest and mutated, and their cases.

@functools.cache
def honest(w: int, c0: int, bps: int, segments: int, seed: int) -> tuple:
    q, rng = P2.P2Params(w, c0, bps, segments), random.Random(f"{w} {c0} {bps} {segments} {seed}")
    data, salt = rng.randbytes(q.w_bytes), rng.randbytes(24)
    enc = [P2.encode_segment(q, None, salt, s, data[s * q.segment_bytes:(s + 1) * q.segment_bytes]) for s in range(segments)]
    return q, data, salt, tuple(tuple(seg) for seg in enc)


@functools.cache
def pinned() -> tuple:
    """The scheme's pinned instance at w = 16,448, one segment of two blocks; its W is `pous_vectors._bytes`'s."""
    small = json.loads(PINNED.read_text())["small"]
    q = P2.P2Params(**{k: small["params"][k] for k in ("w", "c0", "blocks_per_segment", "segments")})
    data = hashlib.shake_256(b"verity-pous/vectors/scheme/p2-16448/v4/w").digest(q.w_bytes)
    return q, data, bytes.fromhex(small["salt"]), tuple(tuple(bytes.fromhex(b) for b in seg) for seg in small["blocks"])


@st.composite
def setups(draw) -> tuple:
    if draw(st.integers(0, 7)) == 0:
        return pinned()
    w, c0 = draw(st.sampled_from(LADDER))
    return honest(w, c0, draw(st.integers(1, 3)), draw(st.integers(1, 3)), draw(st.integers(0, 3)))


def _at(enc, s: int, j: int, block: bytes):
    return [[block if (a, b) == (s, j) else x for b, x in enumerate(seg)] for a, seg in enumerate(enc)]


def _flip(block: bytes, bit: int) -> bytes:
    return (int.from_bytes(block, "big") ^ (1 << bit % (8 * len(block)))).to_bytes(len(block), "big")


#: One change to an honest setup, `(q, W, salt, C*, x) ↦ (commitment, salt, C*)`, `x` choosing where.
MUTATIONS = {
    "flip": lambda q, w, salt, enc, x: (setup.commit(q, w), salt,
                                        _at(enc, x % q.segments, x % q.blocks_per_segment,
                                            _flip(enc[x % q.segments][x % q.blocks_per_segment], x))),
    "p": lambda q, w, salt, enc, x: (setup.commit(q, w), salt, _at(enc, x % q.segments, 0, q.p.to_bytes(q.block_bytes, "big"))),
    "ones": lambda q, w, salt, enc, x: (setup.commit(q, w), salt, _at(enc, 0, x % q.blocks_per_segment, b"\xff" * q.block_bytes)),
    "short": lambda q, w, salt, enc, x: (setup.commit(q, w), salt, _at(enc, x % q.segments, 0, enc[x % q.segments][0][1:])),
    "long": lambda q, w, salt, enc, x: (setup.commit(q, w), salt, _at(enc, 0, 0, enc[0][0] + b"\0")),
    "dropped": lambda q, w, salt, enc, x: (setup.commit(q, w), salt, [seg[:-1] if a == x % q.segments else seg
                                                                       for a, seg in enumerate(enc)]),
    "swap-blocks": lambda q, w, salt, enc, x: (setup.commit(q, w), salt, [seg[::-1] if a == x % q.segments else seg
                                                                           for a, seg in enumerate(enc)]),
    "swap-segments": lambda q, w, salt, enc, x: (setup.commit(q, w), salt, enc[::-1]),
    "missing": lambda q, w, salt, enc, x: (setup.commit(q, w), salt, enc[:-1]),
    "extra": lambda q, w, salt, enc, x: (setup.commit(q, w), salt, [*enc, enc[x % q.segments]]),
    "empty": lambda q, w, salt, enc, x: (setup.commit(q, w), salt, []),
    "salt": lambda q, w, salt, enc, x: (setup.commit(q, w), _flip(salt, x), enc),
    "salt-length": lambda q, w, salt, enc, x: (setup.commit(q, w), [salt + bytes(8), salt[:x % 24], b""][x % 3], enc),
    "commitment": lambda q, w, salt, enc, x: (_flip(setup.commit(q, w), x), salt, enc),
    "other-w": lambda q, w, salt, enc, x: (setup.commit(q, _flip(w, x)), salt, enc),
}


def _evidence_bytes(segments) -> bytes:
    out = len(segments).to_bytes(4, "big")
    for seg in segments:
        out += len(seg).to_bytes(4, "big") + b"".join(len(b).to_bytes(4, "big") + b for b in seg)
    return out


def _case(q, commitment: bytes, salt: bytes, enc, params: dict | None = None) -> dict:
    return {"commitment": (commitment + salt).hex(), "evidence": _evidence_bytes(enc).hex(),
            "params": json.loads(canonical_json_bytes(q.manifest())) if params is None else params}


def _honest_case(s) -> dict:
    q, w, salt, enc = s
    return _case(q, setup.commit(q, w), salt, enc)


def _mutated_case(s, label: str, x: int) -> dict:
    q, w, salt, enc = s
    return _case(q, *MUTATIONS[label](q, w, salt, enc, x))


def _manifest_of(case) -> dict:
    return case["params"]


#: One change to a valid case's bytes or parameters, `(case, x) ↦ case`.
CASE_MUTATIONS = [
    lambda c, x: c | {"evidence": c["evidence"] + "00"},
    lambda c, x: c | {"evidence": c["evidence"][: 2 * (x % (len(c["evidence"]) // 2))]},
    lambda c, x: c | {"evidence": (x % 3 + 1 << 30 * (x % 2)).to_bytes(4, "big").hex() + c["evidence"][8:]},
    lambda c, x: c | {"commitment": c["commitment"][: 2 * (x % 64)]},
    lambda c, x: c | {"commitment": c["commitment"].upper()},
    lambda c, x: c | {"evidence": c["evidence"] + "0"},
    lambda c, x: c | {"params": _manifest_of(c) | {"profile": "p2-1B"}},
    lambda c, x: c | {"params": _manifest_of(c) | {"expander": "chacha8-v2"}},
    lambda c, x: c | {"params": _manifest_of(c) | {"extra": x}},
    lambda c, x: c | {"params": {k: v for k, v in _manifest_of(c).items() if k != sorted(_manifest_of(c))[x % 7]}},
    lambda c, x: c | {"params": _manifest_of(c) | {"c0": _manifest_of(c)["c0"] + 2}},
    lambda c, x: c | {"params": _manifest_of(c) | {"segments": _manifest_of(c)["segments"] + 1}},
    lambda c, x: c | {"params": json.loads(canonical_json_bytes(P2.P2Params(*LADDER[x % 8], 1, 1).manifest()))},
    lambda c, x: {k: v for k, v in c.items() if k != ["commitment", "evidence", "params"][x % 3]},
    lambda c, x: c | {"params": [x]},
]

JSON = st.recursive(st.none() | st.booleans() | st.integers(-3, 3) | st.text(max_size=3),
                    lambda inner: st.lists(inner, max_size=3) | st.dictionaries(st.text(max_size=3), inner, max_size=3),
                    max_leaves=6)
VALID = setups().map(_honest_case)
MUTATED = st.builds(_mutated_case, setups(), st.sampled_from(sorted(MUTATIONS)), st.integers(0, 1 << 20))
MALFORMED = st.builds(lambda c, m, x: CASE_MUTATIONS[m](c, x), VALID, st.integers(0, len(CASE_MUTATIONS) - 1),
                      st.integers(0, 1 << 20))
CASES = st.one_of(VALID, VALID, MUTATED, MUTATED, MALFORMED, JSON)


def setup_port(reference=setup_protocol, inputs=CASES) -> D.ProtocolPort:
    return D.ProtocolPort("pous-setup", reference, inputs, package=LEAN)


protocol_server = D.server_fixture(setup_port())


def test_the_setup_protocol_agrees(protocol_server):
    stats = D.run(setup_port(), examples=600, server=protocol_server)
    assert stats.counts.get("accept", 0) > 100 and stats.counts.get("reject", 0) > 300, stats


def test_each_mutation_is_refused(protocol_server):
    """Every mutation of every width's honest setup and of the pinned one, three times each: Lean and Python agree, and
    only the honest setup is accepted (swapping the pinned setup's one segment leaves it honest)."""
    for s in [honest(w, c0, 2, 2, 0) for w, c0 in LADDER] + [pinned()]:
        case = _honest_case(s)
        assert D.replay(setup_port(), case, protocol_server)[0] == "accept"
        for label in MUTATIONS:
            for x in range(3):
                mutated = _mutated_case(s, label, x)
                want = "accept" if mutated == case else "reject"
                assert D.replay(setup_port(), mutated, protocol_server)[0] == want, label


def test_a_check_that_skips_the_commitment_is_caught(protocol_server):
    """A reference that accepts `C*` whatever `W` it decodes to differs on a mutated commitment."""
    def no_commitment(scheme, w, salt, mode, encoded):
        return protocol.verifier_key(scheme, setup.commit(scheme.params, b"".join(
            P2.decode_segment(scheme.params, None, salt, s, seg) for s, seg in enumerate(encoded))), salt, mode=mode,
            encoded=encoded)

    with pytest.raises(D.Mismatch) as e:
        D.run(setup_port(lambda c: setup_protocol(c, lambda q, c_, ev: _check(q, c_, ev, no_commitment))), examples=600,
              server=protocol_server)
    assert (e.value.lean[0], e.value.python[0]) == ("reject", "accept")


# The functions.

def _fn(name: str, reference, inputs) -> D.Port:
    return D.Port("pous-setup-difftest", name, reference, inputs, package=LEAN)


@st.composite
def map_inputs(draw) -> dict:
    w, c0 = draw(st.sampled_from(LADDER + [FULL] + INVALID[:1]))
    p, ones, rng = (1 << w) - c0, (1 << w) - 1, random.Random(draw(st.integers(0, 1 << 32)))
    z = draw(st.sampled_from([0, 1, p - 1, (p - 1) // 2, rng.randrange(p), rng.randrange(p), p]))
    r = draw(st.sampled_from([rng.randrange(p), p - 1 - rng.randrange(c0), 0]))
    k = draw(st.sampled_from([rng.getrandbits(w), ones, 0, ones ^ r ^ rng.randrange(c0)]))
    t = draw(st.sampled_from([rng.randrange(p), 0, p - 1]))
    n = -(-w // 8)
    return {"w": w, "c0": c0, "z": _hex(z, n), "r": _hex(r, n), "k": _hex(k, n), "t": _hex(t, n)}


KEY_INPUTS = st.builds(lambda wc, salt, s, j: {"w": wc[0], "c0": wc[1], "salt": salt.hex(), "s": s, "j": j},
                       st.sampled_from([REJECTING] * 4 + LADDER + [FULL] + INVALID),
                       st.one_of(st.binary(min_size=24, max_size=24), st.binary(max_size=40)),
                       st.integers(0, (1 << 20) - 1), st.integers(0, (1 << 16) - 1))


def _decode_input(s, label: str, x: int) -> dict:
    q, w, salt, enc = s
    _, salt, enc = MUTATIONS[label](q, w, salt, enc, x) if label != "honest" else (None, salt, enc)
    seg = x % max(len(enc), 1)
    blocks = enc[seg] if enc else []
    return {"params": {"w": q.w, "c0": q.c0, "blocks_per_segment": q.blocks_per_segment, "segments": q.segments},
            "salt": salt.hex(), "s": seg + (q.segments if label == "extra" else 0), "blocks": [b.hex() for b in blocks]}


DECODE_INPUTS = st.one_of(
    st.builds(_decode_input, setups(), st.sampled_from(["honest", "flip", "p", "ones", "short", "long", "dropped",
                                                        "swap-blocks", "salt", "salt-length", "extra"]),
              st.integers(0, 1 << 20)),
    st.builds(lambda i, k: {a: b for a, b in i.items() if a != k}, st.builds(_decode_input, setups(), st.just("honest"),
              st.integers(0, 7)), st.sampled_from(["params", "salt", "s", "blocks"])))


@st.composite
def commit_inputs(draw) -> dict:
    w, c0 = draw(st.sampled_from(LADDER + [FULL] + INVALID[:2]))
    bps, segments, rng = draw(st.integers(1, 5)), draw(st.integers(1, 3)), random.Random(draw(st.integers(0, 1 << 32)))
    size = bps * segments * ((w - 16) // 8 if w > 16 else 1) + draw(st.sampled_from([0, 0, 0, 1, -1]))
    return {"params": {"w": w, "c0": c0, "blocks_per_segment": bps, "segments": segments},
            "w": rng.randbytes(max(size, 0)).hex()}


FNS = {
    "maps": _fn("maps", maps, map_inputs()),
    "keys": _fn("keys", keys, KEY_INPUTS),
    "decode": _fn("decode", decode, DECODE_INPUTS),
    "commit": _fn("commit", commit, commit_inputs()),
}
fn_server = D.server_fixture(FNS["maps"])


@pytest.mark.parametrize("name", sorted(FNS))
def test_the_function_agrees(name, fn_server):
    stats = D.run(FNS[name], examples=400, server=fn_server)
    assert stats.outputs > 100, stats


def test_t_rejects_a_candidate(fn_server):
    """Keys whose t is not the first candidate (`t_index > 0`), found by search at w = 24, which a run draws rarely."""
    q, rng, found = P2.P2Params(*REJECTING, 1 << 16, 1 << 20), random.Random(20261007), 0
    while found < 4:
        i = {"w": q.w, "c0": q.c0, "salt": rng.randbytes(24).hex(), "s": rng.randrange(1 << 20), "j": rng.randrange(1 << 16)}
        if keys(i)["t_index"] > 0:
            assert D.replay(FNS["keys"], i, fn_server)[0] == "output"
            found += 1


def test_decode_refuses_by_kind(fn_server):
    """Each of `decode_segment`'s refusals, by kind, at every width: Lean names the same kind."""
    for s in [honest(w, c0, 2, 2, 1) for w, c0 in LADDER] + [pinned()]:
        kinds = set()
        for label in ["honest", "flip", "p", "short", "dropped", "salt-length", "extra"]:
            kind, line = D.replay(FNS["decode"], _decode_input(s, label, 0), fn_server)
            kinds.add(json.loads(line).get("error", "payload") if kind == "output" else kind)
        assert {"payload", "noncanonical", "width", "shape", "salt", "segments"} <= kinds, kinds


# The spec's §7 vectors.

def _vectors() -> dict:
    raw = SPEC_VECTORS.read_bytes()
    assert hashlib.sha512(raw).hexdigest() == VECTORS_SHA512, "vectors.json is not §7's"
    v = json.loads(raw)
    assert (v["schema"], v["protocol"], v["randomness"]) == ("p2-test-vectors/v4", "p2/v4", "verity/randomness/derive/v2")
    return v


def vector(v) -> dict:
    """What §7 records for the vector: its hash (pinned by §7), keys, rejection index, and `D(c) = m` below p and 2^b."""
    nb = v["w"] // 8
    assert v["decoded_m"] == v["m"], v["id"]
    return {"sha512": v["sha512"], "pinned": True, "K": _hex(int(v["K"], 16), nb), "t": _hex(int(v["t"], 16), nb),
            "t_rejection_index": v["t_rejection_index"], "canonical": True, "decoded_m": _hex(int(v["m"], 16), nb),
            "fits": True}


def sigma(v) -> dict:
    nb = v["w"] // 8
    assert v["branch"] == "reject-xor-and-return-r", v["id"]
    return {"p": _hex(int(v["p"], 16), nb), "r_xor_K": _hex(int(v["r_xor_K"], 16), nb), "rejects": True,
            "sigma": _hex(int(v["sigma"], 16), nb)}


def test_the_spec_vectors(fn_server):
    v = _vectors()
    assert len(v["vectors"]) == 4 and len(v["sigma_rejection_vectors"]) == 2
    for entry in v["vectors"]:
        assert D.replay(_fn("vector", vector, st.nothing()), entry, fn_server)[0] == "output", entry["id"]
    for entry in v["sigma_rejection_vectors"]:
        assert D.replay(_fn("sigma", sigma, st.nothing()), entry, fn_server)[0] == "output", entry["id"]
