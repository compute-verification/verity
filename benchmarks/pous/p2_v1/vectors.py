"""P2 v4's conformance vectors, all of them, from one command:

    cd benchmarks/pous && uv run python -m p2_v1.vectors

writes `vectors.json` (the maps and keys, spec §7), `audit-vectors.json` (a complete two-round w = 64 setup and audit, and
one full-width response), `audit-negative-vectors.json` (31 failing or boundary transcripts, each self-checked against
`wire.verify_audit`) and the package's copy of the keys (`verity/.../pous/tests/fixtures/p2-v4-keys.json`), then replaces
every pinned hash that changed, in `reference.py`, the spec (`p2/PROTOCOL.md`) and the package's keys test, with its new
value.

Every setup salt, audit id, challenge index and nonce is `verity.ml.randomness`'s, so every file but the maps'
`m`, `K`, `t` and `c` at a given salt moves when that derivation does; each file records the derivation it was made under
(`randomness`), and rerunning this is the whole migration.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from verity.ml.commitments.identity import canonical_json_bytes
from verity.ml.randomness import derive, frame

from . import reference as R
from . import wire as Wi

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
AUDIT_VECTORS, NEGATIVE_VECTORS = R.AUDIT_VECTORS, R.NEGATIVE_VECTORS
KEYS_FIXTURE = ROOT / "verity/core/protocols/pous/tests/fixtures/p2-v4-keys.json"
SPEC = ROOT / "verity/core/protocols/pous/p2/PROTOCOL.md"
PINNED_TEXT = (HERE / "reference.py", SPEC, KEYS_FIXTURE.parent.parent / "test_pous_protocol.py")

POINTS = (R.Params(64, 189, 7), R.Params(96, 17, 11), R.Params(128, 173, 15), R.DEPLOYMENT)
"""The least c0 ≡ 1 mod 4 with 2^w − c0 prime at each reduced width (spec §1), and the deployment's."""
PRIMALITY_METHOD = "GMP BPSW plus 64 Miller-Rabin repetitions"
"""The kit's screen of each point's p (`p2-test-vectors/v1`'s `primality`, whose results this carries: they are facts
about p, unchanged by v4, and gmpy2 is not a dependency here)."""

SMALL_SETUP_SOURCE = b"p2-audit-vector/setup-source"
SMALL_AUDIT_SOURCE = b"p2-audit-vector/audit-secret"
FULL_SETUP_SOURCE = b"p2-audit-vector/full-setup-source"
FULL_AUDIT_SOURCE = b"p2-audit-vector/full-audit-secret"
NEGATIVE_FULL_SETUP_SOURCE = b"p2-audit-negative/full-setup"
NEGATIVE_FULL_AUDIT_SOURCE = b"p2-audit-negative/full-audit-secret"


def randomness_version() -> str:
    """Which `verity.ml.randomness` derivation this process has: v1 (SHA-256) or v2 (SHA-512, truncated to the
    32-byte key)."""
    source, domain = b"p2", "verity/pous/p2-probe/v4"
    framed = frame(source) + frame(domain) + frame(b"")
    got = derive(source, domain, {}).value
    if got == hashlib.sha256(b"verity/randomness/derive/v1\0" + framed).digest():
        return "verity/randomness/derive/v1"
    if got == hashlib.sha512(b"verity/randomness/derive/v2\0" + framed).digest()[:32]:
        return "verity/randomness/derive/v2"
    raise AssertionError("verity.ml.randomness is neither v1 nor v2")


def _hex(params: R.Params, value: int) -> str:
    return value.to_bytes(params.width_bytes, "big").hex()


def _sha512(data: bytes) -> str:
    return hashlib.sha512(data).hexdigest()


def _counter(label: str, n: int) -> bytes:
    """The first n bytes of SHA512(label ‖ u32be 0) ‖ SHA512(label ‖ u32be 1) ‖ …: the vectors' own test inputs."""
    return b"".join(hashlib.sha512(label.encode() + i.to_bytes(4, "big")).digest() for i in range(-(-n // 64)))[:n]


def _width_params(w: int, c0: int) -> R.Params:
    return R.DEPLOYMENT if (w, c0) == (R.DEPLOYMENT.w, R.DEPLOYMENT.c0) else R.Params(w, c0, w // 8 - 1)


# --- vectors.json ----------------------------------------------------------------------------------------------------------

def map_vector(params: R.Params, index: int = 0) -> dict:
    source = hashlib.sha512(f"p2-vector-source/{params.w}/{index}".encode()).digest()
    model = hashlib.sha512(f"p2-vector-model/{params.w}".encode()).digest()
    segment, block = index + 2, index + 7
    salt = R.setup_salt(source, params, model)
    k, t, attempt = R.keys(salt, params, model, segment, block)
    bits = 8 * params.payload_bytes
    m = int.from_bytes(_counter(f"p2-vector-m/{params.w}/{index}", params.width_bytes), "big") % (1 << bits)
    c = R.encode(params, m, k, t)
    if R.decode(params, c, k, t) != m:
        raise AssertionError("vector does not round trip")
    row = {"id": f"p2-v4/w{params.w}/{index}", "w": params.w, "c0": params.c0, "source": source.hex(),
           "model_commitment": model.hex(), "setup_salt": salt.hex(), "segment": segment, "block": block,
           "m": _hex(params, m), "K": _hex(params, k), "t": _hex(params, t), "t_rejection_index": attempt,
           "c": _hex(params, c), "decoded_m": _hex(params, m)}
    return {**row, "sha512": _sha512(canonical_json_bytes(row))}


def map_vectors() -> dict:
    rejection = []
    for point in (POINTS[0], R.DEPLOYMENT):
        r, k = 0, (1 << point.w) - 1
        rejection.append({"id": f"sigma-rejection/w{point.w}", "w": point.w, "c0": point.c0, "r": _hex(point, r),
                          "K": _hex(point, k), "r_xor_K": _hex(point, r ^ k), "p": _hex(point, point.p),
                          "sigma": _hex(point, R.sigma(point, r, k)), "branch": "reject-xor-and-return-r"})
    primality = [{"w": p.w, "c0": p.c0, "method": PRIMALITY_METHOD, "result": "probable-prime", "gmp_result": 1,
                  "p_mod_4": p.p % 4, "p_bits": p.p.bit_length()} for p in POINTS]
    return {"schema": "p2-test-vectors/v4", "protocol": R.PROTOCOL, "randomness": randomness_version(),
            "integer_encoding": "fixed w-bit unsigned big-endian", "primality": primality,
            "vectors": [map_vector(p) for p in POINTS], "sigma_rejection_vectors": rejection}


# --- audit-vectors.json ----------------------------------------------------------------------------------------------------

def _setup(params: R.Params, w_bytes: bytes, payloads: list[int], source: bytes) -> dict:
    manifest = canonical_json_bytes(params.manifest())
    commitment = R.w_commitment(params, w_bytes)
    salt = R.setup_salt(source, params, commitment)
    bps, n = params.blocks_per_segment, params.width_bytes
    blocks = []
    for i, value in enumerate(payloads):
        k, t, _ = R.keys(salt, params, commitment, i // bps, i % bps)
        blocks.append(R.encode(params, value, k, t).to_bytes(n, "big"))
    leaves = [R.leaf(salt, i // bps, i % bps, b) for i, b in enumerate(blocks)]
    root = R.root(leaves)
    return {"manifest": manifest, "commitment": commitment, "salt": salt, "blocks": blocks, "leaves": leaves, "root": root,
            "epoch": R.epoch_id(params, commitment, salt, root),
            "messages": Wi.setup_messages(manifest, commitment, salt, blocks, root)}


SMALL = R.Params(64, 189, 7, blocks_per_segment=8, segments=1, profile="p2-63b")
FULL = R.Params(R.DEPLOYMENT.w, R.DEPLOYMENT.c0, 2054, blocks_per_segment=1, segments=1)


def small_vector() -> dict:
    payloads = list(range(1, SMALL.blocks + 1))
    w_bytes = R.pack_payloads(payloads, 63)
    s = _setup(SMALL, w_bytes, payloads, SMALL_SETUP_SOURCE)
    counter, k, delta, rtt, elapsed = 7, 2, R.DELTA_NS, 0, [101_000, 102_000]
    audit_id, key = R.audit_key(SMALL_AUDIT_SOURCE, s["epoch"], s["root"], s["salt"], counter)
    opened = Wi.open_message(s["epoch"], audit_id, 0, k, delta, rtt, 0x01, counter)
    transcript, records, exchanges = [opened], [], []
    for j in range(k):
        index, nonce = key.uniform(SMALL.blocks, j), key.stream(("nonce", j), 32)
        challenge = Wi.challenge_message(s["epoch"], audit_id, j, index, nonce)
        response = Wi.response_message(s["epoch"], audit_id, j, index, s["blocks"][index], R.path(s["leaves"], index))
        transcript += [challenge, response]
        records.append(Wi.record(j, index, elapsed[j], 0x0F, hashlib.sha512(challenge).digest(),
                                 hashlib.sha512(response).digest()))
        exchanges.append({"round": j, "index": index, "nonce": nonce.hex(), "challenge": challenge.hex(),
                          "challenge_sha512": _sha512(challenge), "response": response.hex(),
                          "response_sha512": _sha512(response)})
    final = Wi.final_message(s["epoch"], audit_id, Wi.FAIL_SUCCESS, delta, rtt, transcript, records)
    frames = s["messages"] + transcript
    if Wi.verify_audit(frames, elapsed, verifier_source=SMALL_AUDIT_SOURCE, seen_audits=set()) != final:
        raise AssertionError("the reference verifier does not accept the small transcript")
    return {"w": SMALL.w, "c0": SMALL.c0, "manifest": s["manifest"].decode(), "W": w_bytes.hex(), "W_bit_length": 63 * SMALL.blocks,
            "w_commitment": s["commitment"].hex(), "setup_source": SMALL_SETUP_SOURCE.hex(), "setup_salt": s["salt"].hex(),
            "audit_source": SMALL_AUDIT_SOURCE.hex(), "audit_counter": counter, "elapsed_ns": elapsed,
            "blocks": [b.hex() for b in s["blocks"]], "leaf_hashes": [x.hex() for x in s["leaves"]],
            "verification_root": s["root"].hex(), "epoch_id": s["epoch"].hex(), "audit_id": audit_id.hex(),
            "setup_messages": [{"bytes": m.hex(), "sha512": _sha512(m)} for m in s["messages"]],
            "audit_open": opened.hex(), "audit_open_sha512": _sha512(opened), "exchanges": exchanges,
            "transcript_hash": hashlib.sha512(Wi.TRANSCRIPT_TAG + b"".join(transcript)).hexdigest(),
            "final": final.hex(), "final_sha512": _sha512(final)}


def full_width_response(setup_source: bytes = FULL_SETUP_SOURCE, audit_source: bytes = FULL_AUDIT_SOURCE,
                        counter: int = 0) -> dict:
    s = _setup(FULL, bytes(FULL.payload_bytes), [0], setup_source)
    audit_id = R.audit_id(audit_source, s["epoch"], counter)
    response = Wi.response_message(s["epoch"], audit_id, 0, 0, s["blocks"][0], [])
    return {"w": FULL.w, "manifest": s["manifest"].decode(), "W": bytes(FULL.payload_bytes).hex(),
            "W_bit_length": 8 * FULL.payload_bytes, "w_commitment": s["commitment"].hex(),
            "setup_source": setup_source.hex(), "setup_salt": s["salt"].hex(), "block": s["blocks"][0].hex(),
            "block_sha512": _sha512(s["blocks"][0]), "verification_root": s["root"].hex(), "epoch_id": s["epoch"].hex(),
            "audit_id": audit_id.hex(), "audit_source": audit_source.hex(), "audit_counter": counter,
            "response": response.hex(), "response_sha512": _sha512(response), "response_length": len(response),
            "_setup": s}


def audit_vectors() -> dict:
    full = full_width_response()
    full.pop("_setup")
    return {"schema": "p2-audit-wire-v4/test-vectors", "randomness": randomness_version(),
            "small_end_to_end": small_vector(), "full_width_response": full}


# --- audit-negative-vectors.json -------------------------------------------------------------------------------------------

def _flip(data: bytes, offset: int, value: int | None = None) -> bytes:
    out = bytearray(data)
    out[offset] = out[offset] ^ 1 if value is None else value
    return bytes(out)


def _splice(data: bytes, start: int, stop: int, value: bytes) -> bytes:
    return data[:start] + value + data[stop:]


def _case(name: str, category: str, frames: list[bytes], elapsed: list[int], source: bytes, expected: int, stop_round: int,
          reused: bool = False) -> dict:
    seen: set = set()
    if reused:
        body = next(f for f in frames if f[5] == Wi.AUDIT_OPEN)[10:]
        seen = {(body[:Wi.D], body[Wi.D:Wi.D + Wi.S])}
    final = Wi.verify_audit(frames, elapsed, verifier_source=source, seen_audits=seen)
    verdict, code, count = Wi.parse_final(final)
    if code != expected or verdict != (expected == Wi.FAIL_SUCCESS):
        raise AssertionError(f"{name}: got verdict={verdict}, code={code}, expected {expected}")
    return {"name": name, "category": category, "frames": [f.hex() for f in frames], "elapsed_ns": elapsed,
            "verifier_source": source.hex(), "seen_audit_pair": reused, "expected_verdict": expected == Wi.FAIL_SUCCESS,
            "expected_failure_code": expected, "expected_stop_round": stop_round, "expected_record_count": count,
            "expected_final": final.hex(), "expected_final_sha512": _sha512(final)}


def negative_vectors(vector: dict) -> dict:
    """The kit's 31 cases at v4's offsets (frame offset = body offset + 10): CHALLENGE's epoch at 10, audit id at 74,
    round at 106, index at 108, nonce at 116; AUDIT_OPEN's k at 107 and flags at 125; a w = 64 RESPONSE's block at 120,
    path count at 128 and first sibling at 130; SETUP_ROOT's root at 114."""
    M, F, I, A = Wi.FAIL_MALFORMED, Wi.FAIL_IDENTITY, Wi.FAIL_AUTHENTICATION, Wi.FAIL_SUCCESS
    base = [bytes.fromhex(m["bytes"]) for m in vector["setup_messages"]] + [bytes.fromhex(vector["audit_open"])]
    for x in vector["exchanges"]:
        base += [bytes.fromhex(x["challenge"]), bytes.fromhex(x["response"])]
    elapsed, nblocks = vector["elapsed_ns"], len(vector["blocks"])
    setup_count = len(vector["setup_messages"])
    open_at, c0_at, r0_at, c1_at, r1_at = (setup_count + i for i in range(5))
    cases = []

    def add(name, category, frames, expected, stop, times=None, reused=False):
        cases.append(_case(name, category, frames, elapsed if times is None else times, SMALL_AUDIT_SOURCE, expected, stop,
                           reused))

    def last(at, edit):
        frames = base[:at + 1]
        frames[-1] = edit(frames[-1])
        return frames

    def at_open(edit):
        frames = base[:open_at + 1]
        frames[open_at] = edit(frames[open_at])
        return frames

    add("wrong-magic", "malformed", last(c0_at, lambda f: _flip(f, 0)), M, 0)
    add("wrong-version", "malformed", last(c0_at, lambda f: _flip(f, 4, 2)), M, 0)
    add("wrong-type", "malformed", last(c0_at, lambda f: _flip(f, 5, Wi.RESPONSE)), M, 0)
    add("wrong-body-length", "malformed",
        last(c0_at, lambda f: _splice(f, 6, 10, (int.from_bytes(f[6:10], "big") + 1).to_bytes(4, "big"))), M, 0)
    add("trailing-byte", "malformed", last(r0_at, lambda f: f + b"\x00"), M, 0)
    add("truncated-response", "malformed", last(r0_at, lambda f: f[:-1]), M, 0)
    add("index-outside-population", "malformed", last(c0_at, lambda f: _splice(f, 108, 116, nblocks.to_bytes(8, "big"))), F, 0)
    add("zero-round-count", "malformed", at_open(lambda f: _splice(f, 107, 109, (0).to_bytes(2, "big"))), M, 0)
    add("round-count-over-maximum", "malformed", at_open(lambda f: _splice(f, 107, 109, (106).to_bytes(2, "big"))), M, 0)
    add("reserved-verification-flags", "malformed", at_open(lambda f: _flip(f, 125, 0x81)), M, 0)
    add("payload-compare-without-decode-flag", "malformed", at_open(lambda f: _flip(f, 125, 0x02)), M, 0)
    frames = base[:open_at + 1]
    frames[1] = _flip(frames[1], 10)
    add("setup-salt-commitment-mismatch", "setup", frames, F, 0)

    add("wrong-epoch-id", "identity", last(c0_at, lambda f: _flip(f, 10)), F, 0)
    add("wrong-audit-id", "identity", last(c0_at, lambda f: _flip(f, 74)), F, 0)
    add("wrong-round-number", "identity", last(c0_at, lambda f: _splice(f, 106, 108, (1).to_bytes(2, "big"))), F, 0)
    add("wrong-index", "identity", last(c0_at, lambda f: _splice(
        f, 108, 116, ((int.from_bytes(f[108:116], "big") + 1) % nblocks).to_bytes(8, "big"))), F, 0)
    add("replayed-challenge", "replay", last(c1_at, lambda f: base[c0_at]), F, 1)
    add("replayed-response", "replay", last(r1_at, lambda f: base[r0_at]), F, 1)
    add("reused-epoch-audit-pair", "replay", base[:open_at + 1], F, 0, reused=True)

    epoch, audit_id = bytes.fromhex(vector["epoch_id"]), bytes.fromhex(vector["audit_id"])
    wrong = derive(SMALL_AUDIT_SOURCE, "verity/pous/audit-challenge/v2",
                   {"audit": audit_id, "epoch": epoch, "root": bytes.fromhex(vector["verification_root"]),
                    "salt": bytes.fromhex(vector["setup_salt"])})
    add("wrong-challenge-domain", "domain",
        base[:c0_at] + [Wi.challenge_message(epoch, audit_id, 0, wrong.uniform(nblocks, 0), wrong.stream(("nonce", 0), 32))],
        F, 0)

    add("flipped-sibling", "authentication", last(r0_at, lambda f: _flip(f, 130)), I, 0)
    count = int.from_bytes(base[r0_at][128:130], "big")
    add("declared-path-count-body-mismatch", "malformed",
        last(r0_at, lambda f: _splice(f, 128, 130, (count + 1).to_bytes(2, "big"))), M, 0)

    def drop_sibling(f):
        out = bytearray(f[:-Wi.D])
        out[6:10] = (len(out) - 10).to_bytes(4, "big")
        out[128:130] = (count - 1).to_bytes(2, "big")
        return bytes(out)

    add("well-formed-wrong-sibling-count", "malformed", last(r0_at, drop_sibling), M, 0)
    swapped = bytes.fromhex(vector["blocks"][(vector["exchanges"][0]["index"] + 1) % nblocks])
    add("swapped-leaf", "authentication", last(r0_at, lambda f: _splice(f, 120, 128, swapped)), I, 0)
    frames = base[:open_at + 1]
    frames[setup_count - 1] = _flip(frames[setup_count - 1], 114)
    add("wrong-verification-root", "authentication", frames, I, 0)

    add("late-by-one-nanosecond", "deadline", base[:r0_at + 1], Wi.FAIL_LATE, 0, times=[R.DELTA_NS + 1])
    add("exact-deadline-boundary", "deadline", list(base), A, 2, times=[R.DELTA_NS, R.DELTA_NS])
    add("transport-failure-before-response", "transport", base[:r0_at], Wi.FAIL_TRANSPORT, 0, times=[123_000])
    add("extra-frame-after-k-exchanges", "identity", list(base) + [base[c0_at]], F, len(vector["exchanges"]))

    full = full_width_response(NEGATIVE_FULL_SETUP_SOURCE, NEGATIVE_FULL_AUDIT_SOURCE, 11)
    s = full["_setup"]
    aid = bytes.fromhex(full["audit_id"])
    key = R.audit_key(NEGATIVE_FULL_AUDIT_SOURCE, s["epoch"], s["root"], s["salt"], 11)[1]
    head = s["messages"] + [Wi.open_message(s["epoch"], aid, 0, 1, R.DELTA_NS, 0, 0x01, 11),
                            Wi.challenge_message(s["epoch"], aid, 0, 0, key.stream(("nonce", 0), 32))]
    for name, block in (("full-width-block-equal-p", FULL.p.to_bytes(FULL.width_bytes, "big")),
                        ("full-width-wrong-block-length", s["blocks"][0][:-1])):
        cases.append(_case(name, "malformed", head + [Wi.response_message(s["epoch"], aid, 0, 0, block, [])], [100_000],
                           NEGATIVE_FULL_AUDIT_SOURCE, M, 0))
    return {"schema": "p2-audit-wire-v4/negative-vectors", "randomness": randomness_version(),
            "failure_codes": Wi.FAILURE_CODES,
            "setup_sources": {"small_end_to_end": SMALL_SETUP_SOURCE.hex(), "full_width": NEGATIVE_FULL_SETUP_SOURCE.hex()},
            "case_count": len(cases), "cases": cases}


# --- the package's keys ----------------------------------------------------------------------------------------------------

def keys_fixture(maps: dict, maps_sha512: str) -> dict:
    keep = ("id", "w", "c0", "model_commitment", "setup_salt", "segment", "block", "K", "t", "t_rejection_index")
    return {"source": "benchmarks/pous/p2_v1/vectors.json, P2 v4's p2-test-vectors/v4 (the spec's section 7)",
            "source_sha512": maps_sha512, "randomness": maps["randomness"],
            "vectors": [{k: v[k] for k in keep} for v in maps["vectors"]]}


# --- writing and pinning ---------------------------------------------------------------------------------------------------

def _dump(doc: dict) -> bytes:
    return (json.dumps(doc, indent=2, sort_keys=True) + "\n").encode()


def pins(maps: dict, audit: dict, negative: dict, files: dict[str, bytes]) -> dict[str, str]:
    """Every hash the spec's §7 and `reference.py` pin, by name."""
    small, full = audit["small_end_to_end"], audit["full_width_response"]
    out = {name: _sha512(raw) for name, raw in files.items()}
    out |= {f"entry/{v['id']}": v["sha512"] for v in maps["vectors"]}
    out |= {"small/audit_open": small["audit_open_sha512"], "small/challenge0": small["exchanges"][0]["challenge_sha512"],
            "small/response0": small["exchanges"][0]["response_sha512"], "small/final": small["final_sha512"],
            "full/response": full["response_sha512"]}
    return out


def _old_pins() -> dict[str, str]:
    try:
        maps, audit, negative = (json.loads(p.read_text()) for p in (R.VECTORS, AUDIT_VECTORS, NEGATIVE_VECTORS))
        files = {p.name: p.read_bytes() for p in (R.VECTORS, AUDIT_VECTORS, NEGATIVE_VECTORS)}
        return pins(maps, audit, negative, files)
    except (OSError, KeyError, ValueError):
        return {}


def main() -> None:
    old = _old_pins()
    maps = map_vectors()
    audit = audit_vectors()
    negative = negative_vectors(audit["small_end_to_end"])
    files = {R.VECTORS.name: _dump(maps), AUDIT_VECTORS.name: _dump(audit), NEGATIVE_VECTORS.name: _dump(negative)}
    for name, raw in files.items():
        (HERE / name).write_bytes(raw)
    KEYS_FIXTURE.write_bytes(_dump(keys_fixture(maps, _sha512(files[R.VECTORS.name]))))
    new = pins(maps, audit, negative, files)
    for path in PINNED_TEXT:
        text = path.read_text()
        for name, value in new.items():
            if old.get(name) and old[name] != value:
                text = text.replace(old[name], value)
        path.write_text(text)
    print(f"randomness {maps['randomness']}; {negative['case_count']} negative cases")
    for name, value in new.items():
        print(f"{name} {value}")


if __name__ == "__main__":
    main()
