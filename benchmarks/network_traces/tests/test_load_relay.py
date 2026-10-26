"""load_relay's streamed, multi-process commit of the weights' load against its references: the manifest and generator
are CMP-95 interface 1's (its fixture's TINY manifest and tensor digests, and at H200 its stream's length, head and every
tensor's first and last rows); the relayed roots are interface 1's fixture roots under its salts and domain, over one
process and several, whatever the reads' sizes, and `Registered.commit`'s under the certifier's own salts (the witness
it writes) and the verifier's domain, pads and many chunks a value included; a P2 link (interface 2's layout over a
small field, so that the test encodes its own store) decodes to the same roots, and a block that isn't a codeword, a
nonzero byte past a store tensor's end and a short link are refused; at H200 the store's layout is interface 2's; and
the command signs one load head a value, each registration read by the verifier for the load it expects."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import sys
from pathlib import Path

import pytest

from verity.core.protocols.certifier import transfer_rows as TR
from verity.core.protocols.certifier.certificate import Direction
from verity.core.protocols.pous.schemes import p2 as P2
from verity.core.protocols.sampled_proofs.one_stage import registered as G

HERE = Path(__file__).resolve().parents[1]
FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "weights-v1-tiny.json").read_text())
SEED, SALT_KEY = bytes.fromhex(FIXTURE["seed"]), bytes.fromhex(FIXTURE["salt_key"])
PROGRAM = hashlib.sha512(b"the reading program").hexdigest()
#: a field small enough to encode in a test: p = 2^256 - 189, prime and 3 mod 4
SMALL = (256, 189, 64)


def _load():
    sys.path.insert(0, str(HERE))
    spec = importlib.util.spec_from_file_location("network_traces_load_relay", HERE / "load_relay.py")
    mod = sys.modules[spec.name] = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


LR = _load()
TINY = LR.manifest(LR.TINY)


def interface_1_domain(v):
    return G.domain(FIXTURE["tiny"]["manifest"]["program"], v.name, v.words, v.leaves)


def stream(values=TINY) -> bytes:
    buf = io.BytesIO()
    LR.write_stream(SEED, values, buf)
    return buf.getvalue()


def test_the_manifest_and_generator_are_interface_1s():
    ref = FIXTURE["tiny"]["manifest"]
    fields = lambda vs: [(v.name, v.offset, v.nbytes, v.words, v.leaves, v.kind, v.rows, v.exponent) for v in vs]  # noqa: E731
    assert fields(TINY) == fields(LR.read_manifest(ref))
    assert {v.name: hashlib.sha256(LR.generate(SEED, v)).hexdigest() for v in TINY} == \
        {t["name"]: t["sha256"] for t in ref["tensors"]}
    h200, want = LR.manifest(LR.H200), FIXTURE["h200"]
    assert len(h200) == 733 and h200[-1].offset + h200[-1].nbytes == want["stream_bytes"] == 139_736_443_904
    assert (LR.generate(SEED, h200[0], 0, 1)[:64]).hex() == want["stream_head"]
    rows = [[v.name, hashlib.sha256(LR.generate(SEED, v, 0, 1)).hexdigest(),
             hashlib.sha256(LR.generate(SEED, v, v.rows - 1, 1)).hexdigest()] for v in h200]
    assert hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest() == want["rows_sha256"]
    assert [r for r in rows if r[0] in {x[0] for x in want["rows"]}] == want["rows"]
    assert all(v.leaves & (v.leaves - 1) == 0 for v in h200), "at H200 no tree has a pad"


@pytest.mark.parametrize("workers,read", [(1, 1 << 23), (3, 1000), (2, 7)])
def test_the_relayed_roots_are_interface_1s_registration(workers, read):
    data = stream()
    sink = io.BytesIO()
    got = LR.relay(TINY, io.BytesIO(data), sink, interface_1_domain, salts=LR.keyed_salts(SALT_KEY), workers=workers,
                   chunk_bytes=256, read_bytes=read)
    assert {k: v.hex() for k, v in got.roots.items()} == {e["value"]: e["root"] for e in FIXTURE["tiny"]["values"]}
    assert sink.getvalue() == data and got.link_bytes == len(data)


def odd_values() -> tuple:
    """Values whose rows are no power of two, of several widths, so their trees have pads, in chunks of a few rows."""
    out, at = [], 0
    for name, rows, words in (("a", 5, 3), ("b", 1, 128), ("c", 37, 2), ("d", 64, 1), ("e", 3, 8)):
        out.append(LR.Value(name, at, 2 * words * rows, words))
        at += 2 * words * rows
    return tuple(out)


def test_the_relayed_roots_are_registered_commits_under_the_certifiers_salts(tmp_path):
    values = odd_values()
    data = os.urandom(sum(v.nbytes for v in values))
    witness = tmp_path / "witness.bin"
    got = LR.relay(values, io.BytesIO(data), None, LR.verifier_domain(PROGRAM), witness=witness, workers=2,
                   chunk_bytes=24, read_bytes=13)
    salts = witness.read_bytes()
    assert len(salts) == TR.SALT_BYTES * sum(v.leaves for v in values)
    at = 0
    for v in values:
        rows = [list(memoryview(data[v.offset + v.row_bytes * r:v.offset + v.row_bytes * (r + 1)]).cast("H"))
                for r in range(v.leaves)]
        mine = [salts[at + TR.SALT_BYTES * r:at + TR.SALT_BYTES * (r + 1)] for r in range(v.leaves)]
        t = TR.Transfer.commit(LR.LINK, Direction.INGRESS, 0, "load", PROGRAM, v.name, rows, mine)
        assert t.root == got.roots[v.name] and t.head.domain == LR.verifier_domain(PROGRAM)(v).domain_id, v.name
        at += TR.SALT_BYTES * v.leaves
    assert len(set(salts[i:i + TR.SALT_BYTES] for i in range(0, len(salts), TR.SALT_BYTES))) == len(salts) // 192
    with pytest.raises(LR.Refused, match=f"carried {len(data) - 1} bytes, and the load is {len(data)}"):
        LR.relay(values, io.BytesIO(data[:-1]), None, LR.verifier_domain(PROGRAM))
    with pytest.raises(LR.Refused, match="and the load is"):
        LR.relay(values, io.BytesIO(data + b"\0"), None, LR.verifier_domain(PROGRAM))


def encode_store(params, salt: bytes, values=TINY, shape=LR.TINY) -> tuple[bytes, bytes]:
    """The store's C* of `shape`'s s8 tensors under `salt` (`p2.encode_segment`'s arithmetic), and the rest."""
    by, f, pb, nb = {v.name: v for v in values}, P2._field(params), params.payload_bytes, params.block_bytes
    w = bytearray(params.w_bytes)
    stored = set()
    for t in LR.store_tensors(shape, params):
        w[t.first_block * pb:t.first_block * pb + t.size] = b"".join(LR.generate(SEED, by[n]) for n in t.values)
        stored |= set(t.values)
    out = []
    for i in range(params.blocks):
        s, j = divmod(i, params.blocks_per_segment)
        k, t = P2.segment_keys(params, salt, s)(j)
        m = int.from_bytes(w[i * pb:(i + 1) * pb], "big")
        out.append(f.rho(f.sigma(f.rho((m + t) % f.p), k)).to_bytes(nb, "big"))
    return b"".join(out), b"".join(LR.generate(SEED, v) for v in values if v.name not in stored)


@pytest.fixture(scope="module")
def store():
    params, salt = LR.store_params(LR.TINY, *SMALL), hashlib.sha512(b"the store's salt").digest()[:24]
    cstar, rest = encode_store(params, salt)
    return params, salt, cstar, rest


def p2(link: bytes, store, **kw):
    params, salt, _c, _r = store
    return LR.relay(TINY, io.BytesIO(link), kw.pop("sink", None), interface_1_domain, fmt="p2", shape=LR.TINY,
                    params=params, salt=salt, salts=LR.keyed_salts(SALT_KEY), chunk_bytes=256, **kw)


@pytest.mark.parametrize("workers,read", [(1, 1 << 23), (3, 1000)])
def test_a_p2_link_decodes_to_the_same_roots(store, workers, read):
    params, _s, cstar, rest = store
    sink = io.BytesIO()
    got = p2(cstar + rest, store, sink=sink, workers=workers, read_bytes=read)
    assert {k: v.hex() for k, v in got.roots.items()} == {e["value"]: e["root"] for e in FIXTURE["tiny"]["values"]}
    assert sink.getvalue() == cstar + rest and len(cstar) == params.blocks * params.block_bytes


def test_a_p2_link_that_doesnt_decode_is_refused(store):
    params, salt, cstar, rest = store
    nb, pb = params.block_bytes, params.payload_bytes
    flipped = bytearray(cstar)
    flipped[nb * 3 + 5] ^= 1
    with pytest.raises(LR.Refused, match="block 3 decodes to more than 240 bits: not a codeword"):
        p2(bytes(flipped) + rest, store, workers=2)
    high = cstar[:nb * 7] + params.p.to_bytes(nb, "big") + cstar[nb * 8:]
    with pytest.raises(LR.Refused, match="block 7 is not below p"):
        p2(high + rest, store)
    qkv = LR.store_tensors(LR.TINY, params)[0]
    last = qkv.first_block + qkv.blocks - 1
    s, j = divmod(last, params.blocks_per_segment)
    f, (k, t) = P2._field(params), P2.segment_keys(params, salt, s)(j)
    m = bytearray(LR.decode_blocks(params, salt, last, cstar[nb * last:nb * (last + 1)]))
    assert qkv.size % pb and not any(m[qkv.size % pb:])
    m[-1] ^= 1
    c = f.rho(f.sigma(f.rho((int.from_bytes(m, "big") + t) % f.p), k)).to_bytes(nb, "big")
    with pytest.raises(LR.Refused, match=f"block {last} is nonzero past its tensor's end"):
        p2(cstar[:nb * last] + c + cstar[nb * (last + 1):] + rest, store)
    with pytest.raises(LR.Refused, match="and the load is"):
        p2(cstar + rest[:-1], store)


def test_the_store_layout_is_interface_2s_at_h200():
    params = LR.store_params(LR.H200)
    t = LR.store_tensors(LR.H200, params)
    assert (params.segments, params.blocks, params.blocks * params.block_bytes) == (1038, 68_026_368, 139_862_212_608)
    assert params.w_bytes == 139_726_159_872 and params.prime_id == "p2-p16448-c21065"
    assert [x.blocks for x in t[:4]] == [131_711, 130_690, 522_757, 522_757]
    assert all(t[4 * i].first_block == 1_307_915 * i for i in range(52)) and t[-1].values == ("head",)
    stored = {n for x in t for n in x.values}
    assert sum(v.nbytes for v in LR.manifest(LR.H200) if v.name not in stored) == 36_760_576
    jobs, length = LR.plan(LR.manifest(LR.H200), "p2", LR.H200, params)
    assert length == 139_862_212_608 + 36_760_576 and len(jobs) == len({(j.value, j.chunk) for j in jobs})


def test_the_command_signs_a_load_head_a_value(tmp_path):
    src = tmp_path / "stream.bin"
    src.write_bytes(stream())
    out = tmp_path / "out"
    assert LR.main(["relay", "--program", PROGRAM, "--out", str(out), "--shape", "tiny", "--source", str(src),
                    "--sink", str(tmp_path / "sink.bin"), "--height", "5", "--workers", "2"]) == 0
    doc = json.loads((out / "load.json").read_text())
    assert (tmp_path / "sink.bin").read_bytes() == src.read_bytes() and len(doc["values"]) == len(TINY)
    import certifier_serve as CS
    key = CS.read_public(doc["key"])
    for v, reg in zip(TINY, doc["values"]):
        head, dom = TR.check_registration(reg, key, bytes.fromhex(doc["session"]), link=LR.LINK,
                                          direction=Direction.INGRESS, window=0, kind="load", value=v.name,
                                          program=PROGRAM, words=v.words, rows=v.leaves)
        assert dom.domain_id == TR.domain(PROGRAM, v.name, v.words, v.leaves).domain_id
    salts = (out / "load-witness.bin").read_bytes()
    v = TINY[3]
    rows = [list(memoryview(LR.generate(SEED, v)[v.row_bytes * r:v.row_bytes * (r + 1)]).cast("H"))
            for r in range(v.leaves)]
    at = TR.SALT_BYTES * sum(x.leaves for x in TINY[:3])
    mine = [salts[at + TR.SALT_BYTES * r:at + TR.SALT_BYTES * (r + 1)] for r in range(v.leaves)]
    assert TR.Transfer.commit(LR.LINK, Direction.INGRESS, 0, "load", PROGRAM, v.name, rows, mine).root.hex() == \
        doc["values"][3]["root"]
    src.write_bytes(src.read_bytes()[:-1])
    assert LR.main(["relay", "--program", PROGRAM, "--out", str(tmp_path / "short"), "--shape", "tiny", "--source",
                    str(src), "--key", str(out / "certifier"), "--workers", "1"]) == 1
    assert not (tmp_path / "short" / "load.json").exists()
