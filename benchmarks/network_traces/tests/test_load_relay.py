"""load_relay's streamed, multi-process commit of the weights' load against its references: the manifest and generator
are the seeded weights' (`e2e_h200.served_weights`, interface 1: its fixture's TINY manifest, tensor digests and
entries, and at H200 its stream's length, head and every tensor's first and last rows); the relayed roots are interface
1's fixture roots under its salts, for its reader named by digest, whose domains are the verifier's (``tok`` (H, V) in
row/v2, every tensor 16-bit leaves, an f32 scale two, at H200 a linear's row the 8,192 codes of a FAMILIES ``wrow``
part), over one process and several, whatever the reads' sizes, and `Registered.commit`'s under the certifier's own
salts (the witness it writes) and the verifier's domain, pads, many chunks a value and both row formats included; a P2
link (interface 2's layout over a small field, so that the test encodes its own store) decodes to the same roots, and a
block that isn't a codeword, a nonzero byte past a store tensor's end and a short link are refused; at H200 the store's
layout is interface 2's; and the command signs one load head a value, each registration read by the verifier for the
load it expects.

The gate's load: the store's pins are interface 1's H200 manifest digests (a fetched fixture) and its generator's TINY
ones under the demo seed; the store is written once, read back to its pin, kept when present and refused (unrecorded)
when it isn't interface 1's; the load crosses four streams to a loopback sink (a tunnel's probe ignored) with its
witness, which the sink alone keeps and which reproduces every root; a store whose bytes aren't their digests, a sink
that doesn't answer and a malformed heads document are refused; and the load's registrations read back under the
signer's key and session only."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import socket
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from verity.core.protocols.certifier import transfer_rows as TR
from verity.core.protocols.certifier.certificate import Direction, Issuer
from verity.core.protocols.pous.schemes import p2 as P2
from verity.core.protocols.sampled_proofs.one_stage import registered as G

HERE = Path(__file__).resolve().parents[1]
FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "weights-v1-tiny.json").read_text())
H200_DIGESTS = json.loads((Path(__file__).parent / "fixtures" / "weights-v1-h200-digests.json").read_text())
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


#: interface 1's fixture reader is a program digest, so its domains are the verifier's own
interface_1_domain = LR.verifier_domain(FIXTURE["program"])


def leaves(v, data: bytes) -> list[list[int]]:
    """`data`'s rows of `v` as `Registered.commit` takes them: each row's 16-bit leaves (`v.word_bits`), little-endian, so
    an f32 scale's low half first."""
    cut = memoryview(data).cast("H")
    return [list(cut[v.reg_words * r:v.reg_words * (r + 1)]) for r in range(len(data) // v.row_bytes)]


def stream(values=TINY) -> bytes:
    buf = io.BytesIO()
    LR.write_stream(SEED, values, buf)
    return buf.getvalue()


def test_the_manifest_and_generator_are_interface_1s():
    ref = FIXTURE["tiny"]["manifest"]
    fields = lambda vs: [(v.name, v.offset, v.nbytes, v.words, v.leaves, v.kind, v.rows, v.exponent, v.schema)  # noqa: E731
                         for v in vs]
    assert fields(TINY) == fields(LR.read_manifest(ref))
    assert {v.name: hashlib.sha256(LR.generate(SEED, v)).hexdigest() for v in TINY} == \
        {t["name"]: t["sha256"] for t in ref["tensors"]}
    assert [(v.name, v.leaves, v.reg_words, v.word_bits, interface_1_domain(v).domain_id.hex()) for v in TINY] == \
        [(e["value"], e["leaves"], e["words"], e.get("word_bits", 16), e["domain"]) for e in FIXTURE["tiny"]["values"]]
    assert (TINY[0].name, TINY[0].schema, TINY[0].rows, TINY[0].words) == ("tok", G.SCHEMA_V2, 32, 256)
    h200, want = LR.manifest(LR.H200), FIXTURE["h200"]
    assert len(h200) == 733 and h200[-1].offset + h200[-1].nbytes == want["stream_bytes"] == 139_736_443_904
    assert (LR.generate(SEED, h200[0], 0, 1)[:64]).hex() == want["stream_head"]
    rows = [[v.name, hashlib.sha256(LR.generate(SEED, v, 0, 1)).hexdigest(),
             hashlib.sha256(LR.generate(SEED, v, v.rows - 1, 1)).hexdigest()] for v in h200]
    assert hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest() == want["rows_sha256"]
    assert [r for r in rows if r[0] in {x[0] for x in want["rows"]}] == want["rows"]
    assert all(v.leaves & (v.leaves - 1) == 0 for v in h200), "at H200 no tree has a pad"
    by = {v.name: v for v in h200}
    assert {v.word_bits for v in h200} == {16} and sum(v.kind == "scale" for v in h200) == 313
    assert (by["l0.wup"].reg_words, by["l0.wup"].leaves) == (4096, 65536 * 2)
    assert (by["l0.wup.scale"].reg_words, by["l0.wup.scale"].leaves) == (2, 65536)
    assert sum(v.leaves for v in h200) == 25_284_864


@pytest.mark.parametrize("workers,read", [(1, 1 << 23), (3, 1000), (2, 7)])
def test_the_relayed_roots_are_interface_1s_registration(workers, read):
    data = stream()
    sink = io.BytesIO()
    got = LR.relay(TINY, io.BytesIO(data), sink, interface_1_domain, salts=LR.keyed_salts(SALT_KEY), workers=workers,
                   chunk_bytes=256, read_bytes=read)
    assert {k: v.hex() for k, v in got.roots.items()} == {e["value"]: e["root"] for e in FIXTURE["tiny"]["values"]}
    assert sink.getvalue() == data and got.link_bytes == len(data)


def odd_values() -> tuple:
    """Values whose rows are no power of two, of several widths, so their trees have pads, in chunks of a few rows: one
    of f32 scales (``f``, two 16-bit leaves a row) and one of row-seg rows longer than a segment (``g``)."""
    out, at = [], 0
    for name, rows, words, kind, schema in (("a", 5, 3, "s8", G.SCHEMA_V2), ("b", 1, 128, "s8", G.SCHEMA_V2),
                                            ("c", 37, 2, "s8", G.SCHEMA_V2), ("d", 64, 1, "gain", G.SCHEMA_V2),
                                            ("e", 3, 8, "s8", G.SCHEMA_V2), ("f", 11, 2, "scale", G.SCHEMA_V2),
                                            ("g", 3, 600, "embed", G.SCHEMA_SEG)):
        out.append(LR.Value(name, at, 2 * words * rows, words, kind, schema=schema))
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
        mine = [salts[at + TR.SALT_BYTES * r:at + TR.SALT_BYTES * (r + 1)] for r in range(v.leaves)]
        t = TR.Transfer.commit(LR.LINK, Direction.INGRESS, 0, "load", PROGRAM, v.name,
                               leaves(v, data[v.offset:v.offset + v.nbytes]), mine, v.word_bits, v.schema)
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
                                          program=PROGRAM, words=v.reg_words, rows=v.leaves, word_bits=v.word_bits,
                                          schema=v.schema)
        assert dom.domain_id == TR.domain(PROGRAM, v.name, v.reg_words, v.leaves, v.word_bits).domain_id
    salts = (out / "load-witness.bin").read_bytes()
    for i in (0, 3, 4):
        v, at = TINY[i], TR.SALT_BYTES * sum(x.leaves for x in TINY[:i])
        mine = [salts[at + TR.SALT_BYTES * r:at + TR.SALT_BYTES * (r + 1)] for r in range(v.leaves)]
        t = TR.Transfer.commit(LR.LINK, Direction.INGRESS, 0, "load", PROGRAM, v.name, leaves(v, LR.generate(SEED, v)),
                               mine, v.word_bits, v.schema)
        assert t.root.hex() == doc["values"][i]["root"], v.name
    src.write_bytes(src.read_bytes()[:-1])
    assert LR.main(["relay", "--program", PROGRAM, "--out", str(tmp_path / "short"), "--shape", "tiny", "--source",
                    str(src), "--key", str(out / "certifier"), "--workers", "1"]) == 1
    assert not (tmp_path / "short" / "load.json").exists()


# --- the gate's load: the store, the streams to the H200's sink, the heads and their registrations -----------------------

def demo_stream(values=TINY) -> bytes:
    buf = io.BytesIO()
    LR.write_stream(LR.DEMO_SEED, values, buf)
    return buf.getvalue()


def test_the_stores_pins_are_interface_1s_digests():
    h200, rows = LR.manifest(LR.H200), H200_DIGESTS["tensors"]
    assert [(v.name, v.offset, v.nbytes) for v in h200] == [tuple(r[:3]) for r in rows]
    assert LR.digests_fold(h200, [r[3] for r in rows]) == LR.STORE_DIGESTS[LR.H200.name]
    assert LR.model_id(LR.H200) == H200_DIGESTS["model"] and H200_DIGESTS["stream_bytes"] == 139_736_443_904
    by = {r[0]: r[3] for r in rows}
    small = ("pos", "l0.attn_norm", "l0.wk.scale", "l0.wv.scale", "l51.mlp_norm", "final_norm", "head.scale")
    assert {v.name: hashlib.sha256(LR.generate(LR.DEMO_SEED, v)).hexdigest() for v in h200 if v.name in small} == \
        {n: by[n] for n in small}
    assert LR.digests_fold(TINY, [hashlib.sha256(LR.generate(LR.DEMO_SEED, v)).hexdigest() for v in TINY]) == \
        LR.STORE_DIGESTS[LR.TINY.name]


def test_the_store_is_written_once_and_held_to_its_pin(tmp_path):
    store = tmp_path / "store"
    with pytest.raises(LR.Refused, match="no store directory"):
        LR.write_store(store, LR.TINY)
    store.mkdir()
    with pytest.raises(LR.Refused, match="no store record .*weights-write"):
        LR.read_store(store, LR.TINY)
    got = LR.write_store(store, LR.TINY, workers=2, chunk_bytes=1000)
    data, rec = LR.store_paths(store, LR.TINY)
    assert got["status"] == "written" and data.read_bytes() == demo_stream()
    assert (got["fold"], got["model"]) == (LR.STORE_DIGESTS[LR.TINY.name], LR.model_id(LR.TINY))
    assert sorted(p.name for p in store.iterdir()) == sorted([data.name, rec.name])
    assert LR.read_store(store, LR.TINY) == (data, got["sha256"])
    assert LR.write_store(store, LR.TINY)["status"] == "present"
    raw = bytearray(data.read_bytes())
    raw[100] ^= 1
    data.write_bytes(raw)
    with pytest.raises(LR.Refused, match="isn't interface 1's tiny-h32-l1 weights stream .*left as it is"):
        LR.write_store(store, LR.TINY)
    assert data.exists() and not rec.exists()
    with pytest.raises(LR.Refused, match="no store record"):
        LR.read_store(store, LR.TINY)
    data.unlink()
    assert LR.write_store(store, LR.TINY)["status"] == "written"
    with open(data, "r+b") as fh:
        fh.truncate(len(raw) - 1)
    with pytest.raises(LR.Refused, match=f"isn't its {len(raw):,} bytes"):
        LR.read_store(store, LR.TINY)
    with pytest.raises(LR.Refused, match=f"{len(raw) - 1:,} bytes, not {len(raw):,}"):
        LR.write_store(store, LR.TINY)
    data.unlink()
    LR.write_store(store, LR.TINY)
    doc = json.loads(rec.read_text())
    doc["sha256"][0] = "00" * 32
    rec.write_text(json.dumps(doc))
    with pytest.raises(LR.Refused, match="record .* isn't interface 1's tiny-h32-l1 stream"):
        LR.read_store(store, LR.TINY)
    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(LR.Refused, match="generated tiny-h32-l1 stream isn't interface 1's .*nothing is kept"):
        LR.write_store(other, LR.TINY, pinned={LR.TINY.name: "00" * 64})
    assert not list(other.iterdir())
    with pytest.raises(LR.Refused, match="no pinned digests of tiny-h32-l1"):
        LR.write_store(other, LR.TINY, pinned={})


class Sink:
    """The prover's ``load_relay.py sink`` on a loopback port, in its own process: its `port`, and once `join`ed its
    outcome `res` and exit status `code`."""

    def __init__(self, tmp_path, streams=4):
        self.proc = subprocess.Popen([sys.executable, str(HERE / "load_relay.py"), "sink", "--listen", "127.0.0.1:0",
                                      "--streams", str(streams), "--weights", str(tmp_path / "weights.bin"),
                                      "--witness", str(tmp_path / "witness.bin")], stdout=subprocess.PIPE, text=True)
        self.port = json.loads(self.proc.stdout.readline())["listening"][1]

    def join(self):
        out, _ = self.proc.communicate()
        self.res, self.code = json.loads(out.splitlines()[-1]), self.proc.returncode
        return self.res


def test_the_load_crosses_four_streams_to_a_loopback_sink(tmp_path):
    store = tmp_path / "store"
    store.mkdir()
    LR.write_store(store, LR.TINY)
    s = Sink(tmp_path)
    socket.create_connection(("127.0.0.1", s.port)).close()
    out = tmp_path / "out"
    rep = LR.relay_load(store, LR.TINY, [("127.0.0.1", s.port)] * 4, PROGRAM, out, workers=2, frame_bytes=1000)
    got, data = s.join(), demo_stream()
    assert s.code == 0 and got["accepted"] and got["errors"] == []
    assert (tmp_path / "weights.bin").read_bytes() == data
    salts, rows = (tmp_path / "witness.bin").read_bytes(), sum(v.leaves for v in TINY)
    assert len(salts) == got["witness_bytes"] == rep["witness_bytes"] == TR.SALT_BYTES * rows
    assert stat.S_IMODE((tmp_path / "witness.bin").stat().st_mode) == 0o600
    assert (rep["link_bytes"], rep["rows"], rep["values"], rep["model"]) == (len(data), rows, len(TINY),
                                                                             LR.model_id(LR.TINY))
    assert len(rep["streams"]) == 4 and all(s["weights"] and s["witness"] for s in rep["streams"])
    assert sorted(p.name for p in out.iterdir()) == ["load-heads.json", "load-relay.json"]
    shape, heads = LR.heads_of(json.loads((out / "load-heads.json").read_text()))
    at = 0
    for v, h in zip(TINY, heads):
        mine = [salts[at + TR.SALT_BYTES * r:at + TR.SALT_BYTES * (r + 1)] for r in range(v.leaves)]
        c = TR.Transfer.commit(LR.LINK, Direction.INGRESS, 0, "load", PROGRAM, v.name,
                               leaves(v, data[v.offset:v.offset + v.nbytes]), mine, v.word_bits, v.schema)
        assert c.root == h.root and h.domain == LR.verifier_domain(PROGRAM)(v).domain_id, v.name
        at += TR.SALT_BYTES * v.leaves
    import certifier_serve as CS
    own, session = CS.signer(tmp_path / "key", 5), os.urandom(32)
    regs = LR.sign_heads(Issuer(own, session), heads)
    doc = LR.load_doc(shape, heads, regs, CS.public_json(own.public_key()), session)
    key = CS.read_public(doc["key"])
    assert LR.check_load(doc, key, session) == ({"values": len(TINY), "registered": len(TINY)}, [])
    assert LR.check_load(doc, key, os.urandom(32))[0]["registered"] == 0
    assert len(LR.check_load(dict(doc, values=[regs[1], regs[0], *regs[2:]]), key, session)[1]) == 2
    assert LR.check_load(dict(doc, values=regs[:-1]), key, session)[1] == \
        [f"load: not one registration of each of {len(TINY)} values"]
    assert LR.check_load(dict(doc, format="x"), key, session)[1][0].startswith("load: not a load's registrations")


def test_a_load_that_isnt_the_stores_or_has_no_sink_is_refused(tmp_path):
    store, out = tmp_path / "store", tmp_path / "out"
    store.mkdir()
    with pytest.raises(LR.Refused, match="no store record"):
        LR.relay_load(store, LR.TINY, [("127.0.0.1", 1)], PROGRAM, out)
    assert "no store record" in json.loads((out / "load-relay.json").read_text())["refused"]
    LR.write_store(store, LR.TINY)
    s = socket.create_server(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    with pytest.raises(LR.Refused, match=f"the sink at 127.0.0.1:{port} doesn't answer"):
        LR.relay_load(store, LR.TINY, [("127.0.0.1", port)] * 4, PROGRAM, out)
    data = LR.store_paths(store, LR.TINY)[0]
    raw, v = bytearray(data.read_bytes()), TINY[5]
    raw[v.offset + 3] ^= 1
    data.write_bytes(raw)
    s = Sink(tmp_path)
    with pytest.raises(LR.Refused, match=f"{v.name}: its bytes' SHA-256 .* the store isn't interface 1's weights"):
        LR.relay_load(store, LR.TINY, [("127.0.0.1", s.port)] * 4, PROGRAM, out, read_bytes=64)
    assert s.join()["accepted"] is False and s.res["errors"] and s.code == 1
    assert json.loads((out / "load-relay.json").read_text())["refused"].startswith(f"{v.name}: ")
    assert not (out / "load-heads.json").exists()


def test_a_heads_document_is_the_loads_or_refused():
    roots = {v.name: os.urandom(64) for v in TINY}
    doc = LR.heads_doc(LR.TINY, PROGRAM, roots)
    shape, heads = LR.heads_of(json.loads(json.dumps(doc)))
    assert shape == LR.TINY and [h.root for h in heads] == list(roots.values())
    assert [(h.kind, h.window, h.direction, h.link) for h in heads] == [("load", 0, Direction.INGRESS, LR.LINK)] * len(TINY)
    for bad in (dict(doc, format="x"), dict(doc, window=1), dict(doc, shape="nope"), dict(doc, program=PROGRAM.upper()),
                dict(doc, roots=dict(reversed(list(doc["roots"].items())))), dict(doc, roots={**doc["roots"], "tok": "00"}),
                dict(doc, link=None)):
        with pytest.raises(ValueError):
            LR.heads_of(bad)
