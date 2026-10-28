"""verity/flock-circuit's Python side: the composite circuit, its instance files, and the tail pieces in
the circuit. The Rust statement's own tests are ``flock-circuit selftest`` on a pod (every negative) and the crate's unit tests."""
from __future__ import annotations

import json

import numpy as np
import pytest

from verity_flock import ir_lower as IL
from verity_flock import circuit as FC
from verity_flock import tail as TL
from verity_flock.gf2 import Circuit, const_bits

pytest.importorskip("verity_numerical")
from verity_numerical.bench import templates as TM  # noqa: E402
from verity_numerical.bench.generate import generate  # noqa: E402
from verity_numerical.bench.input_sets import InputSet  # noqa: E402


def _set(sid: str, n: int, tmp_path):
    summary = generate(TM.parse_subcircuit_id(sid), n, 20260926, tmp_path)
    return InputSet.open(summary["path"])


PUBLIC_HEADER = {"format", "statement", "scheme", "set", "subcircuit", "range", "instances", "units", "circuit_sha512", "rows", "frame_v3",
                 "shared_rows"}
PUBLIC_FRAME = {"frame", "roots", "domain_ids", "bindings", "owner", "schemas", "row_key_sha512"}


def assert_public_only(text: str, prover: bytes, verifier: bytes) -> None:
    """The verifier's file holds the frame-v3 roots, each row's hiding leaf ``b || c`` and structure, and nothing else: its
    header is the layout's (names, the range, the unit indices, the circuit's pin, the frame's domains bound to those alone),
    its body exactly the ``b || c`` (then a shared-row file's refs), each the hm96 commit of one of the prover's rows (an output
    row too) under a nonzero salt. An output word, a digest of the set's values or any other value fails it."""
    from verity.ml.commitments import hm96
    from verity.ml.commitments.identity import identity_digest_sha512
    from verity.ml.commitments.rowleaf import ROLE_X, sha512_row_digest

    meta = json.loads(text.split("\n", 2)[1][5:])
    (hp, bp), (hv, bv) = (b.split(b"\n", 1) for b in (prover, verifier))
    hp, hv = json.loads(hp), json.loads(hv)
    assert set(hv) <= PUBLIC_HEADER and set(hv["frame_v3"]) == PUBLIC_FRAME and hv == dict(hp, rows=False)
    assert verifier == FC.public_file(hv, bv), "the header line is C-Flock's, so the file's SHA-512 is public_file_sha512"
    ports = [(nm, w) for nm, w, _ in meta["ports"]]
    names = [nm for nm, _ in ports]
    assert names[meta["input_ports"]:] == [nm for nm, _ in meta["out_ports"]], "one output row per output port"
    fv, (lo, hi), n = hv["frame_v3"], hv["range"], hv["instances"]
    assert all(set(fv[k]) == set(names) for k in ("roots", "domain_ids", "bindings", "schemas"))
    assert set(fv["schemas"].values()) == {FC.ROW_SCHEMA}
    for nm in names:
        assert fv["bindings"][nm] == identity_digest_sha512(FC.BINDING_TAG, {"set": hv["set"], "lo": lo, "hi": hi, "port": nm,
                                                                             "schema": FC.ROW_SCHEMA}).hex()
    shared = "shared_rows" in hv
    counts = hv["shared_rows"] if shared else {nm: n for nm in names}
    n_bcs = sum(counts.values())
    assert len(bv) == FC.BC_BYTES * n_bcs + (4 * n * len(names) if shared else 0), "nothing but the b || c and refs"
    assert bp.endswith(bv)
    rb = sum(2 * w for _, w in ports)
    rows_len = sum(2 * w * counts[nm] for nm, w in ports) if shared else n * rb
    order = [(p, r) for p in range(len(ports)) for r in range(counts[names[p]])] if shared else \
        [(p, i) for i in range(n) for p in range(len(ports))]
    for k, (p, r) in enumerate(order):
        w = ports[p][1]
        at = (sum(2 * ports[q][1] * counts[names[q]] for q in range(p)) + 2 * w * r) if shared else r * rb + sum(2 * x for _, x in ports[:p])
        salt = bp[rows_len + k * FC.SALT_BYTES:][:FC.SALT_BYTES]
        words = np.frombuffer(bp[at:at + 2 * w], dtype="<u2").tolist()
        assert any(salt), "a hiding leaf: the salt is not zero"
        assert bv[k * FC.BC_BYTES:][:FC.BC_BYTES] == hm96.SHA512.commit_string(hm96.SHA512.default_key, sha512_row_digest(words, 16, ROLE_X), salt)


def test_the_verifiers_file_header_is_the_line_c_flock_writes():
    """`public_file`'s header line is serde_json's of the header (`Instances::public_file`): compact, keys sorted, UTF-8
    unescaped, ``rows`` false, so the prover's record's ``public_file_sha512`` is the staged file's SHA-512."""
    got = FC.public_file({"z": [1, {"b": None, "a": "é"}], "rows": True, "m": "s"}, b"\x00\xff")
    assert got == '{"m":"s","rows":false,"z":[1,{"a":"é","b":null}]}'.encode() + b"\n\x00\xff"


@pytest.mark.slow
def test_rope_composes_private_by_default(tmp_path):
    s = _set("rope-head/d64/neox-bf16", 4, tmp_path)
    low = FC.lowering_for_set(s)
    text = FC.compose(low, s.subcircuit)
    meta = json.loads(text.split("\n", 2)[1][5:])
    assert meta["statement"] == FC.STATEMENT and meta["leaf_scheme"] == FC.LEAF_SCHEME
    plain = json.loads(FC.compose(low, s.subcircuit, leaf_scheme=FC.LEAF_SCHEME_PLAIN).split("\n", 2)[1][5:])
    assert plain.pop("leaf_scheme") == FC.LEAF_SCHEME_PLAIN and plain == {k: v for k, v in meta.items() if k != "leaf_scheme"}
    assert not any(k in meta for k in ("tail", "cut", "public_ports", "native")), "no verifier-evaluated part"
    assert {r["net"] for r in meta["ranges"]} == {FC.SHA_NET, FC.HM_NET, "unit", "mask"}
    spans = sorted(((r["pos0"] << r["slot_log"]), ((r["pos0"] + (r["count"] if r.get("packed") else FC._pow2(r["count"]))) << r["slot_log"]),
                    bool(r.get("packed")), r["slot_log"]) for r in meta["ranges"])
    assert all(a[1] <= b[0] for a, b in zip(spans, spans[1:])), "ranges are disjoint"
    assert all(start % ((1 << sl) if packed else end - start) == 0 for start, end, packed, sl in spans), "ranges are aligned"
    assert [r["net"] for r in meta["ranges"] if r.get("packed")] == [FC.SHA_NET], "only the compressions are packed"
    assert meta["program_digests"] and meta["partition"]["rule"] == FC.PARTITION_RULE
    FC.stage(low, s, 0, 4, tmp_path / "st")
    with pytest.raises(ValueError, match="salt key"):
        FC.stage(low, s, 0, 4, tmp_path / "v", public_only=True)
    prover, verifier = (tmp_path / "st" / "inst-4.bin").read_bytes(), (tmp_path / "st" / "pub-4.bin").read_bytes()
    hp, hv = (json.loads(b.split(b"\n", 1)[0]) for b in (prover, verifier))
    assert hp["rows"] and not hv["rows"], "the verifier's file holds no row"
    assert prover.split(b"\n", 1)[1].endswith(verifier.split(b"\n", 1)[1]), "the public part is the same bytes"
    assert hp["circuit_sha512"] == FC.digest(text) and len(FC.digest(text)) == 128
    G = meta["vus_per_block"]
    assert hp["units"]["indices"] == [0, 1, 2, 3] and hp["units"]["blocks"] == [[0, 1, 2, 3][b:b + G] for b in range(0, 4, G)]
    assert len(meta["partition"]["digest"]) == 128, "a 64-byte SHA-512 partition digest"
    # a row's public b || c is hm96-sha512 of its private row and salt
    from verity.ml.commitments import hm96
    from verity.ml.commitments.rowleaf import ROLE_X, sha512_row_digest

    ports = meta["ports"]
    rb, np_ = sum(2 * w for _, w, _ in ports), len(ports)
    body = prover.split(b"\n", 1)[1]
    salt0, bc0 = body[4 * rb:4 * rb + FC.SALT_BYTES], body[4 * rb + 4 * np_ * FC.SALT_BYTES:][:FC.BC_BYTES]
    words = np.frombuffer(body[:2 * ports[0][1]], dtype="<u2").tolist()
    assert bc0 == hm96.SHA512.commit_string(hm96.SHA512.default_key, sha512_row_digest(words, 16, ROLE_X), salt0)
    # instance 0's output row: its output words, zero-padded to whole SHA-512 blocks
    j = meta["input_ports"]
    assert [p[0] for p in ports[j:]] == [p.name for p in s.subcircuit.outputs] and meta["per_vu"][FC.HM_NET] == FC._pow2(len(ports))
    at = sum(2 * w for _, w, _ in ports[:j])
    got = np.frombuffer(body[at:at + 2 * ports[j][1]], dtype="<u2")
    want = np.asarray(s.port(ports[j][0], 0, 1), dtype=np.uint16).reshape(-1)
    assert (got[:len(want)] == want).all() and not got[len(want):].any() and ports[j][1] % 64 == 0
    assert_public_only(text, prover, verifier)


@pytest.mark.slow
def test_shared_rows_are_tables_and_refs(tmp_path):
    """A shared-row file commits each port's distinct rows once (its frame-v3 tree over them) and gives every instance its refs:
    a population whose instances 2 and 3 repeat 0 and 1 has two-row tables and refs 0, 1, 0, 1, and its output rows one row per
    instance (refs 0, 1, 2, 3), their equal values apart."""
    s = _set("rope-head/d64/neox-bf16", 4, tmp_path)
    text = FC.compose(FC.lowering_for_set(s), s.subcircuit)
    meta = json.loads(text.split("\n", 2)[1][5:])
    ports = [(nm, w) for nm, w, _ in meta["ports"]]
    pick, j = [0, 1, 0, 1], meta["input_ports"]
    rows = {nm: np.asarray(s.port(nm, 0, 4), dtype=np.uint16).reshape(4, -1)[pick] for nm, _ in ports[:j]}
    outs = {p.name: np.asarray(s.port(p.name, 0, 4), dtype=np.uint16).reshape(4, -1)[pick] for p in s.subcircuit.outputs}
    head = FC.write(text, s, 0, 4, tmp_path / "inst.bin", tmp_path / "pub.bin", b"k" * 32, share_rows=True, rows=rows, outs=outs)
    assert head["shared_rows"] == {nm: 2 if k < j else 4 for k, (nm, _) in enumerate(ports)}, "an output row is its instance's own"
    assert_public_only(text, *((tmp_path / f).read_bytes() for f in ("inst.bin", "pub.bin")))
    prover, verifier = ((tmp_path / f).read_bytes().split(b"\n", 1)[1] for f in ("inst.bin", "pub.bin"))
    n_bcs = 2 * j + 4 * (len(ports) - j)
    refs = np.frombuffer(verifier[n_bcs * FC.BC_BYTES:][:16 * len(ports)], dtype="<u4").reshape(4, len(ports))
    assert refs.tolist() == [[r] * j + [i] * (len(ports) - j) for i, r in enumerate(pick)]
    # the tables: port-major rows (the input rows' distinct rows, every instance's output row)
    at = 0
    for k, (nm, w) in enumerate(ports):
        table = np.frombuffer(prover[at:at + 2 * head["shared_rows"][nm] * w], dtype="<u2").reshape(-1, w)
        assert (table[:, :rows[nm].shape[1]] == rows[nm][:2]).all() if k < j else (table[:, :outs[nm].shape[1]] == outs[nm]).all()
        at += table.nbytes
    per = FC.write(text, s, 0, 4, None, tmp_path / "per.bin", b"k" * 32, rows=rows, outs=outs)
    assert "shared_rows" not in per and per["frame_v3"]["roots"] != head["frame_v3"]["roots"]


@pytest.mark.slow
def test_shared_rows_as_given_keep_equal_rows_apart(tmp_path):
    """Tables and refs as given (a value-independent layout): an equal row stays a separate table row with its own salt, the
    refs and the committer's salts are written as given."""
    s = _set("rope-head/d64/neox-bf16", 4, tmp_path)
    text = FC.compose(FC.lowering_for_set(s), s.subcircuit)
    meta = json.loads(text.split("\n", 2)[1][5:])
    names = [nm for nm, _, _ in meta["ports"]]
    base = {nm: np.asarray(s.port(nm, 0, 4), dtype=np.uint16).reshape(4, -1) for nm in ("x", "cs")}
    tables = {"x": base["x"][[0, 1, 0]], "cs": base["cs"][[0, 1]]}
    refs = {"x": [0, 1, 2, 1], "cs": [0, 1, 0, 1]}
    outs = {p.name: np.asarray(s.port(p.name, 0, 4), dtype=np.uint16).reshape(4, -1)[[0, 1, 0, 1]] for p in s.subcircuit.outputs}
    counts = {"x": 3, "cs": 2, **{nm: 4 for nm in outs}}
    salts = {nm: [bytes([k + 1, r]) * (FC.SALT_BYTES // 2) for r in range(counts[nm])] for k, nm in enumerate(names)}
    head = FC.write(text, s, 0, 4, tmp_path / "inst.bin", tmp_path / "pub.bin", share_rows=True, outs=outs, tables=tables, refs=refs,
                    salts=salts)
    assert head["shared_rows"] == counts, "the output rows are the instances' own: equal outputs stay apart"
    assert_public_only(text, *((tmp_path / f).read_bytes() for f in ("inst.bin", "pub.bin")))
    prover, verifier = ((tmp_path / f).read_bytes().split(b"\n", 1)[1] for f in ("inst.bin", "pub.bin"))
    n_rows = sum(counts.values())
    got = np.frombuffer(verifier[n_rows * FC.BC_BYTES:][:16 * len(names)], dtype="<u4").reshape(4, len(names))
    assert got.tolist() == [[refs[nm][i] if nm in refs else i for nm in names] for i in range(4)]
    rows_len = sum(2 * t.size for t in tables.values()) + sum(2 * 4 * w for nm, w, _ in meta["ports"] if nm in outs)
    row = 0
    for nm in names:
        for r in range(counts[nm]):
            assert prover[rows_len + row * FC.SALT_BYTES:][:FC.SALT_BYTES] == salts[nm][r]
            row += 1
    o_start = sum(counts[nm] for nm in names[:meta["input_ports"]])
    out0, out2 = (verifier[(o_start + r) * FC.BC_BYTES:][:FC.BC_BYTES] for r in (0, 2))
    assert out0 != out2, "equal outputs, separate rows and salts: different leaves"
    x_start = sum(counts[nm] for nm in names[:names.index("x")])
    x0, x2 = (verifier[(x_start + r) * FC.BC_BYTES:][:FC.BC_BYTES] for r in (0, 2))
    assert x0 != x2, "equal rows, different salts: different leaves"
    with pytest.raises(ValueError, match="one ref per instance"):
        FC.write(text, s, 0, 4, None, tmp_path / "bad.bin", share_rows=True, outs=outs, tables=tables, refs={**refs, "x": [0, 1, 3, 1]})
    reg = {"cs": {"value": "w", "positions": [4, 2], "paths": ["00" * 64, "11" * 64]}}
    shared = dict(share_rows=True, outs=outs, tables=tables, refs=refs)
    assert FC.write(text, s, 0, 4, None, tmp_path / "reg.bin", salts=salts, registered=reg, **shared)["registered"] == reg
    assert json.loads((tmp_path / "reg.bin").read_bytes().split(b"\n", 1)[0])["registered"] == reg
    with pytest.raises(ValueError, match="one position and path per table row"):
        FC.write(text, s, 0, 4, None, tmp_path / "bad.bin", salts=salts, registered={"cs": {**reg["cs"], "paths": ["00" * 64]}}, **shared)
    with pytest.raises(ValueError, match="registered salts"):
        FC.write(text, s, 0, 4, None, tmp_path / "bad.bin", registered=reg, **shared)
    out = next(iter(outs))
    reg_out = {**reg, out: {"value": "o", "positions": [3, 1, 0, 2], "paths": ["22" * 64] * 4}}
    head = FC.write(text, s, 0, 4, None, tmp_path / "reg-out.bin", salts=salts, registered=reg_out, **shared)
    assert head["registered"] == reg_out, "an output row registers with one position and path per instance"
    with pytest.raises(ValueError, match=f"registered port {out}: a value, and one position and path per table row"):
        FC.write(text, s, 0, 4, None, tmp_path / "bad.bin", salts=salts, registered={out: {**reg_out[out], "positions": [0, 1]}},
                 **shared)
    with pytest.raises(ValueError, match="some rows' tables"):
        FC.write(text, s, 0, 4, None, tmp_path / "bad.bin", salts=salts, registered={"nope": reg["cs"]}, **shared)


def test_registered_reads_refuse_a_circuit_naming_a_port_twice():
    """A registered read names its port by name, so with registered reads the writer refuses a circuit naming a port twice,
    as the Lean verifier does (`Registered.namesOnce`); `test_lean_registered_bit_rows.py` holds the two to one case."""
    from types import SimpleNamespace

    meta = {"ports": [["x", 4, 0], ["x", 4, 0]], "input_ports": 2, "out_ports": []}
    tables = {"x": np.zeros((1, 4), np.uint16)}
    with pytest.raises(ValueError, match="the circuit names port x twice, and a registered read names one port"):
        FC.write("", SimpleNamespace(subcircuit=None), 0, 1, None, None, meta=meta, outs={}, share_rows=True, tables=tables,
                 refs={"x": [0]}, salts={}, registered={"x": {"value": "v", "positions": [0], "paths": ["00" * 64]}})


@pytest.mark.slow
def test_the_header_carries_an_rfc8785_identifier(tmp_path):
    """The header's identifier (PROTOCOL.md §16.10) is both files' as given; anything but a JSON object on which canonical
    JSON is RFC 8785 is refused."""
    s = _set("rope-head/d64/neox-bf16", 4, tmp_path)
    text = FC.compose(FC.lowering_for_set(s), s.subcircuit)
    ident = {"scheme": "test/identifier/v0", "s": 7, "U": [3, 1, 2], "nested": {"b": None}}
    assert FC.write(text, s, 0, 4, tmp_path / "inst.bin", tmp_path / "pub.bin", b"k" * 32, identifier=ident)["identifier"] == ident
    for f in ("inst.bin", "pub.bin"):
        assert json.loads((tmp_path / f).read_bytes().split(b"\n", 1)[0])["identifier"] == ident
    assert "identifier" not in FC.write(text, s, 0, 4, None, tmp_path / "plain.bin", b"k" * 32)
    for bad, match in (([1], "not a JSON object"), ({"s": 1.5}, "a float"), ({"s": 2 ** 53}, r"exceeds 2\^53"),
                       ({"s": "café"}, "not ASCII")):
        with pytest.raises(ValueError, match=match):
            FC.write(text, s, 0, 4, None, tmp_path / "bad.bin", b"k" * 32, identifier=bad)


PUBLIC_CS = {"cs": {"seed": "rope/positions", "derivation": "verity/rope/neox-bf16/cos-sin/v1"}}


def test_public_leaves_split_from_the_rows_and_lie_in_whole_words():
    """A public input port's leaves leave the units' row leaf map (a zero leaf there) for the public one, indexed among the
    public ports' words; the in regions are the words holding them as aligned power-of-two blocks, and a row leaf in such a
    word is refused."""
    owner, li, lp = FC._split_leaves([[0, 1, 2, 3, -1], [2, 3, 0, 1, -1]], [("x", 2), ("cs", 2)], {"cs": {}})
    assert owner == [(False, 0), (False, 1), (True, 0), (True, 1)]
    assert li == [[0, 1, -1, -1, -1], [-1, -1, 0, 1, -1]] and lp == [[-1, -1, 0, 1, -1], [0, 1, -1, -1, -1]]
    # 16 leaves of 16 bits in 2 words: the public ones are word 1's eight leaves in every unit
    li, lp = [list(range(8)) + [-1] * 8], [[-1] * 8 + list(range(8))]
    assert FC.public_blocks((4, 2, [16] * 16), li, lp) == [(5, 0)]
    with pytest.raises(ValueError, match="leaf 0 reads a row word inside word 4, which holds a public word"):
        FC.public_blocks((4, 2, [16] * 16), li, [[-1] * 7 + list(range(9))])
    with pytest.raises(ValueError, match="no unit reads a public word"):
        FC.public_blocks((4, 2, [16] * 16), li, [[-1] * 16])
    # a row port of one word, then public ports of 1 + 2 + 2 words (Pearl-C4's row unit: x, then E, F_A, F_B), and a gap
    li, lp = [list(range(8)) + [-1] * 48], [[-1] * 8 + list(range(40)) + [-1] * 8]
    assert FC.public_blocks((0, 7, [16] * 56), li, lp) == [(1, 0), (2, 1), (4, 1)]
    assert FC.public_blocks((0, 7, [16] * 56), li, [[-1] * 8 + list(range(8)) + [-1] * 32 + list(range(8, 16))]) == [(1, 0), (6, 0)]
    assert FC.aligned_blocks([8, *range(9, 137)]) == [(8, 3), (16, 4), (32, 5), (64, 6), (128, 3), (136, 0)]
    assert FC.aligned_blocks([0, 1, 2, 3]) == [(0, 2)] and FC.aligned_blocks([3, 5]) == [(3, 0), (5, 0)]


@pytest.mark.slow
def test_public_input_ports_carry_the_verifiers_recomputation(tmp_path):
    """A public input port (PROTOCOL.md §16.10) is no row: META names it with its seed and derivation, its leaves are zero
    leaves of ``leaves_in`` and public words of ``leaves_public`` in whole words of the leaf group, and the unit range is not
    packed (the in region reads it). Both instance files end with its words; the verifier's own ``--public-inputs`` file holds
    the same words under the same ports. A port naming no recomputation, no input port, or one sharing a word with a row's
    leaf (RoPE's default lowering) is refused."""
    from verity_flock.templates import rope_head as R

    s = _set("rope-head/d64/neox-bf16", 4, tmp_path)
    with pytest.raises(ValueError, match="lay the public ports out as whole words"):
        FC.compose(FC.lowering_for_set(s), s.subcircuit, public_inputs=PUBLIC_CS)
    low = IL.lowering(R.definition(s.subcircuit), R.UNIT + "/x4", group=4)
    for bad, why in (({"cs": {"seed": "", "derivation": "d"}}, "names its recomputation"), ({"cs": {"seed": "s"}}, "names its recomputation"),
                     ({"q": {"seed": "s", "derivation": "d"}}, "not input ports")):
        with pytest.raises(ValueError, match=why):
            FC.compose(low, s.subcircuit, public_inputs=bad)
    FC.stage(low, s, 0, 4, tmp_path / "st", public_inputs=PUBLIC_CS)
    text = (tmp_path / "st" / "circuit.txt").read_text()
    meta = json.loads(text.split("\n", 2)[1][5:])
    assert [p[0] for p in meta["ports"]] == ["x", "out"] and meta["input_ports"] == 1
    assert meta["public_inputs"] == [{"port": "cs", "words": 64, **PUBLIC_CS["cs"]}]
    assert not next(r for r in meta["ranges"] if r["net"] == "unit").get("packed")
    assert all(a == -1 for li, lp in zip(meta["leaves_in"], meta["leaves_public"]) for a, b in zip(li, lp) if b >= 0)
    assert sorted(b for lp in meta["leaves_public"] for b in lp if b >= 0) == list(range(64)), "each cs word read once"
    prover, verifier = ((tmp_path / "st" / f).read_bytes() for f in ("inst-4.bin", "pub-4.bin"))
    (hp, bp), (hv, bv) = (b.split(b"\n", 1) for b in (prover, verifier))
    assert json.loads(hp)["public_inputs"] == json.loads(hv)["public_inputs"] == meta["public_inputs"]
    cs = np.asarray(s.port("cs", 0, 4), dtype="<u2").reshape(4, 64).tobytes()
    assert bp.endswith(cs) and bv.endswith(cs) and len(bv) == 4 * 2 * FC.BC_BYTES + len(cs), "the b || c, then the public words"
    own_head, own = (tmp_path / "st" / "public-inputs-4.bin").read_bytes().split(b"\n", 1)
    assert json.loads(own_head) == {"format": FC.PUBLIC_INPUTS_FORMAT, "instances": 4, "ports": meta["public_inputs"]} and own == cs


def test_program_digests_are_keyed_by_descriptor_id(tmp_path):
    """META ``program_digests`` names the template by core's descriptor id (the canonical partition's and the verifier's), not
    its display id: RMSNorm's EPS is its f64 hex."""
    from verity.ml.circuits.codec import descriptor_id

    s = _set("rmsnorm-triton/n2048-eps1e-05/bf16", 1, tmp_path)
    got = FC.program_digests(s.subcircuit, FC.set_statics(s))
    assert list(got) == [descriptor_id(s.subcircuit.definition(**FC.set_statics(s)))]
    assert '"f64":"0x1.4f8b588e368f1p-17"' in next(iter(got))
    assert list(got.values()) == list(s.subcircuit.program_digests([FC.set_statics(s)]).values())


class _Slices(str):
    """A circuit text that records the longest slice taken of it."""

    widest = 0

    def __getitem__(self, k):
        out = str.__getitem__(self, k)
        if isinstance(k, slice):
            _Slices.widest = max(_Slices.widest, len(out))
        return out


@pytest.mark.parametrize("chunk", [1, 7, 64, 1 << 24])
def test_the_class_digest_hashes_the_body_a_chunk_at_a_time(chunk, monkeypatch):
    """`class_digest` is SHA-512 of the file with META's bindings removed, its body read :data:`circuit._HASH_CHUNK`
    characters at a time and never sliced whole (a GLM-5.3 norm's unit section is 8.6 GB)."""
    import hashlib

    monkeypatch.setattr(FC, "_HASH_CHUNK", chunk)
    meta = {"format": "f", "k_log": 20, "partition": {"rule": "r"}, "program_sha512": "0" * 128, "program_digests": {"a": "b"}}
    head = "flock-circuit u\nMETA "
    body = "\nCIRCUIT unit 400\n" + "".join(f"2 {k} {k + 1} 1 {7 * k}\n" for k in range(400))
    text = _Slices(head + json.dumps(meta, sort_keys=True, separators=(",", ":")) + body)
    kept = {k: v for k, v in meta.items() if k not in FC.BINDING_KEYS}
    _Slices.widest = 0
    assert FC.class_digest(text) == hashlib.sha512((head + json.dumps(kept, sort_keys=True, separators=(",", ":")) + body).encode()).hexdigest()
    assert _Slices.widest <= max(chunk, len(text) - len(body))
    assert FC.digest(text) == hashlib.sha512(str(text).encode()).hexdigest()


def test_leaf_scheme_reserves_what_hm96_needs():
    import hashlib

    from verity.ml.commitments import hm96

    t = FC.LEAF_SCHEME
    assert (t["id"], t["salt_bytes"]) == (hm96.SHA512.name, hm96.SHA512.salt_bytes)
    assert t["key_sha512"] == hashlib.sha512(hm96.SHA512.default_key).hexdigest(), "the key is the scheme's pinned constant"


def _body(path, start: str) -> str:
    src = path.read_text()
    body = src[src.index(start):]
    return body[:body.index("\n\n")]


def test_the_leaf_scheme_is_the_lean_verifiers_and_the_provers():
    """The Lean verifier refuses a META ``leaf_scheme`` other than its pinned object (``Flock/Tags.lean``'s
    ``leafSchemeHm96Os``, the default statement's; V*'s ``compose`` writes it too), and the prover one other than its
    ``leaf_scheme()``: each names the salts, the key and the column digest in ``LEAF_SCHEME``'s words."""
    import re
    from pathlib import Path

    flock, service = Path(__file__).resolve().parents[1], Path(__file__).resolve().parents[3] / "core" / "service"
    words = {k: v for k, v in FC.LEAF_SCHEME.items() if k in ("id", "inner", "salt_source", "key")}
    lean = r'\("(\w+)", (?:s )?"([^"]*)"\)'
    rust = _body(flock / "live" / "src" / "circuit.rs", "pub fn leaf_scheme()")
    copies = {"Flock.Tags.leafSchemeHm96Os": re.findall(lean, _body(service / "flock" / "Flock" / "Tags.lean",
                                                                    "def leafSchemeHm96Os")),
              "VStar.leafScheme": re.findall(lean, _body(service / "flock" / "VStar" / "Compose.lean", "def leafScheme")),
              "circuit.rs leaf_scheme": [("id", re.search(r'\{ \("([^"]+)", 192\) \}', rust)[1]),
                                         *re.findall(r'"(\w+)": "([^"]*)"', rust)]}
    for name, pairs in copies.items():
        assert {k: v for k, v in pairs if k in words} == words, name


def test_the_zk_hooks_name_the_provers_randomness():
    """Under ``--zk`` the identity's ``hooks``, which the statement digest binds, name where the pads and every table's
    draws come from: the Lean verifier's default statements (``Flock.Zk.hooks`` under ``Tags.osMasks``) and the prover's
    ``zk_identity`` and session tables write the same words."""
    import re
    from pathlib import Path

    flock, service = Path(__file__).resolve().parents[1], Path(__file__).resolve().parents[3] / "core" / "service"
    lean = _body(service / "flock" / "Flock" / "Zk.lean", "def hooks")
    os_words = dict(re.findall(r'let (\w+) := if os then "([^"]*)" else', lean))
    assert set(os_words) == {"pads", "draws"}, os_words
    want = {k: re.search(rf'\("{k}", s s!"([^"]*)"\)', lean)[1].replace("{pads}", os_words["pads"])
            .replace("{draws}", os_words["draws"]) for k in ("pads", "tables")}
    rust = (flock / "live" / "src" / "bin" / "flock-circuit.rs").read_text()
    tables = re.search(r'id\["hooks"\]\["tables"\] = json!\("((?:[^"\\]|\\\n)*)"\)', rust)[1]
    assert want == {"pads": re.search(r'"pads": "([^"]*)"', rust)[1], "tables": re.sub(r"\\\n\s*", "", tables)}


def test_returned_outputs_need_a_cut_tail():
    class U:
        cut, tail_rets = [], [0]

    class Low:
        units, name = U(), "attention"

    with pytest.raises(NotImplementedError, match="no verifier-evaluated form"):
        FC.compose(Low(), None)


@pytest.mark.parametrize("prim", sorted(p for p in TL.MUFU if p not in ("MufuEx2Ftz_v1", "Fa2InvSum_v1")))
def test_mufu_pieces_match_the_ir(prim):
    from verity.ml.definitions.numerics import rms_relation
    from verity.ml.definitions.registry import prims as P

    t = rms_relation.tables("cuda")
    tabs = {"rsq": np.asarray(t.rsq, dtype=np.uint32), "sqrt": np.asarray(t.sqrt, dtype=np.uint32), "rcp": np.asarray(t.rcp, dtype=np.uint32)}
    rng = np.random.default_rng(3)
    xs = np.array([0, 0x80000000, 1, 0x80000001, 0x007FFFFF, 0x00800000, 0x7F7FFFFF, 0x7F800000, 0xFF800000, 0x7FC00000, 0x3F800000,
                   0x7E800000, 0x7E800001, 0xFE800001, 0x40000000] + list(rng.integers(0, 2**32, 400, dtype=np.uint64)), dtype=np.uint64)
    n = len(xs)
    pre, post = TL.MUFU[prim]
    c1 = Circuit()
    table, idx, ctx = pre(c1, [c1.input("x", 32)])
    c1.output("idx", idx)
    for i, w in enumerate(ctx):
        c1.output(f"c{i}", w)
    o1 = {k: IL.unslice_words(b, n) for k, b in c1.evaluate({"x": IL.slice_words(xs, 32)}, n)[0].items()}
    v = tabs[table][o1["idx"].astype(np.int64)].astype(np.uint64) & np.uint64((1 << TL.VALUE_BITS) - 1)
    c2 = Circuit()
    cw = [c2.input(f"c{i}", 32) for i in range(len(ctx))]
    c2.output("y", post(c2, cw, c2.input("v", TL.VALUE_BITS)))
    ins = {**{f"c{i}": IL.slice_words(o1[f"c{i}"], 32) for i in range(len(ctx))}, "v": IL.slice_words(v, TL.VALUE_BITS)}
    y = IL.unslice_words(c2.evaluate(ins, n)[0]["y"], n)
    nan = lambda u: (u >> 23) & 0xFF == 0xFF and u & 0x7FFFFF  # noqa: E731
    ref = [getattr(P, prim[:-3]).evaluate(int(a)) for a in xs]
    bad = [(hex(int(a)), hex(int(b)), hex(c)) for a, b, c in zip(xs, y, ref) if int(b) != c and not (nan(int(b)) and nan(c))]
    assert not bad, bad[:5]


def test_lookup_rows_match_the_rust_layout():
    # 128 input rows, decoders of 13 + 10 bits, 31 * 2^10 products, output word, constant (lookup.rs)
    assert FC.lookup_rows(23, 13, 31) == -(-(128 + FC._decode_rows(13) + FC._decode_rows(10) + 31 * 1024) // 128) * 128 + 129
    assert FC._decode_rows(2) == 4 and FC._decode_rows(3) == 8 + 4


def _f32_samples(n, seed):
    rng = np.random.default_rng(seed)
    special = [0, 0x80000000, 1, 0x80000001, 0x007FFFFF, 0x00800000, 0x7F7FFFFF, 0x7F800000, 0xFF800000, 0x7FC00000, 0x3F800000,
               0xBF800000, 0x4E800000, 0xCE800000, 0x42FE0000, 0xC3000000, 0x19DB6BEB, 0x9D60AFCB]
    return special + [int(x) for x in rng.integers(0, 2**32, n, dtype=np.uint64)] + \
        [int(np.float32(x).view(np.uint32)) for x in rng.normal(0, 30, n)]


def _run(C, inputs, t):
    got, _ = C.evaluate({k: [sum(((v >> i) & 1) << j for j, v in enumerate(vs)) for i in range(bits)] for k, (vs, bits) in inputs.items()}, t)
    return {k: [sum(((b >> j) & 1) << i for i, b in enumerate(v)) for j in range(t)] for k, v in got.items()}


@pytest.mark.parametrize("name,arity", [("F32AddFtz_v1", 2), ("F32MulFtz_v1", 2), ("F32FmaFtz_v1", 3), ("F32FmaSubFtz_v1", 3),
                                        ("F32Max_v1", 2), ("F2fpBf16_v1", 1), ("F32SubFtz_v1", 2), ("GuardNegInfZero_v1", 1)])
def test_attention_tail_ops_match_the_ir(name, arity):
    """Exact on encodings, but for a NaN result's payload (canonical here, as for F32Add; every NaN meets a bf16 cast or a
    NaN-blind consumer before it leaves the tail)."""
    from verity.ml.definitions.registry import prims as P

    args = [_f32_samples(150, 11 + i) for i in range(arity)]
    t = len(args[0])
    C = Circuit()
    C.output("y", list(TL.OPS[name](C, *[C.input(f"x{i}", 32) for i in range(arity)])))
    got = _run(C, {f"x{i}": (args[i], 32) for i in range(arity)}, t)["y"]
    nan = lambda u: (u & 0x7F800000) == 0x7F800000 and u & 0x7FFFFF
    for k in range(t):
        want = getattr(P, name[:-3]).evaluate(*[a[k] for a in args]) & 0xFFFFFFFF
        assert got[k] == want or (nan(got[k]) and nan(want)), (name, [hex(a[k]) for a in args], hex(got[k]), hex(want))


#: constant divisors: 2048 and 2^-126 (an exact normal reciprocal), GLM-5.3's 6144, DeepSeek-V4's 7168, 6144 forged x 4, 3, eps,
#: and those without a normal reciprocal: 2^127, a subnormal, -6144's sign, zero, infinity, NaN
DIVISORS = [0x45000000, 0x00800000, 0x45C00000, 0x45E00000, 0x46C00000, 0x40400000, 0x3727C5AC, 0x7F000000, 0x00000001,
            0xC5C00000, 0, 0x7F800000, 0x7FC00000]


@pytest.mark.parametrize("k", DIVISORS, ids=hex)
def test_f32_div_by_a_constant_matches_the_ir(k):
    """A row tail's ``F32Div_v1`` by its constant (RMSNorm's ``/ N``) is the IR's division on every encoding but a NaN result's
    payload, through the exact reciprocal or, for N = 6144, the IEEE divider: quotients rounding into the subnormals included."""
    from verity.ml.definitions.registry import prims as P

    xs = _f32_samples(200, 13) + [0x00C00000, 0x01400000, 0x00800001, 0x7F7FFFFF, 0x80FFFFFF, 0x0A2AAAAB]
    C = Circuit()
    C.output("y", list(TL.OPS["F32Div_v1"](C, C.input("x", 32), const_bits(k, 32))))
    got = _run(C, {"x": (xs, 32)}, len(xs))["y"]
    nan = lambda u: (u & 0x7F800000) == 0x7F800000 and u & 0x7FFFFF  # noqa: E731
    want = [P.F32Div.evaluate(x, k) & 0xFFFFFFFF for x in xs]
    assert [hex(x) for x, g, w in zip(xs, got, want) if g != w and not (nan(g) and nan(w))] == []


def test_mufu_ex2_matches_the_ir():
    """The circuit's EX2 equals the IR's, and every 0 < |x| < 2^-23 gives T[0] = 1.0 (a reduction shift of 64 or more is 0),
    including |x| < 2^-63, where a shift taken mod 64 read ex2(2^-64) as 2.0."""
    from verity.ml.definitions.numerics import fa2_relation
    from verity.ml.definitions.registry import prims as P

    ex2 = np.asarray(fa2_relation.tables().ex2, dtype=np.uint32)
    tiny = [s << 31 | e << 23 | f for s in (0, 1) for e in (1, 2, 40, 63, 64, 65, 103, 104) for f in (0, 1, 0x7FFFFF)]
    xs = _f32_samples(300, 5) + tiny
    t = len(xs)
    pre, post = TL.MUFU["MufuEx2Ftz_v1"]
    c1 = Circuit()
    table, idx, ctx = pre(c1, [c1.input("x", 32)])
    assert table == "ex2"
    c1.output("idx", idx)
    for i, w in enumerate(ctx):
        c1.output(f"c{i}", w)
    o1 = _run(c1, {"x": (xs, 32)}, t)
    c2 = Circuit()
    c2.output("y", post(c2, [c2.input(f"c{i}", 32) for i in range(len(ctx))], c2.input("v", TL.VALUE_BITS)))
    ins = {f"c{i}": (o1[f"c{i}"], 32) for i in range(len(ctx))}
    ins["v"] = ([int(ex2[j]) & 0x7FFFFFFF for j in o1["idx"]], TL.VALUE_BITS)
    got = _run(c2, ins, t)["y"]
    def below(x):
        """0 < |x| < 2^-23, not flushed."""
        return (x >> 23) & 0xFF != 0 and (x & 0x7FFFFFFF) < 0x34000000

    want = [0x3F800000 if below(x) else P.MufuEx2Ftz.evaluate(x) for x in xs]
    assert [hex(x) for x, g, w in zip(xs, got, want) if g != w] == []


def test_fa2_inv_sum_matches_the_ir():
    from verity.ml.definitions.numerics import fa2_relation
    from verity.ml.definitions.registry import prims as P

    rcp = np.asarray(fa2_relation.tables().rcp, dtype=np.uint32)
    xs = _f32_samples(300, 9)
    t = len(xs)
    pre, post = TL.MUFU["Fa2InvSum_v1"]
    c1 = Circuit()
    table, idx, ctx = pre(c1, [c1.input("x", 32)])
    assert table == "rcp"
    c1.output("idx", idx)
    for i, w in enumerate(ctx):
        c1.output(f"c{i}", w)
    o1 = _run(c1, {"x": (xs, 32)}, t)
    c2 = Circuit()
    c2.output("y", post(c2, [c2.input(f"c{i}", 32) for i in range(len(ctx))], c2.input("v", TL.VALUE_BITS)))
    ins = {f"c{i}": (o1[f"c{i}"], 32) for i in range(len(ctx))}
    ins["v"] = ([int(rcp[j]) & 0x7FFFFFFF for j in o1["idx"]], TL.VALUE_BITS)
    got = _run(c2, ins, t)["y"]
    assert [hex(x) for x, g in zip(xs, got) if g != P.Fa2InvSum.evaluate(x)] == []
