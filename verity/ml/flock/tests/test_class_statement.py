"""A unit class of any partition as a ``verity/flock-circuit`` statement (`verity_flock.class_statement`): on circuit-check's tiny
Serve_v1 under `Q_word`, two table-free classes and the embedding gather compose with one row port per input source, their
laid-out unit rows agree with the reference on synthetic lanes, a source read by several instances is one shared row (the
gather's table, broadcast across its Call's members), and, when ``FLOCK_CIRCUIT`` names a proving build of ``flock-circuit``, a
loopback session accepts each, on the coins ``FC_COINS`` names."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("verity.ml.circuits.units")
pytest.importorskip("verity_vllm.program.registry")
from circuit_check import targets as TG  # noqa: E402
from verity.ml.circuits.defs import bind  # noqa: E402
from verity.ml.circuits.program import Program  # noqa: E402

from verity_flock import circuit as FC  # noqa: E402
from verity_flock import class_statement as CS  # noqa: E402
from verity_flock import partition_units as PU  # noqa: E402


@pytest.fixture(scope="module")
def units():
    TG.load_registries()
    from verity_vllm.program.registry import b1

    CS.cache_slot_circuits()
    return PU.partition_units(Program(bind(b1.Serve, C=TG.TINY, LP=3, STEPS=2)), "q-word")


@pytest.mark.slow
def test_table_free_classes_compose_and_prove(units, tmp_path: Path):
    key = PU.derive(PU.SOURCE, PU.DOMAIN, {"program": "test", "seed": 0})
    shared = CS.broadcast_params(units)
    assert shared["GatherBf16x64_v1"] == ({0}, 32)                         # the table, read by all 32 row members of the Call
    picked, gather = [], None
    for s, cl in sorted(units.classes().items(), key=lambda kv: len(kv[1].representative.gates)):
        cc = PU.lower_class(units.activation(cl.definition), cl.representative)
        if cc.circuit is None or cc.circuit.lookups:
            continue
        if cl.definition.id == "GatherBf16x64_v1":
            gather = gather or (s, cl)
        elif len(picked) < 2:
            picked.append((s, cl))
    binary = os.environ.get("FLOCK_CIRCUIT")
    for i, (s, cl) in enumerate([*picked, gather]):
        lanes = CS.class_lanes(units, cl, key, 8, shared)
        d = tmp_path / s[:16]
        st = CS.stage(units, "q-word", s, cl, lanes, d, hashlib.sha256(b"test").digest())
        text = (d / "circuit.txt").read_text()
        assert st["n"] == 8 and st["k_log"] <= FC.K_MAX and st["circuit_sha512"] == FC.digest(text)
        assert '"rule":"verity/partition/v1"' in text                      # the Q_word partition's own digest, not a provisional cover
        assert list(st["shared_rows"]) == [f"p{j}" for j in range(len(st["ports"]))] + ["out"] and st["shared_rows"]["out"] == 8
        assert all(w % CS.BLOCK_WORDS == 0 for w in st["ports"])          # the output row: one per instance, never shared
        if (s, cl) == gather:
            assert st["shared_rows"] == {"p0": 1, "p1": 8, "out": 8}        # one table row shared by every instance; a row per index
            (template,) = json.loads(text.split("\n", 2)[1][5:])["program_digests"]
            assert template.startswith("unit-class/")                       # a primitive has no Program, so the class names itself
        if binary and Path(binary).exists():
            live = CS.prove(binary, d, st["n"], 7690 + i)
            assert live["accepted"] is True and live["proof_bytes"]
            assert live["coins"] == ("live-os" if os.environ.get("FC_COINS") == "os" else "os-seed-prf")


def _gemm_fp():
    spec = importlib.util.spec_from_file_location("gemm_fp", Path(__file__).parents[1] / "pod" / "gemm_fp.py")
    gemm_fp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gemm_fp)
    return gemm_fp


def _fp_class(dtype: str, K: int):
    CS.cache_slot_circuits()
    units = PU.partition_units(_gemm_fp().program(dtype, K, 4), "q-word")
    ((s, cl),) = units.classes().items()
    act, uc = units.activation(cl.definition), cl.representative
    return units, s, cl, act, uc, CS.sources(act, uc, CS.owners_of(units, cl.definition.id))


def test_a_partitions_binding_and_owners_are_its_own_when_a_freed_partitions_address_is_reused(monkeypatch):
    """`partition_binding` and `owners_of` cache per `PartitionUnits` object, so a partition freed while cached leaves no entry
    that a later partition allocated at its address could read (test_wired_links' flat staging once bound the freed wired
    partition's digest that way). Two programs alternate, each partition freed after its binding, with every address made
    one (`id` constant in `class_statement`), so a cache keyed by address reads the other program's entry every time."""
    import gc
    import weakref

    from verity.ml.circuits import partition_object as PO

    monkeypatch.setattr(CS, "id", lambda o: 0, raising=False)
    gemm_fp = _gemm_fp()
    programs = [gemm_fp.program("e4m3", K, 4) for K in (128, 256)]
    bindings = len(CS._BINDINGS)
    for i in range(16):
        units = PU.partition_units(programs[i % 2], "q-word")
        assert CS.partition_binding(units, "q-word")[0] == PO.digest(PO.build(units.program, units.query))
        ((_s, cl),) = units.classes().items()
        assert CS.owners_of(units, cl.definition.id)
        assert len(CS._BINDINGS) == bindings + 1 and len(CS._OWNERS) == 1
        gone = weakref.ref(units)
        del units
        gc.collect()
        assert gone() is None and len(CS._BINDINGS) == bindings and not CS._OWNERS


@pytest.mark.parametrize("dtype", ["e4m3", "nvf4", "mxf4"])
def test_packed_rows_are_the_packed_tensor_bytes_and_the_unit_is_the_definition(dtype, tmp_path: Path):
    """FLOCK_PACK_WORDS's layout on an sm_120 FP coordinate: four FP4 or two FP8 elements a u16 leaf, so a row's little-endian
    bytes are its tensor's packed bytes (two FP4 a byte, low nibble first; one FP8 or scale a byte), every leaf bit before the
    block padding is read, the unit has the default layout's gates, and on circuit-check's vectors it equals the IR evaluator."""
    from circuit_check import checks as CK
    from circuit_check import vectors as V
    from verity.ml.circuits.evaluation.reference import signature_of

    units, s, cl, act, uc, groups = _fp_class(dtype, 128)
    plain = CS.class_lowering(act, uc, "unit-class/plain", groups)
    low = CS.class_lowering(act, uc, "unit-class/packed", groups, packed=True)
    same = ("ands", "xors", "nots", "outputs", "lookups")
    assert {k: low.layout.counts.get(k) for k in same} == {k: plain.layout.counts.get(k) for k in same}
    assert low.layout.counts["rows"] < plain.layout.counts["rows"]
    widths = [[uc.in_widths[t] for t in g] for g in groups]
    assert all(sum(w) % 16 == 0 and CS._slots(w, True)[1] == sum(w) // 16 for w in widths)
    assert low.definition.port_words == tuple(CS._pad(sum(w) // 16, CS.BLOCK_WORDS) for w in widths)

    ins, want, _ids = CS.class_lanes(units, cl, PU.derive(PU.SOURCE, PU.DOMAIN, {"program": "test", "seed": 0}), 4, CS.broadcast_params(units))
    st = CS.stage(units, "q-word", s, cl, (ins, want), tmp_path, hashlib.sha256(b"test").digest(), packed=True)
    assert st["packed"] is True and st["ports"] == list(low.definition.port_words)
    _rows, _outs, per_port = CS.pack(uc, ins, want, low.definition.w_in, low.definition.w_out, groups, low.definition.port_words, True)
    for i, (g, w) in enumerate(zip(groups, widths)):
        for lane in range(4):
            x = [ins[t][lane] for t in g]
            packed = bytes(a | b << 4 for a, b in zip(x[::2], x[1::2])) if w[0] == 4 else bytes(x)
            row = per_port[f"p{i}"][lane].astype("<u2").tobytes()
            assert row == packed + bytes(len(row) - len(packed))

    fn = cl.definition
    params, _ret = signature_of(fn)
    arrays = V.vectors(params, 48, 20260926)
    ref, _raised = CK.reference_outputs(fn, params, arrays)
    leaves = V.flat(arrays)
    lanes = [leaves[:, -1 - k] for k in uc.inputs]
    got = {}
    for name, lw, pk in (("plain", plain, False), ("packed", low, True)):
        rows, _o, _p = CS.pack(uc, lanes, [[0] * len(arrays[0])], lw.definition.w_in, lw.definition.w_out, groups, lw.definition.port_words, pk)
        out, _cut, ok = lw.evaluate(rows.astype(np.uint64))
        got[name] = (out.astype(np.uint64), int(ok))
    assert np.array_equal(got["plain"][0], got["packed"][0]) and got["plain"][1] == got["packed"][1]
    out, ok = got["packed"]
    (w_out,) = uc.out_widths
    word = sum(out[:, h] << np.uint64(16 * h) for h in range(CS._words(w_out)))
    checked = [j for j, r in enumerate(ref) if r is not None and ok >> j & 1]
    assert len(checked) >= 24 and all(int(word[j]) == ref[j][0] for j in checked)


#: a row's bits at K: the codes, then the scales (E4M3: none; NVFP4: one byte per 16 codes; MXFP4: one per 32)
ROW_BITS = {"e4m3": lambda K: 8 * K, "nvf4": lambda K: 4 * K + K // 2, "mxf4": lambda K: 4 * K + K // 4}


def _lanes(units, cl, n: int = 4):
    return CS.class_lanes(units, cl, PU.derive(PU.SOURCE, PU.DOMAIN, {"program": "test", "seed": 0}), n, CS.broadcast_params(units))


def _decode(uc, rows: np.ndarray, groups, port_words, packed: bool) -> list:
    """Every input's value from u16 rows in a layout (`_slots`; every FP input is narrower than a leaf)."""
    vals, base = [None] * len(uc.inputs), 0
    for g, wp in zip(groups, port_words):
        at, _used = CS._slots([uc.in_widths[t] for t in g], packed)
        for t, (leaf, bit) in zip(g, at):
            vals[t] = (rows[:, base + leaf].astype(np.uint64) >> np.uint64(bit)) & np.uint64((1 << uc.in_widths[t]) - 1)
        base += wp
    return vals


@pytest.mark.parametrize("dtype", ["e4m3", "nvf4", "mxf4"])
def test_core_rows_are_core_bit_rows_and_decode_as_the_default_layout(dtype):
    """FLOCK_CORE_ROWS's layout on an sm_120 FP coordinate at K=128: x with its scales one row and the W column with its scales
    another, each row's bytes core's bit row of the lane's codes and scales (here zero-padded, which the stager refuses). The unit
    has the default layout's gates; on rows of every bit random it computes the default lowering's outputs and ok flags on the
    same values re-laid, from either layout to the other; on circuit-check's vectors it equals the IR evaluator."""
    from circuit_check import checks as CK
    from circuit_check import vectors as V
    from verity.ml.commitments.rowleaf import bit_row
    from verity.ml.circuits.evaluation.reference import signature_of

    units, _s, cl, act, uc, groups = _fp_class(dtype, 128)
    merged, joins, bits = CS.element_rows(act, uc, groups)
    assert bits == [ROW_BITS[dtype](128)] * 2
    assert joins == ([(0,), (1,)] if dtype == "e4m3" else [(0, 2), (1, 3)])
    assert merged == [[t for i in j for t in groups[i]] for j in joins]
    plain = CS.class_lowering(act, uc, "unit-class/plain", groups)
    low = CS.class_lowering(act, uc, "unit-class/core", merged, packed=True)
    same = ("ands", "xors", "nots", "outputs", "lookups")
    assert {k: low.layout.counts.get(k) for k in same} == {k: plain.layout.counts.get(k) for k in same}

    ins, want, _ids = _lanes(units, cl)
    _rows, _outs, per_port = CS.pack(uc, ins, want, low.definition.w_in, low.definition.w_out, merged, low.definition.port_words, True)
    code_bits = 8 if dtype == "e4m3" else 4
    for i, (g, n_bits) in enumerate(zip(merged, bits)):
        for lane in range(4):
            v = [int(ins[t][lane]) for t in g]
            row = per_port[f"p{i}"][lane].astype("<u2").tobytes()
            want = bit_row((v[:128], code_bits), (v[128:], 8))
            assert want[1] == n_bits and row[:len(want[0])] == want[0] and not any(row[len(want[0]):])

    layouts = {"plain": (plain, groups, False), "core": (low, merged, True)}
    rng = np.random.default_rng(20261001)
    zero_out = [[0] * 64 for _ in uc.outputs]
    for src, dst in (("core", "plain"), ("plain", "core")):
        lw, grp, pk = layouts[src]
        rows = rng.integers(0, 1 << 16, (64, lw.definition.w_in), dtype=np.uint64)
        out_a, _cut, ok_a = lw.evaluate(rows)
        vals = _decode(uc, rows, grp, lw.definition.port_words, pk)
        lw, grp, pk = layouts[dst]
        re_laid, _o, _p = CS.pack(uc, vals, zero_out, lw.definition.w_in, lw.definition.w_out, grp, lw.definition.port_words, pk)
        out_b, _cut, ok_b = lw.evaluate(re_laid.astype(np.uint64))
        assert np.array_equal(out_a.astype(np.uint64), out_b.astype(np.uint64)) and int(ok_a) == int(ok_b)

    fn = cl.definition
    params, _ret = signature_of(fn)
    arrays = V.vectors(params, 48, 20261001)
    ref, _raised = CK.reference_outputs(fn, params, arrays)
    leaves = V.flat(arrays)
    lanes = [leaves[:, -1 - k] for k in uc.inputs]
    rows, _o, _p = CS.pack(uc, lanes, [[0] * len(arrays[0])], low.definition.w_in, low.definition.w_out, merged, low.definition.port_words, True)
    out, _cut, ok = low.evaluate(rows.astype(np.uint64))
    out, ok = out.astype(np.uint64), int(ok)
    (w_out,) = uc.out_widths
    word = sum(out[:, h] << np.uint64(16 * h) for h in range(CS._words(w_out)))
    checked = [j for j, r in enumerate(ref) if r is not None and ok >> j & 1]
    assert len(checked) >= 24 and all(int(word[j]) == ref[j][0] for j in checked)


def test_a_class_of_whole_activations_names_its_definition_as_the_circuits_template(monkeypatch, tmp_path: Path):
    """An unwired class whose unit is one whole activation of its flat Definition (an FP GEMM coordinate) names that Definition
    in META ``program_digests``, by core's descriptor id with its Program's digest, so that a verifier given the program finds
    each unit an activation of the circuit's template (f05d's refusal named ``unit-class/…``). The name is a binding: the class
    digest is the one the class has naming itself, and only the circuit's pin moves."""
    from verity.ml.circuits.codec import descriptor_id, program_digest

    units, s, cl, act, uc, _groups = _fp_class("e4m3", 128)
    assert uc.computing == act.graph.n
    lanes, key = _lanes(units, cl), hashlib.sha256(b"test").digest()
    meta = lambda d: json.loads((tmp_path / d / "circuit.txt").read_text().split("\n", 2)[1][5:])  # noqa: E731
    named = CS.stage(units, "q-word", s, cl, lanes, tmp_path / "named", key)
    assert meta("named")["program_digests"] == {descriptor_id(act.fn): program_digest(Program(act.fn))}
    shims = CS._shims
    monkeypatch.setattr(CS, "_shims", lambda name, content, ports=1, definition=None: shims(name, content, ports))
    own = CS.stage(units, "q-word", s, cl, lanes, tmp_path / "own", key)
    (template,) = meta("own")["program_digests"]
    assert template.startswith("unit-class/") and meta("own")["program_sha512"] == meta("named")["program_sha512"]
    assert own["class_sha512"] == named["class_sha512"] and own["circuit_sha512"] != named["circuit_sha512"]


def test_a_class_staged_without_shared_rows_opens_every_instances_own_rows_under_the_same_circuit(tmp_path: Path):
    """``share_rows=False``: the same circuit, and public and prover files without ``shared_rows`` (the form Lean's
    ``ProvedScope.sharedRowsOk`` covers without registered reads), every instance's rows its own, one per port, so the prover's
    file is the shared one's with every table as long as the instances; a wired class keeps its link rows shared."""
    units, s, cl, _act, _uc, _groups = _fp_class("e4m3", 128)
    lanes, key = _lanes(units, cl), hashlib.sha256(b"test").digest()
    shared = CS.stage(units, "q-word", s, cl, lanes, tmp_path / "shared", key)
    own = CS.stage(units, "q-word", s, cl, lanes, tmp_path / "own", key, share_rows=False)
    head = lambda d, f: json.loads((tmp_path / d / f"{f}-{own['n']}.bin").read_bytes().split(b"\n", 1)[0])  # noqa: E731
    assert own["circuit_sha512"] == shared["circuit_sha512"] and own["shared_rows"] is None
    assert "shared_rows" in head("shared", "pub") and "shared_rows" in head("shared", "inst")
    assert "shared_rows" not in head("own", "pub") and "shared_rows" not in head("own", "inst")
    meta = json.loads((tmp_path / "own" / "circuit.txt").read_text().split("\n", 2)[1][5:])
    n, words = own["n"], sum(w for _p, w, _c in meta["ports"])
    body = (tmp_path / "own" / f"inst-{n}.bin").read_bytes().split(b"\n", 1)[1]
    assert len(body) == n * (2 * words + (192 + 128) * len(meta["ports"]))
    with pytest.raises(ValueError, match="share_rows"):
        CS.stage(units, "q-word", s, cl, lanes, tmp_path / "wired", key, wiring=SimpleNamespace(), share_rows=False)


def test_core_rows_stage_and_each_row_recomputes_from_the_prover_file_with_core_alone(tmp_path: Path):
    """E4M3 K=128, the smallest whole-block FP row, staged under FLOCK_CORE_ROWS: META names each row's schema and length in bits
    (the output row's, §16.13, its 64 words') and no single ``row_schema``, its row constants are core's prefixes, and its circuit
    is neither other frame's. From the prover's file, every committed row's elements give core's bit-row digest, ``b || c`` with its salt, the frame leaf under the
    hiding schema, and the header's root under the domain its binding names, using ``verity.ml.commitments`` only."""
    from verity.ml.commitments import hm96
    from verity.ml.commitments.frame_v3 import BitRowLeaf, FrameV3, Port
    from verity.ml.commitments.indexed import RangeIndexedDomain
    from verity.ml.commitments.merkle import CommitmentDomain
    from verity.ml.commitments.rowleaf import bit_row, row_leaf, sha512_row_v2_digest, sha512_row_v2_prefix

    units, s, cl, _act, _uc, _groups = _fp_class("e4m3", 128)
    lanes, key = _lanes(units, cl), hashlib.sha256(b"test").digest()
    st = CS.stage(units, "q-word", s, cl, lanes, tmp_path / "core", key, core_rows=True)
    schemas = [["hm96-sha512/row/v2", 1024]] * 3
    assert st["packed"] is True and st["row_schemas"] == schemas and st["ports"] == [64, 64]
    meta = json.loads((tmp_path / "core" / "circuit.txt").read_text().split("\n", 2)[1][5:])
    assert "row_schema" not in meta and meta["row_schemas"] == schemas
    assert [r["prefix"] for r in meta["rows"]] == [sha512_row_v2_prefix(1024).hex()] * 3
    assert [p.bits for p in FC._meta_ports(meta)] == [1024] * 3
    others = {CS.stage(units, "q-word", s, cl, lanes, tmp_path / fr, key, packed=fr == "packed")["circuit_sha512"] for fr in ("plain", "packed")}
    assert len(others) == 2 and st["circuit_sha512"] not in others

    inst = hm96.SHA512
    head_line, body = (tmp_path / "core" / f"inst-{st['n']}.bin").read_bytes().split(b"\n", 1)
    head = json.loads(head_line)
    fv, counts = head["frame_v3"], head["shared_rows"]
    names = [p for p, _w, _c in meta["ports"]]
    at, tables, salts = 0, {}, {}
    for p, w, _c in meta["ports"]:
        tables[p] = [body[at + 2 * w * r:at + 2 * w * (r + 1)] for r in range(counts[p])]
        at += 2 * w * counts[p]
    for p in names:
        salts[p] = [body[at + 192 * r:at + 192 * (r + 1)] for r in range(counts[p])]
        at += 192 * counts[p]
    for (schema, n_bits), p in zip(schemas, names):
        assert fv["schemas"][p] == schema
        dom = CommitmentDomain(bytes.fromhex(fv["bindings"][p]), fv["owner"], RangeIndexedDomain(0, counts[p]), hash="sha512")
        assert dom.domain_id.hex() == fv["domain_ids"][p]
        leaves = []
        for r, row in enumerate(tables[p]):
            assert bit_row((list(row), 8)) == (row, n_bits)
            bc = inst.commit_string(inst.default_key, sha512_row_v2_digest(row, n_bits), salts[p][r])
            assert body[at + 128 * r:at + 128 * (r + 1)] == bc
            leaves.append(row_leaf(dom, r, inst.tree_leaf(inst.default_key, bc), schema))
        at += 128 * counts[p]
        frame = FrameV3({p: Port(dom, BitRowLeaf(n_bits))})
        root = frame.root(p, leaves)
        assert root.hex() == fv["roots"][p]
        assert all(frame.verify_path(p, root, len(leaves), r, leaves[r], frame.path(p, leaves, r)) for r in range(len(leaves)))


def test_core_rows_stage_partial_blocks_and_refuse_tiles_and_sources_that_are_not_whole_parameters(monkeypatch, tmp_path: Path):
    """A bit row of any length stages, its leaves ``ceil(bits / 16)`` words: NVF4 K=256's rows are 1152 bits, 144 bytes, one
    compression and then a last block that holds 16 row bytes and SHA-512's padding; the output row (§16.13) is a bit row of its
    64 words. NVF4 K=128's (576 bits, 72 bytes) fit the padding block with no compression before it, so the output row's is the
    statement's one (with none, the circuit's ``sha512x3`` range, the pin's, would hold no slot). A source that is not one whole
    parameter, a row length that is not its leaves', and tiling are refused."""
    from verity_flock import sha512_circuit as S

    units, s, cl, act, uc, groups = _fp_class("nvf4", 256)
    st = CS.stage(units, "q-word", s, cl, _lanes(units, cl), tmp_path / "k256", b"k" * 32, core_rows=True)
    assert st["row_schemas"] == [["hm96-sha512/row/v2", 1152]] * 2 + [["hm96-sha512/row/v2", 1024]] and st["ports"] == [72, 72]
    meta = json.loads((tmp_path / "k256" / "circuit.txt").read_text().split("\n", 2)[1][5:])
    assert [c for _p, _w, c in meta["ports"]] == [1, 1, 1]
    assert [r["pad"] for r in meta["rows"]] == [S.pad_tail(128 + 144).hex()] * 2 + [S.pad_tail(128 + 128).hex()]
    units, s, cl, act, uc, groups = _fp_class("nvf4", 128)
    CS.stage(units, "q-word", s, cl, _lanes(units, cl), tmp_path / "k128", b"k" * 32, core_rows=True)
    meta = json.loads((tmp_path / "k128" / "circuit.txt").read_text().split("\n", 2)[1][5:])
    assert [c for _p, _w, c in meta["ports"]] == [0, 0, 1]
    merged, _joins, bits = CS.element_rows(act, uc, groups)
    for bad in ([groups[0][:64], groups[0][64:], *groups[1:]], [groups[0][::-1], *groups[1:]], [groups[0] + groups[1], *groups[2:]]):
        with pytest.raises(ValueError, match="not one whole parameter"):
            CS.element_rows(act, uc, bad)
    low = CS.class_lowering(act, uc, "unit-class/padded", merged, packed=True)
    sub, _iset = CS._shims("padded", b"padded", 2)
    with pytest.raises(ValueError, match="a row of 576 bits is 36 u16 leaves, not 64"):
        FC.compose(low, sub, port_bits=bits)
    meta = {"ports": [["p0", 64, 1]], "row_schemas": [["hm96-sha512/row-e4m3/v1", 1024]]}
    with pytest.raises(ValueError, match="hm96-sha512/row/v2"):
        FC._meta_ports(meta)
    meta["row_schemas"] = [["hm96-sha512/row/v2", 1000]]
    with pytest.raises(ValueError, match="1000 bits is 63 u16 leaves, not 64"):
        FC._meta_ports(meta)
    meta["row_schemas"] = [["hm96-sha512/row/v2", 1 << 64]]
    with pytest.raises(ValueError, match="a u64 number of bits"):
        FC._meta_ports(meta)
    assert [p.bits for p in FC._meta_ports({**meta, "row_schemas": [["hm96-sha512/row/v2", 1009]]})] == [1009]
    with pytest.raises(ValueError, match="a row of 1009 bits has a bit set past its length"):
        FC._commit_strings([FC._port("p0", 64, 1009)], {"p0": np.full((1, 64), 1 << 1, np.uint16)}, {"p0": [bytes(192)]})
    monkeypatch.setattr(CS, "CORE_ROWS", True)
    with pytest.raises(ValueError, match="FLOCK_CORE_ROWS"):
        CS.stage_tiled(units, "q-word", s, cl, (), tmp_path, b"", 4, 4, 1)


def _frame_prefixes(dtype: str, K: int) -> tuple[dict, list]:
    """At (dtype, K), from the coordinate Definition's parameters through the stager's own layout and core's prefixes: the default
    and packed frames' port prefixes (one ``sha512/row/v1`` row per parameter, its words as `class_lowering` pads them) and the
    new frame's, ``(prefix, bits)`` per row."""
    gemm_fp = _gemm_fp()
    fn = gemm_fp.coordinate(dtype, K)
    sig = dict(fn.signature()[0])
    params = [[t.elem.w] * t.n for t in sig.values()]
    old = {fr: [FC.Port(f"p{i}", CS._pad(CS._slots(w, fr == "packed")[1], CS.BLOCK_WORDS)).layout().prefix for i, w in enumerate(params)]
           for fr in ("default", "packed")}
    bits = [sum(sig[nm].n * sig[nm].elem.w for nm in row) for row in CS.ELEMENT_ROWS[fn.id.split("{")[0]]]
    ports = [FC._port(f"p{i}", -(-b // 16), b) for i, b in enumerate(bits)]
    return old, [(p.layout().prefix, p.bits) for p in ports]


def _gemm_fp():
    import importlib.util

    spec = importlib.util.spec_from_file_location("gemm_fp", Path(__file__).parents[1] / "pod" / "gemm_fp.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_core_row_prefixes_name_the_row_length_only_and_no_old_frames_prefix():
    """Every port's prefix in the new, default and packed frames, at the test shapes, K = 64 ... 16384 (the 12 cells among them):
    each new prefix is core's ``P(l)`` for the row's own bits (codes then scales), so a prefix names a length and nothing else, x
    and W rows of one shape share it (by design: the frame leaf binds the port's domain), and none equals an old-frame prefix.
    Every shape stages, MXF4 K=2048's partial last block among them: rows that fit SHA-512's padding block (at most 111 bytes)
    have no compression before it, and the output row's (§16.13) is the statement's. The derivation agrees with the stager's
    own lowering and `element_rows` at K=128."""
    from verity.ml.commitments.rowleaf import sha512_row_v2_prefix

    shapes = [(d, k) for d in ROW_BITS for k in (64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384)]
    new: dict[bytes, tuple] = {}
    old: set[bytes] = set()
    for dtype, K in shapes:
        frames, rows = _frame_prefixes(dtype, K)
        for fr, prefixes in frames.items():
            assert prefixes[0] == prefixes[1], (dtype, K, fr, "the old frames' x and W rows share a prefix")
            old.update(prefixes)
        assert [b for _p, b in rows] == [ROW_BITS[dtype](K)] * 2 and [p for p, _b in rows] == [sha512_row_v2_prefix(ROW_BITS[dtype](K))] * 2
        for prefix, b in rows:
            assert new.setdefault(prefix, b) == b, (dtype, K, b, new[prefix])
    assert not set(new) & old
    assert len(new) == len({ROW_BITS[d](k) for d, k in shapes})

    for dtype in ROW_BITS:
        _units, _s, _cl, act, uc, groups = _fp_class(dtype, 128)
        frames, _rows = _frame_prefixes(dtype, 128)
        for fr in ("default", "packed"):
            low = CS.class_lowering(act, uc, f"unit-class/{fr}", groups, fr == "packed")
            assert frames[fr] == [FC.Port(f"p{i}", w).layout().prefix for i, w in enumerate(low.definition.port_words)]
        assert CS.element_rows(act, uc, groups)[2] == [ROW_BITS[dtype](128)] * 2


def test_the_slot_circuits_are_built_only_when_a_shape_is_not_in_the_stage_cache(monkeypatch, tmp_path: Path):
    import verity

    src = {}                                                       # the packages' sources, as the stage key's code digest reads them
    for name, pkg in (("verity_flock", Path(CS.__file__).parent), ("verity", Path(verity.__file__).parent)):
        src[name] = tmp_path / "src" / name
        for f in pkg.rglob("*.py"):
            (src[name] / f.relative_to(pkg)).parent.mkdir(parents=True, exist_ok=True)
            (src[name] / f.relative_to(pkg)).write_bytes(f.read_bytes())
    find_spec = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: SimpleNamespace(submodule_search_locations=[str(src[name])])
                        if name in src else find_spec(name, *a))

    def fresh(s):
        (CS._CTX["out"] / s[:16]).mkdir(parents=True, exist_ok=True)
        (CS._CTX["out"] / s[:16] / "circuit.txt").write_text(s)
        return {"shape": s, "stage": {"n": 16}}

    built = []
    monkeypatch.setenv("FLOCK_STAGE_CACHE", str(tmp_path / "cache"))
    monkeypatch.delenv("RESEARCH_RUN_DIR", raising=False)
    monkeypatch.setitem(CS._CTX, "source_digest", None)
    monkeypatch.setattr(CS, "cache_slot_circuits", lambda: built.append(1))
    monkeypatch.setattr(CS, "_stage_job_fresh", fresh)

    def cached(name):
        CS._CTX.pop("source_digest")                                   # main() reads the code's digest once per process
        assert CS.main(["--model", "tiny", "--partition", "q-word", "--stage-only", "--sample", "2", "--out", str(tmp_path / name)]) == 0
        recs = [json.loads(ln) for ln in (tmp_path / name / "staged.jsonl").read_text().splitlines()]
        assert len(recs) == 2 and len({"stage_cached" in r for r in recs}) == 1
        return "stage_cached" in recs[0]

    assert not cached("first") and built == [1]
    assert cached("again") and built == [1]                            # a cache hit composes nothing
    sha = src["verity"] / "ml" / "commitments" / "gates" / "sha512.py"
    assert "CARRY_EVERY = 5" in sha.read_text()
    sha.write_text(sha.read_text().replace("CARRY_EVERY = 5", "CARRY_EVERY = 4"))
    assert not cached("sha512-parameter") and built == [1, 1]          # a SHA-512 parameter is in the key
    cs = src["verity_flock"] / "class_statement.py"
    cs.write_text(cs.read_text() + "\n# another version\n")
    assert not cached("code") and built == [1, 1, 1]                   # and so is the code's version


def test_the_stage_keys_code_digest_is_one_tree_over_a_namespaces_portions(monkeypatch, tmp_path: Path):
    """Walking `verity`'s portions in path order made the digest depend on sys.path order and on a cwd directory that joined the
    namespace; it is the same over any order, and moves with an edit or a portion that joins."""
    for rel in ("upper/verity/ml/__init__.py", "upper/verity/ml/gemm.py", "lower/verity/ir/__init__.py", "lower/verity/ir/defs.py",
                "cwd/verity/primitives/circuits/src/verity/ir/defs.py"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(f"# {rel}\n")
    find_spec = importlib.util.find_spec

    def digest(*portions: str) -> str:
        locs = [str(tmp_path / p / "verity") for p in portions]
        monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: SimpleNamespace(submodule_search_locations=locs)
                            if name == "verity" else find_spec(name, *a))
        return CS._source_digest()

    a = digest("upper", "lower")
    assert digest("lower", "upper") == a
    assert digest("cwd", "upper", "lower") != a                         # a directory joining from the cwd is a miss, never a hit
    (tmp_path / "lower" / "verity" / "ir" / "defs.py").write_text("# edited\n")
    assert digest("upper", "lower") != a


def test_selftest_jobs_runs_the_case_processes_at_once_and_keeps_their_records_in_case_order(monkeypatch, tmp_path: Path):
    cases = ["gpu_paths_agree", "gpu_proofs_match_cpu", "verify_ahead_matches_serial"]
    mib = dict(zip(cases, (4000, 4500, 4700)))
    st: dict = {"lock": threading.Lock()}

    def run(cmd, **_kw):
        case = cmd[cmd.index("--only") + 1]
        on = "caller" if threading.current_thread() is st["caller"] else "pool"
        with st["lock"]:
            st["argv"][case] = cmd
            st["running"].add(case)
        if on == "pool":
            st["started"].wait()                                   # every case's process is running before any returns,
            st["sampled"].wait()                                   # with its device memory sampled meanwhile,
            if case != cases[-1]:
                st["returned"][cases[cases.index(case) + 1]].wait()   # and they return last case first
        with st["lock"]:
            st["running"].discard(case)
            st["finished"].append(case)
        st["returned"][case].set()
        return CS.subprocess.CompletedProcess(cmd, 0, stdout=f"NEG\t{json.dumps({'case': case, 'pass': True, 'on': on})}\n", stderr="")

    def device():
        with st["lock"]:
            now = {c: mib[c] for c in st["running"]}
        if len(now) == len(cases):
            st["sampled"].set()
        return now

    monkeypatch.delenv("RESEARCH_RUN_DIR", raising=False)
    monkeypatch.setattr(CS.subprocess, "run", run)
    monkeypatch.setattr(CS, "_case_device_mib", device)
    monkeypatch.setattr(CS, "SELFTEST_SAMPLE_S", 0.001)
    monkeypatch.setattr(CS, "cache_slot_circuits", lambda: None)
    monkeypatch.setattr(CS, "_stage_job", lambda s: {"shape": s, "stage": {"n": 16}, "table_reads": {}})
    args = ["--model", "tiny", "--partition", "q-word", "--sample", "1", "--selftest-only", "--selftest", "1", "--selftest-gpu",
            "--selftest-cases", ",".join(cases), "--selftest-binary", "flock-circuit-seeded", "--selftest-case-mib", "6144"]
    argvs, sel = [], {}
    # jobs, the GPU's free MiB, where the cases ran, the order they returned in
    for jobs, free, on, order in (("1", 96000, "caller", cases), ("3", 96000, "pool", cases[::-1]),
                                  ("3", 3 * 6144 - 1, "caller", cases), ("3", None, "caller", cases)):
        monkeypatch.setattr(CS, "_gpu_free_mib", lambda free=free: free)
        st.update(argv={}, running=set(), finished=[], started=threading.Barrier(len(cases)), sampled=threading.Event(),
                  returned={c: threading.Event() for c in cases}, caller=threading.current_thread())
        (tmp_path / "selftests.jsonl").unlink(missing_ok=True)
        assert CS.main([*args, "--selftest-jobs", jobs, "--out", str(tmp_path)]) == 0
        (rec,) = [json.loads(ln) for ln in (tmp_path / "selftests.jsonl").read_text().splitlines()]
        s = sel[jobs, free] = rec["selftest"]
        assert st["finished"] == order
        assert s["all_pass"] is True and s["subset"] == cases and s["did_not_finish"] == []
        assert s["gpu_cases"] == [{"case": c, "pass": True, "on": on} for c in cases]
        assert [(p["case"], p["exit_code"], p["timed_out"], type(p["wall_s"])) for p in s["per_case"]] == [(c, 0, False, float) for c in cases]
        argvs.append(st["argv"])
    assert all(a == argvs[0] for a in argvs)                       # each case its own process, with the same binary and arguments
    assert all(cmd[0] == "flock-circuit-seeded" and "--gpu" in cmd and cmd[-2:] == ["--only", c] for c, cmd in argvs[0].items())
    assert "run" not in sel["1", 96000]
    at_once = sel["3", 96000]
    assert {k: at_once["run"][k] for k in ("jobs_asked", "jobs", "free_mib", "case_bound_mib", "sum_peak_mib")} == \
        {"jobs_asked": 3, "jobs": 3, "free_mib": 96000, "case_bound_mib": 6144, "sum_peak_mib": 13200}
    assert "fallback" not in at_once["run"] and [p["device_peak_mib"] for p in at_once["per_case"]] == [4000, 4500, 4700]
    assert sel["3", 18431]["run"]["jobs"] == 1 and sel["3", 18431]["run"]["fallback"] == "3 x 6144 MiB is more than the 18431 MiB free"
    assert sel["3", None]["run"]["jobs"] == 1 and sel["3", None]["run"]["fallback"] == "the GPU's free memory can't be read"


def test_a_case_process_that_ends_without_its_verdict_is_recorded_as_not_finished(monkeypatch, tmp_path: Path):
    def run(cmd, **_kw):
        if cmd[-1] == "gpu_proofs_match_cpu":
            raise CS.subprocess.TimeoutExpired(cmd, 3600)
        return CS.subprocess.CompletedProcess(cmd, -9, stdout="", stderr="CUDA_ERROR_OUT_OF_MEMORY: out of memory\n")

    monkeypatch.setattr(CS.subprocess, "run", run)
    hung = CS.prove("flock-circuit", tmp_path, 4, 0, selftest="gpu_proofs_match_cpu", gpu=True)
    assert (hung["all_pass"], hung["finished"], hung["timed_out"], hung["exit_code"]) == (False, False, True, None)
    oom = CS.prove("flock-circuit", tmp_path, 4, 0, selftest="gpu_paths_agree", gpu=True)
    assert (oom["all_pass"], oom["finished"], oom["timed_out"], oom["exit_code"]) == (False, False, False, -9)
    assert "out of memory" in oom["stderr_tail"] and "gpu_cases" not in oom


def test_selftest_gpu_runs_the_selftest_on_the_gpu(monkeypatch, tmp_path: Path):
    seen = []

    def run(cmd, **_kw):
        seen.append(cmd)
        return CS.subprocess.CompletedProcess(cmd, 0, stdout='NEG\t{"case": "gpu_paths_agree", "pass": true}\n', stderr="")

    monkeypatch.setattr(CS.subprocess, "run", run)
    gpu = CS.prove("flock-circuit", tmp_path, 4, 0, selftest="gpu_paths_agree", gpu=True)
    assert gpu["all_pass"] is True and gpu["gpu_cases"] == [{"case": "gpu_paths_agree", "pass": True}]
    assert CS.prove("flock-circuit", tmp_path, 4, 0, selftest="honest")["all_pass"] is True
    assert "--gpu" in seen[0] and seen[0][-2:] == ["--only", "gpu_paths_agree"]
    assert "--gpu" not in seen[1]


def test_each_selftest_case_keeps_its_stderr_beside_the_class_records(monkeypatch, tmp_path: Path):
    def run(cmd, **_kw):
        case = cmd[cmd.index("--only") + 1] if "--only" in cmd else None
        return CS.subprocess.CompletedProcess(cmd, 0, stdout=f'NEG\t{{"case": "{case}", "pass": false}}\n',
                                              stderr=f"thread 'main' panicked: {case}: out of memory\n")

    monkeypatch.setattr(CS.subprocess, "run", run)
    rec = CS.prove("flock-circuit", tmp_path, 16, 0, selftest="gpu_proofs_match_cpu", gpu=True)
    assert rec["finished"] is True and rec["all_pass"] is False and "stderr_tail" not in rec   # a verdict: no tail in the record
    CS.prove("flock-circuit", tmp_path, 16, 0, selftest=True)
    assert (tmp_path / "selftest-gpu_proofs_match_cpu.err").read_text() == "thread 'main' panicked: gpu_proofs_match_cpu: out of memory\n"
    assert (tmp_path / "selftest-all.err").read_text() == "thread 'main' panicked: None: out of memory\n"


def test_a_verifier_pod_serves_every_point_of_a_gpu_session_on_the_same_processes(monkeypatch, tmp_path: Path):
    """Each prover process sends its session k to verifier process k mod S from k = 0, so a pod bounded for P points serves
    P provers one after another: each process exits after exactly the sessions the P provers send it."""
    argvs = []

    def popen(argv, stdout, **_kw):
        stdout.write(f"SERVING 0.0.0.0:{9000 + len(argvs)}\n")
        stdout.flush()
        argvs.append(argv)
        return SimpleNamespace(pid=0, kill=lambda: None, wait=lambda: None)

    monkeypatch.setattr(CS.subprocess, "Popen", popen)
    monkeypatch.setenv("FC_VERIFY_SERVERS", "11")
    monkeypatch.setenv("FC_VERIFY_AHEAD", "10")
    monkeypatch.setenv("FC_SERVE_HOST", "10.0.0.7")
    (tmp_path / "circuit.txt").write_text("circuit\n")
    sessions, points = 25, 8
    e = CS.serve_remote("flock-circuit", tmp_path, 2048, tmp_path / "verifiers", sessions, points=points)
    bound = [int(a[a.index("--sessions") + 1]) for a in argvs]
    sent = [sum(1 for _ in range(points) for k in range(sessions) if k % len(argvs) == i) for i in range(len(argvs))]
    assert len(argvs) == 11 and bound == sent and sum(bound) == points * sessions
    doc = json.loads((tmp_path / "verifiers" / CS._served_name(FC.digest("circuit\n"), 2048, sessions)).read_text())
    assert doc == e["doc"] and (doc["sessions"], doc["points"], doc["servers"]) == (sessions, points, 11)
    assert doc["addrs"] == [f"10.0.0.7:{9000 + i}" for i in range(11)]


def test_a_live_statement_is_accepted_only_if_every_session_is_and_prove_exits_0(monkeypatch, tmp_path: Path):
    (tmp_path / "circuit.txt").write_text("circuit")
    st: dict = {}

    class Prove:
        def __init__(self, cmd, stdout, stderr):
            if "-o" in cmd:
                Path(cmd[cmd.index("-o") + 1]).write_text("1024\n")
            stdout.write("".join(f"LIVE\t{json.dumps({'run': i, 'warm': i == 0, 'accepted': v, 'prove_total_s': 1.0 + i})}\n"
                                 for i, v in enumerate(st["verdicts"])))
            self.returncode = st["rc"]

        def wait(self):
            return self.returncode

    monkeypatch.delenv("FC_VERIFIERS", raising=False)
    monkeypatch.setattr(CS, "_serve", lambda *_a, **_kw: ([], ["127.0.0.1:1"]))
    monkeypatch.setattr(CS.subprocess, "Popen", Prove)
    cases = [([True] * 4, 0, True, None), ([True] * 3, 0, False, "3 LIVE records, not warm + runs = 4"),
             ([False, True, True, True], 0, False, "sessions [0] not accepted"), ([True, True, False, True], 0, False, "sessions [2] not accepted"),
             ([True] * 4, 1, False, "prove exited 1")]
    for verdicts, rc, accepted, why in cases:
        st.update(verdicts=verdicts, rc=rc)
        live = CS.prove("flock-circuit", tmp_path, 4, 0, warm=1, runs=3)
        assert (live["accepted"], live["why"]) == (accepted, why), (verdicts, rc)
        assert live["timed_runs"] == len(verdicts) - 1


def test_a_live_statement_records_the_coins_its_sessions_ran_on(monkeypatch, tmp_path: Path):
    (tmp_path / "circuit.txt").write_text("circuit")
    seed, os_ = {"verdict": {"coin_seed": {"commitment": "ab"}}}, {"identity": {"coins": "live-os (every round's coins ...)"}}
    tree = {"identity": {"hooks": {"zk": True}}}
    st: dict = {}

    class Prove:
        def __init__(self, cmd, stdout, stderr):
            if "-o" in cmd:
                Path(cmd[cmd.index("-o") + 1]).write_text("1024\n")
            stdout.write("".join(f"LIVE\t{json.dumps({'run': i, 'warm': i == 0, 'accepted': True, **x})}\n" for i, x in enumerate(st["live"])))
            self.returncode = 0

        def wait(self):
            return 0

    monkeypatch.delenv("FC_VERIFIERS", raising=False)
    monkeypatch.setattr(CS, "_serve", lambda *_a, **_kw: ([], ["127.0.0.1:1"]))
    monkeypatch.setattr(CS.subprocess, "Popen", Prove)
    for live, coins in (([seed] * 2, "os-seed-prf"), ([os_] * 2, "live-os"), ([tree] * 2, None), ([seed, os_], None)):
        st["live"] = live
        assert CS.prove("flock-circuit", tmp_path, 4, 0, warm=1, runs=1)["coins"] == coins, live


def test_serve_wait_accepts_only_every_session_served_and_accepted(tmp_path: Path):
    class Exited:
        pid = 0

        def poll(self):
            return 0

        def kill(self):
            pass

        def wait(self):
            return 0

    def served(name: str, verdicts: list[bool]) -> dict:
        d = tmp_path / name
        (d / "sessions").mkdir(parents=True)
        (d / "sessions" / "index.jsonl").write_text("".join(json.dumps({"accepted": v, "verify_s": 0.5}) + "\n" for v in verdicts))
        (d / "addrs.json").write_text("{}")
        return {"file": d / "addrs.json", "srvs": [Exited()], "doc": {"sessions": 3}, "d": d, "t0": 0.0}

    recs = CS.serve_wait([served("all", [True] * 3), served("short", [True] * 2), served("rejected", [True, False, True]),
                          served("none", [])], 1.0)
    assert [(r["served"], r["accepted"]) for r in recs] == [(3, True), (2, False), (3, False), (0, False)]
    assert not any((tmp_path / n / "addrs.json").exists() for n in ("all", "short", "rejected", "none"))
