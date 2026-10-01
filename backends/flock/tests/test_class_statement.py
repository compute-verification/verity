"""A unit class of any partition as a ``verity/flock-circuit`` statement (`verity_flock.class_statement`): on circuit-check's tiny
Serve_v1 under `Q_word`, two table-free classes and the embedding gather compose with one row port per input source, their
laid-out unit rows agree with the reference on synthetic lanes, a source read by several instances is one shared row (the
gather's table, broadcast across its Call's members), and, when ``FLOCK_CIRCUIT`` names a built ``flock-circuit`` binary, a
loopback session accepts each."""
from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("verity.ir.units")
pytest.importorskip("verity_vllm.program.registry")
from circuit_check import targets as TG  # noqa: E402
from verity.ir.defs import bind  # noqa: E402
from verity.ir.program import Program  # noqa: E402

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
        assert len(st["ports"]) == len(st["shared_rows"]) and all(w % CS.BLOCK_WORDS == 0 for w in st["ports"])
        if (s, cl) == gather:
            assert st["shared_rows"] == {"p0": 1, "p1": 8}                  # one table row shared by every instance; a row per index
        if binary and Path(binary).exists():
            live = CS.prove(binary, d, st["n"], 7690 + i)
            assert live["accepted"] is True and live["proof_bytes"]


def _fp_class(dtype: str, K: int):
    import importlib.util

    spec = importlib.util.spec_from_file_location("gemm_fp", Path(__file__).parents[1] / "pod" / "gemm_fp.py")
    gemm_fp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gemm_fp)
    CS.cache_slot_circuits()
    units = PU.partition_units(gemm_fp.program(dtype, K, 4), "q-word")
    ((s, cl),) = units.classes().items()
    act, uc = units.activation(cl.definition), cl.representative
    return units, s, cl, act, uc, CS.sources(act, uc, CS.owners_of(units, cl.definition.id))


@pytest.mark.parametrize("dtype", ["e4m3", "nvf4", "mxf4"])
def test_packed_rows_are_the_packed_tensor_bytes_and_the_unit_is_the_definition(dtype, tmp_path: Path):
    """FLOCK_PACK_WORDS's layout on an sm_120 FP coordinate: four FP4 or two FP8 elements a u16 leaf, so a row's little-endian
    bytes are its tensor's packed bytes (two FP4 a byte, low nibble first; one FP8 or scale a byte), every leaf bit before the
    block padding is read, the unit has the default layout's gates, and on circuit-check's vectors it equals the IR evaluator."""
    from circuit_check import checks as CK
    from circuit_check import vectors as V
    from verity.evaluation.reference import signature_of

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


FORMAT = {"e4m3": "e4m3", "nvf4": "nvfp4", "mxf4": "mxfp4"}


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
def test_core_rows_are_core_element_rows_and_decode_as_the_default_layout(dtype):
    """FLOCK_CORE_ROWS's layout on an sm_120 FP coordinate at K=128: x with its scales one row of role 1 and the W column with its
    scales one of role 2, each row's bytes core's element row of the lane's codes and scales (here zero-padded, which the stager
    refuses). The unit has the default layout's gates; on rows of every bit random it computes the default lowering's outputs and
    ok flags on the same values re-laid, from either layout to the other; on circuit-check's vectors it equals the IR evaluator."""
    from circuit_check import checks as CK
    from circuit_check import vectors as V
    from verity.commitments.rowleaf import ELEMENT_FORMATS, element_row_bytes
    from verity.evaluation.reference import signature_of

    units, _s, cl, act, uc, groups = _fp_class(dtype, 128)
    merged, joins, kinds = CS.element_rows(act, uc, groups)
    fmt = FORMAT[dtype]
    f = ELEMENT_FORMATS[fmt]
    assert kinds == [(fmt, 1, 128), (fmt, 2, 128)]
    assert joins == ([(0,), (1,)] if dtype == "e4m3" else [(0, 2), (1, 3)])
    assert merged == [[t for i in j for t in groups[i]] for j in joins]
    plain = CS.class_lowering(act, uc, "unit-class/plain", groups)
    low = CS.class_lowering(act, uc, "unit-class/core", merged, packed=True)
    same = ("ands", "xors", "nots", "outputs", "lookups")
    assert {k: low.layout.counts.get(k) for k in same} == {k: plain.layout.counts.get(k) for k in same}

    ins, want, _ids = _lanes(units, cl)
    _rows, _outs, per_port = CS.pack(uc, ins, want, low.definition.w_in, low.definition.w_out, merged, low.definition.port_words, True)
    for i, (g, (_fmt, _role, k)) in enumerate(zip(merged, kinds)):
        for lane in range(4):
            v = [ins[t][lane] for t in g]
            row = per_port[f"p{i}"][lane].astype("<u2").tobytes()
            assert row[:f.row_len(k)] == element_row_bytes(fmt, v[:k], v[k:]) and not any(row[f.row_len(k):])

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


def test_core_rows_stage_and_each_row_recomputes_from_the_prover_file_with_core_alone(tmp_path: Path):
    """E4M3 K=128, the smallest whole-block element row, staged under FLOCK_CORE_ROWS: META names each port's row schema, role
    and k and no single ``row_schema``, its row constants are core's prefixes, and its circuit is neither other frame's. From the
    prover's file, every committed row's codes give core's element-row digest, ``b || c`` with its salt, the frame leaf under the
    hiding schema, and the header's root under the domain its binding names, using ``verity.commitments`` only."""
    from verity.commitments import hm96
    from verity.commitments.frame_v3 import ElementRowLeaf, FrameV3, Port
    from verity.commitments.indexed import RangeIndexedDomain
    from verity.commitments.merkle import CommitmentDomain
    from verity.commitments.rowleaf import row_leaf, sha512_element_row_digest, sha512_element_row_prefix

    units, s, cl, _act, _uc, _groups = _fp_class("e4m3", 128)
    lanes, key = _lanes(units, cl), hashlib.sha256(b"test").digest()
    st = CS.stage(units, "q-word", s, cl, lanes, tmp_path / "core", key, core_rows=True)
    schemas = [["hm96-sha512/row-e4m3/v1", 1, 128], ["hm96-sha512/row-e4m3/v1", 2, 128]]
    assert st["packed"] is True and st["row_schemas"] == schemas and st["ports"] == [64, 64]
    meta = json.loads((tmp_path / "core" / "circuit.txt").read_text().split("\n", 2)[1][5:])
    assert "row_schema" not in meta and meta["row_schemas"] == schemas
    assert [r["prefix"] for r in meta["rows"]] == [sha512_element_row_prefix("e4m3", role, 128).hex() for role in (1, 2)]
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
    for (schema, role, k), p in zip(schemas, names):
        assert fv["schemas"][p] == schema
        dom = CommitmentDomain(bytes.fromhex(fv["bindings"][p]), fv["owner"], RangeIndexedDomain(0, counts[p]), hash="sha512")
        assert dom.domain_id.hex() == fv["domain_ids"][p]
        leaves = []
        for r, row in enumerate(tables[p]):
            bc = inst.commit_string(inst.default_key, sha512_element_row_digest("e4m3", list(row), [], role), salts[p][r])
            assert body[at + 128 * r:at + 128 * (r + 1)] == bc
            leaves.append(row_leaf(dom, r, inst.tree_leaf(inst.default_key, bc), schema))
        at += 128 * counts[p]
        frame = FrameV3({p: Port(dom, ElementRowLeaf("e4m3", role, k))})
        root = frame.root(p, leaves)
        assert root.hex() == fv["roots"][p]
        assert all(frame.verify_path(p, root, len(leaves), r, leaves[r], frame.path(p, leaves, r)) for r in range(len(leaves)))


def test_core_rows_refuse_partial_blocks_tiles_and_sources_that_are_not_whole_parameters(monkeypatch, tmp_path: Path):
    units, s, cl, act, uc, groups = _fp_class("nvf4", 128)
    with pytest.raises(ValueError, match="72 bytes, not whole SHA-512 blocks"):
        CS.stage(units, "q-word", s, cl, _lanes(units, cl), tmp_path, b"k" * 32, core_rows=True)
    merged, _joins, kinds = CS.element_rows(act, uc, groups)
    CS.whole_blocks("cells", [("nvfp4", 1, 2048), ("mxfp4", 2, 4096), ("e4m3", 1, 128), ("nvfp4", 2, 16384)])
    for kind, size in ((("mxfp4", 1, 2048), 1088), (("nvfp4", 1, 1024), 576), (("e4m3", 2, 64), 64)):
        with pytest.raises(ValueError, match=f"{size} bytes, not whole"):
            CS.whole_blocks("cells", [kind])
    for bad in ([groups[0][:64], groups[0][64:], *groups[1:]], [groups[0][::-1], *groups[1:]], [groups[0] + groups[1], *groups[2:]]):
        with pytest.raises(ValueError, match="not one whole parameter"):
            CS.element_rows(act, uc, bad)
    low = CS.class_lowering(act, uc, "unit-class/padded", merged, packed=True)
    sub, _iset = CS._shims("padded", b"padded", 2)
    with pytest.raises(ValueError, match="72 bytes, not 128"):
        FC.compose(low, sub, port_rows=kinds)
    monkeypatch.setattr(CS, "CORE_ROWS", True)
    with pytest.raises(ValueError, match="FLOCK_CORE_ROWS"):
        CS.stage_tiled(units, "q-word", s, cl, (), tmp_path, b"", 4, 4, 1)


def _frame_prefixes(dtype: str, K: int) -> tuple[dict, list | None]:
    """At (dtype, K), from the coordinate Definition's parameters through the stager's own layout and core's prefixes: the default
    and packed frames' port prefixes (one ``sha512/row/v1`` row per parameter, its words as `class_lowering` pads them) and the
    new frame's, ``(prefix, (format, role, k))`` per row, or None where `whole_blocks` refuses the shape."""
    gemm_fp = _gemm_fp()
    fn = gemm_fp.coordinate(dtype, K)
    params = [[t.elem.w] * t.n for _n, t in fn.signature()[0]]
    old = {fr: [FC.Port(f"p{i}", CS._pad(CS._slots(w, fr == "packed")[1], CS.BLOCK_WORDS)).layout().prefix for i, w in enumerate(params)]
           for fr in ("default", "packed")}
    fmt, rows = CS.ELEMENT_ROWS[fn.id.split("{")[0]]
    kinds = [(fmt, role, K) for role, _p in rows]
    try:
        CS.whole_blocks(fn.id, kinds)
    except ValueError:
        return old, None
    from verity.commitments.rowleaf import ELEMENT_FORMATS

    return old, [(FC.Port(f"p{i}", ELEMENT_FORMATS[fmt].row_len(K) // 2, kd).layout().prefix, kd) for i, kd in enumerate(kinds)]


def _gemm_fp():
    import importlib.util

    spec = importlib.util.spec_from_file_location("gemm_fp", Path(__file__).parents[1] / "pod" / "gemm_fp.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_no_core_row_prefix_coincides_with_another_frames_or_another_rows():
    """Every port's prefix in the new, default and packed frames, at the test shapes, K = 64 ... 1024 and the 12 cells: no new
    prefix equals one of another (format, role, k) or any old-frame prefix, where the old frames give x and W one prefix at every
    shape; the refused shapes are exactly the partial-block ones (of the cells, MXF4 K=2048). The derivation agrees with the
    stager's own lowering and `element_rows` at K=128."""
    shapes = [(d, k) for d in FORMAT for k in (64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384)]
    new: dict[bytes, tuple] = {}
    old: set[bytes] = set()
    refused = []
    for dtype, K in shapes:
        frames, rows = _frame_prefixes(dtype, K)
        for fr, prefixes in frames.items():
            assert prefixes[0] == prefixes[1], (dtype, K, fr, "the old frames' x and W rows share a prefix")
            old.update(prefixes)
        if rows is None:
            refused.append((dtype, K))
            continue
        assert rows[0][0] != rows[1][0]
        for prefix, kind in rows:
            assert new.setdefault(prefix, kind) == kind, (dtype, K, kind, new[prefix])
    assert not set(new) & old
    assert refused == [("e4m3", 64)] + [("nvf4", k) for k in (64, 128, 256, 512, 1024)] + [("mxf4", k) for k in (64, 128, 256, 512, 1024, 2048)]
    assert len(new) == 2 * (len(shapes) - len(refused))

    for dtype in FORMAT:
        _units, _s, _cl, act, uc, groups = _fp_class(dtype, 128)
        frames, _rows = _frame_prefixes(dtype, 128)
        for fr in ("default", "packed"):
            low = CS.class_lowering(act, uc, f"unit-class/{fr}", groups, fr == "packed")
            assert frames[fr] == [FC.Port(f"p{i}", w).layout().prefix for i, w in enumerate(low.definition.port_words)]
        assert CS.element_rows(act, uc, groups)[2] == [(FORMAT[dtype], role, 128) for role in (1, 2)]


def test_the_slot_circuits_are_built_only_when_a_shape_is_not_in_the_stage_cache(monkeypatch, tmp_path: Path):
    def hit(s):
        d = tmp_path / "cache" / s[:16]
        d.mkdir(parents=True, exist_ok=True)
        (d / "rec.json").write_text(json.dumps({"shape": s}))
        return d

    built = []
    monkeypatch.delenv("RESEARCH_RUN_DIR", raising=False)
    monkeypatch.setattr(CS, "cache_slot_circuits", lambda: built.append(1))
    monkeypatch.setattr(CS, "_stage_job_fresh", lambda s: {"shape": s, "broke": {"at": "test"}})
    args = ["--model", "tiny", "--partition", "q-word", "--stage-only", "--sample", "2"]
    monkeypatch.setattr(CS, "_stage_cache", hit)
    assert CS.main([*args, "--out", str(tmp_path / "hit")]) == 0 and built == []
    assert all(json.loads(ln)["stage_cached"] for ln in (tmp_path / "hit" / "staged.jsonl").read_text().splitlines())
    monkeypatch.setattr(CS, "_stage_cache", lambda s: None)
    assert CS.main([*args, "--out", str(tmp_path / "miss")]) == 0 and built == [1]


def test_selftest_jobs_runs_the_case_processes_at_once_and_keeps_their_records_in_case_order(monkeypatch, tmp_path: Path):
    cases = ["gpu_paths_agree", "gpu_proofs_match_cpu", "verify_ahead_matches_serial"]
    st: dict = {}

    def run(cmd, **_kw):
        case = cmd[cmd.index("--only") + 1]
        st["argv"][case] = cmd
        on = "caller" if threading.current_thread() is st["caller"] else "pool"
        if on == "pool":
            st["started"].wait()                                   # every case's process is running before any returns
            if case != cases[-1]:
                st["returned"][cases[cases.index(case) + 1]].wait()   # and they return last case first
        st["finished"].append(case)
        st["returned"][case].set()
        return CS.subprocess.CompletedProcess(cmd, 0, stdout=f"NEG\t{json.dumps({'case': case, 'pass': True, 'on': on})}\n", stderr="")

    monkeypatch.delenv("RESEARCH_RUN_DIR", raising=False)
    monkeypatch.setattr(CS.subprocess, "run", run)
    monkeypatch.setattr(CS, "cache_slot_circuits", lambda: None)
    monkeypatch.setattr(CS, "_stage_job", lambda s: {"shape": s, "stage": {"n": 16}, "table_reads": {}})
    args = ["--model", "tiny", "--partition", "q-word", "--sample", "1", "--selftest-only", "--selftest", "1", "--selftest-gpu",
            "--selftest-cases", ",".join(cases), "--selftest-binary", "flock-circuit-seeded"]
    argvs = {}
    for jobs, on, order in (("1", "caller", cases), ("3", "pool", cases[::-1])):
        st.update(argv={}, finished=[], started=threading.Barrier(len(cases)), returned={c: threading.Event() for c in cases},
                  caller=threading.current_thread())
        (tmp_path / "selftests.jsonl").unlink(missing_ok=True)
        assert CS.main([*args, "--selftest-jobs", jobs, "--out", str(tmp_path)]) == 0
        (rec,) = [json.loads(ln) for ln in (tmp_path / "selftests.jsonl").read_text().splitlines()]
        assert st["finished"] == order
        assert rec["selftest"]["all_pass"] is True and rec["selftest"]["subset"] == cases
        assert rec["selftest"]["gpu_cases"] == [{"case": c, "pass": True, "on": on} for c in cases]
        argvs[jobs] = st["argv"]
    assert argvs["3"] == argvs["1"]                                # each case its own process, with the same binary and arguments
    assert all(cmd[0] == "flock-circuit-seeded" and "--gpu" in cmd and cmd[-2:] == ["--only", c] for c, cmd in argvs["3"].items())


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
