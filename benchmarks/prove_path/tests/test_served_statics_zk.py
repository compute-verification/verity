"""`served_statics_zk`'s units on a three-step window at the compile slice's class: one registration holds the honest window
and each case, every unit's served rows are its leaf as the Lean verifier's registered reads open it, the honest blocks
(idle ones included) are accepted and each case gets its flag; a record of another G than the statics' is refused; an
``srb`` row shorter than the scan's piece is found whole; the shared-row public file is read by its refs; and at the
served class the largest block that fits C-Flock's 2^27 is 64 steps."""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from verity.core.protocols.certifier import compile_slice as CS, served_rows as SR, served_statics as SS
from verity.core.protocols.sampled_proofs.one_stage import registered as REG
from verity.ml.commitments import advice as ADV

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE.parents[1] / "one_stage")]
import a8 as A8  # noqa: E402
import served_statics_zk as Z  # noqa: E402

FIXTURE = HERE.parents[2] / "verity/core/protocols/certifier/tests/fixtures/compile_slice_smallkv.json"
WINDOW = [((0, 1, 0, 0, 5), (1, 2, 0, 0, 3)), ((0, 1, 1, 5, 1), (1, 2, 1, 3, 1)), ((0, 1, 1, 6, 1), (1, 2, 1, 4, 1))]
RUN_ROOT = bytes(range(64))
COUNT, POS = 1 << SS.FLAGS.index("count"), 1 << SS.FLAGS.index("pos")


def _S():
    fx = json.loads(FIXTURE.read_text())
    return SimpleNamespace(**CS.statics(fx["R"], fx["W"], fx["A"], fx["engine"], fx["IW"], Q=fx["Q"], D=fx["D"], P=fx["P"],
                                        K=fx["K"], CN=fx["CN"], CTW=fx["CTW"], CU=fx["CU"]))


def test_the_units_read_their_leaves_and_get_their_verdicts():
    S = _S()
    d = Z.plan(S, list(WINDOW), SS.program_of(WINDOW), RUN_ROOT, 1)
    honest, cases = d.units["window"], d.units["cases"]
    assert [u.j for u in honest] == list(range(S.P)) and {u.verdict for u in honest} == {SS.ACCEPT}
    assert {u.kind: u.verdict for u in cases} == {"start": SS.ACCEPT & ~POS, "short": SS.ACCEPT & ~COUNT,
                                                  "dropped": SS.ACCEPT & ~COUNT, "id": SS.ACCEPT, "other": SS.ACCEPT & ~POS}
    every = honest + cases
    assert SS.evaluate(d.st, [SS.Case(u.rows, u.flags, u.verdict) for u in every]) == [u.verdict for u in every]
    assert d.record["count"] == len(Z.KINDS) * S.P and [u.leaf for u in cases] == [S.P * c + 0 for c in (1, 2, 3)] + [
        S.P * c + 1 for c in (4, 5)]
    port = SR.reads_port(d.record, RUN_ROOT, [u.leaf for u in every])
    dom, root = ADV.domain(SR.binding(RUN_ROOT), d.record["count"]), bytes.fromhex(port["root"])
    assert dom.domain_id.hex() == port["domain"]
    reg = d.served.registration
    for u in every:
        assert REG.verify_read(dom, root, u.leaf, d.served.bc(u.leaf), b"".join(reg.open(u.leaf).path), port["row_schema"])


def test_a_shared_row_file_is_read_by_its_refs(tmp_path):
    ports = [SimpleNamespace(name="a"), SimpleNamespace(name="b")]
    bc = {k: bytes([k]) * 128 for k in range(3)}
    head = json.dumps({"instances": 2, "shared_rows": {"a": 1, "b": 2}}).encode()
    refs = b"".join(x.to_bytes(4, "little") for x in (0, 1, 0, 0))
    f = tmp_path / "pub-2.bin"
    f.write_bytes(head + b"\n" + bc[0] + bc[1] + bc[2] + refs)
    assert Z.shared_bcs(f, ports) == [{"a": bc[0], "b": bc[2]}, {"a": bc[0], "b": bc[1]}]


def test_a_record_whose_g_isnt_the_statics_is_refused(monkeypatch):
    """REDTEAM phase 19: the served-rows record's G is held to the G of the statics the unit is built with, where `plan`
    reads the record and where `stage` writes the verifier's reads of it. A record of another G, or of none (read as P),
    is refused."""
    S = _S()
    St = SimpleNamespace(**SS.statics(S, G=1))
    assert Z.same_g(S, St, {"G": 1}) == 1
    for rec in ({"G": S.P}, {}):
        with pytest.raises(SystemExit, match=f"record's G is {S.P} but the unit's statics' is 1"):
            Z.same_g(S, St, rec)
    register = SR.register
    monkeypatch.setattr(SR, "register", lambda S, wins, root, G=None: register(S, wins, root, G=S.P))
    with pytest.raises(SystemExit, match=f"record's G is {S.P} but the unit's statics' is 1"):
        Z.plan(S, list(WINDOW), SS.program_of(WINDOW), RUN_ROOT, 1)
    monkeypatch.undo()
    d = Z.plan(S, list(WINDOW), SS.program_of(WINDOW), RUN_ROOT, 1)
    d.record = {**d.record, "G": S.P}
    with pytest.raises(SystemExit, match=f"record's G is {S.P} but the unit's statics' is 1"):
        Z.stage(None, None, "cases", d.units["cases"], d, bytes(32), True)


def test_an_srb_row_shorter_than_a_piece_is_found_whole(tmp_path):
    """At G = 1 in the slice's class an ``srb`` row is 8 bytes, shorter than the scan's 16-byte piece: `received` finds
    it whole (`leak_scan.find_whole`), so written as hex into what the verifier received it is leaked, while the salts are
    cut into pieces and scanned as before; the instance file is to hold both."""
    S = _S()
    units = Z.plan(S, list(WINDOW), SS.program_of(WINDOW), RUN_ROOT, 1).units["window"]
    rnd = random.Random(3)
    blobs, held = Z.blobs_of(units, {r: [rnd.randbytes(192) for _ in units] for r in ("srb", *Z.OPENED)})
    rows = {k: x for k, x in blobs.items() if "salt" not in k}
    assert list(rows) == ["srb 0", "srb 1", "srb 2"] and {len(x) for x in rows.values()} == {8}
    (tmp_path / "inst.bin").write_bytes(b"".join(blobs.values()))
    vdir = tmp_path / "verifier"
    vdir.mkdir()
    (vdir / "audit.json").write_text(json.dumps({"accepted": True}))
    got = Z.received(A8, vdir, tmp_path / "inst.bin", blobs, held, b"")
    wr = got["whole_rows"]
    assert (got["leaked"], got["control"]["holds_all"], wr["rows"], wr["leaked"], wr["control"]["holds_all"]) == (
        [], True, 3, [], True)
    (vdir / "openings.json").write_text(json.dumps([{"salt/o0": rows["srb 1"].hex(), "salt/pub": blobs["srb salt 2"].hex()}]))
    got = Z.received(A8, vdir, tmp_path / "inst.bin", blobs, held, b"")
    assert sorted(got["leaked"]) == sorted(f"srb salt 2 @{o}" for o in range(0, 192, 16))
    assert (got["whole_rows"]["leaked"], got["whole_rows"]["placed"]) == (["srb 1"], {"srb 1": [("openings.json", None, "file")]})
    (tmp_path / "inst.bin").write_bytes(b"".join(x for k, x in blobs.items() if k != "srb 1"))
    assert Z.received(A8, vdir, tmp_path / "inst.bin", blobs, held, b"")["whole_rows"]["control"]["missing"] == ["srb 1"]


def test_the_served_class_fits_at_64_steps_a_block():
    c = Z.CLASSES["served"]
    fits = {G: Z.zk_block(SS.rows(SimpleNamespace(**SS.statics(SimpleNamespace(**c), G=G))), SS.size(**SS.statics(
        SimpleNamespace(**c), G=G))["ands"])["fits"] for G in (16, 64, 128)}
    assert fits == {16: True, 64: True, 128: False}
