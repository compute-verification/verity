"""Pearl-C4's device-run replay (`pearl_c4/replay.py`) on `test_pouw_pearl_c4`'s lifecycle dumped as a device would dump
it: it counts a dump's scale bytes, accepts the honest transcript (with its workers giving the serial one), declares R1's
rows as the honest prover does, and rejects a wrong leaf, a root the rows don't have, a tampered row, a beacon not the
commitment's, hidden exclusions, a missing opening and a B̃ formed under another epoch's salt or none."""
from __future__ import annotations

import base64
import json
import pickle
import sys
from fractions import Fraction
from pathlib import Path

import pytest

# the replay is imported by name so that its forkserver workers find it
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pearl_c4"))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "verity" / "protocols" / "pouw" / "clear"))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "verity" / "protocols" / "pouw" / "verifier" / "tests"))
import pearl_c4_vectors as V
import replay as R
import test_pouw_pearl_c4 as T
from clear_audit import Prover
import clear_work as CW
from verity.protocols.pouw.verifier.audit import Epoch, Workload
from verity.protocols.pouw.verifier.protocol import Matmul, Shape
from verity.protocols.pouw.prover.schemes import pearl_c4 as F
from verity.protocols.pouw.prover.schemes import pearl_c_work as W

K = T.K


def test_replay_counts_a_dumps_scale_bytes():
    """`device_scales` counts what D-SB refuses in a dump's scale bytes: outside [0x01, 0x7E], bit 7, zero and NaN."""
    counts = R.device_scales(F.NVFP4, bytes([0x38, 0x00, 0x80, 0xB8, 0x7F, 0x01]))
    assert counts == {"bytes": 6, "outside_range": 4, "min": 0, "max": 0xB8, "bit7": 2, "zero": 1, "nan": 1}


def device_dump(path, s: F.PearlC4, swap_leaves=False, flip_root_a=False) -> None:
    """`lifecycle`'s run as a device would dump it (its roots and leaves), optionally with its first two leaves swapped
    or A's root altered."""
    epoch, prover, tr, rows_a, rows_b = T.lifecycle(s)
    leaves = [prover.digest(i) for i in range(2)]
    root_a = tr.activation_roots[0]
    call = {"scheme": s.name, "shape": [32, K, 16], "index": 3, "weight": 9, "real_rows": 20, "epoch_beacon": b"beacon".hex()}
    R.write_dump(path, call, b"".join(rows_a), b"".join(rows_b), bytes([root_a[0] ^ 1]) + root_a[1:] if flip_root_a else root_a,
                 prover.weight_roots[9], b"".join(leaves[::-1] if swap_leaves else leaves))


def test_replay_accepts_the_device_transcript(tmp_path):
    s = V.scheme("nvfp4", True)
    device_dump(tmp_path, s)
    (tmp_path / "scales_b.bin").write_bytes(bytes([0x38, 0x01, 0x7E]))
    t = json.loads(json.dumps(R.prove(tmp_path, "r-test", 16, scheme=s)))
    assert list(t["device_scales"]) == ["b"] and t["device_scales"]["b"]["outside_range"] == 0
    assert t["records"] == [{"real_rows": 20, "k_kept": K, "excluded_a": [3], "excluded_b": []}]
    assert t["host_roots_match"] == {"a": True, "b": True} and [o["tile"] for o in t["openings"]] == [0, 1]
    rep = R.verify(t, scheme=s)
    assert rep["accepted"] and rep["opened"] == 17 and rep["reason"] == ""
    first = rep["tiles"][0]
    assert first["credited"] in ([15, 16], [4, 16]) and Fraction(first["debit"]["total"]) <= Fraction(first["cap"])
    assert pickle.loads(pickle.dumps(s)).fmt is s.fmt and R.verify(t, workers=2, scheme=s)["tiles"] == rep["tiles"]
    assert R.accept_line(rep, "r-test:t.json", "00", "abc").startswith(f"ACCEPT {R.REPORT} {s.name} r-test:t.json sha256:00 ")


def test_replay_rejects(tmp_path):
    """A wrong leaf, a root the rows don't have, a tampered row, a beacon not the commitment's, hidden exclusions, and a
    missing opening."""
    s = V.scheme("nvfp4", True)
    for name, kw, why in (("swap", {"swap_leaves": True}, "checked values"),
                          ("root", {"flip_root_a": True}, "activation opening")):
        device_dump(tmp_path / name, s, **kw)
        rep = R.verify(R.prove(tmp_path / name, "r-test", 8, scheme=s), scheme=s)
        assert not rep["accepted"] and rep["reason"] == why
    device_dump(tmp_path / "ok", s)
    good = R.prove(tmp_path / "ok", "r-test", 8, scheme=s)

    def bad(edit):
        t = json.loads(json.dumps(good))
        edit(t)
        return R.verify(t, scheme=s)

    def tamper(t):
        row = bytearray(base64.b64decode(t["openings"][0]["a"][0]["value"]))
        row[0] ^= 1
        t["openings"][0]["a"][0]["value"] = base64.b64encode(bytes(row)).decode()

    def hide(t):
        t["records"][0]["excluded_a"] = []
        unit = Matmul(3, Shape(32, K, 16), 9)
        epoch = Epoch.unverified(Workload((unit,)), {9: bytes.fromhex(t["weight_roots"]["9"])}, b"beacon")
        rec = W.UnitRecord(unit, 20, K)
        beacon = R.draw_beacon(bytes.fromhex(t["transcript"]["root"]),
                               W.records_root(epoch.salt, epoch.weights_root, {9: W.Split(K)}, [rec]))
        t["draw"]["value"] = beacon.value.hex()

    assert bad(tamper)["reason"] == "activation opening"
    assert bad(lambda t: t["draw"].update(value="00" * 32))["reason"].startswith("the draws' beacon is not")
    assert "A row 3 is credited but fails" in bad(hide)["reason"]
    assert bad(lambda t: t["openings"].pop(0))["reason"] == "'the transcript does not open tile 0'"


@pytest.mark.slow
def test_replay_declares_r1_rows_and_its_workers_give_the_serial_transcript(tmp_path):
    """The replay's prove declares R1's rows as the honest prover does (`clear_work.declare` on the transcript): A row 5,
    the spike row, passes the per-row rules but its own F1′ passes R1's bound, so prove excludes it and the
    verifier accepts.  Credited, as prove declared before R1 (an m64-n512-k2048 unit's B row 386), it
    rejects.  prove maps R1 row by row over its workers, and two workers give the one worker's transcript."""
    s = V.scheme("nvfp4", True)
    rows_a = V.rows("work-life", "A", 20, K) + [F.bf16_row([0] * K)] * 12
    rows_a[5] = F.bf16_row([x >> 16 for x in T.spike_row(0, K)])
    rows_b = V.rows("work-life", "B", 16, K)
    unit = Matmul(3, Shape(32, K, 16), 9)
    prover = Prover(s, {9: rows_b})
    epoch = Epoch.unverified(Workload((unit,)), prover.weight_roots, b"beacon")
    tr = prover.prove(epoch, {3: rows_a})
    assert F.row_passes(s.fmt, F.parse_row(rows_a[5], K))
    honest = CW.declare(s, epoch, {3: rows_a}, {9: rows_b}, {9: W.Split(K)}, real_rows={3: 20}, transcript=tr)
    assert honest == [W.UnitRecord(unit, 20, K, (5,), ())]
    call = {"scheme": s.name, "shape": [32, K, 16], "index": 3, "weight": 9, "real_rows": 20, "epoch_beacon": b"beacon".hex()}
    R.write_dump(tmp_path, call, b"".join(rows_a), b"".join(rows_b), tr.activation_roots[0], prover.weight_roots[9],
                 b"".join(prover.digest(i) for i in range(2)))
    serial = R.prove(tmp_path, "r-test", 12, workers=1, scheme=s)
    assert serial["records"] == [{"real_rows": 20, "k_kept": K, "excluded_a": [5], "excluded_b": []}]
    assert [o["tile"] for o in serial["openings"]] == [0, 1]
    assert R.prove(tmp_path, "r-test", 12, workers=2, scheme=s) == serial
    rep = R.verify(json.loads(json.dumps(serial)), scheme=s)
    assert rep["accepted"] and rep["reason"] == ""

    credited = json.loads(json.dumps(serial))
    credited["records"][0]["excluded_a"] = []
    digest = W.records_root(epoch.salt, epoch.weights_root, {9: W.Split(K)}, [W.UnitRecord(unit, 20, K)])
    credited["draw"]["value"] = R.draw_beacon(bytes.fromhex(serial["transcript"]["root"]), digest).value.hex()
    rep = R.verify(credited, scheme=s)
    assert not rep["accepted"] and rep["reason"].startswith("A row 5 is credited but its own debit passes R1's bound")


def test_replay_refuses_a_b_tilde_keyed_by_another_salt(tmp_path):
    """D-SK in the device replay: a B̃ formed under another epoch's salt, or under none (`stale_b`), fails the checked
    values of a sampled tile, as it fails the work sampling design's (`test_b_tilde_is_keyed_by_this_jobs_salt`)."""
    s = V.scheme("nvfp4", True)
    for n, salt in enumerate((b"another epoch's salt", b"")):
        device_dump(tmp_path / str(n), T.stale_b(s, salt))
        rep = R.verify(R.prove(tmp_path / str(n), "r-test", 8, scheme=s), scheme=s)
        assert not rep["accepted"] and rep["reason"] == "checked values"
