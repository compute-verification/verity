"""served_zk's commitment and its check (`served_zk/served_commit.py`): a served call's fp8-p rows committed as C-Flock commits
a unit's rows, registered and received as one-stage's v1 record, a tile proof's public file accepted only when its committed
rows are the drawn tile's rows of that registration, and the audit record that holds them with the verdicts."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from verity.protocols.accounting.work.pouw.circuit import leaves as L
from verity.protocols.accounting.work.pouw.protocol import Context, Matmul, Shape
from verity.protocols.accounting.work.pouw.schemes import pearl_c as C
from verity.protocols.accounting.work.pouw.schemes.pearl_c_device import DEVICES

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "served_zk"))
import served_commit as SC  # noqa: E402
from verity.protocols.verification.sampled_proofs.one_stage import audit as A  # noqa: E402
from verity.protocols.verification.sampled_proofs.one_stage import registration as R  # noqa: E402

K, M, N = 128, 3, 4


def _rows(seed: int, n: int) -> list[bytes]:
    x = np.random.default_rng(seed).standard_normal((n, K)).astype("<f4")
    return [r.tobytes() for r in x]


@pytest.fixture(scope="module")
def served():
    scheme = C.PearlC(k_min=K, forming="v1", device=DEVICES["sm120"], hashing="h2")
    tag = lambda s: hashlib.sha256(s).digest()  # noqa: E731
    ctx = Context(tag(b"salt"), Matmul(5, Shape(16, K, N), 2), tag(b"root_a"), tag(b"root_b"))
    noise = scheme.noise(ctx)
    a = _rows(1, M) + [bytes(4 * K)]                 # a zero row, which forming refuses
    words = {"a": SC.form(scheme, noise, K, "a", a, list(range(M + 1))),
             "b": SC.form(scheme, noise, K, "b", _rows(2, N), list(range(N)))}
    return scheme, ctx, noise, words


def _commit(words, run="r-request"):
    binding = {"run": run, "manifest_sha256": "00" * 32, "scheme": "pearl-c-sm120-v1-h2", "matmul": 5, "shape": [16, K, N],
               "weight": 2, "real_rows": M + 1}
    st = SC.statement_of(K, M + 1, N, f"{run}/matmul-5")
    sides, priv = SC.commit_words(st, words, K)
    return {"schema": SC.SCHEMA, "binding": binding, "statement": st, "sides": sides, "served": {}}, priv


def _pub(bca: bytes, bcb: bytes, c) -> bytes:
    """A tile proof's public file as `verity_flock.circuit.write` lays it out: its header, then instance 0's b || c in port
    order (the A row, the B row, the tile digest's row)."""
    schemas = {"a0": c["sides"]["a"]["schema"], "b0": c["sides"]["b"]["schema"], "o0": L.ROW_SCHEMA["tile", SC.SIDES["a"]]}
    head = {"instances": 1, "frame_v3": {"schemas": schemas}}
    return json.dumps(head).encode() + b"\n" + bca + bcb + bytes(128)


def _draw(a, b):
    """The Lean verifier's subset:1 draw (`DRAW {…}`) of tile (a, b)."""
    return {"law": "subset", "population": (M + 1) * N, "k": 1, "units": [a * N + b]}


def test_rows_are_the_schemes_operands(served):
    """Each formed row is the fp8-p row of the operands `PearlC.encode` forms for a tile holding it, and the refused row is
    excluded."""
    scheme, ctx, noise, words = served
    assert words["a"][M] is None and all(w is not None for w in words["a"][:M] + words["b"])
    from verity.protocols.accounting.work.pouw.protocol import Tile
    ops = scheme.encode(ctx, noise, _rows(1, M)[1:2], _rows(2, N)[3:4], Tile((1,), (3,)))
    assert words["a"][1] == L.words_fp8_p(ops.a.codes[0], ops.pa[0])
    assert words["b"][3] == L.words_fp8_p(ops.b.codes[0], ops.pb[0])


def test_commitment_is_the_statements(served):
    """A row's b || c is C-Flock's commit of the row under its salt (`leaves.commit`), and each side's root the frame over
    those leaves under one-stage's served domain; the refused row is listed as excluded."""
    *_, words = served
    c, priv = _commit(words)
    assert c["sides"]["a"]["excluded"] == [M] and c["sides"]["b"]["excluded"] == []
    assert priv["b"]["bcs"][2] == L.commit("fp8-p", SC.SIDES["b"], words["b"][2], priv["b"]["salts"][2])
    again, _ = SC.commit_words(c["statement"], words, K, {s: p["salts"] for s, p in priv.items()})
    assert again == c["sides"]
    st = c["statement"]
    for side, s in c["sides"].items():
        assert s["domain_id"] == R.served_domain(st["program"], st["partition"], side, s["schema"], s["rows"], st["run"])


def test_the_registration_is_received_before_the_draw(served):
    """The committer's record is v1 with one root per side, the law subset:1 and a served window, over the call's tiles;
    the verifier's R1-R6 hold it to its own copies and its receipt names it.  A record whose root count or domain is not
    the verifier's own, or a stand-in window under the served run, is refused (no receipt)."""
    *_, words = served
    c, _ = _commit(words)
    rec = SC.registration_of(c)
    assert rec["format"] == R.FORMAT and rec["window"]["kind"] == "served" and rec["population"] == (M + 1) * N
    assert [r["port"] for r in rec["roots"]] == ["a", "b"] and rec["law"] == SC.LAW
    v, receipt = SC.receive(c, rec)
    assert v["ok"] and receipt["registration"] == R.digest(rec)
    bad = json.loads(json.dumps(rec))
    bad["roots"][1]["leaves"] += 1
    v, receipt = SC.receive(c, bad)
    assert not v["ok"] and "R3-leaves" in v["codes"] and receipt is None
    other, _ = _commit(words, run="r-other")
    moved = {**rec, "roots": SC.registration_of(other)["roots"]}
    v, receipt = SC.receive(c, moved)
    assert not v["ok"] and "R3-domain" in v["codes"] and receipt is None


def test_check_accepts_the_drawn_rows_only(served, tmp_path):
    *_, words = served
    c, priv = _commit(words)
    rec = SC.registration_of(c)
    SC.save(tmp_path / "p.npz", priv)
    priv = SC.load(tmp_path / "p.npz")
    op = {"a": SC.opening(c, "a", priv["a"]["bcs"], 1), "b": SC.opening(c, "b", priv["b"]["bcs"], 3)}
    honest = _pub(priv["a"]["bcs"][1], priv["b"]["bcs"][3], c)
    assert SC.check(rec, c, _draw(1, 3), honest, op) is None
    # another row of the same commitment, opened at the drawn position
    assert "not the registered row" in SC.check(rec, c, _draw(1, 3), _pub(priv["a"]["bcs"][0], priv["b"]["bcs"][3], c), op)
    # the drawn rows of another request's commitment, and this proof against another request's registration
    other, opriv = _commit(words, run="r-other")
    orec = SC.registration_of(other)
    assert "not the registered row" in SC.check(rec, c, _draw(1, 3), _pub(opriv["a"]["bcs"][1], opriv["b"]["bcs"][3], c), op)
    oop = {"a": SC.opening(other, "a", opriv["a"]["bcs"], 1), "b": SC.opening(other, "b", opriv["b"]["bcs"], 3)}
    assert "not the registered row" in SC.check(orec, other, _draw(1, 3), honest, oop)
    # roots registered under another run's domain, an opening of another row, an excluded row, a draw of two units
    assert "served domain" in SC.check({**rec, "window": {**rec["window"], "run": "r-other/matmul-5"}}, c, _draw(1, 3), honest, op)
    assert "not the drawn row" in SC.check(rec, c, _draw(2, 3), honest, op)
    assert "excluded" in SC.check(rec, c, _draw(M, 3), honest, op)
    assert "not one" in SC.check(rec, c, {**_draw(1, 3), "k": 2, "units": [3, 4]}, honest, op)


def test_the_audit_record_accepts_only_when_lean_and_the_opening_accept(served):
    """One-stage's audit record of the drawn tile: the registration, receipt and draw, the proved unit and the verdicts.
    It is accepted, with its profile, only when the Lean verdict of record and the opening check both accept and the draw
    is the registration's; a rejected Lean verdict, a rejected opening, or a receipt of another record leave it
    rejected."""
    *_, words = served
    c, _ = _commit(words)
    rec = SC.registration_of(c)
    _, receipt = SC.receive(c, rec)
    stage = {"definition": "Pc8TileHidden", "circuit_sha512": "11" * 64, "class_sha512": "22" * 64, "ands": 7, "k_log": 3,
             "binding": "hash scratch"}
    yes = {"of_record": False, "accepted": True, "why": ""}
    lean = {"of_record": True, "accepted": True, "why": "", "digest": "33" * 64}
    a = SC.audit(rec, receipt, _draw(1, 3), stage, b"pub", {"lean": lean, "opening": yes, "rust": None})
    assert a["format"] == A.FORMAT and a["accepted"] and a["complete"] and a["proved_units"] == [1 * N + 3]
    assert a["statement"]["tile"] == [1, 3] and a["integrity_profile"]["registration"] == R.digest(rec)
    assert a["verdicts"]["opening"]["accepted"] and all(ch["ok"] for ch in a["checks"].values())
    assert not SC.audit(rec, receipt, _draw(1, 3), stage, b"pub", {"lean": {**lean, "accepted": False}, "opening": yes})["accepted"]
    assert not SC.audit(rec, receipt, _draw(1, 3), stage, b"pub", {"lean": lean, "opening": {**yes, "accepted": False}})["accepted"]
    other = SC.registration_of(_commit(words, run="r-other")[0])
    assert not SC.audit(rec, R.receipt(other, SC.VERIFIER), _draw(1, 3), stage, b"pub", {"lean": lean, "opening": yes})["accepted"]


def test_lean_and_rust_verdicts_are_read_from_their_outputs(tmp_path):
    (tmp_path / "lean.out").write_text('setup: 3 ms\nVERDICT {"accepted":false,"why":"ring-switch claim 2 mismatch"}\n')
    v = SC.lean_verdict(tmp_path / "lean.out")
    assert v["of_record"] and v["accepted"] is False and "ring-switch" in v["why"] and len(v["digest"]) == 128
    (tmp_path / "none.out").write_text("")
    assert SC.lean_verdict(tmp_path / "none.out")["accepted"] is None
    (tmp_path / "prove.out").write_text('LIVE\t{"verdict": {"accepted": true, "why": ""}}\n')
    assert SC.rust_verdict(tmp_path / "prove.out")["accepted"] is True and not SC.rust_verdict(tmp_path / "prove.out")["of_record"]


def test_the_cli_registers_receives_checks_and_audits(served, tmp_path):
    """The job's steps through `main`: register, receive (check.json and receipt.json), the opening check's verdict file,
    and the audit record accepting the honest tile."""
    *_, words = served
    c, priv = _commit(words)
    (tmp_path / "commitment.json").write_text(json.dumps(c))
    assert SC.main(["register", str(tmp_path / "commitment.json"), str(tmp_path / "registration.json")]) == 0
    assert SC.main(["receive", str(tmp_path / "commitment.json"), str(tmp_path / "registration.json"), str(tmp_path)]) == 0
    assert json.loads((tmp_path / "check.json").read_text())["ok"]
    (tmp_path / "draw.json").write_text(json.dumps(_draw(1, 3)))
    (tmp_path / "pub.bin").write_bytes(_pub(priv["a"]["bcs"][1], priv["b"]["bcs"][3], c))
    (tmp_path / "opening.json").write_text(json.dumps({"a": SC.opening(c, "a", priv["a"]["bcs"], 1),
                                                       "b": SC.opening(c, "b", priv["b"]["bcs"], 3)}))
    names = ("registration", "commitment", "draw")
    assert SC.main(["check", *(str(tmp_path / f"{n}.json") for n in names), str(tmp_path / "pub.bin"),
                    str(tmp_path / "opening.json"), "--out", str(tmp_path / "open.json")]) == 0
    (tmp_path / "stage.json").write_text(json.dumps({"definition": "d", "circuit_sha512": "1", "class_sha512": "2", "ands": 1,
                                                     "k_log": 1, "binding": "hash scratch"}))
    (tmp_path / "lean.out").write_text('VERDICT {"accepted":true,"why":""}\n')
    assert SC.main(["audit", *(str(tmp_path / f"{n}.json") for n in ("registration", "receipt", "draw", "stage")),
                    str(tmp_path / "pub.bin"), str(tmp_path / "audit.json"), "--lean", str(tmp_path / "lean.out"),
                    "--opening", str(tmp_path / "open.json")]) == 0
    assert json.loads((tmp_path / "audit.json").read_text())["accepted"]
