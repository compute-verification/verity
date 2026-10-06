"""served_zk's window through the proof service (`served_zk/served_commit.py` over `served_zk/service.py`): a window of
served calls' fp8-p rows committed as C-Flock commits a unit's rows, its tiles padded with filler tiles to a grid level,
registered with exactly PoUW's public items, received, drawn once, each drawn tile's proof opened against the window
root, and the outcome publishing exactly the public items."""
from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from verity.protocols.accounting.work.pouw import window as Wd
from verity.protocols.accounting.work.pouw.circuit import leaves as L
from verity.protocols.accounting.work.pouw.protocol import Context, Matmul, Shape
from verity.protocols.accounting.work.pouw.schemes import pearl_c as C
from verity.protocols.accounting.work.pouw.schemes.pearl_c_device import DEVICES
from verity.protocols.verification.sampled_proofs.one_stage import registration as R

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "served_zk"))
import served_commit as SC  # noqa: E402
import service as S  # noqa: E402

K, M, N = 128, 3, 4


def _rows(seed: int, n: int) -> list[bytes]:
    x = np.random.default_rng(seed).standard_normal((n, K)).astype("<f4")
    return [r.tobytes() for r in x]


def _params(run: str = "r-request") -> dict:
    return {"context": f"served-zk/{run}/{hashlib.sha256(run.encode()).hexdigest()[:16]}", "deployment": f"served-zk/{run}",
            "k": K, "eps": "1/2", "delta": "1/10", "grid_bits": Wd.GRID_BITS, "range": [1, 1 << 40], "entries": 4, "w_rows": N}


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


def _calls(words) -> list[dict]:
    """Two calls of the same rows: 2·M·N = 24 real tiles, grid level 23, padded to 26 with a filler entry of 1 x 2 zero rows."""
    return [{"record": {"matmul": m, "shape": [16, K, N], "weight": 2, "real_rows": M + 1}, "words": words} for m in (5, 9)]


def _weights() -> list[bytes]:
    return [np.arange(K, dtype="<u2").tobytes() for _ in range(N)]


def _window(words, run="r-request"):
    params = _params(run)
    root_w, _ = SC.commit_weights(params, _weights())
    public, advice, priv = SC.commit_window(params, _calls(words), root_w)
    return params, public, advice, priv


def _register(params, public, **extra):
    return S.register(SC.spec_of(params, public["level"]), {**public["items"], **extra}, public["roots"])


def _pub(bca: bytes, bcb: bytes) -> bytes:
    """A tile proof's public file as `verity_flock.circuit.write` lays it out: its header, then instance 0's b || c in port
    order (the A row, the B row, the tile digest's row)."""
    schemas = {"a0": L.ROW_SCHEMA["fp8-p", SC.SIDES["a"]], "b0": L.ROW_SCHEMA["fp8-p", SC.SIDES["b"]],
               "o0": L.ROW_SCHEMA["tile", SC.SIDES["a"]]}
    return json.dumps({"instances": 1, "frame_v3": {"schemas": schemas}}).encode() + b"\n" + bca + bcb + bytes(128)


def _unit_pub(advice, priv, t: int) -> bytes:
    e, i, j = Wd.tile_at(advice["entries"], t)
    return _pub(priv[e]["a"]["bcs"][i], priv[e]["b"]["bcs"][j])


def _sample(units):
    return lambda population, k: {"law": "subset", "population": population, "k": k, "units": sorted(units)[:k]}


def test_rows_are_the_schemes_operands(served):
    """Each formed row is the fp8-p row of the operands `PearlC.encode` forms for a tile holding it, and the refused row is
    none of the call's kept rows."""
    scheme, ctx, noise, words = served
    assert words["a"][M] is None and all(w is not None for w in words["a"][:M] + words["b"])
    from verity.protocols.accounting.work.pouw.protocol import Tile
    ops = scheme.encode(ctx, noise, _rows(1, M)[1:2], _rows(2, N)[3:4], Tile((1,), (3,)))
    assert words["a"][1] == L.words_fp8_p(ops.a.codes[0], ops.pa[0])
    assert words["b"][3] == L.words_fp8_p(ops.b.codes[0], ops.pb[0])


def test_the_window_is_padded_to_a_grid_level_with_filler(served):
    """The calls' kept tiles, then a filler entry of zero rows up to the padded count, then empty entries to the bound; the
    window statement holds on the advice, each side's root is under one-stage's served domain of its entry's port, and
    replaying the salts gives the same commitment."""
    *_, words = served
    params, public, advice, priv = _window(words)
    work = Wd.bucket(2 * M * N)
    assert (advice["real"], public["level"], work.credited, work.n) == (2 * M * N, work.level, 24, 26)
    kinds = [e["kind"] for e in advice["entries"]]
    assert kinds == ["call", "call", "filler", "none"]
    call, filler = advice["entries"][0], advice["entries"][2]
    assert call["a"]["keep"] == list(range(M)) and call["a"]["rows"] == M + 1 and (call["lo"], call["hi"]) == (0, M * N)
    assert (filler["lo"], filler["hi"]) == (24, 26) and (filler["a"]["rows"], filler["b"]["rows"]) == (1, 2)
    assert not priv[2]["a"]["words"].any() and not priv[2]["b"]["words"].any()
    assert Wd.window_problem(advice["entries"], work, params["entries"]) is None
    _, _, part = SC.population(K, work.n)
    for e in (0, 2):
        for side in "ab":
            dom, schema = SC.side_domain(params, public["level"], e, side, advice["entries"][e][side]["rows"])
            assert dom.domain_id.hex() == R.served_domain(part["program"], SC.PT.digest(part), f"{e}/{side}", schema,
                                                           advice["entries"][e][side]["rows"], params["context"])
    assert priv[0]["b"]["bcs"][2] == L.commit("fp8-p", SC.SIDES["b"], words["b"][2], priv[0]["b"]["salts"][2])
    salts = {e: {s: priv[e][s]["salts"] for s in "ab"} for e in range(3)}
    again, advice2, _ = SC.commit_window(params, _calls(words), public["items"]["root_W"], {**salts, "entries": priv["entries"]})
    assert again == public and advice2["entries"] == advice["entries"]


def test_register_makes_exactly_the_public_items_public(served):
    """The record's window holds exactly the registered public items (work, check, sampling, root_W) and nothing of the
    calls: no shape, no call index, no row count.  One more item, a root_W that is no registered root, or a commitment
    whose level the range refuses, is refused."""
    *_, words = served
    params, public, _, _ = _window(words)
    rec = _register(params, public)
    assert rec["window"] == {"kind": "served", "context": params["context"], "public": public["items"]}
    assert set(rec["window"]["public"]) == set(Wd.REGISTERED) and rec["population"] == 26
    assert rec["law"] == {"law": "subset", "k": rec["window"]["public"]["sampling"]["law"]["k"]}
    assert [r["port"] for r in rec["roots"]] == ["window", "W"]
    text = json.dumps(rec)
    for leak in ('"matmul"', '"shape"', '"real_rows"', '"rows"', '"keep"'):
        assert leak not in text
    with pytest.raises(S.Refused, match="public-items"):
        _register(params, public, shapes=[[16, K, N]])
    with pytest.raises(S.Refused, match="public-commitment"):
        _register(params, public, root_W="00" * 64)
    with pytest.raises(S.Refused, match="outside"):
        SC.work_at({**params, "range": [100, 1 << 40]}, public["level"])


def test_receive_holds_every_item_to_the_verifiers_own(served):
    """The verifier receives the honest record and its receipt names it.  An extra public key, a raised level (more credit),
    a K′ other than its own, or another request's record (its own context) is refused, with no receipt."""
    *_, words = served
    params, public, _, _ = _window(words)
    rec = _register(params, public)
    v, receipt = SC.receive(params, rec)
    assert v["ok"] and receipt["registration"] == R.digest(rec)
    extra = json.loads(json.dumps(rec))
    extra["window"]["public"]["shapes"] = [[16, K, N]]
    v, receipt = SC.receive(params, extra)
    assert not v["ok"] and "public-items" in v["codes"] and receipt is None
    raised = json.loads(json.dumps(rec))
    raised["window"]["public"]["work"]["level"] += 1
    v, receipt = SC.receive(params, raised)
    assert not v["ok"] and receipt is None and {"public-value", "R2-population", "R3-domain"} <= set(v["codes"])
    k = json.loads(json.dumps(rec))
    k["window"]["public"]["sampling"]["law"]["k"] -= 1
    v, receipt = SC.receive(params, k)
    assert not v["ok"] and "public-value" in v["codes"] and receipt is None
    oparams, opublic, _, _ = _window(words, run="r-other")
    v, receipt = SC.receive(params, _register(oparams, opublic))
    assert not v["ok"] and "public-context" in v["codes"] and receipt is None


def test_select_draws_once_after_the_receipt(served, tmp_path):
    """The draw is the law's subset:K′ over the padded population, once per receipt: a second select is refused, so no
    retry redraws; a receipt of another record and a draw of the wrong size are refused."""
    *_, words = served
    params, public, _, _ = _window(words)
    rec = _register(params, public)
    _, receipt = SC.receive(params, rec)
    spec = SC.spec_of(params, public["level"])
    k = spec.selection["k"]
    with pytest.raises(S.Refused, match="not the law"):
        S.select(spec, rec, receipt, lambda p, _k: _sample(range(p))(p, k + 1), tmp_path / "bad")
    d = S.select(spec, rec, receipt, _sample(range(26)), tmp_path / "log")
    assert d["units"] == list(range(k)) and d["population"] == 26
    with pytest.raises(S.Refused, match="once"):
        S.select(spec, rec, receipt, _sample(range(26)), tmp_path / "log")
    oparams, opublic, _, _ = _window(words, run="r-other")
    with pytest.raises(S.Refused, match="receipt"):
        S.select(spec, rec, R.receipt(_register(oparams, opublic), SC.VERIFIER), _sample(range(26)), tmp_path / "x")


def test_check_opens_the_drawn_units_rows_only(served, tmp_path):
    """A proof's public file opens a drawn unit, a call's tile or the filler's, only when its rows are that unit's rows of
    this window: another row, another window's rows, another window's registration, or an undrawn unit is rejected."""
    *_, words = served
    params, public, advice, priv = _window(words)
    rec = _register(params, public)
    SC.save(tmp_path / "p.npz", priv)
    priv = SC.load(tmp_path / "p.npz")
    draw = {"law": "subset", "population": 26, "k": 3, "units": [5, 13, 25]}
    for t in draw["units"]:
        assert SC.check(params, rec, draw, t, _unit_pub(advice, priv, t), SC.opening(advice, priv, params, t)) is None
    assert SC.opening(advice, priv, params, 25)["entry_value"]["kind"] == "filler"
    op = SC.opening(advice, priv, params, 5)
    assert "not entry 0's row" in SC.check(params, rec, draw, 5, _unit_pub(advice, priv, 6), op)
    assert "not drawn" in SC.check(params, rec, draw, 6, _unit_pub(advice, priv, 6), SC.opening(advice, priv, params, 6))
    assert SC.check(params, rec, draw, 6, _unit_pub(advice, priv, 6), SC.opening(advice, priv, params, 6), drawn=False) is None
    assert "not the unit's row" in SC.check(params, rec, draw, 6, _unit_pub(advice, priv, 6), op, drawn=False)
    oparams, opublic, oadvice, opriv = _window(words, run="r-other")
    orec = _register(oparams, opublic)
    assert "row" in SC.check(params, rec, draw, 5, _unit_pub(oadvice, opriv, 5), op)
    assert "window" in SC.check(params, orec, draw, 5, _unit_pub(advice, priv, 5), op)
    forged = {**op, "entry_value": {**op["entry_value"], "hi": op["entry_value"]["hi"] + 1}}
    assert "not the registered window's entry" in SC.check(params, rec, draw, 5, _unit_pub(advice, priv, 5), forged)


def _unit(t: int, lean: bool = True, opening: bool = True) -> dict:
    stage = {"definition": "Pc8TileHidden", "circuit_sha512": "11" * 64, "class_sha512": "22" * 64, "ands": 7, "k_log": 3,
             "binding": "hash scratch"}
    return {"unit": t, "statement": SC.statement(stage, b"pub"),
            "verdicts": {"lean": {"of_record": True, "accepted": lean, "why": "", "digest": "33" * 64},
                         "opening": {"of_record": False, "accepted": opening, "why": ""}}}


def test_the_outcome_publishes_exactly_the_public_items(served, tmp_path):
    """The outcome over every drawn unit accepts only when each unit's Lean verdict and opening check accept; its public
    items are exactly PoUW's (`window.PUBLIC`); it is made once per registration, and units other than the draw's are
    refused."""
    *_, words = served
    params, public, _, _ = _window(words)
    rec = _register(params, public)
    _, receipt = SC.receive(params, rec)
    spec = SC.spec_of(params, public["level"])
    d = S.select(spec, rec, receipt, _sample([20, 3, 25, 7, 11]), tmp_path / "draw")
    units = [_unit(t) for t in sorted(d["units"])]
    o, audit = S.outcome(spec, rec, receipt, d, units, 0.1, tmp_path / "ok", {})
    assert o.accepted and audit["accepted"] and audit["proved_units"] == sorted(d["units"])
    assert set(o.public) == {p[0] for p in Wd.PUBLIC} and o.public["verdict"] is True and o.public["drawn"] == d["units"]
    assert o.public["work"] == public["items"]["work"] and o.public["root_W"] == public["items"]["root_W"]
    with pytest.raises(S.Refused, match="once"):
        S.outcome(spec, rec, receipt, d, units, 0.1, tmp_path / "ok", {})
    bad = [_unit(t, lean=t != units[1]["unit"]) for t in sorted(d["units"])]
    o, audit = S.outcome(spec, rec, receipt, d, bad, 0.1, tmp_path / "lean", {})
    assert not o.accepted and o.public["verdict"] is False and audit["verdicts"]["lean"]["units"][1] is False
    o, _ = S.outcome(spec, rec, receipt, d, [_unit(t, opening=False) for t in sorted(d["units"])], 0.1, tmp_path / "op", {})
    assert not o.accepted
    with pytest.raises(S.Refused, match="draw"):
        S.outcome(spec, rec, receipt, d, units[:-1], 0.1, tmp_path / "short", {})
    with pytest.raises(S.Refused, match="declared"):
        S.outcome(spec, rec, receipt, d, units, 0.1, tmp_path / "more", {"shapes": []})


def test_lean_and_rust_verdicts_are_read_from_their_outputs(tmp_path):
    (tmp_path / "lean.out").write_text('setup: 3 ms\nVERDICT {"accepted":false,"why":"ring-switch claim 2 mismatch"}\n')
    v = SC.lean_verdict(tmp_path / "lean.out")
    assert v["of_record"] and v["accepted"] is False and "ring-switch" in v["why"] and len(v["digest"]) == 128
    (tmp_path / "none.out").write_text("")
    assert SC.lean_verdict(tmp_path / "none.out")["accepted"] is None
    (tmp_path / "prove.out").write_text('LIVE\t{"verdict": {"accepted": true, "why": ""}}\n')
    assert SC.rust_verdict(tmp_path / "prove.out")["accepted"] is True and not SC.rust_verdict(tmp_path / "prove.out")["of_record"]


def test_the_cli_runs_the_window_through_the_service(served, tmp_path):
    """The job's steps through `main`: register (and the refused extra item), receive (and the refused edited record),
    select by a stand-in for Lean's draw (and the refused second select), the opening checks and the outcome."""
    *_, words = served
    params, public, advice, priv = _window(words)
    (tmp_path / "params.json").write_text(json.dumps(params))
    c = tmp_path / "commit"
    c.mkdir()
    (c / "commitment.json").write_text(json.dumps(public))
    p, t = str(tmp_path / "params.json"), str(tmp_path)
    assert SC.main(["register", str(c), p, f"{t}/registration.json"]) == 0
    assert SC.main(["register", str(c), p, f"{t}/x.json", "--extra", "shapes=[[16,128,4]]", "--expect-refused"]) == 0
    assert SC.main(["receive", p, f"{t}/registration.json", f"{t}/recv"]) == 0
    assert SC.main(["edit", f"{t}/registration.json", f"{t}/extra.json", 'window.public.calls=2']) == 0
    assert SC.main(["receive", p, f"{t}/extra.json", f"{t}/recv-extra", "--expect-refused"]) == 0
    lean = tmp_path / "lean"
    lean.write_text(f"#!{sys.executable}\nimport json, sys\na = sys.argv\np, k = int(a[a.index('--population') + 1]), "
                    "int(a[a.index('--k') + 1])\nprint('DRAW ' + json.dumps({'law': 'subset', 'population': p, 'k': k, "
                    "'units': list(range(p - k, p))}))\n")
    lean.chmod(lean.stat().st_mode | stat.S_IEXEC)
    sel = ["select", p, f"{t}/registration.json", f"{t}/recv/receipt.json", f"{t}/draw.json", "--lean", str(lean), "--log", f"{t}/log"]
    assert SC.main(sel) == 0
    assert SC.main([*sel, "--expect-refused"]) == 0
    d = json.loads((tmp_path / "draw.json").read_text())
    assert d["units"][-1] == 25                      # the draw holds the filler's tiles 24 and 25
    units = []
    for u in d["units"]:
        (tmp_path / f"pub{u}.bin").write_bytes(_unit_pub(advice, priv, u))
        (tmp_path / f"op{u}.json").write_text(json.dumps(SC.opening(advice, priv, params, u)))
        assert SC.main(["check", p, f"{t}/registration.json", f"{t}/draw.json", str(u), f"{t}/pub{u}.bin", f"{t}/op{u}.json",
                        "--out", f"{t}/open{u}.json"]) == 0
        (tmp_path / f"lean{u}.out").write_text('VERDICT {"accepted":true,"why":""}\n')
        (tmp_path / f"stage{u}.json").write_text(json.dumps({"definition": "d", "circuit_sha512": "1", "class_sha512": "2",
                                                             "ands": 1, "k_log": 1, "binding": "hash scratch"}))
        units.append({"unit": u, "stage": f"{t}/stage{u}.json", "pub": f"{t}/pub{u}.bin", "lean": f"{t}/lean{u}.out",
                      "opening": f"{t}/open{u}.json"})
    (tmp_path / "units.json").write_text(json.dumps(units))
    out = ["outcome", p, f"{t}/registration.json", f"{t}/recv/receipt.json", f"{t}/draw.json", f"{t}/units.json"]
    assert SC.main([*out, f"{t}/outcome", "--log", f"{t}/olog"]) == 0
    o = json.loads((tmp_path / "outcome" / "outcome.json").read_text())
    assert o["accepted"] and set(o["public"]) == {x[0] for x in Wd.PUBLIC}
    assert SC.main([*out, f"{t}/outcome2", "--log", f"{t}/olog", "--expect-refused"]) == 0
    assert not os.path.exists(tmp_path / "outcome2")
    assert Fraction(o["public"]["sampling"]["delta"]) == Fraction(1, 10)


def _staged(d: Path, rec: dict, proof: int) -> Path:
    s = d / "sessions" / "l0000-1"
    s.mkdir(parents=True)
    (d / "stage.json").write_text(json.dumps({"circuit_sha512": "11", "class_sha512": "22", "ands": 7, "k_log": 3}))
    (d / "pub.bin").write_bytes(b"h\n" + bytes(384))
    (s / "session.json").write_text(json.dumps(rec))
    (s / "verifier.json").write_text(json.dumps({"host": str(d)}))
    (s / "t.rep0.bin").write_bytes(bytes(proof))
    return d


def test_shape_compares_what_the_prover_sends(tmp_path):
    """Two units' proofs have one shape when the circuit, the public file, each proof file's size and the session record's
    skeleton agree; the verifier's own verdict and timings, and its own files, are not the prover's."""
    rec = {"streams": [{"rounds": [{"k": 0, "msg_len": 96, "coins": ["ab" * 16]}]}], "verdict": {"verify_s": 1.5}, "g": 0.25}
    a = _staged(tmp_path / "a", rec, 100)
    b = _staged(tmp_path / "b", {**rec, "verdict": {"verify_s": 12.25, "why": "x"}, "g": 3.125}, 100)
    assert SC.shape(a, b)["same"]
    c = _staged(tmp_path / "c", {**rec, "streams": [{"rounds": [{"k": 0, "msg_len": 97, "coins": ["ab" * 16]}]}]}, 100)
    v = SC.shape(a, c)
    assert not v["same"] and any("msg_len" in x for x in v["diffs"])
    assert not SC.shape(a, _staged(tmp_path / "d", rec, 101))["same"]
