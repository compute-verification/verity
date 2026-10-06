"""served_zk's window through the proof service (`served_zk/served_commit.py` over P1, `sampled_proofs.service`): a
window of served calls' fp8-p rows committed as C-Flock commits a unit's rows, its tiles padded with filler tiles to a
grid level, registered with exactly PoUW's public items, received by a `Stream`, drawn once, each drawn tile's proof
opened against the window root, delivered once, and the final outcome publishing exactly the public items."""
from __future__ import annotations

import hashlib
import io
import json
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
from verity.protocols.verification.sampled_proofs import service as S
from verity.protocols.verification.sampled_proofs.one_stage import registration as R

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "served_zk"))
import served_commit as SC  # noqa: E402

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
    return SC.register(params, public, extra)


def _codes(e: pytest.ExceptionInfo) -> set[str]:
    return set(e.value.verdict.codes)


SESSION = "5e" * 64


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
    return lambda law, n: list(units)[:law["k"]]


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
    assert set(rec["window"]["public"]) == set(Wd.REGISTERED) and rec["population"] == 26 and rec["scheme"] == S.SCHEME
    assert rec["law"] == {"law": "subset", "k": rec["window"]["public"]["sampling"]["law"]["k"]}
    assert [r["port"] for r in rec["roots"]] == [SC.PORT, "W"]
    assert S.check_public(SC.spec_of(params, public["level"]), rec).ok and SC.public_problem(params, rec).ok
    text = json.dumps(rec)
    for leak in ('"matmul"', '"shape"', '"real_rows"', '"rows"', '"keep"'):
        assert leak not in text
    with pytest.raises(S.Refused) as e:
        _register(params, public, shapes=[[16, K, N]])
    assert _codes(e) == {"public-items"}
    with pytest.raises(S.Refused) as e:
        _register(params, public, root_W="00" * 64)
    assert _codes(e) == {"public-value"}
    undeclared = SC.spec_of(params, public["level"])
    undeclared = S.Spec(undeclared.values, undeclared.selection, undeclared.check, undeclared.public[:2], undeclared.context)
    assert set(S.check_public(undeclared, rec).detail["undeclared"]) == {f"leaves:{SC.PORT}", f"domain:{SC.PORT}", "leaves:W",
                                                                          "domain:W"}
    with pytest.raises(ValueError, match="outside"):
        SC.work_at({**params, "range": [100, 1 << 40]}, public["level"])


def _challenger(params, units=None, coins=b"\x07") -> SC.Challenger:
    return SC.Challenger(params, None if units is None else _sample(units), coins=lambda n: coins * n)


def test_the_challenger_holds_every_item_to_its_own(served):
    """The challenger's stream registers the honest record and its receipt names it.  An extra public key, a raised level
    (more credit), a K′ other than its own, another request's record (its own context) or a field one-stage doesn't define
    is refused, and so is a second registration while the first has no outcome."""
    *_, words = served
    params, public, _, _ = _window(words)
    rec = _register(params, public)
    ch = _challenger(params)

    def refused(r) -> set[str]:
        with pytest.raises(S.Refused) as e:
            ch.register("x", r)
        return _codes(e)

    extra = json.loads(json.dumps(rec))
    extra["window"]["public"]["shapes"] = [[16, K, N]]
    assert refused(extra) == {"public-items"}
    raised = json.loads(json.dumps(rec))
    raised["window"]["public"]["work"]["level"] += 1
    assert refused(raised) == {"public-value", "population"}
    k = json.loads(json.dumps(rec))
    k["window"]["public"]["sampling"]["law"]["k"] -= 1
    assert refused(k) == {"public-value"}
    oparams, opublic, _, _ = _window(words, run="r-other")
    assert "public-context" in refused(_register(oparams, opublic))
    assert refused({**rec, "tiles": [2, 3]}) == {"keys"}
    assert "x" not in ch.regs
    reg = ch.register("honest", rec)
    assert reg.receipt["registration"] == R.digest(rec) == reg.digest and reg.order == 0
    assert refused(rec) == {"open"}


def test_select_draws_once_after_the_receipt(served):
    """The draw is the law's subset:K′ over the padded population, from the challenger's own coins, once: a second select
    is refused, so no retry redraws; a sampler's draw of the wrong size is refused."""
    *_, words = served
    params, public, _, _ = _window(words)
    rec = _register(params, public)
    k = SC.law_of(params, public["level"])["k"]
    ch = _challenger(params)
    ch.register("honest", rec)
    d = ch.select("honest")
    assert len(d.units) == k and d.draw["population"] == 26 and d.units == sorted(d.units) and all(0 <= u < 26 for u in d.units)
    with pytest.raises(S.Refused) as e:
        ch.select("honest")
    assert _codes(e) == {"drawn"}
    again = _challenger(params)
    again.register("honest", rec)
    assert again.select("honest").units == d.units
    bad = SC.Challenger(params, lambda law, n: list(range(law["k"] + 1)))
    bad.register("honest", rec)
    with pytest.raises(S.Refused):
        bad.select("honest")


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


def test_the_outcome_publishes_exactly_the_public_items(served):
    """The outcome over every drawn unit accepts only when each unit's Lean verdict and opening check accept; PoUW's public
    items of it are exactly `window.PUBLIC`; it waits for the proof's delivery, is final, and units other than the draw's
    are refused.  A control stream replays the honest draw: a reject there is final, and its failure budget stops it."""
    *_, words = served
    params, public, _, _ = _window(words)
    rec = _register(params, public)
    ch = _challenger(params, [20, 3, 25, 7, 11])
    ch.register("honest", rec)
    d = ch.select("honest")
    assert d.units == [3, 7, 20, 25]
    units = [_unit(t) for t in d.units]
    with pytest.raises(S.Pending):
        ch.outcome("honest", units)
    assert ch.deliver("honest", [SESSION]) is None
    with pytest.raises(S.Refused) as e:
        ch.outcome("honest", units[:-1])
    assert _codes(e) == {"units"}
    o, pub, body = ch.outcome("honest", units)
    assert o.accepted and o.profile is not None and body["audit"]["proved_units"] == d.units and body["sessions"] == [SESSION]
    assert set(pub) == {p[0] for p in Wd.PUBLIC} and pub["verdict"] is True and pub["drawn"] == d.units
    assert pub["work"] == public["items"]["work"] and pub["root_W"] == public["items"]["root_W"]
    assert body["audit"]["verdicts"]["lean"]["sessions"] == [SESSION]
    for again in (lambda: ch.outcome("honest", units), lambda: ch.deliver("honest", [SESSION])):
        with pytest.raises(S.Refused) as e:
            again()
        assert _codes(e) == {"final"}
    for i, bad in enumerate(([_unit(t, lean=t != 7) for t in d.units], [_unit(t, opening=False) for t in d.units])):
        name = f"ctl-{i}"
        ch.register(name, rec, stream=name, replay="honest")
        assert ch.select(name).units == d.units
        ch.deliver(name, [SESSION])
        o, pub, body = ch.outcome(name, bad)
        assert not o.accepted and o.verdict.codes == ("reject",) and pub["verdict"] is False and o.profile is None
        with pytest.raises(S.Refused) as e:
            ch.register(f"{name}-retry", rec, stream=name)
        assert _codes(e) == {"stopped"}
    assert body["audit"]["verdicts"]["opening"]["units"] == [False] * 4


def test_lean_and_rust_verdicts_are_read_from_their_outputs(tmp_path):
    (tmp_path / "lean.out").write_text('setup: 3 ms\nVERDICT {"accepted":false,"why":"ring-switch claim 2 mismatch"}\n')
    v = SC.lean_verdict(tmp_path / "lean.out")
    assert v["of_record"] and v["accepted"] is False and "ring-switch" in v["why"] and len(v["digest"]) == 128
    (tmp_path / "none.out").write_text("")
    assert SC.lean_verdict(tmp_path / "none.out")["accepted"] is None
    (tmp_path / "prove.out").write_text('LIVE\t{"verdict": {"accepted": true, "why": ""}}\n')
    assert SC.rust_verdict(tmp_path / "prove.out")["accepted"] is True and not SC.rust_verdict(tmp_path / "prove.out")["of_record"]


def test_the_cli_runs_the_window_through_the_service(served, tmp_path, monkeypatch, capsys):
    """The job's steps through `main`: register (and the refused extra item), then one challenger process's requests,
    as the job sends them a line at a time: the refused edited record, the honest one, Lean's draw (a stand-in) and the
    refused second select, the opening checks, the delivery, the outcome and the refused second one."""
    *_, words = served
    params, public, advice, priv = _window(words)
    (tmp_path / "params.json").write_text(json.dumps(params))
    c = tmp_path / "commit"
    c.mkdir()
    (c / "commitment.json").write_text(json.dumps(public))
    p, t = str(tmp_path / "params.json"), str(tmp_path)
    assert SC.main(["register", str(c), p, f"{t}/registration.json"]) == 0
    assert SC.main(["register", str(c), p, f"{t}/x.json", "--extra", "shapes=[[16,128,4]]", "--expect-refused"]) == 0
    assert SC.main(["edit", f"{t}/registration.json", f"{t}/extra.json", 'window.public.calls=2']) == 0
    lean = tmp_path / "lean"
    lean.write_text(f"#!{sys.executable}\nimport json, sys\na = sys.argv\np, k = int(a[a.index('--population') + 1]), "
                    "int(a[a.index('--k') + 1])\nprint('DRAW ' + json.dumps({'law': 'subset', 'population': p, 'k': k, "
                    "'units': list(range(p - k, p))}))\n")
    lean.chmod(lean.stat().st_mode | stat.S_IEXEC)
    k = SC.law_of(params, public["level"])["k"]
    drawn = list(range(26 - k, 26))                  # the stand-in's draw holds the filler's tiles 24 and 25
    rec = json.loads((tmp_path / "registration.json").read_text())
    rows = []
    for u in drawn:
        (tmp_path / f"pub{u}.bin").write_bytes(_unit_pub(advice, priv, u))
        (tmp_path / f"op{u}.json").write_text(json.dumps(SC.opening(advice, priv, params, u)))
        assert SC.check(params, rec, {"units": drawn}, u, _unit_pub(advice, priv, u), SC.opening(advice, priv, params, u)) is None
        (tmp_path / f"open{u}.json").write_text(json.dumps({"of_record": False, "accepted": True, "why": ""}))
        (tmp_path / f"lean{u}.out").write_text('VERDICT {"accepted":true,"why":""}\n')
        (tmp_path / f"stage{u}.json").write_text(json.dumps({"definition": "d", "circuit_sha512": "1", "class_sha512": "2",
                                                             "ands": 1, "k_log": 1, "binding": "hash scratch"}))
        (tmp_path / f"session{u}.json").write_text(json.dumps({"unit": u}))
        rows.append({"unit": u, "stage": f"{t}/stage{u}.json", "pub": f"{t}/pub{u}.bin", "lean": f"{t}/lean{u}.out",
                     "opening": f"{t}/open{u}.json", "session": f"{t}/session{u}.json"})
    (tmp_path / "units.json").write_text(json.dumps(rows))
    reqs = [{"op": "register", "name": "extra", "registration": f"{t}/extra.json"},
            {"op": "register", "name": "honest", "registration": f"{t}/registration.json", "out": f"{t}/receipt.json"},
            {"op": "select", "name": "honest", "out": f"{t}/draw.json"},
            {"op": "select", "name": "honest", "out": f"{t}/draw2.json"},
            {"op": "outcome", "name": "honest", "units": f"{t}/units.json"},
            {"op": "deliver", "name": "honest", "units": f"{t}/units.json"},
            {"op": "outcome", "name": "honest", "units": f"{t}/units.json", "out": f"{t}/outcome"},
            {"op": "outcome", "name": "honest", "units": f"{t}/units.json", "out": f"{t}/outcome2"},
            {"op": "quit"}, {"op": "select", "name": "honest"}]
    monkeypatch.setattr(sys, "stdin", io.StringIO("".join(json.dumps(r) + "\n" for r in reqs)))
    capsys.readouterr()
    assert SC.main(["challenger", p, f"{t}/log", "--lean", str(lean)]) == 0
    words_ = [x.split(" ", 1)[0] for x in capsys.readouterr().out.splitlines()]
    assert words_ == ["REFUSED", "REGISTERED", "DRAWN", "REFUSED", "PENDING", "DELIVERED", "ACCEPT", "REFUSED"]
    d = json.loads((tmp_path / "draw.json").read_text())
    assert d["units"] == drawn and json.loads((tmp_path / "receipt.json").read_text())["registration"] == R.digest(rec)
    assert not (tmp_path / "draw2.json").exists() and not (tmp_path / "outcome2").exists()
    o = json.loads((tmp_path / "outcome" / "outcome.json").read_text())
    assert o["accepted"] and set(o["public"]) == {x[0] for x in Wd.PUBLIC} and o["public"]["drawn"] == drawn
    assert Fraction(o["public"]["sampling"]["delta"]) == Fraction(1, 10)
    audit = json.loads((tmp_path / "outcome" / "audit.json").read_text())
    assert audit["sessions"] == [hashlib.sha512((tmp_path / f"session{u}.json").read_bytes()).hexdigest() for u in drawn]
    assert len((tmp_path / "log" / "requests.jsonl").read_text().splitlines()) == 8


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
