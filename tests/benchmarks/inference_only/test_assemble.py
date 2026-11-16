"""`assemble.py`: a tied serve's `out/registered/`, limits and count assembled after the serve, on made serves. A made
serve is the record check's own shaped case laid out as a serve under the Pearl+ tie writes it: its log's first record
the deployment's registration, its second W and its last the window's registration R. A number printed on one is a
test of the step, not a result."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import assemble as A
import built as B
import pytest
import record as R
import report
from test_served_record import DEV_WHY, _run, case, made, no_out, stand_in  # noqa: F401

SCHEME = "pearl-c-made"
PHI = {"model": B.MODEL, "revision": "made", "execution": "bi-eager", "target": {}}
DAY = "2000-01-01"


def tied(case: dict, where: Path, *, accept: bool = True, limits: bool = False) -> dict:
    """``case`` as a tied serve's files under ``where``: what serve.sh writes (the circuit record, φ, W, the policy at
    `SCHEME`, the verdicts and its own `registered/`, W alone) and no Program or count, and no limits unless ``limits``
    (the case's own, as a serve that fixed them before it served writes them); and the files a caller names, the first
    record and R."""
    made(case, where, accept=accept)
    out, inst, log = where / "out", case["instance"], case["trusted"]["log"]
    for p in [*([] if limits else [out / A.LIMITS]), *out.glob("block-counts-*.json")]:
        p.unlink()
    policy = json.loads((out / A.SERVED["policy"]).read_text())
    (out / A.SERVED["policy"]).write_text(json.dumps({**policy, "pearl_tables": SCHEME, "written_utc": f"{DAY}T00:00:00Z"}))
    (out / A.SERVED["circuit"]).write_text(inst["record"])
    (out / A.SERVED["phi"]).write_text(json.dumps(PHI))
    (out / A.SERVED["window"]).write_text(json.dumps(log[1]))
    (out / "registered").mkdir()
    (out / R.REGISTERED["circuit"]).write_text(inst["record"])
    (out / R.REGISTERED["log"]).write_text(json.dumps(log[1]) + "\n")
    (where / "first.json").write_text(json.dumps(log[0]))
    (where / "R.json").write_text(json.dumps(log[2]))
    return {"first": [where / "first.json"], "registration": where / "R.json"}


def counts(case: dict, calls: list):
    """The count, standing in for the integration's: the case's count limits, the policy's c, k and n."""
    def counted(circuit, roots, pins, policy, day, out, private):
        calls.append({"circuit": circuit, "roots": roots, "pins": pins, "policy": policy, "day": day, "out": out,
                      "private": private})
        out.write_text(json.dumps(case["count"]))
    return counted


def _assemble(case: dict, where: Path, names: dict, *, build=None, calls: list | None = None, **kw) -> list[str]:
    seen = []

    def built(phi, cls, scheme):
        seen.append((phi, cls, scheme))
        return case["instance"]["program"]

    said = A.assemble(where, **{**names, **kw}, build=build or built, counted=counts(case, calls if calls is not None else []))
    if build is None:
        assert seen == [(PHI, (2, 2, 2), SCHEME)]
    return said


def test_a_made_tied_serve_assembled_holds_the_files_the_check_reads_and_its_accept_gets_a_number(
        case, tmp_path, stand_in, capsys) -> None:
    calls: list = []
    said = _assemble(case, tmp_path, tied(case, tmp_path, limits=True), calls=calls)
    out, inst, log = tmp_path / "out", case["instance"], case["trusted"]["log"]
    assert (out / R.REGISTERED["program"]).read_text() == inst["program"]
    assert (out / R.REGISTERED["circuit"]).read_text() == inst["record"]
    assert [json.loads(x) for x in (out / R.REGISTERED["log"]).read_text().splitlines()] == log
    assert "R chained right after W" in said[2] and A.NO_REGISTRATION not in "\n".join(said)
    assert calls == [{"circuit": out / R.REGISTERED["circuit"], "roots": tmp_path / "R.json", "pins": None,
                      "policy": out / A.SERVED["policy"], "day": DAY, "out": out / "block-counts-2-2-2.json",
                      "private": tmp_path / "private" / "block-counts-2-2-2.json"}]
    capsys.readouterr()
    code, doc = _run(["--from", str(tmp_path)], stand_in)
    assert (code, doc["record"]["block"]["value"] is not None, doc["record"]["no_number"]) == (0, True, None)
    assert {x: v["complies"] for x, v in doc["record"]["check"].items()} == {"check": True, "count": True}


def test_a_development_accept_assembled_still_has_no_number(case, tmp_path, stand_in, capsys) -> None:
    _assemble(case, tmp_path, tied(case, tmp_path, accept=False, limits=True))
    capsys.readouterr()
    code, doc = _run(["--from", str(tmp_path)], stand_in)
    assert (code, doc["value"], doc["record"]["no_number"]) == (1, None, DEV_WHY)


def test_the_limits_assemble_fixes_from_the_policy_are_judged_and_refused_on_the_made_case(
        case, tmp_path, stand_in, capsys) -> None:
    _assemble(case, tmp_path, tied(case, tmp_path))
    capsys.readouterr()
    code, doc = _run(["--from", str(tmp_path)], stand_in)
    assert (code, doc["value"], doc["record"]["no_number"]) == (1, None, "the check refused the record (the rate rule)")


def test_without_R_the_log_ends_at_W_so_an_accept_tied_to_W_is_refused_and_the_check_refuses_the_rest(
        case, tmp_path, stand_in, capsys) -> None:
    names = tied(case, tmp_path, limits=True)
    said = _assemble(case, tmp_path, {"first": names["first"]})
    log = case["trusted"]["log"]
    assert [json.loads(x) for x in (tmp_path / "out" / R.REGISTERED["log"]).read_text().splitlines()] == log[:2]
    assert A.NO_REGISTRATION in said[2]
    capsys.readouterr()
    with pytest.raises(SystemExit):
        _run(["--from", str(tmp_path)], stand_in)
    assert "the verdict is about another record than the check's" in capsys.readouterr().err
    dev = tmp_path / "dev"
    _assemble(case, dev, {"first": tied(case, dev, accept=False, limits=True)["first"]})
    capsys.readouterr()
    code, doc = _run(["--from", str(dev)], stand_in)
    assert (code, doc["value"]) == (1, None) and doc["record"]["no_number"].startswith("the check refused the record")


def test_a_program_rebuilt_to_another_digest_is_refused_before_anything_is_written(case, tmp_path) -> None:
    names = tied(case, tmp_path)
    other = json.dumps({**json.loads(case["instance"]["program"]), "made": "another"}, sort_keys=True)
    with pytest.raises(A.Refused, match="the Program rebuilt from the serve's φ, class 2,2,2 and pearl-c-made has digest"):
        _assemble(case, tmp_path, names, build=lambda phi, cls, scheme: other)
    out = tmp_path / "out"
    assert not (out / R.REGISTERED["program"]).exists() and not (out / A.LIMITS).exists()
    assert (out / R.REGISTERED["log"]).read_text() == json.dumps(case["trusted"]["log"][1]) + "\n"


@pytest.mark.parametrize("change, match", [
    (lambda r: {**r, "program": "ab" * 64}, "R names Program abababababababab..., not the circuit record's"),
    (lambda r: {**r, "window": {**r["window"], "circuit": "ab" * 64}}, "R's window.circuit is not the SHA-512"),
    (lambda r: {**r, "previous": "ab" * 64}, "the log's record 2 follows abababababababab..., not"),
])
def test_an_R_that_doesnt_name_the_program_the_record_or_W_is_refused(case, tmp_path, change, match) -> None:
    names = tied(case, tmp_path)
    names["registration"].write_text(json.dumps(change(json.loads(names["registration"].read_text()))))
    with pytest.raises(A.Refused, match=match):
        _assemble(case, tmp_path, names)
    assert not (tmp_path / "out" / R.REGISTERED["program"]).exists()


def test_a_W_chained_after_another_record_than_the_first_given_is_refused(case, tmp_path) -> None:
    names = tied(case, tmp_path)
    with pytest.raises(A.Refused, match="the log's record 0 follows"):
        _assemble(case, tmp_path, {"registration": names["registration"]})


def test_a_registered_circuit_record_other_than_the_serves_is_refused(case, tmp_path) -> None:
    names = tied(case, tmp_path)
    (tmp_path / "out" / R.REGISTERED["circuit"]).write_text(case["instance"]["record"] + " ")
    with pytest.raises(A.Refused, match="is not the serve's circuit record"):
        _assemble(case, tmp_path, names)


def test_a_serve_that_wrote_no_window_record_is_refused_by_name(case, tmp_path) -> None:
    names = tied(case, tmp_path)
    (tmp_path / "out" / A.SERVED["window"]).unlink()
    with pytest.raises(A.Refused, match=r"the serve wrote no window \("):
        _assemble(case, tmp_path, names)


def test_the_limits_are_the_policys_own_and_limits_already_written_are_kept(case, tmp_path) -> None:
    names = tied(case, tmp_path)
    _assemble(case, tmp_path, names)
    out = tmp_path / "out"
    policy = (out / A.SERVED["policy"]).read_bytes()
    doc = json.loads((out / A.LIMITS).read_text())
    c, k = json.loads(policy)["c"], json.loads(policy)["k"]
    assert (doc["alpha"][0] * c, doc["alpha"][1] * k, doc["gamma"]) == (k * doc["alpha"][1], c * doc["alpha"][0], c)
    assert doc["policy_sha512"] == hashlib.sha512(policy).hexdigest() and doc["format"] == A.LIMITS_FORMAT
    (out / A.LIMITS).write_text('{"alpha": [1, 2], "gamma": 3}')
    _assemble(case, tmp_path, names)
    assert json.loads((out / A.LIMITS).read_text()) == {"alpha": [1, 2], "gamma": 3}


def test_main_exits_2_with_the_refusal(case, tmp_path, capsys) -> None:
    tied(case, tmp_path)
    (tmp_path / "out" / A.SERVED["phi"]).unlink()
    assert A.main(["--from", str(tmp_path)]) == 2
    assert "assemble: refused: the serve wrote no phi" in capsys.readouterr().err


def test_the_report_reads_what_assemble_wrote_by_record_REGISTERED() -> None:
    assert set(R.REGISTERED) == {"program", "circuit", "log"} and report.R is R
