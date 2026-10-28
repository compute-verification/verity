"""certifier_piece: the front sends one group of the window circuit's class through the gate's scan rehearsal and keeps
the gate's fetch, witness and all, 0700 under the serve dir's private/certifier, publishing only the bundle less witness/
and its outcome. The piece, on a serve dir whose window was the warden's synthetic run's certified requests (the run
`test_warden_chain.gate_run` publishes in the gate's layout), runs as `served_verdicts.py pieces` runs it and writes
`network-period`'s answer only when the chain accepted and the window's requests are certified ones in the scan's
link-window: the period mirror's certifier/v0 accept (`period_case`) then goes through `served_verdicts.py record` into
the block record's certifier leg, which `block_record.check` holds and its schema admits. A window request the warden
didn't certify is rejected before any chain runs; a chain that refused, requests outside the scan's link-window and a
period the check rejects are rejects; a step that can't run (no kept run, no report, no build, the time limit, no
challenger, coins not the challenger's) refuses, which leaves the certifier missing. The leak guard reads the piece's
DIR, each drawing round's challenger evidence included, against the run's witness, and leaves the evidence."""
from __future__ import annotations

import importlib.util
import json
import os
import shlex
import sys
import tarfile
from pathlib import Path

import jsonschema
import pytest

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(Path(__file__).resolve().parent), str(ROOT / "benchmarks" / "one_stage")]

import certifier_piece as CP  # noqa: E402
import served_verdicts as S  # noqa: E402
from test_served_session import RUN, STUB  # noqa: E402
from test_warden_chain import WC, gate_run  # noqa: E402
from verity_e2e_demo import block_record as BR  # noqa: E402

PERIOD_CASE = ROOT / "verity" / "core" / "protocols" / "network" / "tests" / "period_case.py"
SCHEMA = json.loads((Path(BR.__file__).parent / "schemas" / "block-record.v1.schema.json").read_text())
K = 3


@pytest.fixture(scope="module")
def period():
    spec = importlib.util.spec_from_file_location("period_case", PERIOD_CASE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["period_case"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def warden(tmp_path_factory):
    """The synthetic warden run, its fetch, and the link-window holding the most whole certified requests."""
    d = tmp_path_factory.mktemp("warden")
    run = gate_run(d)
    io, T = WC.io(run), WC.deployment(run)[1].window
    by: dict[int, list[dict]] = {}
    for r in io["records"]:
        ws = {b // T for b in [r["release"], *(b for _, b in r["frames"])]}
        if len(ws) == 1:
            by.setdefault(ws.pop(), []).append(r)
    w = max(by, key=lambda x: (len(by[x]), -x))
    fetch = d / "fetch.tar.gz"
    with tarfile.open(fetch, "w:gz") as tar:
        tar.add(run, arcname=RUN)
    return {"fetch": fetch, "w": w, "records": by[w][:4], "others": [r for x, rs in by.items() if x != w for r in rs],
            "windows": WC.deployment(run)[0]["windows"]}


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _serve_dir(d: Path, warden: dict, records: list[dict] | None = None, kept: bool = True) -> Path:
    """A serve dir whose window K served ``records`` (the warden's, by default those of its link-window) and whose
    gate fetch the front kept."""
    records = warden["records"] if records is None else records
    d.mkdir(parents=True)
    reqs = [{"engine_id": f"e{i}", "user": r["user"], "prompt_ids": r["prompt"], "streamed_ids": r["output"],
             "windows": [K], "check": {"passed": True}} for i, r in enumerate(records)]
    _write(d / "requests.jsonl", reqs)
    _write(d / "seals.jsonl", [{"window": K, "requests": [r["engine_id"] for r in reqs]}])
    _write(d / "windows.jsonl", [{"window": K, "passed": True, "commit": {"steps": 9}}])
    _write(d / "advice_trace.jsonl", [{"step": 0}])
    (d / "serve.json").write_text("{}")
    if kept:
        (d / CP.PRIVATE / RUN).mkdir(parents=True)
        (d / CP.PRIVATE / RUN / "fetch.tar.gz").write_bytes(warden["fetch"].read_bytes())
    return d


#: a drawing round's challenger evidence as the chain leaves it in out/ (`round_coins.Session`), which DIR publishes
EVIDENCE = ("scan/honest/challenger/session.json", "scan/honest/challenger/pins.jsonl", "scan/honest/challenger/coins.json",
            "n2/honest/challenger/coins.json", "n2/honest/verifier/period-certificates.json")


def _chain_out(d: Path, verdict: str, w: int, coins: dict | None = None) -> Path:
    """What the chain leaves in its run dir's out/: its report, its coins step `coins` (accepted on the challenger
    ``local`` by default), each drawing round's challenger evidence and the scan's one link-window."""
    out = d / "out"
    (out / "report").mkdir(parents=True)
    (out / "scan" / "joint").mkdir(parents=True)
    coins = {"verdict": "accepted", "challenger": "local", "why": None} if coins is None else coins
    (out / "report" / "report.json").write_text(json.dumps({"verdict": verdict, "why": None if verdict == "accepted"
                                                            else {"n2": "a window refused"},
                                                            "steps": {"coins": coins} if coins else {}}))
    (out / "scan" / "answers.json").write_text(json.dumps({"registration": {"program": "ab" * 64}, "windows": [
        {"window": w, "answer": {"verdict": "accept"}, "registrations": []}]}))
    (out / "scan" / "joint" / "program.json").write_text("{}")
    for rel in EVIDENCE:
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        (out / rel).write_text(json.dumps({"evidence": rel}) + "\n")
    return d


def _piece_cmd(tmp_path: Path, chain: Path, answer: dict, cmd: str | None = None, lean: Path | None = None,
               coins: str = "--challenger local") -> str:
    if lean is None:
        lean = tmp_path / "network-period"
        (tmp_path / "lean-answer.json").write_text(json.dumps(answer, sort_keys=True))
        lean.write_text(f"#!/bin/sh\ncat {shlex.quote(str(tmp_path / 'lean-answer.json'))}\n")
        lean.chmod(0o755)
    cmd = cmd or f'mkdir -p "$RESEARCH_RUN_DIR" && cp -r {shlex.quote(str(chain / "out"))} "$RESEARCH_RUN_DIR/"'
    return (f"{shlex.quote(sys.executable)} {shlex.quote(str(HERE / 'certifier_piece.py'))} piece \"$@\" "
            f"--lean {shlex.quote(str(lean))} --chain-cmd {shlex.quote(cmd)} {coins}")


def _integrity() -> dict:
    a = {"protocol": "verity/sampled-proofs/one-stage/v1", "verdict": "accept", "reason": None, "commitment": "ab" * 64,
         "public": {"bound": 2, "drawn": [0, 3], "per_template": None, "proved": [0, 3]},
         "profile": {"levels": [{"name": "unit", "size": 4}], "drawn": {"unit": 2}, "delta": "1/100"}}
    return S.row_of(S.INTEGRITY_OWNER, a, Path("/nonexistent"))


def _record(monkeypatch, capsys, serve: Path, pieces: Path) -> tuple[dict | None, dict]:
    """`served_verdicts.py record` over the pieces `pieces` ran, sampled proofs' row an accept: the block record (None
    where none was written) and the ``RECORD`` line."""
    monkeypatch.setattr(S, "sampled_proofs_row", lambda d, run=None: _integrity())
    monkeypatch.setenv("RESEARCH_RUN_ID", "r1")
    out = pieces.parent / "record"
    capsys.readouterr()
    S.main(["record", "--serve-run", str(serve), "--window", str(K), "--sampled-proofs", str(out / "sampled-proofs"),
            "--pieces", str(pieces), "--out", str(out)])
    (line,) = [x.removeprefix("RECORD ") for x in capsys.readouterr().out.splitlines() if x.startswith("RECORD ")]
    rec = out / "block-record.json"
    return (json.loads(rec.read_text()) if rec.exists() else None), json.loads(line)


def test_the_pieces_certifier_row_is_the_period_checks_accept_and_composes_into_the_block_record(tmp_path, warden,
                                                                                                period, capsys,
                                                                                                monkeypatch):
    accept = period.reference(period.case("accept", 1, "clock"))
    assert accept["verdict"] == "accept" and accept["protocol"] == CP.PROTOCOL
    d = _serve_dir(tmp_path / "serve", warden)
    env = dict(os.environ, CERTIFIER_CMD=_piece_cmd(tmp_path, _chain_out(tmp_path / "chain", "accepted", warden["w"]),
                                                    accept))
    got = S.pieces(d, K, tmp_path / "pieces", env=env)
    piece = tmp_path / "pieces" / "certifier"
    assert (got["certifier"]["verdict"], got["certifier"]["rc"]) == ("accept", 0), (piece / "piece.log").read_text()
    assert "MISSING the network certifier" not in capsys.readouterr().out
    assert json.loads((piece / "answer.json").read_text()) == accept
    assert json.loads((piece / "chain" / "report" / "report.json").read_text())["verdict"] == "accepted"
    assert sorted(p.name for p in (piece / "period").iterdir()) == ["commitment.bin", "evidence.json", "params.json"]
    assert {p.name for p in piece.iterdir()} - {"piece.json"} == {"answer.json", "chain", "period", "piece.log"}
    assert all(json.loads((piece / "chain" / rel).read_text()) == {"evidence": rel} for rel in EVIDENCE)
    assert "leak guard (exit 0)" in (piece / "piece.log").read_text()
    keep = d / CP.PRIVATE / RUN
    assert sorted(p.name for p in keep.iterdir()) == ["chain", "endpoint", "fetch.tar.gz"]
    assert [(keep / "endpoint" / f).stat().st_mode & 0o777 for f in CP.ENDPOINT] == [0o600] * 4
    row = S.piece_row("communication", tmp_path / "pieces")
    assert (row["verdict"], row["quantity"], row["owner"]) == ("accept", accept["public"], "network-accounting")
    rec, line = _record(monkeypatch, capsys, d, tmp_path / "pieces")
    jsonschema.Draft202012Validator(SCHEMA).validate(rec)
    assert BR.check(rec) == {} and not line["check"]
    assert "certifier" not in line["record_missing"] and line["verdicts"]["the network certifier"] == "accept"
    leg = rec["legs"]["certifier"]
    assert leg["quantity"]["symbol"] == "K" and leg["quantity"]["value"] == int(accept["public"]["K"])


def test_a_window_request_the_warden_didnt_certify_is_rejected_before_any_chain(tmp_path, warden, capsys, monkeypatch):
    stranger = {"user": "not-the-wardens", "prompt": [1, 2], "output": [3]}
    d = _serve_dir(tmp_path / "serve", warden, [*warden["records"][:1], stranger])
    env = dict(os.environ, CERTIFIER_CMD=_piece_cmd(tmp_path, tmp_path / "none", {}))
    got = S.pieces(d, K, tmp_path / "pieces", env=env)
    assert got["certifier"]["verdict"] == "reject" and "is no certified request" in got["certifier"]["reason"]
    assert not (d / CP.PRIVATE / RUN / "chain").exists()
    rec, line = _record(monkeypatch, capsys, d, tmp_path / "pieces")
    assert rec is None and (line["composed"], line["by"]) == ("reject", ["the network certifier"])
    changed = dict(warden["records"][0], output=[*warden["records"][0]["output"], 7])
    d = _serve_dir(tmp_path / "changed", warden, [changed])
    a, _ = CP.decide(d, K, tmp_path / "out", chain=lambda *x: pytest.fail("no chain"), check=lambda *x: {})
    assert a["verdict"] == "reject" and "streamed_ids aren't the certified output" in a["reason"]


def _decide(tmp_path, warden, verdict="accepted", w=None, records=None, check=None):
    d = _serve_dir(tmp_path / "serve", warden, records)

    def chain(keep, run, ep, windows):
        assert sorted(p.name for p in ep.iterdir()) == sorted(CP.ENDPOINT) and windows == warden["windows"]
        _chain_out(keep / run / "chain", verdict, warden["w"] if w is None else w)
        return 0
    (tmp_path / "dir").mkdir()
    a, unpacked = CP.decide(d, K, tmp_path / "dir", chain=chain, check=check or (lambda c, p: pytest.fail("no check")),
                            case=lambda run, scan: {"params": {}, "commitment": "", "evidence": ""})
    assert unpacked is not None and unpacked.parent == d / CP.PRIVATE / RUN
    return a


def test_a_refused_chain_a_request_off_the_scans_link_window_and_a_rejected_period_are_rejects(tmp_path, warden):
    d = _serve_dir(tmp_path / "serve", warden)
    env = dict(os.environ, CERTIFIER_CMD=_piece_cmd(tmp_path, _chain_out(tmp_path / "chain", "refused", warden["w"]), {}))
    got = S.pieces(d, K, tmp_path / "pieces", env=env)
    assert got["certifier"]["verdict"] == "reject" and got["certifier"]["reason"].startswith("the chain refused")
    assert not (tmp_path / "pieces" / "certifier" / CP.REFUSED).exists()
    a = _decide(tmp_path / "refused", warden, "refused")
    assert a["verdict"] == "reject" and a["reason"].startswith("the chain refused: {\"n2\": \"a window refused\"}")
    assert a == CP.answer("reject", a["reason"]) and S.check_answer(a) == a
    a = _decide(tmp_path / "off", warden, w=warden["w"] + 1)
    assert a["verdict"] == "reject" and f"not the period's {warden['w'] + 1} alone" in a["reason"]
    if warden["others"]:
        a = _decide(tmp_path / "mixed", warden, records=[*warden["records"][:1], warden["others"][0]])
        assert a["verdict"] == "reject" and "spans link-windows" in a["reason"]
    rejected = CP.answer("reject", "window 0: certificates")
    assert _decide(tmp_path / "period", warden, check=lambda c, p: rejected) == rejected


REFUSED_COINS = {"verdict": "refused", "challenger": f"gate:{CP.SS.GATE}",
                 "why": "scan/honest: the challenger's session: the gate is unreachable"}


@pytest.mark.parametrize("case", ["no kept run", "corrupt fetch", "no report", "no build", "time limit",
                                  "foreign answer", "no challenger", "coins refused", "no coins step"])
def test_a_step_that_cant_run_refuses_and_leaves_the_certifier_missing(case, tmp_path, warden, capsys, monkeypatch):
    """The front's failures (the gate refusing or busy, its tunnel or ssh) keep no run; a corrupt fetch, a chain that
    ends without a report, a missing Lean build, SIGTERM from the piece's time limit and a check that answers no
    certifier/v0 verdict can't run: each leaves DIR/refused.txt and its log, no answer, and the certifier missing. So
    do the chain's coins when they aren't the challenger's: the gate named without its key record, or a report whose
    coins step isn't accepted (the gate unreachable, a session refused) or that has none, with nothing drawn on other
    coins in their place."""
    d = _serve_dir(tmp_path / "serve", warden, kept=case != "no kept run")
    if case == "corrupt fetch":
        (d / CP.PRIVATE / RUN / "fetch.tar.gz").write_bytes(b"not a tarball")
    coins = {"coins refused": REFUSED_COINS, "no coins step": {}}.get(case)
    chain = _chain_out(tmp_path / "chain", "accepted", warden["w"], coins)
    cmd = {"no report": "exit 7", "time limit": 'kill -TERM "$PPID"; while :; do :; done'}.get(case)
    lean = tmp_path / "no-such-network-period" if case == "no build" else None
    answer = {"verdict": "accept", "protocol": "other"} if case == "foreign answer" else {}
    flags = "" if case == "no challenger" else "--challenger local"
    env = dict(os.environ, CERTIFIER_CMD=_piece_cmd(tmp_path, chain, answer, cmd, lean, flags))
    S.pieces(d, K, tmp_path / "pieces", env=env)
    piece = tmp_path / "pieces" / "certifier"
    why = (piece / CP.REFUSED).read_text()
    assert {p.name for p in piece.iterdir()} - {"piece.json"} == {CP.REFUSED, "piece.log"}, why
    assert S.piece_row("communication", tmp_path / "pieces")["verdict"] == "missing"
    assert {"no kept run": "no warden run kept", "corrupt fetch": "fetch.tar.gz: ReadError", "no report": "the chain wrote no report (exit 7)",
            "no build": "network-period check: FileNotFoundError", "time limit": "SIGTERM: the piece's time limit",
            "foreign answer": "no verity/certifier/v0 verdict",
            "no challenger": "no challenger for the chain's coins: --challenger gate:… needs --challenger-key",
            "coins refused": f"the challenger's coins (gate:{CP.SS.GATE}): scan/honest: the challenger's session: the "
                             "gate is unreachable",
            "no coins step": "the challenger's coins (None): the report has no coins step"}[case] in why
    if case != "no kept run":
        assert not list((d / CP.PRIVATE / RUN).glob("unpacked-*"))
    rec, line = _record(monkeypatch, capsys, d, tmp_path / "pieces")
    assert rec is not None and line["composed"] == "pending" and "the network certifier" in line["missing"]


def test_n2_and_the_tie_take_every_link_window_of_a_short_run_unless_named():
    assert CP.sized(1, []) == ["CR_N2_DRAW=", "CR_TIE_WINDOWS=1"]
    assert CP.sized(8, ["CR_N2_DRAW=4"]) == ["CR_TIE_WINDOWS=8", "CR_N2_DRAW=4"]
    assert CP.sized(9, ["CR_SCAN_ONLY=1"]) == ["CR_SCAN_ONLY=1"]


def test_the_front_sends_one_group_of_the_class_and_keeps_the_fetch_in_the_serve_dirs_private(tmp_path, monkeypatch):
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / "ssh").write_text(STUB)
    (tmp_path / "bin" / "ssh").chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path / 'bin'}{os.pathsep}{os.defpath}")
    monkeypatch.setenv("STUB_DIR", str(tmp_path))
    serve, out = tmp_path / "serve", tmp_path / "out" / "certifier-front"
    assert CP.main(["front", "18190", str(serve), str(out), "--key", str(tmp_path / "key")]) == 0
    assert (tmp_path / "request").read_text() == (
        "warden-rehearse --peer vy-dc-h200 --port 18190 --minutes 7 --requests 4 --height 10 --scan --prompts "
        "--tokens 4 --window-per-group")
    prompts = json.loads((tmp_path / "prompts.json").read_text())
    assert [len(p) for p in prompts] == [CP.PROMPT_IDS] * CP.PROMPTS and all(t < 128000 for p in prompts for t in p)
    kept = serve / CP.PRIVATE / RUN / "fetch.tar.gz"
    assert [p.stat().st_mode & 0o777 for p in (serve / "private", serve / CP.PRIVATE, kept)] == [0o700, 0o700, 0o600]
    assert sorted(p.name for p in out.iterdir()) == ["bundle.tgz", "session.json", "start.json", "status.json"]
    assert json.loads((out / "session.json").read_text())["keep"] == str(kept)
    assert CP.main(["front", "18190", str(serve), str(out), "--tokens", "64"]) == 2
    assert "past one link-window" in json.loads((out / "session.json").read_text())["refused"]
