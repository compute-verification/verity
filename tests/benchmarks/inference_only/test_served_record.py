"""`verity bench inference_only --from`: a served run's record, made small from the record check's own shaped case.
The check's case is written by its script; its verdicts come from a stand-in that prints the script's reference
verdicts as the compiled command prints them, or, where `lake` is, from the compiled command itself. What the command
keeps of each verdict is its public form; the share is PoUW's verdict of record's, refused where it can't be; and the
record's class under the tie is read from its policy's own k by kind."""
from __future__ import annotations

import importlib.util
import itertools
import json
import shutil
import sys
from pathlib import Path

import built as B
import policy as P
import pytest
import record as R
import report
import served_verdicts as SV
import tied as T

CLASS = (2, 2, 2)
#: the class's tag in its count's file name, `block-counts-<TAG>.json`
TAG = "-".join(map(str, CLASS))
RUN = "made"
SCAN = "the scan state row"
#: the compiled command's printout, made by the check's own script from its reference's verdicts
STAND_IN = """#!{python}
import importlib.util, json, sys
spec = importlib.util.spec_from_file_location("served_record_case", {script!r})
script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(script)
got = script.verdicts(json.loads(open(sys.argv[1]).read()))
print("[0 s] read the case")
print("[0 s] check: " + json.dumps(got["check"]))
print("[1 s] count: " + json.dumps(got["count"]))
"""


def _vectors():
    spec = importlib.util.spec_from_file_location("record_difftest_vectors",
                                                  B.PACKAGE / "scripts" / "record_difftest_vectors.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _script():
    spec = importlib.util.spec_from_file_location("served_record_case", R.SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def case() -> dict:
    return {m["shape"]: m for m in _vectors().shaped()}["2x3 registered"]["case"]


@pytest.fixture(autouse=True)
def no_out(monkeypatch) -> None:
    for name in ("OUT", "RESEARCH_RUN_ID", "RESEARCH_RUN_DIR"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(report.kit, "host_stamp", lambda: {"commit": None})


@pytest.fixture
def stand_in(tmp_path) -> Path:
    exe = tmp_path / "inference-only-record-check"
    exe.write_text(STAND_IN.format(python=sys.executable, script=str(R.SCRIPT)))
    exe.chmod(0o755)
    return exe


def made(case: dict, where: Path, *, accept: bool = False, credited: float = 0.9, macs: int | None = None,
         tie: str | None = "same", n: int | None = None, kinds: bool = True, counts: int = 1,
         limits: dict | None = None, registered: bool = False, unregister: bool = False) -> list[str]:
    """``case`` as a served run's files under ``where``, and the options that read them. Its rate rule's limits are the
    case's, or ``limits``; its policy is the case's count limits (``n`` in place of its n) on class `CLASS`, k by kind
    made of the linears' outputs and a scan state row of 16 bits (none where not ``kinds``). Its PoUW row is a development accept, or with ``accept`` Pearl+'s accept of record
    crediting ``credited`` of the ``macs`` it counts (default the class's linears), its window record the run's
    registration's previous (``tie`` "same"), another, or none. With ``unregister`` its log's first record registers
    no entry for the smallest parameter the record reads, as the case script's ``--unregister smallest`` makes it.
    With ``registered`` the Program, the circuit record and the log are in its out/ (`record.REGISTERED`), and the
    options are ``--from`` with the run's directory alone."""
    inst, tr = case["instance"], case["trusted"]
    run, out = where / "run", where / "out"
    (out / "record").mkdir(parents=True)
    run.mkdir()
    log = tr["log"]
    if unregister:
        made_case, _ = _script().served_case(inst["program"], inst["record"], log,
                                             {"alpha": inst["alpha"], "gamma": inst["gamma"]}, unregister="smallest")
        log = made_case["trusted"]["log"]
    (run / "program.json").write_text(inst["program"])
    (run / "circuit.json").write_text(inst["record"])
    logs = [run / f"log{k}.json" for k in range(len(log))]
    for path, rec in zip(logs, log):
        path.write_text(json.dumps(rec))
    if registered:
        (out / "registered").mkdir()
        (out / R.REGISTERED["program"]).write_text(inst["program"])
        (out / R.REGISTERED["circuit"]).write_text(inst["record"])
        (out / R.REGISTERED["log"]).write_text("".join(json.dumps(rec) + "\n" for rec in log))
    count = {**case["count"], **({"n": n} if n else {})}
    for tag in (TAG, "record", *map(str, range(2, counts)))[:counts]:
        (out / f"block-counts-{tag}.json").write_text(json.dumps(count))
    (out / "inference-only-limits.json").write_text(json.dumps(limits or {"alpha": inst["alpha"], "gamma": inst["gamma"]}))
    work = B.block_work(CLASS, B.MODEL)
    policy = {"format": R.POLICY_FORMAT, **{x: count[x] for x in "ckn"}, "serve": {"model": B.MODEL},
              "class": {"slots": CLASS[0], "prompt": CLASS[1], "outputs": CLASS[2],
                        "positions": P.class_of(CLASS).positions},
              "n_inputs": {"work": {**{x: work[x] for x in P.WORK}, "theta_macs": work[P.TOTAL]}}}
    if kinds:
        policy["k_by_kind"] = {"linears": count["k"] - 16, "scan_state": 16}
    (out / "inference-only-policy.json").write_text(json.dumps(policy))
    if accept:
        macs = CLASS[0] * work["linears"] if macs is None else macs
        window = {"same": tr["log"][-1]["previous"], "other": "ab" * 64, None: None}[tie]
        answer = {"protocol": SV.PEARLPLUS_PROTOCOL, "verdict": "accept", "reason": None, "commitment": "cd" * 64,
                  "public": {"work": {"credited": round(credited * macs), "macs": macs},
                             **({"tie": {"record": window}} if window else {})},
                  "profile": {"source": "ef" * 64}}
        row = {**SV.row_of("compute-accounting", answer, out), "answer_from": SV.PEARLPLUS_FROM}
    else:
        row = {"verdict": "missing", "owner": "compute-accounting", "why": f"{SV.DEV_VERDICT}: not of record",
               "development": {"piece": "pearlplus", "verdict": "accept"}}
    (out / "record" / "verdicts.json").write_text(json.dumps({"run": RUN, "protocols": {"work": row}}))
    if registered:
        return ["--from", str(where)]
    return ["--from", str(out), "--program", str(run / "program.json"), "--circuit", str(run / "circuit.json"),
            "--log", *map(str, logs)]


def _run(args: list[str], exe: Path, out: Path | None = None) -> tuple[int, dict]:
    ticks = itertools.count()
    return report.run(["--stated-only", *args, *(["--out", str(out)] if out else [])], commit=lambda: None,
                      clock=lambda: float(next(ticks)), record_check=exe)


def _main(args: list[str], exe: Path, out: Path | None = None) -> dict:
    return _run(args, exe, out)[1]


def _record(lines: list[str]) -> list[str]:
    start = next(i for i, x in enumerate(lines) if x.startswith("FROM RUN "))
    return lines[start:lines.index("", start)]


DEV_WHY = "no PoUW accept of record (a development accept counts as none)"
#: the kit's lines under a record's first line (`benchmarks/README.md`'s interface)
NOT_STATED, NOT_AN_AUDIT = "ran with: not stated", "a measurement, not an audit"


def test_with_no_accept_of_record_there_is_no_number_and_the_exit_is_1(case, stand_in, tmp_path, capsys) -> None:
    """A development accept is no accept: the kit's first line says there is no number and why, the JSON's value is
    null beside it, the metric is written with no value and that why, no period has a value, and the exit is 1. The
    print names the run and the field that says the verdict is of record, and prints each rule's public form only and
    the check's cost."""
    code, doc = _run(made(case, tmp_path), stand_in, tmp_path / "result")
    out = capsys.readouterr().out
    rec, every, lines = doc["record"], out.splitlines(), _record(out.splitlines())
    assert code == 1 and every[:6] == [f"inference_only: no number: {DEV_WHY}", NOT_STATED, NOT_AN_AUDIT,
                                       report.TITLE, f"The served run's record read: {tmp_path / 'out'}", ""]
    assert lines[0] == f"FROM RUN made's RECORD: no number, {DEV_WHY}"
    assert (rec["run"], rec["block"]["value"], rec["no_number"], rec["pouw"]["share"]) == (RUN, None, DEV_WHY, 1.0)
    shown = json.loads(every[-1])
    assert (shown["value"], shown["run"], shown["no_number"], shown["ran_with"], shown["audit"]) == (
        None, RUN, DEV_WHY, None, False)
    assert rec["check"] == {"check": {"rule": "the rate rule", "public_form": "InferenceOnlyVerifier.Record.check", "complies": True},
                            "count": {"rule": "the count", "public_form": "InferenceOnlyVerifier.RecordCount.check", "complies": True}}
    assert "              the rate rule (InferenceOnlyVerifier.Record.check): complies: true" in lines
    assert "              the count (InferenceOnlyVerifier.RecordCount.check): complies: true" in lines
    assert f"  PoUW      of record by {R.VERDICT} = 'missing'" in lines
    assert any(x.startswith("            no accept of record (a development accept counts as none): " + SV.DEV_VERDICT)
               for x in lines)
    assert any(x.startswith("  share     s = 1: no PoUW accept credits") for x in lines)
    # each child's cost on the injected clock, and the command's own seconds
    assert rec["cost"]["check"]["wall_s"] == rec["cost"]["case"]["wall_s"] == 1.0
    assert rec["seconds"] == {"read": 0, "check": 0, "count": 1}
    assert any(x.startswith("  cost      in the clear, at the record's size: the check 1.0 s wall") for x in lines)
    assert rec["periods"] == [] and "Over each monitoring period" not in out
    assert rec["tied"] is None and report.TIED_LABEL not in out and "Run made's class under the value tie" not in out
    # the private detail is nowhere: neither its keys nor the scratch the check wrote it to
    for word in ("bool_gates", "unregistered", "openings", "failure"):
        assert word not in out and word not in json.dumps(rec)
    result = json.loads((tmp_path / "result" / "result.json").read_text())
    assert [(m["name"], m["value"], m["why"]) for m in result["measurements"]] == [(report.METRIC, None, DEV_WHY)]


def test_an_accept_of_record_prices_the_class_under_the_tie_at_its_share(case, stand_in, tmp_path, capsys) -> None:
    """Pearl+'s accept, tied to the record the check read: s = 1 - C / W, one block's value the smaller of its two
    limits with its reading (partial), and each fleet and period's the least over its readings (modeled). The
    class's k is the policy's kinds but the linears' outputs, with tied.toml's salt."""
    doc = _main(made(case, tmp_path, accept=True), stand_in)
    out = capsys.readouterr().out
    rec, lines = doc["record"], _record(out.splitlines())
    work = B.block_work(CLASS, B.MODEL)
    macs, W = CLASS[0] * work["linears"], CLASS[0] * work[P.TOTAL]
    C = round(0.9 * macs)
    assert rec["pouw"]["share"] == pytest.approx(1 - C / W) and rec["pouw"]["C"] == C and rec["pouw"]["W"] == W
    tie = rec["pouw"]["tie"]
    assert tie["window"] == tie["previous"] == case["trusted"]["log"][-1]["previous"]
    assert any(x.startswith(f"  tie       PoUW's window record {tie['window'][:16]}... = the run's registration's previous")
               for x in lines)
    tied = rec["tied"]
    assert tied["k"] == 16 + 512 and tied["c"] == case["count"]["c"] and tied["n"] == case["count"]["n"]
    assert tied["uncredited"]["share"] == rec["pouw"]["share"] and tied["uncredited"]["pouw_verdict"]["C"] == C
    b = tied["block"]
    want = min((b["limit"], "every word tied"), (b["with_words"]["limit"], T.IN_K))
    assert (rec["block"]["value"], rec["block"]["reading"], rec["block"]["mark"]) == (*want, "partial")
    assert "the record's" in rec["block"]["inputs"] and "premises: the tie" in rec["block"]["inputs"]
    assert lines[0].startswith(f"FROM RUN made's RECORD: {report._n(want[0])} (partial), {want[1]}")
    least = {}
    for r in tied["periods"]:
        if (key := (r["where"], r["period"])) not in least or r["limit"] < least[key][0]:
            least[key] = (r["limit"], r["reading"])
    assert {(r["where"], r["period"]): (r["value"], r["reading"]) for r in rec["periods"]} == least
    assert {r["mark"] for r in rec["periods"]} == {"modeled"} and len(rec["periods"]) == 2 * 4
    section = out[out.index("Run made's class under the value tie"):out.index("\nOver a monitoring period")]
    assert "is run made's: 1 - C / W" in section and "No limit is proved: it is arithmetic on the record's share" in section
    assert "typed" not in section and report.TIED_LABEL not in out


def test_an_accept_with_no_window_record_says_nothing_ties_it(case, stand_in, tmp_path, capsys) -> None:
    doc = _main(made(case, tmp_path, accept=True, tie=None), stand_in)
    out = capsys.readouterr().out
    assert doc["record"]["pouw"]["tie"]["why"] == R.NO_TIE == "nothing ties the PoUW verdict to the record the check read"
    assert f"  tie       {R.NO_TIE}" in out.splitlines() and f"({R.NO_TIE});" in out


def test_with_a_record_read_its_number_heads_the_print_and_the_json_at_its_own_mark(case, stand_in, tmp_path,
                                                                                    capsys) -> None:
    """A PoUW accept of record with the check run and both rules complying: the command's number is the record's, the
    exit is 0, and the record's headline is printed first after the kit's lines: the coins and parameters it states
    (none here) and that it is not an audit. The JSON's value and run, and the measurement under the metric's id at
    the record's own mark, are the record's. The stated policy's number follows, labelled as not the result."""
    code, doc = _run(made(case, tmp_path, accept=True), stand_in, tmp_path / "result")
    out = capsys.readouterr().out
    lines, one = out.splitlines(), doc["record"]["block"]
    assert code == 0 and doc["record"]["no_number"] is None and "no number" not in out
    assert lines[:5] == [report.TITLE, NOT_STATED, NOT_AN_AUDIT,
                         f"The served run's record read: {tmp_path / 'out'}", ""]
    assert lines[5].startswith(f"FROM RUN {RUN}'s RECORD: {report._n(one['value'])} (partial), ")
    stated = [i for i, x in enumerate(lines) if x.startswith("THE SERVED POLICY'S NUMBER, NOT THE RESULT: ")]
    assert len(stated) == 1 and stated[0] > 5 and "AS SERVED" not in out
    shown = json.loads(lines[-1])
    assert list(shown)[:4] == ["value", "unit", "commit", "run"] and (shown["ran_with"], shown["audit"]) == (None, False)
    assert (shown["value"], shown["run"]) == (doc["value"], doc["run"]) == (one["value"], RUN)
    result = json.loads((tmp_path / "result" / "result.json").read_text())
    assert [(m["name"], m["value"], m["mark"], m["verdict"]) for m in result["measurements"]] == [
        (report.METRIC, one["value"], "partial", report.verdict(one["value"]))]


def test_a_record_with_no_figure_writes_no_value_and_says_why(case, stand_in, tmp_path, capsys) -> None:
    code, doc = _run(made(case, tmp_path, accept=True, kinds=False), stand_in, tmp_path / "result")
    one = doc["record"]["block"]
    assert (one["value"], one["mark"], doc["value"]) == (None, None, None) and one["reading"] == f"no figure: {R.NO_KINDS}"
    lines = capsys.readouterr().out.splitlines()
    assert code == 1 and lines[0] == f"inference_only: no number: no figure: {R.NO_KINDS}"
    assert _record(lines)[0] == f"FROM RUN {RUN}'s RECORD: no number, no figure: {R.NO_KINDS}"
    result = json.loads((tmp_path / "result" / "result.json").read_text())
    assert [(m["name"], m["value"], m["mark"], m["why"]) for m in result["measurements"]] == [
        (report.METRIC, None, "partial", one["reading"])]


def test_a_record_that_states_no_run_is_named_by_its_directory(case, stand_in, tmp_path, capsys) -> None:
    args = made(case, tmp_path)
    verdicts = tmp_path / "out" / "record" / "verdicts.json"
    verdicts.write_text(json.dumps({k: v for k, v in json.loads(verdicts.read_text()).items() if k != "run"}))
    doc = _main(args, stand_in)
    lines = capsys.readouterr().out.splitlines()
    assert doc["run"] == doc["record"]["run"] == tmp_path.name
    assert _record(lines)[0] == f"FROM RUN {tmp_path.name}'s RECORD: no number, {DEV_WHY}"


@pytest.mark.parametrize("accept", [False, True])
def test_the_ran_with_line_says_the_coins_and_parameters_the_record_states(case, stand_in, tmp_path, capsys,
                                                                          accept) -> None:
    """PoUW's piece, an accept of record's or a development row's, read under the keys it writes today: a live
    challenger session is a local challenger's coins, never the deployed challenger's, at development parameters with
    its scheme; a record that names no session (in-process coins name none under those keys) is not stated. Either is
    not an audit."""
    piece = {"coins": {"check": {"session": "s-made", "reason": None}}, "scheme": "pearl-c-made"}
    assert R.made_with({"protocols": {"work": {"piece": piece}}}) == {"session": "s-made", "scheme": "pearl-c-made"}
    assert R.made_with({"protocols": {"work": {"development": piece}}}) == {"session": "s-made", "scheme": "pearl-c-made"}
    assert R.made_with({"protocols": {"work": {"piece": {"coins": {}, "scheme": "pearl-c-made"}}}}) == {
        "session": None, "scheme": "pearl-c-made"}
    assert R.made_with({}) == R.made_with({"protocols": {"work": "missing"}}) == {"session": None, "scheme": None}
    args = made(case, tmp_path, accept=accept)
    verdicts = tmp_path / "out" / "record" / "verdicts.json"
    doc = json.loads(verdicts.read_text())
    work = doc["protocols"]["work"]
    work["piece" if accept else "development"] = {**work.get("development", {}), **piece}
    verdicts.write_text(json.dumps(doc))
    code, _ = _run(args, stand_in)
    lines = capsys.readouterr().out.splitlines()
    line = "ran with: coins local challenger; parameters development (scheme pearl-c-made)"
    assert code == (0 if accept else 1) and lines[1:3] == [line, NOT_AN_AUDIT]
    shown = json.loads(lines[-1])
    assert (shown["ran_with"], shown["audit"]) == (
        {"coins": "local challenger", "parameters": "development", "detail": "scheme pearl-c-made"}, False)
    assert "deployed challenger" not in "\n".join(lines)


def test_the_served_class_option_picks_one_count_of_several(case, stand_in, tmp_path, capsys) -> None:
    """Where a record holds counts of more than one class, `--served-class` reads the one it names; a class the record
    holds no count of is refused naming the classes it holds, and the option without a record is refused."""
    args = made(case, tmp_path, counts=2)
    for cls in ("2,2,2", "record"):
        assert _main([*args, R.CLASS_OPTION, cls], stand_in)["record"]["no_number"] == DEV_WHY
    capsys.readouterr()
    with pytest.raises(SystemExit) as e:
        _main([*args, R.CLASS_OPTION, "9,9,9"], stand_in)
    assert e.value.code == 2 and (f"{R.CLASS_OPTION} 9,9,9: the record holds no count of that class; it holds counts "
                                  "of the classes 2,2,2, record") in capsys.readouterr().err
    with pytest.raises(SystemExit) as e:
        _main([R.CLASS_OPTION, "record"], stand_in)
    assert e.value.code == 2 and f"{R.CLASS_OPTION} names a class of a served run's record, and no record is read" in (
        capsys.readouterr().err)


#: rate rule limits the case's record exceeds, where its count stays within its own
TIGHT = {"alpha": [1, 84], "gamma": 84}


@pytest.mark.parametrize(("how", "which", "complies"), [
    ({"n": 1}, "the count", {"check": True, "count": False}),
    ({"limits": TIGHT}, "the rate rule", {"check": False, "count": True}),
    ({"n": 1, "limits": TIGHT}, "both rules", {"check": False, "count": False}),
])
def test_a_record_a_rule_refuses_has_no_number_and_names_only_the_rule(case, stand_in, tmp_path, capsys, how,
                                                                      which, complies) -> None:
    """The count against n = 1 refuses the record's two blocks, and the rate rule refuses it at limits it exceeds,
    each alone or both: there is no number whatever PoUW says, the exit is 1, and the print names the rule that
    refused, not which of its limits failed nor any count."""
    code, doc = _run(made(case, tmp_path, accept=True, **how), stand_in)
    out = capsys.readouterr().out
    rec, why = doc["record"], f"the check refused the record ({which})"
    assert {line: rec["check"][line]["complies"] for line in R.PUBLIC} == complies
    assert rec["check"]["count"]["public_form"] == "InferenceOnlyVerifier.RecordCount.check"
    assert (code, rec["block"]["value"], rec["no_number"], rec["tied"], rec["periods"]) == (1, None, why, None, [])
    assert out.splitlines()[0] == f"inference_only: no number: {why}"
    for line, (rule, form) in R.PUBLIC.items():
        assert f"              {rule} ({form}): complies: {str(complies[line]).lower()}" in out.splitlines()
    for word in ("failure", "fresh", "gates", "unregistered"):
        assert word not in json.dumps(rec["check"])


@pytest.mark.parametrize(("how", "why"), [
    ({"accept": True, "credited": 1.5}, "above W = "),
    ({"accept": True, "credited": 1.1}, "above the"),
    ({"accept": True, "credited": -0.1}, "below 0"),
    ({"accept": True, "macs": 7}, "MACs of linears, not the record's class.slots"),
    ({"accept": True, "tie": "other"}, "the verdict is about another record than the check's"),
    ({"counts": 2}, f"holds counts of 2 classes, 2,2,2, record: name the one to read with {R.CLASS_OPTION} CLASS"),
])
def test_a_record_that_misstates_is_refused_before_the_check_runs(case, tmp_path, capsys, how, why) -> None:
    with pytest.raises(SystemExit) as e:
        _main(made(case, tmp_path, **how), tmp_path / "no-such-check")
    err = capsys.readouterr().err
    assert e.value.code == 2 and "the record is refused" in err and why in err


def test_a_record_with_no_program_has_no_number_and_says_the_check_did_not_run(case, tmp_path, capsys) -> None:
    """A record that holds no Program (a serve under the Pearl+ tie writes none) can't be checked: there is no number,
    whatever PoUW's accept credits, its why says the check did not run, no check runs, and the exit is 1."""
    args = made(case, tmp_path, accept=True)
    (tmp_path / "run" / "program.json").unlink()
    code, doc = _run(args, tmp_path / "no-such-check")
    rec, out = doc["record"], capsys.readouterr().out.splitlines()
    why = f"{R.NO_PROGRAM}, so the check did not run"
    assert (code, rec["block"]["value"], rec["no_number"], rec["cost"], rec["periods"]) == (1, None, why, None, [])
    assert rec["pouw"]["verdict"] == "accept" and out[0] == f"inference_only: no number: {why}"
    assert all("complies" not in v for v in rec["check"].values()) and rec["files"]["program_bytes"] is None
    lines = _record(out)
    assert lines[0] == f"FROM RUN made's RECORD: no number, {why}"
    assert f"  check     {R.CHECK}: not run, {R.NO_PROGRAM}" in lines


@pytest.mark.parametrize(("gone", "what"), [
    ("run/circuit.json", "circuit record"), ("run/log2.json", "record 2 of a registration log"),
    ("out/record/verdicts.json", "compose verdicts"), ("out/inference-only-policy.json", "policy"),
    ("out/inference-only-limits.json", "rate rule's limits"), (f"out/block-counts-{TAG}.json", "count's limits"),
])
def test_a_file_the_run_did_not_write_is_named(case, tmp_path, capsys, gone, what) -> None:
    """Each file is named as one the run did not write, with its path: none has a default."""
    args = made(case, tmp_path)
    (tmp_path / gone).unlink()
    with pytest.raises(SystemExit):
        _main(args, tmp_path / "no-such-check")
    assert f"the run wrote no {what} ({tmp_path / gone.replace(TAG, '*')})" in capsys.readouterr().err


def test_the_files_named_apart_take_served_and_come_all_three_or_none(case, stand_in, tmp_path, capsys) -> None:
    args = made(case, tmp_path)
    for drop in (["--from", args[1]], args[2:4]):
        with pytest.raises(SystemExit):
            _main([x for x in args if x not in drop], stand_in)
        assert "--program, --circuit and --log take --from DIR, and come all three or none" in capsys.readouterr().err


@pytest.mark.parametrize("accept", [False, True])
def test_served_alone_reads_the_run_directorys_registered_files(case, stand_in, tmp_path, capsys, accept) -> None:
    """``--from DIR`` alone, the run's directory or its out/, reads the files the run registered from out/'s
    `REGISTERED`, its log one record a line, and gives what the same files named apart give; the kit's lines say the
    number is a measurement and not an audit, and the command's which record was read. ``--served``, its old name,
    reads the same."""
    named = _main(made(case, tmp_path / "a", accept=accept), stand_in)["record"]
    made(case, tmp_path / "b", accept=accept, registered=True)
    for flag, served in (("--from", tmp_path / "b"), ("--from", tmp_path / "b" / "out"), ("--served", tmp_path / "b")):
        capsys.readouterr()
        got = _main([flag, str(served)], stand_in)["record"]
        lines = capsys.readouterr().out.splitlines()
        assert lines.index(NOT_AN_AUDIT) == 2 and lines.index(report.TITLE) == (0 if accept else 3)
        assert lines[lines.index(report.TITLE) + (3 if accept else 1)] == (
            f"The served run's record read: {tmp_path / 'b' / 'out'}")
        keys = ("run", "check", "pouw", "block", "files")
        assert {x: got[x] for x in keys} == {x: named[x] for x in keys}
        assert got["files"]["log_records"] == 3 and got["pouw"]["verdict"] == ("accept" if accept else "missing")


def test_a_run_that_registered_no_files_where_served_reads_them_is_named(case, tmp_path, capsys) -> None:
    made(case, tmp_path, registered=True)
    (tmp_path / "out" / R.REGISTERED["log"]).unlink()
    with pytest.raises(SystemExit):
        _main(["--from", str(tmp_path)], tmp_path / "no-such-check")
    assert f"the run wrote no registration log ({tmp_path / 'out' / R.REGISTERED['log']})" in capsys.readouterr().err


def test_a_weight_left_unregistered_is_refused_by_the_count(case, stand_in, tmp_path, capsys) -> None:
    """The log's first record without the smallest parameter the record reads: the count refuses the record, and so
    does the rate rule, and there is no number, whatever PoUW's accept credits."""
    rec = _main(made(case, tmp_path, accept=True, tie=None, unregister=True), stand_in)["record"]
    assert {line: rec["check"][line]["complies"] for line in R.PUBLIC} == {"check": False, "count": False}
    assert (rec["block"]["value"], rec["no_number"]) == (None, "the check refused the record (both rules)")


def test_the_data_file_names_the_served_run_of_record_or_none(tmp_path) -> None:
    """Absent, empty or with no `record`, the data file names no run; a path is the repository's root's or absolute;
    any other key or a record that isn't a path is refused."""
    f = tmp_path / "of_record.toml"
    assert R.of_record(f) is None
    for text, want in (("", None), ("# none\n", None), ('record = "runs/a"\n', P.ROOT / "runs/a"),
                       (f'record = "{tmp_path}"\n', tmp_path)):
        f.write_text(text)
        assert R.of_record(f) == want
    for text, why in (('record = 3\n', "a path, not 3"), ('record = ""\n', "a path"), ('run = "a"\n', "its one key"),
                      ("record = [", "not TOML")):
        f.write_text(text)
        with pytest.raises(R.Refused, match=why):
            R.of_record(f)


def test_the_command_reads_the_run_the_data_file_names(case, stand_in, tmp_path, monkeypatch, capsys) -> None:
    """With no --from, the record the data file names is read as --from would read it; --from names another."""
    made(case, tmp_path / "of-record", registered=True)
    f = tmp_path / "of_record.toml"
    f.write_text(f'record = "{tmp_path / "of-record"}"\n')
    monkeypatch.setattr(R, "OF_RECORD", f)
    assert _main([], stand_in)["record"]["run"] == RUN
    assert f"read: {tmp_path / 'of-record' / 'out'}" in capsys.readouterr().out
    made(case, tmp_path / "other", registered=True)
    _main(["--from", str(tmp_path / "other")], stand_in)
    assert f"read: {tmp_path / 'other' / 'out'}" in capsys.readouterr().out
    f.write_text("")
    assert _main([], stand_in)["record"] is None


def test_with_no_lake_the_command_says_in_one_line_what_it_needs(case, tmp_path, monkeypatch, capsys) -> None:
    """Where the check would be built here and `lake` isn't on PATH, the command exits 1 before any work with one line
    naming elan and the toolchain, for a served record and for the built class alike."""
    monkeypatch.setattr(B.shutil, "which", lambda name: None)
    toolchain = (B.PACKAGE / "lean-toolchain").read_text().strip()
    for argv in (["--stated-only", *made(case, tmp_path)], []):
        with pytest.raises(SystemExit) as e:
            report.run(argv, commit=lambda: None)
        out, err = capsys.readouterr()
        assert e.value.code == 1 and out == ""
        assert err.count("\n") == 1 and "`lake` isn't on PATH" in err and "elan" in err and toolchain in err


def test_the_record_class_under_the_tie_is_its_own_kinds(tmp_path) -> None:
    """On tied.toml's own class, a record policy whose k is the served policy's by kind gives tied.toml's k: its kinds
    but the linears' outputs, with tied.toml's salt. A policy with no k by kind (one written before its writer stated
    it) gives no figure for the tied class, and one with other kinds, kinds by their names rather than the writer's
    keys, or kinds that don't sum to its k is refused."""
    served, t = P.stated(), P.tied()
    doc = {"c": served.c, "k": served.k, "n": served.n, "serve": {"model": t.policy.model},
           "class": {"slots": served.cls[0], "prompt": served.cls[1], "outputs": served.cls[2],
                     "positions": P.class_of(served.cls).positions},
           "n_inputs": {"work": {**{x: served.work[x] for x in P.WORK}, "theta_macs": served.work[P.TOTAL]}},
           "k_by_kind": {"linears": served.outside, "scan_state": P.scan_state_bits(served.cls)}}
    got = R.tied_of(doc, RUN)
    assert got.policy.k == t.policy.k == 146_080 and (got.policy.c, got.policy.n, got.policy.work) == (
        t.policy.c, t.policy.n, t.policy.work)
    assert [(kind, bits) for kind, bits, _ in got.policy.parts] == [(SCAN, 145_568), ("the execution phase's salt", 512)]
    assert (got.words, got.blocks, got.word_bits) == (t.words, t.blocks, t.word_bits)
    for kinds, why in ((None, R.NO_KINDS), ({**doc["k_by_kind"], "more": 0}, "exactly"),
                       ({"linears": served.k}, "exactly"), ({R.OUTPUTS: served.outside, "scan_state": 1}, "exactly"),
                       ({"linears": served.outside, "scan_state": 1}, "sum to")):
        with pytest.raises(R.Refused, match="no figure for the tied class" if kinds is None else why):
            R.tied_of({**doc, "k_by_kind": kinds} if kinds else {x: v for x, v in doc.items() if x != "k_by_kind"}, RUN)


def test_a_check_that_fails_is_named_by_its_exit_code_alone(case, tmp_path) -> None:
    """A check that exits nonzero, whatever it wrote on its standard error, raises with its exit code and none of it."""
    exe = tmp_path / "inference-only-record-check"
    exe.write_text(f"#!{sys.executable}\nimport sys\nsys.stderr.write('{{\"bool_gates\": 7, \"failure\": 3}}')\nsys.exit(3)\n")
    exe.chmod(0o755)
    with pytest.raises(RuntimeError) as got:
        _main(made(case, tmp_path / "a", accept=True), exe)
    assert str(got.value) == "inference-only-record-check exited 3"


def test_a_line_keeps_only_its_public_form() -> None:
    assert R.public({"complies": True, "bool_gates": 3, "failure": None}) == {"complies": True}
    assert R.public({"error": "inputs", "message": "m", "at": 2}) == {"error": "inputs", "message": "m"}
    with pytest.raises(RuntimeError, match="neither a judgment nor a refusal"):
        R.public({"bool_gates": 3})


@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_the_compiled_check_gives_the_stand_ins_public_verdicts(case, stand_in, tmp_path, capsys) -> None:
    """The compiled command on the made record, built here, gives what the stand-in gives, in public form."""
    want = _main(made(case, tmp_path / "a", accept=True), stand_in)["record"]["check"]
    got = report.run(["--stated-only", *made(case, tmp_path / "b", accept=True)], commit=lambda: None)[1]["record"]
    assert got["check"] == want and got["cost"]["check"]["peak_mib"] > 0
    assert "FROM RUN made's RECORD: " in capsys.readouterr().out
