"""`verity bench inference_only --served`: a served run's record, made small from the record check's own shaped case.
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
         limits: dict | None = None) -> list[str]:
    """``case`` as a served run's files under ``where``, and the options that read them. Its rate rule's limits are the
    case's, or ``limits``; its policy is the case's count limits (``n`` in place of its n) on class `CLASS`, k by kind
    made of the linears' outputs and a scan state row of 16 bits (none where not ``kinds``). Its PoUW row is a development accept, or with ``accept`` Pearl+'s accept of record
    crediting ``credited`` of the ``macs`` it counts (default the class's linears), its window record the run's
    registration's previous (``tie`` "same"), another, or none."""
    inst, tr = case["instance"], case["trusted"]
    run, out = where / "run", where / "out"
    (out / "record").mkdir(parents=True)
    run.mkdir()
    (run / "program.json").write_text(inst["program"])
    (run / "circuit.json").write_text(inst["record"])
    logs = [run / f"log{k}.json" for k in range(len(tr["log"]))]
    for path, rec in zip(logs, tr["log"]):
        path.write_text(json.dumps(rec))
    count = {**case["count"], **({"n": n} if n else {})}
    for k in range(counts):
        (out / f"block-counts-{k}.json").write_text(json.dumps(count))
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
    return ["--served", str(out), "--program", str(run / "program.json"), "--circuit", str(run / "circuit.json"),
            "--log", *map(str, logs)]


def _main(args: list[str], exe: Path, out: Path | None = None) -> dict:
    ticks = itertools.count()
    return report.main(["--stated-only", *args, *(["--out", str(out)] if out else [])], commit=lambda: None,
                       clock=lambda: float(next(ticks)), record_check=exe)


def _record(lines: list[str]) -> list[str]:
    start = next(i for i, x in enumerate(lines) if x.startswith("FROM RUN "))
    return lines[start:lines.index("", start)]


def test_with_no_accept_of_record_the_value_is_one_and_measured(case, stand_in, tmp_path, capsys) -> None:
    """A development accept is no accept: s = 1, and the value is 1, marked measured, over every fleet and period too.
    The print names the run and the field that says the verdict is of record, prints each rule's public form only and
    the check's cost, and the weekly metric is not written from it."""
    doc = _main(made(case, tmp_path), stand_in, tmp_path / "result")
    out = capsys.readouterr().out
    rec, lines = doc["record"], _record(out.splitlines())
    assert lines[0] == ("FROM RUN made's RECORD: 1 (measured), no PoUW accept of record, so s = 1, and every limit at "
                        "s = 1 is 1 (min_cut.capped)")
    assert (rec["run"], rec["block"]["value"], rec["block"]["mark"], rec["pouw"]["share"]) == (RUN, 1, "measured", 1.0)
    assert rec["check"] == {"check": {"rule": "the rate rule", "public_form": "NciCheck.Record.check", "complies": True},
                            "count": {"rule": "the count", "public_form": "NciCheck.RecordCount.check", "complies": True}}
    assert "              the rate rule (NciCheck.Record.check): complies: true" in lines
    assert "              the count (NciCheck.RecordCount.check): complies: true" in lines
    assert f"  PoUW      of record by {R.VERDICT} = 'missing'" in lines
    assert any(x.startswith("            no accept of record (a development accept counts as none): " + SV.DEV_VERDICT)
               for x in lines)
    assert any(x.startswith("  share     s = 1: no PoUW accept credits") for x in lines)
    # each child's cost on the injected clock, and the command's own seconds
    assert rec["cost"]["check"]["wall_s"] == rec["cost"]["case"]["wall_s"] == 1.0
    assert rec["seconds"] == {"read": 0, "check": 0, "count": 1}
    assert any(x.startswith("  cost      in the clear, at the record's size: the check 1.0 s wall") for x in lines)
    assert {(r["value"], r["mark"]) for r in rec["periods"]} == {(1, "measured")} and len(rec["periods"]) == 2 * 3
    assert rec["tied"] is None and report.TIED_LABEL not in out and "Run made's class under the value tie" not in out
    # the private detail is nowhere: neither its keys nor the scratch the check wrote it to
    for word in ("bool_gates", "unregistered", "openings", "failure"):
        assert word not in out and word not in json.dumps(rec)
    assert not (tmp_path / "result" / "result.json").exists()


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


#: rate rule limits the case's record exceeds, where its count stays within its own
TIGHT = {"alpha": [1, 84], "gamma": 84}


@pytest.mark.parametrize(("how", "which", "complies"), [
    ({"n": 1}, "the count", {"check": True, "count": False}),
    ({"limits": TIGHT}, "the rate rule", {"check": False, "count": True}),
    ({"n": 1, "limits": TIGHT}, "both rules", {"check": False, "count": False}),
])
def test_a_record_a_rule_refuses_is_one_and_measured_and_names_only_the_rule(case, stand_in, tmp_path, capsys, how,
                                                                            which, complies) -> None:
    """The count against n = 1 refuses the record's two blocks, and the rate rule refuses it at limits it exceeds,
    each alone or both: the value is 1 whatever PoUW says, and the print names the rule that refused, not which of its
    limits failed nor any count."""
    doc = _main(made(case, tmp_path, accept=True, **how), stand_in)
    out = capsys.readouterr().out
    rec = doc["record"]
    assert {line: rec["check"][line]["complies"] for line in R.PUBLIC} == complies
    assert rec["check"]["count"]["public_form"] == "NciCheck.RecordCount.check"
    assert (rec["block"]["value"], rec["block"]["mark"], rec["tied"]) == (1, "measured", None)
    assert rec["block"]["reading"].startswith(f"the check refused the record ({which}), ")
    assert {r["reading"] for r in rec["periods"]} == {f"{which} refused"}
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
    ({"counts": 2}, "holds 2"),
])
def test_a_record_that_misstates_is_refused_before_the_check_runs(case, tmp_path, capsys, how, why) -> None:
    with pytest.raises(SystemExit) as e:
        _main(made(case, tmp_path, **how), tmp_path / "no-such-check")
    err = capsys.readouterr().err
    assert e.value.code == 2 and "the record is refused" in err and why in err


@pytest.mark.parametrize(("gone", "what"), [
    ("run/program.json", "Program"), ("run/circuit.json", "circuit record"), ("run/log2.json", "record 2 of a registration log"),
    ("out/record/verdicts.json", "compose verdicts"), ("out/inference-only-policy.json", "policy"),
    ("out/inference-only-limits.json", "rate rule's limits"), ("out/block-counts-0.json", "count's limits"),
])
def test_a_file_the_run_did_not_write_is_named(case, tmp_path, capsys, gone, what) -> None:
    """Each file is named as one the run did not write, with its path: none has a default."""
    args = made(case, tmp_path)
    (tmp_path / gone).unlink()
    with pytest.raises(SystemExit):
        _main(args, tmp_path / "no-such-check")
    assert f"the run wrote no {what} ({tmp_path / gone.replace('block-counts-0', 'block-counts-*')})" in capsys.readouterr().err


def test_served_takes_the_files_it_names_and_they_take_it(case, stand_in, tmp_path, capsys) -> None:
    args = made(case, tmp_path)
    for drop in (["--served", args[1]], args[2:4]):
        with pytest.raises(SystemExit):
            _main([x for x in args if x not in drop], stand_in)
        assert "--served OUT takes --program, --circuit and --log" in capsys.readouterr().err


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
    got = report.main(["--stated-only", *made(case, tmp_path / "b", accept=True)], commit=lambda: None)["record"]
    assert got["check"] == want and got["cost"]["check"]["peak_mib"] > 0
    assert "FROM RUN made's RECORD: " in capsys.readouterr().out
