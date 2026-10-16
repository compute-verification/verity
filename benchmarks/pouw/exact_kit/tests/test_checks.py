import io
import json

from exact_kit.checks import SCHEMA, Checks, dumps
from research import result


def _run(tmp_path, argv=(), body=None):
    script = tmp_path / "probe.py"
    script.write_text("# a script the verdict names\n")
    buf = io.StringIO()
    c = Checks(argv=list(argv), script=str(script), out=buf)
    (body or (lambda c: None))(c)
    return c, buf


def test_lines_match_the_legacy_registries(tmp_path):
    def body(c):
        c.check("A rank", True, "rank 4")
        c.check("B no detail", True)
        c.check("C empty detail", True, "")
        c.check("D false", 0, 7)
        c.mutation("even weights", True, "rank 0")
        c.mutation("odd weights", False)

    c, buf = _run(tmp_path, body=body)
    assert buf.getvalue().splitlines() == [
        "[OK  ] A rank: rank 4",
        "[OK  ] B no detail",
        "[OK  ] C empty detail: ",
        "[FAIL] D false: 7",
        "[OK  ] mutation caught: even weights: rank 0",
        "[FAIL] mutation NOT caught: odd weights",
    ]
    assert c.summary() == "3/4 checks OK, 1/2 mutations caught  <-- FAILURES"


def test_summary_and_exit_code(tmp_path):
    c, buf = _run(tmp_path, body=lambda c: [c.check(f"c{i}", True, i) for i in range(3)])
    assert c.finish() == 0
    assert buf.getvalue().endswith("\n\n3/3 checks OK\n")
    c, _ = _run(tmp_path, body=lambda c: (c.check("a", True), c.mutation("m", True)))
    assert c.summary() == "1/1 checks OK, 1/1 mutations caught" and c.finish() == 0
    c, _ = _run(tmp_path, body=lambda c: (c.check("a", True), c.mutation("m", False)))
    assert c.finish() == 1


def test_argv(tmp_path):
    c, _ = _run(tmp_path, argv=["A", "--mutations", "--verdict", str(tmp_path / "v.json"), "C"])
    assert c.mutations and c.args == ["A", "C"] and c.verdict_path == str(tmp_path / "v.json")
    c, _ = _run(tmp_path, argv=["--verdict=x.json"])
    assert not c.mutations and c.args == [] and c.verdict_path == "x.json"


def test_verdict_is_deterministic_and_named(tmp_path, monkeypatch):
    def body(c):
        c.check("a", True, "x")
        c.mutation("m", True)

    path = tmp_path / "out" / "v.json"
    c, _ = _run(tmp_path, argv=["--verdict", str(path), "--mutations"], body=body)
    c.finish()
    first = path.read_bytes()
    c, _ = _run(tmp_path, argv=["--verdict", str(path), "--mutations"], body=body)
    c.finish()
    assert path.read_bytes() == first
    doc = json.loads(first)
    assert doc["schema"] == SCHEMA and doc["ok"] is True and doc["mutations_run"] is True
    assert doc["script"]["name"] == "probe.py" and len(doc["script"]["sha256"]) == 64
    assert doc["counts"] == {"checks": 1, "checks_ok": 1, "mutations": 1, "mutations_caught": 1}
    assert first.decode() == dumps(doc)

    monkeypatch.setenv("OUT", str(tmp_path / "run-out"))
    c, _ = _run(tmp_path, body=body)
    c.finish()
    assert json.loads((tmp_path / "run-out" / "probe.verdict.json").read_text())["counts"]["checks"] == 1


def test_mutant_is_caught_iff_one_of_its_checks_fails(tmp_path):
    def body(c):
        c.check("honest", True)
        with c.mutant("break it"):
            c.check("inner a", True)
            c.check("inner b", False)
        with c.mutant("leave it"):
            c.check("inner c", True)

    c, buf = _run(tmp_path, body=body)
    assert [x["name"] for x in c.checks] == ["honest"]
    assert [(m["name"], m["caught"]) for m in c.mutants] == [("break it", True), ("leave it", False)]
    assert "[OK  ] mutation caught: break it: 1 of 2 checks fail" in buf.getvalue().splitlines()
    assert c.finish() == 1


def test_result_json_under_a_run(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    (run_dir / "out").mkdir(parents=True)
    monkeypatch.setenv("RESEARCH_RUN_DIR", str(run_dir))
    monkeypatch.setenv("RESEARCH_RUN_ID", "r-test")
    monkeypatch.setenv("OUT", str(run_dir / "out"))
    c, _ = _run(tmp_path, body=lambda c: (c.check("a", True), c.check("b", False)))
    assert c.finish() == 1
    chk = result.check(run_dir, "r-test")
    assert chk.valid, chk.errors
    assert chk.validation_status == "failed" and chk.doc["validation"]["evidence"] == ["out/probe.verdict.json"]
    assert {m["name"]: m["value"] for m in chk.doc["measurements"]}["checks_ok"] == 1
