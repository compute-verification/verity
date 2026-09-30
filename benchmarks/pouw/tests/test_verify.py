"""The verify step (`harness/verify.py`, panel rule (b)) and the verifier declaration (`arm.check_verifier`, arm interface
v0.4): an accepted row carries the verifier's commit, its accept line verbatim and the transcript as `<run>:<path>
sha256:<hex>`, in the form `panel.py append` takes; a rejected, missing or stale transcript, a per-shape command that fails,
an arm without a verifier, and a run without a run id or commit are not accepted; the same per-run command runs once; a bad
declaration is refused before timing. The verifiers are small scripts shaped like GPU 1's `verify.py` (one run, one line per
transcript) and GPU 5's replay (prove, then verify, per shape)."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "harness"))

import arm as A  # noqa: E402
import verify as V  # noqa: E402

RUN, COMMIT = "r20260930-120000-abcd", "0123456789abcdef0123456789abcdef01234567"
SHAPES = {"m8192-n8192-k8192": "prefill", "m32-n8192-k8192": "decode"}
PREFILL, DECODE = SHAPES

RUN_VERIFIER = """
import hashlib, sys
from pathlib import Path
ok = True
for m in sorted(Path(sys.argv[1]).glob("transcripts/*/*/manifest.json")):
    raw = m.read_bytes()
    good = b"bad" not in raw
    ok &= good
    print(("ACCEPT" if good else "REJECT") + f" {m.parent.parent.name}/{m.parent.name}: transcript "
          f"sha256:{hashlib.sha256(raw).hexdigest()}; verifier {sys.argv[2]}")
sys.exit(0 if ok else 1)
"""

SHAPE_VERIFIER = """
import hashlib, json, sys
from pathlib import Path
if sys.argv[1] == "prove":
    Path(sys.argv[3]).write_text(json.dumps({"dump": Path(sys.argv[2]).read_text(), "run": sys.argv[4]}))
else:
    raw = Path(sys.argv[2]).read_bytes()
    good = b"bad" not in raw
    print(("ACCEPT" if good else "REJECT") + f" replay {sys.argv[3]} sha256:{hashlib.sha256(raw).hexdigest()} "
          f"verifier={sys.argv[4]}")
    sys.exit(0 if good else 1)
"""


def per_run(script: Path) -> dict:
    return {"per": "run", "transcript": "transcripts/{arm}/{shape}/manifest.json",
            "commands": [["{python}", str(script), "{run_dir}", "{commit}"]]}


def per_shape(script: Path) -> dict:
    return {"transcript": "transcripts/{arm}-{shape}/transcript.json",
            "commands": [["{python}", str(script), "prove", "{dir}/dump", "{transcript}", "{run_id}"],
                         ["{python}", str(script), "verify", "{transcript}", "{run_id}:{shape}", "{commit}"]]}


def result(run_dir: Path, verifiers: dict, clocks: bool = True) -> Path:
    shapes = []
    for label, phase in SHAPES.items():
        arms = {}
        for name in verifiers:
            p = {"phase": phase, "shape": label, "slowdown": 1.25, "hash_free": 1.1, "baseline": "fp8-e4m3"}
            p |= {"sm_clock_arm": "2090,2092", "sm_clock_base": "2091,2093"} if clocks else {"sm_clock_missing": "x"}
            if phase == "decode":
                p["decode_method"] = "dependent-chain"
            arms[name] = {"graph": {}, "panel": p}
        shapes.append({"label": label, "arms": arms})
    arms = {n: {"stream": "fp8", "verifier": None if v is None else A.check_verifier(n, {"verifier": v})}
            for n, v in verifiers.items()}
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "bench.json"
    path.write_text(json.dumps({"shapes": shapes, "arms": arms}))
    return path


def script(tmp_path: Path, name: str, text: str) -> Path:
    p = tmp_path / name
    p.write_text(text)
    return p


def manifest(run_dir: Path, arm: str, shape: str, text: str = "{}") -> str:
    p = run_dir / "transcripts" / arm / shape / "manifest.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return hashlib.sha256(text.encode()).hexdigest()


def run(path: Path, *extra: str) -> tuple[int, dict]:
    code = V.main([str(path), "--run-id", RUN, "--commit", COMMIT, *extra])
    return code, json.loads(path.read_text())


def panel(res: dict, arm: str, shape: str) -> dict:
    return next(s for s in res["shapes"] if s["label"] == shape)["arms"][arm]["panel"]


def test_an_accepted_row_carries_what_panel_py_needs_and_one_run_command_runs_once(tmp_path, capsys):
    run_dir = tmp_path / "run"
    v = script(tmp_path, "v.py", RUN_VERIFIER)
    path = result(run_dir, {"pc": per_run(v), "pcu": per_run(v)})
    shas = {(a, s): manifest(run_dir, a, s, f'{{"arm": "{a}", "shape": "{s}"}}') for a in ("pc", "pcu") for s in SHAPES}
    code, res = run(path)
    assert code == V.EXIT_OK and res["verification"]["ok"]
    assert len(res["verification"]["commands"]) == 1 and res["verification"]["commands"][0]["exit"] == 0
    assert (run_dir / "verify" / "0.out").read_text().count("ACCEPT") == 4
    for (a, s), sha in shas.items():
        p = panel(res, a, s)
        assert p["verification"] == {"accepted": True}
        assert p["verifier_commit"] == COMMIT
        assert p["verifier_accept"] == f"ACCEPT {a}/{s}: transcript sha256:{sha}; verifier {COMMIT}"
        assert p["transcript"] == f"{RUN}:transcripts/{a}/{s}/manifest.json sha256:{sha}"
        args = dict(zip(p["panel_args"][::2], p["panel_args"][1::2]))
        assert re.fullmatch(r"[0-9a-f]{7,40}", args["--verifier-commit"])
        assert "accept" in args["--verifier-accept"].lower() and RUN in args["--transcript"]
        assert args["--run"] == RUN and args["--kind"] == "measured" and args["--precision"] == "fp8"
        assert args["--sm-clock-arm"] == "2090,2092" and args["--sm-clock-base"] == "2091,2093"
        assert args.get("--decode-method") == ("dependent-chain" if s == DECODE else None)
    out = capsys.readouterr().out
    assert out.count("python panel.py append") == 4 and "not measured" not in out


def test_a_rejected_missing_or_unnamed_transcript_is_not_accepted_and_the_step_fails(tmp_path):
    run_dir = tmp_path / "run"
    v = script(tmp_path, "v.py", RUN_VERIFIER)
    named = run_dir / "transcripts" / "st" / PREFILL / "manifest.json"
    stale = script(tmp_path, "stale.py", "import hashlib\n"
                                         f"print('ACCEPT another transcript sha256:{'0' * 64}')\n"
                                         f"sha = hashlib.sha256(open({str(named)!r}, 'rb').read()).hexdigest()\n"
                                         "print('NOT ACCEPT, only named: sha256:' + sha)\n")
    stale_decl = per_run(stale) | {"commands": [["{python}", str(stale)]]}
    path = result(run_dir, {"good": per_run(v), "bad": per_run(v), "st": stale_decl})
    manifest(run_dir, "good", PREFILL)
    manifest(run_dir, "good", DECODE)
    manifest(run_dir, "bad", PREFILL, '{"bad": 1}')
    manifest(run_dir, "st", PREFILL)
    code, res = run(path)
    assert code == V.EXIT_REJECTED and not res["verification"]["ok"]
    assert panel(res, "good", PREFILL)["verification"]["accepted"]
    assert panel(res, "good", DECODE)["verification"]["accepted"]
    bad = panel(res, "bad", PREFILL)
    assert bad["verification"]["why"] == "the verifier rejected it" and bad["verification"]["reject_line"].startswith("REJECT")
    assert "verifier_accept" not in bad and "panel_args" not in bad
    assert panel(res, "bad", DECODE)["verification"]["why"].startswith("no transcript at ")
    st = panel(res, "st", PREFILL)
    assert st["verification"]["why"].startswith("no line starting ACCEPT names sha256:") and "transcript" not in st
    assert [c["exit"] for c in res["verification"]["commands"]] == [1, 0]


def test_per_shape_commands_run_for_each_shape_and_a_failing_one_rejects_only_its_shape(tmp_path, capsys):
    run_dir = tmp_path / "run"
    v = script(tmp_path, "r.py", SHAPE_VERIFIER)
    path = result(run_dir, {"c4": per_shape(v)})
    for s, text in ((PREFILL, "good dump"), (DECODE, "bad dump")):
        (run_dir / "transcripts" / f"c4-{s}").mkdir(parents=True)
        (run_dir / "transcripts" / f"c4-{s}" / "dump").write_text(text)
    code, res = run(path)
    assert code == V.EXIT_REJECTED
    cmds = res["verification"]["commands"]
    assert len(cmds) == 4 and [c["exit"] for c in cmds] == [0, 0, 0, 1]
    assert cmds[1]["argv"][-2:] == [f"{RUN}:{PREFILL}", COMMIT]
    p = panel(res, "c4", PREFILL)
    t = run_dir / "transcripts" / f"c4-{PREFILL}" / "transcript.json"
    sha = hashlib.sha256(t.read_bytes()).hexdigest()
    assert json.loads(t.read_text()) == {"dump": "good dump", "run": RUN}
    assert p["transcript"] == f"{RUN}:transcripts/c4-{PREFILL}/transcript.json sha256:{sha}"
    assert p["verifier_accept"] == f"ACCEPT replay {RUN}:{PREFILL} sha256:{sha} verifier={COMMIT}"
    assert not panel(res, "c4", DECODE)["verification"]["accepted"]
    assert "c4 m32-n8192-k8192: not measured: the verifier rejected it" in capsys.readouterr().out


def test_a_failing_prove_step_stops_its_shape(tmp_path):
    run_dir = tmp_path / "run"
    v = script(tmp_path, "r.py", SHAPE_VERIFIER)
    path = result(run_dir, {"c4": per_shape(v)})
    code, res = run(path)
    assert code == V.EXIT_REJECTED
    assert [c["argv"][2] for c in res["verification"]["commands"]] == ["prove", "prove"]
    for s in SHAPES:
        assert panel(res, "c4", s)["verification"]["why"].startswith("no transcript at ")


def test_an_arm_without_a_verifier_or_clocks_gets_no_measured_row(tmp_path):
    run_dir = tmp_path / "run"
    v = script(tmp_path, "v.py", RUN_VERIFIER)
    path = result(run_dir, {"none": None, "pc": per_run(v)}, clocks=False)
    for s in SHAPES:
        manifest(run_dir, "pc", s)
    code, res = run(path)
    assert code == V.EXIT_REJECTED
    assert panel(res, "none", PREFILL)["verification"]["why"] == "the arm names no verifier (arm interface v0.4)"
    p = panel(res, "pc", PREFILL)
    assert p["verification"] == {"accepted": True, "not_appendable": "x"} and "panel_args" not in p
    assert p["transcript"].startswith(f"{RUN}:transcripts/pc/{PREFILL}/manifest.json sha256:")


def test_a_run_without_a_run_id_or_a_commit_is_a_usage_error(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    v = script(tmp_path, "v.py", RUN_VERIFIER)
    path = result(run_dir, {"pc": per_run(v)})
    monkeypatch.delenv("RESEARCH_RUN_ID", raising=False)
    monkeypatch.delenv("RESEARCH_SOURCE_SHA", raising=False)
    assert V.main([str(path), "--commit", COMMIT]) == V.EXIT_USAGE
    assert V.main([str(path), "--run-id", RUN]) == V.EXIT_USAGE
    assert V.main([str(path), "--run-id", RUN, "--commit", "unknown"]) == V.EXIT_USAGE
    assert "verification" not in json.loads(path.read_text())
    monkeypatch.setenv("RESEARCH_RUN_ID", RUN)
    monkeypatch.setenv("RESEARCH_SOURCE_SHA", COMMIT[:12])
    for s in SHAPES:
        manifest(run_dir, "pc", s)
    out = tmp_path / "verified.json"
    assert V.main([str(path), "--out", str(out)]) == V.EXIT_OK
    assert panel(json.loads(out.read_text()), "pc", PREFILL)["verifier_commit"] == COMMIT[:12]
    assert "verification" not in json.loads(path.read_text())


def test_a_declared_commit_overrides_the_run_commit(tmp_path):
    run_dir = tmp_path / "run"
    v = script(tmp_path, "v.py", RUN_VERIFIER)
    path = result(run_dir, {"pc": per_run(v) | {"commit": "abcdef1"}})
    for s in SHAPES:
        manifest(run_dir, "pc", s)
    code, res = run(path)
    assert code == V.EXIT_OK
    assert panel(res, "pc", PREFILL)["verifier_commit"] == "abcdef1"
    assert res["verification"]["commands"][0]["argv"][-1] == "abcdef1"


GOOD = {"transcript": "transcripts/{arm}/{shape}/t.json", "commands": [["{python}", "v.py", "{transcript}", "{commit}"]]}


@pytest.mark.parametrize("bad, why", [
    ({"per": "tile"}, "per 'tile'"),
    ({"transcript": "/abs/{arm}/{shape}"}, "inside the run's directory"),
    ({"transcript": "../x/{arm}/{shape}"}, "inside the run's directory"),
    ({"transcript": "t/{arm}.json"}, "must name {arm} and {shape}"),
    ({"transcript": "t/{arm}/{shape}/{run_id}"}, "must name {arm} and {shape}"),
    ({"commands": []}, "non-empty list"),
    ({"commands": [[]]}, "non-empty list"),
    ({"commands": [["v.py", 3]]}, "non-empty list"),
    ({"per": "run", "commands": [["{python}", "{shape}"]]}, "use ['shape']"),
    ({"commands": [["{bogus}"]]}, "use ['bogus']"),
    ({"commands": [["{}"]]}, "use ['']"),
    ({"commands": [["{python"]]}, "bad placeholder"),
    ({"commit": "unknown"}, "git commit"),
])
def test_check_verifier_refuses_a_bad_declaration(bad, why):
    with pytest.raises(A.ArmError, match=re.escape(why)):
        A.check_verifier("x", {"verifier": GOOD | bad})


def test_check_verifier_normalizes_a_good_one_and_allows_none():
    assert A.check_verifier("x", {}) is None
    v = A.check_verifier("x", {"verifier": GOOD | {"name": "the audit"}})
    assert v == {"per": "shape", "transcript": GOOD["transcript"], "commands": GOOD["commands"], "commit": None,
                 "name": "the audit", "control": None}
    assert A.check_verifier("x", {"verifier": v}) == v
