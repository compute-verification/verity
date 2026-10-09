"""A hidden netlist as the recursion's inner statement (``pod/rec_inner.py universal``, ``pod/87-rec-private.sh``): the universal
unit's class statement with plain leaves, its program one row, whose circuit is the class's whatever the netlist, and whose
public file tells two netlists of the class apart only by the salted commitment of their program row; and V*'s fold
(`InnerFold.lean`) taking the verifier's program, so that the setup checks a wired statement's cut wires (Linking)."""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
POD = HERE.parent / "pod"


def test_inner_fold_hands_the_verifiers_program_to_the_setup():
    """`InnerFold.lean --program FILE` reads the verifier's program descriptor and passes it to `Stmt.setupTables`, as
    `verify --program` does: without it a statement with META ``links`` is refused, with it every cut wire is checked."""
    src = (HERE / "lean" / "InnerFold.lean").read_text()
    assert '| "--program" :: f :: rest => (rest, some f)' in src
    assert "(partition := partition) (program := program)" in src
    assert src.index('"--partition" :: f') < src.index('"--program" :: f') < src.index('"os"] =>')


@pytest.mark.slow
def test_universal_staging_hides_the_netlist_but_its_program_row(tmp_path):
    """Two netlists of class ``16,16,8`` staged on the same x and salt key: one circuit (the class's), plain leaves, three
    regions (x, the program, the outputs: rec-v0's eight claims for V*'s algebra), and public files differing in the program
    row's root and its 64-byte commitment, nothing else."""
    root = HERE.parents[3]
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(str(p) for p in (root, root / "verity" / "ml" / "flock" / "python")))
    p = subprocess.run([sys.executable, str(POD / "rec_inner.py"), "universal", "--cls", "16,16,8", "--n", "4", "--out",
                        str(tmp_path)], env=env, capture_output=True, text=True, check=False)
    assert p.returncode == 0, p.stderr[-2000:]
    r = json.loads(p.stdout)
    o = r["other"]
    assert r["leaf_scheme"] == "flock-leaf/sha512-unsalted" and (r["m"], r["k_log"]) == (25, 22)
    assert o["same_circuit"] and o["same_outputs"] and o["netlist_gates_differing"] == 1
    assert r["stage"]["shared_rows"] == {"p0": 4, "p1": 1, "out": 4}
    assert o["header_differing"] == ["frame_v3.roots.p1"]
    assert 0 < o["body_bytes_differing"] <= 64


def _summary():
    spec = importlib.util.spec_from_file_location("rec_private_summary", POD / "rec_private_summary.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _record(o: Path, parts: list[str] | None, verdicts: dict[str, str | None]) -> dict:
    o.mkdir()
    if parts is not None:
        (o / "cap.json").write_text(json.dumps({"parts": parts}))
    for name, why in verdicts.items():
        (o / "outer" / name).mkdir(parents=True)
        v = {"session": "s", "accepted": why == "", "why": why} if why is not None else None
        (o / "outer" / name / "lean.log").write_text(f"VERDICT {json.dumps(v)}\n" if v else "flock-verify: no session\n")
    return _summary().main(o)


SHARED = "refused: a shared-row public file (header shared_rows): a statement form the soundness proof does not cover"


def test_summary_counts_a_form_refusal_as_a_refusal(tmp_path):
    """rec_private_summary fails closed: Lean's refusal of a form its soundness proof does not cover is "Lean refused: form
    not covered", never acceptance; the run's Lean of record accepts only when Lean accepted every part cap.json names in the
    recursive session (NAME-direct, outside it, counts for nothing); and a control refused for its form is not caught, as
    that refusal comes after every other check; a description outside the class, a member of it, is expected accepted."""
    a, b, c, d = (tmp_path / x for x in "abcd")
    s = _record(a, ["L1", "alg-p1"], {"L1": SHARED, "L1-direct": "", "alg-p1": "", "forged-top-L2": SHARED,
                                       "other-alg-p1": "refused: alg/p1: the residuals are not zero", "flipped-top-L1": None,
                                       "outside-alg-p0": ""})
    lean = s["outer"]["L1"]["lean"]
    assert (lean["accepted"], lean["verdict"], lean["why"]) == (False, "Lean refused: form not covered", SHARED)
    r = s["lean_of_record"]
    assert (r["accepted"], r["verdict"]) == (False, "not accepted")
    assert r["refused"] == {"L1": "Lean refused: form not covered"} and r["direct"] == {"L1": "accepted"}
    assert r["controls"] == {"forged-top-L2": {"verdict": "Lean refused: form not covered", "caught": False, "expect": "caught"},
                             "other-alg-p1": {"verdict": "Lean refused: alg/p1: the residuals are not zero", "caught": True,
                                              "expect": "caught"},
                             "flipped-top-L1": {"verdict": "Lean: no verdict", "caught": False, "expect": "caught"},
                             "outside-alg-p0": {"verdict": "accepted", "caught": False,
                                                "expect": "accepted: " + _summary().ACCEPTED["outside-alg-"]}}
    assert s["outer"]["flipped-top-L1"]["lean"]["accepted"] is False
    assert _record(b, ["L1", "alg-p1"], {"L1": "", "alg-p1": ""})["lean_of_record"]["accepted"] is True
    assert _record(c, ["L1", "alg-p1"], {"L1-direct": "", "alg-p1-direct": ""})["lean_of_record"]["refused"] == {
        "L1": "no session in the recursive session", "alg-p1": "no session in the recursive session"}
    assert _record(d, None, {"L1": ""})["lean_of_record"]["accepted"] is False
    assert json.loads((a / "summary.json").read_text())["lean_of_record"] == r
