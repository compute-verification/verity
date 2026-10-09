"""A hidden netlist as the recursion's inner statement (``pod/rec_inner.py universal``, ``pod/87-rec-private.sh``): the universal
unit's class statement with plain leaves, whose circuit is the class's whatever the netlist, and whose public file tells two
netlists of the class apart only by the salted commitment of the program row they differ in; and V*'s fold
(`InnerFold.lean`) taking the verifier's program, so that the setup checks a wired statement's cut wires (Linking)."""
from __future__ import annotations

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
    """Two netlists of class ``16,16,8`` staged on the same x and salt key: one circuit (the class's), plain leaves, and public
    files differing in one program row's root and its 64-byte commitment, nothing else."""
    root = HERE.parents[3]
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(str(p) for p in (root, root / "verity" / "ml" / "flock" / "python")))
    p = subprocess.run([sys.executable, str(POD / "rec_inner.py"), "universal", "--cls", "16,16,8", "--n", "4", "--out",
                        str(tmp_path)], env=env, capture_output=True, text=True, check=False)
    assert p.returncode == 0, p.stderr[-2000:]
    r = json.loads(p.stdout)
    o = r["other"]
    assert r["leaf_scheme"] == "flock-leaf/sha512-unsalted" and (r["m"], r["k_log"]) == (25, 22)
    assert o["same_circuit"] and o["same_outputs"] and o["netlist_gates_differing"] == 1
    [row] = o["header_differing"]
    port = row.removeprefix("frame_v3.roots.")
    assert port != row and port != "p0" and r["stage"]["shared_rows"][port] == 1
    assert 0 < o["body_bytes_differing"] <= 64
