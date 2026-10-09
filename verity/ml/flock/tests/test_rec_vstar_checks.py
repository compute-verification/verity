"""The Lean verifier's checks of V*'s statement family (`Flock.RecVStar`): V*'s statements at an inner session's shape, as
the verifier restates them, are the ones `rec_vstage.stage` writes (each level's `rec-open/l<LANES>-h<H>` over `2 q`
openings, each of the algebra's parts `rec-residuals/m<m>-k<k_log>-c<claims>-p<j>` over two reps, `rec_shape.outer`'s), and a
unit whose name or population is not one of them is refused by the value it differs in: an instance count, a part's shape
against the inner session's, a name V* has no statement under. `verify` refuses one of V*'s statements without the
verifier's own inner shape (`--rec-inner`) or its own partition object and program, and a circuit file that is not byte for
byte the one V* writes for its statement by the byte it differs at. The honest sessions themselves are replayed on the pod: their statements are
hundreds of MB. Each `rec-statements` builds the algebra's parts at its shape, about 5 s, so every test is slow."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from test_rec_outer import BIN, staged  # noqa: F401
from verity_flock import rec_algebra as A
from verity_flock import rec_residuals as RR

pytestmark = [pytest.mark.slow,
              pytest.mark.skipif(not BIN.exists() or shutil.which("lake") is None, reason="the Lean verifier is not built")]

# the fixture session's table shape (`test_rec_shape`'s cap) and the 16,16,8 class's (`87-rec-private.sh`, n64)
SHAPES = [(25, 22, 8), (27, 22, 8)]


def _inner(path: Path, m: int, k: int, c: int) -> Path:
    path.write_text(json.dumps({"format": "verity/rec-inner/v0", "m": m, "k_log": k, "claims": c}))
    return path


def _statements(inner: Path, *check: str) -> tuple[int, dict]:
    p = subprocess.run([str(BIN), "rec-statements", "--rec-inner", str(inner), *check], capture_output=True, text=True)
    [line] = [json.loads(ln[9:]) for ln in p.stdout.splitlines() if ln.startswith("RECVSTAR ")]
    return p.returncode, line


def _v_star(m: int, k: int, c: int) -> list[list]:
    """V*'s statements at the shape, as `rec_vstage.stage` names them and `rec_shape.levels` counts their instances."""
    sh = A.Shape(m, k, c)
    out = [[f"rec-open/l{1 << lv.lanes}-h{lv.d - lv.cap}", 2 * lv.q] for lv in sh.sched.levels]
    return out + [[f"rec-residuals/m{m}-k{k}-c{c}-p{j}", 2] for j in range(len(RR.parts(RR.spec(sh))))]


@pytest.mark.parametrize("shape", SHAPES)
def test_the_verifiers_statements_are_the_ones_v_star_stages(tmp_path, shape):
    """The verifier's statements at the shape are V*'s, in order, and each passes the check under its own name and count."""
    inner = _inner(tmp_path / "inner.json", *shape)
    rc, line = _statements(inner)
    assert rc == 0 and line["statements"] == _v_star(*shape), line
    for name, n in (line["statements"][0], line["statements"][-1]):
        rc, line = _statements(inner, "--unit", name, "--instances", str(n))
        assert rc == 0 and line["checked"] == [name, n], line


def test_a_unit_off_v_stars_statements_is_refused_by_the_value(tmp_path):
    """At the 16,16,8 class's shape (level 0 is `rec-open/l64-h8` over 436 openings, as its run's cap recorded): another
    instance count, a part named at other claims, a part V*'s algebra does not have, and an honest part's name under an inner
    shape of one more claim are each refused by the value compared."""
    inner = _inner(tmp_path / "inner.json", 27, 22, 8)
    parts = len(_v_star(27, 22, 8)) - 4
    cases = [
        (inner, "rec-open/l64-h8", 434, "rec-open/l64-h8: instances 434, statement says 436"),
        (inner, "rec-residuals/m27-k22-c9-p0", 2,
         "rec-residuals/m27-k22-c9-p0: shape (m 27, k_log 22, claims 9), the inner session's (m 27, k_log 22, claims 8)"),
        (inner, "rec-residuals/m27-k22-c8-p0", 4, "rec-residuals/m27-k22-c8-p0: instances 4, statement says 2"),
        (inner, f"rec-residuals/m27-k22-c8-p{parts}", 2,
         f"rec-residuals/m27-k22-c8-p{parts}: not one of V*'s statements at the inner session's shape (m 27, k_log 22, claims 8)"),
        (inner, "rec-open/l64-h9", 436, "rec-open/l64-h9: not one of V*'s statements"),
        (_inner(tmp_path / "more.json", 27, 22, 9), "rec-residuals/m27-k22-c8-p0", 2,
         "rec-residuals/m27-k22-c8-p0: shape (m 27, k_log 22, claims 8), the inner session's (m 27, k_log 22, claims 9)"),
    ]
    assert _v_star(27, 22, 8)[0] == ["rec-open/l64-h8", 436]
    for f, name, n, why in cases:
        rc, line = _statements(f, "--unit", name, "--instances", str(n))
        assert rc == 1 and line["error"].startswith(why), (name, n, line)


def test_a_v_star_statement_is_refused_without_the_verifiers_inner_shape_and_off_it(staged, tmp_path):  # noqa: F811
    """`rec_outer`'s staged level (`rec-open/l8-h1`, four openings: no shape's V* has eight lanes) through the verifier's own
    files: `verify` refuses it at setup without `--rec-inner`, and `statement --rec-inner` refuses it as no statement of V*'s
    at the inner shape."""
    d = staged[0]
    files = ["--statement", "verity/flock-circuit", "--circuit", str(d / "circuit.txt"), "--public", str(d / "pub.bin"),
             "--partition", str(d / "partition.json"), "--program", str(d / "program.json"),
             "--registered", str(d / "registered.json")]
    sess = tmp_path / "session"
    sess.mkdir()
    (sess / "session.json").write_text("{}")
    p = subprocess.run([str(BIN), "verify", *files, "--session", str(sess)], capture_output=True, text=True)
    [v] = [json.loads(ln[8:]) for ln in p.stdout.splitlines() if ln.startswith("VERDICT ")]
    assert p.returncode == 1 and v["why"] == (
        "setup: rec-open/l8-h1: one of V*'s statements; pass the verifier's own inner shape (--rec-inner F, from rec-inner)"), v
    inner = _inner(tmp_path / "inner.json", 25, 22, 8)
    p = subprocess.run([str(BIN), "statement", *files, "--rec-inner", str(inner)], capture_output=True, text=True)
    assert p.returncode == 1 and p.stdout.startswith(
        "setup: rec-open/l8-h1: not one of V*'s statements at the inner session's shape (m 25, k_log 22, claims 8)"), p.stdout


def test_a_v_star_statements_bindings_are_the_verifiers_own_partition_and_program(staged, tmp_path):  # noqa: F811
    """META's bindings are checked against the verifier's own files before the statement is matched to V*'s: without the
    verifier's program `verify --rec-inner` refuses, and a partition object naming another template, or another program,
    is refused by the digest it changes."""
    d = staged[0]
    inner = _inner(tmp_path / "inner.json", 25, 22, 8)
    base = ["--statement", "verity/flock-circuit", "--circuit", str(d / "circuit.txt"), "--public", str(d / "pub.bin"),
            "--registered", str(d / "registered.json"), "--rec-inner", str(inner)]
    sess = tmp_path / "session"
    sess.mkdir()
    (sess / "session.json").write_text("{}")
    p = subprocess.run([str(BIN), "verify", *base, "--partition", str(d / "partition.json"), "--session", str(sess)],
                       capture_output=True, text=True)
    [v] = [json.loads(ln[8:]) for ln in p.stdout.splitlines() if ln.startswith("VERDICT ")]
    assert v["why"] == ("setup: rec-open/l8-h1: one of V*'s statements; pass the verifier's own partition object and program "
                        "(--partition F, --program F)"), v
    obj = json.loads((d / "partition.json").read_text())
    obj["query"]["params"]["template"] = "RecOpen_v4{LANES=8,H=2}"
    (tmp_path / "partition.json").write_text(json.dumps(obj))
    (tmp_path / "program.json").write_bytes((d / "program.json").read_bytes() + b" ")
    for part, prog, why in ((tmp_path / "partition.json", d / "program.json",
                             "setup: partition: the circuit's partition digest is not the verifier's partition object's"),
                            (d / "partition.json", tmp_path / "program.json",
                             "setup: partition: the verifier's program is not the one the partition object names")):
        p = subprocess.run([str(BIN), "statement", *base, "--partition", str(part), "--program", str(prog)],
                           capture_output=True, text=True)
        assert p.returncode == 1 and p.stdout.startswith(why), p.stdout


def test_a_v_star_circuit_file_is_the_one_v_star_writes_byte_for_byte(tmp_path):
    """`rec-open/l16-h1` (V*'s last level at m = 23, over 142 openings) as `compose` writes it, byte-equal to `vstar
    rec-open` (`test_vstar_difftest`): the verifier passes it, and refuses it with a space added inside META, which parses
    to the same bindings, by the byte the two files differ at."""
    from verity_flock import circuit as FC
    from verity_flock import rec_outer as O

    assert _v_star(23, 22, 8)[2] == ["rec-open/l16-h1", 142]
    inner = _inner(tmp_path / "inner.json", 23, 22, 8)
    honest = FC.compose(O.lowering(16, 1), O.subcircuit(16, 1), partition="a" * 128, program="b" * 128).encode()
    (tmp_path / "honest.txt").write_bytes(honest)
    rc, line = _statements(inner, "--circuit", str(tmp_path / "honest.txt"), "--instances", "142")
    assert rc == 0 and line["checked"] == ["rec-open/l16-h1", 142], line
    at = honest.index(b'META {"') + len(b"META {")
    (tmp_path / "spaced.txt").write_bytes(honest[:at] + b" " + honest[at:])
    rc, line = _statements(inner, "--circuit", str(tmp_path / "spaced.txt"), "--instances", "142")
    assert rc == 1 and line["error"] == (
        f"rec-open/l16-h1: the circuit file ({len(honest) + 1} bytes) is not V*'s ({len(honest)} bytes): they differ at "
        f"byte {at}, line 2"), line
