"""The Lean verifier's checks of V*'s statement family (`Flock.RecVStar`): V*'s statements at an inner session's shape, as
the verifier restates them, are the ones `rec_vstage.stage` writes (each level's `rec-open/l<LANES>-h<H>` over `2 q`
openings, each of the algebra's parts `rec-residuals/m<m>-k<k_log>-c<claims>-p<j>` over two reps, `rec_shape.outer`'s), and a
unit whose name or population is not one of them is refused by the value it differs in: an instance count, a part's shape
against the inner session's, a name V* has no statement under. `verify` refuses one of V*'s statements without the
verifier's own inner shape (`--rec-inner`) or its own partition object and program, and a circuit file that is not byte for
byte the one V* writes for its statement by the byte it differs at. V*'s parts are as many as the verifier's own groups, the
inner shape's claims are two and two per region, and the verifier's registered reads hold exactly the statement's
message-round ports. The honest sessions themselves are replayed on the pod: their statements are
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


def _inner(path: Path, m: int, k: int, c: int, regions: int | None = None) -> Path:
    """`rec-inner`'s line at a shape, of `(c - 2) / 2` regions unless given: an extra claim per region at two link points."""
    path.write_text(json.dumps({"format": "verity/rec-inner/v1", "m": m, "k_log": k, "claims": c,
                                "regions": (c - 2) // 2 if regions is None else regions}))
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


def test_v_stars_parts_are_as_many_as_the_verifiers_groups(tmp_path):
    """V*'s parts at the inner shape (`VStar.parts (spec vsh)`) are as many as the verifier's own packing of its algebra
    (`RecResiduals.groups (structureOf sh) (portSizes sh)`): both are functions of the shape alone, so the count refuses no
    shape at which the two agree, and a shape whose packing refuses, where one residual alone reads more than 64 regions
    (claims 16 at m = 22), is refused, naming V*'s parts."""
    for shape in SHAPES:
        rc, line = _statements(_inner(tmp_path / "inner.json", *shape))
        assert rc == 0 and len([s for s in line["statements"] if s[0].startswith("rec-residuals/")]) == len(
            RR.parts(RR.spec(A.Shape(*shape)))), line
    rc, line = _statements(_inner(tmp_path / "wide.json", 22, 6, 16))
    assert rc == 1 and line["error"] == "V*'s parts: residual 20 alone reads more than 64 regions or 160 sha512x3 slots", line


def test_the_inner_shapes_claims_are_two_and_two_per_region(tmp_path):
    """The inner shape's claims are `2 + 2 · regions` of the inner statement (`rec_vstage.verifier`'s `2 + len(extras)`, a
    claim per region at each of two link points): a shape of one region more or fewer is refused, naming both counts, and
    so is `rec-inner`'s previous format, which named no regions."""
    for regions in (2, 4):
        rc, line = _statements(_inner(tmp_path / "inner.json", 27, 22, 8, regions))
        assert rc == 1 and line["error"] == (
            f"the inner session's claims 8, not two and two per region of its {regions} ({2 + 2 * regions})"), line
    (tmp_path / "v0.json").write_text(json.dumps({"format": "verity/rec-inner/v0", "m": 27, "k_log": 22, "claims": 8}))
    p = subprocess.run([str(BIN), "rec-statements", "--rec-inner", str(tmp_path / "v0.json")], capture_output=True, text=True)
    assert p.returncode == 2 and p.stderr.strip() == "rec-inner: not a verity/rec-inner/v1 object", p.stderr


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
    """META's bindings are the verifier's own files' before the statement is matched to V*'s. Without the verifier's program
    the registered reads' check refuses first; a partition object naming another template, or another program, is refused
    by the partition checks that run before `RecVStar.checkBindings` (`HmRow.checkPartition`, `deriveTemplateUnits`); the
    same descriptor written another way (indented) passes the bindings and reaches the shape check."""
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
    assert v["why"] == ("setup: registered reads: the statement's units are taken as stated, and a read's position binds only "
                        "as far as its unit does: the verifier derives them from its own program (--partition, --program)"), v
    obj = json.loads((d / "partition.json").read_text())
    obj["query"]["params"]["template"] = "RecOpen_v4{LANES=8,H=2}"
    (tmp_path / "partition.json").write_text(json.dumps(obj))
    (tmp_path / "program.json").write_text(json.dumps({**json.loads((d / "program.json").read_text()), "x": 1}))
    for part, prog, why in ((tmp_path / "partition.json", d / "program.json",
                             "setup: partition: the circuit's partition digest is not the verifier's partition object's"),
                            (d / "partition.json", tmp_path / "program.json",
                             "setup: partition: the verifier's program is not the one the partition object names")):
        p = subprocess.run([str(BIN), "statement", *base, "--partition", str(part), "--program", str(prog)],
                           capture_output=True, text=True)
        assert p.returncode == 1 and p.stdout.startswith(why), p.stdout
    desc = json.loads((d / "program.json").read_text())
    (tmp_path / "program.json").write_text(json.dumps(desc, indent=1))
    p = subprocess.run([str(BIN), "statement", *base, "--partition", str(d / "partition.json"), "--program",
                        str(tmp_path / "program.json")], capture_output=True, text=True)
    assert p.returncode == 1 and p.stdout.startswith(
        "setup: rec-open/l8-h1: not one of V*'s statements at the inner session's shape"), p.stdout


def _reads(path: Path, ports: list[list]) -> Path:
    """A `verity/registered-reads/v0` file of `[name, value, reads]` ports (the fields `checkOwn` does not read are fixed)."""
    path.write_text(json.dumps({"format": "verity/registered-reads/v0", "ports": {
        nm: {"value": v, "root": "00" * 64, "leaves": 2, "words": 64, "domain": "00" * 64, "word_bits": 16, "reads": rd}
        for nm, v, rd in ports}}))
    return path


def test_the_registered_reads_message_round_ports_are_the_statements(tmp_path):
    """A part's reads file holds its ports `m<k>` (`rec_residuals.write`: each under `rec-c<k>`, rows 0 and 1), in any key
    order, beside its other ports; a level's holds none. A port read at another row, one the statement lacks, one missing,
    a `rec-c<k>` under another name, and a level's file with a part's port are each refused by the port compared."""
    inner = _inner(tmp_path / "inner.json", 27, 22, 8)
    part = ["--unit", "rec-residuals/m27-k22-c8-p0", "--instances", "2"]
    rc, line = _statements(inner, *part)
    own = line["own"]
    assert rc == 0 and own and all(v == f"rec-c{nm[1:]}" and rd == [0, 1] for nm, v, rd in own), line
    other = [["v", "rec-v", [0, 1]], ["out", "rec-out", [0, 1]]]
    honest = _reads(tmp_path / "honest.json", other + own[::-1])
    rc, line = _statements(inner, *part, "--registered", str(honest))
    assert rc == 0 and line["checked"] == ["rec-residuals/m27-k22-c8-p0", 2], line
    (m0, c0, _), (ml, cl, _) = own[0], own[-1]
    cases = [
        ([[m0, c0, [0, 2]]] + own[1:], f"the registered reads' port {m0} under {c0} reading [0, 2] is not the statement's: "
         f"the statement's is {m0} under {c0} reading [0, 1]"),
        (own + [["m99", "rec-c99", [0, 1]]], f"the registered reads' port m99 under rec-c99 reading [0, 1] is not the "
         f"statement's: the statement has no port m99 ({len(own)} message-round ports)"),
        (own[:-1], f"the statement's message-round port {ml} under {cl} reading [0, 1] is not in the registered reads "
         f"({len(own) - 1} message-round ports)"),
        (own + [["w", c0, [0, 1]]], f"the registered reads' port w under {c0} reading [0, 1] is not the statement's: the "
         f"statement has no port w ({len(own)} message-round ports)"),
    ]
    for k, (ports, why) in enumerate(cases):
        rc, line = _statements(inner, *part, "--registered", str(_reads(tmp_path / f"t{k}.json", other + ports)))
        assert rc == 1 and line["error"] == f"rec-residuals/m27-k22-c8-p0: {why}", (k, line)
    level = ["--unit", "rec-open/l64-h8", "--instances", "436"]
    rc, line = _statements(inner, *level, "--registered", str(_reads(tmp_path / "level.json", other)))
    assert rc == 0 and line["own"] == [], line
    rc, line = _statements(inner, *level, "--registered", str(_reads(tmp_path / "level-m.json", other + own[:1])))
    assert rc == 1 and line["error"] == (f"rec-open/l64-h8: the registered reads' port {m0} under {c0} reading [0, 1] is not "
                                         f"the statement's: the statement has no port {m0} (0 message-round ports)"), line


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
