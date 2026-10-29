"""The Lean verifier's checks of V*'s statement family (`Flock.RecVStar`): V*'s statements at an inner session's shape, as
the verifier restates them, are the ones `rec_vstage.stage` writes (each level's `rec-open/l<LANES>-h<H>` over `2 q`
openings, each of the algebra's parts `rec-residuals/m<m>-k<k_log>-c<claims>-p<j>` over two reps, `rec_shape.outer`'s, and
then RecCap at each later level `l`, `rec-cap/n<N>-l<l>` over both reps' tops, and last RecCap, `rec-cap/n<N>` over
level 0's cap of `N` nodes), and a
unit whose name or population is not one of them is refused by the value it differs in: an instance count, a part's shape
against the inner session's, a name V* has no statement under. `verify` refuses one of V*'s statements without the
verifier's own inner shape (`--rec-inner`) or its own partition object and program, and a circuit file that is not byte for
byte the one V* writes for its statement by the byte it differs at. V*'s parts are as many as the verifier's own groups, the
inner shape's claims are two and two per region, and the verifier's registered reads hold exactly the statement's
message-round ports. A session is the statement the verifier names (`--rec-part J`), and META binds the verifier's own
bindings (`--rec-bindings F`, which `verity_flock.rec_bindings` computes from the partition object and program) exactly,
`program_digests` by value. Candidate 3's session adds RecSparse, R's levels under names of their own
(`rec-open-stacked/l<LANES>-h<H>`) and R's opening at META's shape, each needed. The honest sessions themselves are replayed on the pod: their statements are
hundreds of MB. Each `rec-statements` builds the algebra's parts at its shape, about 5 s, so every test is slow; its runs are
kept on the machine while the verifier and their inputs are unchanged (`keyed.run`)."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import keyed
import pytest
from test_rec_opening_unit import opening
from test_rec_outer import BIN, staged  # noqa: F401
from verity_flock import rec_algebra as A
from verity_flock import rec_bindings as RB
from verity_flock import rec_budget as RBG
from verity_flock import rec_outer as O
from verity_flock import rec_residuals as RR

pytestmark = [pytest.mark.slow,
              pytest.mark.skipif(not BIN.exists() or shutil.which("lake") is None, reason="the Lean verifier is not built")]

# the fixture session's table shape (`test_rec_shape`'s cap) and the 16,16,8 class's (`87-rec-private.sh`, n64)
SHAPES = [(25, 22, 8), (27, 22, 8)]


def _inner(path: Path, m: int, k: int, c: int, regions: int | None = None, holo=None, sparse=None, stacked=None) -> Path:
    """`rec-inner`'s line at a shape, of `(c - 2) / 2` regions unless given: an extra claim per region at two link points;
    ``holo``, when given, its `holo` key (`rec-inner --hide`); ``sparse``, the bucket's RecSparse shape (`rec-inner
    --sparse`), and ``stacked``, R's `(m, claims)` (the inner circuit's META's), candidate 3's."""
    path.write_text(json.dumps({"format": "verity/rec-inner/v2", "m": m, "k_log": k, "claims": c,
                                "regions": (c - 2) // 2 if regions is None else regions, **({"holo": holo} if holo is not None else {}),
                                **({"sparse": list(sparse)} if sparse is not None else {}),
                                **({"stacked": {"m": stacked[0], "claims": stacked[1]}} if stacked is not None else {})}))
    return path


def _statements(inner: Path, *check: str) -> tuple[int, dict]:
    p = keyed.run([str(BIN), "rec-statements", "--rec-inner", str(inner), *check])
    [line] = [json.loads(ln[9:]) for ln in p.stdout.splitlines() if ln.startswith("RECVSTAR ")]
    return p.returncode, line


def _bindings(d: Path, path: Path) -> Path:
    """The verifier's own bindings of the statement staged in ``d``, from its partition object and program."""
    path.write_text(json.dumps(RB.bindings(json.loads((d / "partition.json").read_text()),
                                           json.loads((d / "program.json").read_text()))))
    return path


def _meta(circuit: bytes) -> dict:
    return json.loads(circuit.split(b"\n", 2)[1][5:])


def _v_star(m: int, k: int, c: int, holo: bool = False) -> list[list]:
    """V*'s statements at the shape (holo: over a private inner session), as `rec_vstage.stage` names them and
    `rec_shape.levels` counts their instances."""
    sh = A.Shape(m, k, c, holo=holo)
    out = [[f"rec-open/l{1 << lv.lanes}-h{lv.d - lv.cap}", 2 * lv.q] for lv in sh.sched.levels]
    return out + [[f"rec-residuals/m{m}-k{k}-c{c}-p{j}", 2] for j in range(len(RR.parts(RR.spec(sh))))]


def _caps(m: int, k: int, c: int) -> list[list]:
    """RecCap at the shape, V*'s last statements: at each later level ``l`` both reps' ``2 · 2^cap_l`` nodes of its tops, then
    the ``2^cap₀`` nodes of level 0's cap (`rec_cap.unit_name`)."""
    from verity_flock import rec_cap as RC

    lv = A.Shape(m, k, c).sched.levels
    later = [[RC.unit_name(2 << lv[l].cap, l), 2 << lv[l].cap] for l in range(1, len(lv))]
    return later + [[RC.unit_name(1 << lv[0].cap), 1 << lv[0].cap]]


def _listed(m: int, k: int, c: int, holo: bool = False) -> list[list]:
    """The verifier's whole list at the shape: `_v_star`'s, then the RecCaps."""
    return _v_star(m, k, c, holo) + _caps(m, k, c)


@pytest.mark.parametrize("shape", SHAPES)
def test_the_verifiers_statements_are_the_ones_v_star_stages(tmp_path, shape):
    """The verifier's statements at the shape are V*'s, in order, then RecCap over each later level's tops and over level 0's
    128 nodes, and each passes the check under its own name and count."""
    inner = _inner(tmp_path / "inner.json", *shape)
    rc, line = _statements(inner)
    assert rc == 0 and line["statements"] == _listed(*shape) and _caps(*shape)[-1] == ["rec-cap/n128", 128], line
    for name, n in (line["statements"][0], line["statements"][-2], line["statements"][-1]):
        rc, line = _statements(inner, "--unit", name, "--instances", str(n))
        assert rc == 0 and line["checked"] == [name, n], line


def test_a_unit_off_v_stars_statements_is_refused_by_the_value(tmp_path):
    """At the 16,16,8 class's shape (level 0 is `rec-open/l64-h8` over 436 openings, as its run's cap recorded): another
    instance count, a part named at other claims, a part V*'s algebra does not have, an honest part's name under an inner
    shape of one more region, and RecCap over other than its 128 nodes are each refused by the value compared."""
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
        (inner, "rec-cap/n128", 127, "rec-cap/n128: instances 127, statement says 128"),
        (inner, "rec-cap/n64", 64, "rec-cap/n64: not one of V*'s statements"),
        (_inner(tmp_path / "more.json", 27, 22, 10), "rec-residuals/m27-k22-c8-p0", 2,
         "rec-residuals/m27-k22-c8-p0: shape (m 27, k_log 22, claims 8), the inner session's (m 27, k_log 22, claims 10)"),
    ]
    assert _v_star(27, 22, 8)[0] == ["rec-open/l64-h8", 436]
    for f, name, n, why in cases:
        rc, line = _statements(f, "--unit", name, "--instances", str(n))
        assert rc == 1 and line["error"].startswith(why), (name, n, line)


def test_v_stars_parts_are_as_many_as_the_verifiers_groups(tmp_path):
    """V*'s parts at the inner shape (`VStar.parts (spec vsh)`) are as many as the verifier's own packing of its algebra
    (`RecResiduals.groups (structureOf sh) (portSizes sh)`): both are functions of the shape alone, so the count refuses no
    shape at which the two agree, among them claims 16 at m = 22, whose batched T stands alone (172 slots, past MAX_SLOTS);
    and a shape whose packing refuses, where one residual alone reads more than MAX_LONE_SLOTS (claims 22: 236 slots), is
    refused, naming V*'s parts."""
    for shape in (*SHAPES, (22, 6, 16)):
        rc, line = _statements(_inner(tmp_path / "inner.json", *shape))
        assert rc == 0 and len([s for s in line["statements"] if s[0].startswith("rec-residuals/")]) == len(
            RR.parts(RR.spec(A.Shape(*shape)))), line
    rc, line = _statements(_inner(tmp_path / "wide.json", 22, 6, 22))
    assert rc == 1 and line["error"] == "V*'s parts: residual 26 alone reads more than 64 regions or 223 sha512x3 slots", line


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
    assert p.returncode == 2 and p.stderr.strip() == "rec-inner: not a verity/rec-inner/v2 object", p.stderr


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
    own = ["--rec-part", "0", "--rec-bindings", str(_bindings(d, tmp_path / "bindings.json"))]
    p = subprocess.run([str(BIN), "statement", *files, "--rec-inner", str(inner), *own], capture_output=True, text=True)
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
            "--registered", str(d / "registered.json"), "--rec-inner", str(inner), "--rec-part", "0",
            "--rec-bindings", str(_bindings(d, tmp_path / "bindings.json"))]
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


def test_a_v_star_sessions_registration_past_the_call_bound_is_refused_by_both(staged, tmp_path):  # noqa: F811
    """`rec_outer`'s staged level's registered reads, one port's tree made 65 levels deep and the file made to name 4097
    ports: `verify` under the verifier's inner shape refuses each at setup with `RecBudget.regs`'s message, which
    `rec_budget.regs` returns for the same file, and the honest file passes both."""
    d = staged[0]
    honest = json.loads((d / "registered.json").read_text())
    assert RBG.regs(honest) is None
    inner = _inner(tmp_path / "inner.json", 25, 22, 8)
    sess = tmp_path / "session"
    sess.mkdir()
    (sess / "session.json").write_text("{}")
    first, port = sorted(honest["ports"].items())[0]
    deep = {**honest, "ports": {**honest["ports"], first: {**port, "leaves": 2**64 + 1}}}
    wide = {**honest, "ports": {f"w{i:04}": port for i in range(RBG.MAX_PORTS + 1)}}
    for k, obj in enumerate((deep, wide)):
        why = RBG.regs(obj)
        assert why is not None
        with pytest.raises(ValueError, match="the recursion's call bound covers no more"):
            RBG.check(obj, "rec-open/l8-h1", 8)
        (tmp_path / f"r{k}.json").write_text(json.dumps(obj))
        p = subprocess.run([str(BIN), "verify", "--statement", "verity/flock-circuit", "--circuit", str(d / "circuit.txt"),
                            "--public", str(d / "pub.bin"), "--partition", str(d / "partition.json"),
                            "--program", str(d / "program.json"), "--registered", str(tmp_path / f"r{k}.json"),
                            "--rec-inner", str(inner), "--rec-part", "0",
                            "--rec-bindings", str(_bindings(d, tmp_path / "bindings.json")), "--session", str(sess)],
                           capture_output=True, text=True)
        [v] = [json.loads(ln[8:]) for ln in p.stdout.splitlines() if ln.startswith("VERDICT ")]
        assert p.returncode == 1 and v["why"] == "setup: " + why, v


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


def _l16_h1(d: Path, n: int, name: str | None = None) -> Path:
    """`RecOpen{16, 1}` over ``n`` openings (under ``name``, else `rec-open/l16-h1`) as `compose` writes it under its population
    program's partition, staged in ``d`` (`_staged`): the verifier's own bindings of it."""
    from verity.core.protocols.sampled_proofs.one_stage import partition as P
    from verity_flock import circuit as FC
    from verity_flock import rec_open as R

    defn = R.definition(16, 1)
    program = P.population_program(defn, n)
    part = P.partition(program, P.template_instance_query(defn))
    return _staged(d, FC.compose(O.lowering(16, 1, name), O.subcircuit(16, 1, name), partition=P.digest(part),
                                 program=P.program_sha512(program)), part, program)


@pytest.fixture(scope="module")
def l16_h1(once) -> tuple[bytes, Path]:
    """`rec-open/l16-h1` (V*'s statement 2 at m = 23, its last level, over 142 openings) as `_l16_h1` stages it, once on the
    machine, and the verifier's own bindings of it, from its partition object and program."""
    d = once("vstar-l16-h1", lambda d: _l16_h1(d, 142), lambda: keyed.key(code=[__name__]))
    return (d / "circuit.txt").read_bytes(), d / "bindings.json"


def test_a_v_star_circuit_file_is_the_one_v_star_writes_byte_for_byte(l16_h1, tmp_path):
    """`rec-open/l16-h1` as `compose` writes it, byte-equal to `vstar rec-open` (`test_vstar_difftest`): the verifier passes
    it, and refuses it with a space added inside META, which parses to the same bindings, by the byte the two files differ
    at."""
    assert _v_star(23, 22, 8)[2] == ["rec-open/l16-h1", 142]
    inner = _inner(tmp_path / "inner.json", 23, 22, 8)
    honest, own = l16_h1
    flags = ["--instances", "142", "--rec-part", "2", "--rec-bindings", str(own)]
    (tmp_path / "honest.txt").write_bytes(honest)
    rc, line = _statements(inner, "--circuit", str(tmp_path / "honest.txt"), *flags)
    assert rc == 0 and line["checked"] == ["rec-open/l16-h1", 142], line
    at = honest.index(b'META {"') + len(b"META {")
    (tmp_path / "spaced.txt").write_bytes(honest[:at] + b" " + honest[at:])
    rc, line = _statements(inner, "--circuit", str(tmp_path / "spaced.txt"), *flags)
    assert rc == 1 and line["error"] == (
        f"rec-open/l16-h1: the circuit file ({len(honest) + 1} bytes) is not V*'s ({len(honest)} bytes): they differ at "
        f"byte {at}, line 2"), line


def test_the_verifiers_bindings_are_metas_of_v_stars_honest_statements(staged, l16_h1, tmp_path):  # noqa: F811
    """`rec_bindings.bindings`, from the partition object and program alone, is exactly the three bindings of META that
    V*'s honest statements carry: `rec_outer`'s staged level, and `rec-open/l16-h1` under its population program."""
    d = staged[0]
    keys = ("partition", "program", "program_digests")
    m = _meta((d / "circuit.txt").read_bytes())
    assert json.loads(_bindings(d, tmp_path / "staged.json").read_text()) == dict(zip(keys, (
        m["partition"]["digest"], m["program_sha512"], m["program_digests"])))
    honest, own = l16_h1
    m = _meta(honest)
    assert json.loads(own.read_text()) == dict(zip(keys, (m["partition"]["digest"], m["program_sha512"], m["program_digests"])))
    assert list(m["program_digests"]) == ["RecOpen_v4{LANES=16,H=1}"]


def test_a_session_is_the_statement_the_verifier_names(l16_h1, tmp_path):
    """`rec-open/l16-h1` is V*'s statement 2 at m = 23: as session 2 it passes; as session 1 (another level's) or 0, it is
    refused naming the statement the verifier holds the session for, and past V*'s last statement by the count; a circuit
    checked with no `--rec-part` is refused naming the flag."""
    inner = _inner(tmp_path / "inner.json", 23, 22, 8)
    honest, own = l16_h1
    (tmp_path / "honest.txt").write_bytes(honest)
    ss = _listed(23, 22, 8)
    check = ["--circuit", str(tmp_path / "honest.txt"), "--instances", "142", "--rec-bindings", str(own)]
    rc, line = _statements(inner, *check, "--rec-part", "2")
    assert rc == 0 and line["checked"] == ["rec-open/l16-h1", 142], line
    for j in (0, 1):
        rc, line = _statements(inner, *check, "--rec-part", str(j))
        assert rc == 1 and line["error"] == (
            f"rec-open/l16-h1: the session is V*'s statement {j} (--rec-part), {ss[j][0]}"), (j, line)
    rc, line = _statements(inner, *check, "--rec-part", str(len(ss)))
    assert rc == 1 and line["error"] == (
        f"rec-open/l16-h1: the session is V*'s statement {len(ss)} (--rec-part), but V* has {len(ss)} statements at the "
        "inner session's shape"), line
    rc, line = _statements(inner, *check)
    assert rc == 1 and line["error"] == "rec-open/l16-h1: pass which of V*'s statements this circuit is for (--rec-part J)", line
    rc, line = _statements(inner, "--unit", "rec-open/l16-h1", "--instances", "142", "--rec-part", "1")
    assert rc == 1 and line["error"] == f"rec-open/l16-h1: the session is V*'s statement 1 (--rec-part), {ss[1][0]}", line


def test_meta_binds_the_verifiers_own_program_digests_by_value(l16_h1, tmp_path):
    """A `rec-open/l16-h1` file whose META `program_digests` names the template at another digest, otherwise V*'s file
    under those bindings, is refused against the verifier's own bindings by the value; so is the honest file against
    bindings of another digest, and a circuit checked without `--rec-bindings`, naming the flag."""
    from verity.ml.circuits.codec import canonical_json

    inner = _inner(tmp_path / "inner.json", 23, 22, 8)
    honest, own = l16_h1
    mine = json.loads(own.read_text())
    ((tid, digest),) = mine["program_digests"].items()
    other = ("0" if digest[0] != "0" else "1") + digest[1:]
    tampered = honest.replace(digest.encode(), other.encode())
    assert tampered.count(other.encode()) == 1 and len(tampered) == len(honest)
    (tmp_path / "tampered.txt").write_bytes(tampered)
    (tmp_path / "honest.txt").write_bytes(honest)
    canon = lambda d: canonical_json({tid: d}).decode()
    flags = ["--instances", "142", "--rec-part", "2"]
    rc, line = _statements(inner, "--circuit", str(tmp_path / "tampered.txt"), *flags, "--rec-bindings", str(own))
    assert rc == 1 and line["error"] == (
        f"rec-open/l16-h1: META program_digests {canon(other)}, the verifier's bindings' {canon(digest)}"), line
    (tmp_path / "theirs.json").write_text(json.dumps({**mine, "program_digests": {tid: other}}))
    rc, line = _statements(inner, "--circuit", str(tmp_path / "tampered.txt"), *flags, "--rec-bindings",
                           str(tmp_path / "theirs.json"))
    assert rc == 0 and line["checked"] == ["rec-open/l16-h1", 142], line
    rc, line = _statements(inner, "--circuit", str(tmp_path / "honest.txt"), *flags, "--rec-bindings",
                           str(tmp_path / "theirs.json"))
    assert rc == 1 and line["error"] == (
        f"rec-open/l16-h1: META program_digests {canon(digest)}, the verifier's bindings' {canon(other)}"), line
    rc, line = _statements(inner, "--circuit", str(tmp_path / "honest.txt"), *flags)
    assert rc == 1 and line["error"] == "rec-open/l16-h1: pass the verifier's own bindings (--rec-bindings F)", line


def test_verify_refuses_a_v_star_statement_without_the_session_index_or_the_verifiers_bindings(staged, tmp_path):  # noqa: F811
    """`rec_outer`'s staged level through `statement` and `verify` with the verifier's inner shape, partition object and
    program: without `--rec-part`, or without `--rec-bindings`, it is refused at setup naming the flag; with bindings whose
    `program_digests` value is not META's, by that value."""
    d = staged[0]
    inner = _inner(tmp_path / "inner.json", 25, 22, 8)
    own = _bindings(d, tmp_path / "bindings.json")
    files = ["--statement", "verity/flock-circuit", "--circuit", str(d / "circuit.txt"), "--public", str(d / "pub.bin"),
             "--partition", str(d / "partition.json"), "--program", str(d / "program.json"),
             "--registered", str(d / "registered.json"), "--rec-inner", str(inner)]
    sess = tmp_path / "session"
    sess.mkdir()
    (sess / "session.json").write_text("{}")
    why = {
        "--rec-part": "rec-open/l8-h1: one of V*'s statements; pass which of V*'s statements this session is (--rec-part J, "
                      "its index in rec-statements' list)",
        "--rec-bindings": "rec-open/l8-h1: one of V*'s statements; pass the verifier's own bindings (--rec-bindings F, from "
                          "verity_flock.rec_bindings)"}
    for flags, missing in ((["--rec-bindings", str(own)], "--rec-part"), (["--rec-part", "0"], "--rec-bindings")):
        p = subprocess.run([str(BIN), "verify", *files, *flags, "--session", str(sess)], capture_output=True, text=True)
        [v] = [json.loads(ln[8:]) for ln in p.stdout.splitlines() if ln.startswith("VERDICT ")]
        assert p.returncode == 1 and v["why"] == f"setup: {why[missing]}", (missing, v)
        p = subprocess.run([str(BIN), "statement", *files, *flags], capture_output=True, text=True)
        assert p.returncode == 1 and p.stdout.strip() == f"setup: {why[missing]}", (missing, p.stdout)
    mine = json.loads(own.read_text())
    ((tid, digest),) = mine["program_digests"].items()
    other = ("0" if digest[0] != "0" else "1") + digest[1:]
    (tmp_path / "theirs.json").write_text(json.dumps({**mine, "program_digests": {tid: other}}))
    p = subprocess.run([str(BIN), "statement", *files, "--rec-part", "0", "--rec-bindings", str(tmp_path / "theirs.json")],
                       capture_output=True, text=True)
    assert p.returncode == 1 and p.stdout.strip() == (
        f'setup: rec-open/l8-h1: META program_digests {{"{tid}":"{digest}"}}, the verifier\'s bindings\' '
        f'{{"{tid}":"{other}"}}'), p.stdout


def test_the_verifiers_bindings_are_its_own_partition_objects(staged, tmp_path):  # noqa: F811
    """`RecVStar.checkOwnBindings`: bindings whose partition digest is not the verifier's partition object's are refused by
    that digest, before META is compared with them (which would refuse the same file by META's digest instead)."""
    d = staged[0]
    inner = _inner(tmp_path / "inner.json", 25, 22, 8)
    mine = json.loads(_bindings(d, tmp_path / "bindings.json").read_text())
    other = ("0" if mine["partition"][0] != "0" else "1") + mine["partition"][1:]
    (tmp_path / "theirs.json").write_text(json.dumps({**mine, "partition": other}))
    files = ["--statement", "verity/flock-circuit", "--circuit", str(d / "circuit.txt"), "--public", str(d / "pub.bin"),
             "--partition", str(d / "partition.json"), "--program", str(d / "program.json"),
             "--registered", str(d / "registered.json"), "--rec-inner", str(inner), "--rec-part", "0"]
    p = subprocess.run([str(BIN), "statement", *files, "--rec-bindings", str(tmp_path / "theirs.json")],
                       capture_output=True, text=True)
    assert p.returncode == 1 and p.stdout.strip() == (
        f"setup: rec-open/l8-h1: the verifier's bindings' partition digest {other}, its partition object's "
        f"{mine['partition']}"), p.stdout


def test_the_verdict_says_whether_the_inner_sessions_circuit_is_private(staged, tmp_path):  # noqa: F811
    """`rec-inner`'s `holo` (the verifier's own `--hide`) is the verdict's `private_circuit`, on every VERDICT line verified
    with `--rec-inner` and on no other, and the private shape's list is V*'s holo statements, so `rec_outer`'s staged level
    is refused at either shape as none of them; a `holo` that is not a boolean is refused as the verifier's file reads."""
    d = staged[0]
    files = ["--statement", "verity/flock-circuit", "--circuit", str(d / "circuit.txt"), "--public", str(d / "pub.bin"),
             "--partition", str(d / "partition.json"), "--program", str(d / "program.json"),
             "--registered", str(d / "registered.json"), "--rec-part", "0",
             "--rec-bindings", str(_bindings(d, tmp_path / "bindings.json"))]
    sess = tmp_path / "session"
    sess.mkdir()
    (sess / "session.json").write_text("{}")

    def verdict(*flags):
        p = subprocess.run([str(BIN), "verify", *files, *flags, "--session", str(sess)], capture_output=True, text=True)
        [v] = [json.loads(ln[8:]) for ln in p.stdout.splitlines() if ln.startswith("VERDICT ")]
        return p.returncode, v

    rc, v = verdict()
    assert rc == 1 and "private_circuit" not in v, v
    unlisted = "setup: rec-open/l8-h1: not one of V*'s statements at the inner session's shape (m 25, k_log 22, claims 8)"
    hidden = _inner(tmp_path / "hidden.json", 25, 22, 8, holo=True)
    for holo, f in ((False, _inner(tmp_path / "absent.json", 25, 22, 8)),
                    (False, _inner(tmp_path / "plain.json", 25, 22, 8, holo=False)), (True, hidden)):
        rc, v = verdict("--rec-inner", str(f))
        assert rc == 1 and v["private_circuit"] is holo and v["why"].startswith(unlisted), (holo, v)
    p = subprocess.run([str(BIN), "statement", *files, "--rec-inner", str(hidden)], capture_output=True, text=True)
    assert p.returncode == 1 and p.stdout.startswith(unlisted), p.stdout
    rc, line = _statements(hidden)
    assert rc == 0 and line["statements"] == _listed(25, 22, 8, holo=True), line
    p = subprocess.run([str(BIN), "rec-statements", "--rec-inner", str(_inner(tmp_path / "bad.json", 25, 22, 8, holo=1))],
                       capture_output=True, text=True)
    assert p.returncode == 2 and p.stderr.strip() == "rec-inner: holo is not a boolean", p.stderr


def _with_holo(text: str, holo: bool) -> str:
    """``text`` with META's `RR.HOLO_KEY` as `RR.statement` writes it for ``holo`` (present only when true)."""
    head, meta, rest = text.split("\n", 2)
    m = {k: v for k, v in json.loads(meta[5:]).items() if k != RR.HOLO_KEY}
    if holo:
        m[RR.HOLO_KEY] = True
    return f"{head}\nMETA {json.dumps(m, sort_keys=True, separators=(',', ':'))}\n{rest}"


def test_a_private_inner_sessions_part_is_v_stars_holo_part_byte_for_byte(tmp_path):
    """At a private inner shape (Shape(22, 10, 2), whose one part reads h when holo) the verifier's list is V*'s holo
    statements, and the part as `rec_residuals.statement(..., holo=True)` writes it is V*'s file byte for byte as the
    statement after the levels: it passes, its own ports each `m<k>` under `rec-c<k>` at both reps' rows, h's round
    (`rec_algebra.h_round`) among them. The same file without META's `holo` is refused when it loads (its unit is not the
    plain part's), and the holo file at the plain shape as not V*'s (the plain part reads no h). About two minutes:
    composing a holo part is the only way to reach `RecVStar.circuitFile`'s holo branch with a real part's unit, so no
    cheaper test catches the verifier's holo file drifting from the one the prover commits."""
    m, k, c = 22, 10, 2
    sh = A.Shape(m, k, c, holo=True)
    [p] = RR.parts(RR.spec(sh))
    name = f"rec-residuals/m{m}-k{k}-c{c}-p0"
    st = RR.statement(p.S, name, 2, holo=True)
    own = _staged(tmp_path / "holo", st.text, st.part, st.program)
    hidden = _inner(tmp_path / "hidden.json", m, k, c, holo=True)
    plain = _inner(tmp_path / "plain.json", m, k, c)
    ss = _v_star(m, k, c, holo=True)
    assert ss[-1] == [name, 2] and len(ss) == len(_v_star(m, k, c))
    rc, line = _statements(hidden)
    assert rc == 0 and line["statements"] == ss + _caps(m, k, c), line
    flags = ["--instances", "2", "--rec-part", str(len(ss) - 1), "--rec-bindings", str(own)]
    rc, line = _statements(hidden, "--circuit", str(tmp_path / "holo" / "circuit.txt"), *flags)
    nr = len([nm for nm, _ in p.S["ports"] if nm.startswith("m")])
    assert rc == 0 and line["checked"] == [name, 2], line
    assert line["own"] == [[f"m{i}", f"rec-c{i}", [0, 1]] for i in range(nr)] and nr == len(A.rounds(sh)), line
    assert any(nm == f"m{A.h_round(sh)}" for nm, _ in p.S["ports"]), p.S["ports"]
    text, short = st.text, _with_holo(st.text, False)
    assert _with_holo(text, True) == text and len(short) < len(text)
    (tmp_path / "plain-meta.txt").write_text(short)
    rc, line = _statements(hidden, "--circuit", str(tmp_path / "plain-meta.txt"), *flags)
    assert rc == 1 and line["error"].startswith(f"circuit {name}: unit {name} has "), line
    assert "not its plain part's" in line["error"] and line["error"].endswith("(META holo: false)"), line
    rc, line = _statements(plain, "--circuit", str(tmp_path / "holo" / "circuit.txt"), *flags)
    assert rc == 1 and line["error"].startswith(f"{name}: the circuit file ({len(text)} bytes) is not V*'s"), line


# candidate 3's session (`3f97`'s e2e): the circuit table's shape, the bucket's RecSparse shape and R at `table_m` of it
C3 = (25, 22, 8)
SPARSE = (18, 13, 6, 0)
R_SHAPE = (23, 2)


def _past_v_star(j: int) -> int:
    """Candidate 3's statement ``j`` past V*'s holo statements at its inner shape: RecSparse at 0, R's levels at 1 to 3,
    R's opening at 4."""
    return len(_v_star(*C3, holo=True)) + j


def _c3(m: int, k: int, c: int, sparse: tuple, r: tuple) -> list[list]:
    """Candidate 3's V* list, as `rec_vstage.stage --sparse` names its units: V*'s holo statements (candidate 3's inner
    circuits are private: its sparse phase reads h, which only a holo session sends), then RecSparse over the two reps, R's levels
    over its one stream's openings (`rec_vstage.stage_stacked`), the parts of R's opening (`rec_residuals.stage_opening`), and
    the RecCaps last."""
    from verity_flock import rec_sparse as SP

    sh = A.Shape(r[0], 6, r[1], salt_len=0, opening_only=True)
    return (_v_star(m, k, c, holo=True) + [[SP.unit_name(SP.Shape(*sparse)), 2]]
            + [[O.stacked_name(1 << lv.lanes, lv.d - lv.cap), lv.q] for lv in sh.sched.levels]
            + [[f"rec-opening/m{r[0]}-c{r[1]}-p{j}", 1] for j in range(len(RR.parts(RR.spec(sh))))] + _caps(m, k, c))


def test_candidate_3s_statements_are_the_ones_v_star_stages_and_each_is_needed(tmp_path):
    """At candidate 3's inner shape the verifier's list is `_c3`'s, in order: V*'s, RecSparse, R's levels under names of their
    own (`rec-open-stacked/…`) and R's opening at META's shape; an R level passes under its own name. Refused by name: an R
    level under the circuit table's name (`rec-open/l16-h1`, which no level of the table has, and `rec-open/l16-h3` at R's
    count, which the table's has at another), an opening at another shape, a RecSparse at another, R named without the
    sparse phase whose claims it opens, the sparse phase without R, which opens them, and R at another m than the bucket's
    `table_m`."""
    inner = _inner(tmp_path / "c3.json", *C3, holo=True, sparse=SPARSE, stacked=R_SHAPE)
    rc, line = _statements(inner)
    want = _c3(*C3, SPARSE, R_SHAPE)
    assert rc == 0 and line["statements"] == want, line
    assert [s[0] for s in want[_past_v_star(0):]] == ["rec-sparse/n18-e13-s6-t0", "rec-open-stacked/l64-h4", "rec-open-stacked/l16-h3",
                                        "rec-open-stacked/l16-h1", "rec-opening/m23-c2-p0", "rec-cap/n128-l1",
                                        "rec-cap/n128-l2", "rec-cap/n64-l3", "rec-cap/n128"], want
    rc, line = _statements(inner, "--unit", "rec-open-stacked/l16-h1", "--instances", "71")
    assert rc == 0 and line["checked"] == ["rec-open-stacked/l16-h1", 71], line
    cases = [
        ("rec-open/l16-h1", 71, "rec-open/l16-h1: R's level stacked/L2 is rec-open-stacked/l16-h1, never under the circuit "
         "table's name rec-open/…; rec-open/l16-h1: not one of V*'s statements"),
        ("rec-open/l16-h3", 106, "rec-open/l16-h3: R's level stacked/L1 is rec-open-stacked/l16-h3, never under the circuit "
         "table's name rec-open/…; rec-open/l16-h3: instances 106, statement says 142"),
        ("rec-opening/m24-c2-p0", 1, "rec-opening/m24-c2-p0: opens R at (m 24, claims 2), not at R's shape (m 23, claims 2, "
         "the inner circuit's META's stacked)"),
        ("rec-sparse/n18-e13-s6-t1", 2, "rec-sparse/n18-e13-s6-t1: a RecSparse unit, and the bucket's is "
         "rec-sparse/n18-e13-s6-t0 (rec-inner --sparse)"),
    ]
    for name, n, why in cases:
        rc, line = _statements(inner, "--unit", name, "--instances", str(n))
        assert rc == 1 and line["error"].startswith(why), (name, line)
    for f, why in (
            (_inner(tmp_path / "no-sparse.json", *C3, holo=True, stacked=R_SHAPE),
             "candidate 3's V* list: the inner shape names R's shape (m 23, claims 2) and no RecSparse shape (rec-inner "
             "--sparse): R opens the sparse phase's claims, so RecSparse is missing"),
            (_inner(tmp_path / "no-r.json", *C3, holo=True, sparse=SPARSE),
             "candidate 3's V* list: the inner shape names RecSparse (rec-sparse/n18-e13-s6-t0) and no R's shape (stacked): "
             "R opens the sparse phases' claims, so R is missing"),
            (_inner(tmp_path / "r24.json", *C3, holo=True, sparse=SPARSE, stacked=(24, 2)),
             "candidate 3's V* list: R's m 24 is not rec-sparse/n18-e13-s6-t0's table_m 23")):
        rc, line = _statements(f)
        assert rc == 1 and line["error"] == why, line


def test_a_rec_sparse_reads_file_holds_every_message_port(tmp_path):
    """RecSparse's reads file holds `h` and `claim` (`ownPorts`) and every other message port `<p>` under `rec-sparse-<p>`
    at rows 0 and 1 (`sparseMsgs`, as `rec_vstage.sparse_messages` registers them), beside its `v`; one of those missing,
    read at other rows or under another value is refused by name (`checkOwn`'s third loop), so no message column the unit
    reads is left unregistered."""
    from verity_flock import rec_sparse as SP

    inner = _inner(tmp_path / "c3.json", *C3, holo=True, sparse=SPARSE, stacked=R_SHAPE)
    unit = ["--unit", SP.unit_name(SP.Shape(*SPARSE)), "--instances", "2"]
    rc, line = _statements(inner, *unit)
    own = line["own"]
    assert rc == 0 and [p[0] for p in own] == ["h", "claim"], line
    msgs = [[nm, f"rec-sparse-{nm}", [0, 1]] for nm, _ in SP.message_ports(SP.Shape(*SPARSE))[1:-1]]
    assert [m[0] for m in msgs[:2]] == ["pre0", "pre1"] and msgs[-1][0] == "u", msgs
    other = [["v", "rec-sparse-v", [0, 1]]]
    rc, line = _statements(inner, *unit, "--registered", str(_reads(tmp_path / "honest.json", other + own + msgs[::-1])))
    assert rc == 0 and line["checked"] == [unit[1], 2], line
    missing = lambda q: f"{unit[1]}: the statement's message port {q} under rec-sparse-{q} reading [0, 1] is not in the registered reads"  # noqa: E731
    for k, (ports, q) in enumerate([(msgs[1:], "pre0"), (msgs[:-1], "u"), ([[*msgs[0][:2], [1, 0]]] + msgs[1:], "pre0"),
                                    (msgs[:-1] + [["u", "rec-sparse-vt", [0, 1]]], "u")]):
        rc, line = _statements(inner, *unit, "--registered", str(_reads(tmp_path / f"t{k}.json", other + own + ports)))
        assert rc == 1 and line["error"] == missing(q), (k, line)


def _staged(d: Path, text: str, part: dict, program) -> Path:
    from verity.ml.circuits.codec import canonical_json, encode_program

    d.mkdir(exist_ok=True)
    (d / "circuit.txt").write_text(text)
    (d / "partition.json").write_text(json.dumps(part))
    (d / "program.json").write_bytes(canonical_json(encode_program(program)))
    return _bindings(d, d / "bindings.json")


def test_r_levels_circuit_file_is_v_stars_under_its_own_name(l16_h1, tmp_path):
    """R's last level at m 23 (`rec-open-stacked/l16-h1` over 71 openings) as `rec_outer.stage` writes it with
    `stacked_name` is V*'s file byte for byte, as candidate 3's fourth statement past V*'s; the circuit table's
    `rec-open/l16-h1` file is refused at candidate 3's shape as R's level under the table's name."""
    name = O.stacked_name(16, 1)
    own = _l16_h1(tmp_path / "level", 71, name)
    inner = _inner(tmp_path / "c3.json", *C3, holo=True, sparse=SPARSE, stacked=R_SHAPE)
    assert _c3(*C3, SPARSE, R_SHAPE)[_past_v_star(3)] == [name, 71]
    rc, line = _statements(inner, "--circuit", str(tmp_path / "level" / "circuit.txt"), "--instances", "71", "--rec-part",
                           str(_past_v_star(3)), "--rec-bindings", str(own))
    assert rc == 0 and line["checked"] == [name, 71] and line["own"] == [], line
    honest, table_own = l16_h1
    (tmp_path / "table.txt").write_bytes(honest)
    rc, line = _statements(inner, "--circuit", str(tmp_path / "table.txt"), "--instances", "142", "--rec-part",
                           str(_past_v_star(3)), "--rec-bindings", str(table_own))
    assert rc == 1 and line["error"].startswith(
        "rec-open/l16-h1: R's level stacked/L2 is rec-open-stacked/l16-h1, never under the circuit table's name rec-open/…"), line


def test_rec_sparses_circuit_file_is_v_stars(tmp_path):
    """RecSparse at a small shape (NU 1, ELL 6, S_LOG 1), as `rec_sparse.statement` writes it over the two reps, is V*'s file
    byte for byte as candidate 3's statement after V*'s (its bindings `rec_bindings`' alone, which registers
    RecSparse itself), and its own port is `h` under the table's h round
    (`rec-c38` at m 25, k_log 22) at both reps' rows; at an inner shape naming another RecSparse it is refused by the shape."""
    from verity_flock import rec_sparse as SP

    st = SP.statement(SP.Shape(1, 6, 1, 0), 2)
    own = _staged(tmp_path / "sparse", st.text, st.part, st.program)
    fresh = subprocess.run([sys.executable, "-c", "import json, sys; from verity_flock import rec_bindings as RB; "
                            "print(json.dumps(RB.bindings(*(json.load(open(f)) for f in sys.argv[1:]))))",
                            str(tmp_path / "sparse" / "partition.json"), str(tmp_path / "sparse" / "program.json")],
                           capture_output=True, text=True)
    assert fresh.returncode == 0 and json.loads(fresh.stdout) == json.loads(own.read_text()), fresh.stderr
    j = _past_v_star(0)
    flags = ["--circuit", str(tmp_path / "sparse" / "circuit.txt"), "--instances", "2", "--rec-part", str(j),
             "--rec-bindings", str(own)]
    r22 = (SP.table_m(SP.Shape(1, 6, 1, 0)), 2)
    rc, line = _statements(_inner(tmp_path / "s.json", *C3, holo=True, sparse=(1, 6, 1, 0), stacked=r22), *flags)
    assert rc == 0 and line["checked"] == ["rec-sparse/n1-e6-s1-t0", 2], line
    assert line["own"] == [["h", "rec-c38", [0, 1]], ["claim", "rec-claim", [0, 1]]], line
    assert line["statements"] == _c3(*C3, (1, 6, 1, 0), r22), line
    rc, line = _statements(_inner(tmp_path / "s2.json", *C3, holo=True, sparse=(1, 6, 2, 0), stacked=r22), *flags)
    assert rc == 1 and line["error"] == (
        "rec-sparse/n1-e6-s1-t0: a RecSparse unit, and the bucket's is rec-sparse/n1-e6-s2-t0 (rec-inner --sparse)"), line


def test_r_openings_circuit_file_is_v_stars(once, tmp_path):
    """R's opening at m 23 (`rec-opening/m23-c2-p0`, one part), as `rec_residuals.stage_opening` writes it, is V*'s file byte
    for byte as candidate 3's fifth statement past V*'s, its own ports each `m<i>` under `rec-opening-c<i>` at R's one row. About two
    minutes and 2.2 GB (each side builds the opening's unit; Python's is `test_rec_opening_unit.opening`'s, made once): no
    cheaper test writes Lean's opening unit, so none other catches its bytes drifting from the one the prover commits and
    the Lean verifier recomputes."""
    sh = A.Shape(R_SHAPE[0], 6, R_SHAPE[1], salt_len=0, opening_only=True)
    [p] = RR.parts(RR.spec(sh))
    [f] = opening(once, R_SHAPE[0])
    rc, line = _statements(_inner(tmp_path / "c3.json", *C3, holo=True, sparse=SPARSE, stacked=R_SHAPE), "--circuit", str(f),
                           "--instances", "1", "--rec-part", str(_past_v_star(4)), "--rec-bindings",
                           str(_bindings(f.parent, tmp_path / "bindings.json")))
    nr = len([nm for nm, _ in p.S["ports"] if nm.startswith("m")])
    assert rc == 0 and line["checked"] == ["rec-opening/m23-c2-p0", 1], line
    assert line["own"] == ([[f"m{i}", f"rec-opening-c{i}", [0]] for i in range(nr)]
                           + [[f"claim{k}", "rec-claim", [k]] for k in range(R_SHAPE[1])]), line


def test_a_claim_port_reads_its_own_reps_row_of_rec_claim(tmp_path):
    """`checkOwn` holds which row each claim port reads: RecSparse's `claim` reads `rec-claim` at rep r's row in instance r
    (`rec_sparse.reads`), and R's opening's `claim<k>` at row k (`rec_residuals.stage_opening`). `checkPubRows` holds the
    rows a claim port opens to `rec-claim`'s, not which of them it opens, so a reads file swapping or repeating the reps'
    claims is refused here, by the port compared. RecSparse's file also holds its other message ports, each `<p>` under
    `rec-sparse-<p>` at both reps' rows (`rec_vstage.sparse_messages`), and one missing them is refused by the first."""
    from verity_flock import rec_sparse as SP

    inner = _inner(tmp_path / "c3.json", *C3, holo=True, sparse=SPARSE, stacked=R_SHAPE)
    msgs = [[nm, f"rec-sparse-{nm}", [0, 1]] for nm, _ in SP.message_ports(SP.Shape(*SPARSE))[1:-1]]
    for unit, n, other, honest, bad in (
            ("rec-sparse/n18-e13-s6-t0", 2, [["v", "rec-sparse-v", [0, 1]]] + msgs, [["claim", "rec-claim", [0, 1]]],
             [["claim", "rec-claim", [1, 0]]]),
            ("rec-opening/m23-c2-p0", 1, [], [["claim0", "rec-claim", [0]], ["claim1", "rec-claim", [1]]],
             [["claim0", "rec-claim", [1]], ["claim1", "rec-claim", [1]]])):
        check = ["--unit", unit, "--instances", str(n)]
        rc, line = _statements(inner, *check)
        own = line["own"]
        claims = [p for p in own if p[1] == "rec-claim"]
        rest = [p for p in own if p[1] != "rec-claim"]
        assert rc == 0 and claims == honest, line
        rc, line = _statements(inner, *check, "--registered", str(_reads(tmp_path / "honest.json", other + own)))
        assert rc == 0 and line["checked"] == [unit, n], line
        if msgs[0] in other:
            short = [p for p in other if p != msgs[0]] + own
            rc, line = _statements(inner, *check, "--registered", str(_reads(tmp_path / "short.json", short)))
            (nm, v, rd) = msgs[0]
            assert rc == 1 and line["error"] == (f"{unit}: the statement's message port {nm} under {v} reading {rd} is not in "
                                                 "the registered reads"), line
        rc, line = _statements(inner, *check, "--registered", str(_reads(tmp_path / "bad.json", other + rest + bad)))
        (nm, _, rd), (_, _, want) = bad[0], claims[0]
        assert rc == 1 and line["error"] == (f"{unit}: the registered reads' port {nm} under rec-claim reading {rd} is not "
                                             f"the statement's: the statement's is {nm} under rec-claim reading {want}"), line
