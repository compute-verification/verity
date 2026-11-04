"""A recursive session's inner statement whose private circuit's description is registered (`verity_flock.inner_registered`,
PROTOCOL.md §16.16), on a plain-leaves hidden-output RoPE d64 statement shaped as a private circuit's: its ``cs`` port one
table row every instance reads (the description, which the developer registers), ``x`` a row per instance. `register`
takes the table's rows and salts as the registration, so the statement's ``b || c`` and frame-v3 roots are unchanged, and
writes the header's ``registered`` and the verifier's own reads file; `claim` writes another description's statement with
the registration's paths. Composing the 56 MB circuit takes about 30 s, so the module is slow; it catches what no cheaper
test can: the staged files as the verifier reads them."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from verity.ml.commitments import hm96
from verity_flock import circuit as FC
from verity_flock import inner_registered as IR
from verity.core.protocols.sampled_proofs.one_stage import registered as G

pytestmark = pytest.mark.slow
N = 8
LEAN = Path(__file__).resolve().parents[3] / "core" / "service" / "flock"
BIN = LEAN / ".lake" / "build" / "bin" / "flock-verify"
STATEMENT = "verity/flock-circuit+plain-leaves"


@pytest.fixture(scope="module")
def staged(tmp_path_factory):
    """The plain-leaves circuit, the instances' x rows, the description row and another of the class's width."""
    pytest.importorskip("verity_numerical")
    from verity_numerical.bench import templates as TM
    from verity_numerical.bench.generate import generate
    from verity_numerical.bench.input_sets import InputSet

    d = tmp_path_factory.mktemp("inner-registered")
    s = InputSet.open(generate(TM.rope_head(64), N, 20261009, d / "set")["path"])
    text = FC.compose(FC.lowering_for_set(s), s.subcircuit, statics=FC.set_statics(s), leaf_scheme=FC.LEAF_SCHEME_PLAIN)
    (d / "circuit.txt").write_text(text)
    rng = np.random.default_rng(20261009)
    return SimpleNamespace(dir=d, text=text, x=rng.integers(0, 1 << 16, (N, 64)), desc=rng.integers(0, 1 << 16, (1, 64)),
                           other=rng.integers(0, 1 << 16, (1, 64)), x_salts=[hm96.SHA512.fresh_salt() for _ in range(N)],
                           out_salts=[hm96.SHA512.fresh_salt() for _ in range(N)], meta=json.loads(text.split("\n", 2)[1][5:]),
                           template=s.subcircuit.definition(**FC.set_statics(s)))


def stage(s, name: str, cs=None, refs=None, salts=None, text: str | None = None) -> Path:
    """The statement in ``s.dir/name``: the description table ``cs`` (default the description row), each instance's ref into
    it (default row 0), under ``salts`` (default fresh); ``text`` the circuit, if not the staged one."""
    d = s.dir / name
    d.mkdir()
    text = text or s.text
    (d / "circuit.txt").write_text(text)
    cs = s.desc if cs is None else np.asarray(cs)
    inputs = SimpleNamespace(name="inner-registered", subcircuit=SimpleNamespace(id=s.meta.get("subcircuit", "rope-head/d64/neox-bf16")))
    FC.write(text, inputs, 0, N, d / f"inst-{N}.bin", d / f"pub-{N}.bin", None, list(range(N)), share_rows=True,
             outs={"out": np.zeros((N, 64), np.uint16)}, tables={"x": s.x, "cs": cs}, refs={"x": list(range(N)), "cs": refs or [0] * N},
             salts={"x": s.x_salts, "cs": salts or [hm96.SHA512.fresh_salt() for _ in cs], "out": s.out_salts})
    return d


def head(d: Path, f: str = "pub") -> dict:
    return json.loads((d / f"{f}-{N}.bin").read_bytes().split(b"\n", 1)[0])


def partitioned(s):
    """The circuit with META naming the statement's program (`population_program` of the template at N) and its
    ``verity/partition/v1`` object, as `rec_inner.stage` binds it; the program and the object."""
    from verity.core.protocols.sampled_proofs.one_stage import partition as P

    program = P.population_program(s.template, N)
    part = P.partition(program, P.template_instance_query(s.template))
    head_, line, rest = s.text.split("\n", 2)
    meta = dict(s.meta, program_sha512=part["program"], partition={"rule": P.FORMAT, "digest": P.digest(part)})
    return head_ + "\nMETA " + json.dumps(meta, sort_keys=True, separators=(",", ":")) + "\n" + rest, program, part


def test_register_commits_the_description_table_as_its_registration(staged):
    d = stage(staged, "honest")
    before = head(d)
    got = IR.register(d)
    assert set(got) == {"cs"} and got["cs"]["leaves"] == 1 and got["cs"]["positions"] == [0]
    h, hp = head(d), head(d, "inst")
    assert h == dict(hp, rows=False) and h["frame_v3"] == before["frame_v3"], "the rows, salts and roots are the staged ones"
    assert h["registered"] == {"cs": {"value": IR.VALUE, "positions": [0], "paths": [""]}}
    own = json.loads((d / IR.OWN).read_text())
    e = own["ports"]["cs"]
    dom = G.domain(staged.meta["program_sha512"], IR.VALUE, 64, 1)
    assert own["format"] == G.READS and e["reads"] == [0] * N and (e["leaves"], e["words"], e["word_bits"]) == (1, 64, 16)
    assert e["domain"] == dom.domain_id.hex() and e["root"] == got["cs"]["root"]
    _, tabs = IR.tables(d)
    bc = tabs["cs"]["bcs"][0]
    assert G.verify_read(dom, bytes.fromhex(e["root"]), 0, bc, b""), "the statement's b || c opens the registered root at 0"
    assert G.Registered.commit(staged.meta["program_sha512"], IR.VALUE, staged.desc.tolist(), tabs["cs"]["salts"]).bcs == (bc,)


def test_the_description_port_is_the_one_row_every_instance_reads(staged):
    d = stage(staged, "two-rows", cs=np.concatenate([staged.desc, staged.other]), refs=[0, 1] * (N // 2))
    with pytest.raises(ValueError, match="no one description port"):
        IR.register(d)
    got = IR.register(d, {"cs": "desc"})
    assert got["cs"]["leaves"] == 2 and head(d)["registered"]["cs"]["value"] == "desc"
    assert json.loads((d / IR.OWN).read_text())["ports"]["cs"]["reads"] == [0, 1] * (N // 2)


def test_extra_rows_are_registered_past_the_table(staged):
    d = stage(staged, "extra")
    got = IR.register(d, extra=1)
    e = head(d)["registered"]["cs"]
    assert got["cs"]["leaves"] == 2 and e["positions"] == [0] and len(bytes.fromhex(e["paths"][0])) == 64


def test_register_with_the_statements_partition_binds_the_circuit_registration(staged):
    """Given the statement's program and partition object, `register`'s entries join the circuit registration
    (`inner_registered.bind`): one-stage's ``verity/registration/v2`` record over the partition object META names, written
    beside the circuit, its digest META's ``registration`` and both files' headers the rewritten circuit's (its SHA-512 and
    class); the rows, salts and roots are the staged ones. A partition object META doesn't name, or a program without its
    partition object, is refused."""
    from verity.core.protocols.sampled_proofs.one_stage import partition as P, registration as R

    text, program, part = partitioned(staged)
    d = stage(staged, "bound", text=text)
    before = head(d)
    got = IR.register(d, program=program, partition=part)
    rec, now = IR.bound(d), (d / "circuit.txt").read_text()
    meta = IR.meta(d)
    assert meta[IR.REGISTRATION_KEY] == R.digest(rec) and {k: v for k, v in meta.items() if k != IR.REGISTRATION_KEY} == \
        json.loads(text.split("\n", 2)[1][5:])
    assert rec == IR.registration_record(program, part, rec["registered"]) and rec["format"] == R.FORMAT_V2
    assert [(e["value"], e["root"]) for e in rec["registered"]] == [(IR.VALUE, got["cs"]["root"])]
    h = head(d)
    assert h == dict(head(d, "inst"), rows=False) and h["frame_v3"] == before["frame_v3"]
    assert h["circuit_sha512"] == FC.digest(now) and h["units"]["classes"] == [FC.class_digest(now)] * N
    other = stage(staged, "bound-other", text=text)
    with pytest.raises(ValueError, match="META binds program"):
        smaller = P.population_program(staged.template, N - 1)
        IR.register(other, program=smaller, partition=P.partition(smaller, P.template_instance_query(staged.template)))
    with pytest.raises(ValueError, match="program and its partition object together"):
        IR.register(other, program=program)


def test_claim_writes_another_description_with_the_registered_paths(staged):
    honest, other = stage(staged, "claimed-honest"), stage(staged, "claimed-other", cs=staged.other)
    IR.register(honest)
    assert IR.claim(other, honest) == head(honest)["registered"]
    assert head(other)["registered"] == head(honest)["registered"] and head(other) == dict(head(other, "inst"), rows=False)
    _, tabs = IR.tables(other)
    root = bytes.fromhex(json.loads((honest / IR.OWN).read_text())["ports"]["cs"]["root"])
    assert not G.verify_read(G.domain(staged.meta["program_sha512"], IR.VALUE, 64, 1), root, 0, tabs["cs"]["bcs"][0], b"")


# --- the Lean verifier's description check (`Registered.checkDescription`, D14) -------------------------------------------

@pytest.fixture(scope="module")
def verifier():
    if shutil.which("lake") is None:
        pytest.skip("no Lean toolchain")
    subprocess.run(["lake", "build", "flock-verify"], cwd=LEAN, check=True, capture_output=True)
    return BIN


def statement(d: Path, own: Path | None, partition: Path | None = None) -> tuple[int, str]:
    args = [str(BIN), "statement", "--statement", STATEMENT, "--circuit", str(d / "circuit.txt"), "--public", str(d / f"pub-{N}.bin")]
    args += ["--registered", str(own)] if own else []
    p = subprocess.run(args + (["--partition", str(partition)] if partition else []), capture_output=True, text=True, check=False)
    return p.returncode, (p.stdout + p.stderr).strip()


def test_an_honest_inner_statement_reads_its_description_from_the_registration(staged, verifier):
    d = stage(staged, "lean-honest")
    got = IR.register(d)
    code, out = statement(d, d / IR.OWN)
    assert code == 0, out
    line = json.loads(out.splitlines()[0])
    assert line["registered"] == {"cs": {"value": IR.VALUE, "root": got["cs"]["root"]}}
    assert json.loads(line["identity"])["leaf_scheme"] == FC.LEAF_SCHEME_PLAIN
    code, out = statement(d, None)
    assert code == 1 and "the verifier holds no registered roots" in out, out


def test_a_statement_whose_meta_binds_its_circuit_registration_is_set_up(staged, verifier):
    """META's ``registration`` is not a binding key: the Lean verifier sets up the statement whose circuit carries it, and
    the statement's identity holds the rewritten circuit's SHA-512."""
    text, program, part = partitioned(staged)
    d = stage(staged, "lean-bound", text=text)
    IR.register(d, program=program, partition=part)
    (d / "partition.json").write_text(json.dumps(part))
    code, out = statement(d, d / IR.OWN, d / "partition.json")
    assert code == 0, out
    assert FC.digest((d / "circuit.txt").read_text()) in out


def test_an_inner_statement_naming_no_registered_rows_is_refused(staged, verifier):
    honest, plain = stage(staged, "lean-registered"), stage(staged, "lean-plain")
    IR.register(honest)
    assert statement(plain, None)[0] == 0, "without the verifier's registered reads, a statement naming none is set up"
    code, out = statement(plain, honest / IR.OWN)
    assert code == 1 and "names no registered rows" in out, out


def test_another_description_under_the_registered_paths_is_refused(staged, verifier):
    honest, other = stage(staged, "lean-claim-honest"), stage(staged, "lean-claim-other", cs=staged.other)
    IR.register(honest)
    IR.claim(other, honest)
    code, out = statement(other, honest / IR.OWN)
    assert code == 1 and f"port cs: table row 0 does not open {IR.VALUE}'s registered root at position 0" in out, out


def test_a_registered_row_past_the_table_is_refused(staged, verifier):
    d = stage(staged, "lean-extra")
    IR.register(d, extra=1)
    code, out = statement(d, d / IR.OWN)
    assert code == 1 and f"port cs: its table has 1 rows, and {IR.VALUE}'s registration 2" in out, out


def test_a_table_row_no_instance_reads_is_refused(staged, verifier):
    d = stage(staged, "lean-unread", cs=np.concatenate([staged.desc, staged.other]))
    IR.register(d, {"cs": IR.VALUE})
    code, out = statement(d, d / IR.OWN)
    assert code == 1 and f"port cs: none of the {N} instances reads table row 1, registered row 1 of {IR.VALUE}'s 2" in out, out


def test_a_description_table_out_of_order_is_refused(staged, verifier):
    d = stage(staged, "lean-order", cs=np.concatenate([staged.desc, staged.other]), refs=[0, 1] * (N // 2))
    IR.register(d, {"cs": IR.VALUE})
    assert statement(d, d / IR.OWN)[0] == 0, "a two-row description read by the instances in turn is the registration"
    e = head(d)["registered"]["cs"]
    IR._write_header(d, {"cs": {**e, "positions": [1, 0], "paths": e["paths"][::-1]}})
    code, out = statement(d, d / IR.OWN)
    assert code == 1 and "port cs: table row 0 states position 1" in out, out


def test_the_registered_ports_are_exactly_the_verifiers(staged, verifier):
    d = stage(staged, "lean-ports")
    IR.register(d)
    own = json.loads((d / IR.OWN).read_text())
    (d / "renamed.json").write_text(json.dumps({**own, "ports": {"x": own["ports"]["cs"]}}))
    code, out = statement(d, d / "renamed.json")
    assert code == 1 and "names port cs registered, and the verifier's registered reads do not" in out, out
