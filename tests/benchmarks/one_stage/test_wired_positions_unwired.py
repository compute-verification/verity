"""The periods pair (writer ``Gemm_v2{K=16,N=16}``, reader ``Gemm_v2{K=16,N=8}`` whose ``x`` is the writer's ``o0``) staged
under the order-keeping cut, each member through `verity-verify-proof statement` with ``--program`` the cut and
``--wired`` the record's wired Program: the reader's ``x`` is a tree port with ``param`` and no link row, so the verifier
derives its position (`Registered.wiredReads`) and refuses a statement that names another. The two meet by digest:
SPOT's cut check (`Spot.Cut.checkWith`, the spot difftest's `cut_why`) holds ``--wired`` and the registered circuit
record to ``--program``, and Flock refuses a ``--wired`` whose digest is not the one it bound (``--wired-digest``)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from verity.commitments import randomness as RND
from verity.circuits.codec import canonical_json, descriptor_id, encode_program, program_digest
from verity.circuits.defs import bind
from verity.circuits.evaluation import evaluate_batch
from verity.circuits.ops import gemm as G
from verity.circuits.population import wired_population_program
from verity.commitments.periods import interface as IF
from verity.protocols.proofs.spot.prover import docket as R

from test_audit_two_members import RUN, fixed

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "benchmarks" / "one_stage"))
import served_cases as SC  # noqa: E402

pytestmark = pytest.mark.slow

WRITER = bind(G.Gemm, K=16, N=16, **G.GEMM_AMPERE)
READER = bind(G.Gemm, K=16, N=8, **G.GEMM_AMPERE)
WID, RID = descriptor_id(WRITER), descriptor_id(READER)
WIRE = IF.Wire(RID, "x", WID, "o0", at=0, radix=(1,), base=0, step=(1,))


def _bf16(a: np.ndarray) -> np.ndarray:
    return (np.asarray(a, np.float32).view(np.uint32) >> 16).astype(np.uint16)


def _wired(c: IF.Circuit):
    return wired_population_program(list(c.templates), {**c.description(), "external": list(c.external)})[0]


class Staged:
    def __init__(self, exe: Path, tmp: Path, members: list, honest: dict, digest: str, cut: dict):
        self.exe, self.tmp, self.members, self.honest, self.digest, self.cut = exe, tmp, members, honest, digest, cut

    def cut_why(self, wired: str) -> dict:
        """SPOT's cut check of the wired Program in ``wired`` under the registered circuit record."""
        exe = SC.build(ROOT / "verity" / "protocols" / "proofs" / "spot" / "verifier" / "lean", "spot-difftest")
        j = {**self.cut, "wired": json.loads((self.tmp / wired).read_bytes())}
        p = subprocess.run([str(exe)], input=json.dumps({"fn": "cut_why", "input": j}) + "\n", capture_output=True,
                           text=True)
        return json.loads(p.stdout.splitlines()[0])

    def statement(self, k: int, reads: str | None, public: str | None = None, wired: str | None = "wired.json"):
        sd, pub, _ = self.members[k]
        args = ["--circuit", str(sd / "circuit.txt"), "--public", public or str(pub),
                "--partition", str(self.tmp / "partition.json"), "--program", str(self.tmp / "program.json")]
        if wired:
            args += ["--wired", str(self.tmp / wired), "--wired-digest", self.digest]
        if reads:
            args += ["--registered", reads]
        p = subprocess.run([str(self.exe), "statement", *args], capture_output=True, text=True)
        return p.returncode, (p.stdout + p.stderr).strip()

    def reads(self, name: str, **edit) -> str:
        doc = json.loads(json.dumps(self.honest))
        for f, v in edit.items():
            if f in ("template", "field"):
                doc["ports"]["x"]["tree"][f] = v
            else:
                doc["ports"]["x"][f] = v
        path = self.members[1][0] / f"{name}.json"
        path.write_text(json.dumps(doc))
        return str(path)

    def moved_public(self) -> str:
        sd, pub, _ = self.members[1]
        head, body = pub.read_bytes().split(b"\n", 1)
        h = json.loads(head)
        h["registered"]["x"]["positions"] = [p + 1 for p in h["registered"]["x"]["positions"]]
        (sd / "moved.pub").write_bytes(json.dumps(h).encode() + b"\n" + body)
        return str(sd / "moved.pub")


@pytest.fixture(scope="module")
def staged(tmp_path_factory) -> Staged:
    from verity_flock.class_statement import member_stager
    from verity.commitments.periods import flock_inputs as FI, rows as CS
    from verity.protocols.proofs.spot.verifier.one_stage import registered as REG

    exe = SC.build(ROOT / "verity" / "protocols" / "proofs" / "flock" / "verifier", "verity-verify-proof")
    tmp = tmp_path_factory.mktemp("unwired")
    rng = np.random.default_rng(0)
    x, w, w8 = _bf16(rng.standard_normal((1, 16))), _bf16(rng.standard_normal((1, 16, 16)) / 4), \
        _bf16(rng.standard_normal((1, 8, 16)) / 4)
    y = np.asarray(evaluate_batch(WRITER, x, w, kernel="reference"), dtype=np.uint16).reshape(1, 16)
    y8 = np.asarray(evaluate_batch(READER, y, w8, kernel="reference"), dtype=np.uint16).reshape(1, 8)
    circuit = IF.Circuit("wired-two", ((WRITER, 1), (READER, 1)), wires=(WIRE,))
    same_shape = IF.Circuit("wired-same-shape", ((WRITER, 1), (READER, 2)), wires=(WIRE,))
    (tmp / "wired.json").write_bytes(canonical_json(encode_program(_wired(circuit))))
    (tmp / "wired-same-shape.json").write_bytes(canonical_json(encode_program(_wired(same_shape))))

    d = tmp / "window"
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(IF.Circuit, "program", lambda self: wired_population_program(
            list(self.templates), {**self.description(), "external": list(self.external), "wires": []}))
        cut, cut_query = circuit.program()
        win = IF.Window(circuit, RUN, RND.parse_design("stratified:1"), 4)
        win.record(WRITER, [x, w], [y])
        win.record(READER, [None, w8], [y8])
        win.write(win.close(salts_of=fixed(win)), d, {"label": "unwired"}, registrar=R)
    bw = win.window
    assert program_digest(bw.program) == program_digest(cut)
    bind_ = {"partition": bw.partition_digest, "program": bw.partition["program"]}
    roots = {r["port"]: r for r in json.loads((d / "registration.json").read_text())["roots"]}
    want = CS.expected_ports(bw)
    reads = CS.own_reads({}, bw, roots, registered=REG)
    (tmp / "partition.json").write_text(json.dumps(bw.partition, sort_keys=True))
    (tmp / "program.json").write_bytes(canonical_json(encode_program(bw.program)))
    members = []
    for k, (m, defn) in enumerate(zip(bw.members, (WRITER, READER))):
        sd = tmp / f"m{k}"
        sd.mkdir()
        st = member_stager(defn, m, bind_)
        assert "links" not in st.text.split("\n", 2)[1]
        (sd / "circuit.txt").write_text(st.text)
        pub = Path(FI.flock_input_files(d, k, st.text, sd, write=("pub",))["pub"])
        own = None
        if m.key in reads:
            fv = json.loads(pub.read_bytes().split(b"\n", 1)[0])["frame_v3"]["roots"]
            own = REG.bind_ports(reads[m.key], fv, {p: {"root": roots[bw.tree(m, p)]["root"],
                                                        "domain": want[bw.tree(m, p)]["domain"]} for p in fv})
        members.append((sd, pub, own))
    honest = members[1][2]
    assert honest is not None, "the reader has no registered file"
    e = honest["ports"]["x"]
    for f in ("reads", "reads_from", "tree", "param"):
        e.pop(f, None)
    e |= {"tree": {"template": WID, "field": "o0"}, "param": "x", "reads": []}
    record = {**circuit.description(), "external": list(circuit.external)}
    spot_cut = {"cut": json.loads(canonical_json(encode_program(cut))), "record": record, "query": cut_query}
    return Staged(exe, tmp, members, honest, program_digest(_wired(circuit)), spot_cut)


def _refused(rc: int, out: str, *words: str) -> None:
    assert rc != 0, out
    for w in words:
        assert w in out, out


def test_the_writer_member_is_accepted(staged):
    sd, _, own = staged.members[0]
    reads = None
    if own:
        (sd / "reg.json").write_text(json.dumps(own))
        reads = str(sd / "reg.json")
    rc, out = staged.statement(0, reads)
    assert rc == 0, out


def test_the_reader_is_accepted_with_its_position_derived_from_param(staged):
    rc, out = staged.statement(1, staged.reads("honest"))
    assert rc == 0, out


def test_the_tree_port_without_wired_is_refused_by_the_link_rule(staged):
    rc, out = staged.statement(1, staged.reads("honest"), wired=None)
    _refused(rc, out, "a link row reads its writer's whole output row", "HmLinks.checkUnit")


def test_a_stated_position_moved_is_refused(staged):
    rc, out = staged.statement(1, staged.reads("honest"), public=staged.moved_public())
    _refused(rc, out, "does not open", "registered root at position 1")


def test_param_naming_another_parameter_is_refused(staged):
    rc, out = staged.statement(1, staged.reads("param_w", param="w"))
    _refused(rc, out, "is no unit's output in the wired Program")


def test_another_field_is_refused(staged):
    rc, out = staged.statement(1, staged.reads("field_o1", field="o1"))
    _refused(rc, out, "reads field o1", "not that field's first")


def test_another_writer_template_is_refused(staged):
    rc, out = staged.statement(1, staged.reads("writer", template=RID))
    _refused(rc, out, f"reads {RID}'s tree", f"unit 1's writer, unit 0, is {WID}'s")


def test_a_wired_program_of_another_record_of_the_same_shape_is_refused(staged):
    """SPOT's cut check accepts the registered wired Program and refuses the other record's by name (`cut-record`), and
    Flock refuses it given the digest SPOT bound."""
    assert staged.cut_why("wired.json") == {"output": {"refusal": None}}
    assert staged.cut_why("wired-same-shape.json") == {"output": {"refusal": "cut-record"}}
    rc, out = staged.statement(1, staged.reads("honest"), wired="wired-same-shape.json")
    _refused(rc, out, "wired-digest")
