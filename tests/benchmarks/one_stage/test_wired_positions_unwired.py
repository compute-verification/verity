"""Scratch (not for main): the periods pair (writer ``Gemm_v2{K=16,N=16}``, reader ``Gemm_v2{K=16,N=8}`` whose ``x`` is the
writer's ``o0``) staged as a5 stages it under the order-keeping cut (the wired builder over the record with no wires), each
member through `verity-verify-proof statement` with ``--program`` the cut and ``--wired`` the record's wired Program: the
reader's ``x`` a tree port with ``param`` and no link row, its position derived (`Registered.wiredReads`). Prints each case's
verdict; asserts nothing."""
from __future__ import annotations

import json
import subprocess
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
from verity.protocols.proofs.spot.verifier.one_stage import docket as R

from test_audit_two_members import LEAN, RUN, fixed

WRITER = bind(G.Gemm, K=16, N=16, **G.GEMM_AMPERE)
READER = bind(G.Gemm, K=16, N=8, **G.GEMM_AMPERE)
WID, RID = descriptor_id(WRITER), descriptor_id(READER)
DESIGN = "stratified:1"
WIRE = IF.Wire(RID, "x", WID, "o0", at=0, radix=(1,), base=0, step=(1,))


def _bf16(a: np.ndarray) -> np.ndarray:
    return (np.asarray(a, np.float32).view(np.uint32) >> 16).astype(np.uint16)


def _wired(c: IF.Circuit):
    return wired_population_program(list(c.templates), {**c.description(), "external": list(c.external)})[0]


def _lean(*args: str) -> tuple[int, str]:
    p = subprocess.run([str(LEAN / "verity-verify-proof"), "statement", *args], capture_output=True, text=True)
    return p.returncode, (p.stdout + p.stderr).strip()


def test_derived_positions_on_the_unwired_window(tmp_path, monkeypatch):
    from verity_flock.class_statement import member_stager
    from verity.commitments.periods import flock_inputs as FI, rows as CS
    from verity.protocols.proofs.spot.verifier.one_stage import registered as REG

    rng = np.random.default_rng(0)
    x, w, w8 = _bf16(rng.standard_normal((1, 16))), _bf16(rng.standard_normal((1, 16, 16)) / 4), \
        _bf16(rng.standard_normal((1, 8, 16)) / 4)
    y = np.asarray(evaluate_batch(WRITER, x, w, kernel="reference"), dtype=np.uint16).reshape(1, 16)
    y8 = np.asarray(evaluate_batch(READER, y, w8, kernel="reference"), dtype=np.uint16).reshape(1, 8)
    circuit = IF.Circuit("wired-two", ((WRITER, 1), (READER, 1)), wires=(WIRE,))
    other = IF.Circuit("wired-other", ((WRITER, 2), (READER, 1)), wires=(WIRE,))
    wired, wired_other = _wired(circuit), _wired(other)
    monkeypatch.setattr(IF.Circuit, "program", lambda self: wired_population_program(
        list(self.templates), {**self.description(), "external": list(self.external), "wires": []}))
    cut = circuit.program()[0]
    print("DIGESTS wired", program_digest(wired)[:16], "cut", program_digest(cut)[:16], "other", program_digest(wired_other)[:16])
    (tmp_path / "wired.json").write_bytes(canonical_json(encode_program(wired)))
    (tmp_path / "wired-other.json").write_bytes(canonical_json(encode_program(wired_other)))

    d = tmp_path / "window"
    win = IF.Window(circuit, RUN, RND.parse_design(DESIGN), 4)
    win.record(WRITER, [x, w], [y])
    win.record(READER, [None, w8], [y8])
    win.write(win.close(salts_of=fixed(win)), d, {"label": "rows-unwired"}, registrar=R)
    bw = win.window
    assert program_digest(bw.program) == program_digest(cut)
    bind_ = {"partition": bw.partition_digest, "program": bw.partition["program"]}
    rec = json.loads((d / "registration.json").read_text())
    roots = {r["port"]: r for r in rec["roots"]}
    want = CS.expected_ports(bw)
    reads = CS.own_reads({}, bw, roots, registered=REG)
    (tmp_path / "partition.json").write_text(json.dumps(bw.partition, sort_keys=True))
    (tmp_path / "program.json").write_bytes(canonical_json(encode_program(bw.program)))
    staged = {}
    for k, (m, defn) in enumerate(zip(bw.members, (WRITER, READER))):
        sd = tmp_path / f"m{k}"
        sd.mkdir()
        st = member_stager(defn, m, bind_)
        (sd / "circuit.txt").write_text(st.text)
        got = FI.flock_input_files(d, k, st.text, sd, write=("pub",))
        header = json.loads(Path(got["pub"]).read_bytes().split(b"\n", 1)[0])
        print(f"M{k}-META-LINKS", "links" in st.text.split("\n", 2)[1], "HEADER-REGISTERED",
              json.dumps({p: {f: v for f, v in e.items() if f != "paths"} for p, e in (header.get("registered") or {}).items()}))
        own = None
        if m.key in reads:
            fv = header["frame_v3"]["roots"]
            own = REG.bind_ports(reads[m.key], fv, {p: {"root": roots[bw.tree(m, p)]["root"],
                                                        "domain": want[bw.tree(m, p)]["domain"]} for p in fv})
        print(f"M{k}-REGISTERED", json.dumps({p: {f: v for f, v in e.items() if f not in ("root", "domain")}
                                              for p, e in ((own or {}).get("ports") or {}).items()}, sort_keys=True))
        staged[k] = (sd, Path(got["pub"]), own)

    base = ["--partition", str(tmp_path / "partition.json"), "--program", str(tmp_path / "program.json")]
    sd0, pub0, own0 = staged[0]
    m0_reads = []
    if own0:
        (sd0 / "reg.json").write_text(json.dumps(own0))
        m0_reads = ["--registered", str(sd0 / "reg.json")]
    rc, out = _lean("--circuit", str(sd0 / "circuit.txt"), "--public", str(pub0), *base, *m0_reads)
    print("M0", rc, out.splitlines()[-1] if out else "")

    sd, pub, own = staged[1]
    if own is None:
        pytest.fail("the reader has no registered file")
    honest = json.loads(json.dumps(own))
    e = honest["ports"]["x"]
    for f in ("reads", "reads_from", "tree", "param"):
        e.pop(f, None)
    e |= {"tree": {"template": WID, "field": "o0"}, "param": "x", "reads": []}
    print("READER-X", json.dumps({f: v for f, v in e.items() if f not in ("root", "domain")}, sort_keys=True))

    def reg(name: str, **edit) -> str:
        doc = json.loads(json.dumps(honest))
        for f, v in edit.items():
            if f in ("template", "field"):
                doc["ports"]["x"]["tree"][f] = v
            else:
                doc["ports"]["x"][f] = v
        (sd / f"{name}.json").write_text(json.dumps(doc))
        return str(sd / f"{name}.json")

    def pub_moved() -> str:
        head, body = pub.read_bytes().split(b"\n", 1)
        h = json.loads(head)
        h["registered"]["x"]["positions"] = [p + 1 for p in h["registered"]["x"]["positions"]]
        (sd / "moved.pub").write_bytes(json.dumps(h).encode() + b"\n" + body)
        return str(sd / "moved.pub")

    def run(name: str, reads: str, wired_file: str = "wired.json", public: str = str(pub), wired_flag: bool = True):
        rc, out = _lean("--circuit", str(sd / "circuit.txt"), "--public", public, *base,
                        *(["--wired", str(tmp_path / wired_file)] if wired_flag else []), "--registered", reads)
        print(f"CASE {name}: rc={rc}: {out.splitlines()[-1] if out else ''}")

    (sd / "explicit.json").write_text(json.dumps(own))
    run("explicit positions, no --wired", str(sd / "explicit.json"), wired_flag=False)
    run("honest", reg("honest"))
    run("tree port without --wired", reg("honest"), wired_flag=False)
    run("another leaf (stated position moved)", reg("honest"), public=pub_moved())
    run("another leaf (param w)", reg("param_w", param="w"))
    run("another field (o1)", reg("field_o1", field="o1"))
    run("another writer template (the reader's)", reg("writer", template=RID))
    run("a different wired Program (record wired-other)", reg("honest"), wired_file="wired-other.json")
