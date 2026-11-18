"""Scratch (not for main): a wired periods window of two templates through the served audit (`records_audit.py audit`,
`a5.audit`, the CPU `flock-circuit`, the Lean verifier), as `test_audit_two_members` runs its unwired one. The writer
``Gemm_v2{K=16,N=16}`` (one unit) and the reader ``Gemm_v2{K=16,N=8}`` (one unit) whose ``x`` is the writer's ``o0``
row, one `IF.Wire`. Prints each member's verdicts and the Lean verifier's messages; asserts nothing about them."""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np

from verity.commitments import randomness as RND
from verity.circuits.codec import descriptor_id
from verity.circuits.defs import bind
from verity.circuits.evaluation import evaluate_batch
from verity.circuits.ops import gemm as G
from verity.commitments.periods import interface as IF
from verity.protocols.proofs.spot.verifier.one_stage import docket as R

from test_audit_two_members import RUN, _audit, binaries, fixed  # noqa: F401

WRITER = bind(G.Gemm, K=16, N=16, **G.GEMM_AMPERE)
READER = bind(G.Gemm, K=16, N=8, **G.GEMM_AMPERE)
WID, RID = descriptor_id(WRITER), descriptor_id(READER)
DESIGN = "stratified:1"
RULE = "stratified-rate:1/2"


def _bf16(a: np.ndarray) -> np.ndarray:
    return (np.asarray(a, np.float32).view(np.uint32) >> 16).astype(np.uint16)


def _window(d: Path) -> Path:
    rng = np.random.default_rng(0)
    x, w, w8 = _bf16(rng.standard_normal((1, 16))), _bf16(rng.standard_normal((1, 16, 16)) / 4), \
        _bf16(rng.standard_normal((1, 8, 16)) / 4)
    y = np.asarray(evaluate_batch(WRITER, x, w, kernel="reference"), dtype=np.uint16).reshape(1, 16)
    y8 = np.asarray(evaluate_batch(READER, y, w8, kernel="reference"), dtype=np.uint16).reshape(1, 8)
    wire = IF.Wire(RID, "x", WID, "o0", at=0, radix=(1,), base=0, step=(1,))
    circuit = IF.Circuit("wired-two", ((WRITER, 1), (READER, 1)), wires=(wire,))
    print("PROGRAM", type(circuit.program()[0]).__name__, json.dumps(circuit.program()[1], default=str)[:400])
    win = IF.Window(circuit, RUN, RND.parse_design(DESIGN), 4)
    win.record(WRITER, [x, w], [y])
    win.record(READER, [None, w8], [y8])
    print("MISMATCHES", win.mismatches())
    win.write(win.close(salts_of=fixed(win)), d, {"label": "green-wired"}, registrar=R)
    return d


def test_each_member_statement_through_the_lean_verifiers_setup(tmp_path):
    """Each member staged as a5 stages it (`member_stager`, `flock_input_files`, `rows.own_reads` bound by
    `REG.bind_ports`, a5's ``partition.json`` and ``program.json``), then `verity-verify-proof statement` on it: the setup
    and registered-read checks `verify` runs before any proof."""
    import subprocess

    from verity_flock.class_statement import member_stager
    from verity.circuits.codec import canonical_json, encode_program
    from verity.commitments.periods import flock_inputs as FI, rows as CS
    from verity.protocols.proofs.spot.verifier.one_stage import registered as REG
    from test_audit_two_members import LEAN

    print("VARIANT", os.environ.get("GREEN_VARIANT", "wired"))
    d = tmp_path / "window"
    rng = np.random.default_rng(0)
    x, w, w8 = _bf16(rng.standard_normal((1, 16))), _bf16(rng.standard_normal((1, 16, 16)) / 4), \
        _bf16(rng.standard_normal((1, 8, 16)) / 4)
    y = np.asarray(evaluate_batch(WRITER, x, w, kernel="reference"), dtype=np.uint16).reshape(1, 16)
    y8 = np.asarray(evaluate_batch(READER, y, w8, kernel="reference"), dtype=np.uint16).reshape(1, 8)
    circuit = IF.Circuit("wired-two", ((WRITER, 1), (READER, 1)),
                         wires=(IF.Wire(RID, "x", WID, "o0", at=0, radix=(1,), base=0, step=(1,)),))
    win = IF.Window(circuit, RUN, RND.parse_design(DESIGN), 4)
    win.record(WRITER, [x, w], [y])
    win.record(READER, [None, w8], [y8])
    win.write(win.close(salts_of=fixed(win)), d, {"label": "green-wired"}, registrar=R)
    bw = win.window
    print("PROGRAM-ROOT-NODES", [(type(n).__name__) for n in getattr(bw.program, "nodes", [])][:6], bw.partition.get("query"))
    print("MEMBERS", [(m.key, m.template[:16], m.base, m.n, [(v.port, v.writer[:16], v.out) for v in m.wired]) for m in bw.members])
    bind = {"partition": bw.partition_digest, "program": bw.partition["program"]}
    rec = json.loads((d / "registration.json").read_text())
    roots = {r["port"]: r for r in rec["roots"]}
    print("REGISTRATION-ROOTS", sorted(roots), "fields" in rec)
    want = CS.expected_ports(bw)
    reads = CS.own_reads({}, bw, roots, registered=REG)
    (tmp_path / "partition.json").write_text(json.dumps(bw.partition, sort_keys=True))
    (tmp_path / "program.json").write_bytes(canonical_json(encode_program(bw.program)))
    for k, (m, defn) in enumerate(zip(bw.members, (WRITER, READER))):
        sd = tmp_path / f"m{k}"
        sd.mkdir()
        st = member_stager(defn, m, bind)
        (sd / "circuit.txt").write_text(st.text)
        got = FI.flock_input_files(d, k, st.text, sd, write=("pub",))
        header = json.loads(Path(got["pub"]).read_bytes().split(b"\n", 1)[0])
        fv = header["frame_v3"]["roots"]
        own = REG.bind_ports(reads[m.key], fv, {p: {"root": roots[bw.tree(m, p)]["root"],
                                                    "domain": want[bw.tree(m, p)]["domain"]} for p in fv})
        (sd / "registered.json").write_text(json.dumps(own, sort_keys=True))
        print(f"M{k}-REGISTERED", json.dumps({p: {f: v for f, v in e.items() if f not in ("root", "domain")}
                                              for p, e in own["ports"].items()}, sort_keys=True))
        print(f"M{k}-META-LINKS", "links" in st.text.split("\n", 3)[1] if st.text.startswith("flock-circuit") else "?")
        p = subprocess.run([str(LEAN / "verity-verify-proof"), "statement", "--circuit", str(sd / "circuit.txt"),
                            "--public", str(got["pub"]), "--partition", str(tmp_path / "partition.json"),
                            "--program", str(tmp_path / "program.json"), "--registered", str(sd / "registered.json")],
                           capture_output=True, text=True)
        print(f"M{k}-LEAN rc={p.returncode}\n{p.stdout[-3000:]}\n{p.stderr[-3000:]}")


def _tail(p: Path, n: int = 4000) -> str:
    return p.read_text(errors="replace")[-n:]


def test_wired_reader_through_the_served_audit(tmp_path, binaries):  # noqa: F811
    print("VARIANT", os.environ.get("GREEN_VARIANT", "wired"))
    window = _window(tmp_path / "window")
    rc, row, out, said = _audit(tmp_path, window, binaries, more={"FLOCK_CHALLENGER": "0"}, design=RULE)
    print("RC", rc)
    print("SUMMARY", json.dumps({k: row.get(k) for k in ("ok", "refused", "population", "draw", "members", "audit")},
                                default=str, indent=1))
    print("SAID-TAIL", said)
    for f in sorted((out / "window").rglob("*")):
        if f.is_file() and (f.suffix in (".log", ".txt") and "lean" in str(f) or f.name in ("registered.json",)
                            or "registered" in f.name and f.suffix == ".json"):
            print(f"=== {f.relative_to(out)}\n{_tail(f)}")
    dest = os.environ.get("OUT")
    if dest:
        keep = Path(dest) / "audit"
        for f in out.rglob("*"):
            if f.is_file() and f.suffix in (".json", ".log", ".txt", ".jsonl", ".desc") and f.stat().st_size < 20 << 20:
                (keep / f.relative_to(out)).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, keep / f.relative_to(out))
        for f in window.iterdir():
            if f.suffix in (".json", ".desc"):
                (keep / "window-in").mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, keep / "window-in" / f.name)
