"""One-stage audit, run A5, on a wired stand-in, and v2's wired reads on its sessions. The window: a writer
``Bf16AddRow_v1{N=64}`` (its ``a`` external, its ``b`` free) and a reader ``ArgmaxChunkBest_v1{C=64}``, four units each,
reader unit i's ``x`` the writer unit i's ``o0`` (one template-level wire, `interface.Wire`): small circuits, so that the
case and circuits this run keeps are a small test fixture (`tests/test_a5_wired.py`). Committed by `interface.Window` under
``<ROOT>/rank0`` with its registered circuit record, and audited by `a5.audit` as rank 0 of a one-rank step on a local
challenger's coins under ``stratified:1``, one unit of each template: each template's circuit composed by Flock's
`UnitStager` on its rows (`interface.unit_rows`), M0's `--zk` session of each drawn member behind the ZK server, the
Lean verifier of record, the stream's outcome and the negatives. Then v2's per-member verdicts on the audit's case
(``rank0/case-v2.json``, its ``proofs.wires`` the window's wiring from its registered circuit record, `a5.held_wiring`,
and its challenger and coins the audit's draw's, `a5.drawn_by`) by ``sampled-proofs-one-stage-zk verdicts --check``
(``--v2``), five ways:

* ``wired``: the case as the audit made it: every session accepted, each verdict naming the wiring's SHA-512;
* ``unwired``: its parameters without ``wires``: every session accepted, no verdict naming a wiring;
* ``moved``: the wire reading the writer's units in reverse: the reader's session refused by its wired reads (its rows
  don't open the writer's root), since its rows are the writer's at the other positions;
* ``altered``: the reader's public file with each unit's ``x`` row other than its writer's (a byte of its ``b || c``
  flipped), its header's ``x`` root and the commitment's ``<reader>/x`` root the altered rows', the evidence naming that
  file: the reader's session refused by its wired reads;
* ``writer``: the commitment's ``<writer>/o0`` root another, the files as they were: the writer's session refused by
  the statement's root check (`HELD`, `Registered.statementHeld` in each per-member session's setup), the reader's by
  its wired reads.

The window as a whole stays refused by v2's check until a sample over several members' statements is in the lock
(`OneStageZk.MULTI_REFUSED`); these are the sessions' verdicts it reads. The run exits 0 only when the audit is accepted
with every negative refused and the five verdicts are as said. ``--out`` holds the window (``window/``, its private
files too), a5's records (``rank0/``, ``challenger/``), the verdicts' files (``v2/``) and ``summary.json``.

    PYTHONPATH=<tree>:<tree>/benchmarks/numerical/python:<tree>/verity/ml/flock/python \\
      python benchmarks/one_stage/a5_wired.py --m0-pythonpath PATH --bin flock-circuit --lean flock-verify \\
        --firewall flock-firewall --v2 sampled-proofs-one-stage-zk --main FILE --out DIR
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import a5  # noqa: E402
import a5_coins as AC  # noqa: E402
import verity_difftest as DT  # noqa: E402
from composite import canonical  # noqa: E402
from verity.core.protocols.sampled_proofs.one_stage import draw as D, registration as R  # noqa: E402
from verity.ml.circuits.codec import canonical_json, descriptor_id  # noqa: E402
from verity.ml.circuits.defs import bind  # noqa: E402
from verity.ml.circuits.evaluation import evaluate_batch  # noqa: E402
from verity.ml.circuits.ops import argmax as AX, rows as RW  # noqa: E402
from verity.ml.windows import interface as IF, rows as CS, wiring as WI  # noqa: E402

#: the stand-in's session root, as a TP Commit's run root is its ranks' session root
ROOT = hashlib.sha512(b"verity/a5/wired-stand-in/v1").hexdigest()
RULE = "stratified:1"
UNITS = 4
#: rows of 64 words: the composer needs a row of each unit past one SHA-512 block (`verity_flock.circuit.Port.comps`)
WIDTH = 64
WRITER, READER = bind(RW.Bf16AddRow, N=WIDTH), bind(AX.ArgmaxChunkBest, C=WIDTH)
WID, RID = descriptor_id(WRITER), descriptor_id(READER)
OF = {"stand-in": "benchmarks/one_stage/a5_wired.py", "writer": WID, "reader": RID, "units": UNITS}
#: a reader's refusal by its wired reads: its row doesn't open its writer's root (`RegWired.check`)
OPENS = "does not open"
#: a writer's refusal by the statement's root check, its public file's ``o0`` root not the commitment's
#: (`Registered.statementHeld`)
HELD = "the statement's port o0 has another frame-v3-sha512 root or domain in the public file than the registration's"


def circuit() -> IF.Circuit:
    """The writer's units, then the reader's, each reader unit's ``x`` its writer unit's ``o0``: writers before their
    readers in the partition, as v2's wired reads hold them (`WiredReads.ordered`)."""
    return IF.Circuit("a5-wired", ((WRITER, UNITS), (READER, UNITS)), external=(f"{WID}/a",),
                      wires=(IF.Wire(RID, "x", WID, "o0", at=0, radix=(UNITS,), base=0, step=(1,)),))


def values(seed: int = 0) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """The writer's inputs (a, b) and output, then the reader's, its ``x`` the writer's output: bf16 words."""
    rng = np.random.default_rng(seed)

    def bf16(a: np.ndarray) -> np.ndarray:
        return (np.asarray(a, np.float32).view(np.uint32) >> 16).astype(np.uint16)

    def out(fn: Any, *xs: np.ndarray) -> np.ndarray:
        return np.asarray(evaluate_batch(fn, *xs, kernel="reference"), dtype=np.uint16).reshape(UNITS, -1)

    a, b = bf16(rng.standard_normal((UNITS, WIDTH))), bf16(rng.standard_normal((UNITS, WIDTH)))
    y0 = out(WRITER, a, b)
    return [a, b], [y0], [y0], [out(READER, y0)]


def commit(out: Path) -> tuple[CS.Binding, str]:
    """The window under ``<ROOT>/rank0`` committed into ``out`` with fresh salts and registered; -> (the verifier's window,
    the SHA-512 of its registered circuit record, which the verifier holds)."""
    win = IF.Window(circuit(), f"{ROOT}/rank0", D.parse_law(RULE), UNITS)
    wi, wo, ri, ro = values()
    win.record(WRITER, wi, wo)
    win.record(READER, ri, ro)
    c = win.close()
    win.write(c, out, {"of": OF, "committer": {"module": "a5_wired.py"}}, registrar=R, kind="stand-in")
    return win.window, hashlib.sha512(IF.circuit_bytes(win.circuit_record())).hexdigest()


def composer(fn: Any) -> Any:
    """``fn``'s circuit as the verifier composes it from its Definition alone: Flock's `UnitStager` on its rows
    (`interface.unit_rows`), bound to the partition and program (`a5.audit`'s composer)."""
    def compose(path: Path, partition: str, program: str) -> dict:
        from verity_flock.class_statement import UnitStager

        st = UnitStager(fn, "unit/a5-wired", *IF.unit_rows(fn), circuit=path, bind={"partition": partition, "program": program})
        return {"subcircuit": st.meta.get("subcircuit"), "pin": st.pin, "class_sha512": st.class_sha512,
                "k_log": st.meta["k_log"], "meta": st.meta}
    return compose


def verdicts(exe: str, case: dict, d: Path) -> dict:
    """``verdicts --check`` on ``case``'s params, commitment, evidence and blobs, its files written to ``d``: its exit
    code and its line (``{"verdicts"}`` or ``{"refused"}``)."""
    d.mkdir(parents=True, exist_ok=True)
    files = {"params.json": canonical(case["params"]), "commitment.bin": bytes.fromhex(case["commitment"]),
             "evidence.json": bytes.fromhex(case["evidence"])}
    for f, b in files.items():
        (d / f).write_bytes(b)
    (d / "blobs.json").write_text(json.dumps(case["blobs"]))
    p = subprocess.run([exe, "verdicts", "--check", *(str(d / f) for f in files), *case["blobs"]], capture_output=True,
                       text=True, check=False)
    line = p.stdout.strip().splitlines()[-1] if p.stdout.strip() else json.dumps({"refused": f"no line: {p.stderr[-2000:]}"})
    (d / "answer.json").write_text(line + "\n")
    return {"rc": p.returncode, **json.loads(line)}


def evidence_of(case: dict) -> dict:
    return json.loads(bytes.fromhex(case["evidence"]))


def with_evidence(case: dict, e: dict) -> dict:
    return {**case, "evidence": DT.canonical(e).encode().hex()}


def moved(case: dict) -> dict:
    """``case`` with its wire reading the writer's units in reverse (reader unit i the writer's unit 3 - i), which
    `Wires.check` and `WiredReads.ordered` accept."""
    pr = case["params"]["proofs"]
    (w,) = pr["wires"]["wires"]
    wires = {**pr["wires"], "wires": [{**w, "base": UNITS - 1, "step": [-1]}]}
    return {**case, "params": {**case["params"], "proofs": {**pr, "wires": wires}}}


def unwired(case: dict) -> dict:
    pr = {k: v for k, v in case["params"]["proofs"].items() if k != "wires"}
    return {**case, "params": {**case["params"], "proofs": pr}}


def rooted(case: dict, port: str, root: str) -> dict:
    """``case`` with its commitment's root of ``port`` (``<member>/<port>``) ``root``, and the evidence's reported root of
    it when it reports one."""
    rec = json.loads(bytes.fromhex(case["commitment"]))
    (r,) = [r for r in rec["roots"] if r["port"] == port]
    r["root"] = root
    e = evidence_of(case)
    if port in (e.get("session_roots") or {}):
        e["session_roots"][port] = root
    return {**with_evidence(case, e), "commitment": canonical_json(rec).hex()}


def writer(case: dict, writer_key: str) -> dict:
    """``case`` with its commitment's root of the writer's ``o0`` another, its files as they were: the writer's statement
    isn't the root its reader opens."""
    return rooted(case, f"{writer_key}/o0", hashlib.sha512(b"a5_wired: another o0 root").hexdigest())


def altered(case: dict, reader: CS.Member, d: Path) -> dict:
    """``case`` with the reader's public file's every ``x`` row other than its writer's (the first byte of its ``b || c``
    flipped, `wiring.string_at`) and its header's ``x`` root the altered rows' (`rows.port_tree`), and the commitment's
    ``<reader>/x`` root that root: a prover that committed reader rows other than the writer's, which only the wire
    tells from the writer's. The file is written to ``d`` and named by the reader's statement and the blobs."""
    e = evidence_of(case)
    (st,) = [s for s in e["proofs"]["statements"] if s["member"] == reader.key]
    (src,) = [Path(b) for b in case["blobs"] if Path(b).is_file() and hashlib.sha512(Path(b).read_bytes()).hexdigest() == st["public"]]
    head, body = src.read_bytes().split(b"\n", 1)
    body = bytearray(body)
    fv = json.loads(head)["frame_v3"]
    at = [WI.string_at(reader, bytes(body), "x", u) for u in range(reader.n)]

    def root() -> str:
        return CS.port_tree(fv, "x", [bytes(body[i:i + CS.BC_BYTES]) for i in at]).commitment.root.hex()
    was = fv["roots"]["x"]
    if root() != was or head.count(was.encode()) != 1:
        raise ValueError("the reader's public file's x rows are not the tree its header states once")
    for i in at:
        body[i] ^= 1
    head = head.replace(was.encode(), root().encode())
    d.mkdir(parents=True, exist_ok=True)
    f = d / src.name
    f.write_bytes(head + b"\n" + bytes(body))
    st["public"] = hashlib.sha512(f.read_bytes()).hexdigest()
    return rooted({**with_evidence(case, e), "blobs": [*case["blobs"], str(f)]}, f"{reader.key}/x", root())


def cases(case: dict, writer_key: str, reader: CS.Member, d: Path) -> dict[str, tuple[dict, list[str | None]]]:
    """The five cases of ``case`` (module docstring), each with what `held` asks of its sessions' verdicts in the
    statements' order: None accepted, else the refusal's reason. The altered reader's file is written under ``d``."""
    members = [st["member"] for st in evidence_of(case)["proofs"]["statements"] for _ in st["sessions"]]
    reads = [OPENS if m == reader.key else None for m in members]
    return {"wired": (case, [None] * len(members)), "unwired": (unwired(case), [None] * len(members)),
            "moved": (moved(case), reads), "altered": (altered(case, reader, d), reads),
            "writer": (writer(case, writer_key), [HELD if m == writer_key else OPENS for m in members])}


def wiring_digest(case: dict) -> str | None:
    """The SHA-512 of the canonical JSON of ``case``'s ``proofs.wires``, as each verdict names it (`WiredReads.specOf`);
    None for a case without wires."""
    w = case["params"]["proofs"].get("wires")
    return None if w is None else hashlib.sha512(DT.canonical(w).encode()).hexdigest()


def held(got: dict, members: list[str], want: list[str | None], wires: str | None) -> bool:
    """Whether ``got`` lists one verdict per session, in the statements' order (``members``): accepted where ``want`` is
    None, naming the wiring ``wires`` (none when None), and elsewhere refused with a reason holding ``want``'s."""
    vs = got.get("verdicts")
    if got["rc"] != 0 or vs is None or len(vs) != len(members):
        return False
    return all((v["accepted"] is True and v.get("wires") == wires) if why is None else
               (v["accepted"] is False and why in (v.get("why") or "")) for v, why in zip(vs, want))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m0-pythonpath", required=True)
    ap.add_argument("--bin", required=True)
    ap.add_argument("--lean", required=True)
    ap.add_argument("--firewall", required=True)
    ap.add_argument("--v2", required=True, help="sampled-proofs-one-stage-zk, whose `verdicts --check` reads the case")
    ap.add_argument("--main", required=True, help="the verifier's own `git rev-list --first-parent origin/main`")
    ap.add_argument("--out", required=True)
    ap.add_argument("--delta", type=float, default=0.5)
    ap.add_argument("--port", type=int, default=7571)
    ap.add_argument("--session-timeout", type=float, default=3600)
    ap.add_argument("--compose-cache", help="a5.py's: this machine's composed circuits (`compose_circuit.py --cache`)")
    a = ap.parse_args()
    out = Path(a.out)
    mine, digest = commit(out / "window")

    def source(w: dict) -> dict:
        ok = {"kind": w.get("kind") == "stand-in", "run": w.get("run") == mine.run, "of": w.get("of") == OF,
              "circuit": w.get("circuit") == digest}
        return {"ok": all(ok.values()), "codes": [c for c, v in ok.items() if not v], "detail": {"kind": w.get("kind"), "run": w.get("run")}}

    a.rule, a.window_root = RULE, ROOT
    sets = {WID: (composer(WRITER), WRITER), RID: (composer(READER), READER)}
    manager, a.coins, finisher = AC.start("local", ROOT, [0], out / "challenger", a.main)
    try:
        s, ok = a5.audit(a, sets, out / "rank0", mine, out / "window", source, a.port,
                         {"rank": 0, "standin": OF, "circuit": digest}, rank=0)
    finally:
        a.coins.skip(0)
        finisher.join()
        published = a.coins.evidence()
        manager.shutdown()
    case_file = out / "rank0" / "case-v2.json"
    got: dict[str, Any] = {}
    if case_file.is_file():
        case = json.loads(case_file.read_text())
        wiring = case["params"]["proofs"].get("wires")
        e = evidence_of(case)
        members = [st["member"] for st in e["proofs"]["statements"] for _ in st["sessions"]]
        reader = next(m for m in mine.members if m.template == RID)
        writer_key = next(m.key for m in mine.members if m.template == WID)
        for name, (c, w) in cases(case, writer_key, reader, out / "v2" / "altered-public").items():
            r = verdicts(a.v2, c, out / "v2" / name)
            wd = wiring_digest(c)
            got[name] = {"held": held(r, members, w, wd), "rc": r["rc"], "refused": r.get("refused"), "wires": wd,
                         "verdicts": [{x: v.get(x) for x in ("accepted", "session", "why", "wires")} for v in r.get("verdicts") or []]}
        got["members"], got["reader"], got["wiring"] = members, reader.key, wiring_digest(case) if wiring is not None else None
    else:
        why = out / "rank0" / "case-v2-error.txt"
        got["error"] = why.read_text() if why.is_file() else "a5 wrote no v2 case"
    v2_ok = got.get("reader") in got.get("members", []) and got.get("wiring") is not None and all(
        got[n]["held"] for n in ("wired", "unwired", "moved", "altered", "writer"))
    summary = {"root": ROOT, "session": (published.get("plan") or {}).get("session"),
               "window": {"population": mine.population, "members": [[m.key, m.n] for m in mine.members],
                          "wired": {m.key: [[p, o, list(ws)] for p, o, ws in m.wired] for m in mine.members if m.wired},
                          "circuit": digest},
               "challenger": {x: published.get(x) for x in ("trusted", "error")},
               "rank0": {x: s.get(x) for x in ("checks_before_draw", "refused", "audit", "draw", "draw_counts", "negatives", "coins",
                                                "circuits", "wiring_reference", "link_points")},
               "v2": got, "audit_accepted": ok and not published.get("error"), "accepted": ok and not published.get("error") and v2_ok}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str))
    print("A5-WIRED " + json.dumps({"accepted": summary["accepted"], "audit_accepted": summary["audit_accepted"],
                                    "window": summary["window"], "draw": s.get("draw"),
                                    "v2": {n: (g["held"] if isinstance(g, dict) and "held" in g else g) for n, g in got.items()}},
                                   sort_keys=True, default=str))
    return 0 if summary["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
