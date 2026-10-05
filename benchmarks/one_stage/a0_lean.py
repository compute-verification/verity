"""One-stage audit, A0 with the auditor's Lean program driving: `flock-audit` (`FlockAudit`) registers, draws the unit draw
itself and verifies the session at it, and this file's workers answer its requests.

    PYTHONPATH=<as a0.py> python benchmarks/one_stage/a0_lean.py --set DIR --m0-pythonpath PATH --bin flock-circuit \\
      --audit flock-audit --out DIR [--k 16] [--delta 0.01] [--port 7521] [--statement verity/flock-circuit]

The launcher does what a0.py does before the draw (the partition, M0's staging, the registration and its checks), puts
the circuit and the registered public file into the auditor's content store (`OUT/archive/sha512/<key>`), and hands the
process to `flock-audit run`, which spawns `a0_lean.py worker` and prints one `VERDICT` line. The worker serves:

* `developer`: the registered public file's key (the registration's `leaf_layer`);
* `serving`: a live `flock-circuit serve --zk --draw-file` / `prove --zk` session at the draw the auditor sent, its record
  and proofs put into the content store by key;
* `audit`: the audit record (`one_stage.audit.audit_record`) with `flock-audit`'s verdict as the verifier of record,
  written to `OUT/audit.json` and `OUT/summary.json`; its answer is a report and never changes the verdict.

There is no draw file among the auditor's inputs: the draw is the program's own (`Draw.drawWith` on
`IO.getRandomBytes`), and the transcript (`OUT/transcript.bin`, `flock-audit replay`) holds its coins.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import a0  # noqa: E402
from verity.primitives.circuits.codec import canonical_json  # noqa: E402
from verity.protocols.verification.sampled_proofs.one_stage import audit, draw as D, partition as P, registration as R  # noqa: E402
from verity_numerical.bench.input_sets import InputSet  # noqa: E402

CHANNELS = ["developer", "serving", "audit"]


def put(archive: Path, b: bytes) -> str:
    key = hashlib.sha512(b).hexdigest()
    (archive / "sha512").mkdir(parents=True, exist_ok=True)
    (archive / "sha512" / key).write_bytes(b)
    return key


def launch(a: argparse.Namespace) -> int:
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    s = InputSet.open(a.set)
    n = s.n
    template = s.subcircuit.definition()
    law_text = f"subset:{a.k}"
    law = D.parse_law(law_text)
    program = P.population_program(template, n)
    query = P.template_instance_query(template)
    st = a0.stage(a.set, [n], out / "stage", a.m0_pythonpath)
    header = a0.public_header(st["dir"], n)
    fv = header["frame_v3"]
    want = a0.stand_in_ports(s.name, s.content_digest, st["meta"], n)
    leaf_layer = a0.sha512_file(st["dir"] / f"pub-{n}.bin")
    rec = R.record(program, query, law, header["scheme"], a0.registered_roots(header, want), a0.stand_in(a.set),
                   leaf_layer=leaf_layer)
    receipt = R.receipt(rec, "one-stage-e2e verifier (a0_lean.py)")
    checks = {"registration": R.check(rec, program, query, law, a0.SCHEME, want, leaf_layer=leaf_layer).to_json(),
              "public": R.matches_public(rec, header["scheme"], fv["roots"], fv["domain_ids"], header["instances"]).to_json()}
    (out / "registration.json").write_text(json.dumps(rec, indent=1, sort_keys=True))
    (out / "receipt.json").write_text(json.dumps(receipt, indent=1, sort_keys=True))
    if not all(c["ok"] for c in checks.values()) or not a0.template_program_ok(st["meta"], template):
        print("A0-LEAN " + json.dumps({"refused_before_draw": checks}))
        return 1
    archive = out / "archive"
    assert put(archive, (st["dir"] / "circuit.txt").read_bytes()) == st["pin"]
    assert put(archive, (st["dir"] / f"pub-{n}.bin").read_bytes()) == leaf_layer
    state = {"out": str(out), "stage": str(st["dir"]), "pin": st["pin"], "n": n, "law": law_text, "bin": a.bin, "port": a.port,
             "delta": a.delta, "checks_before_draw": {k: v["ok"] for k, v in checks.items()}}
    (out / "state.json").write_text(json.dumps(state, indent=1, sort_keys=True))
    (out / "workers.json").write_text(json.dumps({"workers": [{"name": "a0-lean", "channels": CHANNELS, "argv": [
        sys.executable, str(Path(__file__).resolve()), "worker", "--state", str(out / "state.json")]}]}))
    argv = [a.audit, "run", "--zk", "--statement", a.statement, "--circuit", st["pin"], "--k", str(a.k), "--archive", str(archive),
            "--workers", str(out / "workers.json"), "--transcript", str(out / "transcript.bin")]
    (out / "flock-audit.argv.json").write_text(json.dumps(argv))
    sys.stdout.flush()
    os.execv(a.audit, argv)


class Worker:
    """The developer, the serving side and the audit, answering `flock-audit`'s frames."""

    def __init__(self, state: dict):
        self.state = state
        self.out = Path(state["out"])
        self.archive = self.out / "archive"
        self.st = {"dir": Path(state["stage"]), "pin": state["pin"]}
        self.rec = json.loads((self.out / "registration.json").read_text())
        self.receipt = json.loads((self.out / "receipt.json").read_text())
        self.header = a0.public_header(self.st["dir"], state["n"])
        self.record: dict | None = None
        self.live: dict | None = None

    def register(self, ask: dict) -> dict:
        return {"public": self.rec["leaf_layer"]}

    def session(self, ask: dict) -> dict:
        s = self.out / "session"
        s.mkdir(parents=True, exist_ok=True)
        (s / "draw.json").write_text(json.dumps(ask["draw"]))
        self.record, self.live = a0.session(self.state["bin"], self.st, self.state["n"], self.state["law"], self.state["port"], s,
                                            draw_file=s / "draw.json", zk=True)
        sessions = sorted((s / "sessions").glob("*/session.json"))
        if not sessions:
            return {"error": f"no session recorded: {(self.live or {}).get('why')}"}
        files = {p.name: put(self.archive, p.read_bytes()) for p in sorted(sessions[-1].parent.glob("*.bin"))}
        return {"files": files, "record": put(self.archive, sessions[-1].read_bytes())}

    def audit(self, ask: dict) -> dict:
        v = ask["verdict"]
        line = canonical_json(v)
        verdicts = {"flock-audit": {"accepted": v["accepted"] is True, "why": v["why"], "of_record": True,
                                    "digest": hashlib.sha512(line).hexdigest(), "tool": "flock-audit (FlockAudit, a0_lean.py)"}}
        if self.record is not None:
            m0 = self.record.get("verdict") or {}
            verdicts["m0-verifier"] = {"accepted": m0.get("accepted") is True, "why": m0.get("why"), "tool": "flock-circuit serve --zk"}
        live = self.live or {}
        draw = v["draw"] or {"law": "subset", "population": self.state["n"], "k": 0, "units": []}
        sessions = [a0.sha512_file(f) for f in sorted((self.out / "session" / "sessions").glob("*/session.json"))]
        statement = {"circuit_sha512": live.get("circuit_sha512"), "statement_digest": live.get("statement_digest"),
                     "public_sha512": live.get("public_sha512"), "roots": live.get("roots")}
        result = audit.audit_record(self.rec, self.receipt, draw, D.parse_law(self.state["law"]), a0.proved_units(self.header, live),
                                    live.get("instances", -1), statement, verdicts, self.state["delta"], sessions=sessions,
                                    served=((self.record or {}).get("unit_draw") or {}).get("units", []),
                                    session_roots=live.get("roots") or {})
        text = json.dumps(result, indent=1, sort_keys=True)
        (self.out / "audit.json").write_text(text)
        timing = {k: live.get(k) for k in ("e2e_s", "witness_s", "prove_total_s", "proof_bytes", "k_log", "units")}
        summary = {"verdict": v, "checks_before_draw": self.state["checks_before_draw"],
                   "audit": {k: result.get(k) for k in ("accepted", "complete", "proved_units", "checks", "integrity_profile")},
                   "verdicts": {k: x.get("accepted") for k, x in verdicts.items()}, "timing": timing}
        (self.out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True))
        return {"accepted": result["accepted"], "audit_sha512": hashlib.sha512(text.encode()).hexdigest(),
                "checks": {k: c["ok"] for k, c in result["checks"].items()}}


def serve(state_file: Path) -> int:
    frames = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)
    w = Worker(json.loads(state_file.read_text()))
    inp = sys.stdin.buffer
    while len(head := inp.read(8)) == 8:
        ask = json.loads(inp.read(int.from_bytes(head, "big")))
        answer = {"register": w.register, "session": w.session, "audit": w.audit}.get(ask.get("ask"))
        b = canonical_json(answer(ask) if answer else {"error": f"unknown ask {ask.get('ask')!r}"})
        frames.write(len(b).to_bytes(8, "big") + b)
        frames.flush()
    return 0


def main() -> int:
    if sys.argv[1:2] == ["worker"]:
        ap = argparse.ArgumentParser()
        ap.add_argument("--state", type=Path, required=True)
        return serve(ap.parse_args(sys.argv[2:]).state)
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--m0-pythonpath", required=True)
    ap.add_argument("--bin", required=True, help="flock-circuit (the live build, with --zk)")
    ap.add_argument("--audit", required=True, help="flock-audit")
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--delta", type=float, default=0.01)
    ap.add_argument("--port", type=int, default=7521)
    ap.add_argument("--statement", default=a0.STATEMENT_ID)
    return launch(ap.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
