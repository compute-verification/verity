"""One-stage audit, run A0: a template's captured served values through registration, the verifier's sample, M0's prover and
verifier, the Lean verifier's sample checks, and the audit record with its integrity profile; then the negatives.

    PYTHONPATH=<tree>:<tree>/benchmarks/numerical/python:<tree>/integrations/vllm:<tree>/catalog:<tree>/kernels:<tree>/experimental \\
      python benchmarks/one_stage/a0.py --set DIR --m0-pythonpath PATH --bin flock-circuit --lean flock-verify --out DIR \\
      [--k 16] [--delta 0.01] [--port 7511]

* **Stand-in commitment.** M0 commits the captured values afresh (`frame-v3-sha512`, `hm96-sha512/row/v1` leaves, fresh
  OS salts). The values were read under the served run's `vllm-v1` root and replayed bit-exact there by `vu_export`; that
  link is provenance, not proof, and the registration says so.
* **Partition.** `Q_template_instance` v0 over the population program: one unit per instance, no width rule.
* **Verifier of record.** Lean's `flock-verify verify` replays the session offline with the circuit and the public
  file held by content; Lean also checks U1-U3 on the sample, and M0's own verifier (`flock-circuit serve`) checks it live.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from verity.ml.commitments.identity import identity_digest, identity_digest_sha512
from verity.ml.commitments.indexed import RangeIndexedDomain
from verity.ml.commitments.merkle import CommitmentDomain
from verity.ml.circuits.codec import canonical_json, encode_program, program_digest
from verity.ml.circuits.program import Program
from verity_numerical.bench.input_sets import InputSet
from verity.core.protocols.sampled_proofs.one_stage import audit, draw as D, partition as P, registration as R, served as SV, verifier as V

import r2_population as R2

HERE = Path(__file__).resolve().parent
SCHEME = "frame-v3-sha512/hm96-sha512"
#: M0's set-bound frame domains (verity_flock.circuit.BINDING_TAG), each tag with its identity digest; v1 is M0's before
#: 64-byte bindings, for a pinned tree that old (--binding)
STAND_IN_BINDINGS = {"verity/flock-circuit/binding/v2": identity_digest_sha512,
                     "verity/flock-ir-frame/binding/v1": lambda tag, doc: bytes.fromhex(identity_digest(tag, doc))}
STAND_IN_BINDING = "verity/flock-circuit/binding/v2"
STATEMENT_ID = "verity/flock-circuit"


def sha512_file(path: Path) -> str:
    return hashlib.sha512(path.read_bytes()).hexdigest()


def run(cmd: list[str], log: Path, env: dict | None = None, timeout: float = 3600) -> subprocess.CompletedProcess:
    t = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, **(env or {})}, timeout=timeout)
    log.write_text(f"$ {' '.join(cmd)}\n# rc {p.returncode} in {time.time() - t:.1f} s\n--- stdout\n{p.stdout}\n--- stderr\n{p.stderr}")
    return p


def stage(set_dir: str, ns: list[int], out: Path, m0_path: str, tampered: bool = False, bind: dict | None = None) -> dict:
    """M0's staging; `bind` (the verifier's partition digest, program SHA-512 and unit indices) goes into the statement."""
    out.mkdir(parents=True, exist_ok=True)
    env = {"PYTHONPATH": m0_path}
    extra: list[str] = []
    if bind:
        (out / "unit-indices.json").write_text(json.dumps(bind["indices"]))
        extra = ["--partition", bind["partition"], "--program", bind["program"], "--unit-indices", str(out / "unit-indices.json")]
    if tampered:
        cmd = [sys.executable, str(HERE / "stage_tampered.py"), "--input-set", set_dir, "--n", str(ns[0]), "--out", str(out), *extra]
    else:
        cmd = [sys.executable, "-m", "verity_flock.circuit", "--input-set", set_dir, "--n", *map(str, ns), "--out", str(out), *extra]
    p = run(cmd, out / "stage.log", env)
    if p.returncode != 0:
        raise RuntimeError(f"staging failed: {p.stderr[-2000:]}")
    text = (out / "circuit.txt").read_text()
    meta = json.loads(text.split("\n", 2)[1][5:])
    return {"dir": out, "meta": meta, "pin": hashlib.sha512(text.encode()).hexdigest()}


def hold_bound(st: dict, program, query) -> None:
    """The verifier's own partition object and program beside a bound stage, which the Lean verifier then reads by content."""
    (st["dir"] / "partition.json").write_text(json.dumps(P.partition(program, query), sort_keys=True))
    (st["dir"] / "program.json").write_bytes(canonical_json(encode_program(program)))
    st["partition_file"], st["program_file"] = st["dir"] / "partition.json", st["dir"] / "program.json"


def hold_registered(st: dict, program, query, served, log=(), ports=None) -> SV.ServedRegistration:
    """The verifier's own `--registered` file beside a bound stage (`served.register_served`: each served tree a port, the
    form every served path registers in), which `lean_verify` passes to `verify --zk`. The returned registration's
    `registered` entries go into the record (`R.record(..., registered=)`), and its `values` and `outputs` into R7."""
    sr = SV.register_served(program, query, served, log=log, ports=ports)
    st["registered_file"] = sr.write(st["dir"] / "registered.json")
    return sr


def unit_class(set_dir: str, m0_path: str, out: Path) -> str:
    """The class of every instance, from the verifier's own composition of the circuit (M0's `--class-only`)."""
    p = run([sys.executable, "-m", "verity_flock.circuit", "--input-set", set_dir, "--class-only"], out / "class.log", {"PYTHONPATH": m0_path})
    return json.loads(p.stdout.strip().splitlines()[-1])["class_sha512"]


def template_program_ok(meta: dict, defn) -> bool:
    """META's one `program_digests` entry is this template's: keyed by its descriptor id (or its display id, as M0 keys it
    through 68ae79f2), valued core's `program_digest` of its Program."""
    pd = meta.get("program_digests", {})
    return len(pd) == 1 and next(iter(pd)) in (P.template_id(defn), defn.id) and next(iter(pd.values())) == program_digest(Program(defn))


def bindings(meta: dict, header: dict, rec: dict, cls: str) -> dict:
    """What the staged statement binds, against the registration: the partition, the program, the global unit indices of
    `Q_template_instance` (instance i is unit i), and one class per instance."""
    n = rec["population"]
    got = {"partition": meta.get("partition"), "program": meta.get("program_sha512"),
           "indices": header["units"]["indices"] == list(range(n)), "classes": header["units"].get("classes") == [cls] * n}
    ok = {"partition": got["partition"] == {"rule": P.FORMAT, "digest": rec["partition"]},
          "program": got["program"] == rec["program"], "indices": got["indices"], "classes": got["classes"]}
    return {"ok": all(ok.values()), "each": ok, "class_sha512": cls}


def public_header(stage_dir: Path, n: int) -> dict:
    with open(stage_dir / f"pub-{n}.bin", "rb") as f:
        return json.loads(f.readline())


def hidden_outputs(meta: dict) -> bool:
    """M0 from hidden outputs (META `input_ports`): every output port is a row after the input rows, and no domain binds the
    set's unsalted content digest."""
    return "input_ports" in meta


def ports(meta: dict, n: int) -> dict[str, tuple[str, int]]:
    """Each port the layout derives: its schema and leaf count (before hidden outputs, each output port a tree of u16 words)."""
    out = {name: ("hm96-sha512/row/v1", n) for name, *_ in meta["ports"]}
    if not hidden_outputs(meta):
        out.update({name: ("u16", n * words) for name, words in meta["out_ports"]})
    return out


def stand_in_ports(name: str, content_digest: str, meta: dict, n: int) -> dict[str, dict]:
    """The verifier's own derivation of the stand-in's domains: M0's frame domains bound to the set (before hidden outputs,
    its content digest too), the range, the port and its schema."""
    out = {}
    content = {} if hidden_outputs(meta) else {"content_digest": content_digest}
    for p, (schema, count) in ports(meta, n).items():
        b = STAND_IN_BINDINGS[STAND_IN_BINDING](STAND_IN_BINDING, {"set": name, **content, "lo": 0, "hi": n, "port": p, "schema": schema})
        out[p] = {"leaves": count, "domain": CommitmentDomain(b, -1, RangeIndexedDomain(0, count), hash="sha512").domain_id.hex()}
    return out


def registered_roots(header: dict, want: dict[str, dict]) -> list[dict]:
    fv = header["frame_v3"]
    return [{"port": p, "root": fv["roots"][p], "leaves": want[p]["leaves"], "domain": fv["domain_ids"][p]} for p in want]


def stand_in(set_dir: str) -> dict:
    prov = json.loads((Path(set_dir) / "manifest.json").read_text()).get("provenance", {})
    return {"kind": "stand-in", "of": "vllm-v1 commitment of a served run", "input_set": Path(set_dir).name, "row": prov.get("row"), "run": prov.get("research_run") or prov.get("run"),
            "run_root": prov.get("run_root"), "model": prov.get("model"), "program_digests": prov.get("program_digests"),
            "link": "provenance, not proof: vu_export read these values under run_root and replayed each bit-exact"}


def session(bin_: str, st: dict, n: int, law: str, port: int, out: Path, draw_file: Path | None = None,
            zk: bool = False, env: dict | None = None, prove_timeout: float = 3600) -> tuple[dict | None, dict | None]:
    """One live session: `serve --draw` (the verifier's side: registration, the sample, coins, the verdict) and `prove`.
    `draw_file`: the server serves the verifier's own draw from that file (`--draw-file`) instead of sampling one. `zk`: both
    sides run the masked protocol (`--zk`); without it the Lean verifier accepts a subset sample at hidden outputs and one table
    (verifier PROTOCOL §16.15). `env`: added to
    both sides' environment (a firewalled prover's, `firewall_env`). `prove_timeout`: the prover's limit in seconds."""
    out.mkdir(parents=True, exist_ok=True)
    d = st["dir"]
    tables = d / "tables"
    zf = ["--zk"] if zk else []
    serve_log = open(out / "serve.log", "w")
    serve = subprocess.Popen([bin_, "serve", *zf, "--listen", f"127.0.0.1:{port}", "--out", str(out / "sessions"),
                              "--instances", str(d / f"pub-{n}.bin"), "--circuit", str(d / "circuit.txt"), "--tables", str(tables),
                              "--pin", st["pin"], "--operator", "one-stage-a0",
                              *(["--draw-file", str(draw_file)] if draw_file else ["--draw", law]), "--sessions", "1"],
                             stdout=serve_log, stderr=subprocess.STDOUT, text=True, env={**os.environ, **(env or {})})
    try:
        for _ in range(1800):
            if "SERVING" in (out / "serve.log").read_text() or serve.poll() is not None:
                break
            time.sleep(0.5)
        p = run([bin_, "prove", *zf, "--verifier", f"127.0.0.1:{port}", "--instances", str(d / f"inst-{n}.bin"),
                 "--circuit", str(d / "circuit.txt"), "--tables", str(tables), "--runs", "1", "--warm", "0"], out / "prove.log", env=env,
                timeout=prove_timeout)
        live = next((json.loads(line[5:]) for line in p.stdout.splitlines() if line.startswith("LIVE\t")), None)
        if live is None:
            refused = next((line for line in p.stdout.splitlines() if line.startswith("REFUSED\t")), "")
            live = {"accepted": False, "why": f"prover exited {p.returncode}: {refused[:600]} {p.stderr[-600:]}".strip()}
        serve.wait(timeout=600)
    finally:
        if serve.poll() is None:
            serve.kill()
        serve_log.close()
    recs = sorted((out / "sessions").glob("*/session.json"))
    return (json.loads(recs[-1].read_text()) if recs else None), live


def lean_draw_test(lean: str, cases: list[dict], out: Path, name: str) -> list[dict]:
    f = out / f"{name}.cases.json"
    f.write_text(json.dumps(cases))
    p = run([lean, "draw-test", str(f)], out / f"{name}.lean.log")
    return [json.loads(line.split(" ", 1)[1]) for line in p.stdout.splitlines() if line.startswith("DRAWCHECK")]


def firewall_env(st: dict) -> dict:
    """What a firewalled prover (`FC_FIREWALL=1`) sets its Lean `Hello` up on (live PROTOCOL.md §10.4): the statement
    `lean_verify` verifies under, and the statement's partition and program."""
    args = [x for k in ("partition", "program") if st.get(f"{k}_file") for x in (f"--{k}", str(st[f"{k}_file"]))]
    return {"FC_FIREWALL_STATEMENT": STATEMENT_ID, "FC_FIREWALL_HELLO_ARGS": " ".join(args)}


def lean_verify(lean: str, st: dict, n: int, out: Path, zk: bool = False, log: str = "lean-verify",
                flags: Sequence[str] = ()) -> dict:
    """The verifier of record: Lean `flock-verify verify` on the session's record, with the circuit and the public file
    held by content (`--archive`, SHA-512 keys), and the verifier's own registered reads (`st["registered_file"]`,
    `hold_registered`) when the statement reads registered values; `zk`, a `--zk` session's. `log` names its logs under
    `out` (a negative's own, beside the honest verdict's). `flags`: more of `verify`'s (a plain session's ``--coins os
    --stratified K``). ``st["archived"]``, a dict a run's members share: a circuit file archived once, by its file's
    identity (members' ``circuit.txt`` are links of one file), is linked into every later archive rather than put again;
    Lean holds the object to its key when it reads it (`archiveGet`)."""
    sessions = sorted((out / "sessions").glob("*/session.json"))
    if not sessions:
        return {"accepted": False, "of_record": True, "why": "no session recorded"}
    archive = out / "archive"
    tables = sorted(str(f) for f in (st["dir"] / "tables").glob("*") if f.is_file()) if (st["dir"] / "tables").is_dir() else []
    part = [str(st["partition_file"])] if st.get("partition_file") else []
    prog = [str(st["program_file"])] if st.get("program_file") else []
    reads = [str(st["registered_file"])] if st.get("registered_file") else []
    circuit, held = st["dir"] / "circuit.txt", st.get("archived")
    s = circuit.stat()
    ident = (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)
    pre = held.get(ident) if held is not None else None
    if pre is not None:
        (archive / "sha512").mkdir(parents=True, exist_ok=True)
        if not (archive / "sha512" / pre[0]).exists():
            os.link(pre[1], archive / "sha512" / pre[0])
    p = run([lean, "archive-put", str(archive), *([] if pre else [str(circuit)]), str(st["dir"] / f"pub-{n}.bin"), *tables, *part,
             *prog, *reads], out / ("archive-put.log" if log == "lean-verify" else f"{log}.archive-put.log"))
    keys = {Path(line.split()[1]).name: line.split()[0] for line in p.stdout.splitlines() if len(line.split()) == 2}
    if pre is not None:
        keys["circuit.txt"] = pre[0]
    elif held is not None and "circuit.txt" in keys:
        held[ident] = (keys["circuit.txt"], archive / "sha512" / keys["circuit.txt"])
    bound = ["--partition", keys.get(Path(part[0]).name, "")] if part else []
    bound += ["--program", keys.get(Path(prog[0]).name, "")] if prog else []
    bound += ["--registered", keys.get(Path(reads[0]).name, "")] if reads else []
    t = time.time()
    p = run([lean, "verify", "--statement", STATEMENT_ID, "--archive", str(archive), "--circuit", keys.get("circuit.txt", ""),
             "--public", keys.get(f"pub-{n}.bin", ""), *bound, *(["--zk"] if zk else []), *flags, "--session", str(sessions[-1].parent)],
            out / f"{log}.log")
    line = next((x for x in p.stdout.splitlines() if x.startswith("VERDICT ")), None)
    v = json.loads(line.split(" ", 1)[1]) if line else {"accepted": False, "why": f"no verdict (rc {p.returncode}): {p.stderr[-400:]}"}
    return {"accepted": v.get("accepted") is True, "why": v.get("why"), "of_record": True, "verify_s": round(time.time() - t, 1),
            "partition": v.get("partition"), **({"registered": v["registered"]} if "registered" in v else {}),
            "digest": hashlib.sha512(line.encode()).hexdigest() if line else None}


def lean_refused(v: dict, names: str | None = None) -> bool:
    """Lean's own verdict (a VERDICT line, not a0's stand-in for none) refused the proof, and not at setup, where a missing
    or unreadable input is refused, unless the refusal names `names`, a check Lean makes there."""
    why = v.get("why") or ""
    return v.get("accepted") is False and v.get("digest") is not None and bool(why) and (
        names in why if names else not why.startswith("setup:"))


def drawcheck_refused(cases: list[dict], i: int, code: str) -> bool:
    """Lean's sample check refused case `i` by `code` (U1, U3), not for a sample it could not read."""
    return len(cases) > i and not cases[i]["ok"] and cases[i]["why"].startswith(f"{code}:")


def proved_units(header: dict, live: dict) -> list[int]:
    """The drawn statement's units as global unit indices: the population header's index of each drawn position."""
    pop = header["units"]["indices"]
    return [pop[u] for u in (live.get("unit_draw") or {}).get("units", [])]


def audit_session(rec: dict, receipt: dict, st: dict, header: dict, law: dict, record, live, lean: str, lean118: str, out: Path,
                  delta: float) -> dict:
    """Every check on one session, then the audit record."""
    draw = (record or {}).get("unit_draw")
    verifier = (record or {}).get("verdict") or {}
    lean_u = lean_draw_test(lean, [{"record": draw, "statement": (live or {}).get("unit_draw"),
                                    "instances": (live or {}).get("instances", 0)}], out, "u1-u3") if draw else []
    statement = {"circuit_sha512": (live or {}).get("circuit_sha512"), "statement_digest": (live or {}).get("statement_digest"),
                 "public_sha512": (live or {}).get("public_sha512"), "sigma": (live or {}).get("sigma"),
                 "roots": (live or {}).get("roots"), "coin_commitment": ((record or {}).get("coin_seed") or {}).get("commitment"),
                 "roots_are_registered": (live or {}).get("roots") == {r["port"]: r["root"] for r in rec["roots"]}}
    verdicts = {
        "m0-verifier": {"accepted": verifier.get("accepted") is True, "why": verifier.get("why"),
                        "tool": "flock-circuit serve (M0, PR #83)"},
        "lean-draw-u1-u3": {"accepted": bool(lean_u) and all(c["ok"] for c in lean_u), "detail": lean_u,
                            "tool": "flock-verify draw-test (PR #113 9d0d53df)"},
        "lean-verify": {**lean_verify(lean118, st, header["instances"], out), "tool": "flock-verify verify --archive (PR #118)"},
    }
    if draw is None:
        return {"accepted": False, "why": "no draw recorded", "verdicts": verdicts, "statement": statement}
    sessions = [sha512_file(f) for f in sorted((out / "sessions").glob("*/session.json"))]
    return V.audit_record(rec, receipt, draw, law, proved_units(header, live or {}), (live or {}).get("instances", -1),
                          statement, verdicts, delta, sessions=sessions, session_roots=(live or {}).get("roots") or {})


def audit_negatives(result: dict, law: dict, delta: float, session_roots: dict, served: list | None = None) -> dict[str, dict]:
    """The audit record's own refusals, each re-deciding the real record with one input altered: the verifier of record
    not run or absent, a session root that isn't the registered one, a verifier of record reporting a registered value's
    reads opened against another root than its registered one, and, when the sessions were handed the verifier's sample
    (`served`), a served share that isn't it. Each is decided by the verifier of record (`verifier.audit_record`); `agree`
    is whether the Lean and the Python decided it alike."""
    base = dict(registration=result["registration"], receipt=result["receipt"], draw=result["draw"], law=law,
                proved_units=result["proved_units"], instances=len(result["proved_units"]), statement=result["statement"],
                verdicts=result["verdicts"], delta=delta, served=served, session_roots=session_roots)

    def decide(**change) -> dict:
        r = V.audit_record(**{**base, **change})
        return {"refused": not r["accepted"] and r["profile"] is None, "failed": [k for k, c in r["checks"].items() if not c["ok"]],
                "agree": r["verifiers"]["agree"]}

    vs = base["verdicts"]
    port = next(iter(session_roots), None)
    neg = {"verifier-of-record-did-not-run (C2)": decide(verdicts={k: {**v, "accepted": None} if v.get("of_record") else v
                                                                   for k, v in vs.items()}),
           "no-verifier-of-record (C2)": decide(verdicts={k: v for k, v in vs.items() if not v.get("of_record")}),
           "session-root-not-registered": decide(session_roots={**session_roots, port: "00" * 64})}
    opened = audit.session_registered(vs, base["registration"])
    if opened:
        value = next(iter(opened))
        neg["registered-read-opened-against-another-root"] = decide(verdicts={
            k: {**v, "registered": {p: {**e, "root": "00" * 64} if e["value"] == value else e for p, e in v["registered"].items()}}
            if v.get("of_record") and isinstance(v.get("registered"), dict) else v for k, v in vs.items()})
    if served is not None:
        drawn = set(served)
        other = [u for u in range(result["registration"]["population"]) if u not in drawn][:1]
        neg["served-share-not-the-verifier's-draw (C1)"] = decide(served=sorted(served)[1:] + other)
    return neg


def altered_public(lean: str, st: dict, n: int, session_dir: Path, out: Path, rec: dict, check, zk: bool = False) -> dict:
    """A public file whose body is altered (its last output word's high byte flipped) and whose leaf-layer digest is updated
    to match, so the registration passes R1-R6 and only the verifier of record, recomputing the roots from the file, can
    refuse it. `check(record, leaf_layer)` is the verifier's registration check."""
    stage = out / "stage"
    shutil.rmtree(out, ignore_errors=True)
    (stage / "tables").mkdir(parents=True)
    os.link(st["dir"] / "circuit.txt", stage / "circuit.txt")
    for f in (st["dir"] / "tables").glob("*"):
        if f.is_file():
            os.link(f, stage / "tables" / f.name)
    shutil.copytree(session_dir / "sessions", out / "sessions", copy_function=os.link)
    body = bytearray((st["dir"] / f"pub-{n}.bin").read_bytes())
    body[-1] ^= 1
    (stage / f"pub-{n}.bin").write_bytes(bytes(body))
    digest = hashlib.sha512(bytes(body)).hexdigest()
    registration = check(dict(rec, leaf_layer=digest), digest)
    lean_v = lean_verify(lean, dict(st, dir=stage), n, out, zk=zk)
    return {"refused": registration.ok and lean_refused(lean_v, names="do not hash to the header's"),
            "registration_passes": registration.ok,
            "lean": lean_v.get("why")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--m0-pythonpath", required=True)
    ap.add_argument("--bin", required=True)
    ap.add_argument("--lean", required=True, help="flock-verify for the sample checks (PR #113)")
    ap.add_argument("--lean-verify", default=None, help="flock-verify for the full verify (PR #118); default --lean")
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--delta", type=float, default=0.01)
    ap.add_argument("--port", type=int, default=7511)
    ap.add_argument("--bind", action="store_true", help="run A1's M0 half: the statement binds the partition, program and unit indices")
    ap.add_argument("--statement", default=STATEMENT_ID,
                    help="the Lean verifier's statement name (verity/flock-circuit@967b8d06 for files staged at M0 68ae79f2 or later)")
    ap.add_argument("--binding", default=STAND_IN_BINDING, choices=sorted(STAND_IN_BINDINGS),
                    help="M0's frame-domain binding tag (verity/flock-ir-frame/binding/v1 for an M0 tree before 64-byte bindings)")
    a = ap.parse_args()
    a.lean_verify = a.lean_verify or a.lean
    globals()["STATEMENT_ID"] = a.statement
    globals()["STAND_IN_BINDING"] = a.binding
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    s = InputSet.open(a.set)
    n = s.n
    template = s.subcircuit.definition()
    law_text = f"subset:{a.k}"
    law = D.parse_law(law_text)
    summary: dict = {"set": a.set, "subcircuit": s.subcircuit.id, "n": n, "law": law, "delta": a.delta, "bind": a.bind}

    # the partition: Q_template_instance v0 over the verifier's own population program (unit i is instance i)
    program = P.population_program(template, n)
    query = P.template_instance_query(template)
    bind = None
    if a.bind:
        part = P.partition(program, query)
        bind = {"partition": P.digest(part), "program": part["program"], "indices": list(range(P.units(program, query)))}

    # serving commits (the stand-in): M0 stages the captured values under frame-v3-sha512 / hm96-sha512
    t = time.time()
    st = stage(a.set, [4, n], out / "stage", a.m0_pythonpath, bind=bind)
    if a.bind:
        hold_bound(st, program, query)
    header = public_header(st["dir"], n)
    summary["stage_s"] = round(time.time() - t, 1)
    summary["template_program"] = {"statement": st["meta"]["program_digests"],
                                   "verifier": {P.template_id(template): program_digest(Program(template))},
                                   "ok": template_program_ok(st["meta"], template)}

    # 3. registration, before any draw; the verifier holds its own public file to it
    fv = header["frame_v3"]
    want = stand_in_ports(s.name, s.content_digest, st["meta"], n)
    leaf_layer = sha512_file(st["dir"] / f"pub-{n}.bin")
    rec = R.record(program, query, law, header["scheme"], registered_roots(header, want), stand_in(a.set), leaf_layer=leaf_layer)
    (out / "registration.json").write_text(json.dumps(rec, indent=1, sort_keys=True))
    checks = {"registration": V.check(rec, program, query, law, SCHEME, want, leaf_layer=leaf_layer).to_json(),
              "public": V.matches_public(rec, header["scheme"], fv["roots"], fv["domain_ids"], header["instances"]).to_json()}
    receipt = R.receipt(rec, "one-stage-e2e verifier (a0.py)")
    (out / "receipt.json").write_text(json.dumps(receipt, indent=1, sort_keys=True))
    if a.bind:
        checks["bindings"] = bindings(st["meta"], header, rec, unit_class(a.set, a.m0_pythonpath, out))
    summary["registration_digest"] = R.digest(rec)
    summary["partition_digest"] = rec["partition"]
    summary["checks_before_draw"] = {k: v["ok"] for k, v in checks.items()}
    if not all(c["ok"] for c in checks.values()) or not summary["template_program"]["ok"]:
        summary["checks"] = checks
        (out / "summary.json").write_text(json.dumps(summary, indent=1))
        print("A0 " + json.dumps({"refused_before_draw": checks}))
        return 1

    # 4-6. the verifier samples after Register, M0 proves the sampled proof blocks, M0's verifier and Lean's sample checks
    t = time.time()
    record, live = session(a.bin, st, n, law_text, a.port, out / "session")
    summary["session_s"] = round(time.time() - t, 1)
    result = audit_session(rec, receipt, st, header, law, record, live, a.lean, a.lean_verify, out / "session", a.delta)
    result["checks_before_draw"] = checks
    result["timing"] = {k: (live or {}).get(k) for k in ("e2e_s", "witness_s", "prove_total_s", "tp_s", "proof_bytes", "k_log", "units")}
    result["timing"]["verify_s"] = ((record or {}).get("verdict") or {}).get("verify_s")
    (out / "audit.json").write_text(json.dumps(result, indent=1, sort_keys=True))
    summary["audit"] = {k: result.get(k) for k in ("accepted", "complete", "proved_units", "wrong_units", "checks", "integrity_profile")}
    summary["audit"]["verdicts"] = {k: v.get("accepted") for k, v in result.get("verdicts", {}).items()}
    summary["timing"] = result["timing"]

    # negatives: each must be refused
    neg: dict[str, dict] = {}
    first = rec["roots"][0]["port"]
    other = dict(rec, roots=[dict(rec["roots"][0], root="00" * 64), *rec["roots"][1:]])
    v = V.matches_public(other, header["scheme"], fv["roots"], fv["domain_ids"], header["instances"])
    neg["registration-root-mismatch"] = {"refused": not v.ok, "codes": list(v.codes)}
    v = V.matches_public(rec, header["scheme"], fv["roots"], {**fv["domain_ids"], first: "00" * 64}, header["instances"])
    neg["public-file-domain-mismatch"] = {"refused": not v.ok, "codes": list(v.codes)}
    v = V.check(rec, R2.program([(template, n)], mixed=False), query, law, SCHEME, want, leaf_layer=leaf_layer)
    neg["registration-partition-mismatch"] = {"refused": not v.ok, "codes": list(v.codes)}
    v = V.check(dict(rec, roots=[dict(rec["roots"][0], domain="00" * 64), *rec["roots"][1:]]), program, query, law, SCHEME, want,
                leaf_layer=leaf_layer)
    neg["registration-domain-not-the-layout's"] = {"refused": not v.ok, "codes": list(v.codes)}
    v = V.check(rec, program, query, D.parse_law(f"subset:{a.k + 1}"), SCHEME, want, leaf_layer=leaf_layer)
    neg["registration-law-not-the-agreement's"] = {"refused": not v.ok, "codes": list(v.codes)}
    v = V.check(dict(rec, previous=R.digest(rec)), program, query, law, SCHEME, want, leaf_layer=leaf_layer, log=[rec])
    neg["second-registration-for-the-window"] = {"refused": not v.ok, "codes": list(v.codes)}
    v = V.check(dict(rec, leaf_layer=None), program, query, law, SCHEME, want, leaf_layer=leaf_layer)
    neg["registration-null-leaf-layer (R4)"] = {"refused": not v.ok, "codes": list(v.codes)}
    if "registration" in result:
        neg.update(audit_negatives(result, law, a.delta, (live or {}).get("roots") or {}))
    neg["public-body-altered-leaf-layer-updated (Lean roots)"] = altered_public(
        a.lean_verify, st, n, out / "session", out / "neg-public-body", rec,
        lambda r, digest: V.check(r, program, query, law, SCHEME, want, leaf_layer=digest))
    if a.bind:
        other_part = P.digest(P.partition(R2.program([(template, n)], mixed=False), query))
        b = bindings(dict(st["meta"], partition={"rule": P.FORMAT, "digest": other_part}), header, rec, checks["bindings"]["class_sha512"])
        neg["statement-binds-another-partition"] = {"refused": not b["ok"], "each": b["each"]}
    draw = (record or {}).get("unit_draw") or {"law": "subset", "population": n, "k": a.k, "units": list(range(a.k))}
    for name, bad in (("draw-over-another-population", {**draw, "population": n + 1}),
                      ("draw-outside-the-partition", {**draw, "units": draw["units"][:-1] + [n]}),
                      ("draw-under-another-law", {**draw, "k": a.k - 1, "units": draw["units"][:-1]})):
        v = V.check_draw(bad, rec, law)
        neg[name] = {"refused": not v.ok, "codes": list(v.codes)}
    altered = {**draw, "units": sorted(set(draw["units"][:-1]) | {next(u for u in range(n) if u not in draw["units"])})}
    lean_alt = lean_draw_test(a.lean, [{"record": altered, "statement": draw, "instances": len(draw["units"])},
                                       {"record": draw, "statement": draw, "instances": len(draw["units"]) + 1},
                                       {"record": None, "statement": draw, "instances": len(draw["units"])}], out, "negatives")
    neg["draw-altered-after-session (Lean U1)"] = {"refused": drawcheck_refused(lean_alt, 0, "U1"), "lean": lean_alt[:1],
                                                   "proved-check": V.check_proved(altered, proved_units(header, live or {}),
                                                                                  (live or {}).get("instances", -1)).to_json()}
    neg["instances-not-the-draw (Lean U3)"] = {"refused": drawcheck_refused(lean_alt, 1, "U3"), "lean": lean_alt[1:2]}
    neg["draw-dropped-from-record (Lean U1)"] = {"refused": drawcheck_refused(lean_alt, 2, "U1"), "lean": lean_alt[2:3]}
    for case in ("unit_draw_ignored_refused", "unit_draw_session"):
        p = run([a.bin, "selftest", "--instances", str(st["dir"] / "inst-4.bin"), "--circuit", str(st["dir"] / "circuit.txt"),
                 "--tables", str(st["dir"] / "tables"), "--only", case], out / f"selftest-{case}.log")
        res = next((json.loads(line[4:]) for line in p.stdout.splitlines() if line.startswith("NEG\t")), None)
        key = "prover-ignores-the-draw (M0 selftest)" if case == "unit_draw_ignored_refused" else "m0-selftest-unit-draw-session"
        neg[key] = {"refused" if case == "unit_draw_ignored_refused" else "passed": bool(res and res.get("pass")), "case": res}
    tn = 8
    tprog = P.population_program(template, tn)
    tpart = P.partition(tprog, query)
    tbind = {"partition": P.digest(tpart), "program": tpart["program"], "indices": list(range(tn))} if a.bind else None
    tst = stage(a.set, [tn], out / "stage-tampered", a.m0_pythonpath, tampered=True, bind=tbind)
    if a.bind:
        hold_bound(tst, tprog, query)
    theader = public_header(tst["dir"], tn)
    tlaw = D.parse_law("subset:2")
    twant = {p: {"leaves": c, "domain": theader["frame_v3"]["domain_ids"][p]} for p, (_, c) in ports(tst["meta"], tn).items()}
    trec = R.record(tprog, query, tlaw, theader["scheme"], registered_roots(theader, twant),
                    {"kind": "stand-in", "of": "tampered copy of the captured set: bit 0 of every instance's first output word flipped"},
                    leaf_layer=sha512_file(tst["dir"] / f"pub-{tn}.bin"))
    trecord, tlive = session(a.bin, tst, tn, "subset:2", a.port + 1, out / "session-tampered")
    tres = audit_session(trec, R.receipt(trec, "one-stage-e2e verifier (a0.py)"), tst, theader, tlaw, trecord, tlive, a.lean,
                         a.lean_verify, out / "session-tampered", a.delta)
    tlean = tres.get("verdicts", {}).get("lean-verify", {})
    neg["wrong-output-word"] = {"refused": trecord is not None and tres.get("accepted") is False and tres.get("profile") is None
                                and lean_refused(tlean),
                                "verifier": ((trecord or {}).get("verdict") or {}).get("why"),
                                "lean": tlean.get("why"), "lean_refused": lean_refused(tlean),
                                "prover": (tlive or {}).get("why"), "draw": (trecord or {}).get("unit_draw")}
    (out / "negatives.json").write_text(json.dumps(neg, indent=1, sort_keys=True))
    summary["negatives"] = {k: v.get("refused", v.get("passed")) for k, v in neg.items()}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True))
    ok = bool(result.get("accepted")) and all(summary["negatives"].values())
    print("A0 " + json.dumps(summary, sort_keys=True))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
