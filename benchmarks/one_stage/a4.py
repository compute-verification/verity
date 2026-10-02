"""One-stage audit, run A4: a GEMM cell whose weights are a registered value, bound once at registration and read by every
session, then the same cell with the weights committed in the statement, for the per-session cost.

    PYTHONPATH=<tree>/packages/verity/src:<tree>/backends/numerical/python:<tree>/backends/flock/python:<tree>/integrations/vllm:<tree>/protocols/sampled_proofs \\
      python benchmarks/one_stage/a4.py --bin flock-circuit --lean flock-verify --out DIR [--K 2048] [--n 8] [--cols 16]
      [--k 8] [--windows 2] [--statement verity/flock-circuit@967b8d06+seed-injection]

* **Program.** `RegisteredGemm`: `n` Calls of `GemmCoordinate_v2{K, DOT=AmpereBF16TcDot16_v2}` (the generated BF16 set's
  instances), Call `i` reading its activation row `x[i]` (a per-run input) and the weight column `W[pos[i]]`, `W` a root
  parameter of `cols` columns (the set's columns and decoys, permuted). The partition is `Q_template_instance` v0.
* **Registration.** Window 1's `verity/registration/v2` registers `W` (`one_stage.registered`: one hm96 leaf per column under
  registration salts, its frame-v3-sha512 root); every later window's record names the same entry and the log holds its
  root (R7). Each window also registers its session's statement roots (`x`, the `w` table, `y`) under M0's set-bound
  domains, as A0's stand-in does.
* **Sessions.** Each window's statement is a shared-row file whose `w` table is the registered columns at the read
  positions, under their registration salts with their paths (`Registered.statement`); M0 proves it in a loopback session
  over the verifier's draw, M0's verifier checks it live, and the Lean verifier replays it with the verifier's own reads
  file (`--registered`, `registered.reads_file` from its own program), which opens every table row against the
  registered root. The audit record holds the roots the Lean verdict opened against.
* **Baseline.** The same statement with the `w` table under fresh salts and no `registered` header (`--baseline`): the
  weights committed inside the statement, verified without registered reads.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import a0  # noqa: E402
from verity.commitments import hm96  # noqa: E402
from verity.commitments.identity import identity_digest  # noqa: E402
from verity.commitments.indexed import RangeIndexedDomain  # noqa: E402
from verity.commitments.merkle import CommitmentDomain  # noqa: E402
from verity.ir.codec import canonical_json, encode_program  # noqa: E402
from verity.ir.defs import CompositeDefinition, bind  # noqa: E402
from verity.ir.program import Program  # noqa: E402
from verity.ir.refs import coll_concat  # noqa: E402
from verity.ir.types import Array  # noqa: E402
from verity_numerical.bench import templates as TM  # noqa: E402
from verity_numerical.bench.generate import generate  # noqa: E402
from verity_numerical.bench.input_sets import InputSet  # noqa: E402
from verity_sampled_proofs.one_stage import audit, draw as D, partition as P, registered as G, registration as R  # noqa: E402

SCHEME = a0.SCHEME
VALUE = "W"
#: the template parameter that reads W, and the circuit port it is
PARAM, PORT = "wrow", "w"


def program_for(template, n: int, cols: int, pos: list[int]) -> Program:
    (_, xt), (_, wt) = template.signature()[0]
    ret = template.signature()[1]

    def sig(S):
        return (("x", Array(n, xt)), (VALUE, Array(cols, wt))), Array(n, ret)

    def body(B, S, x, W):
        return coll_concat(sig(S)[1], [B.call(template, x[i], W[p]) for i, p in enumerate(pos)])

    return Program(bind(CompositeDefinition("RegisteredGemm", 0, (), sig, body, register=False)))


def want_ports(s: InputSet, header: dict, n: int) -> dict[str, dict]:
    """The verifier's own derivation of each statement port's leaf count and domain: M0's set-bound frame domains (A0's
    stand-in), over the shared-row tables' rows for an input port."""
    out = {}
    for p, schema in header["frame_v3"]["schemas"].items():
        count = header["shared_rows"][p] if schema == "hm96-sha512/row/v1" else n
        b = identity_digest(a0.STAND_IN_BINDING, {"set": s.name, "content_digest": s.content_digest, "lo": 0, "hi": n, "port": p,
                                                  "schema": schema})
        out[p] = {"leaves": count, "domain": CommitmentDomain(bytes.fromhex(b), -1, RangeIndexedDomain(0, count), hash="sha512").domain_id.hex()}
    return out


def stage_window(text: str, s: InputSet, d: Path, rows_x, outs, reg: G.Registered, pos: list[int], registered: bool,
                 n: int) -> dict:
    """One session's statement: circuit (hard link), the prover's and the verifier's files, an empty tables directory."""
    d.mkdir(parents=True, exist_ok=True)
    (d / "tables").mkdir(exist_ok=True)
    circuit = d / "circuit.txt"
    if not circuit.exists():
        circuit.write_text(text)
    positions, refs = G.table(pos)
    rows_w, reg_salts, entry = reg.statement(positions)
    salts_w = reg_salts if registered else [hm96.SHA512.fresh_salt() for _ in positions]
    t = time.time()
    from verity_flock import circuit as FC

    units = list(range(n))
    head = FC.write(text, s, 0, n, d / f"inst-{n}.bin", d / f"pub-{n}.bin", None, units, share_rows=True, outs=outs,
                    tables={"x": rows_x, PORT: np.asarray(rows_w)}, refs={"x": units, PORT: refs},
                    salts={"x": [hm96.SHA512.fresh_salt() for _ in range(n)], PORT: salts_w},
                    registered={PORT: entry} if registered else None)
    return {"dir": d, "pin": hashlib.sha512(text.encode()).hexdigest(), "stage_s": round(time.time() - t, 2), "header": head}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--lean", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--K", type=int, default=2048)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--cols", type=int, default=16)
    ap.add_argument("--k", type=int, default=None, help="the draw's size (default: every unit)")
    ap.add_argument("--windows", type=int, default=2)
    ap.add_argument("--baseline", type=int, default=1, help="sessions with the weights committed in the statement")
    ap.add_argument("--seed", type=int, default=20261002)
    ap.add_argument("--delta", type=float, default=0.01)
    ap.add_argument("--port", type=int, default=7561)
    ap.add_argument("--statement", default="verity/flock-circuit@967b8d06+seed-injection")
    ap.add_argument("--zk-bin", default=None, help="a flock-circuit build with --zk: one registered and one in-statement "
                                                     "statement proved under M0 and --zk, M0's own verifier")
    ap.add_argument("--zk-runs", type=int, default=2)
    ap.add_argument("--statement-check", type=int, default=0,
                    help="R: stage a registered and an in-statement file (no proof) and time the Lean verifier's statement check "
                         "of each R times, with and without --registered")
    a = ap.parse_args()
    a0.STATEMENT_ID = a.statement
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    n, k = a.n, a.k or a.n
    summary: dict = {"K": a.K, "n": n, "cols": a.cols, "k": k, "statement": a.statement}

    # the cell: the generated BF16 set's instances, and the weight matrix whose columns they read
    sub = TM.gemm_coordinate(a.K)
    s = InputSet.open(generate(sub, n, a.seed, out / "set")["path"])
    template = s.subcircuit.definition()
    rows = {p.name: np.asarray(s.port(p.name, 0, n), dtype=np.uint16).reshape(n, -1) for p in s.subcircuit.inputs}
    outs = {p.name: np.asarray(s.port(p.name, 0, n), dtype=np.uint16).reshape(n, -1) for p in s.subcircuit.outputs}
    rng = np.random.default_rng(a.seed)
    distinct = sorted({tuple(r) for r in rows[PORT].tolist()})
    if len(distinct) > a.cols:
        raise SystemExit(f"--cols {a.cols} < the set's {len(distinct)} distinct columns")
    table = distinct + [tuple(r) for r in rng.integers(0, 1 << 16, (a.cols - len(distinct), a.K)).tolist()]
    table = [table[i] for i in rng.permutation(len(table))]
    pos = [table.index(tuple(r)) for r in rows[PORT].tolist()]

    # the verifier's program, partition, law and read map
    program = program_for(template, n, a.cols, pos)
    query = P.template_instance_query(template)
    part = P.partition(program, query)
    law_text = f"subset:{k}"
    law = D.parse_law(law_text)
    values = {VALUE: a.K}
    reads = G.reads(program, query, values)
    assert reads == {PARAM: {"value": VALUE, "rows": pos}}, reads
    prog_sha = P.program_sha512(program)
    st_dir = out / "stage"
    st_dir.mkdir(parents=True, exist_ok=True)
    (st_dir / "program.json").write_bytes(canonical_json(encode_program(program)))
    (st_dir / "partition.json").write_text(json.dumps(part, sort_keys=True))

    # registration of the value, once: the prover's rows, salts and tree
    t = time.time()
    reg = G.Registered.commit(prog_sha, VALUE, [list(r) for r in table])
    summary["register_value_s"] = round(time.time() - t, 3)
    summary["registered"] = reg.entry()
    own = G.reads_file(program, query, values, {VALUE: reg.root.hex()}, ports={PARAM: PORT})
    (st_dir / "registered.json").write_text(json.dumps(own, indent=1))

    from verity_flock import circuit as FC

    t = time.time()
    text = FC.compose(FC.lowering_for_set(s), s.subcircuit, partition=P.digest(part), program=prog_sha, statics=FC.set_statics(s))
    summary["compose_s"] = round(time.time() - t, 1)
    summary["circuit_sha512"] = FC.digest(text)

    log: list[dict] = []
    windows: list[dict] = []
    for w in range(a.windows + a.baseline):
        registered = w < a.windows
        name = f"window-{w + 1}" if registered else f"baseline-{w - a.windows + 1}"
        wd = out / name
        st = stage_window(text, s, wd / "stage", rows["x"], outs, reg, pos, registered, n)
        st["partition_file"], st["program_file"] = st_dir / "partition.json", st_dir / "program.json"
        if registered:
            st["registered_file"] = st_dir / "registered.json"
        header = st.pop("header")
        pub = st["dir"] / f"pub-{n}.bin"
        want = want_ports(s, header, n)
        fv = header["frame_v3"]
        leaf_layer = a0.sha512_file(pub)
        roots = a0.registered_roots(header, want)
        window = {"kind": "stand-in", "of": f"a4 {name}: the generated set {s.name}, committed by this driver", "window": w + 1}
        prior = log if registered else []
        rec = R.record(program, query, law, SCHEME, roots, window, leaf_layer=leaf_layer,
                       previous=R.digest(prior[-1]) if prior else None, registered=[reg.entry()] if registered else ())
        checks = {"registration": R.check(rec, program, query, law, SCHEME, want, leaf_layer=leaf_layer, log=prior,
                                          **({"registered": values} if registered else {})).to_json(),
                  "public": R.matches_public(rec, header["scheme"], fv["roots"], fv["domain_ids"], header["instances"]).to_json()}
        receipt = R.receipt(rec, "one-stage registered-gemm verifier (a4.py)")
        (wd / "registration.json").write_text(json.dumps(rec, indent=1, sort_keys=True))
        if not all(c["ok"] for c in checks.values()):
            windows.append({"name": name, "refused_before_draw": checks})
            print(f"A4 {name} refused before the draw: {json.dumps(checks)}", flush=True)
            continue
        if registered:
            log.append(rec)
        t = time.time()
        record, live = a0.session(a.bin, st, n, law_text, a.port + w, wd / "session")
        session_s = round(time.time() - t, 1)
        result = a0.audit_session(rec, receipt, st, header, law, record, live, a.lean, a.lean, wd / "session", a.delta)
        (wd / "audit.json").write_text(json.dumps(result, indent=1, sort_keys=True, default=str))
        live = live or {}
        lean_v = result.get("verdicts", {}).get("lean-verify", {})
        windows.append({
            "name": name, "registered": registered, "registration_digest": R.digest(rec), "checks_before_draw": {k: v["ok"] for k, v in checks.items()},
            "accepted": result.get("accepted"), "verdicts": {k: v.get("accepted") for k, v in result.get("verdicts", {}).items()},
            "lean_why": lean_v.get("why"), "lean_registered": lean_v.get("registered"),
            "audit_checks_failed": [c for c, v in (result.get("checks") or {}).items() if not v.get("ok")],
            "integrity_inputs_registered": ((result.get("integrity_profile") or {}).get("inputs") or {}).get("registered"),
            "stage_s": st["stage_s"], "session_s": session_s, "pub_bytes": pub.stat().st_size,
            "pub_header_bytes": len(pub.read_bytes().split(b"\n", 1)[0]),
            "timing": {key: live.get(key) for key in ("e2e_s", "witness_s", "prove_s", "prove_total_s", "tp_s", "proof_bytes", "k_log",
                                                      "dense_m", "units", "instances", "wire_sent", "wire_recv")},
            "m0_verify_s": ((record or {}).get("verdict") or {}).get("verify_s"), "lean_verify_s": lean_v.get("verify_s"),
            "negatives": (negatives(a, st, n, wd, result, law, reg, program, query, values, rec, want, leaf_layer, log) if registered and w == 0
                          else {"lean-weights-committed-afresh-read-as-registered": lean_refuses(
                              a.lean, {**st, "registered_file": st_dir / "registered.json"}, n, wd, "afresh-as-registered")}
                          if w == a.windows else None),
        })
        (out / "summary.json").write_text(json.dumps({**summary, "windows": windows}, indent=1, sort_keys=True, default=str))
        print(f"A4 {name} " + json.dumps(windows[-1], sort_keys=True, default=str), flush=True)
        os.remove(st["dir"] / f"inst-{n}.bin")
    summary["windows"] = windows
    if a.zk_bin:
        summary["zk"] = zk_sessions(a, text, s, out / "zk", rows["x"], outs, reg, pos, n, st_dir)
    if a.statement_check:
        summary["statement_check"] = statement_check(a, text, s, out / "statement-check", rows["x"], outs, reg, pos, n, st_dir)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str))
    ok = all(x.get("accepted") for x in windows) and all(v["refused"] for x in windows for v in (x.get("negatives") or {}).values())
    print("A4 " + json.dumps({"ok": ok}), flush=True)
    return 0 if ok else 1


def zk_sessions(a, text: str, s: InputSet, out: Path, rows_x, outs, reg: G.Registered, pos: list[int], n: int, st_dir: Path) -> dict:
    """A registered statement and an in-statement one, each proved by `--zk-bin` in loopback sessions under M0 and `--zk`
    (`--zk-runs` sessions each, no draw), verified by that build's own server; the Lean verifier's statement check of the
    registered public file (`statement --registered`: every table row's path and every instance's position), and its
    verdict on a `--zk` session."""
    import subprocess

    res: dict = {}
    for i, (name, registered) in enumerate((("registered", True), ("in-statement", False))):
        st = stage_window(text, s, out / name, rows_x, outs, reg, pos, registered, n)
        d = st["dir"]
        res[name] = {"stage_s": st["stage_s"], "pub_bytes": (d / f"pub-{n}.bin").stat().st_size}
        for j, mode in enumerate(("m0", "zk")):
            z = ["--zk"] if mode == "zk" else []
            port = a.port + 50 + 4 * i + j
            log = out / f"{name}-{mode}.serve.log"
            with open(log, "w") as f:
                serve = subprocess.Popen([a.zk_bin, "serve", *z, "--listen", f"127.0.0.1:{port}", "--out", str(out / f"sessions-{name}-{mode}"),
                                          "--instances", str(d / f"pub-{n}.bin"), "--circuit", str(d / "circuit.txt"), "--tables",
                                          str(d / "tables"), "--pin", st["pin"], "--operator", "a4-zk", "--sessions", str(a.zk_runs)],
                                         stdout=f, stderr=subprocess.STDOUT)
            try:
                for _ in range(1800):
                    if "SERVING" in log.read_text() or serve.poll() is not None:
                        break
                    time.sleep(0.5)
                p = a0.run([a.zk_bin, "prove", *z, "--verifier", f"127.0.0.1:{port}", "--instances", str(d / f"inst-{n}.bin"),
                            "--circuit", str(d / "circuit.txt"), "--tables", str(d / "tables"), "--warm", "0", "--runs", str(a.zk_runs)],
                           out / f"{name}-{mode}.prove.log")
                serve.wait(timeout=600)
            finally:
                if serve.poll() is None:
                    serve.kill()
            lives = [json.loads(line[5:]) for line in p.stdout.splitlines() if line.startswith("LIVE\t")]
            res[name][mode] = [{"accepted": v.get("accepted"), "why": v.get("why"), "prove_total_s": v.get("prove_total_s"),
                                "witness_s": v.get("witness_s"), "e2e_s": v.get("e2e_s"), "proof_bytes": v.get("proof_bytes"),
                                "verify_s": (v.get("verdict") or {}).get("verify_s"), "dense_m": v.get("dense_m"),
                                "zk": ((v.get("identity") or {}).get("hooks") or {}).get("zk"),
                                "proof_class": (v.get("identity") or {}).get("proof_class")} for v in lives] or [{"accepted": False, "why": p.stderr[-600:]}]
            print(f"A4 zk {name} {mode} " + json.dumps(res[name][mode]), flush=True)
        if registered:
            stmt = a.statement.split("+", 1)[0]
            p = subprocess.run([a.lean, "statement", "--statement", stmt, "--circuit", str(d / "circuit.txt"), "--public", str(d / f"pub-{n}.bin"),
                                "--partition", str(st_dir / "partition.json"), "--program", str(st_dir / "program.json"),
                                "--registered", str(st_dir / "registered.json")], capture_output=True, text=True, check=False)
            line = (p.stdout.strip().splitlines() or [""])[0]
            res[name]["lean_statement"] = {"rc": p.returncode, "registered": (json.loads(line).get("registered") if line.startswith("{") else None),
                                           "why": None if p.returncode == 0 else (p.stdout + p.stderr)[-400:]}
            zs = sorted((out / f"sessions-{name}-zk").glob("*/session.json"))
            if zs:
                zd = out / "lean-zk"
                zd.mkdir(exist_ok=True)
                if not (zd / "sessions").exists():
                    (zd / "sessions").symlink_to(zs[-1].parent.parent)
                v = a0.lean_verify(a.lean, {**st, "partition_file": st_dir / "partition.json", "program_file": st_dir / "program.json",
                                            "registered_file": st_dir / "registered.json"}, n, zd)
                res[name]["lean_verify_zk_session"] = {"accepted": v["accepted"], "why": v.get("why")}
            print(f"A4 zk {name} lean " + json.dumps({k: v for k, v in res[name].items() if k.startswith("lean")}), flush=True)
        os.remove(d / f"inst-{n}.bin")
    return res


def statement_check(a, text: str, s: InputSet, out: Path, rows_x, outs, reg: G.Registered, pos: list[int], n: int,
                    st_dir: Path) -> dict:
    """The verifier's per-session cost of the registered reads outside the proof: the Lean verifier's `statement` on the
    registered file with `--registered` (every table row's path and every instance's position) against the in-statement
    file without it, `--statement-check` times each, and the public files' sizes."""
    import subprocess

    res: dict = {}
    stmt = a.statement.split("+", 1)[0]
    for name, registered in (("registered", True), ("in-statement", False)):
        st = stage_window(text, s, out / name, rows_x, outs, reg, pos, registered, n)
        d = st["dir"]
        os.remove(d / f"inst-{n}.bin")
        pub = d / f"pub-{n}.bin"
        args = [a.lean, "statement", "--statement", stmt, "--circuit", str(d / "circuit.txt"), "--public", str(pub),
                "--partition", str(st_dir / "partition.json"), "--program", str(st_dir / "program.json"),
                *(["--registered", str(st_dir / "registered.json")] if registered else [])]
        times, rc, line = [], None, ""
        for _ in range(a.statement_check):
            t = time.time()
            p = subprocess.run(args, capture_output=True, text=True, check=False)
            times.append(round(time.time() - t, 3))
            rc, line = p.returncode, (p.stdout.strip().splitlines() or [""])[0]
        res[name] = {"rc": rc, "seconds": times, "stage_s": st["stage_s"], "pub_bytes": pub.stat().st_size,
                     "pub_header_bytes": len(pub.read_bytes().split(b"\n", 1)[0]),
                     "registered": json.loads(line).get("registered") if line.startswith("{") else None,
                     "why": None if rc == 0 else (p.stdout + p.stderr)[-400:]}
        print(f"A4 statement-check {name} " + json.dumps(res[name]), flush=True)
    return res


def lean_refuses(lean: str, st: dict, n: int, wd: Path, name: str) -> dict:
    """The Lean verifier's replay of the window's own session (`wd/session/sessions`, linked) under `st`'s files."""
    d = wd / f"neg-{name}"
    d.mkdir(parents=True, exist_ok=True)
    if not (d / "sessions").exists():
        (d / "sessions").symlink_to(wd / "session" / "sessions")
    v = a0.lean_verify(lean, st, n, d)
    return {"refused": v["accepted"] is False and bool(v.get("why")) and v["why"] != "no session recorded", "why": v.get("why")}


def negatives(a, st: dict, n: int, wd: Path, result: dict, law, reg: G.Registered, program, query, values, rec, want, leaf_layer,
              log) -> dict[str, dict]:
    """Each must be refused: the Lean verifier's replay of the session against a reads file holding another root of W (a
    second registration of the same columns under fresh salts), or moving one unit's position; a later window that
    registers W under another root (R7-held); the audit record with the reads opened against another root."""
    neg = {}
    other = G.Registered.commit(reg.program, reg.value, [list(r) for r in reg.rows])
    own = json.loads(Path(st["registered_file"]).read_text())
    for name, edit in (("lean-reads-another-root-of-W", lambda ps: ps[PORT].__setitem__("root", other.root.hex())),
                       ("lean-reads-moved-position", lambda ps: ps[PORT]["reads"].__setitem__(0, (ps[PORT]["reads"][0] + 1) % reg.leaves))):
        doc = json.loads(json.dumps(own))
        edit(doc["ports"])
        f = wd / f"own-{name}.json"
        f.write_text(json.dumps(doc))
        neg[name] = lean_refuses(a.lean, {**st, "registered_file": f}, n, wd, name)
    later = R.record(program, query, law, SCHEME, rec["roots"], {**rec["window"], "window": 99}, leaf_layer=leaf_layer,
                     previous=R.digest(rec), registered=[other.entry()])
    codes = R.check(later, program, query, law, SCHEME, want, leaf_layer=leaf_layer, log=[*log], registered=values).codes
    neg["registration-re-registers-W-under-another-root (R7-held)"] = {"refused": "R7-held" in codes, "codes": list(codes)}
    neg.update({k: v for k, v in a0.audit_negatives(result, law, a.delta, {r["port"]: r["root"] for r in rec["roots"]}).items()
                if k.startswith("registered") or k.startswith("session-root")})
    return neg


if __name__ == "__main__":
    raise SystemExit(main())
