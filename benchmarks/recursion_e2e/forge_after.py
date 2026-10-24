"""forge-after: V*'s three forgeries that the prover's firewall stops before release (coef, dirs, message), made after an
honest firewalled prove instead, and put to the Lean verifier.

    python3 benchmarks/recursion_e2e/forge_after.py --work FLOCK_WORK --tag TAG --run-out OUT_OF_THE_RUN [--bin NAME=PATH ...]

A forgery staged before proving (`rec_vstage --forge`) puts a wrong row in the prover's ``inst.bin``, and the firewall refuses
to release a proof of it, so Lean never sees a session. Here the session is an honest one pod 85 already proved behind the
firewall (``v-sess/L0`` for coef and dirs, ``v-sess/alg-p0`` for message). Each forgery changes the verifier's statement
alone, on copies of its files: row 0 of the port (``coef``, ``dirs``, ``m0``) gets bit 0 of its first word flipped under
the row's own salt, so that the proved row and the verifier's registered row differ by one bit, as in the staged forgery.

* ``<kind>-bc``: that row's ``b || c`` in ``pub.bin``, and nothing else.
* ``<kind>-value``: the value registered again with the forged row: the ``b || c``, ``pub.bin``'s read paths, and the port's
  root in ``registered.json``.

Each case, and each statement's unmodified copy (the control, which must accept), is verified as pod 85's STEP=lean
verifies it (`flock-verify verify --zk --registered` with its ``--rec-*`` flags), with each binary given. The run's
``out/lean/own-<stmt>.txt`` holds 85's own flags for the binary it built; another binary's are derived here as 85 derives
them. Nothing under FLOCK_WORK is written. Writes ``$OUT/forge-after.json`` and prints one line per verify.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

CASES = {"coef": ("L0", "coef"), "dirs": ("L0", "dirs"), "message": ("alg-p0", "m0")}
VERIFIER_FILES = ("circuit.txt", "pub.bin", "program.json", "partition.json", "registered.json")
BC = 128


def header(data: bytes) -> tuple[dict, int]:
    nl = data.index(b"\n") + 1
    return json.loads(data[:nl]), nl


def table_rows(inst: bytes, text: str, port: str, n: int) -> tuple[list[list[int]], list[bytes]]:
    from verity_flock import rec_vstage as S
    from verity_flock import rec_vstar as V

    words = {p[0]: p[1] for p in json.loads(text.split("\n", 2)[1][5:])["ports"]}[port]
    rows, salts = [], []
    for r in range(n):
        at, st = S._offsets(inst, text, port, r)
        rows.append([int.from_bytes(inst[at + 2 * w:at + 2 * w + 2], "little") for w in range(words)])
        salts.append(inst[st:st + V.SALT])
    return rows, salts


def forge(stmt: Path, inner: Path, port: str, out: Path) -> dict:
    """`out/bc` and `out/value` (the module docstring) from the statement `stmt`; what each changed."""
    from verity.core.protocols.sampled_proofs.one_stage import registered as RG
    from verity_flock import rec_outer as O

    text = (stmt / "circuit.txt").read_text()
    inst, pub = (stmt / "inst.bin").read_bytes(), (stmt / "pub.bin").read_bytes()
    reads = json.loads((stmt / "registered.json").read_text())
    port_ = reads["ports"].get(port)
    if port_ is None:
        raise SystemExit(f"{stmt}: no registered port {port} (ports {sorted(reads['ports'])})")
    ph, pnl = header(pub)
    reg = ph["registered"][port]
    pos = reg["positions"]
    rows, salts = table_rows(inst, text, port, len(pos))
    wb, schema = int(port_.get("word_bits", 16)), port_.get("row_schema", RG.SCHEMA)
    bc = lambda row, salt: RG._INST.commit_string(RG._INST.default_key, RG.row_digest(row, wb, schema), salt)
    honest = bc(rows[0], salts[0])
    at = pub.find(honest, pnl)
    if at < 0 or pub.find(honest, at + 1) >= 0:
        raise SystemExit(f"{stmt}: {port} row 0's b || c is not in pub.bin exactly once")
    row0 = [rows[0][0] ^ 1, *rows[0][1:]]
    forged = bc(row0, salts[0])
    rec = {"statement": str(stmt), "port": port, "value": port_["value"], "position": pos[0], "word0": [rows[0][0], row0[0]],
           "bc_offset": at, "bc_sha512": [hashlib.sha512(honest).hexdigest()[:16], hashlib.sha512(forged).hexdigest()[:16]]}
    for case in ("honest", "bc", "value"):
        (out / case).mkdir(parents=True, exist_ok=True)
        for f in VERIFIER_FILES:
            shutil.copyfile(stmt / f, out / case / f)
    b = bytearray(pub)
    b[at:at + BC] = forged
    (out / "bc" / "pub.bin").write_bytes(bytes(b))
    rec["bc"] = {"files": {"pub.bin": [at, at + BC]}}
    if sorted(pos) != list(range(int(port_["leaves"]))):
        rec["value"] = {"skipped": f"the table reads positions {pos}, not all {port_['leaves']} of the value's rows"}
        shutil.rmtree(out / "value")
        return rec
    program = O.session_program((inner / "public.json").read_bytes())
    by_pos = sorted(zip(pos, rows, salts))
    mk = lambda rs: RG.Registered.commit(program, port_["value"], rs, [s for _, _, s in by_pos], wb, schema)
    h = mk([r for _, r, _ in by_pos])
    if h.root.hex() != port_["root"] or [h.path(p).hex() for p in pos] != reg["paths"]:
        rec["value"] = {"skipped": "the rows and salts do not register to the statement's root and paths"}
        shutil.rmtree(out / "value")
        return rec
    f = mk([row0 if p == pos[0] else r for p, r, _ in by_pos])
    ph2 = json.loads(pub[:pnl])
    ph2["registered"][port]["paths"] = [f.path(p).hex() for p in pos]
    h2 = json.dumps(ph2, sort_keys=True).encode() + b"\n"
    if json.dumps(ph, sort_keys=True).encode() + b"\n" != pub[:pnl]:
        raise SystemExit(f"{stmt}: pub.bin's header is not sort_keys JSON; cannot rewrite it byte for byte")
    (out / "value" / "pub.bin").write_bytes(h2 + bytes(b[pnl:]))
    reads["ports"][port] = {**port_, "root": f.root.hex()}
    (out / "value" / "registered.json").write_text(json.dumps(reads))
    rec["value"] = {"root": [port_["root"][:16], f.root.hex()[:16]], "paths_changed": sum(
        a != c for a, c in zip(reg["paths"], ph2["registered"][port]["paths"]))}
    return rec


def own_flags(binary: Path, dd: Path, n: str, stmts: dict[str, Path], o: Path) -> dict[str, list[str]]:
    """85's STEP=lean flags for each statement (its python block), derived with `binary`."""
    from verity_flock import rec_bindings as RB
    from verity_flock import rec_vstage as VS

    o.mkdir(parents=True, exist_ok=True)
    st = dd / "stage"
    with (o / "rec-inner.json").open("w") as f:
        subprocess.run([str(binary), "rec-inner", "--circuit", str(st / "circuit.txt"), "--public", str(st / f"pub-{n}.bin"),
                        "--partition", str(st / "partition.json"), "--program", str(st / "program.json")], stdout=f, check=True)
    txt = subprocess.run([str(binary), "rec-statements", "--rec-inner", str(o / "rec-inner.json")], capture_output=True,
                         text=True, check=True).stdout
    (line,) = [json.loads(ln[9:]) for ln in txt.splitlines() if ln.startswith("RECVSTAR ")]
    names = [s[0] for s in line["statements"]]
    v = dd / "v"
    shared = VS.shared_rows(sorted(v.glob("alg/*")), sorted(v.glob("stacked/L*")),
                            [d for d in [*sorted(v.glob("L[0-9]*")), v / "cap", *sorted(v.glob("cap/L*"))]
                             if (d / "registered.json").is_file()])["ports"]
    out = {}
    for name, c in stmts.items():
        with (c / "circuit.txt").open() as f:
            unit = f.readline().split()[1]
        ports = json.loads((c / "registered.json").read_text())["ports"]
        rows = o / f"rows-{name}.json"
        rows.write_text(json.dumps({"format": "verity/registered-reads/v0",
                                    "ports": shared | ({"acc0": ports["acc"]} if "acc0" not in shared and "acc" in ports else {})}))
        flags = ["--rec-inner", str(o / "rec-inner.json"), "--rec-rows", str(rows)]
        if unit in names:
            flags += ["--rec-part", str(names.index(unit))]
        b = RB.bindings(json.loads((c / "partition.json").read_text()), json.loads((c / "program.json").read_text()))
        (o / f"bindings-{name}.json").write_text(json.dumps(b))
        out[name] = flags + ["--rec-bindings", str(o / f"bindings-{name}.json")]
    return out


def verify(binary: Path, case: Path, own: list[str], inner_flags: list[str], session: Path, log: Path) -> dict:
    cmd = [str(binary), "verify", "--zk", "--statement", "verity/flock-circuit", "--circuit", str(case / "circuit.txt"),
           "--public", str(case / "pub.bin"), "--partition", str(case / "partition.json"), "--program", str(case / "program.json"),
           "--registered", str(case / "registered.json"), *own, *inner_flags, "--session", str(session)]
    with log.with_suffix(".log").open("w") as so, log.with_suffix(".err").open("w") as se:
        rc = subprocess.run(cmd, stdout=so, stderr=se).returncode
    tail = [ln for ln in (log.with_suffix(".log").read_text() + log.with_suffix(".err").read_text()).splitlines() if ln.strip()]
    return {"rc": rc, "why": tail[-1][:400] if tail else "", "cmd": " ".join(cmd)}


def tree_sha(d: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(x for x in d.rglob("*") if x.is_file()):
        h.update(str(p.relative_to(d)).encode() + b"\0" + hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--work", required=True, type=Path)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--run-out", required=True, type=Path, help="the out/ of the run whose lean step verified FLOCK_WORK")
    ap.add_argument("--bin", action="append", default=[], help="NAME=PATH: a flock-verify; NAME=run uses the run's own flags")
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    out = Path(os.environ.get("OUT", "out")) / "forge-after"
    out.mkdir(parents=True, exist_ok=True)
    dd = a.work / "data" / a.tag
    inner = sorted((dd / "inner").glob("s*"), key=lambda p: int(p.name[1:]))[-1]
    (n,) = [p.name[4:-4] for p in (dd / "stage").glob("pub-*.bin")]
    inner_flags = ["--rec-inner-circuit", str(dd / "stage/circuit.txt"), "--rec-inner-public", str(dd / f"stage/pub-{n}.bin"),
                   "--rec-inner-partition", str(dd / "stage/partition.json"), "--rec-inner-statement",
                   "verity/flock-circuit+plain-leaves", "--rec-inner-statement-coins", "os", "--rec-inner-draw",
                   str(dd / "stage/draw.json"), "--rec-inner-session", str(inner / "session"), "--rec-inner-transcript",
                   str(inner / "public.json")]
    sess = {s: dd / "v-sess" / s for s in ("L0", "alg-p0")}
    stmts = {s: Path((dd / "v-sess" / f"{s}.stmt").read_text().strip()) for s in sess}
    before = {s: tree_sha(p) for s, p in {**{f"sess-{k}": v for k, v in sess.items()}, **{f"stmt-{k}": v for k, v in stmts.items()}}.items()}
    rec = {"work": str(a.work), "tag": a.tag, "inner": str(inner), "sessions_sha256": before, "forgeries": {}, "verify": {}}
    for kind, (s, port) in CASES.items():
        rec["forgeries"][kind] = forge(stmts[s], inner, port, out / "cases" / kind)
        print(kind, json.dumps(rec["forgeries"][kind]))
    jobs = []
    for spec in a.bin:
        name, path = spec.split("=", 1)
        binary = Path(path)
        rec.setdefault("bins", {})[name] = hashlib.sha256(binary.read_bytes()).hexdigest()
        if name == "run":
            own = {s: (a.run_out / "lean" / f"own-{s}.txt").read_text().split() for s in stmts}
        else:
            own = own_flags(binary, dd, n, stmts, out / f"own-{name}")
        for kind, (s, _) in CASES.items():
            for case in ("honest", "bc", "value"):
                d = out / "cases" / kind / case
                if d.is_dir():
                    jobs.append((f"{name}/{kind}-{case}", binary, d, own[s], sess[s], out / f"{name}-{kind}-{case}"))
    with ThreadPoolExecutor(a.jobs) as ex:
        res = dict(zip([j[0] for j in jobs], ex.map(lambda j: verify(j[1], j[2], j[3], inner_flags, j[4], j[5]), jobs)))
    rec["verify"] = res
    after = {s: tree_sha(p) for s, p in {**{f"sess-{k}": v for k, v in sess.items()}, **{f"stmt-{k}": v for k, v in stmts.items()}}.items()}
    rec["untouched"] = after == before
    bad = [k for k, r in res.items() if (r["rc"] == 0) != k.endswith("-honest")]
    rec["verdict"] = "ok" if not bad and rec["untouched"] else f"unexpected: {bad}"
    (out.parent / "forge-after.json").write_text(json.dumps(rec, indent=1))
    for k, r in res.items():
        print(f"{k}: exit {r['rc']}: {r['why']}")
    print("inputs untouched:", rec["untouched"], "| verdict:", rec["verdict"])
    return 0 if rec["verdict"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
