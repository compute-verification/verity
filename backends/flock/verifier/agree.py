"""The agreement check (PROTOCOL.md §17.1): the Lean verifier and upstream's Rust verifier on the same sessions.

    python agree.py --lean BIN --upstream BIN [--statement ID] --circuit F [--comp F] --public F \
        [--tables DIR] [--expect FILE]... --out DIR --sessions DIR...

`--upstream` is PR #83's binary built with the vectors patch (`netlist-vectors.patch` at 9294e161, `flock-netlist`;
`circuit-vectors.patch` at 19c7269a, `flock-circuit`): its `replay` subcommand is upstream's offline verdict. Each
`--expect` file is a list of `{"name", "expect": {"accepted", "why"}}` (`redigest.py`'s `mutants.json`); a session's name
is its directory's basename, and `why` is a prefix of the Lean verifier's reason. An expectation may add
`upstream_accepted` for an intended divergence (PROTOCOL.md §17.1: D1–D4): upstream must then return that verdict instead
of the Lean verifier's. A session directory may carry its own
public file (`pub.bin`, a forgery's false statement) and its expected verdict (`case.json`, from the vectors patch's
`forgeries` subcommand). The Lean verifier loads the circuit, public, compression and table files from DIR/archive by
their digests (PROTOCOL.md §16.8); upstream reads the paths. Writes DIR/agreement.tsv, the two
verdict logs and DIR/result.json; exits 1 if the verifiers disagree on any session or either misses an expectation.
"""
import argparse, hashlib, json, os, shutil, subprocess, sys
from pathlib import Path


def groups(a):
    """Sessions by the public file their verifier reads."""
    g = {}
    for s in a.sessions:
        own = Path(s) / "pub.bin"
        g.setdefault(str(own) if own.exists() else a.public, []).append(s)
    return g


def lean_verdicts(a):
    out = {}
    for public, sessions in groups(a).items():
        out.update(lean_run(a, public, sessions))
    return out


def upstream_verdicts(a):
    out = {}
    for public, sessions in groups(a).items():
        out.update(upstream_run(a, public, sessions))
    return out


def archived(a, path):
    """Hold `path` in the run's archive, `OUT/archive/sha512/<hex>`, and return its SHA-512: the Lean verifier loads the
    statement's objects by content (PROTOCOL.md §16.8)."""
    h = hashlib.sha512(Path(path).read_bytes()).hexdigest()
    d = a.out / "archive" / "sha512"
    d.mkdir(parents=True, exist_ok=True)
    if not (d / h).exists():
        try:
            os.link(path, d / h)
        except OSError:
            shutil.copyfile(path, d / h)
    return h


def lean_run(a, public, sessions):
    if a.tables:
        for t in sorted(Path(a.tables).glob("*.u32")):
            archived(a, t)
    cmd = [a.lean, "verify", "--statement", a.statement, "--archive", str(a.out / "archive"), "--circuit", archived(a, a.circuit),
           "--public", archived(a, public)]
    if a.comp:
        cmd += ["--comp", archived(a, a.comp)]
    if a.coins:
        cmd += ["--coins", a.coins]
    p = subprocess.run(cmd + ["--session", *sessions], capture_output=True, text=True)
    with open(a.out / "lean.txt", "a") as f:
        f.write(p.stdout)
    with open(a.out / "lean.err", "a") as f:
        f.write(p.stderr)
    out = {}
    for line in p.stdout.splitlines():
        if line.startswith("VERDICT "):
            v = json.loads(line[8:])
            out[v["session"]] = {"accepted": v["accepted"], "why": v["why"]}
    return out


def upstream_run(a, public, sessions):
    flag = "--netlist" if a.statement == "verity/flock-netlist/v1" else "--circuit"
    tables = ["--tables", a.tables] if a.tables else []
    p = subprocess.run([a.upstream, "replay", "--instances", public, flag, a.circuit, *tables, "--sessions", *sessions],
                       capture_output=True, text=True)
    with open(a.out / "upstream.txt", "a") as f:
        f.write(p.stdout)
    with open(a.out / "upstream.err", "a") as f:
        f.write(p.stderr)
    out = {}
    for line in p.stdout.splitlines():
        if line.startswith("UPSTREAM\t"):
            v = json.loads(line.split("\t", 1)[1])
            out[v["session"]] = v["verdict"]
    return out


def main():
    ap = argparse.ArgumentParser()
    for k in ("--lean", "--upstream", "--circuit", "--public"):
        ap.add_argument(k, required=True)
    ap.add_argument("--comp", default=None, help="the BLAKE3 row leaf's compression circuit (statements before eb90718f)")
    ap.add_argument("--statement", default="verity/flock-circuit")
    ap.add_argument("--coins", default=None)
    ap.add_argument("--tables", default=None, help="the pinned MUFU tables, for circuits with lookup slots")
    ap.add_argument("--expect", action="append", default=[])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--sessions", nargs="+", required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    for f in ("lean.txt", "lean.err", "upstream.txt", "upstream.err"):
        (a.out / f).write_text("")
    expect = {e["name"]: e["expect"] for f in a.expect for e in json.loads(Path(f).read_text())}
    for s in a.sessions:
        case = Path(s) / "case.json"
        if case.exists():
            expect.setdefault(Path(s).name, {"accepted": json.loads(case.read_text())["expect"] == "accept", "why": ""})
    lean, up = lean_verdicts(a), upstream_verdicts(a)
    rows, bad = ["session\texpect\tlean\tupstream\tok\tlean_why\tupstream_why"], []
    for s in a.sessions:
        name = Path(s).name
        L, U, e = lean.get(s), up.get(s), expect.get(name)
        up_want = e.get("upstream_accepted", L["accepted"] if L else None) if e else (L["accepted"] if L else None)
        ok = L is not None and U is not None and U["accepted"] == up_want
        if e is not None:
            ok = ok and L["accepted"] == e["accepted"] and L["why"].startswith(e["why"])
        if not ok:
            bad.append(name)
        av = lambda v: "-" if v is None else "A" if v["accepted"] else "R"
        rows.append("\t".join([name, av(e), av(L), av(U), "yes" if ok else "NO", (L or {}).get("why", ""), (U or {}).get("why", "")]))
    (a.out / "agreement.tsv").write_text("\n".join(rows) + "\n")
    result = {"statement": a.statement, "sessions": len(a.sessions), "agree": len(a.sessions) - len(bad), "disagree": bad,
              "accepted": sum(1 for s in a.sessions if (lean.get(s) or {}).get("accepted"))}
    (a.out / "result.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
