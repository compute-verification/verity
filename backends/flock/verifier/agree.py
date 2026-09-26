"""The agreement check (PROTOCOL.md §17.1): the Lean verifier and upstream's Rust verifier on the same sessions.

    python agree.py --lean BIN --upstream BIN [--statement ID] --circuit F --comp F --public F \
        [--expect FILE]... --out DIR --sessions DIR...

`--upstream` is PR #83's binary built with the vectors patch (`netlist-vectors.patch` at 9294e161, `flock-netlist`;
`circuit-vectors.patch` at 19c7269a, `flock-circuit`): its `replay` subcommand is upstream's offline verdict. Each
`--expect` file is a list of `{"name", "expect": {"accepted", "why"}}` (`redigest.py`'s `mutants.json`); a session's name
is its directory's basename, and `why` is a prefix of the Lean verifier's reason. Writes DIR/agreement.tsv, the two
verdict logs and DIR/result.json; exits 1 if the verifiers disagree on any session or either misses an expectation.
"""
import argparse, json, subprocess, sys
from pathlib import Path


def lean_verdicts(a):
    cmd = [a.lean, "verify", "--statement", a.statement, "--circuit", a.circuit, "--comp", a.comp, "--public", a.public]
    if a.coins:
        cmd += ["--coins", a.coins]
    p = subprocess.run(cmd + ["--session", *a.sessions], capture_output=True, text=True)
    (a.out / "lean.txt").write_text(p.stdout)
    (a.out / "lean.err").write_text(p.stderr)
    out = {}
    for line in p.stdout.splitlines():
        if line.startswith("VERDICT "):
            v = json.loads(line[8:])
            out[v["session"]] = {"accepted": v["accepted"], "why": v["why"]}
    return out


def upstream_verdicts(a):
    flag = "--netlist" if a.statement == "verity/flock-netlist/v1" else "--circuit"
    p = subprocess.run([a.upstream, "replay", "--instances", a.public, flag, a.circuit, "--sessions", *a.sessions],
                       capture_output=True, text=True)
    (a.out / "upstream.txt").write_text(p.stdout)
    (a.out / "upstream.err").write_text(p.stderr)
    out = {}
    for line in p.stdout.splitlines():
        if line.startswith("UPSTREAM\t"):
            v = json.loads(line.split("\t", 1)[1])
            out[v["session"]] = v["verdict"]
    return out


def main():
    ap = argparse.ArgumentParser()
    for k in ("--lean", "--upstream", "--circuit", "--comp", "--public"):
        ap.add_argument(k, required=True)
    ap.add_argument("--statement", default="verity/flock-circuit")
    ap.add_argument("--coins", default=None)
    ap.add_argument("--expect", action="append", default=[])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--sessions", nargs="+", required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    expect = {e["name"]: e["expect"] for f in a.expect for e in json.loads(Path(f).read_text())}
    lean, up = lean_verdicts(a), upstream_verdicts(a)
    rows, bad = ["session\texpect\tlean\tupstream\tok\tlean_why\tupstream_why"], []
    for s in a.sessions:
        name = Path(s).name
        L, U, e = lean.get(s), up.get(s), expect.get(name)
        ok = L is not None and U is not None and L["accepted"] == U["accepted"]
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
