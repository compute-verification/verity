"""The Lean template-instance queries (`flock-verify template-query`, PROTOCOL.md §16.10) against core's pinned vectors,
`verity/ml/circuits/tests/template_instance_vectors.json` (PR #131): descriptor bytes in, units out, and every refusal.

    python template_query_agree.py --lean BIN --vectors template_instance_vectors.json --out DIR

For each case: the program's SHA-512, the population, the ranges and every unit's root node, member and gate interval; for each
refusal: `query-refused` with core's reason; for each malformed query: its parameters refused. Writes DIR/result.json; exits 1
unless every vector agrees.
"""
import argparse, json, subprocess, sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lean", required=True)
    ap.add_argument("--vectors", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    v = json.loads(a.vectors.read_text())
    a.out.mkdir(parents=True, exist_ok=True)
    files, want = [], {}

    def case(name, desc, q, w):
        f = a.out / f"{name}.json"
        f.write_text(json.dumps({"descriptor": desc, "query": q}))
        files.append(str(f))
        want[str(f)] = w

    for i, (_label, d, q, exp) in enumerate(v["cases"]):
        case(f"case{i}", v["descriptors"][d], q, ("ok", exp["object"]["program"], exp["population"], exp["ranges"], [u[:5] for u in exp["units"]]))
    for i, (_label, d, q, why) in enumerate(v["refused"]):
        case(f"refused{i}", v["descriptors"][d], q, ("refused", why.split(") ", 1)[1]))
    for i, (_label, q, codes) in enumerate(v["malformed"]):
        case(f"malformed{i}", v["descriptors"]["population"], q, ("malformed", codes))
    p = subprocess.run([a.lean, "template-query", *files], capture_output=True, text=True)
    got = {}
    for line in p.stdout.splitlines():
        if line.startswith("TEMPLATEQUERY "):
            j = json.loads(line[len("TEMPLATEQUERY "):])
            got[j["file"]] = j
    bad = []
    for f in files:
        g, w = got.get(f) or {}, want[f]
        if w[0] == "ok":
            ok = g.get("ok") and (g["program_sha512"], g["population"], g["ranges"], g["units"]) == w[1:]
        elif w[0] == "refused":
            ok = not g.get("ok", True) and g.get("error", "").startswith("query-refused: ") and g["error"].split(": ", 1)[1] == w[1]
        else:
            ok = not g.get("ok", True) and any(c in g.get("error", "") for c in w[1])
        if not ok:
            bad.append(f)
    result = {"vectors": str(a.vectors), "cases": len(v["cases"]), "refused": len(v["refused"]), "malformed": len(v["malformed"]),
              "agree": len(files) - len(bad), "disagree": bad}
    (a.out / "result.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
