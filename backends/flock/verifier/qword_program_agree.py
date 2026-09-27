"""Conformance of the Lean verifier's `Q_word` v1 over programs (`flock-verify qword-program`: `Flock.Program` decodes
#120's `verity-ir/descriptor/v1`, `Flock.Extract` builds each Call's graph, `Flock.Qword` cuts it) with core's
(`verity.ir.codec.decode_program`, `verity.ir.cut`, `verity.ir.partition_object`).

    python qword_program_agree.py --lean BIN --out DIR [--vectors partition_vectors.json] [--format format_vectors.json]
        [--program DESCRIPTOR]... [--registry MODULE]

- `--vectors` (#111): the program's SHA-512, the population, every Call's units, owners, derived committed set and graph
  (every field core pins), and every unit's `locate`.
- `--format` (#120): the wide-gate Call's graph, the forward reference Q_word refuses (word for word), and every decoder
  refusal of `test_format_spec.REJECT` (the Lean decoder must refuse each; it holds no registry, so an unknown primitive is
  refused by the query instead).
- `--program` (a real descriptor, e.g. a #101 Build's): per Call its units; per Definition a Call names, its units, the
  SHA-512 of its owners and, for a flat cut, of its graph. `--registry` names the module that registers its primitives.
Writes DIR/result.json; exits 1 unless everything agrees.
"""
import argparse, gzip, hashlib, importlib, json, subprocess, sys
from pathlib import Path


def lean(a, path, *extra):
    p = subprocess.run([a.lean, "qword-program", str(path), *extra], capture_output=True, text=True)
    line = next((l for l in p.stdout.splitlines() if l.startswith("QWORDPROGRAM ")), None)
    if line is None:
        raise RuntimeError(p.stdout + p.stderr)
    return json.loads(line.split(" ", 1)[1])


def graph_dict(g):
    return {"total": g.total, "structure": g.free, "n": g.n, "keep": g.keep.tolist(), "width": g.width.tolist(),
            "prim": [g.prim_names[x] for x in g.prim.tolist()], "external": list(g.external), "input_ids": list(g.input_ids),
            "src": g.src.tolist(), "dst": g.dst.tolist(), "input_reads": [list(x) for x in g.input_reads],
            "outputs": list(g.outputs), "returned_inputs": list(g.returned_inputs), "recomputed": [list(x) for x in g.recomputed]}


def sha512(x):
    from verity.ir.codec import canonical_json
    return hashlib.sha512(canonical_json(x)).hexdigest()


def check_vectors(a, out, bad):
    v = json.loads(Path(a.vectors).read_text())
    (out / "pv.desc").write_text(v["descriptor"])
    got = lean(a, out / "pv.desc")
    if not got["ok"]:
        return bad.append(("vectors", got["error"]))
    if got["program_sha512"] != v["program_sha512"] or got["population"] != v["population"]:
        bad.append(("vectors", "program_sha512 or population"))
    det = {c["call"]: c for c in got["detail"]}
    for c in v["calls"]:
        g = det.get(c["call"], {})
        for k in ("units", "owner", "committed"):
            if g.get(k) != c[k]:
                bad.append(("vectors", c["call"], k))
        for k, x in c["graph"].items():
            if g.get("graph", {}).get(k) != x:
                bad.append(("vectors", c["call"], "graph." + k))
    if got["locate"] != v["locate"]:
        bad.append(("vectors", "locate"))
    return len(v["calls"])


def check_format(a, out, bad):
    f = json.loads(Path(a.format).read_text())
    sys.path.insert(0, str(Path(a.format).parent))
    spec = importlib.import_module("test_format_spec")
    (out / "wide.desc").write_text(f["wide"]["descriptor"])
    got = lean(a, out / "wide.desc")
    if not got["ok"]:
        bad.append(("wide", got["error"]))
    else:
        g = got["detail"][0]["graph"]
        for k, x in f["wide"]["graph"]["wide"].items():
            if g.get(k) != x:
                bad.append(("wide", "graph." + k, g.get(k), x))
    d = json.loads(f["descriptor"])
    d["definitions"]["TPair_v1"]["body"]["nodes"][0]["args"][0]["refs"] = {"affine": ["n", 1, 0, 1]}
    (out / "forward.desc").write_text(json.dumps(d))
    got = lean(a, out / "forward.desc")
    if got["ok"] or f["forward_reference"]["q_word"] not in got["error"]:
        bad.append(("forward_reference", got.get("error")))
    refused = 0
    for label, edit, _why in spec.REJECT:
        d = json.loads(f["descriptor"])
        edit(d)
        (out / "reject.desc").write_text(json.dumps(d, ensure_ascii=False))
        got = lean(a, out / "reject.desc")
        if got["ok"]:
            bad.append(("decoder_rejects", label))
        else:
            refused += 1
    return refused, len(spec.REJECT)


def check_program(a, path, out, bad):
    from verity.ir import partition_object as po
    from verity.ir.codec import _spec_id, decode_program
    from verity.ir.cut import CallGraph
    from verity.ir.defs import REGISTRY
    raw = gzip.open(path).read() if str(path).endswith(".gz") else Path(path).read_bytes()
    (out / "program.json").write_bytes(raw)
    got = lean(a, out / "program.json", "16", "--digests")
    if not got["ok"]:
        return bad.append(("program", str(path), got["error"]))
    P = decode_program(json.loads(raw), REGISTRY)
    part = po.evaluate(P, po.query(16, 32))
    if got["program_sha512"] != po.program_sha512(json.loads(raw)) or got["population"] != part.population:
        bad.append(("program", str(path), "program_sha512 or population"))
    want_calls = [[k, _spec_id(c.fn), c.units] for k, c in part.calls]
    if got["calls"] != want_calls:
        bad.append(("program", str(path), "calls"))
    defs = {}
    for _k, c in part.calls:
        key = _spec_id(c.fn)
        if key in defs:
            continue
        e = {"units": c.units, "owner_sha512": sha512(c.owner())}
        if c.parts is None and hasattr(c.fn, "body"):
            e["graph_sha512"] = sha512(graph_dict(CallGraph(c.fn)))
        defs[key] = e
    for key, e in defs.items():
        if got["definitions"].get(key) != e:
            bad.append(("program", str(path), key, got["definitions"].get(key), e))
    return len(want_calls), len(defs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lean", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--vectors")
    ap.add_argument("--format")
    ap.add_argument("--program", action="append", default=[])
    ap.add_argument("--registry")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.registry:
        importlib.import_module(a.registry)
    bad, res = [], {}
    if a.vectors:
        res["vectors_calls"] = check_vectors(a, out, bad)
    if a.format:
        res["decoder_rejects_refused"] = check_format(a, out, bad)
    for p in a.program:
        res[f"program {Path(p).parent.name}/{Path(p).name}"] = check_program(a, p, out, bad)
    res["disagree"] = bad[:50]
    (out / "result.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
