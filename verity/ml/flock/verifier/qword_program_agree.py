"""Conformance of the Lean verifier's `Q_word` v1 over programs (`flock-verify qword-program`: `Flock.Program` decodes
#120's `verity-ir/descriptor/v1`, `Flock.Extract` builds each Call's graph, `Flock.Qword` cuts it) with core's
(`verity.ml.circuits.codec.decode_program`, `verity.ml.circuits.cut`, `verity.ml.circuits.partition_object`).

    python qword_program_agree.py --lean BIN --out DIR [--vectors partition_vectors.json] [--format format_vectors.json]
        [--widths CASES.jsonl] [--program DESCRIPTOR]... [--registry MODULE]

- `--vectors` (#111): the program's SHA-512, the population, every Call's units, owners, derived committed set and graph
  (every field core pins), and every unit's `locate`.
- `--format` (#120): the wide-gate Call's graph, the forward reference Q_word refuses (word for word), and every decoder
  refusal of `test_format_spec.REJECT` (the Lean decoder must refuse each; it holds no registry, so an unknown primitive is
  refused by the query instead).
- `--qword-vectors` (#111): every evaluation, the order refusals, the verify refusals (the query's, and the cuts'), and
  every descriptor's cuts at several (X, W) against core's `verify`.
- `--widths` (PROTOCOL.md §17.1 D22): descriptors whose references name leaves of other widths than their types
  give them, and their neighbours, each with the codec's refusal: the Lean decoder refuses each `E_REINTERPRET` the codec
  does, word for word, and no other.
- `--program` (a real descriptor, e.g. a #101 Build's): per Call its units; per Definition a Call names, its units, the
  SHA-512 of its owners and, for a flat cut, of its graph. `--registry` names the module that registers its primitives.
  With `--cuts`, per Definition its cut on its Call's whole graph against core's `check_cut` (`--core-skip PREFIX`: one core
  can't hold here, checked by Lean only).
Writes DIR/result.json; exits 1 unless everything agrees.
"""
import argparse, gc, gzip, hashlib, importlib, json, subprocess, sys
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
    from verity.ml.circuits.codec import canonical_json
    return hashlib.sha512(canonical_json(x)).hexdigest()


def check_vectors(a, out, bad):
    v = json.loads(Path(a.vectors).read_text())
    (out / "pv.desc").write_text(v["descriptor"])
    got = lean(a, out / "pv.desc")
    if not got["ok"]:
        return bad.append(("vectors", got["error"]))
    if got["program_sha512"] != v["program_digest"] or got["population"] != v["population"]:
        bad.append(("vectors", "program_digest or population"))
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
    fr = f["forward_reference"]
    d = json.loads(f["descriptor"])
    d["definitions"]["TPair_v1"]["body"]["nodes"][0]["args"][0]["refs"] = {"affine": ["n", 1, 0, 1]}
    (out / "forward.desc").write_text(json.dumps(d))
    got = lean(a, out / "forward.desc")
    want = fr.get("call_refused")
    if got["ok"] or (want is not None and got.get("inapplicable") != want) or (want is None and fr["q_word"] not in got["error"]):
        bad.append(("forward_reference", got.get("error")))
    if "root_refused" in fr:
        d = json.loads(f["descriptor"])
        _root_body(d)["nodes"][3]["args"][1]["refs"] = {"affine": ["n", 9, 0, 1]}
        (out / "forward-root.desc").write_text(json.dumps(d))
        got = lean(a, out / "forward-root.desc")
        if got["ok"] or got.get("inapplicable") != fr["root_refused"]:
            bad.append(("forward_reference root", got.get("inapplicable")))
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


def check_widths(a, out, bad):
    """Each `--widths` line is `{"program", "codec"}`: a descriptor and the codec's refusal of it (`decode_program`; null
    where it decodes). The Lean decoder refuses every one the codec refuses as `E_REINTERPRET` with the codec's message,
    and none other as `E_REINTERPRET`."""
    both = neither = 0
    for k, line in enumerate(Path(a.widths).read_text().splitlines()):
        case = json.loads(line)
        (out / "widths.desc").write_text(case["program"])
        got = lean(a, out / "widths.desc")
        err = "" if got["ok"] else got.get("error", "")
        why = case["codec"] or ""
        if "E_REINTERPRET" in why:
            if err == why:
                both += 1
            else:
                bad.append(("widths", k, err[:300], why[:300]))
        elif "E_REINTERPRET" in err:
            bad.append(("widths", k, err[:300], why[:300]))
        else:
            neither += 1
    return {"both_refuse": both, "neither": neither}


def _root_body(d):
    return d["definitions"][d["root"]]["body"]


def _text(x):
    return x if isinstance(x, str) else json.dumps(x)


def check_qword_vectors(a, out, bad):
    """`qword_vectors.json` (#111): every evaluation (population, owners digest, every Call's graph, owners, kinds, units and
    committed set), every order refusal (the `order_violation` detail), and the verify refusals that are the query's
    (`query-inapplicable`); the cut refusals are the width rule and the invariant, which `qword-program` does not check."""
    v = json.loads(Path(a.qword_vectors).read_text())
    runs_checked = 0
    for name, runs in v["evaluations"]:
        (out / f"{name}.desc").write_text(_text(v["descriptors"][name]))
        for run in runs:
            got = lean(a, out / f"{name}.desc", str(run["X"]))
            runs_checked += 1
            if not got["ok"]:
                bad.append(("evaluation", name, run["X"], got.get("error")))
                continue
            if got["population"] != run["population"] or got["owners_digest"] != run["owners_digest"]:
                bad.append(("evaluation", name, run["X"], "population or owners_digest"))
            det = {c["call"]: c for c in got["detail"]}
            for c in run["calls"]:
                g = det.get(c["call"], {})
                for k, x in c.items():
                    if k == "graph":
                        for gk, gx in x.items():
                            if g.get("graph", {}).get(gk) != gx:
                                bad.append(("evaluation", name, run["X"], c["call"], "graph." + gk))
                    elif g.get(k) != x:
                        bad.append(("evaluation", name, run["X"], c["call"], k))
    orders = 0
    for label, dname, definition, node, arg, refs, want in v["order_refusals"]:
        d = json.loads(_text(v["descriptors"][dname]))
        body = d["definitions"][definition]["body"] if definition else _root_body(d)
        body["nodes"][node]["args"][arg]["refs"] = refs
        (out / "order.desc").write_text(json.dumps(d))
        got = lean(a, out / "order.desc")
        orders += 1
        if got["ok"] or got.get("inapplicable") != want["detail"][0]:
            bad.append(("order_refusals", label, got.get("inapplicable")))
    inapplicable = 0
    cut_refusals = 0
    for label, dname, X, W, want in v["verify_refusals"]:
        (out / "refused.desc").write_text(_text(v["descriptors"][dname]))
        if want["codes"] == ["cut"]:
            got = lean(a, out / "refused.desc", str(X), str(W), "--cuts")
            cut_refusals += 1
            if got.get("cut") != want["cut"]:
                bad.append(("verify_refusals", label, got.get("cut"), want["cut"]))
            continue
        if want["codes"] != ["query-inapplicable"]:
            continue
        got = lean(a, out / "refused.desc", str(X))
        inapplicable += 1
        if got["ok"] or not got.get("error", "").startswith("query-inapplicable"):
            bad.append(("verify_refusals", label, got.get("error")))
    cuts = check_vector_cuts(a, v, out, bad)
    return {"evaluations": runs_checked, "order_refusals": orders, "inapplicable": inapplicable, "cut_refusals": cut_refusals,
            "cuts": cuts, "v2": check_v2_vectors(a, v, out, bad)}


def check_v2_vectors(a, v, out, bad):
    """`qword_vectors.json` → `v2`: each evaluation at v2 (`--query-version 2`: v1's population, owners digest and Calls), and
    each verify refusal's and acceptance's cut at v2 (`--cuts`): the failing Calls, and the recompute across units reported per
    Call (Lean's per Definition, so summed by Call)."""
    v2 = v["v2"]
    runs = 0
    for name, rs in v2["evaluations"]:
        (out / f"{name}.desc").write_text(_text(v["descriptors"][name]))
        for run in rs:
            got = lean(a, out / f"{name}.desc", str(run["X"]), "--query-version", "2")
            runs += 1
            if not got["ok"] or got["population"] != run["population"] or got["owners_digest"] != run["owners_digest"]:
                bad.append(("v2 evaluation", name, run["X"], got.get("error")))
                continue
            det = {c["call"]: c for c in got["detail"]}
            for c in run["calls"]:
                if any(det.get(c["call"], {}).get(k) != c[k] for k in ("owner", "units", "committed")):
                    bad.append(("v2 evaluation", name, run["X"], c["call"]))
    checked = {"cut": 0, "accepted": 0, "inapplicable": 0}
    for label, dname, X, W, want in v2["verify_refusals"]:
        (out / "v2.desc").write_text(_text(v["descriptors"][dname]))
        if want["codes"] in (["cut"], []):
            got = lean(a, out / "v2.desc", str(X), str(W), "--cuts", "--query-version", "2")
            checked["cut" if want["codes"] else "accepted"] += 1
            if got.get("cut") != want["cut"] or _across_by_call(got) != _report_by_call(want, got):
                bad.append(("v2 verify", label, got.get("cut"), want["cut"], _across_by_call(got)))
        elif want["codes"] == ["query-inapplicable"]:
            got = lean(a, out / "v2.desc", str(X), "--query-version", "2")
            checked["inapplicable"] += 1
            if got["ok"] or not got.get("error", "").startswith("query-inapplicable"):
                bad.append(("v2 verify", label, got.get("error")))
    return {"evaluations": runs, **checked}


def _across_by_call(got):
    """Lean's `recomputed_across` per Call: its Definition's, from `qword-program --cuts --query-version 2`."""
    return {k: got.get("cuts", {}).get(key, {}).get("recomputed_across") for k, key, _u in got.get("calls", [])}


def _report_by_call(core, got):
    """Core's `recomputed_across` per Call (`verify`'s detail lists only Calls with a pair), over the Calls Lean lists."""
    rep = {c["call"]: c["n"] for c in core["recomputed_across"]["gates"]}
    return {k: rep.get(k, 0) for k, _key, _u in got.get("calls", [])}


#: the (X, W) the vectors' programs are cut and checked at, against core's `verify`
CUT_LIMITS = [(16, 32), (16, 16), (32, 32), (8, 16), (64, 64)]


def check_vector_cuts(a, v, out, bad):
    """Every descriptor of `qword_vectors.json` at each of `CUT_LIMITS`: the Calls whose cut fails and, per Definition, its
    codes and too-wide units (`qword-program --cuts`), against core's `partition_object.verify` (its programs decoded with
    the vectors' registry of test primitives, `test_qword_vectors.registry`). None when that module can't be imported."""
    from verity.ml.circuits import codec, partition_object as po
    sys.path.insert(0, str(Path(a.qword_vectors).resolve().parent))
    try:
        import test_qword_vectors as TQ
    except ImportError:
        return None
    reg = TQ.registry()
    checked = 0
    for name, text in v["descriptors"].items():
        (out / f"{name}.desc").write_text(_text(text))
        prog = codec.decode_program(json.loads(_text(text)), reg)
        for (X, W), version in ((xw, ver) for ver in (1, 2) for xw in CUT_LIMITS):
            q = po.query(X, W, version)
            obj = {"format": po.FORMAT, "program": codec.program_digest(prog), "query": q}
            want = po.verify(obj, prog)
            got = lean(a, out / f"{name}.desc", str(X), str(W), "--cuts", *(["--query-version", "2"] if version == 2 else []))
            checked += 1
            if "query-inapplicable" in want.codes:
                if got.get("ok", True) and "error" not in got:
                    bad.append(("cuts", name, X, W, version, "core: query-inapplicable", got.get("cut")))
                continue
            core_cut = [[c["call"], c["codes"]] for c in want.detail.get("cut", {}).get("gates", [])]
            if got.get("cut") != core_cut:
                bad.append(("cuts", name, X, W, version, got.get("cut"), core_cut))
            if version == 2 and _across_by_call(got) != _report_by_call(want.detail, got):
                bad.append(("cuts", name, X, W, version, "recomputed_across", _across_by_call(got), _report_by_call(want.detail, got)))
            def_of = {k: key for k, key, _u in got.get("calls", [])}
            for c in want.detail.get("cut", {}).get("gates", []):
                wide = c["detail"].get("unit-too-wide", {}).get("gates", [])
                r = got.get("cuts", {}).get(def_of.get(c["call"]), {})
                if r.get("wide") != wide:
                    bad.append(("cuts", name, X, W, version, c["call"], "too-wide units", r.get("wide"), wide))
    return checked


def check_program(a, path, out, bad):
    from verity.ml.circuits import partition_object as po
    from verity.ml.circuits.codec import _spec_id, decode_program, program_digest
    from verity.ml.circuits.cut import CallGraph
    from verity.ml.circuits.defs import REGISTRY
    raw = gzip.open(path).read() if str(path).endswith(".gz") else Path(path).read_bytes()
    (out / "program.json").write_bytes(raw)
    got = lean(a, out / "program.json", "16", "--digests")
    if not got["ok"]:
        return bad.append(("program", str(path), got["error"]))
    P = decode_program(json.loads(raw), REGISTRY)
    part = po.evaluate(P, po.query(16, 32))
    if got["program_sha512"] != program_digest(json.loads(raw)) or got["population"] != part.population:
        bad.append(("program", str(path), "program_digest or population"))
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
    cuts = check_program_cuts(a, P, part, out, bad) if a.cuts else None
    return [len(want_calls), len(defs)] + ([cuts] if cuts else [])


def check_program_cuts(a, P, part, out, bad):
    """Per Definition a Call names, its cut against its Call's whole graph at X = 16 and W = 32, 16 (`qword-program --cuts`)
    against core's `cut.check_cut` on `Partition.graph` and `owner`, as `verify` runs it: the codes, the too-wide units, and
    the graph's computing gates, recomputed pairs, derived committed set, widest boundary and boundary gates. Definitions
    named by `--core-skip` (too large for core here) are checked by Lean only, and listed."""
    from verity.ml.circuits.codec import _spec_id
    from verity.ml.circuits.cut import boundary_widths, check_cut, derived_committed
    lean_runs = {W: lean(a, out / "program.json", "16", str(W), "--digests", "--cuts") for W in (32, 16)}
    compared, lean_only, done = [], [], set()
    for p, (_k, c) in enumerate(part.calls):
        key = _spec_id(c.fn)
        if key in done:
            continue
        done.add(key)
        if any(key.startswith(s) for s in a.core_skip):
            lean_only.append({"definition": key, **{f"W={W}": r["cuts"].get(key) for W, r in lean_runs.items()}})
            continue
        g, own = part.graph(p), part.owner(p)
        ob, og = boundary_widths(g, own)
        stats = [g.n, len(g.recomputed), len(derived_committed(g, own)), max(ob, default=0), sum(og)]
        for W, r in lean_runs.items():
            v = check_cut(g, own, None, (16, W, 0))
            want = {"codes": list(v.codes), "wide": v.detail.get("unit-too-wide", {}).get("gates", []), "stats": stats}
            if r["cuts"].get(key) != want:
                bad.append(("program cuts", key, W, r["cuts"].get(key), want))
        compared.append(key)
        del g, own
        part._graphs.clear()
        gc.collect()
    for W, r in lean_runs.items():
        want = [[k, cc["codes"]] for k, cc in ((k, r["cuts"][_spec_id(c.fn)]) for k, c in part.calls) if cc["codes"]]
        if r["cut"] != want:
            bad.append(("program cut", W, r["cut"][:8], want[:8]))
    return {"compared": compared, "lean_only": lean_only,
            "failing_at_W16": sorted({k for k, r in lean_runs[16]["cuts"].items() if r["codes"]})}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lean", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--vectors")
    ap.add_argument("--format")
    ap.add_argument("--qword-vectors")
    ap.add_argument("--widths")
    ap.add_argument("--program", action="append", default=[])
    ap.add_argument("--registry")
    ap.add_argument("--cuts", action="store_true", help="with --program: every Definition's cut, against core's check_cut")
    ap.add_argument("--core-skip", action="append", default=[], metavar="PREFIX",
                    help="with --cuts: a Definition core can't hold in memory here, checked by Lean only")
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
    if a.qword_vectors:
        res["qword_vectors"] = check_qword_vectors(a, out, bad)
    if a.widths:
        res["widths"] = check_widths(a, out, bad)
    for p in a.program:
        res[f"program {Path(p).parent.name}/{Path(p).name}"] = check_program(a, p, out, bad)
    res["disagree"] = bad[:50]
    (out / "result.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
