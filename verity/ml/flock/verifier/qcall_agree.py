"""Conformance of the Lean `Q_call` v1 and input decoding (`flock-verify qcall-program`, `input-ids`; `Flock/Qcall.lean`,
`Flock/Program.lean`) with core's (`verity.ml.circuits.cut`'s "Q_call v1", `verity.ml.circuits.partition_object`, `verity.ml.circuits.program`).

    python qcall_agree.py --lean BIN --out DIR --qcall qcall_vectors.json --format format_vectors.json

- `--qcall` (PROTOCOL.md §11): every evaluation, field for field (units, owners, committed sets, each Call's gates, `locate`, the
  verdict), every verify refusal (the program's own object at X, or another program's), and every order refusal; and which
  units a Flock statement over a template may claim (`--template --units`), from core's `locate` paths.
- `--format` (PROTOCOL.md §5): `binding_ids` (each id's binding time and width, or none) and `registered_program` (its digest,
  input nodes with their binding times, and Calls).
- Live: a program with a registered input (bound at registration) and run inputs, built here, evaluated by both at several X,
  and its `Q_word` v1 units (`qword-program`) against core's: registering an input changes no cut.
Writes DIR/result.json; exits 1 unless everything agrees.
"""
import argparse, importlib, json, subprocess, sys
from pathlib import Path


def run(a, cmd, tag, *args):
    p = subprocess.run([a.lean, cmd, *map(str, args)], capture_output=True, text=True)
    line = next((l for l in p.stdout.splitlines() if l.startswith(tag + " ")), None)
    if line is None:
        raise RuntimeError(p.stdout + p.stderr)
    return json.loads(line.split(" ", 1)[1])


def qcall(a, path, *extra):
    return run(a, "qcall-program", "QCALLPROGRAM", path, *extra)


def check_members(a, path, desc, ev, bad):
    """`--template T --units`: per template, its whole activations are members, a unit of another Definition or a batch or scan's
    members taken together (`[k]` of n > 1) is not, nor an index past the population; expected from core's `locate` paths."""
    d = json.loads(desc)
    keys = d["callees"]
    whole, other = {}, {}
    for i, (p, _k, _u), steps, _span in ev["locate"]:
        key, members = ev["calls"][p]["definition"], 1
        for step in steps:
            node = d["definitions"][key]["body"]["nodes"][step[0]]
            members = node.get("n", 1) if len(step) == 1 and node["form"] in ("batch", "scan") else 1
            key = keys[node["fn"]]
        (whole if members == 1 else other).setdefault(key, []).append(i)
    n = 0
    for t in sorted(set(whole) | set(other)):
        cases = [(whole.get(t, []), True)] if t in whole else []
        cases += [([i], False) for i in other.get(t, [])[:1]]
        cases += [([i], False) for u, xs in sorted(whole.items()) if u != t for i in xs[:1]][:1]
        cases += [([len(ev["locate"])], False)]
        for units, want in cases:
            got = qcall(a, path, ev["X"], "--template", t, "--units", ",".join(map(str, units)))
            n += 1
            if got.get("members") is not want:
                bad.append(("members", t, units, want, got.get("note") or got.get("error")))
    return n


def check_qcall(a, out, bad):
    v = json.loads(Path(a.qcall).read_text())
    for name, desc in v["descriptors"].items():
        (out / f"{name}.desc").write_text(desc)
    n = 0
    for name, evs in v["evaluations"]:
        for i, want in enumerate(evs):
            got = qcall(a, out / f"{name}.desc", want["X"], *(["--full"] if i == 0 else []))
            n += 1
            if not got["ok"]:
                bad.append(("evaluation", name, want["X"], got["error"]))
                continue
            for k, x in want.items():
                if got["evaluation"].get(k) != x:
                    bad.append(("evaluation", name, want["X"], k))
            if "locate" in want:
                n += check_members(a, out / f"{name}.desc", v["descriptors"][name], want, bad)
    for label, name, X, want in v["verify_refusals"]:
        extra = ["--program-digest", "f" * 128] if label == "another program" else []
        got = qcall(a, out / f"{name}.desc", X, *extra)
        n += 1
        if not got["ok"] or got["evaluation"]["verify"] != want:
            bad.append(("verify refusal", label, got.get("evaluation", {}).get("verify", got.get("error"))))
    for label, name, key, node, arg, refs, want in v["order_refusals"]:
        d = json.loads(v["descriptors"][name])
        d["definitions"][key or d["root"]]["body"]["nodes"][node]["args"][arg]["refs"] = refs
        path = out / f"order-{n}.desc"
        path.write_text(json.dumps(d, separators=(",", ":"), sort_keys=True))
        got = qcall(a, path)
        n += 1
        vv = got.get("evaluation", {}).get("verify", {})
        if not got["ok"] or vv.get("codes") != want["codes"] or vv.get("inapplicable") != want["detail"]:
            bad.append(("order refusal", label, vv or got.get("error")))
    return n


def check_format(a, out, bad):
    f = json.loads(Path(a.format).read_text())
    ids = [i for i, _b in f["binding_ids"]]
    got = run(a, "input-ids", "INPUTID", *ids)
    if got != f["binding_ids"]:
        bad.append(("binding_ids", got))
    reg = f["registered_program"]
    (out / "registered.desc").write_text(reg["descriptor"])
    got = qcall(a, out / "registered.desc")
    want_nodes = [[k, fid, {"Input": "run", "RegisteredInput": "registration"}[fid.rsplit("Input", 1)[0] + "Input"], n]
                  for k, fid, n in reg["input_nodes"]]
    if got.get("program_sha512") != reg["program_digest"] or got.get("input_nodes") != want_nodes \
            or got.get("calls") != [k for k, _fid in reg["calls"]]:
        bad.append(("registered_program", got.get("error") or {k: got.get(k) for k in ("program_sha512", "input_nodes", "calls")}))
    return len(ids) + 1


def check_live(a, out, bad):
    """A program with a registered input: Q_call at several X field for field, and Q_word's units, against core."""
    sys.path.insert(0, str(Path(a.qcall).parent))
    spec = importlib.import_module("test_qcall_vectors")
    from verity.ml.circuits import codec, partition_object as PO
    from verity.ml.circuits.defs import CompositeDefinition, bind
    from verity.ml.circuits.program import Program
    from verity.ml.circuits.refs import tuple_of
    from verity.ml.circuits.types import Array, Tuple, Value
    from verity.ml.circuits.ops.scalar import I32Add

    I32 = Value(32)
    by = bind(CompositeDefinition("QcScaleBy", 1, (), lambda S: ((("x", I32), ("t", I32), ("eps", I32)), I32),
                                  lambda B, S, x, t, eps: B.call(I32Add, B.call(I32Add, x, t), eps), register=False))

    def body(B, S, x, eps, temp, top_p, positions):
        a1 = B.call(by, x, temp, eps)
        return tuple_of(B.call(by, a1, positions[1], eps), B.call(I32Add, x, top_p))

    top = CompositeDefinition("QcTopInputs", 1, (), lambda S: ((("x", I32), ("eps", I32), ("temp", I32), ("top_p", I32),
                              ("positions", Array(3, I32))), Tuple(I32, I32)), body, register=False)
    n = 0
    for registered in [(), ("eps",), ("eps", "top_p")]:
        prog = Program(top, registered=registered)
        desc = codec.encode_program(prog)
        path = out / f"live-{len(registered)}.desc"
        path.write_bytes(codec.canonical_json(desc))
        D = codec.decode_program(json.loads(codec.canonical_json(desc)), spec.registry())
        for X in (32, 64):
            want = spec.evaluation(D, X)
            got = qcall(a, path, X, "--full")
            n += 1
            if not got["ok"] or got["evaluation"] != want:
                bad.append(("live", registered, X, got.get("error") or [k for k in want if got["evaluation"].get(k) != want[k]]))
            elif "locate" in want:
                n += check_members(a, path, path.read_text(), want, bad)
        w = run(a, "qword-program", "QWORDPROGRAM", path)
        part = PO.evaluate(D)
        units = [[k, c.fn.id, c.units] for k, c in part.calls]
        if not w.get("ok") or w.get("population") != part.population or [[k, d, u] for k, d, u in w.get("calls", [])] != units:
            bad.append(("live Q_word", registered, w.get("error") or w.get("calls")))
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lean", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--qcall", required=True)
    ap.add_argument("--format")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    bad, res = [], {}
    res["qcall"] = check_qcall(a, out, bad)
    if a.format:
        res["format"] = check_format(a, out, bad)
    res["live"] = check_live(a, out, bad)
    res["disagree"] = bad[:50]
    (out / "result.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
