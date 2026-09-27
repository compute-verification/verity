"""One-stage audit, run A4: #101's whole layer 0 on serving's own roots, every member template in one audit, one
registration, one draw and one integrity profile, with the Lean verifier deriving the units from the verifier's own program.

    PYTHONPATH=<tree>/packages/verity/src:<tree>/backends/numerical/python:<tree>/integrations/vllm:<tree>/protocols/one_stage \\
      python benchmarks/one_stage/a4.py --served DIR --template-sets DIR --layout P6|P4 --m0-pythonpath PATH \\
      --bin flock-circuit --lean flock-verify --out DIR [--k 1024] [--delta 2^-20]

* **The layout of record** is `a4_layout.json` (the verifier's own copy of vllm-serving-commit handoff 20260927T0905Z):
  members in order with their counts, and the GEMM members' grid row maps. The verifier builds `TemplatePopulations` over
  the members' Definitions (from `--template-sets`, one captured set per member) and the canonical `Q_template_instances`
  partition, and refuses unless its digest is the layout's.
* **`--served DIR`** holds `registration.json`, `index.json` and per member k `m<k>-pub-<n>.bin` and `m<k>-inst-<n>.bin`
  (prover only). For each member the verifier composes the bound circuit itself and holds the file's pin, bindings, global
  unit indices and served domains to it; then R1-R6 over one registration.
* **GEMM members** use M0's shared-row files (flock-netlist handoff 20260927T0905Z): header `shared_rows` gives each input
  port's table size, and the public file lists every instance's row references. The references must be exactly what the
  layout's grid rule gives (`gemm-grid/v0`: instance i of group g, j = i - base_g, reads `x` row `x_base + j div columns`
  and `w` row `w_base + j mod columns`); the verifier checks all of them against its own copy of the rule before the draw.
* **`--session-m0-pythonpath`**: after every check on serving's files as received, the sessions run on the verifier's own copies
  re-headered for the circuit that M0 tree composes (e.g. `program_digests` keyed by descriptor id): the leaf layer is byte
  for byte serving's, and only the header's circuit pin (and each class, unchanged since the class leaves the bindings out)
  follows the verifier's circuit.
* **One draw** of `subset:k` over the whole population by the verifier's Lean `flock-verify draw`; each member's server
  serves its share (`serve --draw-file`). Each member's session is one table; Lean verifies each with `--partition` and
  `--program` (units derived).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import a0  # noqa: E402
from verity.ir.codec import canonical_json, encode_program  # noqa: E402
from verity_numerical.bench.input_sets import InputSet  # noqa: E402
from verity_one_stage import audit, draw as D, partition as P, registration as R  # noqa: E402

ROW = "hm96-sha512/row/v1"


def member_ports(meta: dict, n: int, row_map: dict | None) -> dict[str, tuple[str, int]]:
    """Each port the member's layout derives: its schema and leaf count. A GEMM member's row ports are tables whose sizes
    the grid rule fixes."""
    if not row_map:
        return a0.ports(meta, n)
    groups = row_map["groups"]
    rows = {"x": max(g["x_base"] + g["tokens"] for g in groups), "w": max(g["w_base"] + g["columns"] for g in groups)}   # the table sizes
    out = {name: (ROW, rows[name]) for name, *_ in meta["ports"]}
    out.update({name: ("u16", n * words) for name, words in meta["out_ports"]})
    return out


def grid_refs(row_map: dict, n: int):
    """Every instance's (x row, w row) under the grid rule, as an n x 2 uint32 array."""
    import numpy as np
    out = np.empty((n, 2), dtype=np.uint32)
    at = 0
    for g in row_map["groups"]:
        size = g["tokens"] * g["columns"]
        j = np.arange(size, dtype=np.int64)
        out[at:at + size, 0] = g["x_base"] + j // g["columns"]
        out[at:at + size, 1] = g["w_base"] + j % g["columns"]
        at += size
    if at != n:
        raise ValueError(f"the grid covers {at} instances, the member has {n}")
    return out


def check_shared_rows(pub: Path, header: dict, meta: dict, n: int, row_map: dict) -> dict:
    """M0's shared-row file against the grid rule: `shared_rows` sizes, and every instance's references."""
    import numpy as np
    want = grid_refs(row_map, n)
    sizes = {"x": int(want[:, 0].max()) + 1, "w": int(want[:, 1].max()) + 1}
    ports = [name for name, *_ in meta["ports"]]
    ok = {"shared_rows": header.get("shared_rows") == {p: sizes[p] for p in ports}}
    with open(pub, "rb") as f:
        f.readline()
        f.seek(sum(sizes[p] for p in ports) * 128, 1)
        refs = np.frombuffer(f.read(n * len(ports) * 4), dtype="<u4").reshape(n, len(ports))
    ok["refs"] = bool(ok["shared_rows"]) and refs.shape == (n, len(ports)) and bool(
        (refs == want[:, [("x", "w").index(p) for p in ports]]).all())
    return ok


def rekey(m: dict, pd: str, part: dict, a, out: Path) -> dict:
    """The verifier's session copies of member m's files for the circuit `--session-m0-pythonpath` composes."""
    sd = out / f"m{m['k']}-session"
    sd.mkdir(parents=True, exist_ok=True)
    p = a0.run([sys.executable, str(HERE / "compose_circuit.py"), "--input-set", str(m["set"]), "--out", str(sd / "circuit.txt"),
                "--partition", pd, "--program", part["program"]], sd / "compose.log", {"PYTHONPATH": a.session_m0_pythonpath})
    made = json.loads(p.stdout.strip().splitlines()[-1])
    for kind in ("pub", "inst"):
        with open(m["st"]["dir"] / f"{kind}-{m['n']}.bin", "rb") as f:
            head = json.loads(f.readline())
            body = f.read()
        head["circuit_sha512"] = made["pin"]
        if made.get("class_sha512"):
            head["units"]["classes"] = [made["class_sha512"]] * m["n"]
        with open(sd / f"{kind}-{m['n']}.bin", "wb") as f:
            f.write(json.dumps(head, sort_keys=True).encode() + b"\n" + body)
    text = (sd / "circuit.txt").read_text()
    return {"dir": sd, "meta": json.loads(text.split("\n", 2)[1][5:]), "pin": made["pin"], "partition_file": m["st"]["partition_file"],
            "program_file": m["st"]["program_file"], "pin_as_served": m["st"]["pin"], "class_unchanged":
            made.get("class_sha512") == m["header"]["units"].get("classes", [None])[0]}


def expected_draws(members: list[dict], n_all: int, k: int) -> dict[str, dict]:
    """Per member under the uniform k-subset: the expected number of drawn units, k n / N, and the chance of at least one,
    1 - C(N - n, k) / C(N, k), so a small template's rare draws are visible in the report."""
    from math import lgamma, exp

    def log_comb(a: int, b: int) -> float:
        return lgamma(a + 1) - lgamma(b + 1) - lgamma(a - b + 1)
    return {m["tid"]: {"n": m["n"], "expected": k * m["n"] / n_all,
                       "p_at_least_one": 1.0 - (exp(log_comb(n_all - m["n"], k) - log_comb(n_all, k)) if n_all - m["n"] >= k else 0.0)}
            for m in members}


def link(src: Path, dst: Path) -> None:
    if not dst.exists():
        try:
            os.link(src.resolve(), dst)
        except OSError:
            shutil.copyfile(src, dst)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--served", required=True)
    ap.add_argument("--template-sets", required=True)
    ap.add_argument("--layout", default="P6")
    ap.add_argument("--layout-file", default=str(HERE / "a4_layout.json"))
    ap.add_argument("--m0-pythonpath", required=True)
    ap.add_argument("--bin", required=True)
    ap.add_argument("--lean", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=1024)
    ap.add_argument("--delta", type=float, default=2.0 ** -20)
    ap.add_argument("--port", type=int, default=7541)
    ap.add_argument("--session-m0-pythonpath", default=None,
                    help="run the sessions on copies re-headered for the circuit this M0 tree composes (after the checks)")
    ap.add_argument("--statement", default=a0.STATEMENT_ID,
                    help="the Lean verifier's statement name (verity/flock-circuit@967b8d06 for files staged at M0 68ae79f2 or later)")
    a = ap.parse_args()
    a0.STATEMENT_ID = a.statement
    out, served, sets_root = Path(a.out), Path(a.served), Path(a.template_sets)
    out.mkdir(parents=True, exist_ok=True)
    lay = json.loads(Path(a.layout_file).read_text())[a.layout]

    # the verifier's own program and partition, from the layout of record
    members, base = [], 0
    for k, mm in enumerate(lay["members"]):
        d = next(x for x in sorted(sets_root.iterdir()) if x.name.startswith(mm["set"]) and (x / "manifest.json").exists())
        defn = InputSet.open(str(d)).subcircuit.definition()
        members.append({"k": k, "set": d, "defn": defn, "tid": P.template_id(defn), "n": int(mm["count"]), "base": base,
                        "row_map": mm.get("row_map")})
        base += int(mm["count"])
    program = P.mixed_population_program([(m["defn"], m["n"]) for m in members])
    query = P.template_instances_query([m["defn"] for m in members])
    part = P.partition(program, query)
    pd = P.digest(part)
    n_all = P.units(program, query)
    summary: dict = {"layout": a.layout, "population": n_all, "partition": pd, "k": a.k, "delta": a.delta, "statement": a.statement,
                     "members": [{"k": m["k"], "template": m["tid"], "n": m["n"], "base": m["base"]} for m in members],
                     "expected_draws": expected_draws(members, n_all, a.k)}
    if pd != lay["digest"]:
        summary["refused"] = {"layout-digest": {"layout": lay["digest"], "verifier": pd}}
        (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True))
        print("A4 " + json.dumps(summary["refused"]))
        return 1
    (out / "partition.json").write_text(json.dumps(part, sort_keys=True))
    (out / "program.json").write_bytes(canonical_json(encode_program(program)))

    rec = json.loads((served / "registration.json").read_text())
    shutil.copy(served / "index.json", out / "index.json")
    run_id = rec.get("window", {}).get("run", "")
    law = D.parse_law(f"subset:{a.k}")

    # every member's served files against the verifier's own circuit, bindings and served domains
    want, publics, checks = {}, {}, {}
    for m in members:
        st_dir = out / f"m{m['k']}"
        st_dir.mkdir(parents=True, exist_ok=True)
        p = a0.run([sys.executable, str(HERE / "compose_circuit.py"), "--input-set", str(m["set"]), "--out", str(st_dir / "circuit.txt"),
                    "--partition", pd, "--program", part["program"]], st_dir / "compose.log", {"PYTHONPATH": a.m0_pythonpath})
        made = json.loads(p.stdout.strip().splitlines()[-1])
        for kind in ("pub", "inst"):
            link(served / f"m{m['k']}-{kind}-{m['n']}.bin", st_dir / f"{kind}-{m['n']}.bin")
        text = (st_dir / "circuit.txt").read_text()
        meta = json.loads(text.split("\n", 2)[1][5:])
        header = a0.public_header(st_dir, m["n"])
        fv = header["frame_v3"]
        m.update(st={"dir": st_dir, "meta": meta, "pin": made["pin"], "partition_file": out / "partition.json",
                     "program_file": out / "program.json"}, header=header)
        ok = {"pin": header.get("circuit_sha512") == made["pin"],
              "partition": meta.get("partition") == {"rule": P.FORMAT, "digest": pd},
              "program": meta.get("program_sha512") == part["program"],
              "indices": header["units"]["indices"] == list(range(m["base"], m["base"] + m["n"])),
              "classes": header["units"].get("classes") == [made.get("class_sha512")] * m["n"],
              "template_program": a0.template_program_ok(meta, m["defn"])}
        if m["row_map"]:
            ok.update(check_shared_rows(st_dir / f"pub-{m['n']}.bin", header, meta, m["n"], m["row_map"]))
        else:
            ok["no_shared_rows"] = "shared_rows" not in header
        for port, (schema, leaves) in member_ports(meta, m["n"], m["row_map"]).items():
            name = f"{m['tid']}/{port}"
            want[name] = {"leaves": leaves, "domain": R.served_domain(part["program"], pd, name, schema, leaves, run_id)}
            reg = next((r for r in rec["roots"] if r["port"] == name), None)
            ok[f"root:{port}"] = reg is not None and reg["root"] == fv["roots"].get(port) and reg["domain"] == fv["domain_ids"].get(port)
        checks[f"m{m['k']}"] = {"ok": all(ok.values()), "codes": [c for c, v in ok.items() if not v], "detail": {}}
        publics[m["tid"]] = a0.sha512_file(st_dir / f"pub-{m['n']}.bin")
    leaf_layer = hashlib.sha512(canonical_json(publics)).hexdigest()
    checks["registration"] = R.check(rec, program, query, law, a0.SCHEME, want, leaf_layer=leaf_layer).to_json()
    checks["source"] = {"ok": rec.get("window", {}).get("kind") == "served", "codes": [], "detail": rec.get("window", {}).get("kind")}
    receipt = R.receipt(rec, "one-stage-e2e verifier (a4.py)")
    (out / "receipt.json").write_text(json.dumps(receipt, indent=1, sort_keys=True))
    summary.update(registration=R.digest(rec), checks_before_draw={c: v["ok"] for c, v in checks.items()})
    if not all(v["ok"] for v in checks.values()):
        summary["checks"] = checks
        (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True))
        print("A4 " + json.dumps({"refused_before_draw": summary["checks_before_draw"]}))
        return 1

    if a.session_m0_pythonpath:
        summary["rekeyed"] = {}
        for m in members:
            m["st"] = rekey(m, pd, part, a, out)
            m["header"] = a0.public_header(m["st"]["dir"], m["n"])
            summary["rekeyed"][m["tid"]] = {"pin_as_served": m["st"]["pin_as_served"], "pin": m["st"]["pin"],
                                            "class_unchanged": m["st"]["class_unchanged"]}

    # one draw over the whole population, each member served its share
    p = a0.run([a.lean, "draw", "--population", str(n_all), "--k", str(a.k)], out / "draw.log")
    union_draw = json.loads(next(x for x in p.stdout.splitlines() if x.startswith("DRAW ")).split(" ", 1)[1])
    (out / "draw.json").write_text(json.dumps(union_draw, sort_keys=True))
    served_units, proved, instances, verdicts, statement, sessions, counts, session_roots = [], [], 0, {}, {}, [], {}, {}
    for m in members:
        local = sorted(u - m["base"] for u in union_draw["units"] if m["base"] <= u < m["base"] + m["n"])
        counts[m["tid"]] = len(local)
        if not local:
            continue
        sdir = out / f"session-m{m['k']}"
        sdir.mkdir(parents=True, exist_ok=True)
        (sdir / "draw.json").write_text(json.dumps({"law": "subset", "population": m["n"], "k": len(local), "units": local}, sort_keys=True))
        t = time.time()
        record, live = a0.session(a.bin, m["st"], m["n"], f"subset:{len(local)}", a.port + m["k"], sdir, draw_file=sdir / "draw.json")
        draw_t = (record or {}).get("unit_draw") or {}
        v = D.check_draw(draw_t, {"population": m["n"]}, {"law": "subset", "k": len(local)})
        served_units += [m["base"] + u for u in draw_t.get("units", [])]
        session_roots.update({f"{m['tid']}/{p}": r for p, r in ((live or {}).get("roots") or {}).items()})
        proved += a0.proved_units(m["header"], live or {})
        instances += (live or {}).get("instances", 0)
        lean_u = a0.lean_draw_test(a.lean, [{"record": draw_t or None, "statement": (live or {}).get("unit_draw"),
                                             "instances": (live or {}).get("instances", 0)}], sdir, "u1-u3")
        verdicts[f"m0-verifier/m{m['k']}"] = {"accepted": ((record or {}).get("verdict") or {}).get("accepted") is True
                                              and (live or {}).get("roots") == m["header"]["frame_v3"]["roots"] and v.ok,
                                              "why": ((record or {}).get("verdict") or {}).get("why")}
        verdicts[f"lean-draw/m{m['k']}"] = {"accepted": bool(lean_u) and all(c["ok"] for c in lean_u), "detail": lean_u}
        verdicts[f"lean-verify/m{m['k']}"] = {**a0.lean_verify(a.lean, m["st"], m["n"], sdir), "tool": "flock-verify verify --archive (PR #118)"}
        statement[m["tid"]] = {"session_s": round(time.time() - t, 1), "circuit_sha512": (live or {}).get("circuit_sha512"),
                               "statement_digest": (live or {}).get("statement_digest"), "sigma": (live or {}).get("sigma"),
                               "timing": {x: (live or {}).get(x) for x in ("e2e_s", "prove_total_s", "k_log", "proof_bytes")},
                               "lean_verify_s": verdicts[f"lean-verify/m{m['k']}"].get("verify_s"),
                               "lean_partition": verdicts[f"lean-verify/m{m['k']}"].get("partition")}
        sessions += [a0.sha512_file(f) for f in sorted((sdir / "sessions").glob("*/session.json"))]
    result = audit.audit_record(rec, receipt, union_draw, law, sorted(proved), instances, statement, verdicts, a.delta, sessions=sessions,
                                served=sorted(served_units), session_roots=session_roots)
    result["draw_counts"] = counts
    result["served_is_lean_draw"] = sorted(served_units) == sorted(union_draw["units"])
    result["checks_before_draw"] = checks
    (out / "audit.json").write_text(json.dumps(result, indent=1, sort_keys=True))

    neg: dict[str, dict] = {}
    other = P.mixed_population_program([(m["defn"], m["n"] - (1 if i == 0 else 0)) for i, m in enumerate(members)])
    v = R.check(rec, other, query, law, a0.SCHEME, want, leaf_layer=leaf_layer)
    neg["registration-partition-mismatch (R2)"] = {"refused": not v.ok, "codes": list(v.codes)}
    v = R.check(rec, program, query, law, a0.SCHEME, {nm: {**w, "domain": "00" * 64} for nm, w in want.items()}, leaf_layer=leaf_layer)
    neg["domains-not-the-layout's (R3)"] = {"refused": not v.ok, "codes": list(v.codes)}
    v = D.check_draw({**union_draw, "units": union_draw["units"][:-1] + [n_all]}, rec, law)
    neg["draw-outside-the-partition"] = {"refused": not v.ok, "codes": list(v.codes)}
    if proved:
        kept = proved[1:]
        v = D.check_proved(union_draw, kept, len(kept))
        neg["proved-set-short-of-the-draw"] = {"refused": not v.ok, "codes": list(v.codes)}
    v = R.check(dict(rec, leaf_layer=None), program, query, law, a0.SCHEME, want, leaf_layer=leaf_layer)
    neg["registration-null-leaf-layer (R4)"] = {"refused": not v.ok, "codes": list(v.codes)}
    neg.update(a0.audit_negatives(result, law, a.delta, session_roots, served=sorted(served_units)))
    (out / "negatives.json").write_text(json.dumps(neg, indent=1, sort_keys=True))

    summary["audit"] = {x: result.get(x) for x in ("accepted", "complete", "wrong_units")}
    summary["audit"]["verdicts"] = {x: y.get("accepted") for x, y in verdicts.items()}
    summary.update(draw_counts=counts, served_is_lean_draw=result["served_is_lean_draw"], timing=statement,
                   negatives={x: y["refused"] for x, y in neg.items()})
    (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str))
    ok = bool(result.get("accepted")) and all(summary["negatives"].values())
    print("A4 " + json.dumps({x: summary[x] for x in ("audit", "draw_counts", "negatives")}, sort_keys=True, default=str))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
