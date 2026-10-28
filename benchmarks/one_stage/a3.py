"""One-stage audit, run A3: every unit type M0's statement proves in one #101 decoder layer, as one audit with one sample and
one integrity profile.

    PYTHONPATH=<tree>:<tree>/benchmarks/numerical/python:<tree>/integrations/vllm:<tree>/catalog:<tree>/kernels:<tree>/experimental \\
      python benchmarks/one_stage/a3.py --sets DIR --m0-pythonpath PATH --bin flock-circuit --lean flock-verify --out DIR \\
      [--layer 0] [--k 128] [--delta 2^-20]

* **Population.** The captured instances of layer `--layer` in each set under `--sets` (one directory per template, in the
  layer's order), staged by M0 as one stand-in population per template (`stage_subset.py`: the frame domains bind the
  selection). The program is `TemplatePopulations_v1` over the templates' Definitions, and the partition is
  `Q_template_instances` v0: one unit per instance, no width rule.
* **One registration** lists every template's roots (ports named `<set>/<port>`), and `leaf_layer` is the SHA-512 of the
  canonical map from set to the SHA-512 of its public file.
* **One sample, `Subset(k)` over the union.** The verifier's Lean `flock-verify draw` draws k units over the whole population.
  With `--draw-file` (M0 `serve --draw-file`) each template's server serves its share of that very draw, so the audited
  draw is the verifier's; without it, the Lean draw fixes only how many units fall in each template and each server draws
  a uniform subset of that size, which is still exactly `Subset(k)` over the union (conditioned on the counts, a uniform
  k-subset of the union is uniform within each template). The per-template sessions are one table each (M0 has no
  multi-table statement yet).
* **Stand-in.** The values are #101's served values, committed afresh; every record says so.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import a0  # noqa: E402
import r2_population as R2  # noqa: E402
from verity.ml.circuits.codec import canonical_json  # noqa: E402
from verity_numerical.bench.input_sets import InputSet  # noqa: E402
from verity.core.protocols.sampled_proofs.one_stage import draw as D, partition as P, registration as R, verifier as V  # noqa: E402

ORDER = ("rmstriton", "gemm2048", "rope", "rmsfused", "silu", "gemm8192")


def layer_indices(set_dir: Path, layer: int) -> list[int]:
    idx = json.loads((set_dir / "index.json").read_text())
    col = idx["columns"].index("op_path")
    return [k for k, row in enumerate(idx["rows"]) if row[col].startswith(f"model.layers.{layer}.")]


def stage_member(set_dir: Path, indices: list[int], layer: int, out: Path, m0_path: str) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    (out / "indices.json").write_text(json.dumps(indices))
    p = a0.run([sys.executable, str(HERE / "stage_subset.py"), "--input-set", str(set_dir), "--indices", str(out / "indices.json"),
                "--tag", f"layer{layer}", "--out", str(out)], out / "stage.log", {"PYTHONPATH": m0_path})
    if p.returncode != 0:
        raise RuntimeError(f"staging {set_dir.name} failed: {p.stderr[-2000:]}")
    info = json.loads(p.stdout.strip().splitlines()[-1])
    text = (out / "circuit.txt").read_text()
    return {"dir": out, "meta": json.loads(text.split("\n", 2)[1][5:]), "pin": hashlib.sha512(text.encode()).hexdigest(), **info}


def lean_split(lean: str, n: int, k: int, out: Path) -> dict:
    p = a0.run([lean, "draw", "--population", str(n), "--k", str(k)], out / "split-draw.log")
    line = next(x for x in p.stdout.splitlines() if x.startswith("DRAW "))
    return json.loads(line.split(" ", 1)[1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", required=True)
    ap.add_argument("--m0-pythonpath", required=True)
    ap.add_argument("--bin", required=True)
    ap.add_argument("--lean", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--k", type=int, default=128)
    ap.add_argument("--delta", type=float, default=2.0 ** -20)
    ap.add_argument("--port", type=int, default=7521)
    ap.add_argument("--draw-file", action="store_true", help="serve each template its share of the verifier's Lean draw (M0 --draw-file)")
    ap.add_argument("--statement", default=a0.STATEMENT_ID, help="the Lean verifier's statement name")
    ap.add_argument("--binding", default=a0.STAND_IN_BINDING, choices=sorted(a0.STAND_IN_BINDINGS), help="M0's frame-domain binding tag")
    a = ap.parse_args()
    a0.STATEMENT_ID = a.statement
    a0.STAND_IN_BINDING = a.binding
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(a.sets)
    dirs = sorted((d for d in root.iterdir() if (d / "manifest.json").exists()),
                  key=lambda d: next((i for i, o in enumerate(ORDER) if d.name.startswith(o)), len(ORDER)))
    summary: dict = {"layer": a.layer, "k": a.k, "delta": a.delta, "members": []}

    # serving commits (the stand-in), one population per template
    members = []
    for d in dirs:
        s = InputSet.open(str(d))
        idx = layer_indices(d, a.layer)
        if not idx:
            continue
        t = time.time()
        st = stage_member(d, idx, a.layer, out / f"stage-{d.name}", a.m0_pythonpath)
        header = a0.public_header(st["dir"], len(idx))
        defn = s.subcircuit.definition()
        members.append({"name": d.name, "dir": d, "idx": idx, "n": len(idx), "st": st, "header": header, "defn": defn,
                        "program_ok": a0.template_program_ok(st["meta"], defn),
                        "stage_s": round(time.time() - t, 1)})
    summary["members"] = [{"set": m["name"], "template": m["defn"].id, "n": m["n"], "stage_s": m["stage_s"],
                           "template_program_ok": m["program_ok"]} for m in members]

    # the partition over the verifier's own program
    program = P.mixed_population_program([(m["defn"], m["n"]) for m in members])
    query = P.template_instances_query([m["defn"] for m in members])
    ranges = P.ranges(program, query)
    n = P.units(program, query)
    for m, (tid, base, cnt) in zip(members, ranges):
        assert tid == P.template_id(m["defn"]) and cnt == m["n"]
        m["base"] = base

    # one registration, before the sample
    law = D.parse_law(f"subset:{a.k}")
    want, roots, publics = {}, [], {}
    for m in members:
        mine = a0.stand_in_ports(m["st"]["name"], m["st"]["content_digest"], m["st"]["meta"], m["n"])
        fv = m["header"]["frame_v3"]
        for p, w in mine.items():
            want[f"{m['name']}/{p}"] = w
            roots.append({"port": f"{m['name']}/{p}", "root": fv["roots"][p], "leaves": w["leaves"], "domain": fv["domain_ids"][p]})
        publics[m["name"]] = a0.sha512_file(m["st"]["dir"] / f"pub-{m['n']}.bin")
    leaf_layer = hashlib.sha512(canonical_json(publics)).hexdigest()
    prov = {m["name"]: {"selected": m["n"], "selection_digest": m["st"]["content_digest"],
                        **{k: v for k, v in a0.stand_in(str(m["dir"])).items() if k in ("row", "run", "run_root", "model")}}
            for m in members}
    window = {"kind": "stand-in", "of": "vllm-v1 commitment of served row #101", "layer": a.layer, "sets": prov,
              "link": "provenance, not proof: vu_export read these values under run_root and replayed each bit-exact"}
    rec = R.record(program, query, law, a0.SCHEME, roots, window, leaf_layer=leaf_layer)
    (out / "registration.json").write_text(json.dumps(rec, indent=1, sort_keys=True))
    checks = {"registration": V.check(rec, program, query, law, a0.SCHEME, want, leaf_layer=leaf_layer).to_json()}
    for m in members:
        fv = m["header"]["frame_v3"]
        mine = {r["port"].split("/", 1)[1]: r for r in rec["roots"] if r["port"].startswith(m["name"] + "/")}
        ok = all(fv["roots"][p] == r["root"] and fv["domain_ids"][p] == r["domain"] for p, r in mine.items()) and m["header"]["instances"] == m["n"]
        checks[f"public/{m['name']}"] = {"ok": ok, "codes": [] if ok else ["public"], "detail": {}}
    receipt = R.receipt(rec, "one-stage-e2e verifier (a3.py)")
    (out / "receipt.json").write_text(json.dumps(receipt, indent=1, sort_keys=True))
    summary.update(population=n, partition=rec["partition"], registration=R.digest(rec),
                   checks_before_draw={k: v["ok"] for k, v in checks.items()})
    if not all(v["ok"] for v in checks.values()) or not all(m["program_ok"] for m in members):
        (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True))
        print("A3 " + json.dumps({"refused_before_draw": summary["checks_before_draw"]}))
        return 1

    # the sample: the verifier's split over the union, then each template's live sample of its share
    split = lean_split(a.lean, n, a.k, out)
    counts = {m["name"]: sum(1 for u in split["units"] if m["base"] <= u < m["base"] + m["n"]) for m in members}
    (out / "split.json").write_text(json.dumps({"lean_draw": split, "counts": counts}, indent=1))
    union_units, proved, instances, verdicts, statement, sessions, session_roots = [], [], 0, {}, {}, [], {}
    for j, m in enumerate(members):
        k_t = counts[m["name"]]
        if k_t == 0:
            continue
        sdir = out / f"session-{m['name']}"
        sdir.mkdir(parents=True, exist_ok=True)
        dfile = None
        if a.draw_file:
            dfile = sdir / "draw.json"
            local = sorted(u - m["base"] for u in split["units"] if m["base"] <= u < m["base"] + m["n"])
            dfile.write_text(json.dumps({"law": "subset", "population": m["n"], "k": k_t, "units": local}, sort_keys=True))
        t = time.time()
        record, live = a0.session(a.bin, m["st"], m["n"], f"subset:{k_t}", a.port + j, sdir, draw_file=dfile)
        m["session_s"] = round(time.time() - t, 1)
        draw_t = (record or {}).get("unit_draw") or {}
        v = V.check_draw(draw_t, {"population": m["n"]}, {"law": "subset", "k": k_t})
        union_units += [m["base"] + u for u in draw_t.get("units", [])]
        session_roots.update({f"{m['name']}/{p}": r for p, r in ((live or {}).get("roots") or {}).items()})
        proved += [m["base"] + u for u in a0.proved_units(m["header"], live or {})]
        instances += (live or {}).get("instances", 0)
        lean_u = a0.lean_draw_test(a.lean, [{"record": draw_t or None, "statement": (live or {}).get("unit_draw"),
                                             "instances": (live or {}).get("instances", 0)}], sdir, "u1-u3")
        verdicts[f"m0-verifier/{m['name']}"] = {"accepted": ((record or {}).get("verdict") or {}).get("accepted") is True
                                                and (live or {}).get("roots") == m["header"]["frame_v3"]["roots"] and v.ok,
                                                "why": ((record or {}).get("verdict") or {}).get("why"), "draw_check": v.to_json()}
        verdicts[f"lean-draw/{m['name']}"] = {"accepted": bool(lean_u) and all(c["ok"] for c in lean_u), "detail": lean_u}
        verdicts[f"lean-verify/{m['name']}"] = {**a0.lean_verify(a.lean, m["st"], m["n"], sdir), "tool": "flock-verify verify --archive (PR #118)"}
        statement[m["name"]] = {k: (live or {}).get(k) for k in ("circuit_sha512", "statement_digest", "public_sha512", "sigma")}
        statement[m["name"]]["timing"] = {k: (live or {}).get(k) for k in ("e2e_s", "witness_s", "prove_total_s", "k_log", "units", "proof_bytes")}
        statement[m["name"]]["timing"]["verify_s"] = ((record or {}).get("verdict") or {}).get("verify_s")
        statement[m["name"]]["timing"]["lean_verify_s"] = verdicts[f"lean-verify/{m['name']}"].get("verify_s")
        sessions += [a0.sha512_file(f) for f in sorted((sdir / "sessions").glob("*/session.json"))]
    union = {"law": "subset", "population": n, "k": a.k, "units": sorted(union_units)}
    # with --draw-file the audited draw is the verifier's Lean draw, and every session must have served exactly its share
    audited, served = (split, sorted(union_units)) if a.draw_file else (union, None)
    result = V.audit_record(rec, receipt, audited, law, sorted(proved), instances, statement, verdicts, a.delta,
                            sessions=sessions, served=served, session_roots=session_roots)
    result["split"] = {"counts": counts, "lean_draw_sha512": hashlib.sha512(canonical_json(split)).hexdigest(),
                       "served_is_lean_draw": sorted(union_units) == sorted(split["units"]),
                       "rule": ("each template served its share of the verifier's Lean draw (--draw-file)" if a.draw_file else
                                "counts from the verifier's Lean draw over the union; positions from each template's live draw")}
    result["checks_before_draw"] = checks
    (out / "audit.json").write_text(json.dumps(result, indent=1, sort_keys=True))

    # negatives
    neg: dict[str, dict] = {}
    fewer = R2.program([(m["defn"], m["n"]) for m in members], mixed=True)
    v = V.check(rec, fewer, query, law, a0.SCHEME, want, leaf_layer=leaf_layer)
    neg["registration-partition-mismatch"] = {"refused": not v.ok, "codes": list(v.codes)}
    v = V.check(rec, program, P.template_instances_query([m["defn"] for m in members][:-1]), law, a0.SCHEME, want, leaf_layer=leaf_layer)
    neg["registration-query-drops-a-template"] = {"refused": not v.ok, "codes": list(v.codes)}
    v = V.check_draw({**union, "units": union["units"][:-1] + [n]}, rec, law)
    neg["union-draw-outside-the-partition"] = {"refused": not v.ok, "codes": list(v.codes)}
    for m in members:
        if counts[m["name"]]:
            lo, hi = m["base"], m["base"] + m["n"]
            kept = [u for u in proved if not lo <= u < hi]
            v = V.check_proved(union, kept, len(kept))
            neg[f"session-dropped ({m['name']})"] = {"refused": not v.ok, "codes": list(v.codes)}
            break
    moved = sorted(proved)
    if moved:
        moved[0] = next(u for u in range(n) if u not in union["units"])
        v = V.check_proved(union, sorted(moved), instances)
        neg["proved-unit-not-drawn"] = {"refused": not v.ok, "codes": list(v.codes)}
    v = V.check(dict(rec, leaf_layer=None), program, query, law, a0.SCHEME, want, leaf_layer=leaf_layer)
    neg["registration-null-leaf-layer (R4)"] = {"refused": not v.ok, "codes": list(v.codes)}
    neg.update(a0.audit_negatives(result, law, a.delta, session_roots, served=served))
    (out / "negatives.json").write_text(json.dumps(neg, indent=1, sort_keys=True))

    summary["audit"] = {k: result.get(k) for k in ("accepted", "complete", "wrong_units")}
    summary["audit"]["verdicts"] = {k: v.get("accepted") for k, v in verdicts.items()}
    summary["counts"] = counts
    summary["timing"] = {m: statement[m]["timing"] for m in statement}
    summary["negatives"] = {k: v["refused"] for k, v in neg.items()}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True))
    ok = bool(result.get("accepted")) and all(summary["negatives"].values())
    print("A3 " + json.dumps(summary, sort_keys=True))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
