"""One-stage audit, run A5: a deferred Commit's window, a stand-in. The registration is over serving rows of the Commit's own
values, committed after serving from its sealed replay bundle opened against its record and bound to its vllm-v1 run root, and
the drawn units go to M0's prover and the Lean verifier of record. Serving's commitment is the vllm-v1 roots, so the window is
core's `stand-in`, its link to them provenance (`of`: the run root, the bundle's manifest digest and `verity-vllm serving-rows`):
that the rows hold the values under the vllm-v1 roots is the committer's opening, not proved. It becomes `served` once serving's
record is frame-v3-sha512 / hm96-sha512 itself, or once the circuit opens the vllm-v1 SHA-256 leaves (`pipeline.deferred_rows`).

    PYTHONPATH=<tree>:<tree>/benchmarks/numerical/python:<tree>/integrations/vllm \\
      python benchmarks/one_stage/a5.py --bundle DIR --row-dir DIR --partition F --window DIR --template-sets DIR \\
      --m0-pythonpath PATH --bin flock-circuit --lean flock-verify --out DIR [--delta 0.5]

* **Serving's files.** `--window DIR` is what `verity-vllm serving-rows` wrote: `registration.json` (core's v1 record, window
  kind `stand-in`, its run the Commit's run root, its `of` and `link` `deferred_rows.stand_in`'s), `window.json`, and per member
  k `m<k>-public.bin` and `m<k>-private.bin` (prover only). The prover wraps each member's bodies under the circuit for its
  template (`serving_rows.flock_input_files`).
* **The verifier's own copies.** `--bundle` and `--row-dir` are the deferred Commit's record and its Build's Programs of record;
  the verifier derives its window from them alone (`deferred_rows.window_of_record`): the population program and partition
  over the served Programs, the law, and the run root of record. It composes each member's circuit itself from an input set
  of its template under `--template-sets` (the lowering is all it reads; the values are never used), bound to its partition
  and program. It holds every public file to that circuit, its unit indices and classes, and the registered roots and served
  domains. Then R1-R7 run over the one registration with the leaf layer over the public bodies it holds
  (`serving_rows.leaf_layer`), and the window is held to its own link (`source`: a stand-in whose run, `of` and `link` are
  `deferred_rows.stand_in` over the run root of record and the digest of the bundle's manifest it opened).
* **One draw** by the registered law (the partition file's `subset:k`) over the whole population, made by the verifier's
  Lean `flock-verify draw`. Each member's server
  serves its share (`serve --draw-file`). The sessions are `--zk` (the masked protocol), because Lean accepts a subset draw
  only under `--zk` (verifier PROTOCOL §16.15). Lean `verify --zk` holds each session with `--partition` and `--program`
  (units derived).
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
from verity.ml.circuits.codec import canonical_json, encode_program  # noqa: E402
from verity_numerical.bench.input_sets import InputSet  # noqa: E402
from verity.core.protocols.sampled_proofs.one_stage import partition as P, registration as R, verifier as V  # noqa: E402
from verity_vllm.commit import serving_rows as CS  # noqa: E402
from verity_vllm.pipeline import deferred_rows as DRW  # noqa: E402
from verity_vllm.pipeline import replay_bundle as RB  # noqa: E402


def template_sets(root: Path) -> dict[str, tuple[Path, object]]:
    """Every input set under ``root`` by its template's descriptor id, with its Definition."""
    out = {}
    for d in sorted(p.parent for p in root.rglob("manifest.json")):
        defn = InputSet.open(str(d)).subcircuit.definition()
        out.setdefault(P.template_id(defn), (d, defn))
    return out


def body(path: Path) -> bytes:
    with open(path, "rb") as f:
        f.readline()
        return f.read()


def link(src: Path, dst: Path) -> None:
    if dst.exists():
        dst.unlink()
    try:
        os.link(src.resolve(), dst)
    except OSError:
        shutil.copyfile(src, dst)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True, help="the deferred Commit's sealed replay bundle (<commit out>/replay_bundle_p<pair>)")
    ap.add_argument("--row-dir", required=True, help="the row dir of the Build's Programs of record")
    ap.add_argument("--partition", required=True, help="the verifier's partition file: {templates, law, scope?}")
    ap.add_argument("--window", required=True, help="serving's window (verity-vllm serving-rows --out)")
    ap.add_argument("--template-sets", required=True, help="one input set per member template, for the lowering")
    ap.add_argument("--m0-pythonpath", required=True)
    ap.add_argument("--bin", required=True)
    ap.add_argument("--lean", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--delta", type=float, default=0.5)
    ap.add_argument("--port", type=int, default=7551)
    a = ap.parse_args()
    out, served = Path(a.out), Path(a.window)
    out.mkdir(parents=True, exist_ok=True)

    # the verifier's own window, from the Commit's record and the Build's Programs of record
    t0 = time.time()
    mine = DRW.window_of_record(a.bundle, a.row_dir, a.partition)
    program, query, part, pd, n_all, law = mine.program, mine.query, mine.partition, mine.partition_digest, mine.population, mine.law
    if law.get("law") != "subset":
        raise SystemExit(f"a5 draws by the subset law; the partition file's is {law}")
    want = CS.expected_ports(mine)
    sets = template_sets(Path(a.template_sets))
    summary: dict = {"run_root": mine.run, "population": n_all, "partition": pd, "law": law, "delta": a.delta,
                     "members": [{"k": k, "template": m.template, "n": m.n, "base": m.base} for k, m in enumerate(mine.members)],
                     "verifier_window_s": round(time.time() - t0, 2)}
    (out / "partition.json").write_text(json.dumps(part, sort_keys=True))
    (out / "program.json").write_bytes(canonical_json(encode_program(program)))

    rec = json.loads((served / "registration.json").read_text())
    shutil.copy(served / "registration.json", out / "registration.json")

    # each member: the verifier's circuit, the prover's files under it, and every check on the public file before the draw
    members, publics, checks = [], {}, {}
    for k, m in enumerate(mine.members):
        if m.template not in sets:
            summary["refused"] = {"no-template-set": m.template}
            (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True))
            print("A5 " + json.dumps(summary["refused"]))
            return 1
        set_dir, defn = sets[m.template]
        st_dir = out / f"m{k}"
        st_dir.mkdir(parents=True, exist_ok=True)
        t = time.time()
        p = a0.run([sys.executable, str(HERE / "compose_circuit.py"), "--input-set", str(set_dir), "--out", str(st_dir / "circuit.txt"),
                    "--partition", pd, "--program", part["program"]], st_dir / "compose.log", {"PYTHONPATH": a.m0_pythonpath})
        if p.returncode != 0:
            raise RuntimeError(f"composing member {k}'s circuit failed: {p.stderr[-2000:]}")
        made = json.loads(p.stdout.strip().splitlines()[-1])
        text = (st_dir / "circuit.txt").read_text()
        compose_s = round(time.time() - t, 1)
        got = CS.flock_input_files(served, k, text, st_dir / "prover")
        link(Path(got["pub"]), st_dir / f"pub-{m.n}.bin")
        link(Path(got["inst"]), st_dir / f"inst-{m.n}.bin")
        meta = json.loads(text.split("\n", 2)[1][5:])
        header = a0.public_header(st_dir, m.n)
        fv = header["frame_v3"]
        ok = {"pin": header.get("circuit_sha512") == made["pin"],
              "partition": meta.get("partition") == {"rule": P.FORMAT, "digest": pd},
              "program": meta.get("program_sha512") == part["program"],
              "indices": header["units"]["indices"] == list(range(m.base, m.base + m.n)),
              "classes": header["units"].get("classes") == [made.get("class_sha512")] * m.n,
              "template_program": a0.template_program_ok(meta, defn),
              "no_shared_rows": "shared_rows" not in header}
        for port in fv["roots"]:
            name = m.port(port)
            reg = next((r for r in rec["roots"] if r["port"] == name), None)
            ok[f"root:{port}"] = (name in want and reg is not None and reg["root"] == fv["roots"][port]
                                  and reg["domain"] == fv["domain_ids"][port] == want[name]["domain"])
        checks[f"m{k}"] = {"ok": all(ok.values()), "codes": [c for c, v in ok.items() if not v], "detail": {}}
        publics[m.template] = body(st_dir / f"pub-{m.n}.bin")
        members.append({"k": k, "m": m, "defn": defn, "header": header,
                        "st": {"dir": st_dir, "meta": meta, "pin": made["pin"], "partition_file": out / "partition.json",
                               "program_file": out / "program.json"},
                        "compose_s": compose_s, "pin": made["pin"], "class": made.get("class_sha512")})
    leaf_layer = CS.leaf_layer(publics)
    checks["registration"] = V.check(rec, program, query, law, CS.SCHEME, want, leaf_layer=leaf_layer).to_json()
    w = rec.get("window", {})
    mine_link = DRW.stand_in(mine.run, hashlib.sha256((Path(a.bundle) / RB.MANIFEST).read_bytes()).hexdigest())
    src = {"kind": w.get("kind") == "stand-in", "run": w.get("run") == mine.run, "of": w.get("of") == mine_link["of"],
           "link": w.get("link") == mine_link["link"]}
    checks["source"] = {"ok": all(src.values()), "codes": [c for c, v in src.items() if not v],
                        "detail": {"kind": w.get("kind"), "run": w.get("run"), "of": w.get("of")}}
    summary["source"] = w.get("kind")
    receipt = R.receipt(rec, "one-stage-e2e verifier (a5.py)")
    (out / "receipt.json").write_text(json.dumps(receipt, indent=1, sort_keys=True))
    summary.update(registration=R.digest(rec), leaf_layer=leaf_layer, checks_before_draw={c: v["ok"] for c, v in checks.items()},
                   circuits={mm["m"].template: {"pin": mm["pin"], "class": mm["class"], "compose_s": mm["compose_s"],
                                                "k_log": mm["st"]["meta"].get("k_log"), "unit_log": mm["st"]["meta"].get("unit_log")}
                             for mm in members})
    if not all(v["ok"] for v in checks.values()):
        summary["checks"] = checks
        (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True))
        print("A5 " + json.dumps({"refused_before_draw": summary["checks_before_draw"], "checks": checks}, default=str))
        return 1

    # one draw over the whole population; each member serves its share under --zk
    p = a0.run([a.lean, "draw", "--population", str(n_all), "--k", str(law["k"])], out / "draw.log")
    union_draw = json.loads(next(x for x in p.stdout.splitlines() if x.startswith("DRAW ")).split(" ", 1)[1])
    (out / "draw.json").write_text(json.dumps(union_draw, sort_keys=True))
    served_units, proved, instances, verdicts, statement, sessions, counts, session_roots = [], [], 0, {}, {}, [], {}, {}
    for mm in members:
        m, k = mm["m"], mm["k"]
        local = sorted(u - m.base for u in union_draw["units"] if m.base <= u < m.base + m.n)
        counts[m.template] = len(local)
        if not local:
            continue
        sdir = out / f"session-m{k}"
        sdir.mkdir(parents=True, exist_ok=True)
        (sdir / "draw.json").write_text(json.dumps({"law": "subset", "population": m.n, "k": len(local), "units": local}, sort_keys=True))
        t = time.time()
        record, live = a0.session(a.bin, mm["st"], m.n, f"subset:{len(local)}", a.port + k, sdir, draw_file=sdir / "draw.json", zk=True)
        session_s = round(time.time() - t, 1)
        draw_t = (record or {}).get("unit_draw") or {}
        v = V.check_draw(draw_t, {"population": m.n}, {"law": "subset", "k": len(local)})
        served_units += [m.base + u for u in draw_t.get("units", [])]
        session_roots.update({m.port(port): r for port, r in ((live or {}).get("roots") or {}).items()})
        proved += a0.proved_units(mm["header"], live or {})
        instances += (live or {}).get("instances", 0)
        lean_u = a0.lean_draw_test(a.lean, [{"record": draw_t or None, "statement": (live or {}).get("unit_draw"),
                                             "instances": (live or {}).get("instances", 0)}], sdir, "u1-u3")
        verdicts[f"m0-verifier/m{k}"] = {"accepted": ((record or {}).get("verdict") or {}).get("accepted") is True
                                         and (live or {}).get("roots") == mm["header"]["frame_v3"]["roots"] and v.ok,
                                         "why": ((record or {}).get("verdict") or {}).get("why"), "draw_check": v.to_json()}
        verdicts[f"lean-draw/m{k}"] = {"accepted": bool(lean_u) and all(c["ok"] for c in lean_u), "detail": lean_u}
        verdicts[f"lean-verify/m{k}"] = {**a0.lean_verify(a.lean, mm["st"], m.n, sdir, zk=True), "tool": "flock-verify verify --zk --archive"}
        statement[m.template] = {"session_s": session_s, **{x: (live or {}).get(x) for x in ("circuit_sha512", "statement_digest", "sigma")},
                                 "timing": {x: (live or {}).get(x) for x in ("e2e_s", "witness_s", "prove_total_s", "k_log", "units", "proof_bytes")},
                                 "verify_s": ((record or {}).get("verdict") or {}).get("verify_s"),
                                 "lean_verify_s": verdicts[f"lean-verify/m{k}"].get("verify_s"),
                                 "lean_partition": verdicts[f"lean-verify/m{k}"].get("partition")}
        sessions += [a0.sha512_file(f) for f in sorted((sdir / "sessions").glob("*/session.json"))]
    result = V.audit_record(rec, receipt, union_draw, law, sorted(proved), instances, statement, verdicts, a.delta, sessions=sessions,
                            served=sorted(served_units), session_roots=session_roots)
    result["draw_counts"] = counts
    result["checks_before_draw"] = checks
    (out / "audit.json").write_text(json.dumps(result, indent=1, sort_keys=True, default=str))

    # negatives: each re-decides one input altered
    neg: dict[str, dict] = {}
    defs = [(mm["defn"], mm["m"].n) for mm in members]
    fewer = P.mixed_population_program([(d, n - (1 if i == 0 else 0)) for i, (d, n) in enumerate(defs)]) if len(defs) > 1 \
        else P.population_program(defs[0][0], defs[0][1] - 1)
    v = V.check(rec, fewer, query, law, CS.SCHEME, want, leaf_layer=leaf_layer)
    neg["registration-partition-mismatch (R2)"] = {"refused": not v.ok, "codes": list(v.codes)}
    other = CS.expected_ports(CS.Window(program, query, law, "00" * 32, mine.members))
    v = V.check(rec, program, query, law, CS.SCHEME, other, leaf_layer=leaf_layer)
    neg["domains-bound-to-another-run-root (R3)"] = {"refused": not v.ok, "codes": list(v.codes)}
    first = members[0]["m"].template
    flipped = dict(publics, **{first: publics[first][:-1] + bytes([publics[first][-1] ^ 1])})
    v = V.check(rec, program, query, law, CS.SCHEME, want, leaf_layer=CS.leaf_layer(flipped))
    neg["public-body-not-the-registered-leaf-layer (R4)"] = {"refused": not v.ok, "codes": list(v.codes)}
    v = V.check_draw({**union_draw, "units": union_draw["units"][:-1] + [n_all]}, rec, law)
    neg["draw-outside-the-partition"] = {"refused": not v.ok, "codes": list(v.codes)}
    if proved:
        v = V.check_proved(union_draw, proved[1:], len(proved) - 1)
        neg["proved-set-short-of-the-draw"] = {"refused": not v.ok, "codes": list(v.codes)}
    neg.update(a0.audit_negatives(result, law, a.delta, session_roots, served=sorted(served_units)))
    (out / "negatives.json").write_text(json.dumps(neg, indent=1, sort_keys=True))

    summary["audit"] = {x: result.get(x) for x in ("accepted", "complete", "wrong_units")}
    summary["audit"]["verdicts"] = {x: y.get("accepted") for x, y in verdicts.items()}
    summary["audit"]["why"] = {x: y.get("why") for x, y in verdicts.items() if y.get("accepted") is not True}
    summary.update(draw=union_draw["units"], draw_counts=counts, served_is_lean_draw=sorted(served_units) == sorted(union_draw["units"]),
                   timing=statement, negatives={x: y["refused"] for x, y in neg.items()})
    (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str))
    ok = bool(result.get("accepted")) and all(summary["negatives"].values())
    print("A5 " + json.dumps({x: summary[x] for x in ("source", "audit", "draw", "draw_counts", "negatives")}, sort_keys=True, default=str))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
