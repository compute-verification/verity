"""One-stage audit, run A6: the W8A8 linears of an int8 decoder layer, each output coordinate a unit, sampled by the stratified
law in one audit. A coordinate is `ScaledMmS8Coordinate_v1{K, DOT=BlackwellS8ImmaDot32_v1}` (K/32 satfinite s8 IMMA steps on
sm_120, then `benchmarks/ppl/steps.py`'s `int8_linear` scaling, the int8 step vLLM's int8 profile serves), one template per K.

    PYTHONPATH=<tree>:<tree>/verity/ml/flock/python:<tree>/integrations/vllm:<tree>/tools/verity/circuit_check/src \\
      python benchmarks/one_stage/a6.py --bin flock-circuit --lean flock-verify --out DIR [--members 2048:21504,8192:2048] \\
      [--k 32] [--delta 2^-20]

* **Population.** `--members K:N,...`: N coordinates at each K. The default is one Llama-3.2-1B decoder layer for one token:
  qkv's 3072, o's 2048 and gate_up's 16384 coordinates at K = 2048, and down's 2048 at K = 8192. The program is
  `TemplatePopulations` over the templates, and the partition `Q_template_instances` v0, one unit per coordinate.
* **Stand-in values.** Each coordinate's rows are seeded N(0, 1) and N(0, 0.02) rows, each quantized to s8 as steps.py's
  `quant_rows` does (`benchmarks/served_units/stage.py`'s s8-coord draw), committed afresh. The circuit's output is held to
  that numpy twin's. The window says so; vllm's int8 Program and its served rows replace these values.
* **Statements.** One per template, staged by C-Flock's `UnitStager` bound to the partition: META names the template, the
  partition and the program, and the public file the member's global unit indices. The verifier derives each port's frame
  domain itself from META's row schemas (C-Flock's set-bound domains), holds the public files to the registration, and the
  leaf layer is the SHA-512 of the canonical map from member to the SHA-512 of its public file.
* **One draw, the stratified law** (`stratified:K`: one stratum per template, k_s = max(1, round(k n_s / n))), made by the
  verifier's Lean `flock-verify draw --program --partition --stratified K` over its own program and partition object. A
  stratum is one member, so each member's server serves its stratum's draw (`serve --zk --draw-file`), a uniform
  k_s-subset of the member, and Lean `verify --zk --partition --program` holds each session (units derived).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "served_units"))

import a0  # noqa: E402
import stage as SU  # noqa: E402
from verity.ml.circuits.codec import canonical_json, encode_program  # noqa: E402
from verity.ml.circuits.defs import bind  # noqa: E402
from verity.ml.circuits.ops import gemm as OG  # noqa: E402
from verity.ml.commitments.identity import identity_digest_sha512  # noqa: E402
from verity.ml.commitments.indexed import RangeIndexedDomain  # noqa: E402
from verity.ml.commitments.merkle import CommitmentDomain  # noqa: E402
from verity.ml.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_SEG as SEG, SCHEMA_HM96_SHA512_ROW_V2 as V2  # noqa: E402
from verity_flock import class_statement as CS  # noqa: E402
from verity.core.protocols.sampled_proofs.one_stage import draw as D, partition as P, registration as R, verifier as V  # noqa: E402

#: one Llama-3.2-1B decoder layer's W8A8 coordinates for one token, by K
LLAMA_3_2_1B_LAYER = "2048:21504,8192:2048"
WINDOW_OF = ("seeded s8 rows quantized as benchmarks/ppl/steps.py's quant_rows (benchmarks/served_units/stage.py's s8-coord "
             "draw), committed afresh")


def template(K: int):
    return bind(OG.ScaledMmS8Coordinate, K=K, **OG.GEMM_SM120_S8)


def parse_members(text: str) -> list[tuple[int, int]]:
    out = [(int(k), int(n)) for k, n in (m.split(":") for m in text.split(","))]
    if len({k for k, _ in out}) != len(out) or any(k % 32 or n < 1 for k, n in out):
        raise SystemExit(f"--members: distinct K, each a multiple of 32, and N >= 1 each: {text!r}")
    return out


def frame_ports(set_name: str, meta: dict, n: int) -> dict[str, dict]:
    """The verifier's own derivation of a bit-row statement's frame domains: C-Flock's set-bound domain (`circuit.BINDING_TAG`)
    of each port over the instances [0, n), with the port's META row schema, one leaf per instance."""
    out = {}
    for (port, _w, _c), (schema, _bits) in zip(meta["ports"], meta["row_schemas"], strict=True):
        b = identity_digest_sha512(a0.STAND_IN_BINDING, {"set": set_name, "lo": 0, "hi": n, "port": port, "schema": schema})
        out[port] = {"leaves": n, "domain": CommitmentDomain(b, -1, RangeIndexedDomain(0, n), hash="sha512").domain_id.hex()}
    return out


def stage_member(K: int, n: int, base: int, bound: dict, out: Path) -> dict:
    """Member K's statement: n stand-in coordinates staged by a `UnitStager` bound to the partition, as units base..base+n."""
    t0 = time.time()
    tag = f"s8-coord-k{K}"
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(f"verity/one-stage/a6/v1\0{tag}".encode()).digest()[:8], "little"))
    rows, twin = SU._s8_coord_draw({"K": K}, n, rng)
    in_rows = [("x", SEG, 16 * rows["x"].shape[1], ["x", "sx"]), ("w", V2, 16 * rows["w"].shape[1], ["wrow", "sw"])]
    st = CS.UnitStager(template(K), f"unit/a6-{tag}", in_rows, [(V2, (0,))], bind=bound)
    t_build = time.time() - t0
    salt_key = hashlib.sha256(f"verity/one-stage/a6/salts\0{tag}".encode()).digest()
    got = st.stage(rows, out, salt_key, indices=range(base, base + n))
    (out / "inst.bin").rename(out / f"inst-{n}.bin")
    (out / "pub.bin").rename(out / f"pub-{n}.bin")
    (out / "tables").mkdir(exist_ok=True)
    o0 = np.asarray(got["outs"]["o0"], dtype=np.uint16)[:, :twin.shape[1]]
    return {"tag": tag, "dir": out, "meta": st.meta, "pin": st.pin, "class": st.class_sha512, "set": st.iset.name,
            "twin": {"agrees": bool(np.array_equal(o0, twin)), "instances": n}, "ands": got["ands"], "unit_nnz": got["unit_nnz"],
            "seconds": {"build": round(t_build, 1), "total": round(time.time() - t0, 1)}}


def statement_checks(m: dict, rec: dict, want: dict) -> dict:
    """What member m's public file and META bind, against the registration and the verifier's own derivation."""
    meta, header, fv = m["st"]["meta"], m["header"], m["header"]["frame_v3"]
    ok = {"scheme": header.get("scheme") == a0.SCHEME, "pin": header.get("circuit_sha512") == m["st"]["pin"],
          "partition": meta.get("partition") == {"rule": P.FORMAT, "digest": rec["partition"]},
          "program": meta.get("program_sha512") == rec["program"],
          "template_program": a0.template_program_ok(meta, m["defn"]),
          "instances": header["instances"] == m["n"],
          "indices": header["units"]["indices"] == list(range(m["base"], m["base"] + m["n"])),
          "classes": header["units"].get("classes") == [m["st"]["class"]] * m["n"]}
    registered = {r["port"]: r for r in rec["roots"]}
    for port, root in fv["roots"].items():
        name = f"{m['name']}/{port}"
        r = registered.get(name)
        ok[f"root:{port}"] = (r is not None and r["root"] == root and r["domain"] == fv["domain_ids"][port] == want[name]["domain"])
    return {"ok": all(ok.values()), "codes": [c for c, v in ok.items() if not v], "detail": {}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--lean", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--members", default=LLAMA_3_2_1B_LAYER, help="K:N,...: N coordinates at each K")
    ap.add_argument("--k", type=int, default=32, help="the stratified rule's k")
    ap.add_argument("--delta", type=float, default=2.0 ** -20)
    ap.add_argument("--port", type=int, default=7561)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    spec = parse_members(a.members)
    summary: dict = {"members_spec": a.members, "k": a.k, "delta": a.delta}

    # the verifier's program and partition
    defs = [template(K) for K, _ in spec]
    program = P.mixed_population_program([(d, n) for d, (_K, n) in zip(defs, spec)])
    query = P.template_instances_query(defs)
    part = P.partition(program, query)
    pd = P.digest(part)
    n_all = P.units(program, query)
    ranges = P.ranges(program, query)
    (out / "partition.json").write_text(json.dumps(part, sort_keys=True))
    (out / "program.json").write_bytes(canonical_json(encode_program(program)))
    law = D.parse_law(f"stratified:{a.k}")
    strata = P.strata(program, query)
    assert [t for t, _ in strata] == [P.template_id(d) for d in defs] and all(len(rs) == 1 for _, rs in strata)

    # serving commits (the stand-in): each member's statement, bound to the partition
    members = []
    for d, (K, n), (tid, base, cnt) in zip(defs, spec, ranges):
        assert tid == P.template_id(d) and cnt == n
        st = stage_member(K, n, base, {"partition": pd, "program": part["program"]}, out / f"stage-k{K}")
        st.update(partition_file=out / "partition.json", program_file=out / "program.json")
        members.append({"name": st["tag"], "K": K, "n": n, "base": base, "defn": d, "st": st, "header": a0.public_header(st["dir"], n)})
        print(f"A6 staged {st['tag']}: {n} units from {base}, {st['ands']} ANDs, {st['unit_nnz']} nonzeros, twin {st['twin']['agrees']}, "
              f"{st['seconds']['total']} s", flush=True)
    summary["members"] = [{"name": m["name"], "template": P.template_id(m["defn"]), "n": m["n"], "base": m["base"],
                           "ands": m["st"]["ands"], "unit_nnz": m["st"]["unit_nnz"], "k_log": m["st"]["meta"].get("k_log"),
                           "unit_log": m["st"]["meta"].get("unit_log"), "pin": m["st"]["pin"], "class": m["st"]["class"],
                           "twin": m["st"]["twin"], "stage_s": m["st"]["seconds"]} for m in members]

    # one registration, before the draw
    want, roots, publics = {}, [], {}
    for m in members:
        fv = m["header"]["frame_v3"]
        for port, w in frame_ports(m["st"]["set"], m["st"]["meta"], m["n"]).items():
            want[f"{m['name']}/{port}"] = w
            roots.append({"port": f"{m['name']}/{port}", "root": fv["roots"][port], "leaves": w["leaves"], "domain": fv["domain_ids"][port]})
        publics[m["name"]] = a0.sha512_file(m["st"]["dir"] / f"pub-{m['n']}.bin")
    leaf_layer = hashlib.sha512(canonical_json(publics)).hexdigest()
    window = {"kind": "stand-in", "of": WINDOW_OF, "members": {m["name"]: {"template": P.template_id(m["defn"]), "n": m["n"]} for m in members},
              "link": "none: no served run; vllm's int8 Program and its served rows replace these values"}
    rec = R.record(program, query, law, a0.SCHEME, roots, window, leaf_layer=leaf_layer)
    (out / "registration.json").write_text(json.dumps(rec, indent=1, sort_keys=True))
    checks = {"registration": V.check(rec, program, query, law, a0.SCHEME, want, leaf_layer=leaf_layer).to_json()}
    for m in members:
        checks[f"statement/{m['name']}"] = statement_checks(m, rec, want)
        checks[f"twin/{m['name']}"] = {"ok": m["st"]["twin"]["agrees"], "codes": [] if m["st"]["twin"]["agrees"] else ["twin"], "detail": {}}
    receipt = R.receipt(rec, "one-stage-e2e verifier (a6.py)")
    (out / "receipt.json").write_text(json.dumps(receipt, indent=1, sort_keys=True))
    summary.update(population=n_all, partition=pd, registration=R.digest(rec), law=rec["law"],
                   checks_before_draw={c: v["ok"] for c, v in checks.items()})
    if not all(v["ok"] for v in checks.values()):
        summary["checks"] = checks
        (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True))
        print("A6 " + json.dumps({"refused_before_draw": summary["checks_before_draw"], "checks": checks}, default=str))
        return 1

    # the verifier's stratified draw, by its own Lean over its own program and partition object; each member serves its stratum
    p = a0.run([a.lean, "draw", "--program", str(out / "program.json"), "--partition", str(out / "partition.json"),
                "--stratified", str(a.k)], out / "draw.log")
    lean_draw = json.loads(next(x for x in p.stdout.splitlines() if x.startswith("DRAW ")).split(" ", 1)[1])
    (out / "draw.json").write_text(json.dumps(lean_draw, sort_keys=True))
    served, proved, instances, verdicts, statement, sessions, counts, session_roots = [], [], 0, {}, {}, [], {}, {}
    for j, m in enumerate(members):
        local = sorted(u - m["base"] for u in lean_draw["units"] if m["base"] <= u < m["base"] + m["n"])
        counts[m["name"]] = len(local)
        if not local:
            continue
        sdir = out / f"session-{m['name']}"
        sdir.mkdir(parents=True, exist_ok=True)
        (sdir / "draw.json").write_text(json.dumps({"law": "subset", "population": m["n"], "k": len(local), "units": local}, sort_keys=True))
        t = time.time()
        record, live = a0.session(a.bin, m["st"], m["n"], f"subset:{len(local)}", a.port + j, sdir, draw_file=sdir / "draw.json", zk=True)
        session_s = round(time.time() - t, 1)
        draw_t = (record or {}).get("unit_draw") or {}
        v = V.check_draw(draw_t, {"population": m["n"]}, {"law": "subset", "k": len(local)})
        served += [m["base"] + u for u in draw_t.get("units", [])]
        session_roots.update({f"{m['name']}/{port}": r for port, r in ((live or {}).get("roots") or {}).items()})
        proved += a0.proved_units(m["header"], live or {})
        instances += (live or {}).get("instances", 0)
        lean_u = a0.lean_draw_test(a.lean, [{"record": draw_t or None, "statement": (live or {}).get("unit_draw"),
                                             "instances": (live or {}).get("instances", 0)}], sdir, "u1-u3")
        verdicts[f"m0-verifier/{m['name']}"] = {"accepted": ((record or {}).get("verdict") or {}).get("accepted") is True
                                                and (live or {}).get("roots") == m["header"]["frame_v3"]["roots"] and v.ok,
                                                "why": ((record or {}).get("verdict") or {}).get("why"), "draw_check": v.to_json()}
        verdicts[f"lean-draw/{m['name']}"] = {"accepted": bool(lean_u) and all(c["ok"] for c in lean_u), "detail": lean_u}
        verdicts[f"lean-verify/{m['name']}"] = {**a0.lean_verify(a.lean, m["st"], m["n"], sdir, zk=True),
                                                "tool": "flock-verify verify --zk --archive --partition --program"}
        statement[m["name"]] = {"session_s": session_s, **{x: (live or {}).get(x) for x in ("circuit_sha512", "statement_digest", "sigma")},
                                "timing": {x: (live or {}).get(x) for x in ("e2e_s", "witness_s", "prove_total_s", "k_log", "units", "proof_bytes")},
                                "verify_s": ((record or {}).get("verdict") or {}).get("verify_s"),
                                "lean_verify_s": verdicts[f"lean-verify/{m['name']}"].get("verify_s"),
                                "lean_partition": verdicts[f"lean-verify/{m['name']}"].get("partition")}
        sessions += [a0.sha512_file(f) for f in sorted((sdir / "sessions").glob("*/session.json"))]
        lv = verdicts[f"lean-verify/{m['name']}"]
        print(f"A6 session {m['name']}: {len(local)} units, Lean {lv['accepted']} {lv.get('why') or ''}, {session_s} s", flush=True)
    result = V.audit_record(rec, receipt, lean_draw, rec["law"], sorted(proved), instances, statement, verdicts, a.delta,
                            sessions=sessions, served=sorted(served), session_roots=session_roots)
    result["draw_counts"] = counts
    result["checks_before_draw"] = checks
    (out / "audit.json").write_text(json.dumps(result, indent=1, sort_keys=True, default=str))

    # negatives: each re-decides one input altered
    neg: dict[str, dict] = {}
    fewer = P.mixed_population_program([(d, n - (1 if i == 0 else 0)) for i, (d, (_K, n)) in enumerate(zip(defs, spec))])
    v = V.check(rec, fewer, query, law, a0.SCHEME, want, leaf_layer=leaf_layer)
    neg["registration-partition-mismatch (R2)"] = {"refused": not v.ok, "codes": list(v.codes)}
    if len(defs) > 1:
        v = V.check(rec, program, P.template_instances_query(defs[:-1]), law, a0.SCHEME, want, leaf_layer=leaf_layer)
        neg["registration-query-drops-a-template (R2)"] = {"refused": not v.ok, "codes": list(v.codes)}
    for name, other in (("registration-law-another-k (R1)", D.parse_law(f"stratified:{a.k + 1}")),
                        ("registration-law-subset-not-stratified (R1)", D.parse_law(f"subset:{a.k}"))):
        v = V.check(rec, program, query, other, a0.SCHEME, want, leaf_layer=leaf_layer)
        neg[name] = {"refused": not v.ok, "codes": list(v.codes)}
    first = members[0]["name"]
    v = V.check(rec, program, query, law, a0.SCHEME, want,
                leaf_layer=hashlib.sha512(canonical_json({**publics, first: "00" * 64})).hexdigest())
    neg["public-file-not-the-registered-leaf-layer (R4)"] = {"refused": not v.ok, "codes": list(v.codes)}
    v = V.check_draw({**lean_draw, "units": lean_draw["units"][:-1] + [n_all]}, rec, rec["law"])
    neg["draw-outside-the-partition"] = {"refused": not v.ok, "codes": list(v.codes)}
    if len(members) > 1:
        a_lo, a_hi = members[0]["base"], members[0]["base"] + members[0]["n"]
        b_lo, b_hi = members[1]["base"], members[1]["base"] + members[1]["n"]
        drop = next(u for u in lean_draw["units"] if a_lo <= u < a_hi)
        add = next((u for u in range(b_lo, b_hi) if u not in lean_draw["units"]), None)
        if add is not None:
            v = V.check_draw({**lean_draw, "units": sorted(set(lean_draw["units"]) - {drop} | {add})}, rec, rec["law"])
            neg["draw-moves-a-unit-across-strata"] = {"refused": not v.ok, "codes": list(v.codes)}
    if proved:
        v = V.check_proved(lean_draw, proved[1:], len(proved) - 1)
        neg["proved-set-short-of-the-draw"] = {"refused": not v.ok, "codes": list(v.codes)}
        moved = sorted(proved)
        moved[0] = next(u for u in range(n_all) if u not in lean_draw["units"])
        v = V.check_proved(lean_draw, sorted(moved), instances)
        neg["proved-unit-not-drawn"] = {"refused": not v.ok, "codes": list(v.codes)}
    m0 = members[0]
    other_pd = P.digest(P.partition(fewer, query))
    c = statement_checks({**m0, "st": {**m0["st"], "meta": {**m0["st"]["meta"], "partition": {"rule": P.FORMAT, "digest": other_pd}}}}, rec, want)
    neg["statement-binds-another-partition"] = {"refused": not c["ok"], "codes": c["codes"]}
    sdir = out / f"session-{m0['name']}"
    if (sdir / "sessions").is_dir():
        bad = out / "neg-other-program"
        bad.mkdir(exist_ok=True)
        (bad / "partition.json").write_text(json.dumps(P.partition(fewer, query), sort_keys=True))
        (bad / "program.json").write_bytes(canonical_json(encode_program(fewer)))
        lv = a0.lean_verify(a.lean, {**m0["st"], "partition_file": bad / "partition.json", "program_file": bad / "program.json"},
                            m0["n"], sdir, zk=True)
        neg["lean-verify-under-another-program"] = {"refused": a0.lean_refused(lv, names="partition"), "lean": lv.get("why")}
    neg.update(a0.audit_negatives(result, rec["law"], a.delta, session_roots, served=sorted(served)))
    (out / "negatives.json").write_text(json.dumps(neg, indent=1, sort_keys=True))

    summary["audit"] = {x: result.get(x) for x in ("accepted", "complete", "wrong_units", "profile")}
    summary["audit"]["verdicts"] = {x: y.get("accepted") for x, y in verdicts.items()}
    summary["audit"]["why"] = {x: y.get("why") for x, y in verdicts.items() if y.get("accepted") is not True}
    summary.update(draw=lean_draw["units"], draw_counts=counts, served_is_lean_draw=sorted(served) == sorted(lean_draw["units"]),
                   timing=statement, negatives={x: y["refused"] for x, y in neg.items()})
    (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str))
    ok = bool(result.get("accepted")) and all(summary["negatives"].values())
    print("A6 " + json.dumps({x: summary[x] for x in ("population", "draw_counts", "audit", "negatives")}, sort_keys=True, default=str))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
