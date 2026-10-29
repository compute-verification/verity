"""One-stage audit, run A2: A1 with the serving commit taken from serving itself. vllm-serving-commit commits #101's RoPE
heads at serving time in the circuit prover's row format and hands over the registration, the verifier's public file,
the prover's file and the unit index; nothing else changes.

    PYTHONPATH=<tree>:<tree>/benchmarks/numerical/python:<tree>/integrations/vllm:<tree>/catalog:<tree>/kernels:<tree>/experimental \\
      python benchmarks/one_stage/a2.py --served DIR --template-set DIR --m0-pythonpath PATH --bin flock-circuit \\
      --lean flock-verify --out DIR [--k 256] [--delta 2^-20]

* **`--served DIR`** holds `registration.json`, `pub-N.bin`, `inst-N.bin` (rows and salts: the prover's only) and
  `index.json`. `--template-set` is any input set of the template, for the verifier's own lowering.
* **The verifier's own copies.** It builds the population program and the partition itself, recomputes every served
  domain (`registration.window_domain`) from the record's program, partition and `window.run`, composes the bound circuit
  itself and holds the public file's pin and bindings to it. N is taken from the registration (the served index); the
  verifier doesn't yet derive it from #101's program, and the audit record says so.
* **`--unbind`.** After checking serving's files as received (R1-R6, the pin, the bindings), the session runs on the
  verifier's own re-headered copies for the circuit prover's unbound statement (no `units.classes`, the unbound
  circuit's pin): the leaf layer, every `b || c` and output word, is byte for byte serving's, so the roots are
  unchanged; only the statement no longer carries the partition. For a Lean verifier that doesn't yet read the bound
  header.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import a0  # noqa: E402
import r2_population as R2  # noqa: E402
from verity.ml.circuits.codec import canonical_json, encode_program  # noqa: E402
from verity_numerical.bench.input_sets import InputSet  # noqa: E402
from verity_flock import circuit as FC  # noqa: E402
from verity.core.protocols.sampled_proofs.one_stage import audit, draw as D, partition as P, registration as R, verifier as V  # noqa: E402


def unbound_copy(st: dict, header: dict, n: int, a, out: Path) -> tuple[dict, dict]:
    """The verifier's own copies of the public and prover files for the circuit prover's unbound statement (its provisional unit cover): the
    same body, and a header whose `circuit_sha512` is the unbound circuit's pin, composed by the verifier. A class leaves
    META's bindings out, so `units.classes` is unchanged where that tree's circuit prover has classes and dropped where
    it hasn't."""
    ub = out / "unbound"
    ub.mkdir(parents=True, exist_ok=True)
    p = a0.run([sys.executable, str(HERE / "compose_circuit.py"), "--input-set", a.template_set, "--out", str(ub / "circuit.txt")],
               out / "compose-unbound.log", {"PYTHONPATH": a.unbound_m0_pythonpath or a.m0_pythonpath})
    made = json.loads(p.stdout.strip().splitlines()[-1])
    pin = made["pin"]
    for name in (f"pub-{n}.bin", f"inst-{n}.bin"):
        with open(st["dir"] / name, "rb") as f:
            head = json.loads(f.readline())
            body = f.read()
        head["units"] = {k: v for k, v in head["units"].items() if k != "classes"}
        if made.get("class_sha512"):
            head["units"]["classes"] = [made["class_sha512"]] * n
        head["circuit_sha512"] = pin
        (ub / name).write_bytes(FC.public_file(head, body) if name.startswith("pub-")
                                else json.dumps(head, sort_keys=True).encode() + b"\n" + body)
    text = (ub / "circuit.txt").read_text()
    new = {"dir": ub, "meta": json.loads(text.split("\n", 2)[1][5:]), "pin": pin}
    return new, a0.public_header(ub, n)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--served", required=True)
    ap.add_argument("--template-set", required=True)
    ap.add_argument("--m0-pythonpath", required=True)
    ap.add_argument("--bin", required=True)
    ap.add_argument("--lean", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=256)
    ap.add_argument("--delta", type=float, default=2.0 ** -20)
    ap.add_argument("--port", type=int, default=7531)
    ap.add_argument("--unbind", action="store_true", help="run the session on re-headered copies for the circuit prover's unbound statement")
    ap.add_argument("--legacy-partition", action="store_true",
                    help="the served files bind the pre-canonical partition object {program, query: {name, version, template, "
                         "width_rule}}; check them against it")
    ap.add_argument("--unbound-m0-pythonpath", default=None, help="the circuit prover's tree for the unbound circuit (default --m0-pythonpath)")
    ap.add_argument("--statement", default=a0.STATEMENT_ID, help="the Lean verifier's statement name")
    a = ap.parse_args()
    a0.STATEMENT_ID = a.statement
    out, served = Path(a.out), Path(a.served)
    out.mkdir(parents=True, exist_ok=True)
    rec = json.loads((served / "registration.json").read_text())
    n = int(rec["population"])
    law = D.parse_law(f"subset:{a.k}")
    template = InputSet.open(a.template_set).subcircuit.definition()
    program = P.population_program(template, n)
    query = P.template_instance_query(template)
    part = P.partition(program, query)
    if a.legacy_partition:
        part = {"program": part["program"], "query": {"name": query["name"], "version": query["version"], "template": P.template_id(template),
                                                      "width_rule": None}}
    summary: dict = {"n": n, "k": a.k, "delta": a.delta, "source": rec.get("window", {}).get("kind"), "n_from": "registration",
                     "partition_envelope": "pre-canonical {program, query}" if a.legacy_partition else "verity/partition/v1"}

    # the verifier's own circuit, bound to its own partition and program
    st_dir = out / "stage"
    st_dir.mkdir(parents=True, exist_ok=True)
    p = a0.run([sys.executable, str(HERE / "compose_circuit.py"), "--input-set", a.template_set, "--out", str(st_dir / "circuit.txt"),
                "--partition", P.digest(part), "--program", part["program"]], out / "compose.log", {"PYTHONPATH": a.m0_pythonpath})
    mine = json.loads(p.stdout.strip().splitlines()[-1])
    for name in (f"pub-{n}.bin", f"inst-{n}.bin"):
        target = st_dir / name
        if not target.exists():
            try:
                os.link((served / name).resolve(), target)
            except OSError:
                shutil.copyfile(served / name, target)
    shutil.copy(served / "index.json", out / "index.json")
    text = (st_dir / "circuit.txt").read_text()
    (st_dir / "partition.json").write_text(json.dumps(part, sort_keys=True))
    (st_dir / "program.json").write_bytes(canonical_json(encode_program(program)))
    st = {"dir": st_dir, "meta": json.loads(text.split("\n", 2)[1][5:]), "pin": mine["pin"], "partition_file": st_dir / "partition.json",
          "program_file": st_dir / "program.json"}
    header = a0.public_header(st_dir, n)
    fv = header["frame_v3"]

    # registration: R1-R6 against the verifier's own copies, the served domains recomputed
    run_id = rec.get("window", {}).get("run", "")
    want = {port: {"leaves": leaves, "domain": R.window_domain(part["program"], P.digest(part), port, schema, leaves, run_id)}
            for port, (schema, leaves) in a0.ports(st["meta"], n).items()}
    leaf_layer = a0.sha512_file(st_dir / f"pub-{n}.bin")
    checks = {"registration": V.check(rec, program, query, law, a0.SCHEME, want, leaf_layer=leaf_layer, partition=part).to_json(),
              "public": V.matches_public(rec, header["scheme"], fv["roots"], fv["domain_ids"], header["instances"]).to_json(),
              "circuit": {"ok": header.get("circuit_sha512") == mine["pin"], "codes": [] if header.get("circuit_sha512") == mine["pin"] else ["pin"],
                          "detail": {"public": header.get("circuit_sha512"), "verifier": mine["pin"]}},
              "bindings": a0.bindings(st["meta"], header, rec, mine.get("class_sha512", ""))}
    if not checks["registration"]["ok"] or rec.get("window", {}).get("kind") != "served":
        checks["source"] = {"ok": rec.get("window", {}).get("kind") == "served", "codes": [], "detail": rec.get("window")}
    receipt = R.receipt(rec, "one-stage-e2e verifier (a2.py)")
    (out / "receipt.json").write_text(json.dumps(receipt, indent=1, sort_keys=True))
    summary.update(registration=R.digest(rec), partition=rec["partition"], checks_before_draw={k: v["ok"] for k, v in checks.items()})
    if not all(v["ok"] for v in checks.values()):
        summary["checks"] = checks
        (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True))
        print("A2 " + json.dumps({"refused_before_draw": summary["checks_before_draw"]}))
        return 1

    # the sample, the session, the verifiers, the audit record
    if a.unbind:
        st, header = unbound_copy(st, header, n, a, out)
        summary["unbound"] = {"pin": st["pin"], "rule": "leaf layer byte for byte serving's; the statement no longer binds the partition"}
    t = time.time()
    record, live = a0.session(a.bin, st, n, f"subset:{a.k}", a.port, out / "session")
    summary["session_s"] = round(time.time() - t, 1)
    result = a0.audit_session(rec, receipt, st, header, law, record, live, a.lean, a.lean, out / "session", a.delta)
    result["checks_before_draw"] = checks
    result["n_from"] = "registration (the served index); the verifier does not yet derive N from #101's program"
    (out / "audit.json").write_text(json.dumps(result, indent=1, sort_keys=True))

    neg: dict[str, dict] = {}
    body = (st_dir / f"pub-{n}.bin").read_bytes()
    nl = body.index(b"\n") + 1
    flipped = body[:nl] + bytes([body[nl] ^ 1]) + body[nl + 1:]
    import hashlib
    v = V.check(rec, program, query, law, a0.SCHEME, want, leaf_layer=hashlib.sha512(flipped).hexdigest(), partition=part)
    neg["public-file-b||c-byte-flipped (R4)"] = {"refused": not v.ok, "codes": list(v.codes)}
    v = V.check(rec, program, query, law, a0.SCHEME,
                {p_: {**w, "domain": R.window_domain(part["program"], P.digest(part), p_, s_, w["leaves"], run_id + "-other")}
                 for p_, w, s_ in ((p_, w, a0.ports(st["meta"], n)[p_][0]) for p_, w in want.items())}, leaf_layer=leaf_layer,
                partition=part)
    neg["domains-of-another-run (R3)"] = {"refused": not v.ok, "codes": list(v.codes)}
    v = V.check(rec, R2.program([(template, n)], mixed=False), query, law, a0.SCHEME, want, leaf_layer=leaf_layer, partition=part)
    neg["registration-partition-mismatch (R2)"] = {"refused": not v.ok, "codes": list(v.codes)}
    draw = (record or {}).get("unit_draw") or {"law": "subset", "population": n, "k": a.k, "units": list(range(a.k))}
    v = V.check_draw({**draw, "units": draw["units"][:-1] + [n]}, rec, law)
    neg["draw-outside-the-partition"] = {"refused": not v.ok, "codes": list(v.codes)}
    v = V.check(dict(rec, leaf_layer=None), program, query, law, a0.SCHEME, want, leaf_layer=leaf_layer, partition=part)
    neg["registration-null-leaf-layer (R4)"] = {"refused": not v.ok, "codes": list(v.codes)}
    if "registration" in result:
        neg.update(a0.audit_negatives(result, law, a.delta, (live or {}).get("roots") or {}))
    (out / "negatives.json").write_text(json.dumps(neg, indent=1, sort_keys=True))

    summary["audit"] = {k: result.get(k) for k in ("accepted", "complete", "wrong_units", "checks")}
    summary["audit"]["verdicts"] = {k: v_.get("accepted") for k, v_ in result.get("verdicts", {}).items()}
    summary["timing"] = {k: (live or {}).get(k) for k in ("e2e_s", "witness_s", "prove_total_s", "k_log", "units", "proof_bytes")}
    summary["negatives"] = {k: v_["refused"] for k, v_ in neg.items()}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True))
    ok = bool(result.get("accepted")) and all(summary["negatives"].values())
    print("A2 " + json.dumps(summary, sort_keys=True))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
