"""Fresh-seed FP4 rechecks beyond the gate, split so that no CPU verification runs inside a GPU lease (``tc_recheck_fp4``).

    python benchmarks/tc_probe_fp4/recheck.py capture --instruction nvf4 --seeds 20261101,20261102 --n-random 625000 \
        --prebuilt SO --expect-uuid "$GPU_LEASE_UUID" --out DIR --budget-s 360   # a fill chunk; exit 99 while seeds remain
    python benchmarks/tc_probe_fp4/recheck.py verify --instruction nvf4 --seed 20261101 --n-random 625000 --capture DIR \
        --prebuilt SO --out OUT                                                    # CPU (recorded run): the models, bit for bit

``capture``: the device's identity, the UUID gate, the SASS gate and probe.py's layout check, then for each seed the families of
probe.py's ``--sweep`` (the same generators and rng) through the tile kernel.  Each family's words go to
DIR/<instruction>/<seed>/<family>.npy beside <family>.json (the sha256 of its operands), the seed's done.json last, so a chunk
resumes at the first (seed, family) without a record; no family starts once ``--budget-s`` has passed (exit 99).
``verify``: regenerates the operands from the seed, checks their sha256 against the capture's, evaluates every model on them
(``probe.model_eval``) and counts each model's mismatches.  The gate: the pinned model (``PINNED``: nvf4 ``verity`` =
BLACKWELL_SM120_NVF4, f8f6f4 ``gs32_w26_native`` = BLACKWELL_SM120_E2M1_M16N8K32's parameters; ``--expect-model`` overrides)
reproduces every word of every gated family; record-only families are counted.  ``--fixed-mult`` multiplies the gated families
of fixed size (probe.py's and k32.py's ``fixed_mult``), so that every gated family reaches the element count a recheck asks for.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe as P
import k32
from verity.ml.circuits.silicon import models as M

from verity_config.settings import SHARED

INSTRUCTIONS = ("nvf4", "f8f6f4")
SCHEMA = "pouw-rtxpro/tc-recheck-fp4/v1"
#: the registry model each recheck gates on, by its name among probe.verity_models(instr)
PINNED = {"nvf4": "verity", "f8f6f4": "gs32_w26_native"}


def families(instr: str, seed: int, n_random: int, fixed_mult: int = 1):
    rng = np.random.default_rng(seed)
    if instr in k32.ROWS:
        return k32.families_k32(rng, n_random, instr, fixed_mult)
    return P.families_nvf4(rng, n_random, P.SCALE_KIND[instr], fixed_mult)


def operands_sha256(*arrays) -> str:
    h = hashlib.sha256()
    for a in arrays:
        h.update(b"-" if a is None else np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def primary_model(instr: str, expect: str | None) -> str:
    if expect:
        return expect
    if instr in k32.ROWS:
        c, m = k32.candidates(instr)[PINNED[instr]], M.BLACKWELL_SM120_E2M1_M16N8K32
        if (c.fmt_a, c.fmt_b, tuple(c.groups), c.width, c.zero_exponent, c.e2m1_as_e4m3, c.scale) != \
                ("e2m1", "e2m1", tuple(m.groups), m.width, m.zero_exponent, False, None):
            raise SystemExit(f"{PINNED[instr]} no longer has {m.name}'s parameters")
    return PINNED[instr]


def write_json(path: Path, doc: dict) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, indent=1))
    os.replace(tmp, path)


def capture(args) -> int:
    t_all = time.time()
    root = Path(os.path.expandvars(args.out)) / args.instruction
    root.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",") if s]
    todo = [s for s in seeds if not (root / str(s) / "done.json").exists()]
    if not todo:
        print("RECHECK CAPTURE: every seed recorded", flush=True)
        return 0
    if args.fake_silicon:
        so, dev = None, P.FakeProbe(args.instruction, args.fake_silicon)
        identity = {**dev.identity(), "backend": "fake"}
    else:
        so = Path(os.path.expandvars(args.prebuilt))
        dev = P.Probe(so)
        identity = dev.identity()
    uuid = identity.get("gpu_uuid")
    expect = os.path.expandvars(args.expect_uuid) if args.expect_uuid else None
    if not args.fake_silicon and not (expect and P.uuid_allowed(uuid, expect)):
        print(f"FAILURES device UUID {uuid} != expected {expect}: stopped before any launch", flush=True)
        return 2
    head = {"schema": SCHEMA, "instruction": args.instruction, "instruction_id": P.INSTRUCTION_ID[args.instruction],
            "n_random": args.n_random, "fixed_mult": args.fixed_mult, "identity": identity, "expect_uuid": expect,
            "fake_silicon": args.fake_silicon,
            "library": None if so is None else {"path": str(so), "sha256": hashlib.sha256(so.read_bytes()).hexdigest()}}
    if so is not None:
        sass = P.sass_check(so)
        bad = [k for k, r in sass.items() if r["mma_count"] != 1] + ([] if P.KERNEL[args.instruction] in sass else ["missing"])
        if bad:
            print("FAILURES sass", bad, flush=True)
            return 1
        head["sass"] = {P.KERNEL[args.instruction]: sass[P.KERNEL[args.instruction]]["opcodes"]}
    head["layout_check"] = P.layout_check(dev, args.instruction)
    if head["layout_check"] != "passed":
        print("FAILURES layout check", head["layout_check"], flush=True)
        return 1
    captured = 0
    for seed in todo:
        d = root / str(seed)
        d.mkdir(exist_ok=True)
        fams = {}
        for name, A, Bt, SFA, SFB, C in families(args.instruction, seed, args.n_random, args.fixed_mult):
            rec_path = d / f"{name}.json"
            if rec_path.exists():
                fams[name] = json.loads(rec_path.read_text())
                continue
            if captured and time.time() - t_all > args.budget_s:
                print(f"RECHECK CAPTURE: budget reached after {captured} families; exit 99", flush=True)
                return 99
            t0 = time.time()
            D = dev.tiles(args.instruction, A, Bt, C, SFA, SFB)
            gpu_s = time.time() - t0
            np.save(d / f"{name}.tmp.npy", D)
            os.replace(d / f"{name}.tmp.npy", d / f"{name}.npy")
            rec = {"family": name, "tiles": int(A.shape[0]), "elements": int(A.shape[0]) * 128, "gpu_seconds": round(gpu_s, 3),
                   "operands_sha256": operands_sha256(A, Bt, SFA, SFB, C),
                   "words_sha256": hashlib.sha256((d / f"{name}.npy").read_bytes()).hexdigest()}
            write_json(rec_path, rec)
            fams[name] = rec
            captured += 1
            print(args.instruction, seed, name, rec["tiles"], "tiles", f"{gpu_s:.2f}s", flush=True)
        write_json(d / "done.json", {**head, "seed": seed, "families": fams, "clocks": dev.clocks(uuid)})
        print(f"RECHECK CAPTURE: {args.instruction} seed {seed} recorded ({len(fams)} families)", flush=True)
    return 0


def verify(args) -> int:
    t_all = time.time()
    out = Path(os.path.expandvars(args.out))
    out.mkdir(parents=True, exist_ok=True)
    d = Path(os.path.expandvars(args.capture)) / args.instruction / str(args.seed)
    failures: list[str] = []
    done = json.loads((d / "done.json").read_text())
    ident = done.get("identity") or {}
    primary = primary_model(args.instruction, args.expect_model)
    if done.get("fake_silicon") or ident.get("backend") == "fake":
        failures.append("the capture ran on fake silicon (a dry run's invented words)")
    if not done.get("expect_uuid") or not P.uuid_allowed(ident.get("gpu_uuid"), done["expect_uuid"]):
        failures.append(f"the capture's device {ident.get('gpu_uuid')} was not gated on its lease ({done.get('expect_uuid')})")
    for k, v in (("instruction", args.instruction), ("n_random", args.n_random), ("seed", args.seed), ("schema", SCHEMA),
                 ("fixed_mult", args.fixed_mult)):
        if done.get(k, 1 if k == "fixed_mult" else None) != v:
            failures.append(f"the capture's {k} {done.get(k)!r} != {v!r}")
    if args.prebuilt and hashlib.sha256(Path(os.path.expandvars(args.prebuilt)).read_bytes()).hexdigest() != \
            (done.get("library") or {}).get("sha256"):
        failures.append("the capture loaded another library than --prebuilt")
    fams: dict = {}
    if not failures:
        with mp.Pool(args.procs, initializer=P._init, initargs=(args.instruction, 16)) as pool:
            for name, A, Bt, SFA, SFB, C in families(args.instruction, args.seed, args.n_random, args.fixed_mult):
                rec = done["families"].get(name)
                if rec is None:
                    failures.append(f"{name}: not in the capture")
                    continue
                if operands_sha256(A, Bt, SFA, SFB, C) != rec["operands_sha256"]:
                    failures.append(f"{name}: the regenerated operands differ from the captured ones")
                    continue
                if hashlib.sha256((d / f"{name}.npy").read_bytes()).hexdigest() != rec["words_sha256"]:
                    failures.append(f"{name}: the words file changed since the capture")
                    continue
                D = np.load(d / f"{name}.npy")
                t0 = time.time()
                models, errs = P.model_eval(pool, A, Bt, SFA, SFB, C, args.procs)
                record_only = name in P.RECORD_ONLY_FAMILIES
                fam = {"tiles": rec["tiles"], "elements": rec["elements"], "record_only": record_only,
                       "mismatches_vs_model": {m: int(((w != D) | errs[m]).sum()) for m, w in models.items()},
                       "domain_errors": {m: int(e.sum()) for m, e in errs.items()}, "model_seconds": round(time.time() - t0, 1)}
                if primary not in models:
                    failures.append(f"{name}: no model {primary!r} ({sorted(models)})")
                elif not record_only and fam["mismatches_vs_model"][primary]:
                    t, i, j = next(zip(*np.nonzero((models[primary] != D) | errs[primary])))
                    fam["first_mismatch"] = {"tile": int(t), "i": int(i), "j": int(j), "device": f"{int(D[t, i, j]):08x}",
                                             "model": "error" if errs[primary][t, i, j] else f"{int(models[primary][t, i, j]):08x}"}
                    failures.append(f"{name}: {fam['mismatches_vs_model'][primary]} words differ from {primary}")
                fams[name] = fam
                print(name, json.dumps(fam), flush=True)
    gated = {k: f for k, f in fams.items() if not f["record_only"]}
    doc = {"schema": SCHEMA + "/verify", "instruction": args.instruction, "seed": args.seed, "n_random": args.n_random,
           "fixed_mult": args.fixed_mult, "primary_model": primary, "capture": {"dir": str(d), "done_sha256": hashlib.sha256((d / "done.json").read_bytes()).hexdigest()},
           "identity": ident, "families": fams, "gated_elements": sum(f["elements"] for f in gated.values()),
           "min_gated_family_elements": min((f["elements"] for f in gated.values()), default=0),
           "failures": failures, "seconds": round(time.time() - t_all, 1)}
    (out / "recheck.json").write_text(json.dumps(doc, indent=1))
    ok = not failures and bool(gated)
    if SHARED.RESEARCH_RUN_ID:
        res = {"schema": "research/result/v0.1", "kind": "tc-evidence/v1", "run_id": SHARED.need("RESEARCH_RUN_ID"),
               "instruction": P.INSTRUCTION_ID[args.instruction], "ptx": P.PTX[args.instruction], "sku": ident.get("device"),
               "gpu_uuid": ident.get("gpu_uuid"), "driver": ident.get("driver"), "compute_capability": ident.get("compute_capability"),
               "model_agrees": ok, "primary_model": primary,
               "counts": {"families": len(fams), "gated_elements": doc["gated_elements"],
                          "min_gated_family_elements": doc["min_gated_family_elements"]},
               "workload_fingerprint": {"tool": "tc_recheck_fp4", "instruction": args.instruction, "seed": args.seed,
                                        "n_random": args.n_random, "library_sha256": (done.get("library") or {}).get("sha256")},
               "validation": {"status": "passed" if ok else "failed",
                              "checker": f"a fill chunk's capture (UUID, SASS, layout check), its operands regenerated from the seed "
                                         f"and their sha256 checked, {primary} on every word of every gated family",
                              "detail": "; ".join(failures) or f"{len(gated)} gated families, {doc['gated_elements']} words, 0 differ",
                              "evidence": ["recheck.json"]},
               "measurements": [{"name": "gated_elements", "value": doc["gated_elements"], "unit": "words"}],
               "measurement_files": ["recheck.json"], "artifacts": []}
        write_json(out / "result.json", res)
    print("RECHECK VERIFY", "ok" if ok else "FAILED", failures[:5], f"{doc['seconds']}s", flush=True)
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture")
    c.add_argument("--instruction", required=True, choices=INSTRUCTIONS)
    c.add_argument("--seeds", required=True)
    c.add_argument("--n-random", type=int, required=True)
    c.add_argument("--fixed-mult", type=int, default=1)
    c.add_argument("--prebuilt", default=None)
    c.add_argument("--expect-uuid", default=None)
    c.add_argument("--budget-s", type=float, default=360.0)
    c.add_argument("--fake-silicon", default=None)
    c.add_argument("--out", required=True)
    v = sub.add_parser("verify")
    v.add_argument("--instruction", required=True, choices=INSTRUCTIONS)
    v.add_argument("--seed", type=int, required=True)
    v.add_argument("--n-random", type=int, required=True)
    v.add_argument("--fixed-mult", type=int, default=1)
    v.add_argument("--capture", required=True)
    v.add_argument("--prebuilt", default=None)
    v.add_argument("--expect-model", default=None)
    v.add_argument("--procs", type=int, default=max(1, min(32, (os.cpu_count() or 4) - 2)))
    v.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if getattr(args, "fake_silicon", None) and SHARED.RESEARCH_RUN_ID:
        ap.error("--fake-silicon is a dry run and never publishes: unset RESEARCH_RUN_ID")
    if args.cmd == "capture" and not args.fake_silicon and not args.prebuilt:
        ap.error("capture needs --prebuilt (compile off the lease)")
    return capture(args) if args.cmd == "capture" else verify(args)


if __name__ == "__main__":
    sys.exit(main())
