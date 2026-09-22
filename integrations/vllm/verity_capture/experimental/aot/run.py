"""Harness: ENCODE (φ + record -> h), DECODE (φ + h -> P′ in a directory containing ONLY the declared inputs), CHECK (P′ vs P).

    python -m verity_capture.experimental.aot.run encode  --record <run dir> --profile <profile module> --out <dir>
        writes <dir>/phi.json, <dir>/h.json (the JSON interchange C-codec compresses to `a`), <dir>/encode_report.json
    python -m verity_capture.experimental.aot.run decode  --phi phi.json --hints h.json --out program.json [--workdir <dir>]
        copies φ and h into a fresh (or empty) work directory, asserts it holds nothing else, chdirs there, and builds with
        file reads outside that directory refused (`_jail`); records the input hashes in the output
    python -m verity_capture.experimental.aot.run check   --built program.json --accepted <program.json | fold_summary.json> [--out check.json]
        digest equality (`verity.ir.codec.program_digest`) or the exact structural diff (`check.py`); a fold_summary.json
        gives digest-only evidence (the accepted descriptor is not in the small archives)
    python -m verity_capture.experimental.aot.run roundtrip --record <run dir> --profile <module> --out <dir> [--accepted <program.json>]
        the three in sequence; <dir>/roundtrip.json is the one-line-per-run evidence record

Exit code 0 only when the check verdict is IDENTICAL (digest) / PREFIX-EQUAL; 3 for Unexpressible; 2 for a mismatch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from .build import BUILDER_ID, Unexpressible, build_program, max_rss_gb
from .check import check_descriptors
from .encode import encode_hints, observe_schedule
from .hints import Hints
from .phi import load_phi, phi_from_log, save_phi
from .predict import PREDICTOR_ID, schedule_summary

PROGRAM_SCHEMA = "verity-gen/program/v1"      # same top-level layout as run_config.persist_program's program.json


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _generator_identity() -> dict:
    """The identity of G as loaded: sha256 over (module name, sha256 of its source) for every imported `verity_capture` /
    `verity.ir` module, sorted by name.  Content-based, so it is meaningful on shipped trees without `.git`; the profile module
    and the registry definitions the build touched are included because the build imported them."""
    items: list[tuple[str, str]] = []
    for name, mod in sorted(sys.modules.items()):
        if not (name == "verity_capture" or name == "verity.ir" or name.startswith(("verity_capture.", "verity.ir."))):
            continue
        f = getattr(mod, "__file__", None)
        if not f or not f.endswith(".py"):
            continue
        try:
            items.append((name, hashlib.sha256(Path(f).read_bytes()).hexdigest()))
        except OSError:
            continue
    h = hashlib.sha256()
    for name, d in items:
        h.update(f"{name}\0{d}\n".encode())
    # tree identity: every .py under the two packages, imported or not — one id per shipped tree, independent of which profile /
    # registry modules a particular φ made the build import (the per-import sha above differs between model families).
    t = hashlib.sha256()
    n_files = 0
    for pkg in ("verity_capture", "verity.ir"):
        mod = sys.modules.get(pkg)
        root = Path(getattr(mod, "__file__", "")).parent if mod else None
        if not root or not root.is_dir():
            continue
        for f in sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts):
            t.update(f"{pkg}/{f.relative_to(root).as_posix()}\0".encode() + hashlib.sha256(f.read_bytes()).digest())
            n_files += 1
    return {"modules": len(items), "sha256": h.hexdigest(), "tree_files": n_files, "tree_sha256": t.hexdigest(),
            "builder": BUILDER_ID, "predictor": PREDICTOR_ID}


def _rev() -> str:
    """The code revision: `git rev-parse HEAD` in a checkout, else the shipped tree's EXPORT.json commit (pods have no .git)."""
    root = Path(__file__).resolve().parents[3]
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:  # noqa: BLE001
        pass
    exp = root / "EXPORT.json"
    if exp.exists():
        try:
            d = json.loads(exp.read_text())
            return f"{d.get('commit', 'unknown')} (shipped; dirty_patch_bytes={d.get('dirty_patch_bytes')})"
        except Exception:  # noqa: BLE001
            pass
    return "unknown"


def _record_log(record: Path) -> Path:
    for cand in (record / "capture" / "log.jsonl.gz", record / "capture" / "log.jsonl", record / "log.jsonl.gz"):
        if cand.exists():
            return cand
    raise SystemExit(f"{record}: no capture log")


# ---------------------------------------------------------------------------------------------------------------- encode

def cmd_encode(a) -> int:
    record = Path(a.record)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log = _record_log(record)
    profile = a.profile or json.loads((record / "run.json").read_text()).get("profile")
    if not profile:
        raise SystemExit("--profile is required (run.json has none)")
    t0 = time.time()
    phi = phi_from_log(log, profile)
    # explicit baseline facts (--engine key=value): pre-execution engine facts the RunHeader does not record (e.g. num_gpu_blocks,
    # the KV capacity the V1 scheduler's preemption depends on).  They become part of φ (phi.json, φ digest) and are listed here.
    baseline: dict[str, int | float | str | bool] = {}
    for kv in getattr(a, "engine", None) or []:
        k, _, v = kv.partition("=")
        if not k or not v:
            raise SystemExit(f"--engine expects key=value, got {kv!r}")
        baseline[k] = json.loads(v) if v[0] in "-0123456789tf[{\"" else v
    phi.engine.update(baseline)
    observed = observe_schedule(log, phi)
    predictor = getattr(a, "predictor", None) or PREDICTOR_ID
    try:
        hints, rep = encode_hints(phi, observed, predictor=predictor)
        status = "ok"
    except RuntimeError as e:
        hints, rep, status = Hints(predictor=predictor), {"error": str(e)}, "inexpressible-schedule"
    save_phi(phi, out / "phi.json")
    (out / "h.json").write_text(json.dumps(hints.to_json(), indent=1, sort_keys=True) + "\n")
    reqs = phi.logical_requests()
    doc = {"schema": "verity-aot/encode-report/v1", "status": status, "record": str(record), "log": str(log), "log_sha256": _sha(log),
           "profile_module": profile, "profile_id": phi.profile_id, "phi_sha256": phi.digest(), "hints_sha256": hints.digest(),
           "predictor": predictor, "observed": schedule_summary(observed.plans), "run_ended": observed.ended, "engine_baseline": baseline,
           "hints_naive_bits": hints.naive_bits(max_tokens=max((int(r.get("max_tokens") or 0) for r in reqs), default=1), n_requests=len(reqs),
                                                max_position=max((len(r["prompt_token_ids"]) + int(r.get("max_tokens") or 0) for r in reqs), default=1)),
           "hints_json_bytes": len(hints.canonical()), "report": rep, "seconds": round(time.time() - t0, 3), "code_rev": _rev()}
    (out / "encode_report.json").write_text(json.dumps(doc, indent=1) + "\n")
    print(f"[encode] {record.name}: {status}; observed {doc['observed']}; hints empty={hints.empty} naive_bits={doc['hints_naive_bits']} "
          f"json_bytes={doc['hints_json_bytes']} -> {out}")
    return 0 if status == "ok" else 3


# ---------------------------------------------------------------------------------------------------------------- decode

class _Jail:
    """Refuse file reads outside `root` for the duration of the build (accidental access to the record is a bug, not a hint)."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.violations: list[str] = []

    def __enter__(self):
        import builtins
        import io
        self._open, self._io_open = builtins.open, io.open
        root = self.root

        def guarded(file, mode="r", *args, **kw):
            if isinstance(file, (str, bytes, os.PathLike)) and any(c in str(mode) for c in "r+") and "w" not in str(mode):
                p = Path(os.fsdecode(file)).resolve() if not isinstance(file, Path) else file.resolve()
                if root not in p.parents and p != root and "site-packages" not in str(p) and not str(p).startswith(sys.prefix):
                    self.violations.append(str(p))
                    raise PermissionError(f"decode jail: read of {p} outside {root}")
            return self._open(file, mode, *args, **kw)

        builtins.open = guarded
        io.open = guarded
        return self

    def __exit__(self, *exc):
        import builtins
        import io
        builtins.open, io.open = self._open, self._io_open
        return False


def cmd_decode(a) -> int:
    phi_src, h_src = Path(a.phi).resolve(), Path(a.hints).resolve()
    out = Path(a.out).resolve()
    work = Path(a.workdir).resolve() if a.workdir else Path(tempfile.mkdtemp(prefix="aot-decode-"))
    work.mkdir(parents=True, exist_ok=True)
    if any(work.iterdir()):
        raise SystemExit(f"decode work directory {work} is not empty")
    shutil.copyfile(phi_src, work / "phi.json")
    shutil.copyfile(h_src, work / "h.json")
    listing = sorted(p.name for p in work.iterdir())
    if listing != ["h.json", "phi.json"]:
        raise SystemExit(f"decode directory holds {listing}; only phi.json and h.json are allowed")
    inputs = {"phi.json": _sha(work / "phi.json"), "h.json": _sha(work / "h.json")}
    cwd = os.getcwd()
    os.chdir(work)
    t0 = time.time()
    try:
        with _Jail(work) as jail:
            phi = load_phi(work / "phi.json")
            hints = Hints.from_json(json.loads((work / "h.json").read_text()))
            try:
                res = build_program(phi, hints, with_meta=False)
            except Unexpressible as e:
                doc = {"schema": PROGRAM_SCHEMA, "status": "unexpressible", "reason": str(e), "inputs": inputs, "builder": BUILDER_ID,
                       "workdir": str(work), "code_rev": _rev()}
                out.write_text(json.dumps(doc, indent=1) + "\n")
                print(f"[decode] UNEXPRESSIBLE: {e}")
                return 3
        doc = {"schema": PROGRAM_SCHEMA, "status": "ok", "digest": res.digest, "n_instances": res.n_instances,
               "n_input_nodes": len(res.program.root.body.nodes) - res.n_instances, "profile": phi.profile["module"],
               "profile_id": phi.profile_id, "workload_digest": phi.workload_digest(), "inputs": inputs, "workdir": str(work),
               "workdir_listing": listing, "jail_violations": jail.violations, "builder": BUILDER_ID,
               "predictor": hints.predictor,        # the predictor the hints name is the one build_program ran (PREDICTOR_ID is only the default)
               "schedule": schedule_summary(res.plans), "params": [[n, repr(t)] for n, t in res.params], "ret_type": repr(res.ret_type),
               "timings": res.timings, "max_rss_gb": max_rss_gb(), "descriptor_bytes": len(json.dumps(res.desc, sort_keys=True, separators=(",", ":"))),
               "code_rev": _rev(), "generator": _generator_identity(), "program": res.desc}
    finally:
        os.chdir(cwd)
    out.write_text(json.dumps(doc, separators=(",", ":")) + "\n")
    print(f"[decode] digest={res.digest[:16]} instances={res.n_instances} steps={doc['schedule']['steps']} "
          f"time={res.timings['total_s']}s rss={doc['max_rss_gb']}GB bytes={doc['descriptor_bytes']} -> {out} (work {work})")
    return 0


# ---------------------------------------------------------------------------------------------------------------- check

def _load_built(path: Path) -> dict:
    d = json.loads(path.read_text())
    if d.get("status") == "unexpressible":
        raise SystemExit(f"built program is unexpressible: {d.get('reason')}")
    return d


def cmd_check(a) -> int:
    built = _load_built(Path(a.built))
    acc_path = Path(a.accepted)
    acc = json.loads(acc_path.read_text())
    rep: dict = {"schema": "verity-aot/check/v1", "built": str(a.built), "accepted": str(acc_path), "accepted_sha256": _sha(acc_path),
                 "built_digest": built.get("digest"), "code_rev": _rev()}
    if "program" in acc and isinstance(acc["program"], dict) and "definitions" in acc["program"]:
        r = check_descriptors(built["program"], acc["program"], prefix_nodes=a.prefix_nodes)
        rep.update(r)
        rep["evidence"] = "descriptor (full structural comparison)"
    elif "digest" in acc:
        rep["digest_accepted"] = acc["digest"]
        rep["digest_built"] = built.get("digest")
        rep["digest_equal"] = acc["digest"] == built.get("digest")
        rep["verdict"] = "IDENTICAL (digest)" if rep["digest_equal"] else "DIFFERS (digest only; accepted descriptor unavailable for a structural diff)"
        rep["evidence"] = "digest only (fold_summary.json)"
        for k in ("n_instances", "n_steps"):
            if k in acc:
                rep[f"accepted_{k}"] = acc[k]
        rep["built_n_instances"] = built.get("n_instances")
    else:
        raise SystemExit(f"{acc_path}: neither a program.json nor a fold_summary.json")
    if a.out:
        Path(a.out).write_text(json.dumps(rep, indent=1) + "\n")
    print(f"[check] {rep['verdict']}  built={str(rep.get('digest_built'))[:16]} accepted={str(rep.get('digest_accepted'))[:16]}")
    ok = rep["verdict"].startswith(("IDENTICAL", "PREFIX-EQUAL"))
    return 0 if ok else 2


# ---------------------------------------------------------------------------------------------------------------- roundtrip

def cmd_roundtrip(a) -> int:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    record = Path(a.record)
    rc_e = cmd_encode(argparse.Namespace(record=a.record, profile=a.profile, out=str(out), predictor=getattr(a, "predictor", None),
                                         engine=getattr(a, "engine", None)))
    enc = json.loads((out / "encode_report.json").read_text())
    work = out / "decode-work"
    if work.exists():
        shutil.rmtree(work)
    rc_d = cmd_decode(argparse.Namespace(phi=str(out / "phi.json"), hints=str(out / "h.json"), out=str(out / "program.json"), workdir=str(work)))
    summary = {"schema": "verity-aot/roundtrip/v1", "record": record.name, "encode_status": enc["status"], "encode_rc": rc_e, "decode_rc": rc_d,
               "hints_empty": enc.get("report", {}).get("hints_empty"), "hints_naive_bits": enc["hints_naive_bits"], "hints_json_bytes": enc["hints_json_bytes"],
               "observed": enc["observed"], "code_rev": _rev()}
    if rc_d == 0:
        acc = Path(a.accepted) if a.accepted else None
        if acc is None:
            for cand in (record / "program.json", record / "fold_summary.json"):
                if cand.exists():
                    acc = cand
                    break
        if acc is not None:
            rc_c = cmd_check(argparse.Namespace(built=str(out / "program.json"), accepted=str(acc), out=str(out / "check.json"), prefix_nodes=None))
            chk = json.loads((out / "check.json").read_text())
            summary.update({"check_rc": rc_c, "verdict": chk["verdict"], "evidence": chk["evidence"], "digest_built": chk.get("digest_built"),
                            "digest_accepted": chk.get("digest_accepted")})
        else:
            summary.update({"check_rc": None, "verdict": "NO-ACCEPTED-REFERENCE"})
        built = json.loads((out / "program.json").read_text())
        summary.update({"n_instances": built["n_instances"], "timings": built["timings"], "max_rss_gb": built["max_rss_gb"],
                        "descriptor_bytes": built["descriptor_bytes"], "schedule": built["schedule"]})
    else:
        built = json.loads((out / "program.json").read_text())
        summary.update({"verdict": "UNEXPRESSIBLE", "reason": built.get("reason")})
    (out / "roundtrip.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(f"[roundtrip] {record.name}: {summary.get('verdict')}  hints_empty={summary['hints_empty']} bits={summary['hints_naive_bits']}")
    return 0 if summary.get("check_rc") == 0 else (rc_d or rc_e or 2)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="verity_capture.experimental.aot.run", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("encode"); e.add_argument("--record", required=True); e.add_argument("--profile"); e.add_argument("--out", required=True)
    e.add_argument("--predictor", help="predictor id whose residual to encode (default: the current one; predict.PREDICTORS)")
    e.add_argument("--engine", action="append", metavar="KEY=VALUE", help="explicit baseline engine fact added to φ (e.g. num_gpu_blocks=20)")
    d = sub.add_parser("decode"); d.add_argument("--phi", required=True); d.add_argument("--hints", required=True); d.add_argument("--out", required=True); d.add_argument("--workdir")
    c = sub.add_parser("check"); c.add_argument("--built", required=True); c.add_argument("--accepted", required=True); c.add_argument("--out"); c.add_argument("--prefix-nodes", type=int, dest="prefix_nodes")
    r = sub.add_parser("roundtrip"); r.add_argument("--record", required=True); r.add_argument("--profile"); r.add_argument("--out", required=True); r.add_argument("--accepted")
    r.add_argument("--predictor")
    r.add_argument("--engine", action="append", metavar="KEY=VALUE")
    a = ap.parse_args(argv)
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 200000))
    # The IR's reference graphs are deep (Concat chains over thousands of K/V parts): build, encode and — notably — their
    # deallocation recurse deeply.  Run in a thread with a 1 GiB stack and free everything there (PHI3 b1 c256 crashed at
    # interpreter teardown on the main thread after a correct decode: 2026-09-18).
    import gc
    import threading
    out: list[int] = [1]
    exc: list[BaseException] = []

    def body():
        try:
            out[0] = {"encode": cmd_encode, "decode": cmd_decode, "check": cmd_check, "roundtrip": cmd_roundtrip}[a.cmd](a)
        except SystemExit as e:
            out[0] = int(e.code) if isinstance(e.code, int) else (0 if e.code is None else 1)
            if not isinstance(e.code, int) and e.code is not None:
                print(e.code, file=sys.stderr)
        except BaseException as e:  # noqa: BLE001
            exc.append(e)
        finally:
            gc.collect()

    threading.stack_size(1 << 30)
    t = threading.Thread(target=body, name="aot")
    t.start()
    t.join()
    if exc:
        raise exc[0]
    sys.stdout.flush()
    sys.stderr.flush()
    return out[0]


if __name__ == "__main__":
    rc = main()
    os._exit(rc)      # skip the main-thread interpreter teardown of whatever remains (module-level caches are shallow; the deep graphs died in the worker)
