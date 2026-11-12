"""`verity bench overhead`'s node side: the operating points' sweep and a scheme's timed run, each `bench.py` runs of the
PoUW harness (`benchmarks/pouw/harness`) on the one GPU the run's lease holds. `overhead_node.sh` runs it in the node's
Python with torch and vLLM, under the `research run` that `overhead_records` launches; this side writes `$OUT` and the run's
`result.json` (its measurements through the kit's writer, under the kit's host stamp), and the store side reads them.

    python -m verity_bench.overhead_node points --card h200 --uuid GPU-... --out DIR --run-dir DIR --run-id RUN
    python -m verity_bench.overhead_node run --card h200 --scheme pearl-plus --points FILE --point C|L|all \\
        --uuid GPU-... --out DIR --run-dir DIR --run-id RUN --commit SHA --scratch DIR [--nsys PATH] [--points-art ART] \\
        [--cubin FILE --cubin-art ART]

`points` times every linear of the model at each m `overhead.next_ms` asks for (`CHUNK` m's a `bench.py` process, every
scaling of the precision, graph replays), until both brackets close, and writes `points.json`: C, L, t(m) and each linear's
fastest kernel at every m, each round's process, and the card's identity and timed clocks.

`run` times the scheme's arm beside its family's stock finalists and the fastest one's twin (`--twins`) at the stored points,
in one `bench.py` process under Nsight Systems, the arm loading the cubin the run was sent (`$VERITY_BENCH_CUBIN`, which
the SASS gate must pass): the timed pass untraced, then at each shape a traced pass of the same
entries that nsys captures alone (SMs Active at 10 kHz, each item an NVTX range; `--traced`). It then runs the arm's
reference verifier on every row (`verify.py --every-row`), reads each traced item's busy SM-time from the reports
(`nsys.report`, the reports kept in `--scratch`, outside `$OUT`; a report that collected past its traced pass, or a traced
item no report holds, refuses the run), and writes `overhead.json`: per point, the layer's γ over
the fastest stock arm by wall time, beside it over the least-SM-time arm, with the weight side as a bound, the A/A control,
the cross-check, ε from the scheme's function at the same point and card, and why each point's γ isn't of record. A kernel
point runs no model, so it has no perplexity; nor does it serve requests, so it has no load counts to keep on the node.
Beside the arm it records whether the arm's timed call commits y (`arm.commits_y`, the scheme's entry), which the table's
note reads.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from verity_bench import kit, nsys
from verity_bench import overhead as O

HARNESS = Path("benchmarks/pouw/harness")
#: the traced pass alone (`bench.py --traced` brackets it by cudaProfilerStart and Stop, once a shape): one report a shape
NSYS_PROFILE = ("profile", "--trace=cuda,nvtx", "--cuda-graph-trace=node", "--gpu-metrics-devices=cuda-visible",
                "--gpu-metrics-frequency=10000", "--capture-range=cudaProfilerApi", "--capture-range-end=repeat",
                "--force-overwrite=true")
#: the m's one sweep process times, at every linear
CHUNK = 4
#: the sweep's bench.py: wall time is the graph replays' (the served form)
POINTS_BENCH = ("--methods", "graph", "--decode-method", "independent")
#: a timed run's: the graph method gives each stock kernel's wall time, the burst method the host's enqueue, and the traced
#: pass beside them each item's busy SM-time
RUN_BENCH = ("--methods", "graph", "burst", "--decode-method", "independent", "--traced", "--twins")
#: the cubin the run was sent (`run --cubin ART`), which the scheme's arm loads and lists in its build record's `binaries`
CUBIN_ENV = "VERITY_BENCH_CUBIN"

Runner = Callable[[list[str], Path], int]


def _run(argv: list[str], log: Path) -> int:
    with log.open("w") as f:
        return subprocess.run(argv, stdout=f, stderr=subprocess.STDOUT).returncode


def bench_argv(card: O.Gpu, uuid: str, shapes: Sequence[str], families: Sequence[str], out: Path,
               extra: Sequence[str] = ()) -> list[str]:
    return [sys.executable, str(HARNESS / "bench.py"), "--gpu", "lease", "--expect-uuid", uuid, "--shapes", *shapes,
            "--families", *families, *(["--bridge"] if card.bridge else []), *extra, "--out", str(out)]


def model_shapes(model: str, ms: Sequence[int]) -> list[str]:
    return list(dict.fromkeys(O.label(m, n, k) for m in ms for n, k in O.MODELS[model].values()))


def _card_why(card: O.Gpu, result: dict) -> str | None:
    got = ((result.get("identity") or {}).get("card") or {}).get("name")
    if got is None or card.name not in got:
        return f"the run's card is {got!r}, not the {card.name} the door was asked to measure"
    return None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, doc: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1, default=str) + "\n")
    os.replace(tmp, path)


def write_result(run_dir: Path, run_id: str, out: Path, record: str, ok: bool, detail: str, fingerprint: dict,
                 measurements: Sequence[dict] = (), *, measured: Sequence[dict] = ()) -> None:
    """The run's `result.json`: its validation, then each of `measured` (the kit writer's arguments) through the kit's
    writer, which stamps it with the run, the commit and the host. `measurements` are entries a caller wrote whole, held
    as given: serving cost's (`benchmarks/serve_tax`), until it writes them through the kit's writer."""

    rel = os.path.relpath(out / record, run_dir)
    _write(run_dir / kit.RESULT, {
        "schema": kit.RESULT_SCHEMA, "run_id": run_id, "workload_fingerprint": fingerprint,
        "validation": {"status": "passed" if ok else "failed", "checker": "verity bench overhead", "detail": detail,
                       "evidence": [rel]},
        "measurements": list(measurements), "measurement_files": [rel], "artifacts": []})
    if measured:
        writer = kit.Writer(run_dir, run_id, kit.host_stamp())
        for m in measured:
            writer.measurement(**m, inputs=[rel])


# ---- points ------------------------------------------------------------------------------------------------------------------


def points(a: argparse.Namespace, run: Runner = _run) -> int:
    card = O.card(a.card)
    fams = card.precisions.get(a.precision)
    if fams is None:
        raise ValueError(f"the {card.name} has no {a.precision} in the door ({', '.join(card.precisions)})")
    out = a.out
    times: dict[int, float] = {}
    results: list[dict] = []
    rounds: list[dict] = []
    doc: dict = {"schema": O.POINTS_SCHEMA, "card": card.gpu, "model": a.model, "precision": a.precision,
                 "families": list(fams), "rule": dict(O.RULE), "rounds": rounds}
    fingerprint = {"door": "points", "card": card.gpu, "model": a.model, "precision": a.precision}
    why = None
    while why is None and (todo := O.next_ms(times, a.model)):
        for i in range(0, len(todo), CHUNK):
            ms = todo[i:i + CHUNK]
            d = out / "rounds" / f"{len(rounds):02d}"
            d.mkdir(parents=True, exist_ok=True)
            rc = run(bench_argv(card, a.uuid, model_shapes(a.model, ms), fams, d / "bench.json", POINTS_BENCH),
                     d / "bench.log")
            res = json.loads((d / "bench.json").read_text()) if (d / "bench.json").is_file() else {}
            rounds.append({"ms": ms, "bench": str((d / "bench.json").relative_to(out)), "exit": rc,
                           "error": res.get("error"), "clocks": (res.get("clocks") or {}).get("timed")})
            why = (f"bench.py exited {rc} at m = {ms}: {res.get('error') or 'see ' + str(d / 'bench.log')}" if rc
                   else _card_why(card, res) or O.libraries_why(card, res))
            if why:
                break
            doc.setdefault("identity", res.get("identity"))
            results.append(res)
        if why is None:
            try:
                times, chosen = O.sweep_times(results, a.model)
            except ValueError as ex:
                why = str(ex)
    if why is None:
        try:
            p = O.final(times, a.model)
        except ValueError as ex:
            why = str(ex)
    if why is None:
        doc |= {"c": p.c, "l": p.l, "best_flop_per_ms": p.best_flop_per_ms,
                "times_ms": {str(m): t for m, t in sorted(times.items())},
                "tflops": {str(m): O.layer_flop(a.model, m) / t / 1e9 for m, t in sorted(times.items())},
                "linears": {str(m): v for m, v in sorted(chosen.items())}}
    doc["error"] = why
    _write(out / "points.json", doc)
    said = why or f"C = {doc.get('c')}, L = {doc.get('l')}"
    write_result(a.run_dir, a.run_id, out, "points.json", why is None, said, fingerprint, measured=[] if why else [
        {"name": "C", "value": doc["c"], "unit": "m"}, {"name": "L", "value": doc["l"], "unit": "m"}])
    print(f"points: {said}")
    return 0 if why is None else 1


# ---- run ---------------------------------------------------------------------------------------------------------------------


def epsilon_at(sch: O.Scheme, card: O.Gpu, m: int, model: str) -> dict:
    """The scheme's ε at each linear of the layer at m on the card, from its own function; or why there is none."""

    if sch.epsilon is None:
        return {"why": f"{sch.id} has no ε: it is the A/A control"}
    mod, _, name = sch.epsilon.partition(":")
    try:
        if sch.arm_path and sch.arm_path not in sys.path:
            sys.path.insert(0, sch.arm_path)
        fn = getattr(importlib.import_module(mod), name)
        return {linear: fn(m, n, k, card.gpu) for linear, (n, k) in O.MODELS[model].items()}
    except Exception as ex:  # noqa: BLE001 - recorded beside γ: a missing ε is shown, never invented
        return {"why": f"{sch.epsilon}: {type(ex).__name__}: {ex}"}


def verified(result: dict, arm: str | None, m: int, model: str) -> dict:
    """Each linear's verdict from the arm's reference verifier (`verify.py --every-row`), as the row carries it."""

    if arm is None:
        return {}
    rows = {r["label"]: r for r in result.get("shapes", [])}
    out = {}
    for linear, (n, k) in O.MODELS[model].items():
        r = ((rows.get(O.label(m, n, k)) or {}).get("arms") or {}).get(arm) or {}
        v = (r.get("panel") or r.get("verified") or {}).get("verification")
        out[linear] = v or {"accepted": False, "why": "no verdict: the reference verifier didn't run on this row"}
    return out


def run_scheme(a: argparse.Namespace, run: Runner = _run) -> int:
    card, sch = O.card(a.card), O.scheme(a.scheme)
    sent = a.points if a.points.is_file() else a.run_dir / "inputs" / a.points
    pts = json.loads(sent.read_text())
    cubin = None
    if sch.arm and a.cubin:
        cubin = (a.cubin if a.cubin.is_file() else a.run_dir / "inputs" / a.cubin).absolute()
    out, scratch = a.out, a.scratch
    fingerprint = {"door": "run", "card": card.gpu, "scheme": sch.id, "point": a.point, "points_art": a.points_art,
                   "points_sha256": _sha256(sent), "cubin_art": a.cubin_art if cubin else None}
    doc: dict = {"schema": O.RUN_SCHEMA, "scheme": sch.id, "family": sch.family, "card": card.gpu,
                 "model": pts.get("model"), "points_art": a.points_art, "points_sha256": fingerprint["points_sha256"],
                 "cpu_sm_s_per_core_s": card.cpu_sm_s_per_core_s, "points": {}, "whys": []}
    whys = doc["whys"]
    if pts.get("schema") != O.POINTS_SCHEMA or pts.get("card") != card.gpu or pts.get("error") or None in (
            pts.get("c"), pts.get("l")):
        whys.append(f"{a.points} isn't a finished {O.POINTS_SCHEMA} sweep of the {card.name}")
    elif sch.family not in pts.get("families", ()):
        whys.append(f"the points' sweep didn't time {sch.family}, the scheme's scaling")
    if whys:
        _write(out / "overhead.json", doc)
        write_result(a.run_dir, a.run_id, out, "overhead.json", False, whys[0], fingerprint)
        print(f"run: {whys[0]}")
        return 1
    model = pts["model"]
    at = {p: pts[p.lower()] for p in (O.POINTS if a.point == "all" else (a.point,))}
    arm_args = ["--arm", sch.arm, "--arm-path", sch.arm_path] if sch.arm else []
    prof = scratch / "prof"
    bench_json = out / "bench.json"
    rc = run([*(["env", f"{CUBIN_ENV}={cubin}"] if cubin else []), a.nsys, *NSYS_PROFILE, "-o", str(prof),
              *bench_argv(card, a.uuid, model_shapes(model, list(at.values())), [sch.family], bench_json,
                          [*RUN_BENCH, *arm_args])], out / "bench.log")
    result = json.loads(bench_json.read_text()) if bench_json.is_file() else {}
    doc |= {"bench_exit": rc, "identity": result.get("identity"), "clocks": result.get("clocks"),
            "bridge": result.get("bridge"), "sass_gate_ok": (result.get("sass_gate") or {}).get("ok")}
    if rc:
        whys.append(f"bench.py exited {rc}: {result.get('error') or 'see bench.log'}")
    if why := _card_why(card, result):
        whys.append(why)
    if bench_json.is_file() and (why := O.libraries_why(card, result)):
        whys.append(why)
    arm = next(iter(result.get("arms") or {}), None) if sch.arm else None
    if sch.arm:
        rec = (result.get("arms") or {}).get(arm) or {}
        build = rec.get("build") or {}
        gated = [{k: b.get(k) for k in ("path", "sha256", "ok")}
                 for b in (result.get("sass_gate") or {}).get("binaries") or []]
        sha = _sha256(cubin) if cubin else None
        doc["arm"] = {"name": arm, "cubin_art": a.cubin_art if cubin else None, "cubin_sha256": sha,
                      "binaries": build.get("binaries"), "gated": gated, "verifier": rec.get("verifier"),
                      "commits_y": sch.commits_y}
        if not str(doc["arm"]["cubin_art"] or "").startswith("art:"):
            whys.append("the run was sent no cubin art (`run --cubin ART`): a cubin built in the run is development only")
        elif bench_json.is_file() and sha not in {b["sha256"] for b in gated if b["ok"]}:
            whys.append(f"no binary the SASS gate passed is the cubin the run was sent ({a.cubin_art}): the arm must load "
                        f"${CUBIN_ENV} and list it in its binaries")
    if bench_json.is_file() and result.get("items"):
        busy: dict = {}
        reports = sorted(scratch.glob(prof.name + "*.nsys-rep"))
        if not reports:
            whys.append("nsys wrote no report: the traced pass ran no capture range")
        exports = []
        for i, report in enumerate(reports):
            sqlite = report.with_name(report.name.removesuffix(".nsys-rep") + ".sqlite")
            erc = run([a.nsys, "export", "--type", "sqlite", "--force-overwrite", "true", "-o", str(sqlite), str(report)],
                      out / f"export-{i:02d}.log")
            exports.append({"report": report.name, "exit": erc})
            if erc:
                whys.append(f"nsys export of {report.name} exited {erc}")
                continue
            try:
                got = nsys.report(sqlite, card, "item ")
            except ValueError as ex:
                whys.append(f"{report.name} has no busy SM-time: {ex}")
                continue
            busy |= got["busy"]
            exports[-1] |= {"items": got["ranges"], "stray_kernels": got["stray"]}
            if got["ranges"] and got["stray"] > got["ranges"]:
                whys.append(f"{report.name} traced {got['stray']} kernels outside its {got['ranges']} traced items, more "
                            "than a flush each: its capture range ran past the traced pass, so a timed pass may have run "
                            "traced")
        traced = len(result.get("traced_items") or [])
        if missing := [i for i in range(traced) if f"item {i}" not in busy]:
            whys.append(f"nsys recorded {traced - len(missing)} of the {traced} traced items (item {missing[0]} the first "
                        "missing): a capture range opened after its items")
        doc["reports"], doc["traced_prime"] = exports, result.get("traced_prime")
        _write(out / "busy.json", busy)
        if sch.arm:
            vrc = run([sys.executable, str(HARNESS / "verify.py"), str(bench_json), "--every-row", "--no-ledger",
                       "--run-id", a.run_id, *(["--commit", a.commit] if a.commit else [])], out / "verify.log")
            doc["verify_exit"] = vrc
            result = json.loads(bench_json.read_text())
        for name, m in at.items():
            p = O.layer_gamma(result, busy, card, m, sch, model)
            p["verified"] = verified(result, arm, m, model)
            p["epsilon"] = epsilon_at(sch, card, m, model)
            p["perplexity"] = {"value": None, "why": "a kernel point runs no model: perplexity is a served point's"}
            g = (p.get("gamma") or {}).get("gamma")
            p["gate"] = None if g is None else {"cap": O.GAMMA_CAP, "passes": g <= O.GAMMA_CAP}
            p["whys"] = O.point_whys(p)
            doc["points"][name] = p
    _write(out / "overhead.json", doc)
    gammas = {n: (p.get("gamma") or {}).get("gamma") for n, p in doc["points"].items()}
    ok = not rc and bool(doc["points"]) and None not in gammas.values()
    write_result(a.run_dir, a.run_id, out, "overhead.json", ok,
                 "; ".join(f"γ at {n} {g:.4f}" if g is not None else f"no γ at {n}" for n, g in gammas.items())
                 or (whys[0] if whys else "nothing timed"),
                 fingerprint, measured=[{"name": f"gamma_{n}", "value": gammas[n], "unit": "ratio",
                                         "interval": p["gamma"].get("ci95")}
                                        for n, p in doc["points"].items() if gammas[n] is not None])
    for n, p in doc["points"].items():
        stands = "of record" if not (p["whys"] or whys) else "not of record: " + "; ".join(p["whys"] + whys)
        print(f"run: {sch.id} at {n} (m = {p['m']}): γ {gammas[n]}; {stands}")
    for why in whys if not doc["points"] else ():
        print(f"run: {why}")
    return 0 if ok else 1


def parse(argv: Sequence[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m verity_bench.overhead_node", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("points", "run"):
        s = sub.add_parser(name)
        s.add_argument("--card", required=True, choices=list(O.CARDS))
        s.add_argument("--uuid", required=True, help="the leased GPU's UUID, which bench.py must find")
        s.add_argument("--out", type=Path, required=True)
        s.add_argument("--run-dir", type=Path, required=True)
        s.add_argument("--run-id", required=True)
        if name == "points":
            s.add_argument("--model", default=O.KERNEL_MODEL, choices=list(O.MODELS))
            s.add_argument("--precision", default="fp8")
        else:
            s.add_argument("--scheme", required=True, choices=list(O.SCHEMES))
            s.add_argument("--points", type=Path, required=True,
                           help="the stored points.json (a relative path not found here: the run's inputs/, where --send puts it)")
            s.add_argument("--points-art", help="the art it was fetched from")
            s.add_argument("--cubin", type=Path, help=f"the cubin the arm loads (${CUBIN_ENV}), as sent (a relative path "
                                                      "not found here: the run's inputs/)")
            s.add_argument("--cubin-art", help="the art it was fetched from")
            s.add_argument("--point", choices=[*O.POINTS, "all"], default="all")
            s.add_argument("--commit", help="the run's source commit (the arm's verifier runs it)")
            s.add_argument("--scratch", type=Path, required=True, help="where the nsys report goes, outside --out")
            s.add_argument("--nsys", default="nsys")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None, run: Runner = _run) -> int:
    a = parse(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    return points(a, run) if a.command == "points" else run_scheme(a, run)


if __name__ == "__main__":
    sys.exit(main())
