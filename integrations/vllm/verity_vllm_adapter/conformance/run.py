"""Orchestration and CLI of the configuration-conformance harness.

    python -m verity_vllm_adapter.conformance validate  --port b1-eager
    python -m verity_vllm_adapter.conformance bindings  --port b1-eager                # regenerate bindings.json
    python -m verity_vllm_adapter.conformance run       --port b1-eager --out DIR
            [--from-results DIR ...] [--correspondence FILE|DIR ...] [--obligation FILE ...] [--proof-record DIR ...]
            [--int-record DIR ...]                                   # C5: INT `run_composed` output directories
            [--pod-stage production|replay|all --pod-tag TAG]      # GPU stages through pod/run_conformance.sh
            [--commit STORE --checkpoint SHARD ... --request R --positions T ... --layers L ...]   # live C2 / Level 2
            [--full-descriptor] [--expect-level N|C<n>|L<n>] [--claim N|C<n>|L<n>]
    python -m verity_vllm_adapter.conformance reproduce --port b1-eager [--out DIR] [--all]
    python -m verity_vllm_adapter.conformance matrix    [--results fixtures/results] [--out docs/vllm-poc/conformance-matrix.md]

Every rung is decided from what `evidence.py` / `typed.py` / `ladder.py` found (C1–C4 by `level.py` under their
public names); the harness never claims above the evidence, and `--claim C<n>` (or `L<n>`, or the bare level
number) makes it exit non-zero when that rung is not met. `level.json` and `ladder.json` are both written.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from . import OUTPUT_FILES
from . import evidence as E
from . import ladder as LD
from . import level as LV
from . import typed as T
from .port import Port, PortError, list_ports, load_port, profiles_dir, repo_root, write_bindings
from .registry import REPRODUCE
from .report import write_outputs

POD_SCRIPT = Path(__file__).resolve().parents[1] / "pod" / "run_conformance.sh"


def utc_stamp() -> str:
    return time.strftime("%Y%m%dT%H%MZ", time.gmtime())


def _abs(p: str | Path) -> Path:
    q = Path(p)
    return q if q.is_absolute() else (repo_root() / q if (repo_root() / q).exists() else Path.cwd() / q)


def _program_doc(port: Port, full_descriptor: bool, out: Path) -> dict[str, Any]:
    if not port.level2_claimed:
        return {"status": "not-claimed", "reason": port.level2_reason, "bindings": port.bindings_document()["bindings"],
                "static_config": port.static_config_record()}
    from verity.ir import descriptor_size, encode_program, program_digest  # noqa: PLC0415

    probe = dict(getattr(port.static_config, "PROBE_SHAPE", {"LP": 2, "STEPS": 1}))
    t0 = time.time()
    prog = port.program(**probe)
    desc = encode_program(prog)
    doc: dict[str, Any] = {"status": "constructed", "function": prog.fn.id, "probe_shape": probe, "gates": prog.gates,
                           "input_gates": prog.input_gates, "descriptor_bytes": descriptor_size(desc), "program_digest": program_digest(desc),
                           "definitions": len(desc["definitions"]), "build_seconds": round(time.time() - t0, 2),
                           "config_digest": port.config_digest(), "static_config": port.static_config_record(),
                           "bindings": port.bindings_document()["bindings"], "capture_mapping": port.mapping_record()}
    # the descriptor itself is reproducible from the port (program_digest names it); it is written only on request
    # so a results directory stays small enough to commit (B1's probe-shape descriptor is ~2.8 MB)
    if full_descriptor:
        (out / "program_descriptor.json").write_text(json.dumps(desc, sort_keys=True, separators=(",", ":")))
        doc["descriptor_file"] = "program_descriptor.json"
    else:
        doc["descriptor_file"] = None
        doc["descriptor_note"] = "not written (pass --full-descriptor); program_digest identifies it and descriptor_stats.json carries its byte profile"
    return doc


def _description(dirs: list[E.ResultsDir]) -> dict | None:
    for rd in dirs:
        if rd.matches_port and rd.commit is not None and (rd.commit / "description.json").exists():
            return E._json(rd.commit / "description.json")
    return None


def _collect_correspondence(paths: list[str], dirs: list[E.ResultsDir]) -> list[Path]:
    files: list[Path] = []
    for p in paths:
        q = _abs(p)
        if q.is_dir():
            files += sorted(q.glob("*.json"))
        elif q.exists():
            files.append(q)
        else:
            raise FileNotFoundError(f"correspondence {p} does not exist")
    for rd in dirs:
        for c in rd.correspondence:
            if c not in files:
                files.append(c)
    return files


def run_port(port: Port, out: Path, *, from_results: list[str], correspondence: list[str], obligations: list[str],
             proof_records: list[str], full_descriptor: bool = False, argv: list[str] | None = None,
             live: dict[str, Any] | None = None, int_records: list[str] | None = None) -> dict[str, Any]:
    """Assemble every rung from the supplied artefacts and write the standard outputs into `out`.

    Returns the `level.json` document (as before; HP/TP read `level`/`summary`/`levels` from it) with the C0–C5
    verdict attached under `"ladder"` (the same object written to `ladder.json`)."""
    out.mkdir(parents=True, exist_ok=True)
    dirs = [rd for p in from_results for rd in E.discover_all(_abs(p), port)]
    for rd in dirs:
        if not rd.matches_port:
            print(f"[conformance] {rd.path}: does not match port {port.name}: {rd.manifest_violations or 'no profile manifest'}", file=sys.stderr)
    production = E.production_repro(dirs)
    replay = E.replay_repro(dirs)
    corr_files = _collect_correspondence(correspondence, dirs)
    if live:
        corr_files.append(T.live_run(port, _abs(live["commit"]), [_abs(c) for c in live["checkpoint"]], live["request"], live["positions"],
                                     live["layers"], out / "correspondence" / f"corr-{live['request'][:8]}.json", live.get("coord_samples", 32),
                                     live.get("full"), live.get("seed", 0), live.get("cos_sin_npy"), live.get("cos_sin_cpu_npy")))
    corr = T.correspond(port, corr_files, dirs, _description(dirs))
    program = _program_doc(port, full_descriptor, out)
    descriptor = T.descriptor_stats(port, corr["files"], full=full_descriptor) if port.level2_claimed else {"status": "not-claimed", "reason": port.level2_reason}
    l3 = E.level3_evidence(dirs, port)
    l4 = E.level4_evidence([_abs(o) for o in obligations], [_abs(p) for p in proof_records], dirs)
    results = [LV.level1(production, replay), LV.level2(corr, port.level2_claimed), LV.level3(l3), LV.level4(l4)]
    if not port.level2_claimed:
        results[1] = LV.LevelResult(2, False, "not-claimed", port.level2_reason, [])
    stamp = utc_stamp()
    level = LV.verdict(port.name, results, utc=stamp, run_dir=E.rel(out))
    level["inputs"] = {"results": [rd.as_json() for rd in dirs]}
    # the promotion ladder: C0 from the production manifests, C1–C4 = the levels, C5 from INT records, plus the taxonomy outcome
    c0 = LD.rung_c0(production, dirs, port)
    c5 = LD.rung_c5([_abs(p) for p in (int_records or [])], port)
    rungs = {"C0": c0, "C5": c5, **{LD.RUNG_OF_LEVEL[r.level]: LD.from_level(r) for r in results}}
    outcome, unsupported = LD.unsupported_outcome(port, corr, rungs)
    ladder = LD.ladder(port.name, level, c0, c5, outcome, unsupported, utc=stamp, run_dir=E.rel(out), axis=port.config.axis,
                       declared_blockers=port.expectations.blockers)
    inputs = {"results": [E.rel(rd.path) for rd in dirs], "results_matching_port": [E.rel(rd.path) for rd in dirs if rd.matches_port],
              "correspondence": [E.rel(c) for c in corr_files], "obligations": obligations, "proof_records": proof_records,
              "int_records": list(int_records or [])}
    corr_doc = {"status": corr["status"], "reason": corr["reason"], "port": port.name, "capture_mapping": port.mapping_record(),
                "files": corr["files"], "summary": corr["summary"], "records": corr["records"], "mismatches": len(corr["mismatches"]),
                "expectations": corr["expectations"], "level3": l3, "level4": l4}
    write_outputs(out, port_doc=port.as_json(), production=production, replay=replay, program=program, correspondence=corr_doc,
                  missing=corr["missing"], mismatches=corr["mismatches"], descriptor=descriptor, level=level, inputs=inputs,
                  argv=argv or sys.argv, ladder=ladder)
    level["ladder"] = ladder
    return level


def shell_flags(port: Port) -> str:
    """The port's declared flags as `run_case.sh` arguments and environment exports (one assignment per line)."""
    c = port.config
    prod = []
    if c.execution_mode == "eager":
        prod.append("--enforce-eager")
    co = (c.engine_args.get("compilation_config") or {}).get("custom_ops")
    if co:
        prod += ["--custom-ops", co[0] if isinstance(co, list) else str(co)]
    if c.engine_args.get("max_num_batched_tokens"):
        prod += ["--max-num-batched-tokens", str(c.engine_args["max_num_batched_tokens"])]
    if c.tensor_parallel_size > 1:
        prod += ["--tensor-parallel-size", str(c.tensor_parallel_size)]
    if c.quantization:  # [QZ, additive] a quantised port (b1-fp8) declares its scheme as a production flag
        prod += ["--quantization", str(c.quantization)]
    lines = [f"CONF_PORT={port.name}", f"CONF_CASE={c.checkpoint_role}", f"CONF_REPO={c.model_repo}",
             f"CONF_EXECUTION_MODE={c.execution_mode}", f"CONF_REPLAY_EXEC={c.replay_exec}",
             f"CONF_TP={c.tensor_parallel_size}", f"CONF_PER_RANK={'1' if c.capture_per_rank else '0'}",
             f"CONF_GPU_GENERATION={c.hardware.gpu_generation}", f"CONF_CC='{' '.join(c.hardware.compute_capability)}'",
             f"CONF_PRODUCTION_ARGS='{' '.join(prod)}'",
             f"CONF_REPLAY_ARGS='--replay-exec {c.replay_exec}'",
             "CONF_ENV='" + " ".join(f"{k}={v}" for k, v in c.environment_flags.items()) + "'"]
    return "\n".join(lines)


def pod_stage(port: Port, stage: str, tag: str, extra: list[str]) -> Path:
    """Run the GPU stages through `pod/run_conformance.sh` (pods only) and return the pod results directory."""
    if not POD_SCRIPT.exists():
        raise FileNotFoundError(POD_SCRIPT)
    cmd = ["bash", str(POD_SCRIPT), port.name, tag, stage, *extra]
    print("[conformance] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    return Path(os.environ.get("VERITY_RESULTS_ROOT", "/workspace/results")) / port.config.checkpoint_role / tag


# ---------------------------------------------------------------------------------------------------------
# matrix
# ---------------------------------------------------------------------------------------------------------


def collect_levels(results_dir: Path) -> list[dict[str, Any]]:
    """Every `CONF-*/level.json` (kept: the pre-ladder reader; HP/TP's runs write only this file)."""
    rows = []
    for d in sorted(results_dir.glob("CONF-*")):
        lv = d / "level.json"
        if lv.exists():
            doc = json.loads(lv.read_text())
            doc["_dir"] = d.name
            rows.append(doc)
    return rows


def collect_ladders(results_dir: Path) -> list[dict[str, Any]]:
    """Every run as a ladder: `CONF-*/ladder.json` when the run wrote one, else its `level.json` lifted by
    `ladder.ladder_from_level` (C0/C5 `not-attempted`: the run predates the ladder)."""
    rows = []
    for d in sorted(results_dir.glob("CONF-*")):
        ld, lv = d / "ladder.json", d / "level.json"
        if ld.exists():
            doc = json.loads(ld.read_text())
            LD.ensure_blockers(doc, _port_blockers(str(doc.get("port"))))
        elif lv.exists():
            lvl = json.loads(lv.read_text())
            doc = LD.ladder_from_level(lvl, axis=_port_axis(str(lvl.get("port"))), declared_blockers=_port_blockers(str(lvl.get("port"))))
        else:
            continue
        doc["_dir"] = d.name
        rows.append(doc)
    return rows


def as_ladder(row: dict[str, Any]) -> dict[str, Any]:
    """A matrix row in either shape (`ladder.json` or `level.json` document) as a ladder document."""
    if "rungs" in row:
        return LD.ensure_blockers(row, _port_blockers(str(row.get("port"))))
    out = LD.ladder_from_level(row, axis=_port_axis(str(row.get("port"))), declared_blockers=_port_blockers(str(row.get("port"))))
    out["_dir"] = row.get("_dir")
    return out


def latest_per_port(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for r in sorted(rows, key=lambda r: (r.get("port", ""), r.get("utc") or "", r.get("_dir") or "")):
        latest[r["port"]] = r
    return latest


def _port_blockers(port: str) -> list[dict[str, Any]]:
    """The port's declared blockers (`expectations.json` `blockers`), read directly like `_port_axis`."""
    p = profiles_dir() / port / "expectations.json"
    if not p.exists():
        return []
    try:
        v = json.loads(p.read_text()).get("blockers")
    except json.JSONDecodeError:
        return []
    return [b for b in v if isinstance(b, dict) and b.get("rung") and b.get("class") in LD.BLOCKERS] if isinstance(v, list) else []


def _port_axis(port: str) -> str | None:
    """The port's declared axis, read directly (the matrix must render even when a port fails to load)."""
    p = profiles_dir() / port / "vllm_config.json"
    if not p.exists():
        return None
    try:
        v = json.loads(p.read_text()).get("axis")
    except json.JSONDecodeError:
        return None
    return v if isinstance(v, str) and v.strip() else None


_MATRIX_POINTER_KINDS: dict[str, tuple[str, ...]] = {
    "C0": ("production-manifest",), "C1": ("production", "replay-commit"), "C2": ("correspondence",),
    "C3": ("table1-row",), "C4": ("proof-record",), "C5": ("int-record",),
}


def _first_pointer(ladder_doc: dict[str, Any], rung: str, kinds: tuple[str, ...]) -> str:
    """The first evidence pointer of one of `kinds` that supports the claim (a negative record — a proof
    record whose statement did not match, a replay with problems — is evidence in `ladder.json`, not a pointer
    the matrix cites for a met rung)."""
    for e in ((ladder_doc.get("rungs") or {}).get(rung, {}).get("evidence") or []):
        if e.get("kind") in kinds and e.get("path"):
            if e.get("matches_obligation") is False or e.get("problems") or e.get("status") in ("not-reproducible", "incomplete", "not-verified"):
                continue
            return e["path"]
    return ""


def render_matrix(rows: list[dict[str, Any]], ports: list[str]) -> str:
    """`docs/vllm-poc/conformance-matrix.md`: `Profile | axis | C0 | C1 | C2 | C3 | C4 | C5 | Main unsupported reason | Blocker | run`.
    `rows` are ladder documents (`collect_ladders`) or level documents (`collect_levels`; lifted here)."""
    latest = latest_per_port([as_ladder(r) for r in rows])
    L = ["# Configuration-conformance matrix", "",
         "Generated by `python -m verity_vllm_adapter.conformance matrix` from `fixtures/results/CONF-*/ladder.json` (the latest run per "
         "port; a run that wrote only `level.json` is lifted with C0/C5 not-attempted). Do not edit by hand. The promotion ladder "
         "(`docs/vllm-poc/coverage-wave-2026-09-07.md` Part 3/4): **C0** runs · **C1** replay · **C2** representation · **C3** protocol · "
         "**C4** proof · **C5** vertical slice; C1–C4 are the former Levels 1–4. Rungs are independent claims: **met** cells cite evidence; "
         "every other cell says why not (not-attempted / not-claimed / not-met / absent). The headline (in the Profile cell) is the highest "
         "rung met; gaps below it are named. `axis` is the one architecture question the profile answers. \"Main unsupported reason\" is the "
         "closed taxonomy (`verity_vllm_adapter/conformance/taxonomy.py`): `supported-exactly`, one of the four `unsupported:*` reasons "
         "with the function/value named, or `unclassified` (C2 pending, not an unsupported outcome). \"Blocker\" is *why the profile "
         "did not advance* (GUIDANCE §24): the class of the first rung above the headline that is not met, from the closed enum "
         "`SEMANTICS_MISSING` / `REPRESENTATION_MISSING` / `CAPTURE_MISSING` / `AUTHENTICATION_MISSING` / `PROOF_LOWERING_MISSING` / "
         "`IMPLEMENTATION_BUG` / `NOT_ATTEMPTED` (`ladder.py` docstring: declared by the port in `expectations.json`, else derived from "
         "the rung's status and the taxonomy outcome); a fixed implementation bug stays visible as history (`↺`). How to read a port: "
         "`docs/vllm-poc/config-conformance.md`.", "",
         "| Profile | axis | C0 runs | C1 replay | C2 representation | C3 protocol | C4 proof | C5 vertical slice | Main unsupported reason | Blocker (next rung) | run |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for p in ports:
        r = latest.get(p)
        axis = _short((r or {}).get("axis") or _port_axis(p) or "—", 120)
        if r is None:
            L.append(f"| `{p}` | {axis} | not run | not run | not run | not run | not run | not run | — | `NOT_ATTEMPTED` | — |")
            continue
        cells = []
        for rung in LD.RUNGS:
            rec = r["rungs"][rung]
            if rec["met"]:
                ptr = _first_pointer(r, rung, _MATRIX_POINTER_KINDS[rung])
                cells.append(f"**met** — `{ptr}`" if ptr else "**met**")
            else:
                cells.append((f"`{rec['blocker']['class']}` " if rec.get("blocker") else "") + f"{rec['status']}" + (f": {_short(rec.get('reason'))}" if rec.get("reason") else ""))
        un = r.get("unsupported")
        outcome = str(r.get("outcome") or "unclassified")
        if un and un.get("reason"):
            reason_cell = f"`{un['reason']}` — {_short(un.get('detail'), 120)} (at {un.get('rung')})"
        elif outcome.startswith("supported-exactly"):
            reason_cell = f"`supported-exactly`" + (f" ({_short(outcome.split(';', 1)[1].strip(), 80)})" if ";" in outcome else "")
        else:
            reason_cell = f"`unclassified`" + (f": {_short(r['rungs']['C2'].get('reason'), 100)}" if r["rungs"]["C2"].get("reason") else "")
        b = r.get("blocker")
        if b:
            blocker_cell = f"{b['rung']}: `{b['class']}` — {_short(b.get('detail'), 100)}"
        elif r.get("reached") == "C5" and r.get("gaps"):
            # the headline is the top rung, so there is no *next* rung; the cell names the first gap below it instead
            g = r["gaps"][0]
            gb = (r["rungs"][g].get("blocker") or {})
            blocker_cell = f"gap {g}: `{gb.get('class', 'NOT_ATTEMPTED')}` — {_short(gb.get('detail') or r['rungs'][g].get('reason'), 100)}"
        elif r.get("reached") == "C5":
            blocker_cell = "`none` (every rung met)"
        else:
            blocker_cell = "—"
        for h in r.get("blocker_history") or []:
            blocker_cell += f"<br>↺ {h['rung']} was `{h['class']}` — {_short(h.get('detail'), 80)}" + (f" (resolved: {_short(h.get('resolved'), 60)})" if h.get("resolved") else "")
        L.append(f"| `{p}` — **{r['summary']}** | {axis} | " + " | ".join(cells) + f" | {reason_cell} | {blocker_cell} | `{r['_dir']}` |")
    L.append("")
    L += ["Evidence pointers are repository-relative. `met` for C0 names the production run's profile manifest whose execution choices "
          "are recorded (`ladder.json` lists them: backends, compile mode, dtype, TP/PP, GPU+driver, wheel, checkpoint digest, runtime); "
          "C1 names a production run and the commit whose replay reproduced it (`production_repro.json` / `replay_repro.json`); C2 the "
          "correspondence file (`correspondence.json`, `mismatches.jsonl`, `missing_values.json`); C3 the Table-1 row whose §10 verdict was "
          "recomputed (`ladder.json` also cites the verifier report and the re-hashed bundle digests); C4 the host-verified proof record "
          "(the obligation is cited too); C5 the INT `run_composed` record re-verified by a second process.", ""]
    return "\n".join(L)


def _short(s: Any, n: int = 140) -> str:
    s = str(s).replace("|", "\\|").replace("\n", " ")
    return s if len(s) <= n else s[: n - 1] + "…"


# ---------------------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------------------


def _live_from_args(a: argparse.Namespace) -> dict[str, Any] | None:
    if not a.commit:
        return None
    for k in ("checkpoint", "request", "positions"):
        if not getattr(a, k):
            raise SystemExit(f"--commit needs --{k}")
    return {"commit": a.commit, "checkpoint": a.checkpoint, "request": a.request, "positions": a.positions, "layers": a.layers or [0],
            "coord_samples": a.coord_samples, "full": {tuple(int(x) for x in s.split(",")) for s in a.full}, "seed": a.seed,
            "cos_sin_npy": _abs(a.cos_sin_npy) if a.cos_sin_npy else None, "cos_sin_cpu_npy": _abs(a.cos_sin_cpu_npy) if a.cos_sin_cpu_npy else None}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m verity_vllm_adapter.conformance", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate", help="load and validate a port (fail closed, names the field)")
    v.add_argument("--port", required=True)
    b = sub.add_parser("bindings", help="regenerate <port>/bindings.json from static_config.bindings()")
    b.add_argument("--port", required=True)
    fl = sub.add_parser("flags", help="print the port's declared flags as shell assignments (used by pod/run_conformance.sh)")
    fl.add_argument("--port", required=True)
    r = sub.add_parser("run", help="produce the standard outputs and the level verdict for a port")
    r.add_argument("--port", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--from-results", nargs="*", default=[], help="existing run dirs / bundles / commit stores")
    r.add_argument("--correspondence", nargs="*", default=[], help="verity-ir/correspondence/v0 files or directories")
    r.add_argument("--obligation", nargs="*", default=[], help="typed-obligation/v0 record(s) exported from this port's capture")
    r.add_argument("--proof-record", nargs="*", default=[], help="proof record directories (host verify*.json)")
    r.add_argument("--int-record", nargs="*", default=[], help="C5: INT `run_composed` output directories (manifest.json, summary.json, reverify/reverify.json)")
    r.add_argument("--pod-stage", choices=["production", "replay", "all"], default=None, help="run GPU stages via pod/run_conformance.sh first")
    r.add_argument("--pod-tag", default=None)
    r.add_argument("--pod-extra", nargs="*", default=[], help="extra arguments passed to run_conformance.sh")
    r.add_argument("--commit", default=None, help="live Level 2: commit store")
    r.add_argument("--checkpoint", nargs="*", default=[], help="live Level 2: safetensors shard(s)")
    r.add_argument("--request", default=None)
    r.add_argument("--positions", type=int, nargs="*", default=[])
    r.add_argument("--layers", type=int, nargs="*", default=[])
    r.add_argument("--coord-samples", type=int, default=32)
    r.add_argument("--full", nargs="*", default=[])
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--cos-sin-npy", default=None)
    r.add_argument("--cos-sin-cpu-npy", default=None)
    r.add_argument("--full-descriptor", action="store_true", help="also lower the full-shape program and embed the descriptor")
    r.add_argument("--expect-level", default=None, help="exit 3 unless the headline rung equals this (C<n>, L<n> or the level number)")
    r.add_argument("--claim", default=None, help="exit 3 unless this rung is met (C<n>, L<n> or the level number; over-claims are refused)")
    p = sub.add_parser("reproduce", help="run --from-results on the committed artefacts of a port (registry.py), no GPU")
    p.add_argument("--port", default=None)
    p.add_argument("--all", action="store_true")
    p.add_argument("--out", default=None, help="output dir (default fixtures/results/CONF-<port>-<utc>)")
    p.add_argument("--full-descriptor", action="store_true")
    m = sub.add_parser("matrix", help="write docs/vllm-poc/conformance-matrix.md from fixtures/results/CONF-*/level.json")
    m.add_argument("--results", default=None)
    m.add_argument("--out", default=None)
    m.add_argument("--check", action="store_true", help="exit 1 if the file on disk differs from what would be written")
    a = ap.parse_args(argv)

    if a.cmd == "validate":
        try:
            port = load_port(a.port)
        except PortError as e:
            print(f"INVALID {a.port}: {e}")
            return 2
        print(json.dumps({"port": port.name, "valid": True, "axis": port.config.axis, "level2_claimed": port.level2_claimed, "level2_reason": port.level2_reason,
                          "level2_taxonomy": list(port.level2_taxonomy) if port.level2_taxonomy else None,
                          "unsupported_functions": len(port.expectations.unsupported_functions),
                          "mapping_implemented": port.mapping_implemented, "bindings": len(port.bindings), "config_digest": port.config_digest()}, indent=1))
        return 0
    if a.cmd == "bindings":
        port = load_port(a.port, check_generated=False)
        print(write_bindings(port))
        return 0
    if a.cmd == "flags":
        port = load_port(a.port)
        print(shell_flags(port))
        return 0
    if a.cmd == "run":
        port = load_port(a.port)
        from_results = list(a.from_results)
        if a.pod_stage:
            from_results.append(str(pod_stage(port, a.pod_stage, a.pod_tag or utc_stamp(), a.pod_extra)))
        out = Path(a.out)
        level = run_port(port, out, from_results=from_results, correspondence=a.correspondence, obligations=a.obligation,
                         proof_records=a.proof_record, full_descriptor=a.full_descriptor, argv=sys.argv, live=_live_from_args(a),
                         int_records=a.int_record)
        ladder = level["ladder"]
        print(f"[conformance] {port.name}: {ladder['summary']} (level.json: {level['summary']}); outcome {ladder['outcome']} -> {out}")
        for c in ladder["claims"]:
            print("  " + c)
        if a.claim is not None:
            try:
                LD.assert_claim(ladder, a.claim)
            except (LD.OverClaim, ValueError) as e:
                print(f"REFUSED: {e}")
                return 3
        if a.expect_level is not None:
            try:
                want = LD.normalize_rung(a.expect_level)
            except ValueError as e:
                print(f"REFUSED: {e}")
                return 3
            if ladder["reached"] != want:
                print(f"EXPECTATION FAILED: headline {ladder['reached']} != {want}")
                return 3
        return 0
    if a.cmd == "reproduce":
        names = list(REPRODUCE) if a.all else ([a.port] if a.port else [])
        if not names:
            ap.error("reproduce needs --port or --all")
        rc = 0
        for name in names:
            spec = REPRODUCE.get(name)
            if spec is None:
                print(f"{name}: no committed artefacts registered (registry.py) — nothing to reproduce")
                rc = 2
                continue
            port = load_port(name)
            out = Path(a.out) if (a.out and not a.all) else repo_root() / "fixtures" / "results" / f"CONF-{name}-{utc_stamp()}"
            level = run_port(port, out, from_results=spec.get("results", []), correspondence=spec.get("correspondence", []),
                             obligations=spec.get("obligations", []), proof_records=spec.get("proof_records", []),
                             full_descriptor=a.full_descriptor, argv=sys.argv, int_records=spec.get("int_records", []))
            print(f"[conformance] {name}: {level['ladder']['summary']} (level.json: {level['summary']}) -> {E.rel(out)}")
        return rc
    if a.cmd == "matrix":
        results = Path(a.results) if a.results else repo_root() / "fixtures" / "results"
        out = Path(a.out) if a.out else repo_root() / "docs" / "vllm-poc" / "conformance-matrix.md"
        text = render_matrix(collect_ladders(results), list_ports())
        if a.check:
            if not out.exists() or out.read_text() != text:
                print(f"{out} is stale")
                return 1
            print(f"{out} up to date")
            return 0
        out.write_text(text)
        print(out)
        return 0
    return 0
