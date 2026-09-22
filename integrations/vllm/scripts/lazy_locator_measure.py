"""Measure the verifier's lazy program path (LZ, 2026-09-10): `LazyProgram` + streamed `program_digest` + `LazyProgramLocator` + `locate()`.

Two modes, both reporting per-stage wall seconds and max RSS (`resource.ru_maxrss`; wrap in `/usr/bin/time -l|-v` for the process peak):

* ``--archive DIR`` -- a real record (REC-P1-B0): `build_pinned_program` from the instance's `program.identity` (digest checked against the
  population pin), `make_locator`, `locate_selected` over EVERY row of the verifier's `chronology/selected.json`, `compare_units` against the
  prover's `mapping.units.jsonl`; with ``--compare-full`` the same over `Program(sp)` / `ProgramLocator` and a unit-by-unit comparison
  (identical `locate` for every selected VU is the equality the contract asks for).
* ``--case B0 --lp LP --steps STEPS --gates N`` -- no record: N uniformly random gates; for each, what the verifier would do for a selected
  VU there: `root_of`, the Gemm call and GemmCoordinate member containing it (`locate_gemm_coordinate` of that member) or `locate_input`
  for an Input gate.  ``--digest`` adds the streamed digest (MEASURED over the full structure; ``--expect-digest`` compares).

Launch control: run through ``zk/campaign/lane_run.py`` when a full build is requested (``--compare-full`` at 32x15 is ~2.4 GB; the lazy
path alone stays under 100 MB at 1024x511).
"""

from __future__ import annotations

import argparse
import json
import random
import resource
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
for p in (HERE.parent.parent, HERE.parent, HERE.parent.parent / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def rss_gb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1024 ** 3) if sys.platform == "darwin" else r / (1024 ** 2)


class Stages:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def mark(self, name: str, t0: float, **extra: Any) -> None:
        rec = {"stage": name, "seconds": round(time.perf_counter() - t0, 3), "max_rss_gb": round(rss_gb(), 3), **extra}
        self.rows.append(rec)
        print(json.dumps(rec), flush=True)


def config_of(case: str) -> dict[str, Any]:
    from verity_vllm.registry import b1
    if case == "TINY":
        from e2e.population import TINY
        return dict(TINY)
    return dict(getattr(b1, f"{case}_CONFIG"))


def member_containing(loc, g: int) -> dict[str, Any] | None:
    """What a selected VU at gate g would be located as: the GemmCoordinate member containing g (or None when g is in no Gemm member)."""
    name, root = loc.root_of(g)
    if root is None:
        return None
    for gm in loc.gemms_of(name, root):
        if gm.lo <= g < gm.hi:
            key = (name, loc.gemms_of(name, root).index(gm))
            if key not in loc._members:
                kids = list(gm.children())
                loc._members[key] = {(c.lo, c.hi): k for k, c in enumerate(kids) if c.name.startswith("GemmCoordinate")}, len(kids)
            for (lo, hi) in loc._members[key][0]:
                if lo <= g < hi:
                    return loc.locate_gemm_coordinate(lo, hi)
            return None
    return None


def run_shape(a: argparse.Namespace, st: Stages, out: dict[str, Any]) -> int:
    from verity.ir import bind
    from verity_vllm.ir.lazy import LazyProgram, lazy_program_digest
    from verity_vllm.registry import b1

    from e2e.lazy_locator import LazyProgramLocator

    C = config_of(a.case)
    t0 = time.perf_counter()
    sp = bind(b1.Serve, C=C, LP=a.lp, STEPS=a.steps)
    prog = LazyProgram(sp)
    out.update(function=sp.id, gates=prog.gates, input_gates=prog.input_gates, root_nodes=prog.layout.n_nodes)
    st.mark("LazyProgram", t0)
    rc = 0
    if a.digest:
        t0 = time.perf_counter()
        d = lazy_program_digest(prog)
        out["program_digest"] = d
        out["program_digest_how"] = "MEASURED: sha256 of verity_vllm.ir.lazy.encode_program_stream over the full structure (every definition encoded by codec._encode_definition, every root node by ServeLayout)"
        st.mark("streamed encode + sha256", t0)
        if a.expect_digest:
            out["digest_matches_expected"] = d == a.expect_digest
            print("digest", d, "expected", a.expect_digest, "EQUAL" if d == a.expect_digest else "DIFFERENT", flush=True)
            rc = 0 if d == a.expect_digest else 2
    t0 = time.perf_counter()
    loc = LazyProgramLocator(prog, C)
    st.mark("LazyProgramLocator", t0, weight_ranges=len(loc.w_ranges))
    if a.gates:
        rng = random.Random(a.seed)
        t0 = time.perf_counter()
        counts = {"gemm_member": 0, "input": 0, "other": 0, "refused": 0}
        roots: set[str] = set()
        for _ in range(a.gates):
            g = rng.randrange(prog.gates)
            if g < loc.input_gates:
                r = loc.locate_input(g)
                counts["input"] += 1
            else:
                r = member_containing(loc, g)
                if r is None:
                    counts["other"] += 1
                    name, _root = loc.root_of(g)
                    roots.add(name)
                    continue
                counts["gemm_member"] += 1
                roots.add(r.get("root", "?"))
            if isinstance(r, dict) and "refused" in r:
                counts["refused"] += 1
        dt = time.perf_counter() - t0
        out["random_gates"] = {"n": a.gates, "seconds": round(dt, 3), "ms_per_gate": round(1000 * dt / a.gates, 4), "counts": counts,
                               "distinct_roots_touched": len(roots), "seed": a.seed}
        st.mark(f"{a.gates} random gates located", t0)
    if a.check_dataflow:
        rng = random.Random(a.seed + 1)
        P = prog.layout.P
        pos = sorted({0, prog.layout.LP - 1, prog.layout.LP, P - 1} | {rng.randrange(P) for _ in range(a.check_dataflow)})
        t0 = time.perf_counter()
        errs = loc.check_dataflow(pos)
        out["check_dataflow"] = {"positions": len(pos), "violations": errs[:5]}
        st.mark(f"check_dataflow at {len(pos)} positions", t0)
        if errs:
            rc = rc or 3
    return rc


def run_archive(a: argparse.Namespace, st: Stages, out: dict[str, Any]) -> int:
    from e2e.verifier_v3 import locator as vloc

    rec = Path(a.archive)
    inst_doc = json.loads((rec / "protocol_instance.json").read_text())
    sel_doc = json.loads((rec / "chronology" / "selected.json").read_text())
    proto = json.loads((rec / "protocol.json").read_text())
    expr_path = Path(a.expressibility) if a.expressibility else HERE.parent.parent / "out" / "record" / "expressibility.json"
    expr = json.loads(expr_path.read_text())
    klass = {r["kind"]: r["class"] for r in expr["rows"]}
    ru_request = {r["ru"]: r["request_id"] for r in proto["ru_map"]["rus"]}
    n_v = int(expr["vus_per_request"])
    rows = sel_doc["selected"]
    out.update(archive=str(rec), identity=inst_doc["program"]["identity"], selected=len(rows), program_digest_pinned=sel_doc["program_digest"])

    patched = hasattr(vloc, "make_locator")  # data/contract/lz-lazy-program/patches/verifier_v3.diff applied?

    def build(lazy: bool, tm: dict[str, Any]):
        """`build_pinned_program(..., lazy=lazy)` with the patch; the same steps spelled out without it."""
        identity = inst_doc["program"]["identity"]
        if patched:
            return vloc.build_pinned_program(identity, expected_digest=sel_doc["program_digest"], expected_gates=inst_doc["program"]["gates"]["value"],
                                             timings=tm, lazy=lazy)
        from verity.ir import Program, bind, encode_program, program_digest
        from verity_vllm.ir.lazy import LazyProgram, lazy_program_digest
        from verity_vllm.registry import b1
        C = dict(getattr(b1, f"{identity['case']}_CONFIG"))
        sp = b1.serve_program(C, int(identity["LP"]), int(identity["STEPS"]), family=identity.get("family"), attention=identity.get("attention"))
        t0 = time.perf_counter()
        program = LazyProgram(sp) if lazy else Program(sp)
        tm["program_build_s"], tm["program_build_mode"] = time.perf_counter() - t0, "lazy" if lazy else "full"
        t0 = time.perf_counter()
        d = lazy_program_digest(program) if lazy else program_digest(encode_program(program))
        tm["program_digest_s"] = time.perf_counter() - t0
        problems = [] if (sp.id == identity["function"] and program.gates == inst_doc["program"]["gates"]["value"] and d == sel_doc["program_digest"]) else \
            [f"id/gates/digest mismatch: {sp.id} {program.gates} {d[:16]}"]
        return program, C, problems

    def locator(program, C):
        if patched:
            return vloc.make_locator(program, C)
        from verity_vllm.ir.lazy import LazyProgram

        from e2e.lazy_locator import LazyProgramLocator
        return LazyProgramLocator(program, C) if isinstance(program, LazyProgram) else vloc.ProgramLocator(program, C)

    def one(lazy: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        tm: dict[str, Any] = {}
        t0 = time.perf_counter()
        program, C, problems = build(lazy, tm)
        st.mark(f"build_pinned_program(lazy={lazy}){'' if patched else ' [unpatched verifier_v3: equivalent steps]'}", t0, problems=problems, mode=tm.get("program_build_mode"))
        if program is None or problems:
            raise SystemExit(f"build_pinned_program refused: {problems}")
        t0 = time.perf_counter()
        loc = locator(program, C)
        st.mark(f"make_locator -> {type(loc).__name__}", t0)
        t0 = time.perf_counter()
        units = vloc.locate_selected(loc, rows, klass=klass, ru_request=ru_request, n_v=n_v)
        dt = time.perf_counter() - t0
        st.mark(f"locate_selected x{len(units)} ({type(loc).__name__})", t0, ms_per_unit=round(1000 * dt / max(1, len(units)), 4),
                refused=sum(1 for u in units if isinstance(u.get("locate"), dict) and "refused" in u["locate"]))
        return units, tm

    units_lazy, tm_lazy = one(True)
    out["lazy_timings"] = tm_lazy
    prover_path = rec / "mapping.units.jsonl"
    if prover_path.exists():
        prover = [json.loads(ln) for ln in prover_path.read_text().splitlines() if ln.strip()]
        t0 = time.perf_counter()
        dis = vloc.compare_units(units_lazy, prover)
        out["vs_prover_mapping"] = {"prover_units": len(prover), "disagreements": len(dis), "first": dis[:3]}
        st.mark("compare_units(lazy, prover mapping.units.jsonl)", t0, disagreements=len(dis))
    rc = 0
    if a.compare_full:
        units_full, tm_full = one(False)
        out["full_timings"] = tm_full
        dis = vloc.compare_units(units_lazy, units_full)
        # every field, not just LOCATE_FIELDS
        diff_any = sum(1 for x, y in zip(units_lazy, units_full) if x != y)
        out["lazy_vs_full"] = {"units": len(units_full), "disagreements_locate_fields": len(dis), "disagreements_any_field": diff_any, "first": dis[:3]}
        print("lazy vs full:", "IDENTICAL" if not dis and not diff_any else f"DIFFERENT ({len(dis)}/{diff_any})", flush=True)
        rc = 0 if not dis and not diff_any else 2
    return rc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--archive", help="record directory (protocol_instance.json, chronology/selected.json, protocol.json, mapping.units.jsonl)")
    ap.add_argument("--expressibility", help="expressibility.json (default out/record/expressibility.json)")
    ap.add_argument("--compare-full", action="store_true", help="also build Program(sp) / ProgramLocator and compare every located unit")
    ap.add_argument("--case", default="B0")
    ap.add_argument("--lp", type=int)
    ap.add_argument("--steps", type=int)
    ap.add_argument("--digest", action="store_true")
    ap.add_argument("--expect-digest")
    ap.add_argument("--gates", type=int, default=0)
    ap.add_argument("--check-dataflow", type=int, default=0, help="run LazyProgramLocator.check_dataflow at N random positions (+ the 4 boundary ones)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    st = Stages()
    out: dict[str, Any] = {"label": "MEASURED", "stages": st.rows, "argv": sys.argv[1:]}
    if a.archive:
        rc = run_archive(a, st, out)
    else:
        if a.lp is None or a.steps is None:
            ap.error("--lp and --steps (or --archive)")
        out.update(case=a.case, LP=a.lp, STEPS=a.steps)
        rc = run_shape(a, st, out)
    out["max_rss_gb"] = round(rss_gb(), 3)
    print("DONE", json.dumps({k: v for k, v in out.items() if k != "stages"}), flush=True)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=1))
    return rc


if __name__ == "__main__":
    sys.exit(main())
