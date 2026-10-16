"""The hierarchical VU store: a run's sampled units kept inside the circuit they came from, and the query that cuts instance sets.

The stored form of a VU export (`vllm-vu-store/v1`); flat instance sets (`vllm-vu-set/v1`) are the output of `extract`, and the
default extraction is exactly the sets `vu_export` defines.

    <out>/store/run.json                               run root, provenance (row, model, serving, run, source, epoch, manifest digest),
                                                       the draw's limits, the weight sample rule and the checkpoint of record
    <out>/store/programs/<program digest>/program.json the Program by reference (digest, request labels, Build dir)
    <out>/store/programs/<program digest>/units.jsonl  one line per exported VU: its Call (i, spec, module op_path, request, step,
                                                       row, phase, stratum), its operand identities, and every row evaluation it
                                                       performed (interior producers first): family, statics, args, outs
    <out>/store/values/<sha256>.<u8|u16|u32|u64>       the words of every operand and output, once each (content-addressed)

An evaluation's arg is `{"value": sha, "dtype", "words"}` (committed words, or a small registered weight in full) or, for a GEMM's
weight, `{"weight": {field, range, shape, sha256 of the whole slice}, "sample": {"rows", "value", "index"}}`: a fixed sample of
its rows (`gemm_coords_per_row`, drawn from the selection key at the evaluation's own index), the full weight referenced by
checkpoint field and hash.  `extract(weights=...)` reads the full weight from a checkpoint when one is given (any rows); without
one it cuts the stored rows.

`extract(store, where=, module=)`: `where` is a Definition selector in the IR query grammar (`verity.ml.circuits.query_ast.NameSel`:
`Gemm_v1`, `Gemm_v1{K=2048,*}`, `Attention_v3{D=64,*}`), matched against each evaluation's Definition and statics; `module`
keeps the units whose module is that node or below it (a module path, as `program_graph` names the nodes).  The selected
evaluations go through `vu_export.decompose` unchanged and are written as flat sets.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np

from verity.ml.circuits.query_ast import NameSel

from verity_vllm.check.replay.index import ProgramIndex
from verity_vllm.check.replay.opening import CommittedStore
from verity_vllm.check.replay.population import phase_anchor, population, split_not_executed
from verity_vllm.check.replay import driver as SR
from verity_vllm.pipeline import vu_export as VX

STORE_SCHEMA = "vllm-vu-store/v1"
SR.keep_population(True)            # the Commit imports this module before its replay: the export reuses that replay's population
WEIGHT_SAMPLE_RULE = ("GEMM weight rows: selection_key(run_root).subset(k, N, ('row', request, i, evaluation)) with k = "
                      "gemm_coords_per_row; the whole [N, K] slice referenced by checkpoint field, range and sha256 (an FP8 block GEMM: "
                      "the e4m3 rows; a MoE expert GEMM: the routed expert's rows of the [E, N, K] slab, `expert` = the committed router id)")

_FORK: dict[str, Any] = {}          # the per-VU context the forked workers read (set in the parent right before the fork)


def _blob(blobs: dict[str, tuple[str, bytes]], a: np.ndarray) -> dict[str, Any]:
    a = VX._unsigned(a)
    b = np.ascontiguousarray(a).astype(a.dtype.newbyteorder("<"), copy=False).tobytes()
    h = hashlib.sha256(b).hexdigest()
    dt = VX._WORD[a.dtype.itemsize][0]
    blobs[h] = (dt, b)
    return {"value": h, "dtype": dt, "words": int(a.size)}


def _evaluation(k: tuple, fam: str, st: dict, ops: list, out: dict, refs: list, blobs: dict, limits: VX.Limits) -> dict[str, Any] | None:
    """One row evaluation as stored; None when an operand is too large to keep and is not a GEMM weight (counted, not written)."""
    args = []
    wg, rows_of = VX.WEIGHT_ROWS_OF.get(fam, (None, None))
    for g, a in enumerate(ops):
        ref = refs[g] if g < len(refs) else None
        if ref is not None and g == wg:
            w, extra = rows_of(st, ops)
            N, K = w.shape
            rows = VX._Picker(k[0], k[1])(N, limits.gemm_coords_per_row)
            full = hashlib.sha256(np.ascontiguousarray(w).astype(w.dtype.newbyteorder("<")).tobytes()).hexdigest()
            args.append({"weight": dict(ref, shape=[N, K], dtype=VX._WORD[w.dtype.itemsize][0], sha256=full, **extra),
                         "sample": {"rows": rows, "index": list(k[1]), **_blob(blobs, w[rows])}})
        elif ref is not None and int(np.asarray(a).size) > limits.max_row_words:
            return None
        else:
            args.append(dict(_blob(blobs, a), **({"weight": ref} if ref is not None else {})))
    return {"family": fam, "statics": st, "args": args, "outs": {m: _blob(blobs, v) for m, v in VX._outputs(out).items()}}


def _one_vu(k: int) -> tuple[dict[str, Any], dict[str, tuple[str, bytes]], float]:
    """Evaluate VU `k` of the draw: (its unit record, the value blobs it references, seconds)."""
    c = _FORK
    vu, tv = c["picked"][k], time.perf_counter()
    rid = vu["request_id"]
    rec, evals = VX.capture(c["indices"][rid], c["store"], c["weights"], vu, arrive=int(c["arrivals"].get(rid, 0)))
    unit = {"vu": k, "request": rid, "i": vu["i"], "family": vu["family"], "spec": vu["spec"], "stratum": vu["stratum"],
            "engine_step": rec.get("engine_step"), "step": rec.get("step"), "op_path": rec.get("op_path"), "row": rec.get("row"),
            "phase": vu["phase"], "result": rec.get("result"), "why": rec.get("why"), "inputs": rec.get("inputs")}
    blobs: dict[str, tuple[str, bytes]] = {}
    if rec.get("result") is True:
        ev = [_evaluation((c["key"], ("row", str(rid), int(vu["i"]), j)), fam, st, ops, out, refs, blobs, c["limits"])
              for j, (fam, st, ops, out, refs) in enumerate(evals)]
        if any(e is None for e in ev):
            unit["stored"], blobs = False, {}
        else:
            unit["evaluations"], unit["stored"] = ev, True
    return unit, blobs, time.perf_counter() - tv


def _run_draw(n: int, workers: int, budget_s: float, t0: float) -> Iterator[tuple[dict, dict, float] | None]:
    """The draw's results in order; None for a VU the spent time budget left undrawn.  `workers` > 1: a forked pool in chunks (the
    replay's pattern: weights and twin libraries pre-warmed in the parent, the heap frozen before the fork, children never touch CUDA)."""
    if workers <= 1:
        for k in range(n):
            yield None if time.perf_counter() - t0 > budget_s else _one_vu(k)
        return
    import gc
    from verity_vllm.check.replay import driver as SR
    from verity_vllm.check.replay.fork import fork_pool
    SR._prewarm_weights(_FORK["indices"], _FORK["picked"], _FORK["weights"])
    SR._prewarm_evaluators(_FORK["picked"])
    gc.collect()
    gc.freeze()
    try:
        with fork_pool(workers) as pool:
            chunk = workers * 4
            for lo in range(0, n, chunk):
                if time.perf_counter() - t0 > budget_s:
                    yield from (None for _ in range(lo, n))
                    return
                yield from pool.map(_one_vu, range(lo, min(n, lo + chunk)))
    finally:
        gc.unfreeze()


def export_vus(programs: dict[str, dict[str, Any]], bmap: dict[str, Any], committed: Callable, weights: Callable, *, out_dir: str | Path,
               run_root: bytes, arrivals: dict[str, int] | None = None, manifest: dict[str, Any] | None = None,
               executed_prefix: dict[str, dict] | None = None, program_dirs: dict[str, str] | None = None,
               provenance: dict[str, Any] | None = None, limits: VX.Limits | None = None,
               population_of_record: dict[str, Any] | None = None) -> dict[str, Any]:
    """Draw and evaluate, write the store (`<out_dir>/store`), then the default extraction (`<out_dir>/sets`).  Returns the export
    summary (also `<out_dir>/export.json`).  `population_of_record`: the replay's own population (`driver.take_population()`), reused
    instead of rebuilt (on a B32 row the rebuild alone took > 18 min, inside the Commit, before its verdict)."""
    t0 = time.perf_counter()
    limits = limits or VX.Limits()
    arrivals = arrivals or {}
    out = Path(out_dir)
    root = out / "store"
    (root / "values").mkdir(parents=True, exist_ok=True)
    indices: dict[str, ProgramIndex] = {}
    by_inst: dict[int, ProgramIndex] = {}
    all_vus: list[dict] = []
    skipped: list[str] = []
    reuse = population_of_record if (population_of_record or {}).get("indices") and set(population_of_record["indices"]) >= set(map(str, programs)) else None
    for n_built, (rid, inst) in enumerate([] if reuse else programs.items()):
        # the population build has its own budget, checked between request Programs (a B32 row's 33 Programs took > 18 min):
        # the requests left are named, the draw runs over the ones built
        if time.perf_counter() - t0 > float(limits.max_population_seconds):
            skipped = [str(r) for r in list(programs)[n_built:]]
            break
        pi = by_inst.get(id(inst)) or ProgramIndex(inst, program_dir=(program_dirs or {}).get(rid))
        by_inst[id(inst)] = indices[rid] = pi
        last = int(phase_anchor(pi.n_steps, (executed_prefix or {}).get(str(rid)))["last_step"])
        vus, _ne, _nef = population(pi, rid, arrive=int(arrivals.get(rid, 0)), last_step=last)
        all_vus.extend(vus)
    if reuse:
        indices, all_vus = {r: reuse["indices"][str(r)] for r in programs}, list(reuse["vus"])
        pool, _not_exec = list(reuse["pool"]), list(reuse["not_executed"])
    else:
        pool, _not_exec = split_not_executed(all_vus, executed_prefix)
    key = VX.selection_key(run_root)
    picked = VX.select(pool, limits, key)
    t_draw = time.perf_counter()                                             # the draw's budget starts here, never inside the build
    print(f"[vu-export] population: {len(indices)}/{len(programs)} request Programs, {len(all_vus)} VUs, {len(picked)} drawn, "
          f"{t_draw - t0:.0f} s" + (" (the replay's own)" if reuse else "") + (f"; {len(skipped)} Programs past the population budget" if skipped else ""),
          flush=True)
    _FORK.update(picked=picked, indices=indices, store=CommittedStore.of_record(bmap, committed, manifest), weights=weights,
                 arrivals=arrivals, key=key, limits=limits)
    digest_of = {rid: str(inst.get("digest") or "program") for rid, inst in programs.items()}
    fam_stats: dict[str, Counter] = {}
    fam_time: Counter = Counter()
    units: dict[str, list[dict]] = {}
    try:
        for k, res in enumerate(_run_draw(len(picked), int(limits.workers), float(limits.max_seconds), t_draw)):
            st = fam_stats.setdefault(picked[k]["family"], Counter())
            if res is None:
                st["not_drawn_time_budget"] += 1
                continue
            unit, blobs, dt = res
            st["drawn"] += 1
            fam_time[unit["family"]] += dt
            if unit["result"] is True:
                st["replayed_equal"] += 1
                st["stored" if unit.get("stored") else "not_stored_operand_too_large"] += 1
            else:
                st["not_exported_" + ("mismatch" if unit["result"] is False else "not_evaluated")] += 1
            for h, (dt_, b) in blobs.items():
                f = root / "values" / f"{h}.{dt_}"
                if not f.exists():
                    f.write_bytes(b)
            units.setdefault(digest_of[unit["request"]], []).append(unit)
            if k % 100 == 0:
                print(f"[vu-export] {k}/{len(picked)} VUs, {time.perf_counter() - t_draw:.0f} s", flush=True)
    finally:
        _FORK.clear()
    prov = dict(provenance or {}, run_root=bytes(run_root).hex(),
                selection=f"verity.ml.randomness.derive({VX.SELECTION_SOURCE!r}, {VX.SELECTION_DOMAIN!r}, {{run_root}}); per family key.shard(('family', "
                          "family)) through challenge.stratum_picks; sub-indices key.subset(k, n, ('row', request, i, evaluation))")
    prov.setdefault("weights", {})
    prov["weights"] = dict(prov["weights"], sample=WEIGHT_SAMPLE_RULE, rows_per_gemm=limits.gemm_coords_per_row)
    for d, us in units.items():
        pd = root / "programs" / d
        pd.mkdir(parents=True, exist_ok=True)
        reqs = sorted({u["request"] for u in us})
        (pd / "program.json").write_text(json.dumps({"program_digest": d, "requests": reqs,
                                                      "build_dirs": {r: (program_dirs or {}).get(r) for r in reqs}}, indent=1) + "\n")
        with (pd / "units.jsonl").open("w") as f:
            for u in us:
                f.write(json.dumps(u, separators=(",", ":"), default=str) + "\n")
    (root / "run.json").write_text(json.dumps({"schema": STORE_SCHEMA, "provenance": prov, "limits": dataclasses.asdict(limits)},
                                              indent=1, default=str) + "\n")
    with (out / "vus.jsonl").open("w") as f:
        for d in units:
            for u in units[d]:
                f.write(json.dumps({x: u.get(x) for x in ("vu", "request", "i", "family", "spec", "stratum", "engine_step", "op_path", "row",
                                                          "phase", "result", "why", "inputs", "stored")}, separators=(",", ":"), default=str) + "\n")
    t_extract = time.perf_counter()
    sets, inst_by_family = extract(root, out / "sets")
    for f, n in inst_by_family.items():
        fam_stats.setdefault(f, Counter())["instances"] += n
    summary = {"schema": VX.EXPORT_SCHEMA, "provenance": prov, "limits": dataclasses.asdict(limits),
               "population": {"vus": len(all_vus), "pool": len(pool), "not_executed": len(_not_exec),
                              "programs": len(programs), "programs_built": len(indices), "programs_past_budget": skipped,
                              "source": "the sampled replay's own population (reused)" if reuse else "built here",
                              "pool_by_family": dict(Counter(v["family"] for v in pool)), "not_exported_families": VX.NOT_EXPORTED},
               "by_family": {f: dict(c) | {"seconds": round(fam_time[f], 3)} for f, c in sorted(fam_stats.items())},
               "store": {"values": len(list((root / "values").iterdir())), "bytes": sum(p.stat().st_size for p in root.rglob("*") if p.is_file())},
               "sets": sets, "wall_s": round(time.perf_counter() - t0, 3),
               "timing": {"population_s": round(t_draw - t0, 3), "draw_s": round(t_extract - t_draw, 3), "extract_s": round(time.perf_counter() - t_extract, 3),
                          "rule": "max_seconds budgets the draw from after the population is built; max_population_seconds budgets the build"}}
    (out / "export.json").write_text(json.dumps(summary, indent=1, default=str) + "\n")
    return summary


def _matches(sel: NameSel | None, fam: str, st: dict[str, Any]) -> bool:
    """`NameSel`'s rule on a stored evaluation: the name is the Definition id (`Gemm_v1`) or its name (`Gemm`); every listed static
    equals the binding as the spec renders it (`*` = any)."""
    if sel is None:
        return True
    if sel.name not in (fam, fam.rsplit("_v", 1)[0]):
        return False
    return all(k in st and (v == "*" or str(st[k]) == v) for k, v in (sel.statics or ()))


def _in_module(op_path: str | None, module: str | None) -> bool:
    if not module:
        return True
    op = str(op_path or "")
    return any(op == m or op.startswith(m + ".") for m in (module, module.split(".", 1)[1] if "." in module else module))


def _value(root: Path, a: dict[str, Any]) -> np.ndarray:
    return np.frombuffer((root / "values" / f"{a['value']}.{a['dtype']}").read_bytes(), dtype=np.dtype(VX._WORD[int(a["dtype"][1:]) // 8][1]).newbyteorder("<"))


def _operands(root: Path, ev: dict[str, Any], weights: Callable | None) -> list:
    ops = []
    for a in ev["args"]:
        if "sample" in a:
            w = a["weight"]
            full = weights(w["field"], *w["range"]) if weights is not None else None
            if full is not None:
                ops.append(np.asarray(full, dtype=VX._WORD[int(w.get("dtype", "u16")[1:]) // 8][1]).reshape(-1))
            else:
                ops.append(VX.RowSample(list(a["sample"]["rows"]), _value(root, a["sample"]).reshape(len(a["sample"]["rows"]), -1)))
        else:
            ops.append(_value(root, a))
    return ops


def extract(store: str | Path, out_dir: str | Path, *, where: str | None = None, module: str | None = None,
            limits: VX.Limits | None = None, weights: Callable[[str, int, int], np.ndarray | None] | None = None) -> tuple[list[dict], dict[str, int]]:
    """Flat instance sets of the stored units the query selects, written under `out_dir`: (set summaries, instances by family).
    No query = the default extraction: every stored unit, the draw's own limits."""
    root = Path(store)
    run = json.loads((root / "run.json").read_text())
    limits = limits or VX.Limits(**run["limits"])
    key = VX.selection_key(bytes.fromhex(run["provenance"]["run_root"]))
    sel = NameSel.parse(where) if where else None
    units = [json.loads(line) for p in sorted((root / "programs").glob("*/units.jsonl")) for line in p.read_text().splitlines()]
    writers: dict[str, VX._SetWriter] = {}
    by_family: Counter = Counter()
    for u in sorted(units, key=lambda u: u["vu"]):
        if not u.get("evaluations") or not _in_module(u.get("op_path"), module):
            continue
        evs = u["evaluations"]
        for j, ev in enumerate(evs):
            if not _matches(sel, ev["family"], ev["statics"]):
                continue
            pick = VX._Picker(key, ("row", str(u["request"]), int(u["i"]), j))
            outs = {m: _value(root, a) for m, a in ev["outs"].items()}
            try:
                insts = VX.decompose(ev["family"], ev["statics"], _operands(root, ev, weights), outs, pick, limits)
            except VX.DecompositionMismatch as e:
                by_family[f"decomposition_error:{u['family']}"] += 1
                print(f"[vu-export] unit {u['vu']} evaluation {j}: {e}", flush=True)
                continue
            for inst in insts:
                w = writers.get(inst.set_name) or writers.setdefault(inst.set_name, VX._SetWriter(inst.set_name, inst.relation))
                if len(w.index) >= limits.max_instances_per_set:
                    continue
                w.add(inst, [u["vu"], u["request"], u["engine_step"], u["op_path"], u["row"], u["phase"],
                             "committed" if j == len(evs) - 1 else "interior"])
                by_family[u["family"]] += 1
    prov = dict(run["provenance"], extraction={"where": where, "module": module, "weights": "checkpoint" if weights else "stored sample"})
    return [w.write(Path(out_dir), prov) for _n, w in sorted(writers.items())], dict(by_family)


def verify_store(store: str | Path) -> dict[str, Any]:
    """Every value file hashes to its name and every unit's references resolve."""
    root = Path(store)
    bad = [p.name for p in (root / "values").iterdir() if hashlib.sha256(p.read_bytes()).hexdigest() != p.name.split(".")[0]]
    have = {p.name for p in (root / "values").iterdir()}
    missing = 0
    for p in (root / "programs").glob("*/units.jsonl"):
        for line in p.read_text().splitlines():
            for ev in json.loads(line).get("evaluations") or []:
                for a in [*ev["args"], *ev["outs"].values()]:
                    a = a.get("sample", a)
                    missing += f"{a['value']}.{a['dtype']}" not in have
    return {"ok": not bad and not missing, "bad_values": bad[:8], "missing_values": missing, "values": len(have)}


#: below this fraction of the memory limit free, the export runs serial: a forked pool's pages could OOM-kill the Commit before
#: its summary and verdict are written, which no `except` catches
HEADROOM_MIN = 0.20


def memory_headroom() -> float | None:
    """Free fraction of this process's memory limit: the cgroup's (v2 `memory.max` / `memory.current`, v1 `limit_in_bytes` /
    `usage_in_bytes`), else the host's (`MemAvailable` / `MemTotal`); None when none is readable."""
    for lim_f, cur_f in (("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),
                         ("/sys/fs/cgroup/memory/memory.limit_in_bytes", "/sys/fs/cgroup/memory/memory.usage_in_bytes")):
        try:
            lim, cur = Path(lim_f).read_text().strip(), int(Path(cur_f).read_text().strip())
        except (OSError, ValueError):
            continue
        if lim.isdigit() and int(lim) < (1 << 60):
            return max(0.0, 1.0 - cur / int(lim))
    try:
        mi = {ln.split(":")[0]: int(ln.split()[1]) for ln in Path("/proc/meminfo").read_text().splitlines() if ln.split()[1:]}
        return mi["MemAvailable"] / mi["MemTotal"]
    except (OSError, KeyError, ValueError, ZeroDivisionError):
        return None


def _limits_for(out: Path, sr: dict[str, Any], headroom: float | None = None) -> VX.Limits:
    lim = VX.Limits.from_dir(out)
    if (sr.get("evaluator") or {}).get("retention") == "device":
        lim.workers = 1                                                      # forked children cannot read device-retained blocks
    free = memory_headroom() if headroom is None else headroom
    if free is not None and free < HEADROOM_MIN:
        lim.workers = 1
    return lim


def after_replay(out_dir: str | None, sr: dict[str, Any], pair: int, run_root: bytes, programs: dict[str, dict[str, Any]], bmap: dict[str, Any],
                 reader: Callable, weights: Callable, arrivals: dict[str, int], manifest: Any, replay_kw: dict[str, Any], loaded: dict[str, dict],
                 program_digests: dict[str, str], versions: dict[str, Any], eng: Any) -> dict[str, Any] | None:
    """The Commit stage's call, right after pair 0's replay record and value source are taken: exports when `out_dir` is given and the
    replay passed.  A failure is printed and returned by name; nothing here reaches the record or a verdict."""
    kept = SR.take_population()                                              # taken (and dropped) whether or not this pair exports
    if not out_dir or pair != 0 or sr.get("result") is not True:
        return None
    out = Path(out_dir)
    extra = json.loads((out / "provenance.json").read_text()) if (out / "provenance.json").is_file() else {}
    man = manifest if isinstance(manifest, dict) else None
    prov = dict(extra, model=(getattr(eng, "cp", None) or {}).get("repo"), revision=(getattr(eng, "cp", None) or {}).get("revision"),
                program_digests=program_digests, manifest_digest=(man or {}).get("manifest_digest"), pair=pair,
                versions={k: v for k, v in versions.items() if isinstance(v, (str, int, float))},
                weights={"checkpoint": {"repo": (getattr(eng, "cp", None) or {}).get("repo"), "revision": (getattr(eng, "cp", None) or {}).get("revision"),
                                        "weights_of_record_root": (sr.get("weights") or {}).get("committed_root")}})
    try:
        s = export_vus(programs, bmap, reader, weights, out_dir=out, run_root=run_root, arrivals=arrivals, manifest=man,
                       executed_prefix=replay_kw.get("executed_prefix"), program_dirs={k: v["dir"] for k, v in loaded.items() if v.get("dir")},
                       provenance=prov, limits=_limits_for(out, sr), population_of_record=kept)
    except Exception as e:  # noqa: BLE001  -- not part of the record: named, never a verdict
        print(f"[vu-export] FAILED: {e!r}"[:400], flush=True)
        return {"ok": False, "why": repr(e)[:400]}
    print(f"[vu-export] {len(s['sets'])} sets, {sum(x['n'] for x in s['sets'])} instances, {sum(x['bytes'] for x in s['sets'])} bytes "
          f"in {s['wall_s']} s -> {out}", flush=True)
    return s
