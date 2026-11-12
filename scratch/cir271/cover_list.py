"""cover_list.py OUT: CIR-271 Part 2's sweep inputs, read on `main` at GLM-5.3 TP8 H200 class G, one request (0, 256, 32), greedy.

Phases, each its own JSON under OUT, written as it ends (a later phase's failure leaves the earlier ones):

  list.json    rank 0's n_t (`plan_steps`, then `TemplateCounts(reg.phi).unit(u)`), eager and compiled: every template's
               descriptor id, instances, its Definition's family, name, version and bindings
  ranks.json   ranks 1 to 7 the same way (eager), and each one's difference from rank 0
  pins.json    each template's pins (circuit-check's `load_pins`, kinds keyed by a Definition's id) at its id and at other
               bindings of its family, its owner module (`sidecars.owner`) and whether that module has a sidecar
  cuts.json    `serving_rows.unit_templates` of each template at this tree (the census's re-cut rule)
  names.json   for each template and each card block cutting its family: `stage.py`'s tables matched at its statics, and
               `census:NAME` resolved as CIR-259's names.py resolves it (`serving_rows.template`, `chunk_rows.statement`)
  roots.json   circuit-check's catalog roots made unbuilt (`targets.binding_roots`): their ids and families
"""
import importlib.util
import json
import resource
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
GUARD_GB = float(sys.argv[2]) if len(sys.argv) > 2 else 24.0
T0 = time.time()


def rss_gb() -> float:
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * resource.getpagesize() / 2**30


def say(*a) -> None:
    print(f"COVER {time.time() - T0:7.1f}s {rss_gb():5.2f}GiB", *a, flush=True)


def put(name: str, obj) -> None:
    (OUT / name).write_text(json.dumps(obj, indent=1, default=str) + "\n")
    say("wrote", name)


def err(e: BaseException) -> str:
    return f"{type(e).__name__}: {e}"[:500]


spec = importlib.util.spec_from_file_location("compile_cost", "benchmarks/compile_cost/run.py")
cc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cc)
from verity.circuits.codec import descriptor_id  # noqa: E402
from verity.definitions.call_scope import spec_base, spec_statics  # noqa: E402
from verity_vllm.program.predict.steps import plan_steps  # noqa: E402
from verity_vllm.program.predict.template_counts import TemplateCounts  # noqa: E402

advice, ingress = cc.session([(0, 256, 32)])


def counts(rank: int, ex: str):
    reg = cc.registered(rank, None, ex)
    plan = plan_steps(reg, advice, ingress)
    u = [x for x in plan.requests if x is not None][0]
    tc = TemplateCounts(reg.phi)
    n = tc.unit(u)
    return reg, u, tc, n


def describe(tc, n: Counter) -> list[dict]:
    rows = []
    for tid, c in sorted(n.items(), key=lambda kv: (-kv[1], kv[0])):
        f = tc._fns.get(tid)
        d = getattr(f, "defn", f)
        rows.append({"id": tid, "n": int(c), "family": getattr(f, "base_id", None) or getattr(f, "id", None),
                     "spec_base": spec_base(tid), "name": getattr(d, "name", None), "version": getattr(d, "version", None),
                     "bindings": {k: (descriptor_id(v) if hasattr(v, "signature") else v)
                                  for k, v in (getattr(f, "bindings", None) or {}).items()},
                     "kind": type(f).__name__})
    return rows


# ---- list ----------------------------------------------------------------------------------------------------------------
lst: dict = {"request": [0, 256, 32], "commit_note": "see the run's source commit", "by": {}}
fns: dict = {}
for ex in ("eager", "compiled"):
    try:
        reg, u, tc, n = counts(0, ex)
        lst["by"][ex] = {"LP": u.LP, "T": u.T, "C": u.C, "chunks": list(u.chunks), "templates": len(n),
                         "instances": int(sum(n.values())), "dsa_table_reads": getattr(reg, "dsa_table_reads", None),
                         "rows": describe(tc, n)}
        if ex == "eager":
            fns = dict(tc._fns)
            n0 = n
        say("list", ex, lst["by"][ex]["templates"], lst["by"][ex]["instances"])
    except Exception as e:  # noqa: BLE001
        lst["by"][ex] = {"error": err(e), "trace": traceback.format_exc()[-3000:]}
        say("list", ex, "failed", err(e))
if "rows" in lst["by"].get("eager", {}) and "rows" in lst["by"].get("compiled", {}):
    a = {r["id"]: r["n"] for r in lst["by"]["eager"]["rows"]}
    b = {r["id"]: r["n"] for r in lst["by"]["compiled"]["rows"]}
    lst["eager_equals_compiled"] = a == b
put("list.json", lst)
if not fns:
    sys.exit(2)
#: names outside the list, swept the same way: Friday's select at vllm's binding (NET-390, 12:19 PM PST)
EXTRA = ["GreedySelect_v1{V=154880,N=7744}"]
for x in EXTRA:
    try:
        from verity_vllm.pipeline.serving_rows import template as _tpl
        fns[x] = _tpl(x).fn
    except Exception as e:  # noqa: BLE001
        say("extra", x, "names no template:", err(e))
IDS = [r["id"] for r in lst["by"]["eager"]["rows"]] + [x for x in EXTRA if x in fns]

# ---- ranks ---------------------------------------------------------------------------------------------------------------
ranks: dict = {}
for rank in range(1, 8):
    try:
        _reg, _u, _tc, n = counts(rank, "eager")
        only_r = sorted(set(n) - set(n0))
        only_0 = sorted(set(n0) - set(n))
        moved = {t: [int(n0[t]), int(n[t])] for t in set(n) & set(n0) if n[t] != n0[t]}
        ranks[rank] = {"templates": len(n), "instances": int(sum(n.values())), "same_ids": not only_r and not only_0,
                       "same_counts": not only_r and not only_0 and not moved, "only_here": {t: int(n[t]) for t in only_r},
                       "only_rank0": {t: int(n0[t]) for t in only_0}, "count_differs": moved}
        say("rank", rank, ranks[rank]["templates"], ranks[rank]["instances"], "same ids", ranks[rank]["same_ids"])
    except Exception as e:  # noqa: BLE001
        ranks[rank] = {"error": err(e)}
        say("rank", rank, "failed", err(e))
put("ranks.json", ranks)

# ---- pins ----------------------------------------------------------------------------------------------------------------
pins_out: dict = {}
try:
    from circuit_check import checks as CK
    from circuit_check import sidecars as SC
    from circuit_check import targets as TG

    TG.load_registries()
    pins = CK.load_pins()
    by_fam: dict = {}
    for kind, d in pins.items():
        for key in d:
            by_fam.setdefault(spec_base(key.partition("/")[0]), []).append([kind, key])
    disc = SC.discover()
    for tid in IDS:
        f = fns[tid]
        fam = spec_base(tid)
        try:
            mod = SC.owner(f)
        except Exception as e:  # noqa: BLE001
            mod = f"error: {err(e)}"
        pins_out[tid] = {"exact": [k for k in pins if tid in pins[k]],
                         "family": [x for x in by_fam.get(fam, []) if x[1] != tid],
                         "owner": mod, "sidecar_py": str(disc.get(mod, {}).get("py")) if isinstance(mod, str) else None,
                         "sidecar_json": str(disc.get(mod, {}).get("json")) if isinstance(mod, str) else None}
    put("pins.json", {"kinds": {k: len(v) for k, v in pins.items()}, "templates": pins_out})
except Exception as e:  # noqa: BLE001
    put("pins.json", {"error": err(e), "trace": traceback.format_exc()[-3000:], "templates": pins_out})

# ---- cuts ----------------------------------------------------------------------------------------------------------------
from verity_vllm.pipeline import serving_rows as SRW  # noqa: E402

cuts: dict = {}
stopped = None
for tid in IDS:
    if rss_gb() > GUARD_GB:
        stopped = f"stopped at {tid}: {rss_gb():.1f} GiB over the guard {GUARD_GB}"
        break
    t = time.time()
    try:
        cuts[tid] = {"units": [list(x) for x in (SRW.unit_templates(tid) or ())]}
    except Exception as e:  # noqa: BLE001
        cuts[tid] = {"error": err(e)}
    cuts[tid]["s"] = round(time.time() - t, 2)
put("cuts.json", {"stopped": stopped, "templates": cuts})

# ---- names ---------------------------------------------------------------------------------------------------------------
spec = importlib.util.spec_from_file_location("su_stage", "benchmarks/served_units/stage.py")
stage = importlib.util.module_from_spec(spec)
sys.modules["su_stage"] = stage
spec.loader.exec_module(stage)
from verity_vllm.pipeline import chunk_rows as CR  # noqa: E402
from verity_vllm.pipeline.serving_rows import template  # noqa: E402

CARD = json.loads(Path("benchmarks/sampling_plan/glm53_blocks.json").read_text())["definitions"]


def look(name: str) -> dict:
    rec: dict = {}
    t = time.time()
    try:
        tp = template(name)
        stmt = CR.statement(tp)
        rec.update(ok=True, id=tp.id, word=tp.word, layout=tp.layout, statement=stmt, zk_refusal=CR.zk_refusal(tp))
        try:
            params, _ret = template(tp.word).fn.signature()
            rec["leaf_widths"] = {p: sorted(set(ty.leaf_widths())) for p, ty in params}
        except Exception as e:  # noqa: BLE001
            rec["leaf_widths_error"] = err(e)
    except BaseException as e:  # noqa: BLE001
        rec.update(ok=False, error=err(e))
    rec["s"] = round(time.time() - t, 2)
    return rec


def tables(name: str) -> list[dict]:
    """The `UNITS`, `PARTS`, `CENSUS` and `READS` entries that name ``name`` at some statics: each entry of the same family, at
    its defaults overridden by ``name``'s statics where it has them, bound and compared by descriptor id (a census entry's name)."""
    fam, st = spec_base(name), {}
    try:
        st = spec_statics(name)
    except Exception:  # noqa: BLE001
        pass
    hits = []
    for tname, table in (("UNITS", stage.UNITS), ("PARTS", stage.PARTS), ("CENSUS", stage.CENSUS), ("READS", stage.READS)):
        for key, entry in table.items():
            S = dict(entry.defaults)
            over = {k: v for k, v in st.items() if k in S}
            S.update(over)
            try:
                if tname == "CENSUS":
                    got = entry.name(S)
                elif tname == "READS":
                    from verity.commitments.gates import embedding as E
                    got = descriptor_id(E.read(S["V"], S["H"]))
                else:
                    got = descriptor_id(entry.definition(S))
            except Exception:  # noqa: BLE001  (an entry that doesn't bind at these statics names nothing here)
                continue
            if spec_base(got) == fam:
                spec_str = key + (":" + ",".join(f"{k}={v}" for k, v in over.items()) if over else "")
                hits.append({"table": tname, "name": key, "spec": spec_str, "id": got, "exact": got == name})
    return hits


names: dict = {"templates": {}, "blocks": {}}
stopped = None
blocks_wanted = set()
for tid in IDS:
    fam = spec_base(tid)
    blocks_wanted |= {d for d, v in CARD.items() if fam in v["of"]}
    if tid in EXTRA:
        blocks_wanted |= {u for u, _n in cuts.get(tid, {}).get("units", []) if u != tid}
for tid in IDS:
    if rss_gb() > GUARD_GB:
        stopped = f"stopped at template {tid}: {rss_gb():.1f} GiB"
        break
    names["templates"][tid] = {"tables": tables(tid), "census": look(tid)}
for b in sorted(blocks_wanted):
    if stopped or rss_gb() > GUARD_GB:
        stopped = stopped or f"stopped at block {b}: {rss_gb():.1f} GiB"
        break
    names["blocks"][b] = {"tables": tables(b), "census": look(b)}
names["stopped"] = stopped
put("names.json", names)

# ---- roots ---------------------------------------------------------------------------------------------------------------
try:
    from circuit_check import targets as TG

    roots = TG.binding_roots()
    ids = sorted({r.id for r in roots})
    fams = sorted({TG.family(r) for r in roots})
    put("roots.json", {"roots": len(roots), "ids": ids, "families": fams, "root_errors": TG.ROOT_ERRORS[:50]})
except BaseException as e:  # noqa: BLE001
    put("roots.json", {"error": err(e), "trace": traceback.format_exc()[-3000:]})
say("done")
