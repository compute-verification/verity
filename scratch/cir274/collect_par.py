"""CIR-274 (REQBLOCK): `scratch/cir271/collect43.py` with the W ranks compiled in parallel, one process a rank (fork), then
composed as `tp_steps.compile_tp` composes them, and `pipeline.tp.wires.sites_circuit` over the bindings. A rank's
`RankSites` comes back with its Definitions by reference (a registered one by id, a binding as its Definition and record),
so the parent's are the ones it would have built. Run from adapters/vllm:

    ../../.venv/bin/python ../../scratch/cir274/collect_par.py OUT LAYERS LP

Writes OUT/collect_<LP>_l<LAYERS>.json with collect43.py's fields."""
import importlib.util
import io
import json
import multiprocessing as mp
import pickle
import re
import resource
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
from tests.pipeline.test_tp_wires import in_order, proof_blocks  # noqa: E402
from tests.program.predict.test_steps import GREEDY, _session  # noqa: E402

spec = importlib.util.spec_from_file_location("compile_cost", Path.cwd().parents[1] / "benchmarks/compile_cost/run.py")
cc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cc)
from verity_vllm.pipeline.tp import wires as TW  # noqa: E402
from verity_vllm.program import tp_binding as TB  # noqa: E402
from verity_vllm.program.predict.steps import compile_steps  # noqa: E402
from verity_vllm.program.predict.tp_steps import at_rank  # noqa: E402
from verity_vllm.program.tp_workload import TpWorkload  # noqa: E402

from verity.circuits import defs as D  # noqa: E402
from verity.circuits.codec import descriptor_id  # noqa: E402


def _rebind(defn, bindings):
    return D.bind(defn, **bindings)


def _registered(fid):
    return D.REGISTRY[fid]


class _ByRef(pickle.Pickler):
    def reducer_override(self, obj):
        if isinstance(obj, D.SpecializedDefinition):
            return _rebind, (obj.defn, dict(obj.bindings))
        if isinstance(obj, D.FunctionDefinition) and D.REGISTRY.defs.get(obj.id) is obj:
            return _registered, (obj.id,)
        return NotImplemented


def _dumps(x) -> bytes:
    f = io.BytesIO()
    _ByRef(f, protocol=pickle.HIGHEST_PROTOCOL).dump(x)
    return f.getvalue()


ARGS: dict = {}


def _rank(r: int) -> bytes:
    reg, advice, ingress, world = ARGS["reg"], ARGS["advice"], ARGS["ingress"], ARGS["world"]
    t = time.perf_counter()
    c = compile_steps(at_rank(reg, r), advice, ingress)
    rows = [[(x.request, x.start, x.n, x.samples) for x in step] for step in c.steps]
    mine = tuple(None if u is None else u.digest for u in c.requests)
    sites = {d: TB.rank_sites(c.programs[d].fn.body, world, r, d) for d in set(mine) - {None}}
    return _dumps((rows, c.workload.digest, mine, sites, round(time.perf_counter() - t, 1),
                   resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))


def main() -> None:
    out, layers, lp = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    t0, rec = time.perf_counter(), {"layers": layers, "LP": lp, "outputs": 2, "driver": "collect_par"}
    try:
        once = {**GREEDY, "max_tokens": 2, "ignore_eos": True}
        advice, ingress = _session([(lp, 2, once)], [[(0, 0, lp)], [(0, lp, 1)]])
        reg = cc.registered(0, layers, "eager")
        world = int(reg.phi.tp.get("world", 1))
        ARGS.update(reg=reg, advice=advice, ingress=ingress, world=world)
        with mp.get_context("fork").Pool(world) as pool:
            got = [pickle.loads(b) for b in pool.map(_rank, range(world))]
        rec["rank_compile_s"] = [g[4] for g in got]
        rec["rank_peak_gib"] = [round(g[5] / 1048576, 2) for g in got]
        if any(g[0] != got[0][0] for g in got):
            raise ValueError("a rank compiles the advice to other rows than rank 0")
        digests, units = [g[1] for g in got], [g[2] for g in got]
        bindings = []
        for i in range(len(units[0])):
            key = tuple(u[i] for u in units)
            bindings.append(None if key[0] is None else TB.pair([got[r][3][d] for r, d in enumerate(key)]))
        workload = TpWorkload(digests, bindings)
        rec["compile_s"] = round(time.perf_counter() - t0, 1)
        live = [b for b in bindings if b is not None]
        sites = TW.sites_circuit(bindings, "class-g")
        try:
            in_order(sites)
            backward = None
        except AssertionError as e:
            backward = str(e)[:2000]
        kinds = Counter(re.sub(r"\[\d+, \d+\)", "[..)", re.sub(r"request \d+'s collective \d+ \[\d+, \d+\): rank \d+'s node \d+ ",
                                                                "", u)) for u in sites.unwired)
        rec.update({"workload": workload.digest, "sites": [len(x.sites) for x in live],
                    "combines": sorted({b.fns[0][s.combine[0][0]].id for b in live for s in b.sites}),
                    "units": dict(sorted(sites.circuit.units.items())), "templates": len(sites.circuit.templates),
                    "template_ids": sorted(descriptor_id(fn) for fn, _ in sites.circuit.templates),
                    "wires": len(sites.circuit.wires), "unwired": dict(sorted(kinds.items())),
                    "proof_blocks": proof_blocks(sites), "backward": backward})
    except BaseException as e:  # noqa: BLE001 - a refusal is a result
        rec.update({"error": f"{type(e).__name__}: {e}"[:3000], "trace": traceback.format_exc()[-3000:]})
    rec["wall_s"] = round(time.perf_counter() - t0, 1)
    rec["peak_gib"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576, 2)
    (out / f"collect_{lp}_l{layers}.json").write_text(json.dumps(rec, indent=1) + "\n")
    print("REQBLOCK collect_par " + json.dumps({k: rec.get(k) for k in ("layers", "LP", "sites", "wires", "templates", "unwired",
                                                                       "proof_blocks", "backward", "compile_s", "wall_s",
                                                                       "peak_gib", "error")}))


if __name__ == "__main__":
    main()
