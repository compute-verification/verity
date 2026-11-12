"""CIR-265, the lead's question: does a unit a partition file, the served statements' test or a card names lower through
`DsaCountSum_v1`'s builder?  Read off the tree at ROOT, nothing staged:

  names    every template of ``benchmarks/one_stage/*.json`` (``templates`` and the keys of ``chunk_cap``) and every Definition
           of the card ``benchmarks/sampling_plan/glm53_blocks.json`` (``definitions``)
  units    each name with the units the adapter cuts it into (`serving_templates.unit_templates`)
  reach    every Definition a unit's Definition holds, walked as `codec._walk` walks a Program (a body's nodes, a static's
           functions); a unit reaches the count sum when one of them is `DsaCountSum_v1`
  digest   the served statements' digest as `adapters/vllm/tests/pipeline/test_served_unit_rows.py` computes it, with the
           builder as it is and with the builder committing its sum, and how often the builder ran for each

    python served_reach.py ROOT [MAX_GB]

The process exits itself past MAX_GB of RSS (1.0)."""
import hashlib
import json
import os
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(sys.argv[1]).resolve()
MAX_GB = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
sys.path[:0] = [str(ROOT), str(ROOT / "verity/protocols/flock/prover/python"), str(ROOT / "adapters/vllm")]


def guard():
    while True:
        rss = int(open("/proc/self/statm").read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**30
        if rss > MAX_GB:
            print(f"GUARD: RSS {rss:.2f} GB past {MAX_GB} GB", file=sys.stderr, flush=True)
            os._exit(9)
        time.sleep(0.2)


threading.Thread(target=guard, daemon=True).start()
t0 = time.monotonic()

import verity  # noqa: E402
import verity_flock  # noqa: E402
import verity_vllm  # noqa: E402
from verity.circuits import codec as CD  # noqa: E402
from verity.circuits.defs import CompositeDefinition, SpecializedDefinition, bind  # noqa: E402
from verity.commitments.windows import interface as IF  # noqa: E402
from verity.definitions.registry import boolean_dsa_topk_mask as BM  # noqa: E402
from verity_vllm.pipeline import serving_rows as SRW, serving_templates as TS, weight_rows as WT  # noqa: E402
from verity_vllm.query.program_view import spec_base, spec_statics  # noqa: E402

SERVED = [ROOT / "benchmarks/one_stage" / f"glm53_{p}.json" for p in ("all_families_tp8_step16", "sparse_mla_tp8_step16", "prefill_tp8_step0")]
PINNED = ("eed75699e3df3e72e98379acfda035f1b2a72d19482b103a1f0d1a8e083833fc"
          "083efb3005628542c59f613d48082ae5b1ff58f2df9eed86cd8646e380d7110a")
CARD = ROOT / "benchmarks/sampling_plan/glm53_blocks.json"
COUNT_TREE = ("DsaCountSum_v1", "DsaRadixTies_v2", "DsaRadixPass_v2", "DsaButterflyTagsCount_v1", "DsaTopKMaskBlock_v2",
              "DsaTopKMaskWhole_v2", "DsaMaskPositions_v2")

out = {"root": str(ROOT), "modules": {m.__name__: str(Path(m.__file__).parent) for m in (verity, verity_flock, verity_vllm)}}

names: dict[str, list[str]] = {}
for f in sorted((ROOT / "benchmarks/one_stage").glob("*.json")):
    d = json.loads(f.read_text())
    if isinstance(d, dict) and "templates" in d:
        for t in [*d["templates"], *d.get("chunk_cap", {})]:
            names.setdefault(t, []).append(f.name)
out["partition_files"] = sorted({f for fs in names.values() for f in fs})
out["partition_templates"] = len(names)
card = json.loads(CARD.read_text())["definitions"]
for t in card:
    names.setdefault(t, []).append(CARD.name)
out["card_definitions"] = len(card)
out["names"] = len(names)

TS.template  # the registry loads as `template` loads it
from verity.definitions import boolean as BB  # noqa: E402
from verity.definitions.call_scope import specialization  # noqa: E402
import importlib  # noqa: E402

importlib.import_module("verity.commitments.gates.embedding")
out["registry_import_errors"] = [str(e) for e in (BB.load() or [])]


def definition(t: str):
    fn = specialization(spec_base(t), spec_statics(t))
    fn = bind(fn) if isinstance(fn, CompositeDefinition) else fn
    assert CD.descriptor_id(fn) == t, (t, CD.descriptor_id(fn))
    return fn


units: dict[str, set[str]] = {}
unresolved: dict[str, str] = {}
for t in names:
    if os.environ.get("REACH_TRACE"):
        print("name", t[:120], round(int(open("/proc/self/statm").read().split()[1]) * 4096 / 2**30, 2), round(time.monotonic() - t0, 1), file=sys.stderr, flush=True)
    try:
        got = TS.unit_templates(t)
    except Exception as e:  # noqa: BLE001
        got, unresolved[t] = (), f"unit_templates: {type(e).__name__}: {e}"[:200]
    for u, _n in ((t, 1), *got):
        units.setdefault(u, set()).add(t)
out["units"] = len(units)

visited: dict[int, object] = {}
reach_of: dict[int, frozenset[str]] = {}


def reach(fn) -> frozenset[str]:
    """The families (name_vN) of every Definition ``fn`` holds, itself among them, as `codec._walk` visits them."""
    if visited.get(id(fn)) is fn:
        return reach_of[id(fn)]
    visited[id(fn)] = fn
    reach_of[id(fn)] = frozenset()  # a Definition never holds itself
    got = {spec_base(CD.descriptor_id(fn))}
    if isinstance(fn, SpecializedDefinition):
        held = list({id(n.fn): n.fn for n in fn.body.nodes}.values())
        fn._body = None  # a unit's gates are gigabytes: the Definitions it holds are read, the body is built again if asked
        for g in held:
            got |= reach(g)
        fid = CD.descriptor_id(fn)
        for k in fn.defn.statics:
            for f in CD._static_functions(fn.bindings[k]):
                got |= reach(CD._as_specialized(f, fid))
    reach_of[id(fn)] = frozenset(got)
    return reach_of[id(fn)]


hits: dict[str, dict] = {}
families: set[str] = set()
for u, via in sorted(units.items()):
    if os.environ.get("REACH_TRACE"):
        print("unit", u[:120], round(int(open("/proc/self/statm").read().split()[1]) * 4096 / 2**30, 2), round(time.monotonic() - t0, 1), file=sys.stderr, flush=True)
    try:
        got = reach(definition(u))
    except Exception as e:  # noqa: BLE001
        unresolved[u] = f"{type(e).__name__}: {e}"[:200]
        continue
    families |= got
    if got & set(COUNT_TREE):
        hits[u] = {"reaches": sorted(got & set(COUNT_TREE)), "named_by": sorted(via)}
out["unresolved"] = unresolved
out["walked_units"] = len(units) - len([u for u in unresolved if u in units])
out["families_reached"] = len(families)
out["dsa_families_reached"] = sorted(f for f in families if f.startswith("Dsa"))
out["count_tree_families"] = list(COUNT_TREE)
out["units_reaching_the_count_tree"] = hits
out["walk_seconds"] = round(time.monotonic() - t0, 1)

IF.templates_of = lambda _w: []
WT.reads = lambda _ms, _rows, _units: None
IF.circuit_record = lambda _w, _name, _templates, sts, **_kw: sts
calls = {"n": 0}


def counting(build):
    def run(C, *a):
        calls["n"] += 1
        return build(C, *a)
    return run


def committing(build):
    def run(C, *a):
        return [C.commit(list(w)) for w in build(C, *a)]
    return run


def digest():
    tmpls = sorted({t for f in SERVED for t in json.loads(f.read_text())["chunk_cap"]})
    calls["n"] = 0
    sts = SRW.circuit_record(SimpleNamespace(members=[SimpleNamespace(template=t, schemas=True) for t in tmpls], row_tables=()), [])
    return {"count": len(sts), "sha512": hashlib.sha512(IF.circuit_bytes(sts)).hexdigest(), "count_sum_builder_calls": calls["n"]}


plain = BM.DsaCountSum.build
BM.DsaCountSum.build = counting(plain)
out["statements_as_is"] = digest()
BM.DsaCountSum.build = counting(committing(plain))
TS._TEMPLATES.clear()
out["statements_sums_committed"] = digest()
out["statements_equal_pin"] = [out[k]["sha512"] == PINNED for k in ("statements_as_is", "statements_sums_committed")]
out["seconds"] = round(time.monotonic() - t0, 1)
out["maxrss_gib"] = round(int(open("/proc/self/status").read().split("VmHWM:")[1].split()[0]) / 2**20, 2)
json.dump(out, sys.stdout, indent=1)
print()
