"""CIR-274 DENSE join trial: rank 0's window with all six family sets (frame, dense, sparse, multimem, embed, select) at the
test's GLM request Program, and the three facts the lead expects: only StopCheck unlowered; no free rows of TokenSelect, the
decode's shard reads taking ``tok`` by wire from the select's combine units; the prefill's shard reads still row 0 of the
``SIZE=192`` combine's ``parts``. The 5-set window beside it, to compare with EMBED's digest.

Usage (from adapters/vllm): join6_census.py OUT [--ids 43] [--layers 4] [--no-five]   Writes $OUT/join6_census_<layers>l.json."""
import argparse
import json
import resource
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, "tests/program/predict")
from test_node_window import glm_request  # noqa: E402

from verity.commitments.windows import interface as IF  # noqa: E402
from verity_vllm.program.predict import node_window as NW  # noqa: E402
from verity_vllm.program.predict import node_window_dense as ND  # noqa: E402
from verity_vllm.program.predict import node_window_embed as NE  # noqa: E402
from verity_vllm.program.predict import node_window_multimem as NM  # noqa: E402
from verity_vllm.program.predict import node_window_select as NSEL  # noqa: E402
from verity_vllm.program.predict import node_window_sparse as NS  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--ids", type=int, default=43)
ap.add_argument("--layers", type=int, default=4)
ap.add_argument("--no-five", action="store_true")
a = ap.parse_args()
T0 = time.perf_counter()
path = Path(a.out, f"join6_census_{a.layers}l.json")
FIVE = {**NW.FAMILIES, **ND.FAMILIES, **NS.FAMILIES, **NM.FAMILIES, **NE.FAMILIES}
SIX = {**FIVE, **NSEL.FAMILIES}
p = glm_request(a.ids, a.layers)
fn = p.program.fn
rec: dict = {"ids": a.ids, "layers": a.layers, "outputs": 2, "rank": 0, "program_digest": p.digest,
             "nodes": len(fn.body.nodes)}


def save():
    rec["wall_s"] = round(time.perf_counter() - T0, 1)
    rec["peak_rss_gib"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2)
    path.write_text(json.dumps(rec, indent=1, default=str))


def summary(c, w):
    cen = w.census(c)
    return {"digest": cen["digest"], "units": sum(c.units.values()), "templates": len(c.ids),
            "wire_patterns": len(c.wires), "unlowered": cen["unlowered"],
            "free_words_by_kind": {k: v["words"] for k, v in cen["free"].items()},
            "free": cen["free"], "units_by_family": cen["units"], "templates_by_family": cen["templates"]}


if not a.no_five:
    c5, w5 = NW.node_window(fn, families=FIVE)
    rec["five"] = summary(c5, w5)
    print("five", {k: v for k, v in rec["five"].items() if k not in ("free", "units_by_family", "templates_by_family")},
          flush=True)
    save()
    del c5, w5
c, w = NW.node_window(fn, families=SIX)
rec["six"] = summary(c, w)
rec["six"]["settle"] = "passed (NodeWindow.circuit: settle, check_proof_block per template, check_wires, check_inputs)"
IF.check_wires(c.layouts, c.wires, c.cuts, c.order)
fns = dict(zip(c.ids, (f for f, _n in c.templates)))
refused = {}
for t, f in fns.items():
    try:
        IF.check_proof_block(f, c.cuts.get(t))
    except Exception as e:  # noqa: BLE001 - a refusal is a result
        refused[t] = f"{type(e).__name__}: {e}"[:300]
rec["six"]["check_wires"] = "passed"
rec["six"]["check_proof_block"] = {"templates": len(fns), "refused": refused}
print("six", {k: v for k, v in rec["six"].items() if k not in ("free", "units_by_family", "templates_by_family")}, flush=True)

sel = {t for t in c.ids if t.startswith(("GreedyChunk_v2{", "GreedyCombine_v2{"))}
comb = next(t for t in sel if t.startswith("GreedyCombine_v2{"))
read = next(t for t in c.ids if t.startswith("EmbeddingRead_v1"))
facts: dict = {}
facts["1_only_stopcheck_unlowered"] = rec["six"]["unlowered"] == {"StopCheck": 1}
facts["2_no_tokenselect_free"] = "TokenSelect" not in rec["six"]["free"] and not [f for f in w.free if f.family == "TokenSelect"]
tok_wires = [x for x in c.wires if x.reader == read and x.port == "tok"]
facts["2_read_tok_wires"] = [{"writer": x.writer, "out": x.out, "at": x.at, "radix": list(x.radix), "base": x.base,
                              "step": list(x.step), "cut": list(x.cut), "reader_units": x.reader_units} for x in tok_wires]
from_comb = [x for x in tok_wires if x.writer == comb]
facts["2_read_units_tok_from_select"] = sum(x.reader_units for x in from_comb)
facts["2_select_combine_units_read"] = sorted({int(s) for x in from_comb for s in x.writers(x.at, x.reader_units)})
facts["2_read_tok_free"] = [{"kind": f.kind, "family": f.family, "name": f.name, "at": f.at, "count": f.count}
                            for f in w.free if f.tid == read and f.port == "tok"]
facts["2_holds"] = facts["2_no_tokenselect_free"] and facts["2_read_units_tok_from_select"] == 32
prefill = [x for x in c.wires if x.writer == read and x.reader.startswith("AllReduceMultimem_v1")]
facts["3_read_to_allreduce_wires"] = [{"reader": x.reader, "port": x.port, "out": x.out, "at": x.at, "radix": list(x.radix),
                                       "base": x.base, "step": list(x.step), "cut": list(x.cut),
                                       "reader_units": x.reader_units} for x in prefill]
facts["3_holds"] = (len(prefill) >= 1 and all(x.port == "parts" and x.cut[0] == 0 and "SIZE=192" in x.reader for x in prefill)
                    and sum(x.reader_units for x in prefill) == a.ids * 32)
facts["3_combine"] = {x.reader: {"units": c.units[x.reader], "cuts": c.cuts.get(x.reader)} for x in prefill}
facts["read_units"] = c.units[read]
facts["read_wires_out"] = dict(Counter(x.reader for x in c.wires if x.writer == read))
rec["facts"] = facts
rec["select"] = {"templates": len(sel), "units": sum(c.units[t] for t in sel),
                 "free_rows": sum(f.count for f in w.free if f.tid in sel),
                 "combine_out_readers": dict(Counter(x.reader for x in c.wires if x.writer == comb))}
rec["free_ports"] = dict(sorted(Counter(f"{f.tid.split('{')[0]}/{f.port}: {f.kind} {f.family}" for f in w.free).items()))
if not a.no_five:
    rec["six_minus_five"] = {"units": rec["six"]["units"] - rec["five"]["units"],
                             "templates": rec["six"]["templates"] - rec["five"]["templates"],
                             "wire_patterns": rec["six"]["wire_patterns"] - rec["five"]["wire_patterns"],
                             "units_by_family": {f: rec["six"]["units_by_family"].get(f, 0) - rec["five"]["units_by_family"].get(f, 0)
                                                 for f in sorted(set(rec["six"]["units_by_family"]) | set(rec["five"]["units_by_family"]))
                                                 if rec["six"]["units_by_family"].get(f, 0) != rec["five"]["units_by_family"].get(f, 0)}}
save()
print("facts", json.dumps(facts, default=str)[:4000], flush=True)
print("select", rec["select"], flush=True)
print("six_minus_five", rec.get("six_minus_five"), flush=True)
