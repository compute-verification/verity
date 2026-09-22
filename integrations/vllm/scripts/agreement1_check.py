#!/usr/bin/env python3
"""[SUMMARY-C] Agreement-1 test: does PROTO-A's protocol instance (e2e/population.py) name the validated
partition and its populations exactly, on a dense and on a run-aware (UnitRuns) plan?  TINY Serve_v1, plans
P3/cu=block and P1/cu=operator.  Findings: docs/vllm-poc/agreements-review.md (SUMMARY-C 2026-09-09 17:27 UTC).

    cd vllm-poc && PYTHONPATH=../src:.. ../.venv/bin/python scripts/agreement1_check.py
"""
import sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
for p in (str(HERE.parent), str(HERE.parent.parent), str(HERE.parent.parent / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)
from e2e import population as pop
from veritor.plans.canonical import canonical_plan, Classifier, PLAN_NAMES
from veritor.plans.validate import validate_plan
from veritor.plans.summary import summarize, AddressBook
from veritor.plans.verity_ir_adapter import adapt
from verity.ir import Program, bind
from verity_vllm.registry import b1
from verity.ir.codec import program_digest, encode_program
TINY = {"model": "tiny", "revision": None, "H": 32, "NL": 2, "NH": 2, "KVH": 1, "D": 16, "I": 32, "V": 64, "QKV": 64, "GU": 64, "EPS": 1e-5, "ROPE_THETA": 1e4, "MAX_POS": 16, "BIAS": True, "tie_word_embeddings": True, "profile": {}}
P = Program(bind(b1.Serve, C=TINY, LP=4, STEPS=8)); pd = program_digest(encode_program(P))
a = adapt(P, compact=True); Q = a.program; c = Q.circuit()
runs = {}
for fid, orig, m in a.unrolled: runs.setdefault(fid, []).append((a.node_index[(fid, orig)], m))
cl = Classifier({k: tuple(sorted(v)) for k, v in runs.items()})
for spec in ("P3/cu=block", "P1/cu=operator"):
    head, _, cu = spec.partition("/cu=")
    dense = canonical_plan(c, PLAN_NAMES[head], cu, cl, name=spec).plan
    compact = canonical_plan(c, PLAN_NAMES[head], cu, cl, name=spec, compact=True).plan
    vd, vc = validate_plan(c, dense), validate_plan(c, compact)
    pd_, pc_ = pop.population(Q, dense, vd, pd), pop.population(Q, compact, vc, pd)
    print(spec, "N_V/N_R", pd_.vu_count, pd_.ru_count, "| dense descriptor", pd_.plan_digest[:12], "compact descriptor", pc_.plan_digest[:12], "SAME" if pd_.plan_digest == pc_.plan_digest else "DIFFER")
    ad, ac = pop.Addresser(Q, c, dense, vd), pop.Addresser(Q, c, compact, vc)
    t = time.perf_counter(); mism = 0
    for i in range(ad.count):
        if ad.resolve(i).to_json() != ac.resolve(i).to_json(): mism += 1
    print("  Addresser.resolve dense==compact for all", ad.count, "VUs:", mism == 0, f"({time.perf_counter()-t:.2f}s)")
    s = summarize(c, compact, vc, program_digest=("verity-ir/program_digest", pd))
    sp = s.sampler_populations()
    print("  Summary.sampler_populations N_V,N_R =", sp["N_V"], sp["N_R"], "== instance vu_count/ru_count:", (sp["N_V"], sp["N_R"]) == (pc_.vu_count, pc_.ru_count))
    book = AddressBook(c, compact, vc)
    sample = range(0, ad.count, max(1, ad.count // 50))
    print("  AddressBook.vu(i).replay_unit == Addresser.resolve(i).ru (sampled indices):", all(book.vu(i).replay_unit == ac.resolve(i).ru for i in sample))
    bad = [i for i in sample if book.vu(i).intervals[0] != ac.resolve(i).interval]
    print("  AddressBook.vu(i).intervals[0] == Addresser.resolve(i).interval (sampled indices):", not bad, "" if not bad else f"-- objection 2 (part.a ignored), e.g. VU {bad[0]}: Addresser {ac.resolve(bad[0]).interval} vs partition {book.vu(bad[0]).intervals[0]}")
    print("  W_out from summary:", s.W_out, "outputs slots:", len(s.outputs.slots), "domain bits per slot:", sorted({o.domain_bits for o in s.outputs.slots}))
