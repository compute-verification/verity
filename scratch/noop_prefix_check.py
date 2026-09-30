"""Is the cap Program's executed prefix already the length-L Program?  (proofs-noop design study, experiment 0)

For `Serve_v1{TINY, LP=4, STEPS}` (the pure-IR greedy request, one root node per step piece), compare the cap Program
(STEPS = cap-1) with each length-L Program (STEPS = L-1):

1. root body: the length-L Program's nodes are the cap Program's first nodes (form, callee descriptor id, operand refs, type);
2. layout: the same gate offsets on that prefix;
3. partitions: `Q_word{16,32}` v1 and `Q_template_instances` give the length-L Program the cap Program's first units;
4. values: the length-L Program's outputs are the cap Program's first L outputs on the same inputs;
5. the output-prefix rule: output k is produced by root node j_k, j is increasing, and the length-L prefix is nodes [0, j_{L-1}];
6. committed sets: which values the cap Program's partition commits in the prefix that the length-L Program's does not;
7. cost: gates of the prefix against the cap.

Run: PYTHONPATH=packages/verity/src:integrations/vllm python scratch/noop_prefix_check.py
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np

from verity.ir import Program, bind
from verity.ir.codec import descriptor_id
from verity.ir import partition_object as PO
from verity_vllm.program.registry import b1
from verity_vllm.program.registry import serve3_reference as ref
from verity_vllm.program.registry.b1 import weights_type

TINY = {"model": "tiny", "revision": None, "H": 32, "NL": 2, "NH": 2, "KVH": 1, "D": 16, "I": 32, "V": 64,
        "QKV": 64, "GU": 64, "EPS": 1e-5, "ROPE_THETA": 1e4, "MAX_POS": 16, "BIAS": True,
        "tie_word_embeddings": True, "profile": {}}
LP, CAP = 4, 9


def program(tokens: int) -> Program:
    return Program(bind(b1.Serve, C=TINY, LP=LP, STEPS=tokens - 1))


def inputs() -> list[list[int]]:
    rng = np.random.default_rng(1001)
    w = ((rng.standard_normal(weights_type(TINY).leaves) * 0.05).astype(np.float32).view(np.uint32) >> 16)
    prompt = np.random.default_rng(2002).integers(0, TINY["V"], LP)
    return [[int(x) for x in prompt], [int(x) for x in w]]


def node_key(n) -> tuple:
    return (n.form, descriptor_id(n.fn), repr([(repr(a.type), repr(a.refs)) for a in n.args]), repr(n.out),
            getattr(n, "axes", None), getattr(n, "n", None))


def producer_nodes(p: Program) -> list[int]:
    """Output leaf k -> the root node whose gates hold it."""
    offs = p.root.body.offsets
    out = []
    for g in p.circuit.root_outputs():
        out.append(max(k for k in range(len(p.root.body.nodes)) if offs[k] <= g))
    return out


def main() -> int:
    t0 = time.perf_counter()
    cap = program(CAP)
    ins = inputs()
    tc = ref.materialize(cap, ins)
    y_cap = [tc[g] for g in cap.circuit.root_outputs()]
    j_cap = producer_nodes(cap)
    tids = sorted({descriptor_id(fn) for _k, fn in PO.calls(cap)})
    q_ti = PO.template_instances_query(tids)
    part_w_cap = PO.evaluate(cap, PO.query())
    part_t_cap = PO.evaluate(cap, q_ti)
    committed_cap = set(part_t_cap.committed())
    rows = []
    ok = all(a < b for a, b in zip(j_cap, j_cap[1:]))
    for L in (1, 4, CAP):
        p = program(L)
        nodes_L, nodes_c = p.root.body.nodes, cap.root.body.nodes
        k = len(nodes_L)
        same_nodes = [node_key(a) for a in nodes_L] == [node_key(b) for b in nodes_c[:k]]
        same_offsets = list(p.root.body.offsets[:k + 1]) == list(cap.root.body.offsets[:k + 1])
        pw = PO.evaluate(p, PO.query())
        same_qword = [(kk, c.units) for kk, c in pw.calls] == [(kk, c.units) for kk, c in part_w_cap.calls if kk < k]
        pt = PO.evaluate(p, PO.template_instances_query(sorted({descriptor_id(fn) for _k, fn in PO.calls(p)})))
        same_ti = [r[:2] + r[2:] for r in pt.ranges] == [r for r in part_t_cap.ranges if r[0] < k]
        t = ref.materialize(p, ins)
        y = [t[g] for g in p.circuit.root_outputs()]
        same_values = y == y_cap[:L] and all(t[g] == tc[g] for g in range(p.gates))
        rule_k = j_cap[L - 1] + 1
        committed_L = set(pt.committed())
        prefix_gates = p.gates
        extra = sorted(g for g in committed_cap if g < prefix_gates and g not in committed_L)
        row = {"L": L, "root_nodes": k, "output_prefix_rule_nodes": rule_k, "rule_matches": rule_k == k,
               "same_nodes": same_nodes, "same_offsets": same_offsets, "same_qword_units": same_qword,
               "qword_population": pw.population, "cap_qword_units_before": part_w_cap._start[len(pw.calls)] if len(pw.calls) < len(part_w_cap.calls) else part_w_cap.population,
               "same_template_units": same_ti, "template_population": pt.population,
               "same_values": same_values, "outputs": y,
               "gates": p.gates, "gates_fraction_of_cap": round(p.gates / cap.gates, 4),
               "committed_in_prefix_cap_only": len(extra), "committed_L": len(committed_L)}
        ok = ok and same_nodes and same_offsets and same_qword and same_ti and same_values and rule_k == k
        rows.append(row)
    per_step = [cap.root.body.offsets[j_cap[s] + 1] - (cap.root.body.offsets[j_cap[s - 1] + 1] if s else cap.input_gates)
                for s in range(CAP)]
    print(json.dumps({"program": f"Serve_v1{{TINY, LP={LP}, STEPS={CAP - 1}}}", "cap_tokens": CAP, "cap_gates": cap.gates,
                      "cap_input_gates": cap.input_gates, "cap_root_nodes": len(cap.root.body.nodes),
                      "cap_qword_population": part_w_cap.population, "cap_template_population": part_t_cap.population,
                      "templates": len(tids), "output_producers_increasing": all(a < b for a, b in zip(j_cap, j_cap[1:])),
                      "gates_per_step": per_step, "cap_outputs": y_cap, "rows": rows, "all_ok": ok,
                      "seconds": round(time.perf_counter() - t0, 1)}, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
