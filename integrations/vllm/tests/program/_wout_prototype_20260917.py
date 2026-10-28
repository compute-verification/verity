"""Prototype: per-body Out(S)/W_out without expanded-circuit traversal, checked against GateSelection.out (oracle).

A *part* is (scope, items) where scope is one activation of a specialization S and items are (node j, member m) of S's
body (m = 0 for primitive/call nodes).  Out(part) = gates of the part that are (a) consumed by a body item outside the
part, or (b) returned by S's body (escape to the parent; exact when S is the root, conservative otherwise)."""
import sys
sys.path.insert(0, "/private/tmp/lane-coverage")
from verity.ml.circuits.defs import PrimitiveDefinition, bind
from verity.ml.circuits.layout import Scope, GateSelection, resolve
from verity.ml.circuits.program import Program
from verity.ml.circuits.refs import Affine, Concat, Strided, runs
from verity.ml.definitions.registry import b1


def node_leaf_gate(sc: Scope, j: int, leaf: int) -> int:
    """Root gate producing output leaf `leaf` of node j of body sc (same walk as layout.resolve, one step)."""
    return resolve(sc, ("n", j, leaf))


def item_interval(sc: Scope, j: int, m: int):
    body = sc.spec.body
    node = body.nodes[j]
    lo = sc.offset + body.offsets[j] + m * node.fn.gates
    return lo, lo + node.fn.gates


def consumed_leaves(sc: Scope, jp: int, mp: int):
    """(producer node j, lo, hi) leaf ranges consumed by item (jp, mp) of body sc, n-space only."""
    body = sc.spec.body
    node = body.nodes[jp]
    out = []
    if node.form in ("primitive", "call"):
        for a in node.args:
            out.extend((idx, lo, hi) for sp, idx, lo, hi in runs(a.refs) if sp == "n")
    elif node.form == "batch":
        for a, ax in zip(node.args, node.axes):
            if ax is None:
                refs = a.refs
            elif ax == 0:
                L = a.type.elem.leaves
                refs = a.refs.slice(mp * L, (mp + 1) * L)
            else:
                refs = a.take(ax, mp).refs
            out.extend((idx, lo, hi) for sp, idx, lo, hi in runs(refs) if sp == "n")
    else:  # scan: (init, *xs, *shared); the carry of iteration mp>0 is iteration mp-1's return (intra-node edge)
        for pi, a in enumerate(node.args):
            if pi == 0:
                if mp == 0:
                    out.extend((idx, lo, hi) for sp, idx, lo, hi in runs(a.refs) if sp == "n")
                else:
                    out.append(("carry", jp, mp - 1))
            elif pi <= node.n_xs:
                L = a.type.elem.leaves
                out.extend((idx, lo, hi) for sp, idx, lo, hi in runs(a.refs.slice(mp * L, (mp + 1) * L)) if sp == "n")
            else:
                out.extend((idx, lo, hi) for sp, idx, lo, hi in runs(a.refs) if sp == "n")
    return out


def leaf_to_member(node, leaf: int) -> int:
    if node.form in ("primitive", "call"):
        return 0
    if node.form == "batch":
        return leaf // node.fn.out_leaves
    c = node.out.fields[0].leaves
    if leaf < c:
        return node.n - 1
    y = node.out.fields[1].elem.leaves
    return (leaf - c) // y


def out_pred(sc: Scope, items: set[tuple[int, int]], designated: set[int]) -> set[int]:
    body = sc.spec.body
    esc: set[int] = set()
    by_node: dict[int, set[int]] = {}
    for j, m in items:
        by_node.setdefault(j, set()).add(m)
    # (a) consumers in the body outside the part
    for jp, nodep in enumerate(body.nodes):
        nmem = nodep.n if nodep.form in ("batch", "scan") else 1
        for mp in range(nmem):
            if (jp, mp) in items:
                continue
            for c in consumed_leaves(sc, jp, mp):
                if c[0] == "carry":
                    _, j, m = c
                    if (j, m) in items:
                        # the carry leaves of iteration m: its returned carry gates
                        carry = body.nodes[j].out.fields[0].leaves
                        sub = Scope(body.nodes[j].fn, sc.offset + body.offsets[j] + m * body.nodes[j].fn.gates, sc, j, m)
                        for r in range(carry):
                            esc.add(resolve(sub, body.nodes[j].fn.body.ret[r]))
                    continue
                j, lo, hi = c
                if j not in by_node:
                    continue
                for leaf in range(lo, hi):
                    if leaf_to_member(body.nodes[j], leaf) in by_node[j]:
                        esc.add(node_leaf_gate(sc, j, leaf))
    # (b) returned by the body (escape to the parent / designated outputs)
    for sp, j, lo, hi in runs(body.ret):
        if sp == "n" and j in by_node:
            for leaf in range(lo, hi):
                if leaf_to_member(body.nodes[j], leaf) in by_node[j]:
                    esc.add(node_leaf_gate(sc, j, leaf))
    # only gates produced *inside* the part can be out
    ivs = [item_interval(sc, j, m) for j, m in items]
    return {g for g in esc if any(lo <= g < hi for lo, hi in ivs)}


def check(program: Program, sc: Scope, parts: list[set[tuple[int, int]]], label: str):
    C = program.circuit
    designated = set(C.root_outputs())
    n_ok = 0; bad = []
    for items in parts:
        gates = []
        for j, m in items:
            lo, hi = item_interval(sc, j, m)
            gates.extend(range(lo, hi))
        oracle = set(GateSelection(C, gates).out)
        pred = out_pred(sc, items, designated)
        if pred == oracle:
            n_ok += 1
        else:
            bad.append((sorted(items)[:3], sorted(oracle)[:5], sorted(pred)[:5]))
    widths = [sum(C.gate(g).prim.ret.w for g in out_pred(sc, items, designated)) for items in parts]
    print(f"{label}: {n_ok}/{len(parts)} parts agree with GateSelection.out; W_out max={max(widths)} "
          f"(circuit {C.size} gates)")
    for b in bad[:3]:
        print("   MISMATCH items", b[0], "oracle", b[1], "pred", b[2])
    return n_ok == len(parts), widths


def body_parts(sc: Scope, skip_input=True):
    """Every node/member of body sc as its own part (the finest hierarchy parts)."""
    parts = []
    for j, node in enumerate(sc.spec.body.nodes):
        if skip_input and getattr(node.fn, "family", None) == "Input":
            continue
        nmem = node.n if node.form in ("batch", "scan") else 1
        for m in range(nmem):
            parts.append({(j, m)})
    return parts


if __name__ == "__main__":
    # 1. GEMM: parts = every GemmCoordinate member of the batch node
    P = Program(bind(b1.Gemm, K=32, N=6))
    sc = P.circuit.root_scope
    ok1, w = check(P, sc, body_parts(sc), "Gemm_v1{K=32,N=6} members")
    # 1b. inside one GemmCoordinate member: every node of DotBf16 and the cvt
    jb = [j for j, n in enumerate(sc.spec.body.nodes) if n.form == "batch" and not isinstance(n.fn, PrimitiveDefinition)][0]
    sub = sc.child(jb, 2)
    ok1b, w = check(P, sub, body_parts(sub), "  GemmCoordinate#2 body nodes")
    sub2 = sub.child(0, 0)
    ok1c, w = check(P, sub2, body_parts(sub2), "  DotBf16 body nodes (accumulator chain)")
    # 2. RMSNorm
    P2 = Program(bind(b1.RMSNormTriton, N=32, EPS=1e-5))
    sc2 = P2.circuit.root_scope
    ok2, w2 = check(P2, sc2, body_parts(sc2), "RMSNormTriton_v1{N=32} nodes/members")
    # 2b. merged normalize path per output j: {Bf16ToF32 w[j], F32Mul(inv,xf[j]), F32Mul, F2fpBf16}
    nodes2 = sc2.spec.body.nodes
    tail = [j for j, n in enumerate(nodes2) if n.form == "batch"][-4:]
    merged = [{(j, m) for j in tail} for m in range(nodes2[tail[0]].n)]
    ok2b, w2b = check(P2, sc2, merged, "  normalize-path merge per output")
    # 3. Attention head -> AttnBlock interior
    P3 = Program(bind(b1.AttentionHeadV3, T=40, D=16, BN=16))
    sc3 = P3.circuit.root_scope
    ok3, w3 = check(P3, sc3, body_parts(sc3), "AttentionHead_v3{T=40,D=16,BN=16} nodes")
    blocks = [j for j, n in enumerate(sc3.spec.body.nodes) if n.form == "call" and "AttnBlock" in n.fn.id]
    for jb in blocks[:2]:
        sb = sc3.child(jb, 0)
        okb, wb = check(P3, sb, body_parts(sb), f"  {sb.spec.id} body nodes/members")
        # PV chain merge per d: all AmpereBF16TcDot16 batch members m=d across the k-step batches
        pv = [j for j, n in enumerate(sb.spec.body.nodes) if n.form == "batch" and "Ampere" in n.fn.id]
        if pv:
            chain = [{(j, d) for j in pv} for d in range(sb.spec.body.nodes[pv[0]].n)]
            okc, wc = check(P3, sb, chain, f"    PV chain merge per d ({len(pv)} k-steps)")
    # 4. MoE router: scans (carry 96 bits) and NvExpf calls
    try:
        from verity_vllm.program.quarantine.ov_moe import moe as Q
        P4 = Program(bind(Q.MoeRouterTopK, E=8, TOPK=2, VPT=8))
        sc4 = P4.circuit.root_scope
        ok4, w4 = check(P4, sc4, body_parts(sc4), "MoeRouterTopK_v1{E=8,TOPK=2} nodes/iterations")
    except Exception as e:  # noqa: BLE001
        print("MoE skipped:", type(e).__name__, str(e)[:120])
    # 5. Gumbel: the Const64 gates
    try:
        from verity.ml.definitions.registry import sampling as G
        P5 = Program(bind(G.GumbelArgmaxBf16, V=8, SEED=12345, POS=3))
        sc5 = P5.circuit.root_scope
        ok5, w5 = check(P5, sc5, body_parts(sc5), "GumbelArgmaxBf16_v1{V=8} nodes")
        wide = [(j, n.fn.id) for j, n in enumerate(sc5.spec.body.nodes) if getattr(n.fn, "ret", None) is not None and n.fn.ret.w > 32]
        print("   gates wider than 32 bits:", wide)
    except Exception as e:  # noqa: BLE001
        print("Gumbel skipped:", type(e).__name__, str(e)[:160])
