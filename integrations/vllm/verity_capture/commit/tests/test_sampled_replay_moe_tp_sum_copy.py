"""[MOE-TP-02 / moe, tp M-1528] On a TP rank the MoeSum row's ONE output is a required identity under TWO names: the MoE module's
`part_rank<r>` (tp_rank_partials -- TP-04/TP-06, the all-reduce input the rank committed) and the MOE-01 plane `moe/L<k>/sum`
(moe_block_stream -- MoeBlockSource stages it on the rank exactly as at TP1).  `required_manifest` requires both (TP-06 test); the replay
addressed the row at the partial only (TP-06 in `describe`), so F-r17b-38's identity-by-identity reconciliation read every `moe/L<k>/sum`
identity as `identities_without_rows` -- OLMoE-1B-7B TP2 B8 1024/128 on r17-int-6 (tp M-1528): 6,944 per rank = 434 rows x 16 layers,
COMMIT FAIL by the population gate.  The fix records the plane address as a `copy` of the same output: the population reconciles, and the
evaluation reads the plane copy back and compares it against the same evaluated words (two committed copies, one relation, both checked).
TP1 is byte-identical: no all-reduce under the MoE module -> no copy."""
from __future__ import annotations

from verity_capture.commit import required_manifest as RM
from verity_capture.commit import sampled_replay as SR
from verity_capture.commit.tests.test_required_manifest_tp import H, W, _tp_program


def _tp_moe_program(rank: int, *, layers: int = 2, I: int = 96, E: int = 8, TOPK: int = 2, tp: bool = True) -> dict:
    """The TP-06 manifest test's rank Program: layer 1's dense MLP replaced by a fused-MoE block (router, TOPK x (up, act, down), sum) under
    `model.model.layers.1.mlp.experts`, whose all-reduce that module issues -- with the block's pair-major DATAFLOW wired (the plane mapping
    `_plane_dataflow_error` checks: up p reads router t's id slot k, act p reads up p, down p reads act p + router t's id and weight slot k,
    sum t reads down t*TOPK..t*TOPK+TOPK-1).  `tp=False`: the same Program without any collective row (a TP1 Program)."""
    prog = _tp_program(rank, layers=layers)
    rows = prog["instances"]["rows"]
    out: list[dict] = []
    wires: list[tuple[dict, list[list[list]]]] = []                              # (row, args built from row REFERENCES) -> indices after numbering
    for r in rows:
        name = r["name"]
        mod = name.split("/")[0]
        if not tp and name.endswith("/all_reduce"):
            continue
        if mod.startswith("model.model.layers.1.mlp."):
            if mod.endswith("gate_up_proj"):
                sfx = name.split("[")[1].rstrip("]")
                step_tag = name.split("/")[1]
                mx = "model.model.layers.1.mlp.experts"
                router = {"i": 0, "spec": f"MoeRouterTopK_v1{{E={E},TOPK={TOPK}}}", "form": "call", "name": f"{mx}/router_{step_tag}[{sfx}]", "args": []}
                out.append(router)
                downs: list[dict] = []
                for k in range(TOPK):
                    up = {"i": 0, "spec": f"MoeExpertGemm_v1{{K={H},N={2 * I}}}", "form": "call", "name": f"{mx}/up_{step_tag}_{k}[{sfx}]", "args": []}
                    act = {"i": 0, "spec": f"SiluMul_v1{{I={I}}}", "form": "call", "name": f"{mx}/act_{step_tag}_{k}[{sfx}]", "args": []}
                    down = {"i": 0, "spec": f"MoeExpertGemmW_v1{{K={I},N={H}}}", "form": "call", "name": f"{mx}/down_{step_tag}_{k}[{sfx}]", "args": []}
                    out += [up, act, down]
                    wires.append((up, [[["n", "x", 0, H]], [["n", router, TOPK + k, TOPK + k + 1]]]))
                    wires.append((act, [[["n", up, 0, 2 * I]]]))
                    wires.append((down, [[["n", act, 0, I]], [["n", router, TOPK + k, TOPK + k + 1]], [["n", router, k, k + 1]]]))
                    downs.append(down)
                s = {"i": 0, "spec": f"MoeSum_v1{{H={H},TOPK={TOPK}}}", "form": "call", "name": f"{mx}/moe_sum_{step_tag}[{sfx}]", "args": []}
                out.append(s)
                wires.append((s, [[["n", d, 0, H] for d in downs]]))
                if tp:
                    out.append({"i": 0, "spec": f"AllReduce2_v1{{N={H}}}", "form": "call", "name": f"{mx}/all_reduce2_{step_tag}[{sfx}]/all_reduce", "args": []})
            continue
        out.append(r)
    for i, r in enumerate(out):
        r["i"] = i
    for r, args in wires:
        r["args"] = [[[a[0], (a[1]["i"] if isinstance(a[1], dict) else 0), a[2], a[3]] for a in grp] for grp in args]
    prog["instances"]["rows"] = out
    return prog


def _moe_sum_rows(pi: SR.ProgramIndex) -> list[dict]:
    return [r for r in pi.rows if RM.spec_family(r["spec"]) == "MoeSum_v1"]


def test_tp_rank_moe_sum_row_is_addressed_at_its_partial_and_its_plane_copy():
    prog = _tp_moe_program(1)
    pi = SR.ProgramIndex(prog["instances"], tp={"rank": 1, "world": W})
    sums = _moe_sum_rows(pi)
    assert sums, "fixture carries no MoeSum rows"
    for r in sums:
        d = pi.describe(r)
        assert d is not None and d["op_path"] == "model.layers.1.mlp.experts", d
        assert [m[0] for m in d["members"]] == ["part_rank1"], d["members"]                     # TP-06: the partial, unchanged
        assert "copy_error" not in d, d
        (c,) = d["copies"]
        assert (c["op_path"], c["member"], c["width"], c["dtype"], c["of"]) == ("moe/L1", "sum", H, "bf16", "part_rank1"), c
        # the plane copy's row is the checked plane ordinal (token t within the step), the partial's is the request-local row: equal here
        assert c["row"] == d["row"], (c, d["row"])


def test_tp_rank_population_reconciles_the_sum_plane_identities_the_manifest_requires():
    """The record's failure shape (tp M-1528) and its closure: on the TP rank manifest every moe/L1/sum identity has a producing row."""
    prog = _tp_moe_program(1)
    man = RM.build_manifest(prog, {"sweep": {"row_id": "tp2-moe-sum-copy", "batch": 1, "arrive_steps": [0]}}, rank=1, ranks=W)
    assert man["complete"] is True, man["unmodelled"]
    sum_ids = [i for i in man["identities"] if i["family"] == "moe_block_stream" and i["output_member"] == "sum"]
    parts = [i for i in man["identities"] if i["op_path"] == "model.layers.1.mlp.experts" and i["output_member"] == "part_rank1"]
    assert sum_ids and parts and len(sum_ids) == len(parts) == len(man["steps"]), (len(sum_ids), len(parts))      # both required, per step
    pi = SR.ProgramIndex(prog["instances"], tp={"rank": 1, "world": W})
    addresses: dict = {}
    SR.population(pi, "r0", rank=1, addresses=addresses)
    for i in sum_ids + parts:
        k = (str(i["request_id"]), int(i["step"]), str(i["op_path"]), str(i["output_member"]))
        assert k in addresses, (k, sorted(a for a in addresses if a[2].startswith("moe/") or "experts" in a[2]))
        assert addresses[k]["rows"] == int(i["rows"]), (k, addresses[k], i["rows"])              # rows agree with the manifest's count
    q = SR.query_population(man, addresses, set(), ranks=[1])
    chk = q["query_checks"]
    assert chk["identities_without_rows"]["n"] == 0, chk["identities_without_rows"]
    assert chk["rows_disagree"]["n"] == 0 and chk["vus_outside_query"]["n"] == 0, chk
    # the pre-fix shape, reproduced on the same manifest: drop the copies and the sum identities are exactly the ones without rows
    bare = {k: v for k, v in addresses.items() if not (k[2] == "moe/L1" and k[3] == "sum")}
    q0 = SR.query_population(man, bare, set(), ranks=[1])
    assert q0["query_checks"]["identities_without_rows"]["n"] == len(sum_ids)
    assert all(m["op_path"] == "moe/L1" and m["member"] == "sum" for m in q0["query_checks"]["identities_without_rows"]["first"])
    assert q0["query_checks"]["ok"] is False


def test_tp1_moe_sum_row_has_no_copy_and_is_the_plane_itself():
    """No all-reduce under the MoE module (TP1: the row is the MOE-01 plane, one name) -> no copy, byte-identical to before."""
    prog = _tp_moe_program(0, tp=False)
    pi = SR.ProgramIndex(prog["instances"])
    for r in _moe_sum_rows(pi):
        d = pi.describe(r)
        assert d is not None and d["op_path"] == "moe/L1" and [m[0] for m in d["members"]] == ["sum"], d
        assert "copies" not in d and "copy_error" not in d
