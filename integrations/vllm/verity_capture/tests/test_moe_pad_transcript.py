"""R17-42 Stage 3a: `verity_capture.moe_pad_transcript` reads the padded fixed-slot assignment off a MOE-01 record (torch-free, synthetic row).

A row directory with `match/instances.jsonl` in the collector's shape (router -> TOPK pair rows whose ids ref points at the router output's
element TOPK+k, GemmW alpha ref at element k, MoeSum over the W outputs in k order) and a `match/capture` snapshot store carrying the router
outputs.  The OK case maps every (token, layer) to E slots (TOPK from the pairs, E-TOPK bot-assigned) with the expanded / compact populations
as separate quantities; the negatives are reported BY NAME: a missing pair row (incomplete token-layer), an out-of-range id (never a default
slot), a repeated id, MoeSum operands out of k order.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from verity_capture import moe_pad_transcript as T

E, TOPK, VPT, H, I = 4, 2, 2, 16, 16


def _view(alloc, shape, dtype, offset=0):
    return {"alloc": alloc, "offset": offset, "shape": list(shape), "strides": [int(np.prod(shape[i + 1:])) for i in range(len(shape))], "dtype": dtype, "device": "cuda:0"}


def make_row(tmp_path, token_layers, *, drop_pair=None, sum_order=None, snapshots=True):
    """token_layers: list of (step, position, layer, ids[TOPK], alpha[TOPK]).  Node ids are allocated sequentially."""
    root = tmp_path / "row"
    (root / "match" / "capture").mkdir(parents=True)
    node, event, alloc = 1000, 500, 600
    inst, snaps = [], []
    for ti, (step, pos, layer, ids, alpha) in enumerate(token_layers):
        rnode, ev = node, event
        a_alloc, i_alloc = alloc, alloc + 1
        inst.append({"node": rnode, "step": step, "event": ev, "spec": f"MoeRouterTopK_v1{{E={E},TOPK={TOPK},VPT={VPT}}}", "row": 0, "rows": 1,
                     "name": f"t={pos}/layers.{layer}/router", "request": "r0", "position": pos,
                     "output_views": [_view(a_alloc, (TOPK,), "float32"), _view(i_alloc, (TOPK,), "int32")], "operand_refs": [[["n", rnode - 1, 0, E]]]})
        if snapshots:
            fa, fi = root / "match" / "capture" / f"a{ti}.bin", root / "match" / "capture" / f"i{ti}.bin"
            np.asarray(alpha, dtype=np.float32).tofile(fa)
            np.asarray(ids, dtype=np.int32).tofile(fi)
            snaps.append({"event": ev, "arg": "out0", "view": _view(a_alloc, (TOPK,), "float32"), "file": fa.name, "byte_offset": 0, "nbytes": 4 * TOPK, "step": step})
            snaps.append({"event": ev, "arg": "out1", "view": _view(i_alloc, (TOPK,), "int32"), "file": fi.name, "byte_offset": 0, "nbytes": 4 * TOPK, "step": step})
        node += 1
        gemm_nodes, w_nodes = [], []
        for k in range(TOPK):
            if drop_pair == (ti, k):
                continue
            inst.append({"node": node, "step": step, "event": ev + 1, "spec": f"MoeExpertGemm_v1{{E={E},K={H},N={2 * I}}}", "row": k, "rows": TOPK,
                         "name": f"layers.{layer}/w13_weight", "request": None, "position": None, "output_views": [_view(700, (2 * I,), "bfloat16")],
                         "operand_refs": [[["n", rnode - 1, 0, H]], [["n", rnode, TOPK + k, TOPK + k + 1]], [["p", 1, 0, E * 2 * I * H]]]})
            gemm_nodes.append(node)
            node += 1
        for k in range(TOPK):
            if drop_pair == (ti, k):
                continue
            inst.append({"node": node, "step": step, "event": ev + 2, "spec": f"MoeExpertGemmW_v1{{E={E},K={I},N={H}}}", "row": k, "rows": TOPK,
                         "name": f"layers.{layer}/w2_weight", "request": None, "position": None, "output_views": [_view(701, (H,), "bfloat16", offset=k * H)],
                         "operand_refs": [[["n", gemm_nodes[min(k, len(gemm_nodes) - 1)], 0, I]], [["n", rnode, TOPK + k, TOPK + k + 1]], [["n", rnode, k, k + 1]], [["p", 1, 0, E * H * I]]]})
            w_nodes.append(node)
            node += 1
        order = sum_order if sum_order is not None else list(range(len(w_nodes)))
        inst.append({"node": node, "step": step, "event": ev + 3, "spec": f"MoeSum_v1{{TOPK={TOPK},H={H}}}", "row": 0, "rows": 1,
                     "name": f"t={pos}/layers.{layer}/moe_sum", "request": "r0", "position": pos, "output_views": [_view(702, (H,), "bfloat16")],
                     "operand_refs": [[["n", w_nodes[j], 0, H] for j in order]]})
        node += 1
        event += 10
        alloc += 2
    with open(root / "match" / "instances.jsonl", "w") as f:
        for r in inst:
            f.write(json.dumps(r) + "\n")
    if snapshots:
        json.dump({"schema": "verity-capture/snapshots/v1", "layout": "spanned-storage-bytes", "entries": snaps}, open(root / "match" / "capture" / "snapshots.json", "w"))
    return str(root)


GOOD = [(0, 0, 0, [3, 1], [0.7, 0.3]), (0, 1, 0, [0, 2], [0.6, 0.4]), (1, 2, 0, [2, 3], [0.55, 0.45]), (1, 2, 1, [1, 0], [0.9, 0.1])]


def test_ok_record_maps_every_token_layer_to_e_slots_with_the_populations_separate(tmp_path):
    out = T.build(make_row(tmp_path, GOOD), examples=4)
    assert out["verdict"] == "OK" and out["n_anomalies"] == 0
    assert out["envelope"] == {"E": E, "TOPK": TOPK, "VPT": VPT, "H": H, "I2": 2 * I, "I": I}
    assert out["token_layers"] == out["complete"] == out["with_recorded_ids"] == out["ids_admissible"] == 4
    p = out["population"]
    assert p["expanded_slot_bodies"] == 4 * E and p["compact_executed_pairs"] == 4 * TOPK and p["expansion_factor"] == E / TOPK
    assert p["disabled_slots"] == 4 * (E - TOPK) and p["disabled_internal_module_words"] == 4 * (E - TOPK) * (H + 1)
    assert p["disabled_internal_fine_words"] == 4 * (E - TOPK) * (3 * I + H + 1)
    assert p["per_step"] == {"0": {"token_layers": 2, "pairs_executed": 4}, "1": {"token_layers": 2, "pairs_executed": 4}}
    ex = {(e["step"], e["position"], e["layer"]): e for e in out["examples"]}
    e0 = ex[(0, 0, "layers.0")]
    assert e0["ids"] == [3, 1] and e0["alpha"] == pytest.approx([0.7, 0.3])
    assert sorted(e0["enabled_slots"]) == [1, 3] and e0["enabled_slots"][3]["k"] == 0 and e0["enabled_slots"][1]["k"] == 1
    assert e0["enabled_slots"][3]["x"].startswith("wired")          # R17-43: the slot input is the token's activation, never replaced
    assert e0["disabled_slots"] == [0, 2] and e0["n_disabled_internal_module_words"] == (E - TOPK) * (H + 1)
    assert out["slot_usage_histogram"] == {"0": 2, "1": 2, "2": 2, "3": 2}


def test_a_missing_pair_row_is_an_incomplete_token_layer_by_name(tmp_path):
    out = T.build(make_row(tmp_path, GOOD, drop_pair=(2, 1)))
    assert out["verdict"] == "ANOMALIES" and out["incomplete"] == 1 and out["complete"] == 3
    kinds = {a["kind"] for a in out["anomalies"]}
    assert "incomplete-token-layer" in kinds
    assert out["population"]["compact_executed_pairs"] == 4 * TOPK - 1


def test_an_out_of_range_or_repeated_id_is_inadmissible_never_a_default_slot(tmp_path):
    bad = [(0, 0, 0, [E + 2, 1], [0.7, 0.3]), (0, 1, 0, [2, 2], [0.5, 0.5]), (1, 2, 0, [0, 3], [0.6, 0.4])]
    out = T.build(make_row(tmp_path, bad), examples=3)
    assert out["verdict"] == "ANOMALIES" and out["ids_admissible"] == 1 and out["n_ids_inadmissible"] == 2
    whys = sorted(a["why"] for a in out["ids_inadmissible"])
    assert whys == ["out-of-range", "repeated"]
    # the inadmissible token-layers contribute NO slot (no default slot 0): the histogram counts only the admissible one
    assert out["slot_usage_histogram"] == {"0": 1, "1": 0, "2": 0, "3": 1}
    assert all("enabled_slots" not in e for e in out["examples"] if e["ids"] in ([E + 2, 1], [2, 2]))


def test_moesum_operands_out_of_k_order_are_named(tmp_path):
    out = T.build(make_row(tmp_path, GOOD[:1], sum_order=[1, 0]))
    assert out["verdict"] == "ANOMALIES"
    assert [a["kind"] for a in out["anomalies"]] == ["moesum-operands-not-in-k-order"]


def test_without_a_snapshot_store_the_structure_is_mapped_but_the_verdict_is_structure_only(tmp_path):
    out = T.build(make_row(tmp_path, GOOD, snapshots=False))
    assert out["complete"] == 4 and out["with_recorded_ids"] == 0 and out["slot_usage_histogram"] is None
    assert [a["kind"] for a in out["anomalies"]] == ["no-snapshot-store"]
    assert out["verdict"] == "STRUCTURE-ONLY"          # never OK without the ids of every token-layer read off the record (F-r17b-44 (1))


def test_cli_writes_the_assignment_beside_the_row(tmp_path, capsys):
    root = make_row(tmp_path, GOOD)
    assert T.main([root]) == 0
    assert os.path.exists(os.path.join(root, "moe_pad_assignment.json"))
    assert "[moe-pad] OK" in capsys.readouterr().out
