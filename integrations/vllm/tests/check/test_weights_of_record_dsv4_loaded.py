"""DeepSeek-V4's non-expert weights as vLLM serves them after load: its FP8 linears keep the checkpoint's e4m3 words and upcast their e8m0
block scales to f32 (no ue8m0 requant), the stacked fields (`fused_wqa_wkv`, the compressor's `fused_wkv_wgate`, the shared experts'
gate|up from w1|w3) stack their parts, wo_a is the rank's groups viewed [g, r, d], the sink pads the rank's heads with -inf, tid2eid is
served as i32, and layer 0's `hc_attn_fn_broadcast` sums its four streams in order.  CPU, a synthetic checkpoint (real safetensors)."""
from __future__ import annotations

import json
import re

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from verity_vllm.check import weights_of_record as WR

from tests.check.test_weights_of_record import live_digests, write_safetensors
from tests.check.test_weights_of_record_routed_experts import _program

H, G, GROUPS, R, HEADS, NH, M, WORLD, RANK = 256, 128, 4, 128, 8, 16, 24, 2, 1
L0 = "model.layers.0"
REQUANT = {"weight": {"role": "fp8_w_requant_ue8m0"}, "weight_scale_inv": {"role": "fp8_ws_requant_ue8m0"}}


@pytest.fixture
def dsv4(tmp_path):
    rng = np.random.default_rng(9)

    def e4m3(*shape):
        return "F8_E4M3", rng.integers(0, 256, shape, dtype=np.uint8)

    def e8m0(*shape):
        return "F8_E8M0", rng.integers(100, 140, shape, dtype=np.uint8)

    t = {"layers.0.attn.wq_a.weight": e4m3(G, H), "layers.0.attn.wq_a.scale": e8m0(1, H // G),
         "layers.0.attn.wkv.weight": e4m3(G, H), "layers.0.attn.wkv.scale": e8m0(1, H // G),
         "layers.0.attn.wq_b.weight": e4m3(4 * G, G), "layers.0.attn.wq_b.scale": e8m0(4, 1),
         "layers.0.attn.wo_a.weight": e4m3(GROUPS * R, H), "layers.0.attn.wo_a.scale": e8m0(GROUPS * R // G, H // G),
         "layers.0.attn.wo_b.weight": e4m3(H, 2 * H), "layers.0.attn.wo_b.scale": e8m0(H // G, 2 * H // G),
         "layers.0.attn.compressor.wkv.weight": ("BF16", rng.integers(0, 1 << 16, (64, H), dtype=np.uint16)),
         "layers.0.attn.compressor.wgate.weight": ("BF16", rng.integers(0, 1 << 16, (64, H), dtype=np.uint16)),
         "layers.0.attn.attn_sink": ("F32", rng.standard_normal(HEADS).astype(np.float32)),
         "layers.0.ffn.gate.tid2eid": ("I64", rng.integers(0, 64, (10, 2), dtype=np.int64)),
         "layers.0.hc_attn_fn": ("F32", (rng.standard_normal((M, 4 * H)) * np.exp2(rng.integers(-20, 20, (M, 4 * H)))).astype(np.float32))}
    for p in ("w1", "w3", "w2"):
        t[f"layers.0.ffn.shared_experts.{p}.weight"], t[f"layers.0.ffn.shared_experts.{p}.scale"] = e4m3(H, H), e8m0(H // G, H // G)
    d = tmp_path / "snap"
    d.mkdir()
    write_safetensors(str(d / "model.safetensors"), t)
    (d / "config.json").write_text(json.dumps({"model_type": "deepseek_v4", "expert_dtype": "fp4", "quantization_config": {"weight_block_size": [G, G]}}))
    man = tmp_path / "checkpoints.json"
    man.write_text(json.dumps({"checkpoints": [{"repo": "test/dsv4", "revision": "abc", "local_path": str(d),
                                                "files": {"model.safetensors": {"sha256": WR.sha256_file(str(d / "model.safetensors"))}}}]}))
    return {"t": t, "ck": WR.CheckpointIndex(str(d), checkpoints_manifest=str(man)), "dir": tmp_path}


def _served(t):
    """Rank 1 of 2's served words of each field, from the checkpoint's words by vLLM's load rules; and each field's Build role."""
    def raw(name):
        return t[name][1]

    def up(name):
        return np.ldexp(np.float32(1), raw(name).astype(np.int32) - 127).astype(np.float32).view(np.uint32)

    half = slice(RANK * H // 2, (RANK + 1) * H // 2)
    sink = np.full(NH, np.float32(-np.inf))
    sink[:HEADS // 2] = raw("layers.0.attn.attn_sink")[RANK * HEADS // 2:]
    want = {f"{L0}.attn.fused_wqa_wkv.weight": np.concatenate([raw("layers.0.attn.wq_a.weight"), raw("layers.0.attn.wkv.weight")]),
            f"{L0}.attn.fused_wqa_wkv.weight_scale_inv": np.concatenate([up("layers.0.attn.wq_a.scale"), up("layers.0.attn.wkv.scale")]),
            f"{L0}.attn.wq_b.weight": raw("layers.0.attn.wq_b.weight")[RANK * 2 * G:(RANK + 1) * 2 * G],
            f"{L0}.attn.wq_b.weight_scale_inv": up("layers.0.attn.wq_b.scale")[RANK * 2:(RANK + 1) * 2],
            f"{L0}.attn.wo_a.weight": raw("layers.0.attn.wo_a.weight").reshape(GROUPS, R, H)[RANK * 2:(RANK + 1) * 2],
            f"{L0}.attn.wo_a.weight_scale_inv": up("layers.0.attn.wo_a.scale").reshape(GROUPS, R // G, H // G)[RANK * 2:(RANK + 1) * 2],
            f"{L0}.attn.wo_b.weight": raw("layers.0.attn.wo_b.weight")[:, RANK * H:(RANK + 1) * H],
            f"{L0}.attn.wo_b.weight_scale_inv": up("layers.0.attn.wo_b.scale")[:, RANK * 2:(RANK + 1) * 2],
            f"{L0}.attn.compressor.fused_wkv_wgate.weight": np.concatenate([raw("layers.0.attn.compressor.wkv.weight"),
                                                                           raw("layers.0.attn.compressor.wgate.weight")]),
            f"{L0}.attn.attn_sink": sink.view(np.uint32),
            f"{L0}.ffn.gate.tid2eid": raw("layers.0.ffn.gate.tid2eid").astype(np.int32).view(np.uint32),
            f"{L0}.ffn.shared_experts.gate_up_proj.weight": np.concatenate([raw(f"layers.0.ffn.shared_experts.{p}.weight")[half] for p in ("w1", "w3")]),
            f"{L0}.ffn.shared_experts.gate_up_proj.weight_scale_inv": np.concatenate([up(f"layers.0.ffn.shared_experts.{p}.scale")[RANK:RANK + 1]
                                                                                      for p in ("w1", "w3")]),
            f"{L0}.ffn.shared_experts.down_proj.weight": raw("layers.0.ffn.shared_experts.w2.weight")[:, half],
            f"{L0}.ffn.shared_experts.down_proj.weight_scale_inv": up("layers.0.ffn.shared_experts.w2.scale")[:, RANK:RANK + 1],
            f"{L0}.hc_attn_fn": raw("layers.0.hc_attn_fn").view(np.uint32),
            f"{L0}.hc_attn_fn_broadcast": torch.from_numpy(raw("layers.0.hc_attn_fn")).view(-1, 4, H).sum(dim=1).numpy().view(np.uint32)}
    roles = {f"model.{n}": REQUANT[n.rsplit(".", 1)[1]] for n in want if re.search(r"(wq_b|wo_b|fused_wqa_wkv|shared_experts\.\w+)\.weight", n)}
    roles |= {f"model.{L0}.attn.wo_a.weight": {"role": "fp8_einsum_w"}, f"model.{L0}.attn.wo_a.weight_scale_inv": {"role": "fp8_einsum_ws"},
              f"model.{L0}.attn.attn_sink": {"role": "dsv4.attn_sink", "NH": NH},
              f"model.{L0}.hc_attn_fn_broadcast": {"role": "mhc.fn_broadcast", "HC": 4, "H": H}}
    return want, roles


def _record(dsv4, want, roles):
    prog = _program([(f"model.{n}", list(w.shape), w.dtype.itemsize * 8) for n, w in want.items()])
    prog["result"]["tp"], prog["dir"] = {"world": WORLD, "rank": RANK}, str(dsv4["dir"])
    (dsv4["dir"] / "derive-report.json").write_text(json.dumps({"weight_roles": roles}))
    return WR.weights_of_record(prog, dsv4["ck"], prefix="model.")


def test_a_ranks_loaded_weights_are_the_checkpoints_words_by_vllms_load_rules(dsv4):
    want, roles = _served(dsv4["t"])
    rec = _record(dsv4, want, roles)
    assert rec["complete"] is True and rec["unresolved"] == [], rec["unresolved"]
    rules = {f["field"]: f.get("rule") for f in rec["fields"]}
    assert rules[f"model.{L0}.attn.wq_b.weight_scale_inv"] == "vLLM keeps the weight and upcasts its e8m0 scales (DeepGEMM linear: no requant)"
    ok = WR.check(rec, live_digests(want), program_digests_of_record={"e" * 128}, checkpoint_of_record={"repo": "test/dsv4", "revision": "abc"})
    assert ok["result"] is True and ok["compared"] == len(want), ok
    as_bytes = {**want, f"{L0}.attn.wq_b.weight_scale_inv": dsv4["t"]["layers.0.attn.wq_b.scale"][1][RANK * 2:(RANK + 1) * 2]}
    bad = WR.check(rec, live_digests(as_bytes), program_digests_of_record={"e" * 128}, checkpoint_of_record={"repo": "test/dsv4", "revision": "abc"})
    assert bad["result"] is False, "a served scale left as its e8m0 bytes is not the record"


def test_the_e8m0_upcast_is_vllms_shift_and_the_streams_order_shows_in_the_broadcast(dsv4):
    from verity.ml.definitions.numerics.deepgemm_twins import upcast_e8m0
    assert upcast_e8m0(np.array([0, 1, 127, 128, 254, 255], np.uint8)).tolist() == [0, 1 << 23, 0x3F800000, 0x40000000, 0x7F000000, 0x7F800000]
    fn = dsv4["t"]["layers.0.hc_attn_fn"][1].reshape(M, 4, H)
    pairwise = ((fn[:, 0] + fn[:, 1]) + (fn[:, 2] + fn[:, 3])).view(np.uint32)
    assert not np.array_equal(pairwise, _served(dsv4["t"])[0][f"{L0}.hc_attn_fn_broadcast"]), "the fixture tells the streams' order apart"


@pytest.mark.parametrize("fault", ["tid2eid out of int32", "more rank heads than the padded sink"])
def test_a_load_rule_outside_its_range_leaves_the_field_unresolved(dsv4, fault):
    want, roles = _served(dsv4["t"])
    field = f"model.{L0}.ffn.gate.tid2eid" if fault.startswith("tid2eid") else f"model.{L0}.attn.attn_sink"
    if fault.startswith("tid2eid"):
        t = dict(dsv4["t"], **{"layers.0.ffn.gate.tid2eid": ("I64", np.full((10, 2), 1 << 31, np.int64))})
        snap = dsv4["dir"] / "snap"
        write_safetensors(str(snap / "model.safetensors"), t)
        dsv4["ck"] = WR.CheckpointIndex(str(snap))
    else:
        roles[field] = {"role": "dsv4.attn_sink", "NH": 2}
        want[field[len("model."):]] = want[field[len("model."):]][:2]
    rec = _record(dsv4, want, roles)
    assert rec["complete"] is False and [u["field"] for u in rec["unresolved"]] == [field], rec["unresolved"]
