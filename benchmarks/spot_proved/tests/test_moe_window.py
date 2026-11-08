"""`moe_window.py` on a synthetic capture in `capture.py`'s format (`write_capture`: one rank, one opened MoE layer at small
statics, every recorded tensor and opened weight under its SHA-512, the rank and step roots as `stage.Capture` checks them):

* each pair's expert is its place in the launch's ``sorted_token_ids`` and ``expert_ids``, and its values are the
  capture's tensors of its token and slot;
* the match gate names a pair whose served word is not the twin's and a token whose slots are not the router's;
* the cut's tables are the ones `expert_tables` registers for the same weights, by name, so the verifier's pins are the
  registration's;
* a staged layer is a window the verifier's side takes from its records: its circuit record the one ``--circuit-only``
  derives without values, each table read pinned by the window's registration, and a forged arm staged past the gate;
* ``--fused`` cuts the weight reads into the GEMMs, which hold their declarations, and its staged units are their
  templates' on the recorded values."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
LAYER, STEP, E, TOPK, S = 3, 5, 32, 2, 3   # E = 32: a router unit's logits are a whole SHA-512 block (a statement row needs one)
G = 32
UP = {"K": 64, "N": 64, "config": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32, "num_warps": 4}}
DOWN = {"K": 64, "N": 64, "config": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "num_warps": 4}}
BLOCKS = ["--block", "w13=8", "--block", "w2=8"]
DT = {np.uint8: "torch.float8_e4m3fn", np.uint16: "torch.bfloat16", np.uint32: "torch.float32", np.int32: "torch.int32"}


@pytest.fixture(scope="module")
def MW():
    spec = importlib.util.spec_from_file_location("spot_moe_window", HERE / "moe_window.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sha(b: bytes) -> str:
    return hashlib.sha512(b).hexdigest()


def write_capture(d: Path, MW, seed: int = 0) -> dict:
    """A one-rank capture of step `STEP` whose opened MoE layer `LAYER` routes `S` tokens to `TOPK` of `E` experts: the router's
    logits and bias random and its slots the router's, the activations random, each launch's C its twin's. -> its values."""
    from verity.ml.definitions.registry import moe as MOE
    from verity_vllm import TargetProfile

    rng = np.random.default_rng(seed)
    rd, sd = d / "rank0", d / "rank0" / f"step{STEP}"
    (sd / "open").mkdir(parents=True)
    events, params, opened = [], {}, {}

    def blob(a: np.ndarray) -> dict:
        raw = np.ascontiguousarray(a).tobytes()
        h = _sha(raw)
        (rd / "blobs" / h[:2]).mkdir(parents=True, exist_ok=True)
        (rd / "blobs" / h[:2] / h).write_bytes(raw)
        return {"sha512": h, "dtype": DT[a.dtype.type], "shape": list(a.shape)}

    def param(name: str, a: np.ndarray) -> dict:
        raw = np.ascontiguousarray(a).tobytes()
        (sd / "open" / f"{name}.bin").write_bytes(raw)
        meta = {"sha512": _sha(raw), "dtype": DT[a.dtype.type], "shape": list(a.shape)}
        params[name], opened[name] = meta, meta
        return {"param": name, "dtype": meta["dtype"], "shape": meta["shape"]}

    logits = (rng.standard_normal((S, E)) * 2).astype(np.float32).view(np.uint32)
    bias = (rng.standard_normal(E) * 1e-2).astype(np.float32).view(np.uint32)
    router = MW.bind(MOE.MoeRouterSigmoidBiasSlots, E=E, TOPK=TOPK, SCALE=2.5, KERNEL="block")
    slots = np.array([MW.evaluate(router, logits[t].tolist(), bias.tolist()) for t in range(S)], dtype=np.int64)
    g, ex = slots[:, :TOPK].astype(np.uint32), slots[:, TOPK:]
    pre = f"model.layers.{LAYER}.mlp"
    events.append({"seq": 0, "op": "aten::mm", "layer": LAYER, "in": {"self": blob(rng.integers(0, 1 << 16, (S, UP["K"]), dtype=np.uint16)),
                   "mat2": param(f"{pre}.gate.weight", rng.integers(0, 1 << 16, (UP["K"], E), dtype=np.uint16))},
                   "out": {"ret0": blob(logits)}})
    param(f"{pre}.gate.e_score_correction_bias", bias)
    order = np.argsort(ex.reshape(-1), kind="stable")
    sorted_ids, expert_ids = [], []
    for e in np.unique(ex):
        mine = [int(x) for x in order if ex.reshape(-1)[x] == e]
        for lo in range(0, len(mine), 16):
            chunk = mine[lo:lo + 16]
            sorted_ids += chunk + [S * TOPK] * (16 - len(chunk))
            expert_ids.append(int(e))
    profile = TargetProfile(compute_capability=(9, 0), batch_invariant=False)
    v = {"tokens": list(range(S)), "topk": TOPK, "g": g, "e": ex, "logits": logits, "bias": bias}
    for seq, (proj, sh, w) in enumerate((("up", UP, "w13"), ("down", DOWN, "w2")), start=1):
        K, N = sh["K"], sh["N"]
        W = MW._e4m3(rng, E, N, K)
        WS = MW._f32(rng, 1e-4, 1e-2, E, N // G, K // G)
        rows = S if proj == "up" else S * TOPK
        x, sx = MW._e4m3(rng, rows, K), MW._f32(rng, 1e-3, 1e-1, rows, K // G)
        ev = {"seq": seq, "op": "moe_launch", "layer": LAYER,
              "in": {"A": blob(x), "A_scale": blob(sx), "B": param(f"{pre}.experts.{w}_weight", W),
                     "B_scale": param(f"{pre}.experts.{w}_weight_scale_inv", WS),
                     "expert_ids": blob(np.array(expert_ids, dtype=np.int32)),
                     "sorted_token_ids": blob(np.array(sorted_ids, dtype=np.int32)),
                     "topk_weights": blob(g.copy()), "config": sh["config"], "block_shape": [G, G],
                     "mul_routed_weight": proj == "down"}, "out": {}}
        v[proj] = MW.tile(profile, ev)
        v.update({f"x{'13' if proj == 'up' else '2'}": x, f"s{'13' if proj == 'up' else '2'}": sx,
                  "W13" if proj == "up" else "W2": W, "WS13" if proj == "up" else "WS2": WS})
        events.append(ev)
    v["y13"], v["y2"] = MW.twins(v)
    events[1]["out"]["C"], events[2]["out"]["C"] = blob(v["y13"]), blob(v["y2"])
    events.append({"seq": 3, "op": "model_output", "layer": None, "in": {}, "out": {"hidden": {"shape": [S, UP["K"]]}}})
    ops = "".join(json.dumps(e) + "\n" for e in events).encode()
    pj = json.dumps(params).encode()
    (sd / "ops.jsonl").write_bytes(ops)
    (sd / "params.json").write_bytes(pj)
    rank_root = _sha(f"verity/spot/rank-root\0{_sha(ops)}\0{_sha(pj)}".encode())
    (sd / "meta.json").write_text(json.dumps({"ops_sha512": _sha(ops), "params_sha512": _sha(pj), "rank_root": rank_root,
                                              "scheduled": {"r0": S}, "check": "ok"}))
    root = _sha(("verity/spot/step-root\0" + rank_root).encode())
    (sd / "commit.json").write_text(json.dumps({"root": root, "coins": "c0" * 32, "rank_roots": [rank_root],
                                                "layers": {"moe": [LAYER]}, "opened": opened, "kept_params": {}}))
    return v


@pytest.fixture(scope="module")
def captured(MW, tmp_path_factory):
    d = tmp_path_factory.mktemp("capture")
    return d, write_capture(d, MW)


def test_each_pair_reads_its_expert_and_its_tokens_values_from_the_capture(MW, captured):
    from verity_vllm import TargetProfile

    d, want = captured
    cap = MW.ST.Capture(d, STEP)
    got = MW.served(cap, 0, LAYER, S, TargetProfile(compute_capability=(9, 0), batch_invariant=False))
    assert got["tokens"] == list(range(S)) and got["topk"] == TOPK
    for k in ("e", "g", "logits", "bias", "x13", "s13", "x2", "s2", "W13", "WS13", "W2", "WS2", "y13", "y2"):
        assert np.array_equal(np.asarray(got[k]), np.asarray(want[k])), k
    assert got["up"]["SWAP"] is False and got["down"]["SWAP"] is True
    assert (got["up"]["twin"], got["down"]["twin"]) == ("mmav2", "hopper")
    one = MW.served(cap, 0, LAYER, 1, TargetProfile(compute_capability=(9, 0), batch_invariant=False))
    t = one["tokens"][0]
    assert np.array_equal(one["e"][0], want["e"][t]) and np.array_equal(one["x2"], want["x2"][t * TOPK:(t + 1) * TOPK])


def test_the_gate_names_a_pair_off_its_twin_and_a_token_off_its_router(MW, captured):
    from verity.ml.definitions.registry import moe as MOE

    _d, v = captured
    router = MW.bind(MOE.MoeRouterSigmoidBiasSlots, E=E, TOPK=TOPK, SCALE=2.5, KERNEL="block")
    assert MW.gate(v, router)["ok"]
    y13 = v["y13"].copy()
    y13[1, 0, 7] ^= 1
    e = v["e"].copy()
    e[2, 1] = (e[2, 1] + 1) % E
    bad = MW.gate({**v, "y13": y13}, router)["mismatched"]
    assert (bad["up"], bad["down"], bad["router"]) == ([[1, 0]], [], [])
    bad = MW.gate({**v, "e": e}, router)["mismatched"]
    assert bad["router"] == [2] and [2, 1] in bad["up"] and [2, 1] in bad["down"]


def test_the_cuts_tables_are_the_ones_expert_tables_registers(MW, captured):
    from verity_vllm.pipeline import expert_tables as XT

    _d, v = captured
    blocks = {"w13": 8, "w2": 8}
    fns = MW.templates(v, 2.5, "block", blocks)
    lab = MW.labels({"up": ("model.layers.3.mlp.experts.w13_weight", "model.layers.3.mlp.experts.w13_weight_scale_inv"),
                     "down": ("model.layers.3.mlp.experts.w2_weight", "model.layers.3.mlp.experts.w2_weight_scale_inv")}, None)
    circuit, plan = MW.cut(v, fns, lab, blocks, "test")
    fields = [(f"model.layers.3.mlp.experts.{w}_weight", (E, sh["N"], sh["K"]), 8) for w, sh in (("w13", UP), ("w2", DOWN))]
    fields += [(f"model.layers.3.mlp.experts.{w}_weight_scale_inv", (E, sh["N"] // G, sh["K"] // G), 32)
               for w, sh in (("w13", UP), ("w2", DOWN))]
    registered = {p for ts in XT.tables_of(fields, blocks).values() for _b, p, _w, _n in ts}
    assert set(circuit.registered.params) == registered
    prefixed = MW.labels({"up": ("model.layers.3.mlp.experts.w13_weight", "model.layers.3.mlp.experts.w13_weight_scale_inv")},
                         [f"weights/language_model.{p[len('weights/'):]}" for p in registered])
    assert prefixed["up"]["w"] == "language_model.model.layers.3.mlp.experts.w13_weight"
    want: dict[str, int] = {MW.descriptor_id(fns["router"]): S}
    for p, B, NBG in (("up", 8, 2), ("down", 8, 2)):
        for tid, n in ((plan[p]["rows"][0], B), (plan[p]["scales"][0], NBG), (MW.descriptor_id(fns[p]), B)):
            want[tid] = want.get(tid, 0) + n * TOPK * S
    for part, first in (("rows", 8), ("scales", 2)):
        assert plan["up"][part][0] == plan["down"][part][0] and plan["down"][part][3] == first * TOPK * S
    assert circuit.units == want and len(want) == 5
    lays = circuit.layouts

    def bits(fn, ports) -> int:
        tid = MW.descriptor_id(fn)
        return 16 * sum(dict(lays[tid][0].rows)[p] for p in ports) * lays[tid][1]

    pins = {p: f"{i:0128x}" for i, p in enumerate(sorted(registered))}
    held = MW.WI.input_count(circuit.description(), pins=pins, roots=pins)
    left = bits(fns["router"], ("logits", "bias")) + bits(fns["up"], ("xq", "sx")) + bits(fns["down"], ("xq", "sx"))
    assert held["ingress"] == 0 and held["registered"] > 0 and held["free"] == left


def test_the_fused_cut_reads_each_blocks_rows_inside_its_gemm(MW, captured):
    _d, v = captured
    blocks = {"w13": 8, "w2": 8}
    lab = MW.labels({"up": ("model.layers.3.mlp.experts.w13_weight", "model.layers.3.mlp.experts.w13_weight_scale_inv"),
                     "down": ("model.layers.3.mlp.experts.w2_weight", "model.layers.3.mlp.experts.w2_weight_scale_inv")}, None)
    plain, _ = MW.cut(v, MW.templates(v, 2.5, "block", blocks), lab, blocks, "test")
    fns = MW.templates(v, 2.5, "block", blocks, fused=True)
    circuit, plan = MW.cut(v, fns, lab, blocks, "test", fused=True)
    up, down, scales = MW.descriptor_id(fns["up"]), MW.descriptor_id(fns["down"]), plan["up"]["scales"][0]
    assert (up.split("{")[0], down.split("{")[0]) == ("MoeExpertFp8ReadGemm_v1", "MoeExpertFp8ReadGemmW_v1")
    assert (plan["up"]["rows"][0], plan["down"]["rows"][0], plan["down"]["scales"][0]) == (up, down, scales)
    assert circuit.units == {MW.descriptor_id(fns["router"]): S, up: 8 * TOPK * S, down: 8 * TOPK * S, scales: 2 * 2 * TOPK * S}
    assert circuit.registered.params == plain.registered.params
    assert {o.reader for o in circuit.registered.openings} == {r.reader for r in circuit.registered.roots} == {up, down, scales}
    lays = circuit.layouts

    def bits(tid, ports) -> int:
        return 16 * sum(dict(lays[tid][0].rows)[p] for p in ports) * lays[tid][1]

    pins = {p: f"{i:0128x}" for i, p in enumerate(sorted(plain.registered.params))}
    held = MW.WI.input_count(circuit.description(), pins=pins, roots=pins)
    left = bits(MW.descriptor_id(fns["router"]), ("logits", "bias")) + bits(up, ("xq", "sx")) + bits(down, ("xq", "sx"))
    assert held["ingress"] == 0 and held["registered"] > 0 and held["free"] == left


@pytest.mark.slow
def test_a_staged_layer_is_a_window_its_verifier_takes_from_the_records(MW, captured, tmp_path, capsys):
    """The whole staging on the synthetic capture, its forged arm and ``--circuit-only``, and the verifier's side of the window
    (`interface.of_records`, `registered_reads` under the registration's pins), which no cheaper test reaches: slow for the
    population Programs of the cut's seven templates the window and the verifier each build."""
    from verity.core.protocols.sampled_proofs.one_stage import draw as D
    from verity.core.protocols.sampled_proofs.one_stage import registered as REG
    from verity_vllm.records import expert_tables as XR

    d, _v = captured
    base = ["--capture", str(d), "--step", str(STEP), "--layer", str(LAYER), "--tokens", "2", "--cap", "4", *BLOCKS,
            "--self-register", "--key", str(tmp_path / "key")]
    assert MW.main([*base, "--out", str(tmp_path / "w")]) == 0
    st = json.loads((tmp_path / "w" / "stage.json").read_text())
    assert st["gate"]["ok"] and st["tables"]["not_registered_rows"] == [] and st["run"] == MW.ST.Capture(d, STEP).root
    doc = json.loads((tmp_path / "w" / "tables.json").read_text())
    assert XR.ExpertTables.of(doc, "tables.json").dump() == doc
    for value, e in doc["values"].items():
        want = REG.domain(doc["program"], value, e["words"], e["leaves"], e.get(REG.WIDTH_KEY, REG.WORD_BITS))
        assert e["value"] == value and e["domain"] == want.domain_id.hex()
    mine, rec = MW.WI.of_records(tmp_path / "w" / "window", D.parse_law("stratified:1"), MW.WI.resolve)
    files = MW.WI.registered_reads(rec, mine, doc["values"], registered=REG)
    reads = [m for m in mine.members if m.template.startswith("EmbeddingRead_v1")]
    assert sorted(files) == sorted(m.key for m in reads) and reads
    assert all(set(f["ports"]) == {"root", "domain", "o1"} for f in files.values())
    reg = json.loads((tmp_path / "w" / "window" / "registration.json").read_text())
    roots = reg["window"]["registered_roots"]
    assert reg["window"]["circuit"] == st["circuit"] and all(roots[p] == e["root"] for p, e in doc["values"].items())
    capsys.readouterr()
    assert MW.main([*base, "--out", str(tmp_path / "c"), "--circuit-only"]) == 0
    assert json.loads(capsys.readouterr().out.splitlines()[-1])["circuit"] == st["circuit"]
    assert MW.main([*base, "--out", str(tmp_path / "f"), "--forge", "w13"]) == 0
    forged = json.loads((tmp_path / "f" / "stage.json").read_text())
    assert forged["forge"] == "w13" and forged["circuit"] == st["circuit"]


@pytest.mark.slow
def test_a_fused_layer_stages_units_their_templates_compute_and_its_verifier_pins_their_reads(MW, captured, tmp_path, capsys):
    """``--fused`` staged with ``--full-gate`` on the synthetic capture, and taken back by the verifier's side: the only check
    that the fused GEMMs' recorded rows, salts, paths and activations are their templates' (`Window.mismatches`) before a
    session proves them; slow for the fused templates' population Programs and their evaluation."""
    from verity.core.protocols.sampled_proofs.one_stage import draw as D
    from verity.core.protocols.sampled_proofs.one_stage import registered as REG

    d, _v = captured
    base = ["--capture", str(d), "--step", str(STEP), "--layer", str(LAYER), "--tokens", "2", "--cap", "4", *BLOCKS,
            "--self-register", "--key", str(tmp_path / "key"), "--fused"]
    assert MW.main([*base, "--out", str(tmp_path / "w"), "--full-gate"]) == 0
    st = json.loads((tmp_path / "w" / "stage.json").read_text())
    assert st["fused"] and st["gate"]["ok"] and st["tables"]["not_registered_rows"] == []
    assert st["full_gate"]["mismatched"] == {}
    assert st["units"] == {"router": 2, "up.ws": 2 * TOPK * 2, "up.fused": 8 * TOPK * 2, "down.ws": 2 * TOPK * 2,
                           "down.fused": 8 * TOPK * 2}
    doc = json.loads((tmp_path / "w" / "tables.json").read_text())
    mine, rec = MW.WI.of_records(tmp_path / "w" / "window", D.parse_law("stratified:1"), MW.WI.resolve)
    files = MW.WI.registered_reads(rec, mine, doc["values"], registered=REG)
    reads = [m for m in mine.members if m.template.split("{")[0] in
             ("EmbeddingRead_v1", "MoeExpertFp8ReadGemm_v1", "MoeExpertFp8ReadGemmW_v1")]
    assert sorted(files) == sorted(m.key for m in reads) and any(m.template.startswith("MoeExpertFp8ReadGemm") for m in reads)
    assert all(set(f["ports"]) == {"root", "domain", "o1"} for f in files.values())
    capsys.readouterr()
    assert MW.main([*base, "--out", str(tmp_path / "c"), "--circuit-only"]) == 0
    assert json.loads(capsys.readouterr().out.splitlines()[-1])["circuit"] == st["circuit"]
