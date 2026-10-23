"""GLM-5.3's linears under pearl-c-h100-v1-h3 (benchmarks/pouw/pearl_c_sm90/glm53.py): one TP8 rank's shapes from the
checkpoint's config, the sm_90a kernels' domain verdict on each, γ's formula against a Lean price twin, and the H200
session's shapes the table's."""

from __future__ import annotations

import importlib.util
import re
import sys
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parents[1] / "pearl_c_sm90"
spec = importlib.util.spec_from_file_location("pearl_c_sm90_glm53", HERE / "glm53.py")
G = sys.modules[spec.name] = importlib.util.module_from_spec(spec)
spec.loader.exec_module(G)


def test_linears_are_one_tp8_ranks():
    """The capture map's shapes (glm-moe-dsa-eager-hopper-tp8-v0) at GLM-5.3's config: KVA = QL + KVL + DR, NHL = NH / 8,
    IKW = ID + IH, IL = I / 8, MIL = SIL = MI / 8; compute-accounting's per-rank K (256, 1536, 2048, 6144) are the FP8
    linears' k but kv_b_proj's 512."""
    got = {lin.name: (lin.n, lin.k) for lin in G.linears()}
    assert got == {
        "self_attn.fused_qkv_a_proj": (2624, 6144), "self_attn.q_b_proj": (2048, 2048), "self_attn.kv_b_proj": (3584, 512),
        "self_attn.o_proj": (6144, 2048), "self_attn.indexer.wq_b": (4096, 2048), "self_attn.indexer.wk_weights_proj": (160, 6144),
        "mlp.gate_up_proj": (3072, 6144), "mlp.down_proj": (6144, 1536), "mlp.shared_experts.gate_up_proj": (512, 6144),
        "mlp.shared_experts.down_proj": (6144, 256), "mlp.experts.w13": (512, 6144), "mlp.experts.w2": (6144, 256)}
    fp8 = {lin.k for lin in G.linears() if lin.stock != "cublas-bf16-gemm"}
    assert fp8 == {256, 512, 1536, 2048, 6144}
    assert G.expert_rows(64) == 2 and G.expert_rows(8192) == 256


def test_domain_refuses_the_short_k_and_narrow_n():
    """k = 512 is under the record's k ≥ 1,024 and not its census k = 256 (which takes both down projections); n = 160 is
    not a multiple of 64.  Every other linear is in, with γ at both FADD readings, the data sheet's (cheaper add) a little
    above the record's, and γ falling with n."""
    rows = {r["name"]: r for r in G.table()}
    refused = {name for name, r in rows.items() if r["refused"]}
    assert refused == {"self_attn.kv_b_proj", "self_attn.indexer.wk_weights_proj"}
    assert "1024 <= k" in rows["self_attn.kv_b_proj"]["refused"][0]
    assert "n of 64" in rows["self_attn.indexer.wk_weights_proj"]["refused"][0]
    for name, r in rows.items():
        if name not in refused:
            assert 0 < r["gamma"]["record"] < r["gamma"]["sheet"] < 0.011, name
    assert rows["mlp.experts.w13"]["gamma"]["record"] > rows["self_attn.o_proj"]["gamma"]["record"]


def test_expert_parallel_brings_the_experts_into_the_domain():
    """With the routed experts expert-parallel (EP = TP = 8) a rank holds 32 whole experts, w13 (4096, 6144) and w2
    (6144, 2048), both in the domain; the shared expert stays sharded, its down at the census's k = 256."""
    rows = {r["name"]: r for r in G.table(ep=True)}
    assert {x: (rows[x]["n"], rows[x]["k"]) for x in ("mlp.experts.w13", "mlp.experts.w2")} == {
        "mlp.experts.w13": (4096, 6144), "mlp.experts.w2": (6144, 2048)}
    assert {x for x, r in rows.items() if r["refused"]} == {"self_attn.kv_b_proj", "self_attn.indexer.wk_weights_proj"}
    assert (rows["mlp.shared_experts.down_proj"]["k"], rows["mlp.shared_experts.down_proj"]["refused"]) == (256, [])
    shapes = G.in_domain()
    assert (4096, 6144) in shapes and shapes[(4096, 6144)]["ep"] and not shapes[(6144, 2048)]["ep"]
    assert shapes[(6144, 2048)]["name"] == "self_attn.o_proj"


def test_gamma_is_the_lean_twins():
    """`Pouw.PearlC.pearlCGammaSm120v1Cast32Rev1_8192`: v1 under rev1 (cap 1/400) at the sm_120 record, 8192³, the honest cast
    at 32 (24 over the credited 8), γ = 173561/22302500."""
    from verity.core.protocols.pouw.schemes import pearl_c as C
    from verity.core.protocols.pouw.schemes.pearl_c_device import SM120

    s = C.PearlC(forming="v1", device=replace(SM120, cap=Fraction(1, 400)), hashing="h3")
    assert G.gamma(s, 8192, 8192, Fraction(24)) == Fraction(173561, 22302500)
    h = G.scheme()
    assert h.name == G.SCHEME and h.prices.fadd == 32 and G.scheme(Fraction(1979, 67)).prices.fadd == Fraction(1979, 67)


def test_the_profile_gives_each_launch_to_the_step_that_launched_it():
    """glm53_profile's report: a GPU activity is the innermost step's around its launch (the API call's thread and time, not
    when it ran), one launched outside every call is dropped, the GPU span is the calls' first start to their last end over
    the calls, the solo call's is its own, and a reference SHA-512 launch is named as a fallback."""
    spec = importlib.util.spec_from_file_location("pearl_c_sm90_glm53_profile", HERE / "glm53_profile.py")
    P = sys.modules[spec.name] = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(P)
    t, label = 7, "6144:2048 m16 hash"
    ranges = [(t, 0, 100, f"call {label} #0"), (t, 10, 30, "step commit_a"), (t, 40, 60, "step gemm_pearlc"),
              (t, 70, 90, "host after"), (t, 75, 85, "host screen"),
              (t, 100, 200, f"call {label} #1"), (t, 110, 130, "step commit_a"), (t, 140, 160, "step hash_leaf"),
              (t, 300, 400, f"solo {label}"), (t, 310, 320, "step commit_a"), (t, 500, 900, f"pass_end {label}")]
    acts = [(t, 15, 1000, 3000, "kernel", "pd_commit_851"), (t, 45, 3000, 5000, "kernel", "gemm_pearlc_d"),
            (t, 80, 5000, 5500, "kernel", "screen"), (t, 72, 5500, 5600, "memcpy", "memcpy"),
            (t, 105, 6000, 6100, "memcpy", "memcpy"), (t, 115, 7000, 9000, "kernel", "pd_commit_851"),
            (t, 145, 9000, 10000, "kernel", "s512_tile_leaves"), (t, 250, 20000, 21000, "kernel", "stray"),
            (t + 1, 15, 21000, 22000, "kernel", "other thread"), (t, 315, 30000, 32000, "kernel", "pd_commit_851")]
    out = P.report(sorted(ranges), acts)
    assert set(out) == {label}
    r, steps = out[label], out[label]["steps"]
    assert r["calls"] == 2 and r["host_wall_us"] == 0.1 and r["pass_end_host_us"] == 0.4
    assert {s: round(v["gpu_us"], 9) for s, v in steps.items()} == {
        "step commit_a": 2.0, "step gemm_pearlc": 2.0, "host screen": 0.5, "host after": 0.1, "host other": 0.05, "step hash_leaf": 1.0}
    assert steps["step commit_a"]["host_us"] == 0.02 and steps["step commit_a"]["kernels"]["pd_commit_851"]["count"] == 1
    assert round(steps["host other"]["host_us"], 9) == 0.05 and steps["host screen"]["host_us"] == 0.01
    assert r["gpu_busy_us"] == 3.85 and r["gpu_span_us_per_call"] == 4.5 and r["solo_gpu_span_us"] == 2.0
    assert (r["hash_decode"], r["hash_fast"], r["fallbacks"]) == (["pd_commit_851"], [], ["s512_tile_leaves"])


def test_h200_session_runs_the_tables_shapes():
    """h200.sh's run.py SHAPES are every in-domain (n, k) (`in_domain`), at decode's 64 and prefill's 8192."""
    sh = (HERE / "h200.sh").read_text()
    listed = re.search(r"for nk in ([\dx ]+); do", sh).group(1).split()
    assert {tuple(map(int, x.split("x"))) for x in listed} == set(G.in_domain())
    assert 'SHAPES="$SHAPES 64x$nk 8192x$nk"' in sh
