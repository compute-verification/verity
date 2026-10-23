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
    """k = 256 and 512 are under the record's k ≥ 1,024; n = 160 is not a multiple of 64.  Every other linear is in, with γ
    at both FADD readings, the data sheet's (cheaper add) a little above the record's, and γ falling with n."""
    rows = {r["name"]: r for r in G.table()}
    refused = {name for name, r in rows.items() if r["refused"]}
    assert refused == {"self_attn.kv_b_proj", "self_attn.indexer.wk_weights_proj", "mlp.shared_experts.down_proj",
                       "mlp.experts.w2"}
    assert all("1024 <= k" in rows[x]["refused"][0] for x in refused - {"self_attn.indexer.wk_weights_proj"})
    assert "n of 64" in rows["self_attn.indexer.wk_weights_proj"]["refused"][0]
    for name, r in rows.items():
        if name not in refused:
            assert 0 < r["gamma"]["record"] < r["gamma"]["sheet"] < 0.011, name
    assert rows["mlp.experts.w13"]["gamma"]["record"] > rows["self_attn.o_proj"]["gamma"]["record"]


def test_gamma_is_the_lean_twins():
    """`Pouw.PearlC.pearlCGammaSm120v1Cast32Rev1_8192`: v1 under rev1 (cap 1/400) at the sm_120 record, 8192³, the honest cast
    at 32 (24 over the credited 8), γ = 173561/22302500."""
    from verity.core.protocols.pouw.schemes import pearl_c as C
    from verity.core.protocols.pouw.schemes.pearl_c_device import SM120

    s = C.PearlC(forming="v1", device=replace(SM120, cap=Fraction(1, 400)), hashing="h3")
    assert G.gamma(s, 8192, 8192, Fraction(24)) == Fraction(173561, 22302500)
    h = G.scheme()
    assert h.name == G.SCHEME and h.prices.fadd == 32 and G.scheme(Fraction(1979, 67)).prices.fadd == Fraction(1979, 67)


def test_h200_session_runs_the_tables_shapes():
    """h200.sh's run.py SHAPES are every in-domain (n, k), at decode's 64 and prefill's 8192."""
    sh = (HERE / "h200.sh").read_text()
    listed = re.search(r"for nk in ([\dx ]+); do", sh).group(1).split()
    assert {tuple(map(int, x.split("x"))) for x in listed} == {(r["n"], r["k"]) for r in G.table() if not r["refused"]}
    assert 'SHAPES="$SHAPES 64x$nk 8192x$nk"' in sh
